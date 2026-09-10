"""Estimate how sampled graph topologies affect TabICL graph performance.

The script uses TabArena's own train/test split and passes the raw pandas data
through :class:`TabICLClassifier`, which keeps the benchmark preprocessing
unchanged. For every requested number of graph topologies, it samples a new
graph set for each pass and reports the resulting ROC AUC distribution.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	confusion_matrix,
	precision_recall_curve,
	precision_score,
	recall_score,
	roc_auc_score,
)

from tabicl import TabICLClassifier
from tabicl._model.graph import CompactGraphSet


ABLATIONS = (
	"matched-baselines",
	"undersampling",
	"candidate-quality",
	"attention-selectivity",
	"all",
)


def load_tabarena_task(dataset_name: str, preset: str):
	from tabarena.benchmark.task.metadata import TaskMetadataCollection
	from tabarena.benchmark.task.spec import task_spec_from_task_id_str

	collection = (
		TaskMetadataCollection.from_preset(preset)
		.subset_tasks(dataset_names=[dataset_name])
		.materialize()
	)
	metadata = collection.task_metadata_by_dataset().get(dataset_name)
	if metadata is None:
		raise ValueError(f"Dataset {dataset_name!r} was not found in preset {preset!r}")
	return task_spec_from_task_id_str(metadata.task_id_str).with_task_metadata(metadata).load()


def print_dataset_statistics(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> None:
	"""Print split dimensions, label counts, and feature cardinalities."""
	labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
	if len(labels) != 2:
		raise ValueError("Dataset statistics currently support binary labels only")
	negative_label, positive_label = labels
	discrete_columns = X_train.select_dtypes(
		include=["object", "category", "string", "bool", "boolean"]
	).columns
	all_features = pd.concat([X_train, X_test], axis=0, ignore_index=True)
	cardinalities = all_features[discrete_columns].nunique(dropna=True)

	print("Dataset statistics")
	print(f"  train samples: {len(X_train)}")
	print(f"  test samples: {len(X_test)}")
	print(
		f"  train labels: positive={int(np.sum(np.asarray(y_train) == positive_label))}, "
		f"negative={int(np.sum(np.asarray(y_train) == negative_label))}"
	)
	print(
		f"  test labels: positive={int(np.sum(np.asarray(y_test) == positive_label))}, "
		f"negative={int(np.sum(np.asarray(y_test) == negative_label))}"
	)
	print(f"  features: continuous={X_train.shape[1] - len(discrete_columns)}, discrete={len(discrete_columns)}")
	print(
		"  discrete cardinality: "
		f"max={int(cardinalities.max()) if not cardinalities.empty else 'N/A'}, "
		f"mean={cardinalities.mean():.2f}"
		if not cardinalities.empty
		else "  discrete cardinality: max=N/A, mean=N/A"
	)


def build_classifier(
	model_path: Path,
	device: str,
	topology_count: int,
	seed: int,
	cross_label_fraction: float,
	min_train_neighbors: int,
	max_train_neighbors: int,
	train_neighbors_per_test: int,
	skip_gat: bool = False,
) -> TabICLClassifier:
	return TabICLClassifier(
		model_path=model_path,
		allow_auto_download=False,
		device=device,
		n_estimators=1,
		batch_size=1,
		kv_cache=False,
		use_amp=device != "cpu",
		offload_mode=False if device == "cpu" else "cpu",
		norm_methods="none",
		softmax_temperature=0.9,
		graph_config={
			"graph_v1_prob": 1.0,
			"graph_v2_prob": 0.0,
			"graph_prob": 0.0,
			"graph_num_graphs": topology_count,
			"graph_seed": seed,
			"graph_cross_label_fraction": cross_label_fraction,
			"graph_min_train_neighbors": min_train_neighbors,
			"graph_max_train_neighbors": max_train_neighbors,
			"graph_train_neighbors_per_test": train_neighbors_per_test,
			"skip_gat": skip_gat,
		},
	)


def class_separability_score(representations: np.ndarray, labels: np.ndarray) -> float:
	"""Return a Fisher-style between-class / within-class scatter ratio."""
	features = np.asarray(representations, dtype=np.float64)
	labels = np.asarray(labels)
	global_mean = features.mean(axis=0)
	between = 0.0
	within = 0.0
	for label in np.unique(labels):
		class_features = features[labels == label]
		class_mean = class_features.mean(axis=0)
		between += len(class_features) * np.sum((class_mean - global_mean) ** 2)
		within += np.sum((class_features - class_mean) ** 2)
	return float(between / max(within, np.finfo(np.float64).eps))


def linear_probe_roc_auc(
	representations: np.ndarray,
	train_labels: np.ndarray,
	test_labels: np.ndarray,
) -> float:
	"""Fit a linear probe on train representations and score held-out rows."""
	features = np.asarray(representations, dtype=np.float32)
	train_labels = np.asarray(train_labels)
	test_labels = np.asarray(test_labels)
	train_features = features[: len(train_labels)]
	test_features = features[len(train_labels) :]
	probe = LogisticRegression(max_iter=1000, solver="lbfgs")
	probe.fit(train_features, train_labels)
	return float(roc_auc_score(test_labels, probe.predict_proba(test_features)[:, 1]))


@torch.no_grad()
def extract_semantic_features(
	classifier: TabICLClassifier,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	) -> torch.Tensor:
	"""Return normalized pre-GAT row features, cached across graph passes."""
	engine = classifier.model_
	model = getattr(engine, "model", engine)
	device = next(model.parameters()).device
	train_size = len(y_train)
	total_nodes = train_size + len(X_test)
	X_encoded = classifier.X_encoder_.transform(X_test)
	representations = []
	for subset, generator in zip(classifier.feature_subsets_, classifier.ensemble_generators_):
		data = generator.transform(X_encoded[:, subset], mode="both")
		norm_method, (Xs, ys) = next(iter(data.items()))
		feature_shuffles = generator.feature_shuffles_[norm_method]
		X_tensor = torch.from_numpy(Xs).float().to(device)
		y_tensor = torch.from_numpy(ys).float().to(device)
		graph_input = engine._graph_input(
			X_tensor,
			y_tensor,
			getattr(engine, "inference_config_", classifier.inference_config_),
			feature_shuffles,
		)
		row_representation = graph_input[0].reshape(total_nodes, -1).float()
		representations.append(row_representation)
	if not representations:
		raise ValueError("No fitted feature representation was available for semantic graph construction")
	features = torch.stack(representations).mean(dim=0)
	return torch.nn.functional.normalize(features, dim=1)


@torch.no_grad()
def build_semantic_graph_set(
	semantic_features: torch.Tensor,
	y_train: pd.Series,
	prior_graph_set: CompactGraphSet,
	seed: int,
	neighbor_range: tuple[int, int],
	frac_k_unconditional: float,
	frac_k_label_conditional: float,
	train_neighbors_per_test: int,
) -> CompactGraphSet:
	"""Build a sparse semantic, label-conditioned, and random graph."""
	device = semantic_features.device
	train_size = len(y_train)
	total_nodes = semantic_features.shape[0]
	if prior_graph_set.num_datasets != 1 or prior_graph_set.num_nodes != total_nodes:
		raise ValueError("Semantic graph construction expects one dataset-sized prior graph")

	similarities = semantic_features[:train_size] @ semantic_features.T
	labels = np.asarray(y_train)
	unique_labels = np.unique(labels)
	min_neighbors, max_neighbors = neighbor_range
	all_graph_edges = []
	for graph_index in range(prior_graph_set.num_graphs):
		rng = np.random.default_rng(seed + graph_index)
		edges: set[tuple[int, int]] = set()
		target_degrees = [
			int(rng.integers(min_neighbors, max_neighbors + 1))
			if destination < train_size
			else train_neighbors_per_test
			for destination in range(total_nodes)
		]

		for destination in range(total_nodes):
			candidate_count = train_size - (destination < train_size)
			if candidate_count <= 0:
				continue
			per_destination = target_degrees[destination]
			unconditional_k = min(
				per_destination, int(round(per_destination * frac_k_unconditional))
			)
			label_k = min(
				per_destination - unconditional_k,
				int(round(per_destination * frac_k_label_conditional)),
			)
			candidate_mask = torch.ones(train_size, dtype=torch.bool, device=device)
			if destination < train_size:
				candidate_mask[destination] = False
			candidate_indices = torch.where(candidate_mask)[0]
			k = min(unconditional_k, candidate_indices.numel())
			semantic_sources = candidate_indices[torch.topk(
				similarities[candidate_indices, destination], k=k
			).indices] if k else candidate_indices[:0]
			for source in semantic_sources.cpu().tolist():
				edges.add((source, destination))

			# For test rows, distribute label-conditioned candidates across the
			# observed training classes because the test label is unavailable.
			if destination < train_size:
				label_groups = [labels[candidate_indices.cpu().numpy()] == labels[destination]]
			else:
				candidate_labels = labels[candidate_indices.cpu().numpy()]
				label_groups = [candidate_labels == label for label in unique_labels]
			for label_mask in label_groups:
				group = candidate_indices[torch.from_numpy(label_mask).to(device)]
				if group.numel() == 0:
					continue
				k = min(
					label_k // max(len(label_groups), 1),
					group.numel(),
				)
				if not k:
					continue
				label_sources = group[torch.topk(similarities[group, destination], k=k).indices]
				for source in label_sources.cpu().tolist():
					edges.add((source, destination))

		# Sample exploration edges directly instead of building every possible pair.
		# Fill duplicate collisions from the semantic selections with random edges.
		degrees = np.zeros(total_nodes, dtype=np.int64)
		for _, destination in edges:
			degrees[destination] += 1
		for destination in range(total_nodes):
			while degrees[destination] < target_degrees[destination]:
				source = int(rng.integers(train_size))
				if source != destination:
					edge = (source, destination)
					if edge not in edges:
						edges.add(edge)
						degrees[destination] += 1
		edge_array = torch.tensor(list(edges), dtype=torch.long, device=device).T
		if edge_array.numel() == 0:
			edge_array = torch.empty((2, 0), dtype=torch.long, device=device)
		all_graph_edges.append(edge_array)

	parts = []
	offsets = []
	running = 0
	for edge_index in all_graph_edges:
		parts.append(edge_index)
		running += edge_index.shape[1]
		offsets.append([running - edge_index.shape[1], running])
	edge_index = torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.long, device=device)
	return CompactGraphSet(
		edge_index=edge_index,
		edge_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
		num_nodes=total_nodes,
	)


def compact_graph_edges(graph_set: CompactGraphSet) -> list[torch.Tensor]:
	"""Return one local edge index per graph topology for one dataset."""
	if graph_set.num_datasets != 1:
		raise ValueError("Diagnostics currently expect one dataset per graph set")
	return [
		graph_set.edge_index[:, start:end]
		for start, end in zip(
			graph_set.edge_offsets[:, 0].detach().cpu().tolist(),
			graph_set.edge_offsets[:, 1].detach().cpu().tolist(),
		)
	]


def candidate_quality_rows(
	semantic_features: torch.Tensor,
	graph_set: CompactGraphSet,
	y_train: pd.Series,
	y_test: pd.Series,
	) -> list[dict[str, object]]:
	"""Measure graph coverage and purity against dense semantic neighbors."""
	train_size = len(y_train)
	labels = np.asarray(y_train)
	test_labels = np.asarray(y_test)
	similarities = (semantic_features[:train_size] @ semantic_features.T).detach().cpu().numpy()
	rows = []
	for topology, edge_index in enumerate(compact_graph_edges(graph_set)):
		edges = edge_index.detach().cpu().numpy()
		for destination in range(train_size, semantic_features.shape[0]):
			sources = edges[0, edges[1] == destination]
			if len(sources) == 0:
				continue
			top_k = min(len(sources), len(sources))
			dense_order = np.argsort(-similarities[:, destination])[:top_k]
			graph_labels = labels[sources]
			best_same = np.flatnonzero(labels[dense_order] == test_labels[destination - train_size])
			rows.append(
				{
					"topology": topology,
					"destination": destination - train_size,
					"degree": len(sources),
					"mean_similarity": float(similarities[sources, destination].mean()),
					"max_similarity": float(similarities[sources, destination].max()),
					"same_label_fraction": float(np.mean(graph_labels == test_labels[destination - train_size])),
					"class_count": int(np.unique(graph_labels).size),
					"topk_dense_recall": float(len(np.intersect1d(sources, dense_order)) / top_k),
					"topk_same_label_recall": float(len(np.intersect1d(sources, dense_order[best_same])) / max(len(best_same), 1)),
				}
			)
	return rows


def attention_selectivity_rows(
	classifier: TabICLClassifier,
	graph_set: CompactGraphSet,
	y_train: pd.Series,
	) -> list[dict[str, object]]:
	"""Summarize the attention distributions cached by each GAT block."""
	model = getattr(classifier.model_, "model", classifier.model_)
	gat = model.icl_predictor.gat_icl
	labels = np.asarray(y_train)
	rows = []
	for layer, block in enumerate(gat.graph_blocks, start=1):
		weights = block.attn.last_attention_weights if hasattr(block, "attn") else None
		sources = block.attn.last_attention_edge_src if hasattr(block, "attn") else None
		destinations = block.attn.last_attention_edge_dst if hasattr(block, "attn") else None
		if weights is None or sources is None or destinations is None:
			continue
		weights = weights.mean(dim=(1, 2)).detach().cpu().numpy()
		sources = sources.detach().cpu().numpy()
		destinations = destinations.detach().cpu().numpy()
		for destination in np.unique(destinations):
			mask = destinations == destination
			edge_weights = weights[mask]
			probabilities = edge_weights / max(edge_weights.sum(), np.finfo(float).eps)
			entropy = float(-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum())
			valid_sources = sources[mask]
			is_test = destination >= len(y_train)
			rows.append(
				{
					"layer": layer,
					"destination": int(destination),
					"is_test": is_test,
					"attention_entropy": entropy,
					"effective_support": float(np.exp(entropy)),
					"max_attention": float(edge_weights.max()),
					"same_label_attention_mass": np.nan if is_test else float(
						probabilities[labels[valid_sources] == labels[destination]].sum()
					),
				}
			)
	return rows


def layerwise_diagnostics(
	classifier: TabICLClassifier,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
	graph_set: object,
	skip_probabilities: np.ndarray,
	use_linear_probe: bool = False,
	attention_rows: list[dict[str, object]] | None = None,
	semantic_features: torch.Tensor | None = None,
) -> list[dict[str, object]]:
	"""Decode and score the pre-GAT representation and every graph layer."""
	engine = classifier.model_
	model = getattr(engine, "model", engine)
	if model.icl_backend not in {"encoder", "graph-1d", "graph-1d-pyg"}:
		raise ValueError("Layerwise diagnostics currently support graph-1d checkpoints only")
	model_device = next(model.parameters()).device
	is_encoder = model.icl_backend == "encoder"
	semantic_similarity = None
	if semantic_features is not None:
		semantic_similarity = (
			semantic_features[: len(y_train)] @ semantic_features.T
		).detach().cpu().numpy()
	inference_config = getattr(engine, "inference_config_", classifier.inference_config_)
	outputs = []
	X_encoded = classifier.X_encoder_.transform(X_test)
	for subset, generator in zip(classifier.feature_subsets_, classifier.ensemble_generators_):
		data = generator.transform(X_encoded[:, subset], mode="both")
		for norm_method, (Xs, ys) in data.items():
			feature_shuffles = generator.feature_shuffles_[norm_method]
			class_shuffle = generator.class_shuffles_[norm_method][0]
			X_tensor = torch.from_numpy(Xs).float().to(model_device)
			y_tensor = torch.from_numpy(ys).float().to(model_device)
			if is_encoder:
				col_embeddings = model.col_embedder(
					X_tensor,
					y_train=y_tensor,
					feature_shuffles=feature_shuffles,
					mgr_config=inference_config.COL_CONFIG,
				)
				representation = model.row_interactor(
					col_embeddings, mgr_config=inference_config.ROW_CONFIG
				).clone()
			else:
				graph_input = engine._graph_input(
					X_tensor, y_tensor, inference_config, feature_shuffles
				)
				compact_graph = engine._compact_graph_set(graph_set, X_tensor.shape[0])
				compact_edges = compact_graph.edge_index.to(graph_input.device, dtype=torch.long)
				graph_edges = []
				for graph_index in range(compact_graph.num_graphs):
					starts = compact_graph.edge_offsets[graph_index, :-1].detach().cpu().tolist()
					ends = compact_graph.edge_offsets[graph_index, 1:].detach().cpu().tolist()
					graph_edges.append([
						compact_edges[:, start:end] for start, end in zip(starts, ends)
					])
				representation = graph_input.clone()

			representation[:, : y_tensor.shape[1]] += model.icl_predictor.y_encoder(
				y_tensor.float()
			)
			stages = [(0, representation)]
			if is_encoder:
				encoder = model.icl_predictor.tf_icl
				for layer_index, block in enumerate(encoder.blocks, start=1):
					representation = block(
						q=representation.to(dtype=next(encoder.parameters()).dtype),
						train_size=y_tensor.shape[1],
						rope=encoder.rope,
					)
					stages.append((layer_index, representation))
			else:
				gat = model.icl_predictor.gat_icl
				for layer_index, block in enumerate(gat.graph_blocks, start=1):
					graph_index = (layer_index - 1) // gat.layers_per_graph
					representation = block(
						representation.to(dtype=next(gat.parameters()).dtype).unsqueeze(2),
						graph_edges[graph_index],
					).squeeze(2)
					if attention_rows is not None:
						attention = block.attn
						weights = attention.last_attention_weights
						sources = attention.last_attention_edge_src
						destinations = attention.last_attention_edge_dst
						if weights is not None and sources is not None and destinations is not None:
							weights = weights.mean(dim=(1, 2)).detach().cpu().numpy()
							sources = sources.detach().cpu().numpy()
							destinations = destinations.detach().cpu().numpy()
							for destination in np.unique(destinations):
								mask = destinations == destination
								edge_weights = weights[mask]
								probabilities = edge_weights / max(edge_weights.sum(), 1e-12)
								source_labels = np.asarray(y_train)[sources[mask]]
								target_label = (
									np.asarray(y_train)[destination]
									if destination < len(y_train)
									else np.asarray(y_test)[destination - len(y_train)]
								)
								same_label = source_labels == target_label
								entropy = float(
									-(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()
								)
								attention_rows.append(
									{
										"layer": layer_index,
										"destination": int(destination),
										"is_test": bool(destination >= len(y_train)),
										"attention_entropy": entropy,
										"effective_support": float(np.exp(entropy)),
										"max_attention": float(edge_weights.max()),
										"same_label_attention_mass": float(probabilities[same_label].sum()),
										"candidate_same_label_fraction": float(np.mean(same_label)),
										"candidate_positive_count": int(np.sum(source_labels == np.asarray(classifier.classes_)[1])),
										"candidate_negative_count": int(np.sum(source_labels == np.asarray(classifier.classes_)[0])),
										"top_dense_similarity_attention_mass": np.nan,
										"similarity_attention_correlation": np.nan,
									}
								)
								if semantic_similarity is not None:
									similarity = semantic_similarity[sources[mask], destination]
									top_sources = np.argsort(-semantic_similarity[:, destination])[: len(similarity)]
								top_mask = np.isin(sources[mask], top_sources)
								correlation = np.nan
								if np.std(similarity) > 0 and np.std(edge_weights) > 0:
									correlation = float(np.corrcoef(similarity, edge_weights)[0, 1])
								attention_rows[-1]["top_dense_similarity_attention_mass"] = float(
									probabilities[top_mask].sum()
								)
								attention_rows[-1]["similarity_attention_correlation"] = correlation
								attention_rows[-1]["attention_purity_gain"] = (
									attention_rows[-1]["same_label_attention_mass"]
									- attention_rows[-1]["candidate_same_label_fraction"]
								)
					stages.append((layer_index, representation))

			train_size = len(y_train)
			for layer_index, representation in stages:
				representation_numpy = representation.detach().cpu().numpy()[0]
				decoder_representation = representation
				if is_encoder and model.icl_predictor.norm_first:
					decoder_representation = model.icl_predictor.ln(decoder_representation)
				if is_encoder:
					logits = model.icl_predictor._decode_representations(
						decoder_representation, y_tensor, train_size
					)
				else:
					logits = engine._decode(decoder_representation, y_tensor)
				logits = logits[:, train_size:]
				probabilities = torch.softmax(logits / 0.9, dim=-1).detach().cpu().numpy()[0]
				probabilities = probabilities[:, class_shuffle]
				positive_prob = probabilities[:, 1]
				prediction_delta = positive_prob - skip_probabilities[:, 1]
				if attention_rows is not None and not is_encoder:
					positive_label = np.asarray(classifier.classes_)[1]
					test_labels = np.asarray(y_test)
					predicted_positive = positive_prob >= 0.5
					correct = predicted_positive == (test_labels == positive_label)
					for attention_row in attention_rows:
						if attention_row["layer"] != layer_index or not attention_row["is_test"]:
							continue
						test_index = int(attention_row["destination"]) - train_size
						if not 0 <= test_index < len(test_labels):
							continue
						attention_row.update(
							{
								"true_label": test_labels[test_index],
								"true_is_positive": bool(test_labels[test_index] == positive_label),
								"predicted_label": positive_label if predicted_positive[test_index] else np.asarray(classifier.classes_)[0],
								"positive_probability": float(positive_prob[test_index]),
								"skip_positive_probability": float(skip_probabilities[test_index, 1]),
								"prediction_margin": float(abs(positive_prob[test_index] - 0.5)),
								"prediction_change": float(prediction_delta[test_index]),
								"correct": bool(correct[test_index]),
							}
						)
				outputs.append(
					{
						"layer": layer_index,
						"class_separability": class_separability_score(
							representation.detach().cpu().numpy()[0, :train_size],
							y_train,
						),
						"class_separability_test": class_separability_score(
							representation_numpy[train_size:],
							y_test,
						),
						"linear_probe_roc_auc": (
							linear_probe_roc_auc(
								representation_numpy,
								np.asarray(y_train),
								np.asarray(y_test),
							)
							if use_linear_probe
							else np.nan
						),
						"validation_roc_auc": roc_auc_score(y_test, positive_prob),
						"roc_auc": roc_auc_score(y_test, positive_prob),
						"prediction_change_mean_abs": float(np.abs(prediction_delta).mean()),
						"prediction_change_rmse": float(np.sqrt(np.mean(prediction_delta**2))),
						"prediction_change_max_abs": float(np.abs(prediction_delta).max()),
					}
				)
	return outputs


def plot_candidate_quality(rows: pd.DataFrame, output: Path) -> None:
	"""Plot graph coverage, similarity, and label purity diagnostics."""
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	for column, axis, title in [
		("topk_dense_recall", axes[0], "Dense-neighbor recall"),
		("mean_similarity", axes[1], "Mean cosine similarity"),
		("same_label_fraction", axes[2], "Same-label fraction"),
	]:
		rows.boxplot(column=column, by="configuration", ax=axis, grid=False)
		axis.set_title(title)
		axis.set_xlabel("")
	fig.suptitle("")
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def plot_attention_selectivity(rows: pd.DataFrame, output: Path) -> None:
	"""Plot attention concentration and alignment by GAT layer."""
	grouped = rows.groupby(["configuration", "layer"], as_index=False).mean(numeric_only=True)
	fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
	axes = axes.ravel()
	for configuration, group in grouped.groupby("configuration"):
		axes[0].plot(group["layer"], group["attention_entropy"], marker="o", label=configuration)
		axes[1].plot(group["layer"], group["effective_support"], marker="o", label=configuration)
		axes[2].plot(group["layer"], group["max_attention"], marker="o", label=configuration)
		axes[3].plot(group["layer"], group["same_label_attention_mass"], marker="o", label=configuration)
		axes[4].plot(group["layer"], group["top_dense_similarity_attention_mass"], marker="o", label=configuration)
		axes[5].plot(group["layer"], group["similarity_attention_correlation"], marker="o", label=configuration)
	for axis, title in zip(
		axes,
		(
			"Attention entropy",
			"Effective support",
			"Maximum attention",
			"Same-label attention mass",
			"Top dense-neighbor attention mass",
			"Cosine-attention correlation",
		),
	):
		axis.set_title(title)
		axis.set_xlabel("GAT layer")
		axis.grid(alpha=0.3)
		axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def plot_attention_error_analysis(rows: pd.DataFrame, output: Path) -> None:
	"""Compare attention alignment for correct and incorrect test predictions."""
	valid = rows.dropna(subset=["correct"])
	grouped = valid.groupby(["correct", "layer"], as_index=False).mean(numeric_only=True)
	fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
	metrics = [
		("same_label_attention_mass", "Same-label attention mass"),
		("top_dense_similarity_attention_mass", "Top dense-neighbor attention mass"),
		("similarity_attention_correlation", "Cosine-attention correlation"),
	]
	for correct, group in grouped.groupby("correct"):
		label = "correct" if correct else "incorrect"
		for axis, (metric, title) in zip(axes, metrics):
			axis.plot(group["layer"], group[metric], marker="o", label=label)
	for axis, (_, title) in zip(axes, metrics):
		axis.set_title(title)
		axis.set_xlabel("GAT layer")
		axis.grid(alpha=0.3)
		axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def plot_attention_purity_margin(rows: pd.DataFrame, output: Path) -> None:
	"""Relate neighborhood purity and prediction confidence per test row."""
	valid = rows.dropna(
		subset=[
			"correct",
			"prediction_margin",
			"candidate_same_label_fraction",
			"same_label_attention_mass",
		]
	)
	fig, axes = plt.subplots(1, 3, figsize=(16, 4))
	for correct, group in valid.groupby("correct"):
		label = "correct" if correct else "incorrect"
		axes[0].scatter(
			group["candidate_same_label_fraction"],
			group["same_label_attention_mass"],
			alpha=0.35,
			s=14,
			label=label,
		)
		axes[1].scatter(
			group["same_label_attention_mass"],
			group["prediction_margin"],
			alpha=0.35,
			s=14,
			label=label,
		)
		axes[2].scatter(
			group["attention_purity_gain"],
			group["prediction_margin"],
			alpha=0.35,
			s=14,
			label=label,
		)
	axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
	axes[0].set_xlabel("Candidate same-label fraction")
	axes[0].set_ylabel("Attention-weighted same-label mass")
	axes[0].set_title("GAT selection vs candidate purity")
	axes[1].set_xlabel("Attention-weighted same-label mass")
	axes[1].set_ylabel("Prediction margin")
	axes[1].set_title("Purity vs confidence")
	axes[2].set_xlabel("Attention purity gain")
	axes[2].set_ylabel("Prediction margin")
	axes[2].set_title("Selection gain vs confidence")
	for axis in axes:
		axis.grid(alpha=0.3)
		axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def _safe_roc_auc(labels: pd.Series, scores: pd.Series) -> float:
	"""Return ROC AUC or NaN when a subset contains one class."""
	if labels.nunique(dropna=True) < 2:
		return np.nan
	return float(roc_auc_score(labels.astype(bool), scores))


def attention_class_analysis(rows: pd.DataFrame) -> pd.DataFrame:
	"""Summarize class-wise prediction quality and purity-linked errors."""
	valid = rows.dropna(
		subset=[
			"correct",
			"true_is_positive",
			"positive_probability",
			"skip_positive_probability",
			"candidate_same_label_fraction",
			"same_label_attention_mass",
		]
	).copy()
	if valid.empty:
		return pd.DataFrame()
	valid["true_class"] = np.where(valid["true_is_positive"], "positive", "negative")
	valid["high_purity_confident_error"] = (
		(~valid["correct"].astype(bool))
		& (valid["same_label_attention_mass"] >= 0.75)
		& (valid["prediction_margin"] >= 0.25)
	)
	rows_out = []
	for (layer, true_class), group in valid.groupby(["layer", "true_class"]):
		positive_predictions = group["positive_probability"] >= 0.5
		rows_out.append(
			{
				"layer": int(layer),
				"true_class": true_class,
				"rows": len(group),
				"accuracy": float(group["correct"].mean()),
				"recall": float((positive_predictions == group["true_is_positive"]).mean()),
				"mean_positive_probability": float(group["positive_probability"].mean()),
				"mean_skip_positive_probability": float(group["skip_positive_probability"].mean()),
				"mean_candidate_purity": float(group["candidate_same_label_fraction"].mean()),
				"mean_attention_purity": float(group["same_label_attention_mass"].mean()),
				"high_purity_confident_error_rate": float(group["high_purity_confident_error"].mean()),
				"roc_auc": _safe_roc_auc(group["true_is_positive"], group["positive_probability"]),
			}
		)
	for layer, group in valid.groupby("layer"):
		rows_out.append(
			{
				"layer": int(layer),
				"true_class": "all",
				"rows": len(group),
				"accuracy": float(group["correct"].mean()),
				"recall": np.nan,
				"mean_positive_probability": float(group["positive_probability"].mean()),
				"mean_skip_positive_probability": float(group["skip_positive_probability"].mean()),
				"mean_candidate_purity": float(group["candidate_same_label_fraction"].mean()),
				"mean_attention_purity": float(group["same_label_attention_mass"].mean()),
				"high_purity_confident_error_rate": float(group["high_purity_confident_error"].mean()),
				"correctness_auc_attention_purity": _safe_roc_auc(
					group["correct"], group["same_label_attention_mass"]
				),
				"correctness_auc_candidate_purity": _safe_roc_auc(
					group["correct"], group["candidate_same_label_fraction"]
				),
			}
		)
	return pd.DataFrame(rows_out)


def plot_attention_class_analysis(rows: pd.DataFrame, output: Path) -> None:
	"""Plot class-specific probability shifts and purity-linked errors."""
	valid = rows.dropna(
		subset=["correct", "true_is_positive", "positive_probability", "skip_positive_probability"]
	).copy()
	if valid.empty:
		return
	valid["true_class"] = np.where(valid["true_is_positive"], "positive", "negative")
	valid["correct_numeric"] = valid["correct"].astype(float)
	fig, axes = plt.subplots(1, 3, figsize=(16, 4))
	for true_class, group in valid.groupby("true_class"):
		grouped = group.groupby("layer", as_index=False).mean(numeric_only=True)
		axes[0].plot(
			grouped["layer"], grouped["skip_positive_probability"], marker="o",
			label=f"{true_class} pre-GAT",
		)
		axes[0].plot(
			grouped["layer"], grouped["positive_probability"], marker="o", linestyle="--",
			label=f"{true_class} post-GAT",
		)
		axes[1].plot(
			grouped["layer"], grouped["correct_numeric"], marker="o", label=true_class,
		)
	incorrect = valid[~valid["correct"].astype(bool)]
	for true_class, group in incorrect.groupby("true_class"):
		axes[2].scatter(
			group["same_label_attention_mass"], group["prediction_margin"],
			alpha=0.35, s=14, label=f"{true_class} incorrect",
		)
	axes[0].set_title("Probability by true class")
	axes[0].set_ylabel("Positive probability")
	axes[1].set_title("Accuracy by true class")
	axes[1].set_ylabel("Accuracy")
	axes[2].set_title("Incorrect predictions")
	axes[2].set_xlabel("Attention-weighted same-label mass")
	axes[2].set_ylabel("Prediction margin")
	for axis in axes:
		axis.set_xlabel(axis.get_xlabel() or "GAT layer")
		axis.grid(alpha=0.3)
		axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def _semantic_graph_for_run(
	classifier: TabICLClassifier,
	X_train: pd.DataFrame,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	args: argparse.Namespace,
) -> tuple[torch.Tensor, CompactGraphSet]:
	"""Fit a graph classifier and build one semantic graph for diagnostics."""
	classifier.fit(X_train, y_train)
	engine = classifier.model_
	model = getattr(engine, "model", engine)
	class_indices = {label: index for index, label in enumerate(classifier.classes_.tolist())}
	encoded = np.asarray([class_indices[label] for label in np.asarray(y_train)])
	device = next(model.parameters()).device
	train_labels = torch.as_tensor(encoded, device=device).reshape(1, -1)
	prior = engine._make_graph_set(train_labels, len(y_train) + len(X_test))
	features = extract_semantic_features(classifier, X_test, y_train)
	graph_set = build_semantic_graph_set(
		features,
		y_train,
		prior,
		args.seed,
		args.semantic_neighbor_range,
		args.frac_k_unconditional,
		args.frac_k_label_conditional,
		args.train_neighbors_per_test_semantic,
	)
	return features, graph_set


def run_candidate_quality(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
	classifier = build_classifier(
		args.model_path, args.device, args.topology_count, args.seed, 0.1,
		args.semantic_neighbor_range[0], args.semantic_neighbor_range[1], args.train_neighbors_per_test_semantic,
	)
	features, graph_set = _semantic_graph_for_run(classifier, X_train, X_test, y_train, args)
	rows = candidate_quality_rows(features, graph_set, y_train, y_test)
	frame = pd.DataFrame(rows)
	frame.insert(0, "dataset", args.dataset)
	frame.insert(1, "fold", args.fold)
	frame.insert(2, "repeat", args.repeat)
	frame["configuration"] = (
		f"u={args.frac_k_unconditional:.2f},l={args.frac_k_label_conditional:.2f}"
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(args.output, index=False)
	plot_candidate_quality(frame, args.plot)


def run_attention_selectivity(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
	classifier = build_classifier(
		args.model_path, args.device, args.topology_count, args.seed, 0.1,
		args.semantic_neighbor_range[0], args.semantic_neighbor_range[1], args.train_neighbors_per_test_semantic,
	)
	semantic_features, graph_set = _semantic_graph_for_run(classifier, X_train, X_test, y_train, args)
	skip_owner = classifier.model_.model.icl_predictor
	previous_skip_gat = skip_owner.skip_gat
	try:
		skip_owner.skip_gat = True
		skip_probabilities = np.asarray(classifier.predict_proba(X_test))
	finally:
		skip_owner.skip_gat = previous_skip_gat
	attention_rows: list[dict[str, object]] = []
	layerwise_diagnostics(
		classifier, X_test, y_train, y_test, graph_set, skip_probabilities,
		attention_rows=attention_rows, semantic_features=semantic_features,
	)
	frame = pd.DataFrame(attention_rows)
	frame.insert(0, "dataset", args.dataset)
	frame.insert(1, "fold", args.fold)
	frame.insert(2, "repeat", args.repeat)
	frame["configuration"] = (
		f"u={args.frac_k_unconditional:.2f},l={args.frac_k_label_conditional:.2f}"
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	frame.to_csv(args.output, index=False)
	plot_attention_selectivity(frame[frame["is_test"]], args.plot)
	plot_attention_error_analysis(
		frame[frame["is_test"]],
		args.plot.with_name(f"{args.plot.stem}-error-analysis{args.plot.suffix}"),
	)
	plot_attention_purity_margin(
		frame[frame["is_test"]],
		args.plot.with_name(f"{args.plot.stem}-purity-margin{args.plot.suffix}"),
	)
	class_analysis = attention_class_analysis(frame[frame["is_test"]])
	class_analysis.to_csv(
		args.output.with_name(f"{args.output.stem}-class-analysis.csv"),
		index=False,
	)
	plot_attention_class_analysis(
		frame[frame["is_test"]],
		args.plot.with_name(f"{args.plot.stem}-class-analysis{args.plot.suffix}"),
	)


def plot_matched_confusion_matrices(
	prediction_rows: list[dict[str, object]],
	output: Path,
) -> None:
	"""Plot one normalized confusion matrix per matched-baseline configuration."""
	configurations = list(dict.fromkeys(row["configuration"] for row in prediction_rows))
	figure, axes = plt.subplots(
		1,
		len(configurations),
		figsize=(4 * len(configurations), 4),
		squeeze=False,
	)
	for axis, configuration in zip(axes.ravel(), configurations):
		rows = [row for row in prediction_rows if row["configuration"] == configuration]
		true_labels = np.asarray([row["true_label"] for row in rows], dtype=bool)
		predictions = np.asarray([row["prediction"] for row in rows], dtype=bool)
		matrix = confusion_matrix(true_labels, predictions, labels=[False, True])
		row_totals = matrix.sum(axis=1, keepdims=True)
		normalized = matrix / np.maximum(row_totals, 1)
		image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
		for row_index in range(2):
			for column_index in range(2):
				axis.text(
					column_index,
					row_index,
					f"{matrix[row_index, column_index]}\n{normalized[row_index, column_index]:.2f}",
					ha="center",
					va="center",
					color="white" if normalized[row_index, column_index] > 0.5 else "black",
				)
		axis.set_title(configuration)
		axis.set_xlabel("Predicted label")
		axis.set_ylabel("True label")
		axis.set_xticks([0, 1], ["negative", "positive"])
		axis.set_yticks([0, 1], ["negative", "positive"])
		figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row proportion")
	figure.suptitle("Matched-baseline confusion matrices")
	figure.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output, dpi=160)
	plt.close(figure)


def plot_matched_precision_recall(
	prediction_rows: list[dict[str, object]],
	output: Path,
) -> None:
	"""Plot precision-recall curves for matched-baseline configurations."""
	figure, axis = plt.subplots(figsize=(7, 5))
	for configuration in dict.fromkeys(row["configuration"] for row in prediction_rows):
		rows = [row for row in prediction_rows if row["configuration"] == configuration]
		true_labels = np.asarray([row["true_label"] for row in rows], dtype=bool)
		positive_probabilities = np.asarray(
			[row["positive_probability"] for row in rows], dtype=float
		)
		precision, recall, _ = precision_recall_curve(true_labels, positive_probabilities)
		axis.plot(recall, precision, label=configuration)
	axis.set_xlabel("Recall")
	axis.set_ylabel("Precision")
	axis.set_xlim(0.0, 1.0)
	axis.set_ylim(0.0, 1.0)
	axis.set_title("Matched-baseline precision-recall curves")
	axis.grid(alpha=0.3)
	axis.legend()
	figure.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output, dpi=160)
	plt.close(figure)


def run_matched_baselines(args: argparse.Namespace, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
	"""Compare encoder, graph, semantic graph, and skip-GAT under one budget."""
	rows = []
	prediction_rows = []
	configs = [
		("graph-prior", args.model_path, False, False),
		("graph-semantic", args.model_path, False, True),
		("graph-skip-gat", args.model_path, True, False),
	]
	if args.encoder_model_path is not None:
		configs.append(("encoder", args.encoder_model_path, False, False))
	for name, model_path, skip_gat, semantic in configs:
		classifier = build_classifier(
			model_path, args.device, args.topology_count, args.seed, 0.1,
			args.semantic_neighbor_range[0], args.semantic_neighbor_range[1], args.train_neighbors_per_test_semantic,
			skip_gat=skip_gat,
		)
		classifier.fit(X_train, y_train)
		graph_set = None
		if name in {"graph-prior", "graph-semantic"}:
			engine = classifier.model_
			model = getattr(engine, "model", engine)
			indices = {label: index for index, label in enumerate(classifier.classes_.tolist())}
			encoded = torch.as_tensor([indices[label] for label in np.asarray(y_train)], device=next(model.parameters()).device).reshape(1, -1)
			graph_set = engine._make_graph_set(encoded, len(y_train) + len(X_test))
			if semantic:
				features = extract_semantic_features(classifier, X_test, y_train)
				graph_set = build_semantic_graph_set(
					features, y_train, graph_set, args.seed, args.semantic_neighbor_range,
					args.frac_k_unconditional, args.frac_k_label_conditional,
					args.train_neighbors_per_test_semantic,
				)
		probabilities = classifier.predict_proba(X_test, graph_set=graph_set)
		positive_probabilities = probabilities[:, 1]
		predictions = positive_probabilities >= args.threshold
		true_labels = np.asarray(y_test) == classifier.classes_[1]
		rows.append(
			{
				"dataset": args.dataset,
				"fold": args.fold,
				"repeat": args.repeat,
				"configuration": name,
				"roc_auc": roc_auc_score(true_labels, positive_probabilities),
				"precision": precision_score(true_labels, predictions, zero_division=0),
				"recall": recall_score(true_labels, predictions, zero_division=0),
				"sensitivity": recall_score(true_labels, predictions, zero_division=0),
			}
		)
		prediction_rows.extend(
			{
				"configuration": name,
				"true_label": bool(true_label),
				"prediction": bool(prediction),
				"positive_probability": float(positive_probability),
			}
			for true_label, prediction, positive_probability in zip(
				true_labels, predictions, positive_probabilities
			)
		)
	frame = pd.DataFrame(rows)
	frame.to_csv(args.output, index=False)
	metrics = ["roc_auc", "precision", "recall", "sensitivity"]
	labels = ["ROC AUC", "Precision", "Recall", "Sensitivity"]
	fig, axis = plt.subplots(figsize=(10, 5))
	positions = np.arange(len(frame))
	bar_width = 0.2
	for metric_index, (metric, label) in enumerate(zip(metrics, labels)):
		axis.bar(
			positions + (metric_index - (len(metrics) - 1) / 2) * bar_width,
			frame[metric],
			bar_width,
			label=label,
		)
	axis.set_ylabel("Score")
	axis.set_ylim(0.0, 1.0)
	axis.set_xticks(positions, frame["configuration"])
	axis.tick_params(axis="x", rotation=20)
	axis.legend()
	axis.grid(axis="y", alpha=0.3)
	fig.tight_layout()
	args.plot.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(args.plot, dpi=160)
	plt.close(fig)
	plot_matched_confusion_matrices(
		prediction_rows,
		args.plot.with_name(f"{args.plot.stem}-confusion-matrix{args.plot.suffix}"),
	)
	plot_matched_precision_recall(
		prediction_rows,
		args.plot.with_name(f"{args.plot.stem}-precision-recall{args.plot.suffix}"),
	)


def undersample_training_context(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	positive_to_negative_ratio: float,
	seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
	"""Keep all negatives and randomly undersample positives to a target ratio."""
	if positive_to_negative_ratio <= 0:
		raise ValueError("undersampling ratios must be positive")
	labels = np.asarray(y_train)
	unique_labels = np.unique(labels)
	if len(unique_labels) != 2:
		raise ValueError("undersampling currently supports binary labels only")
	negative_label, positive_label = unique_labels
	negative_indices = np.flatnonzero(labels == negative_label)
	positive_indices = np.flatnonzero(labels == positive_label)
	target_positive_count = min(
		len(positive_indices),
		max(1, int(round(positive_to_negative_ratio * len(negative_indices)))),
	)
	rng = np.random.default_rng(seed)
	selected_positive = rng.choice(
		positive_indices,
		size=target_positive_count,
		replace=False,
	)
	selected_indices = np.concatenate([negative_indices, selected_positive])
	rng.shuffle(selected_indices)
	return (
		X_train.iloc[selected_indices].reset_index(drop=True),
		y_train.iloc[selected_indices].reset_index(drop=True),
	)


def run_undersampling(
	args: argparse.Namespace,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
) -> None:
	"""Run matched baselines after positive-class undersampling at each ratio."""
	base_output = args.output
	base_plot = args.plot
	for ratio in args.undersampling_ratios:
		undersampled_X, undersampled_y = undersample_training_context(
			X_train, y_train, ratio, args.seed + int(round(ratio * 1000)),
		)
		ratio_label = f"p{ratio:g}to1"
		args.output = base_output.with_name(
			f"{base_output.stem}-{ratio_label}{base_output.suffix}"
		)
		args.plot = base_plot.with_name(f"{base_plot.stem}-{ratio_label}{base_plot.suffix}")
		run_matched_baselines(args, undersampled_X, undersampled_y, X_test, y_test)
	args.output = base_output
	args.plot = base_plot


def evaluate_topology_count(
	classifier: TabICLClassifier,
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_test: pd.DataFrame,
	y_test: pd.Series,
	passes: int,
	seed: int,
	topology_count: int,
	cross_label_fraction: float,
	min_train_neighbors: int,
	max_train_neighbors: int,
	train_neighbors_per_test: int,
	semantic_neighbor_range: tuple[int, int],
	frac_k_unconditional: float,
	frac_k_label_conditional: float,
	train_neighbors_per_test_semantic: int,
	use_linear_probe: bool = False,
	use_semantic_graph: bool = False,
	) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
	classifier.fit(X_train, y_train)
	engine = classifier.model_
	model = getattr(engine, "model", engine)
	is_encoder = model.icl_backend == "encoder"
	if not is_encoder and not hasattr(engine, "_make_graph_set"):
		raise ValueError("The checkpoint does not use the GAT graph inference engine")
	if not is_encoder and engine.num_layers % topology_count != 0:
		raise ValueError(
			f"topology_count={topology_count} must divide the checkpoint's "
			f"{engine.num_layers} GAT blocks"
		)

	class_indices = {label: index for index, label in enumerate(classifier.classes_.tolist())}
	encoded_train_labels = np.asarray([class_indices[label] for label in np.asarray(y_train)])
	model_device = next(model.parameters()).device
	train_labels = torch.as_tensor(encoded_train_labels, device=model_device).reshape(1, -1)
	total_nodes = len(X_train) + len(X_test)
	skip_owner = model.icl_predictor
	previous_skip_gat = skip_owner.skip_gat
	try:
		skip_owner.skip_gat = True
		skip_probabilities = np.asarray(classifier.predict_proba(X_test))
	finally:
		skip_owner.skip_gat = previous_skip_gat
	results = []
	layer_results = []
	semantic_features = (
		extract_semantic_features(classifier, X_test, y_train)
		if use_semantic_graph and not is_encoder
		else None
	)
	for pass_index in range(passes if not is_encoder else 1):
		# The engine's seeded graph prior advances its seed per call, so every
		# pass receives a fresh topology while remaining reproducible.
		prior_graph_set = None if is_encoder else engine._make_graph_set(train_labels, total_nodes)
		graph_set = (
			None
			if is_encoder
			else (
				build_semantic_graph_set(
					semantic_features,
					y_train,
					prior_graph_set,
					seed + pass_index,
					semantic_neighbor_range,
					frac_k_unconditional,
					frac_k_label_conditional,
					train_neighbors_per_test_semantic,
				)
				if use_semantic_graph
				else prior_graph_set
			)
		)
		probabilities = np.asarray(classifier.predict_proba(X_test, graph_set=graph_set))
		if probabilities.ndim != 2 or probabilities.shape[1] != 2:
			raise ValueError("This diagnostic currently supports binary classification only")
		results.append(
			{
				"topology_count": topology_count,
				"cross_label_fraction": cross_label_fraction,
				"min_train_neighbors": min_train_neighbors,
				"max_train_neighbors": max_train_neighbors,
				"train_neighbors_per_test": train_neighbors_per_test,
				"neighbor_ranges_semantic": f"{semantic_neighbor_range[0]}-{semantic_neighbor_range[1]}",
				"frac_k_unconditional": frac_k_unconditional,
				"frac_k_label_conditional": frac_k_label_conditional,
				"train_neighbors_per_test_semantic": train_neighbors_per_test_semantic,
				"pass": pass_index,
				"seed": seed + pass_index,
				"backend": "encoder" if is_encoder else "graph",
				"graph_construction": "semantic" if use_semantic_graph else "prior",
				"roc_auc": roc_auc_score(y_test, probabilities[:, 1]),
			}
		)
		layer_results.extend(
			{
				**row,
				"topology_count": topology_count,
				"cross_label_fraction": cross_label_fraction,
				"min_train_neighbors": min_train_neighbors,
				"max_train_neighbors": max_train_neighbors,
				"train_neighbors_per_test": train_neighbors_per_test,
				"neighbor_ranges_semantic": f"{semantic_neighbor_range[0]}-{semantic_neighbor_range[1]}",
				"frac_k_unconditional": frac_k_unconditional,
				"frac_k_label_conditional": frac_k_label_conditional,
				"train_neighbors_per_test_semantic": train_neighbors_per_test_semantic,
				"pass": pass_index,
				"backend": "encoder" if is_encoder else "graph",
				"graph_construction": "semantic" if use_semantic_graph else "prior",
			}
			for row in layerwise_diagnostics(
				classifier,
				X_test,
				y_train,
				y_test,
				graph_set,
				skip_probabilities,
				use_linear_probe=use_linear_probe,
			)
		)
	return results, layer_results


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--ablation",
		choices=("legacy", *ABLATIONS),
		default="legacy",
		help="Diagnostic to run; legacy runs the original topology/layerwise benchmark",
	)
	parser.add_argument("--dataset", required=True, help="TabArena dataset name, e.g. Amazon_employee_access")
	parser.add_argument("--model-path", required=True, type=Path, help="Graph-backend TabICL checkpoint")
	parser.add_argument(
		"--encoder-model-path",
		type=Path,
		help="Optional encoder-backend checkpoint to compare against --model-path",
	)
	parser.add_argument("--preset", default="TabArena-v0.1", help="TabArena metadata preset")
	parser.add_argument("--fold", type=int, default=0)
	parser.add_argument("--repeat", type=int, default=0)
	parser.add_argument("--sample", type=int, default=0)
	parser.add_argument("--passes", type=int, default=16, help="Graph samples per topology count")
	parser.add_argument(
		"--topology-counts",
		default="1,2,3,6",
		help="Comma-separated graph topology counts; six is the benchmark setting",
	)
	parser.add_argument(
		"--cross-label-fractions",
		default="0.1",
		help="Comma-separated cross-label edge fractions; 0.1 is the benchmark setting",
	)
	parser.add_argument(
		"--neighbor-ranges",
		default="8-15",
		help="Comma-separated min-max training-neighbor ranges; 8-15 is the benchmark setting",
	)
	parser.add_argument(
		"--train-neighbors-per-test",
		default="8",
		help="Comma-separated numbers of training neighbors per test node; 8 is the benchmark setting",
	)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("eval/tabicl_graph/graph_influence.csv"),
		help="Output path for per-pass graph influence results",
	)
	parser.add_argument(
		"--summary-output",
		type=Path,
		default=Path("eval/tabicl_graph/graph_influence_summary.csv"),
		help="Output path for mean/std results",
	)
	parser.add_argument(
		"--layerwise-output",
		type=Path,
		default=Path("eval/tabicl_graph/graph_layerwise.csv"),
		help="Output path for per-layer diagnostics",
	)
	parser.add_argument(
		"--layerwise-plot",
		type=Path,
		default=Path("eval/tabicl_graph/graph_encoder_layerwise.png"),
		help="Side-by-side layerwise performance/separability plot",
	)
	parser.add_argument(
		"--density-plot",
		type=Path,
		default=Path("eval/tabicl_graph/graph_density_sweep.png"),
		help="ROC AUC plot for train-train and test-node graph density sweeps",
	)
	parser.add_argument(
		"--plot",
		type=Path,
		default=Path("eval/tabicl_graph/ablation.png"),
		help="Plot path for the selected focused ablation",
	)
	parser.add_argument(
		"--topology-count",
		type=int,
		default=1,
		help="Number of graph topologies for focused ablations",
	)
	parser.add_argument(
		"--undersampling-ratios",
		default="1,2,4",
		help="Positive:negative training-context ratios for the undersampling ablation",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.5,
		help="Positive-probability threshold for hard classification metrics",
	)
	parser.add_argument(
		"--linear-probe",
		action="store_true",
		help="Fit a linear probe on each layer's train representation and report test ROC AUC",
	)
	parser.add_argument(
		"--semantic-graph",
		action="store_true",
		help="Use pre-GAT semantic, label-conditioned, and random sparse edges",
	)
	parser.add_argument(
		"--neighbor-ranges-semantic",
		default="8-15",
		help="Min-max range for semantic train-train neighbors",
	)
	parser.add_argument(
		"--frac-k-unconditional",
		type=float,
		default=1 / 3,
		help="Fraction of semantic neighbors allocated to unconditional similarity",
	)
	parser.add_argument(
		"--frac-k-label-conditional",
		type=float,
		default=1 / 3,
		help="Fraction of semantic neighbors allocated to label-conditioned similarity",
	)
	parser.add_argument(
		"--train-neighbors-per-test-semantic",
		type=int,
		default=8,
		help="Number of semantic training neighbors per test node",
	)
	return parser


def save_layerwise_plot(layer_results: pd.DataFrame, output: Path) -> None:
	plot_probe = layer_results["linear_probe_roc_auc"].notna().any()
	fig, axes = plt.subplots(1, 4 if plot_probe else 3, figsize=(24 if plot_probe else 18, 4), sharex=True)
	axes = np.atleast_1d(axes)
	for backend, group in layer_results.groupby("backend"):
		group = group.groupby("layer", as_index=False).mean(numeric_only=True)
		axes[0].plot(group["layer"], group["validation_roc_auc"], marker="o", label=backend)
		axes[1].plot(group["layer"], group["class_separability"], marker="o", label=backend)
		axes[2].plot(group["layer"], group["class_separability_test"], marker="o", label=backend)
		if plot_probe:
			axes[3].plot(group["layer"], group["linear_probe_roc_auc"], marker="o", label=backend)
	axes[0].set_title("Validation ROC AUC")
	axes[1].set_title("Train class separability")
	axes[2].set_title("Test class separability")
	if plot_probe:
		axes[3].set_title("Linear probe ROC AUC")
	for axis in axes:
		axis.set_xlabel("Layer index")
		axis.grid(alpha=0.3)
		axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=160)
	plt.close(fig)


def save_density_sweep_plot(summary: pd.DataFrame, output: Path) -> None:
	"""Plot ROC AUC against test-node degree for each train-neighbor range."""
	if summary.empty:
		return
	plot_data = (
		summary.groupby(
			["backend", "min_train_neighbors", "max_train_neighbors", "train_neighbors_per_test"],
			as_index=False,
		)
		.agg(roc_auc_mean=("roc_auc_mean", "mean"))
	)
	backends = list(plot_data["backend"].drop_duplicates())
	figure, axes = plt.subplots(
		1,
		len(backends),
		figsize=(7 * len(backends), 5),
		squeeze=False,
		sharey=True,
	)
	for axis, backend in zip(axes.ravel(), backends):
		backend_data = plot_data[plot_data["backend"] == backend]
		for (minimum, maximum), group in backend_data.groupby(
			["min_train_neighbors", "max_train_neighbors"]
		):
			group = group.sort_values("train_neighbors_per_test")
			axis.plot(
				group["train_neighbors_per_test"],
				group["roc_auc_mean"],
				marker="o",
				label=f"train {minimum}-{maximum}",
			)
		axis.set_title(backend)
		axis.set_xlabel("Neighbors per test node")
		axis.set_ylabel("Mean ROC AUC")
		axis.set_xticks(sorted(backend_data["train_neighbors_per_test"].unique()))
		axis.set_ylim(0.0, 1.0)
		axis.grid(alpha=0.3)
		axis.legend()
	figure.suptitle("Graph density sweep")
	figure.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output, dpi=160)
	plt.close(figure)


def main() -> None:
	args = build_parser().parse_args()
	if args.passes <= 0:
		raise ValueError("--passes must be positive")
	topology_counts = [int(value) for value in args.topology_counts.split(",") if value.strip()]
	if not topology_counts or any(value <= 0 for value in topology_counts):
		raise ValueError("--topology-counts must contain positive integers")
	cross_label_fractions = [
		float(value) for value in args.cross_label_fractions.split(",") if value.strip()
	]
	if not cross_label_fractions or any(not 0.0 <= value <= 1.0 for value in cross_label_fractions):
		raise ValueError("--cross-label-fractions must be between 0 and 1")
	neighbor_ranges = []
	for value in args.neighbor_ranges.split(","):
		if value.strip():
			minimum, maximum = value.split("-", maxsplit=1)
			neighbor_ranges.append((int(minimum), int(maximum)))
	if not neighbor_ranges or any(minimum <= 0 or maximum < minimum for minimum, maximum in neighbor_ranges):
		raise ValueError("--neighbor-ranges must contain positive min-max ranges")
	train_neighbors_per_test = [
		int(value) for value in args.train_neighbors_per_test.split(",") if value.strip()
	]
	if not train_neighbors_per_test or any(value <= 0 for value in train_neighbors_per_test):
		raise ValueError("--train-neighbors-per-test must contain positive integers")
	semantic_range_parts = args.neighbor_ranges_semantic.split(",")
	if len(semantic_range_parts) != 1 or not semantic_range_parts[0].strip():
		raise ValueError("--neighbor-ranges-semantic must contain one min-max range")
	semantic_min, semantic_max = semantic_range_parts[0].split("-", maxsplit=1)
	semantic_neighbor_range = (int(semantic_min), int(semantic_max))
	if semantic_neighbor_range[0] <= 0 or semantic_neighbor_range[1] < semantic_neighbor_range[0]:
		raise ValueError("--neighbor-ranges-semantic must contain a positive min-max range")
	if not 0.0 <= args.frac_k_unconditional <= 1.0:
		raise ValueError("--frac-k-unconditional must be between 0 and 1")
	if not 0.0 <= args.frac_k_label_conditional <= 1.0:
		raise ValueError("--frac-k-label-conditional must be between 0 and 1")
	if args.frac_k_unconditional + args.frac_k_label_conditional > 1.0:
		raise ValueError("semantic neighbor fractions must sum to at most 1")
	if args.train_neighbors_per_test_semantic <= 0:
		raise ValueError("--train-neighbors-per-test-semantic must be positive")
	if args.topology_count <= 0:
		raise ValueError("--topology-count must be positive")
	if not 0.0 <= args.threshold <= 1.0:
		raise ValueError("--threshold must be between 0 and 1")
	args.undersampling_ratios = [
		float(value) for value in args.undersampling_ratios.split(",") if value.strip()
	]
	if not args.undersampling_ratios or any(value <= 0 for value in args.undersampling_ratios):
		raise ValueError("--undersampling-ratios must contain positive values")
	model_path = args.model_path.expanduser().resolve()
	if not model_path.is_file():
		raise FileNotFoundError(f"TabICL checkpoint not found: {model_path}")

	task = load_tabarena_task(args.dataset, args.preset)
	X_train, y_train, X_test, y_test = task.get_train_test_split(
		fold=args.fold,
		repeat=args.repeat,
		sample=args.sample,
	)
	print_dataset_statistics(X_train, y_train, X_test, y_test)
	if len(np.unique(y_train)) != 2:
		raise ValueError("This diagnostic currently supports binary classification only")
	args.model_path = model_path
	args.semantic_neighbor_range = semantic_neighbor_range
	if args.ablation != "legacy":
		ablation_names = ABLATIONS[:-1] if args.ablation == "all" else (args.ablation,)
		base_output = args.output
		base_plot = args.plot
		for ablation_name in ablation_names:
			if args.ablation == "all":
				args.output = base_output.with_name(f"{base_output.stem}-{ablation_name}{base_output.suffix}")
				args.plot = base_plot.with_name(f"{base_plot.stem}-{ablation_name}{base_plot.suffix}")
			if ablation_name == "matched-baselines":
				run_matched_baselines(args, X_train, y_train, X_test, y_test)
			elif ablation_name == "undersampling":
				run_undersampling(args, X_train, y_train, X_test, y_test)
			elif ablation_name == "candidate-quality":
				run_candidate_quality(args, X_train, y_train, X_test, y_test)
			elif ablation_name == "attention-selectivity":
				run_attention_selectivity(args, X_train, y_train, X_test, y_test)
		return
	rows = []
	layer_rows = []
	for (
		topology_count,
		cross_label_fraction,
		(min_train_neighbors, max_train_neighbors),
		neighbors_per_test,
	) in product(topology_counts, cross_label_fractions, neighbor_ranges, train_neighbors_per_test):
		classifier = build_classifier(
			model_path,
			args.device,
			topology_count,
			args.seed,
			cross_label_fraction,
			min_train_neighbors,
			max_train_neighbors,
			neighbors_per_test,
		)
		configuration_rows, configuration_layer_rows = evaluate_topology_count(
				classifier,
				X_train,
				y_train,
				X_test,
				y_test,
				args.passes,
				args.seed,
				topology_count,
				cross_label_fraction,
				min_train_neighbors,
				max_train_neighbors,
				neighbors_per_test,
				semantic_neighbor_range,
				args.frac_k_unconditional,
				args.frac_k_label_conditional,
				args.train_neighbors_per_test_semantic,
				use_linear_probe=args.linear_probe,
				use_semantic_graph=args.semantic_graph,
			)
		rows.extend(configuration_rows)
		layer_rows.extend(configuration_layer_rows)
	if args.encoder_model_path is not None:
		encoder_path = args.encoder_model_path.expanduser().resolve()
		if not encoder_path.is_file():
			raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
		encoder_classifier = build_classifier(
			encoder_path,
			args.device,
			topology_count=1,
			seed=args.seed,
			cross_label_fraction=0.0,
			min_train_neighbors=1,
			max_train_neighbors=1,
			train_neighbors_per_test=1,
		)
		encoder_rows, encoder_layer_rows = evaluate_topology_count(
			encoder_classifier,
			X_train,
			y_train,
			X_test,
			y_test,
			args.passes,
			args.seed,
			1,
			0.0,
			1,
			1,
			1,
			semantic_neighbor_range,
			args.frac_k_unconditional,
			args.frac_k_label_conditional,
			args.train_neighbors_per_test_semantic,
			use_linear_probe=args.linear_probe,
			use_semantic_graph=args.semantic_graph,
		)
		rows.extend(encoder_rows)
		layer_rows.extend(encoder_layer_rows)

	results = pd.DataFrame(rows)
	results.insert(0, "dataset", args.dataset)
	results.insert(1, "fold", args.fold)
	results.insert(2, "repeat", args.repeat)
	layer_results = pd.DataFrame(layer_rows)
	layer_results.insert(0, "dataset", args.dataset)
	layer_results.insert(1, "fold", args.fold)
	layer_results.insert(2, "repeat", args.repeat)
	summary = (
		results.groupby(
			[
				"dataset",
				"fold",
				"repeat",
				"backend",
				"topology_count",
				"cross_label_fraction",
				"min_train_neighbors",
				"max_train_neighbors",
				"train_neighbors_per_test",
				"neighbor_ranges_semantic",
				"frac_k_unconditional",
				"frac_k_label_conditional",
				"train_neighbors_per_test_semantic",
			],
			as_index=False,
		)
		.agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), passes=("roc_auc", "size"))
	)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.summary_output.parent.mkdir(parents=True, exist_ok=True)
	args.layerwise_output.parent.mkdir(parents=True, exist_ok=True)
	results.to_csv(args.output, index=False)
	summary.to_csv(args.summary_output, index=False)
	layer_results.to_csv(args.layerwise_output, index=False)
	save_layerwise_plot(layer_results, args.layerwise_plot)
	save_density_sweep_plot(summary, args.density_plot)
	print(summary.to_string(index=False))
	print(f"Per-pass results written to {args.output}")
	print(f"Summary written to {args.summary_output}")
	print(f"Layerwise diagnostics written to {args.layerwise_output}")
	print(f"Layerwise plot written to {args.layerwise_plot}")
	print(f"Density sweep plot written to {args.density_plot}")


if __name__ == "__main__":
	main()
