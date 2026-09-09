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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tabicl import TabICLClassifier


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


def layerwise_diagnostics(
	classifier: TabICLClassifier,
	X_test: pd.DataFrame,
	y_train: pd.Series,
	y_test: pd.Series,
	graph_set: object,
	skip_probabilities: np.ndarray,
	use_linear_probe: bool = False,
) -> list[dict[str, object]]:
	"""Decode and score the pre-GAT representation and every graph layer."""
	engine = classifier.model_
	model = getattr(engine, "model", engine)
	if model.icl_backend not in {"encoder", "graph-1d", "graph-1d-pyg"}:
		raise ValueError("Layerwise diagnostics currently support graph-1d checkpoints only")
	model_device = next(model.parameters()).device
	is_encoder = model.icl_backend == "encoder"
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
	use_linear_probe: bool = False,
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
	for pass_index in range(passes if not is_encoder else 1):
		# The engine's seeded graph prior advances its seed per call, so every
		# pass receives a fresh topology while remaining reproducible.
		graph_set = None if is_encoder else engine._make_graph_set(train_labels, total_nodes)
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
				"pass": pass_index,
				"seed": seed + pass_index,
				"backend": "encoder" if is_encoder else "graph",
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
				"pass": pass_index,
				"backend": "encoder" if is_encoder else "graph",
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
		"--linear-probe",
		action="store_true",
		help="Fit a linear probe on each layer's train representation and report test ROC AUC",
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
	model_path = args.model_path.expanduser().resolve()
	if not model_path.is_file():
		raise FileNotFoundError(f"TabICL checkpoint not found: {model_path}")

	task = load_tabarena_task(args.dataset, args.preset)
	X_train, y_train, X_test, y_test = task.get_train_test_split(
		fold=args.fold,
		repeat=args.repeat,
		sample=args.sample,
	)
	if len(np.unique(y_train)) != 2:
		raise ValueError("This diagnostic currently supports binary classification only")
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
				use_linear_probe=args.linear_probe,
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
			use_linear_probe=args.linear_probe,
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
	print(summary.to_string(index=False))
	print(f"Per-pass results written to {args.output}")
	print(f"Summary written to {args.summary_output}")
	print(f"Layerwise diagnostics written to {args.layerwise_output}")
	print(f"Layerwise plot written to {args.layerwise_plot}")


if __name__ == "__main__":
	main()
