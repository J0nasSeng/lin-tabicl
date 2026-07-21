"""Evaluate a graph-backend TabICL checkpoint against classical baselines.

The prior is sampled independently for every class count. Consequently, the
plots produced by this script compare models on exactly the same datasets for
each value of ``K``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, recall_score

from tabicl.eval._run import _build_model_from_checkpoint, _evaluate_one_dataset
from tabicl._preprocessing.normalizer import RobustScaler, Standardizer, infer_feature_types
from tabicl.prior._dataset import PriorDataset


# Registries make adding another baseline or score a local change rather than
# requiring changes to the evaluation loop and plotting code.
BASELINE_FACTORIES: dict[str, Callable[[int, int], object]] = {
	"random_forest": lambda seed, n_estimators: RandomForestClassifier(
		n_estimators=n_estimators, random_state=seed, n_jobs=1
	),
}
SCORES: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]] = {
	"balanced_accuracy": lambda y_true, y_pred, _y_proba, _labels: _balanced_accuracy(y_true, y_pred),
	"cross_entropy": lambda y_true, _y_pred, y_proba, labels: _cross_entropy(
		y_true, y_proba, labels
	),
	"entropy": lambda _y_true, _y_pred, y_proba, _labels: _mean_entropy(y_proba),
}


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""Compute balanced accuracy without warning on unknown predictions.

	TabICL can assign a test row to a class that is absent from the sampled
	test partition. ``balanced_accuracy_score`` delegates to recall and emits a
	warning in that case. Balanced accuracy is the mean recall over the true
	classes, so explicitly supplying those labels gives the same value without
	allowing a spurious prediction class to alter the class set.

	This evaluation script calls the low-level TabICL model directly; it does not
	construct the sklearn ensemble and therefore has no class-label permutation
	setting to disable. The training labels are passed to the model unchanged.
	"""
	labels = np.unique(y_true)
	return float(
		recall_score(
			y_true,
			y_pred,
			labels=labels,
			average="macro",
			zero_division=0,
		)
	)


def _cross_entropy(
	y_true: np.ndarray, y_proba: np.ndarray, labels: np.ndarray
) -> float:
	"""Compute cross entropy for the supplied, shared class support."""
	y_proba = _normalize_probabilities(y_proba)
	return float(log_loss(y_true, y_proba, labels=np.asarray(labels)))


def _has_complete_label_support(y_train: np.ndarray, y_test: np.ndarray) -> bool:
	"""Return whether every test label is represented in the training labels."""
	return np.isin(np.unique(y_test), np.unique(y_train)).all()


def _normalize_probabilities(y_proba: np.ndarray) -> np.ndarray:
	"""Convert finite nonnegative scores into row-normalized probabilities."""
	probabilities = np.asarray(y_proba, dtype=np.float64)
	probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
	probabilities = np.clip(probabilities, 0.0, None)
	row_sums = probabilities.sum(axis=1, keepdims=True)
	zero_rows = row_sums.squeeze(1) <= 0
	if np.any(zero_rows):
		probabilities[zero_rows] = 1.0 / probabilities.shape[1]
		row_sums = probabilities.sum(axis=1, keepdims=True)
	return probabilities / row_sums


def _mean_entropy(y_proba: np.ndarray) -> float:
	"""Compute the mean predictive entropy over test rows."""
	probabilities = _normalize_probabilities(y_proba)
	positive = probabilities > 0
	entropy_terms = np.zeros_like(probabilities)
	entropy_terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
	entropy = entropy_terms.sum(axis=1)
	return float(entropy.mean())


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("checkpoint", type=Path, nargs="?", help="TabICL checkpoint (.ckpt) to evaluate")
	parser.add_argument("--num-datasets", type=int, default=1000, help="Datasets per class count")
	parser.add_argument("--batch-size", type=int, default=32, help="Prior datasets generated per batch")
	parser.add_argument("--min-features", type=int, default=2)
	parser.add_argument("--max-features", type=int, default=10)
	parser.add_argument("--max-seq-len", type=int, default=1024)
	parser.add_argument("--min-seq-len", type=int, default=None)
	parser.add_argument("--min-train-size", type=float, default=0.2)
	parser.add_argument("--max-train-size", type=float, default=0.6)
	parser.add_argument("--prior-type", choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl"], default="nanotabicl")
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--device", default=None, help="Inference device (default: cuda when available)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--normalization",
		choices=("none", "std", "robust"),
		default="none",
		help="Optional feature normalization applied before evaluation (default: none).",
	)
	parser.add_argument(
		"--normalize-features",
		action="store_true",
		help="Compatibility alias for enabling feature normalization.",
	)
	parser.add_argument(
		"--rf-n-estimators",
		type=int,
		default=50,
		help="Number of RF trees per dataset; lower values speed up diagnostics (default: 50)",
	)
	parser.add_argument(
		"--pretrained-tabicl-n-estimators",
		type=int,
		default=1,
		help="Number of ensemble members for the pretrained TabICL baseline (default: 8)",
	)
	parser.add_argument(
		"--umap-n-neighbors",
		type=int,
		default=10,
		help="UMAP neighborhood size for plots; lower values are faster (default: 10)",
	)
	parser.add_argument(
		"--umap-n-epochs",
		type=int,
		default=100,
		help="UMAP optimization epochs for plots; lower values are faster (default: 100)",
	)
	parser.add_argument(
		"--skip-baselines",
		action="store_true",
		help="Only generate per-dataset UMAP plots; skip baseline evaluation and comparison plots.",
	)
	parser.add_argument("--output-dir", type=Path, default=Path("gat_eval"))
	parser.add_argument("--results-output", type=Path, default=None, help="JSON output path (default: output-dir/results.json)")
	parser.add_argument(
		"--plot-results-only",
		action="store_true",
		help="Only produce plots from results.json; skip checkpoint loading, prior generation, and model inference.",
	)
	return parser


def _as_regular_tensors(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
	"""Convert variable-length prior batches to tensors usable by the loop."""
	return tuple(
		value.to_padded_tensor(padding=0.0) if value.is_nested else value
		for value in batch
	)


def _normalize_features(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
	"""Normalize each dataset using statistics computed from its train rows only."""
	x, y, d, seq_lens, train_sizes = batch
	x = x.clone()
	for index in range(x.shape[0]):
		seq_len = int(seq_lens[index].item())
		train_size = int(train_sizes[index].item())
		feature_count = int(d[index].item())
		features = x[index, :seq_len, :feature_count]
		train_features = features[:train_size]
		finite = torch.isfinite(train_features)
		count = finite.sum(dim=0, keepdim=True).clamp_min(1)
		finite_train_features = torch.where(finite, train_features, 0.0)
		mean = finite_train_features.sum(dim=0, keepdim=True) / count
		centered = torch.where(finite, train_features - mean, 0.0)
		std = (centered.square().sum(dim=0, keepdim=True) / count).sqrt().clamp_min(1e-6)
		x[index, :seq_len, :feature_count] = (features - mean) / std
	return x, y, d, seq_lens, train_sizes


def _build_pretrained_tabicl(n_estimators: int, device: str, seed: int):
	"""Build the pretrained sklearn-compatible TabICL baseline.

	The checkpoint is downloaded automatically on the first ``fit`` when it is
	not already present in the Hugging Face cache, as documented in README.md.
	"""
	try:
		from tabicl import TabICLClassifier
	except ImportError as exc:  # pragma: no cover - package import failure
		raise ModuleNotFoundError("The pretrained TabICL baseline requires tabicl.") from exc

	return TabICLClassifier(
		n_estimators=n_estimators,
		device=device,
		random_state=seed,
		n_jobs=1,
		verbose=False,
	)


def _plot_umap(
	sample,
	num_classes: int,
	dataset_index: int,
	output_dir: Path,
	seed: int,
	rf_accuracy: float,
	tabicl_entropy: float,
	umap_n_neighbors: int,
	umap_n_epochs: int,
) -> None:
	"""Plot train and test representations before the soft-kNN layer side by side."""
	try:
		import matplotlib
		matplotlib.use("Agg", force=True)
		import matplotlib.pyplot as plt
	except ImportError as exc:  # pragma: no cover - runtime dependency
		raise ModuleNotFoundError("matplotlib is required for UMAP plots") from exc

	try:
		from umap import UMAP
	except ImportError as exc:  # pragma: no cover - runtime dependency
		raise ModuleNotFoundError("umap-learn is required for UMAP plots") from exc

	representations = sample.repr_full.numpy()
	labels = sample.y_full.numpy().astype(int)
	train_size = sample.train_size
	train_representations, test_representations = (
		representations[:train_size],
		representations[train_size:],
	)
	train_labels, test_labels = labels[:train_size], labels[train_size:]
	tabicl_accuracy = _balanced_accuracy(
		sample.y_true_test.numpy().astype(int), sample.y_pred_test.numpy().astype(int)
	)
	if train_representations.shape[0] < 3 or test_representations.shape[0] < 3:
		return

	# Fit one embedding jointly so train/test coordinates are comparable. This
	# also halves the expensive UMAP optimization work. A single worker avoids
	# joblib worker teardown issues in headless Python 3.14 runs.
	n_neighbors = min(umap_n_neighbors, representations.shape[0] - 1)
	embeddings = UMAP(
		n_components=2,
		n_neighbors=n_neighbors,
		min_dist=0.1,
		metric="euclidean",
		n_epochs=umap_n_epochs,
		# Avoid joblib worker teardown issues in headless Python 3.14 runs.
		n_jobs=1,
		random_state=None,
	).fit_transform(representations)
	train_embedding, test_embedding = embeddings[:train_size], embeddings[train_size:]

	fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
	all_labels = np.concatenate((train_labels, test_labels))
	label_min, label_max = all_labels.min(), all_labels.max()
	if label_min == label_max:
		label_min -= 0.5
		label_max += 0.5
	for ax, embedding, partition_labels, title in zip(
		axes,
		(train_embedding, test_embedding),
		(train_labels, test_labels),
		("Training", "Test"),
	):
		scatter = ax.scatter(
			embedding[:, 0],
			embedding[:, 1],
			c=partition_labels,
			cmap="tab10",
			vmin=label_min,
			vmax=label_max,
			s=16,
			alpha=0.8,
		)
		ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
		ax.grid(alpha=0.2)

	fig.colorbar(
		scatter,
		ax=axes,
		orientation="horizontal",
		location="bottom",
		label="Class label",
		pad=0.12,
		fraction=0.08,
	)
	fig.suptitle(
		f"TabICL representations before soft-kNN (K={num_classes}, dataset={dataset_index})\n"
		f"TabICL accuracy: {tabicl_accuracy:.3f} | RF accuracy: {rf_accuracy:.3f} | "
		f"TabICL predictive entropy: {tabicl_entropy:.3f}"
	)
	output_dir.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_dir / f"umap_k{num_classes}_dataset{dataset_index:05d}.png")
	plt.close(fig)


def _evaluate_class_count(model, args: argparse.Namespace, num_classes: int, device: str) -> list[dict]:
	prior = PriorDataset(
		batch_size=args.batch_size,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=num_classes,
		min_seq_len=args.min_seq_len,
		max_seq_len=args.max_seq_len,
		min_train_size=args.min_train_size,
		max_train_size=args.max_train_size,
		prior_type=args.prior_type,
		device=args.prior_device,
		n_jobs=1,
		normalization=("std" if args.normalize_features and args.normalization == "none" else args.normalization),
	)

	rows: list[dict] = []
	pretrained_tabicl = _build_pretrained_tabicl(
		args.pretrained_tabicl_n_estimators,
		device,
		args.seed,
	)
	processed = 0
	batch_index = 0
	while processed < args.num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= args.num_datasets:
				break

			# The direct model path is intentional: it exercises the graph
			# backend exactly as represented by the supplied checkpoint.
			seq_len = int(batch[3][index].item())
			train_size = int(batch[4][index].item())
			y = batch[1][index, :seq_len].numpy().astype(int)
			y_train, y_test = y[:train_size], y[train_size:]
			if not _has_complete_label_support(y_train, y_test):
				# Such a dataset cannot be evaluated fairly because a classifier
				# fitted on the training split has no probability column for the
				# unseen test label.
				processed += 1
				continue

			sample = _evaluate_one_dataset(model, "tabicl", batch, index, device)
			x = sample.x.numpy()
			train_size = sample.train_size

			if args.skip_baselines:
				random_forest = BASELINE_FACTORIES["random_forest"](
					args.seed + processed, args.rf_n_estimators
				)
				random_forest.fit(
					np.nan_to_num(x[:train_size]), y[:train_size]
				)
				rf_prediction = random_forest.predict(np.nan_to_num(x[train_size:])).astype(int)
				rf_accuracy = _balanced_accuracy(y[train_size:], rf_prediction)
				if getattr(model.icl_predictor, "decoder_type", None) in ("soft_kmeans", "rbf", "euclidean"):
					tabicl_scores = sample.y_logits_test.float().exp().numpy()
				else:
					tabicl_scores = torch.softmax(sample.y_logits_test.float(), dim=-1).numpy()
				tabicl_entropy = _mean_entropy(tabicl_scores)
				try:
					_plot_umap(
						sample,
						num_classes,
						processed,
						args.output_dir / "umap",
						args.seed + processed,
						rf_accuracy,
						tabicl_entropy,
						args.umap_n_neighbors,
						args.umap_n_epochs,
					)
				except Exception as exc:
					print(f"Warning: UMAP plot failed for K={num_classes}, dataset={processed}: {exc}")
				processed += 1
				continue

			n_features = x.shape[1]
			x_train, x_test = x[:train_size], x[train_size:]

			if getattr(model.icl_predictor, "decoder_type", None) in ("soft_kmeans", "rbf", "euclidean"):
				# Kernel decoders already return class masses; applying another
				# softmax is incorrect.
				tabicl_scores = sample.y_logits_test.float().exp().numpy()
			else:
				tabicl_scores = torch.softmax(sample.y_logits_test.float(), dim=-1).numpy()
			tabicl_proba = _normalize_probabilities(tabicl_scores)
			predictions = {
				"tabicl": (
					sample.y_pred_test.numpy().astype(int),
					tabicl_proba,
					np.arange(tabicl_proba.shape[1]),
				)
			}
			for name, factory in BASELINE_FACTORIES.items():
				baseline = factory(args.seed + processed, args.rf_n_estimators)
				# RF versions differ in NaN support; replacing non-finite
				# values keeps this baseline portable across sklearn versions.
				baseline.fit(np.nan_to_num(x_train), y_train)
				predictions[name] = (
					baseline.predict(np.nan_to_num(x_test)).astype(int),
					baseline.predict_proba(np.nan_to_num(x_test)),
					baseline.classes_,
				)

			# Use the public sklearn-compatible API so the pretrained model applies
			# its own documented preprocessing, ensemble averaging, and label mapping.
			pretrained_tabicl.fit(x_train, y_train)
			pretrained_proba = pretrained_tabicl.predict_proba(x_test)
			predictions["pretrained_tabicl"] = (
				pretrained_tabicl.classes_[np.argmax(pretrained_proba, axis=1)].astype(int),
				pretrained_proba,
				pretrained_tabicl.classes_,
			)

			for model_name, (prediction, probabilities, class_labels) in predictions.items():
				for score_name, score_fn in SCORES.items():
					rows.append({
						"num_classes": num_classes,
						"dataset": processed,
						"batch": current_batch_index,
						"n_features": n_features,
						"model": model_name,
						"score": score_name,
						"value": score_fn(y_test, prediction, probabilities, class_labels),
					})
			processed += 1

	return rows


def _plot_results(rows: list[dict], output_dir: Path) -> None:
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	output_dir.mkdir(parents=True, exist_ok=True)
	by_classes: dict[int, list[dict]] = defaultdict(list)
	for row in rows:
		by_classes[row["num_classes"]].append(row)

	for num_classes, class_rows in sorted(by_classes.items()):
		for score_name in dict.fromkeys(row["score"] for row in class_rows):
			print(f"Making plots for K={num_classes}, score={score_name}")
			score_ylim = 4 if score_name in {"cross_entropy", "entropy"} else 1
			score_rows = [row for row in class_rows if row["score"] == score_name]
			models = list(dict.fromkeys(row["model"] for row in score_rows))
			values = {
				model: np.asarray([row["value"] for row in score_rows if row["model"] == model])
				for model in models
			}

			print(f"Box Plots")
			fig, ax = plt.subplots(figsize=(7, 5))
			boxplot = ax.boxplot(
				[values[model] for model in models],
				labels=models,
				patch_artist=True,
				showmeans=True,
			)
			for patch, color in zip(
				boxplot["boxes"], ["#4472C4", "#ED7D31", "#70AD47"][:len(models)]
			):
				patch.set_facecolor(color)
				patch.set_alpha(0.75)
			ax.set(title=f"K={num_classes}: {score_name} distribution", ylabel=score_name, ylim=(0, score_ylim))
			ax.grid(axis="y", alpha=0.25)
			fig.tight_layout()
			fig.savefig(output_dir / f"bar_{score_name}_k{num_classes}.png")
			plt.close(fig)

			tabicl = values["tabicl"]
			fig, ax = plt.subplots(figsize=(6, 6))
			print(f"Scatter Plots")
			for baseline in (model for model in models if model != "tabicl"):
				baseline_values = values[baseline]
				ax.scatter(tabicl, baseline_values, s=18, alpha=0.45, label=baseline)
			ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="equal performance")
			ax.set(
				title=f"K={num_classes}: TabICL vs baselines",
				xlabel="TabICL",
				ylabel="baseline",
				xlim=(0, score_ylim),
				ylim=(0, score_ylim),
			)
			ax.grid(alpha=0.25)
			ax.legend()
			fig.tight_layout()
			fig.savefig(output_dir / f"scatter_{score_name}_k{num_classes}.png")
			plt.close(fig)

			fig, ax = plt.subplots(figsize=(7, 5))
			print(f"Feature Count Plots")
			for model in models:
				feature_counts = np.asarray([
					row["n_features"] for row in score_rows if row["model"] == model
				])
				ax.scatter(feature_counts, values[model], s=18, alpha=0.45, label=model)
			ax.set(
				title=f"K={num_classes}: {score_name} versus number of features",
				xlabel="Number of features",
				ylabel=score_name,
				ylim=(0, score_ylim),
			)
			ax.grid(alpha=0.25)
			ax.legend()
			fig.tight_layout()
			fig.savefig(output_dir / f"features_vs_{score_name}_k{num_classes}.png")
			plt.close(fig)

		accuracy_rows = [row for row in class_rows if row["score"] == "balanced_accuracy"]
		entropy_rows = [row for row in class_rows if row["score"] == "entropy"]
		cross_entropy_rows = [row for row in class_rows if row["score"] == "cross_entropy"]
		if accuracy_rows and entropy_rows:
			accuracy = {
				(row["model"], row["dataset"]): row["value"] for row in accuracy_rows
			}
			entropy = {
				(row["model"], row["dataset"]): row["value"] for row in entropy_rows
			}
			tabicl_pairs = [
				(uncertainty, accuracy[key])
				for key, uncertainty in entropy.items()
				if key[0] == "tabicl" and key in accuracy
			]
			if tabicl_pairs:
				entropy_values, accuracy_values = map(np.asarray, zip(*tabicl_pairs))
				fig, ax = plt.subplots(figsize=(6, 5))
				ax.scatter(entropy_values, accuracy_values, s=18, alpha=0.45, label="tabicl")
				ax.set(
					title=f"K={num_classes}: TabICL accuracy versus entropy",
					xlabel="Mean predictive entropy",
					ylabel="Balanced accuracy",
					ylim=(0, 1),
				)
				ax.grid(alpha=0.25)
				ax.legend()
				fig.tight_layout()
				fig.savefig(output_dir / f"tabicl_accuracy_vs_entropy_k{num_classes}.png")
				plt.close(fig)

		if accuracy_rows and cross_entropy_rows:
			accuracy = {
				(row["model"], row["dataset"]): row["value"] for row in accuracy_rows
			}
			cross_entropy = {
				(row["model"], row["dataset"]): row["value"] for row in cross_entropy_rows
			}
			tabicl_pairs = [
				(cross_entropy_value, accuracy[key])
				for key, cross_entropy_value in cross_entropy.items()
				if key[0] == "tabicl" and key in accuracy
			]
			if tabicl_pairs:
				cross_entropy_values, accuracy_values = map(np.asarray, zip(*tabicl_pairs))
				fig, ax = plt.subplots(figsize=(6, 5))
				ax.scatter(cross_entropy_values, accuracy_values, s=18, alpha=0.45, label="tabicl")
				ax.set(
					title=f"K={num_classes}: TabICL accuracy versus cross entropy",
					xlabel="Cross entropy",
					ylabel="Balanced accuracy",
					ylim=(0, 1),
				)
				ax.grid(alpha=0.25)
				ax.legend()
				fig.tight_layout()
				fig.savefig(output_dir / f"tabicl_accuracy_vs_cross_entropy_k{num_classes}.png")
				plt.close(fig)

		_plot_batch_statistics(class_rows, num_classes, output_dir)


def _plot_batch_statistics(class_rows: list[dict], num_classes: int, output_dir: Path) -> None:
	"""Plot per-batch summary statistics for accuracy and cross-entropy."""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	for score_name in ("balanced_accuracy", "cross_entropy"):
		score_rows = [row for row in class_rows if row["score"] == score_name]
		models = list(dict.fromkeys(row["model"] for row in score_rows))
		fig, ax = plt.subplots(figsize=(10, 6))
		colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

		for model_index, model in enumerate(models):
			model_rows = [row for row in score_rows if row["model"] == model]
			by_batch: dict[int, list[float]] = defaultdict(list)
			for row in model_rows:
				by_batch[int(row["batch"])].append(float(row["value"]))
			if not by_batch:
				continue

			batch_indices = np.asarray(sorted(by_batch))
			values = [np.asarray(by_batch[int(batch)]) for batch in batch_indices]
			means = np.asarray([np.mean(batch_values) for batch_values in values])
			stds = np.asarray([np.std(batch_values) for batch_values in values])
			medians = np.asarray([np.median(batch_values) for batch_values in values])
			minimums = np.asarray([np.min(batch_values) for batch_values in values])
			maximums = np.asarray([np.max(batch_values) for batch_values in values])
			color = colors[model_index % len(colors)]

			ax.fill_between(
				batch_indices,
				minimums,
				maximums,
				color=color,
				alpha=0.08,
			)
			ax.errorbar(
				batch_indices,
				means,
				yerr=stds,
				fmt="o-",
				color=color,
				capsize=3,
				linewidth=1,
				label=f"{model}: mean ± std",
			)
			ax.plot(
				batch_indices,
				medians,
				"--",
				color=color,
				linewidth=1,
				label=f"{model}: median",
			)

		ylabel = "Balanced accuracy" if score_name == "balanced_accuracy" else "Cross-entropy"
		ax.set(
			title=f"Batch-wise {ylabel} statistics (K={num_classes})",
			xlabel="Batch index",
			ylabel=ylabel,
		)
		if score_name == "balanced_accuracy":
			ax.set_ylim(0.0, 1.0)
		ax.grid(alpha=0.25)
		ax.legend(ncol=2)
		fig.tight_layout()
		fig.savefig(output_dir / f"batch_{score_name}_k{num_classes}.png", dpi=150)
		plt.close(fig)


def main() -> None:
	args = build_parser().parse_args()
	if args.num_datasets < 1:
		raise ValueError("--num-datasets must be positive")
	if args.rf_n_estimators < 1:
		raise ValueError("--rf-n-estimators must be positive")
	if args.pretrained_tabicl_n_estimators < 1:
		raise ValueError("--pretrained-tabicl-n-estimators must be positive")
	if args.umap_n_neighbors < 2:
		raise ValueError("--umap-n-neighbors must be at least 2")
	if args.umap_n_epochs < 1:
		raise ValueError("--umap-n-epochs must be positive")
	if args.max_seq_len < 2:
		raise ValueError("--max-seq-len must be at least 2")
	if args.plot_results_only:
		results_path = args.results_output or args.output_dir / "results.json"
		if not results_path.exists():
			raise FileNotFoundError(f"Results file not found: {results_path}")
		rows = json.loads(results_path.read_text())
		_plot_results(rows, args.output_dir)
		print(f"Loaded results from {results_path}")
		print(f"Saved plots to {args.output_dir}")
		return
	if args.checkpoint is None:
		raise ValueError("checkpoint is required unless --plot-results-only is specified")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
	model, model_type = _build_model_from_checkpoint(str(args.checkpoint), device)
	if model_type != "tabicl" or getattr(model, "icl_backend", None) != "graph":
		raise ValueError("The checkpoint must contain a TabICL model with icl_backend='graph'.")

	rows = []
	for num_classes in range(10, 11):
		print(f"Evaluating {args.num_datasets} datasets for K={num_classes}...")
		rows.extend(_evaluate_class_count(model, args, num_classes, device))
	if args.skip_baselines:
		print(f"Generated UMAP plots for {args.num_datasets} datasets for each K=2,...,10")
		print(f"Saved UMAP plots to {args.output_dir / 'umap'}")
		return

	print(f"Making plots")
	output_dir = args.output_dir
	results_output = args.results_output or output_dir / "results.json"
	results_output.parent.mkdir(parents=True, exist_ok=True)
	results_output.write_text(json.dumps(rows, indent=2))
	_plot_results(rows, output_dir)
	print(f"Evaluated {args.num_datasets} datasets for each K=2,...,10")
	print(f"Saved results to {results_output}")
	print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
	main()
