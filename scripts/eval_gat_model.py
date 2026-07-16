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
	"""Compute cross entropy while accounting for classes absent during fitting."""
	all_labels = np.unique(np.concatenate((y_true, np.asarray(labels))))
	probabilities = np.zeros((y_proba.shape[0], all_labels.size), dtype=y_proba.dtype)
	columns = np.searchsorted(all_labels, labels)
	probabilities[:, columns] = y_proba
	return float(log_loss(y_true, probabilities, labels=all_labels))


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
		"--normalize-features",
		action="store_true",
		help="Normalize each feature using its training-sample mean and standard deviation.",
	)
	parser.add_argument(
		"--rf-n-estimators",
		type=int,
		default=50,
		help="Number of RF trees per dataset; lower values speed up diagnostics (default: 50)",
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


def _plot_umap(
	sample,
	num_classes: int,
	dataset_index: int,
	output_dir: Path,
	seed: int,
	rf_accuracy: float,
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
		f"TabICL accuracy: {tabicl_accuracy:.3f} | RF accuracy: {rf_accuracy:.3f}"
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
	)

	rows: list[dict] = []
	processed = 0
	while processed < args.num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		if args.normalize_features:
			batch = _normalize_features(batch)
		for index in range(int(batch[0].shape[0])):
			if processed >= args.num_datasets:
				break

			# The direct model path is intentional: it exercises the graph
			# backend exactly as represented by the supplied checkpoint.
			sample = _evaluate_one_dataset(model, "tabicl", batch, index, device)
			if args.skip_baselines:
				x = sample.x.numpy()
				y = sample.y_full.numpy().astype(int)
				train_size = sample.train_size
				random_forest = BASELINE_FACTORIES["random_forest"](
					args.seed + processed, args.rf_n_estimators
				)
				random_forest.fit(
					np.nan_to_num(x[:train_size]), y[:train_size]
				)
				rf_prediction = random_forest.predict(np.nan_to_num(x[train_size:])).astype(int)
				rf_accuracy = _balanced_accuracy(y[train_size:], rf_prediction)
				_plot_umap(
					sample,
					num_classes,
					processed,
					args.output_dir / "umap",
					args.seed + processed,
					rf_accuracy,
					args.umap_n_neighbors,
					args.umap_n_epochs,
				)
				processed += 1
				continue

			x = sample.x.numpy()
			y = sample.y_full.numpy().astype(int)
			train_size = sample.train_size
			n_features = x.shape[1]
			x_train, x_test = x[:train_size], x[train_size:]
			y_train, y_test = y[:train_size], y[train_size:]

			tabicl_proba = torch.softmax(sample.y_logits_test, dim=-1).numpy()
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

			for model_name, (prediction, probabilities, class_labels) in predictions.items():
				for score_name, score_fn in SCORES.items():
					rows.append({
						"num_classes": num_classes,
						"dataset": processed,
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
			score_ylim = 2.3 if score_name == "cross_entropy" else 1
			score_rows = [row for row in class_rows if row["score"] == score_name]
			models = list(dict.fromkeys(row["model"] for row in score_rows))
			values = {
				model: np.asarray([row["value"] for row in score_rows if row["model"] == model])
				for model in models
			}

			print(f"Bar Plots")
			fig, ax = plt.subplots(figsize=(7, 5))
			means = [values[model].mean() for model in models]
			stds = [values[model].std() for model in models]
			ax.bar(models, means, yerr=stds, capsize=5, color=["#4472C4", "#ED7D31"][:len(models)])
			ax.set(title=f"K={num_classes}: mean {score_name}", ylabel=score_name, ylim=(0, score_ylim))
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


def main() -> None:
	args = build_parser().parse_args()
	if args.num_datasets < 1:
		raise ValueError("--num-datasets must be positive")
	if args.rf_n_estimators < 1:
		raise ValueError("--rf-n-estimators must be positive")
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
	for num_classes in range(2, 5):
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
