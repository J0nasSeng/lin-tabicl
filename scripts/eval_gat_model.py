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
BASELINE_FACTORIES: dict[str, Callable[[int], object]] = {
	"random_forest": lambda seed: RandomForestClassifier(
		n_estimators=200, random_state=seed, n_jobs=-1
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
	parser.add_argument("--max-features", type=int, default=100)
	parser.add_argument("--max-seq-len", type=int, default=512)
	parser.add_argument("--min-seq-len", type=int, default=None)
	parser.add_argument("--min-train-size", type=float, default=0.1)
	parser.add_argument("--max-train-size", type=float, default=0.9)
	parser.add_argument("--prior-type", choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl"], default="nanotabicl")
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--device", default=None, help="Inference device (default: cuda when available)")
	parser.add_argument("--seed", type=int, default=42)
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


def _plot_umap(sample, num_classes: int, dataset_index: int, output_dir: Path, seed: int) -> None:
	"""Plot sample representations before the TabICL soft-kNN prediction layer."""
	try:
		import matplotlib.pyplot as plt
	except ImportError as exc:  # pragma: no cover - runtime dependency
		raise ModuleNotFoundError("matplotlib is required for UMAP plots") from exc

	try:
		from umap import UMAP
	except ImportError as exc:  # pragma: no cover - runtime dependency
		raise ModuleNotFoundError("umap-learn is required for UMAP plots") from exc

	representations = sample.repr_full.numpy()
	labels = sample.y_full.numpy().astype(int)
	balanced_accuracy = _balanced_accuracy(
		sample.y_true_test.numpy().astype(int), sample.y_pred_test.numpy().astype(int)
	)
	if representations.shape[0] < 3:
		return

	n_neighbors = min(15, representations.shape[0] - 1)
	embeddings = UMAP(
		n_components=2,
		n_neighbors=n_neighbors,
		min_dist=0.1,
		metric="euclidean",
		random_state=seed,
	).fit_transform(representations)

	fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
	scatter = ax.scatter(
		embeddings[:, 0],
		embeddings[:, 1],
		c=labels,
		cmap="tab10",
		s=16,
		alpha=0.8,
	)
	fig.colorbar(scatter, ax=ax, label="Class label")
	ax.set(
		title=f"TabICL sample representations before soft-kNN (K={num_classes}, dataset={dataset_index})",
		xlabel="UMAP-1",
		ylabel="UMAP-2",
	)
	ax.text(
		0.02,
		0.98,
		f"Balanced accuracy: {balanced_accuracy:.3f}",
		transform=ax.transAxes,
		va="top",
		bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
	)
	ax.grid(alpha=0.2)
	fig.tight_layout()
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
		for index in range(int(batch[0].shape[0])):
			if processed >= args.num_datasets:
				break

			# The direct model path is intentional: it exercises the graph
			# backend exactly as represented by the supplied checkpoint.
			sample = _evaluate_one_dataset(model, "tabicl", batch, index, device)
			if args.skip_baselines:
				_plot_umap(sample, num_classes, processed, args.output_dir / "umap", args.seed + processed)
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
				baseline = factory(args.seed + processed)
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
	import matplotlib.pyplot as plt

	output_dir.mkdir(parents=True, exist_ok=True)
	by_classes: dict[int, list[dict]] = defaultdict(list)
	for row in rows:
		by_classes[row["num_classes"]].append(row)

	for num_classes, class_rows in sorted(by_classes.items()):
		for score_name in dict.fromkeys(row["score"] for row in class_rows):
			score_ylim = 2.3 if score_name == "cross_entropy" else 1
			score_rows = [row for row in class_rows if row["score"] == score_name]
			models = list(dict.fromkeys(row["model"] for row in score_rows))
			values = {
				model: np.asarray([row["value"] for row in score_rows if row["model"] == model])
				for model in models
			}

			fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
			means = [values[model].mean() for model in models]
			stds = [values[model].std() for model in models]
			ax.bar(models, means, yerr=stds, capsize=5, color=["#4472C4", "#ED7D31"][:len(models)])
			ax.set(title=f"K={num_classes}: mean {score_name}", ylabel=score_name, ylim=(0, score_ylim))
			ax.grid(axis="y", alpha=0.25)
			fig.tight_layout()
			fig.savefig(output_dir / f"bar_{score_name}_k{num_classes}.png")
			plt.close(fig)

			tabicl = values["tabicl"]
			fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
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

			fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
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
	for num_classes in range(2, 11):
		rows.extend(_evaluate_class_count(model, args, num_classes, device))
	if args.skip_baselines:
		print(f"Generated UMAP plots for {args.num_datasets} datasets for each K=2,...,10")
		print(f"Saved UMAP plots to {args.output_dir / 'umap'}")
		return

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
