"""Evaluate random forests on datasets sampled from the TabICL prior."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

from tabicl.prior._dataset import PriorDataset


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--num-datasets",
		"-N",
		type=int,
		default=1000,
		help="Number of prior datasets evaluated for each K (default: 1000).",
	)
	parser.add_argument(
		"--n-ensembles",
		type=int,
		default=100,
		help="Number of trees in each random forest (default: 100).",
	)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--min-features", type=int, default=2)
	parser.add_argument("--max-features", type=int, default=100)
	parser.add_argument("--min-seq-len", type=int, default=None)
	parser.add_argument("--max-seq-len", type=int, default=1024)
	parser.add_argument("--min-train-size", type=float, default=0.3)
	parser.add_argument("--max-train-size", type=float, default=0.9)
	parser.add_argument(
		"--prior-type",
		choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl"],
		default="nanotabicl",
	)
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--rf-jobs", type=int, default=-1, help="Random-forest parallelism.")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("prior_eval/accuracy_histogram.png"),
		help="Path of the output histogram image.",
	)
	return parser


def _regular_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
	"""Convert nested prior outputs to padded tensors when necessary."""
	return tuple(
		value.to_padded_tensor(padding=0.0) if value.is_nested else value
		for value in batch
	)


def _evaluate_class_count(
	num_classes: int,
	num_datasets: int,
	batch_size: int,
	n_ensembles: int,
	min_features: int,
	max_features: int,
	min_seq_len: int | None,
	max_seq_len: int,
	min_train_size: float,
	max_train_size: float,
	prior_type: str,
	prior_device: str,
	rf_jobs: int,
	seed: int,
) -> tuple[int, list[float], list[float]]:
	"""Sample and evaluate datasets for one class count in a worker process."""
	prior = PriorDataset(
		batch_size=batch_size,
		min_features=min_features,
		max_features=max_features,
		max_classes=num_classes,
		min_seq_len=min_seq_len,
		max_seq_len=max_seq_len,
		min_train_size=min_train_size,
		max_train_size=max_train_size,
		prior_type=prior_type,
		device=prior_device,
		n_jobs=1,
	)

	scores: list[float] = []
	base_frequencies: list[float] = []
	processed = 0
	while processed < num_datasets:
		batch = _regular_batch(prior.get_batch())
		x_batch, y_batch, feature_counts, seq_lens, train_sizes = batch
		for index in range(int(x_batch.shape[0])):
			if processed >= num_datasets:
				break

			seq_len = int(seq_lens[index].item())
			train_size = int(train_sizes[index].item())
			n_features = int(feature_counts[index].item())
			x = x_batch[index, :seq_len, :n_features].cpu().numpy()
			y = y_batch[index, :seq_len].cpu().numpy().astype(int)

			x_train, x_test = x[:train_size], x[train_size:]
			y_train, y_test = y[:train_size], y[train_size:]
			classifier = RandomForestClassifier(
				n_estimators=n_ensembles,
				random_state=seed + processed,
				n_jobs=rf_jobs,
			)
			classifier.fit(np.nan_to_num(x_train), y_train)
			prediction = classifier.predict(np.nan_to_num(x_test))
			scores.append(float(balanced_accuracy_score(y_test, prediction)))

			class_counts = np.bincount(y, minlength=num_classes)
			base_frequencies.append(float(class_counts.max() / class_counts.sum()))
			processed += 1

	return num_classes, scores, base_frequencies


def evaluate_prior(args: argparse.Namespace) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
	"""Sample and evaluate datasets for all K values in parallel."""
	worker_args = [
		(
			num_classes,
			args.num_datasets,
			args.batch_size,
			args.n_ensembles,
			args.min_features,
			args.max_features,
			args.min_seq_len,
			args.max_seq_len,
			args.min_train_size,
			args.max_train_size,
			args.prior_type,
			args.prior_device,
			args.rf_jobs,
			args.seed + num_classes * args.num_datasets,
		)
		for num_classes in range(2, 11)
	]
	with ProcessPoolExecutor(
		max_workers=len(worker_args),
		mp_context=mp.get_context("spawn"),
	) as executor:
		results = list(executor.map(_evaluate_class_count, *zip(*worker_args)))

	scores_by_k = {num_classes: scores for num_classes, scores, _ in results}
	base_frequencies_by_k = {
		num_classes: base_frequencies for num_classes, _, base_frequencies in results
	}
	for num_classes in sorted(scores_by_k):
		scores = scores_by_k[num_classes]
		print(
			f"K={num_classes}: n={len(scores)}, "
			f"mean balanced accuracy={np.mean(scores):.4f}, "
			f"std={np.std(scores):.4f}, "
			f"mean base frequency={np.mean(base_frequencies_by_k[num_classes]):.4f}"
		)
	return scores_by_k, base_frequencies_by_k


def plot_histograms(scores_by_k: dict[int, list[float]], output: Path) -> None:
	"""Save one accuracy histogram image for every K."""
	output.parent.mkdir(parents=True, exist_ok=True)
	bins = np.linspace(0.0, 1.0, 41)
	for num_classes, scores in scores_by_k.items():
		fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
		ax.hist(
			scores,
			bins=bins,
			density=False,
			edgecolor="black",
			linewidth=0.4,
		)
		ax.set(
			title=f"Random-forest balanced accuracy on prior datasets (K={num_classes})",
			xlabel="Balanced accuracy",
			ylabel="Number of datasets",
			xlim=(0.0, 1.0),
		)
		ax.grid(axis="y", alpha=0.25)
		fig.tight_layout()
		output_path = output.with_name(f"{output.stem}_k{num_classes}{output.suffix}")
		fig.savefig(output_path)
		plt.close(fig)


def plot_base_frequency_histograms(
	base_frequencies_by_k: dict[int, list[float]], output: Path
) -> None:
	"""Save one base-frequency histogram image for every K."""
	output.parent.mkdir(parents=True, exist_ok=True)
	bins = np.linspace(0.0, 1.0, 41)
	for num_classes, frequencies in base_frequencies_by_k.items():
		fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
		ax.hist(frequencies, bins=bins, edgecolor="black", linewidth=0.4)
		ax.set(
			title=f"Base class frequency on prior datasets (K={num_classes})",
			xlabel="Largest class frequency",
			ylabel="Number of datasets",
			xlim=(0.0, 1.0),
		)
		ax.grid(axis="y", alpha=0.25)
		fig.tight_layout()
		output_path = output.with_name(f"{output.stem}_base_frequency_k{num_classes}{output.suffix}")
		fig.savefig(output_path)
		plt.close(fig)


def plot_base_frequency_scatter(
	scores_by_k: dict[int, list[float]],
	base_frequencies_by_k: dict[int, list[float]],
	output: Path,
) -> None:
	"""Save a scatter plot of base frequency versus RF accuracy."""
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
	for num_classes in sorted(scores_by_k):
		ax.scatter(
			base_frequencies_by_k[num_classes],
			scores_by_k[num_classes],
			alpha=0.45,
			s=16,
			label=f"K={num_classes}",
		)
	ax.set(
		title="Random-forest accuracy versus base class frequency",
		xlabel="Largest class frequency",
		ylabel="Balanced accuracy",
		xlim=(0.0, 1.0),
		ylim=(0.0, 1.0),
	)
	ax.grid(alpha=0.25)
	ax.legend(title="Number of classes", ncol=2)
	fig.tight_layout()
	output_path = output.with_name(f"{output.stem}_base_frequency_vs_accuracy{output.suffix}")
	fig.savefig(output_path)
	plt.close(fig)


def main() -> None:
	args = build_parser().parse_args()
	if args.num_datasets < 1:
		raise ValueError("--num-datasets must be positive")
	if args.n_ensembles < 1:
		raise ValueError("--n-ensembles must be positive")
	if args.batch_size < 1:
		raise ValueError("--batch-size must be positive")

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	started_at = time.perf_counter()
	scores_by_k, base_frequencies_by_k = evaluate_prior(args)
	plot_histograms(scores_by_k, args.output)
	plot_base_frequency_histograms(base_frequencies_by_k, args.output)
	plot_base_frequency_scatter(scores_by_k, base_frequencies_by_k, args.output)
	elapsed = time.perf_counter() - started_at
	print(f"Saved evaluation plots with prefix {args.output}")
	print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60.0:.2f} minutes)")


if __name__ == "__main__":
	main()
