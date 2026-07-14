"""Evaluate random forests on datasets sampled from the TabICL prior."""

from __future__ import annotations

import argparse
import time
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
	parser.add_argument("--max-seq-len", type=int, default=512)
	parser.add_argument("--min-train-size", type=float, default=0.1)
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


def evaluate_prior(args: argparse.Namespace) -> dict[int, list[float]]:
	"""Sample and evaluate ``args.num_datasets`` datasets for every K."""
	prior = PriorDataset(
		batch_size=args.batch_size,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=10,
		min_seq_len=args.min_seq_len,
		max_seq_len=args.max_seq_len,
		min_train_size=args.min_train_size,
		max_train_size=args.max_train_size,
		prior_type=args.prior_type,
		device=args.prior_device,
		n_jobs=1,
	)

	scores_by_k: dict[int, list[float]] = {}
	for num_classes in range(2, 11):
		# PriorDataset samples the exact number of classes when max_classes is
		# set to K. A fresh instance also makes the K loop explicit and keeps
		# the evaluation independent if the prior implementation changes.
		prior.max_classes = num_classes
		prior.prior.max_classes = num_classes
		scores: list[float] = []
		processed = 0

		while processed < args.num_datasets:
			batch = _regular_batch(prior.get_batch())
			x_batch, y_batch, feature_counts, seq_lens, train_sizes = batch

			for index in range(int(x_batch.shape[0])):
				if processed >= args.num_datasets:
					break

				seq_len = int(seq_lens[index].item())
				train_size = int(train_sizes[index].item())
				n_features = int(feature_counts[index].item())
				x = x_batch[index, :seq_len, :n_features].cpu().numpy()
				y = y_batch[index, :seq_len].cpu().numpy().astype(int)

				x_train, x_test = x[:train_size], x[train_size:]
				y_train, y_test = y[:train_size], y[train_size:]
				classifier = RandomForestClassifier(
					n_estimators=args.n_ensembles,
					random_state=args.seed + processed,
					n_jobs=args.rf_jobs,
				)
				classifier.fit(np.nan_to_num(x_train), y_train)
				prediction = classifier.predict(np.nan_to_num(x_test))
				scores.append(float(balanced_accuracy_score(y_test, prediction)))
				processed += 1

		scores_by_k[num_classes] = scores
		print(
			f"K={num_classes}: n={len(scores)}, "
			f"mean balanced accuracy={np.mean(scores):.4f}, "
			f"std={np.std(scores):.4f}"
		)

	return scores_by_k


def plot_histograms(scores_by_k: dict[int, list[float]], output: Path) -> None:
	"""Save overlaid accuracy histograms, one distribution for every K."""
	output.parent.mkdir(parents=True, exist_ok=True)
	fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
	bins = np.linspace(0.0, 1.0, 21)
	for num_classes, scores in scores_by_k.items():
		ax.hist(
			scores,
			bins=bins,
			alpha=0.35,
			density=False,
			label=f"K={num_classes}",
			edgecolor="black",
			linewidth=0.4,
		)
	ax.set(
		title="Random-forest balanced accuracy on prior datasets",
		xlabel="Balanced accuracy",
		ylabel="Number of datasets",
		xlim=(0.0, 1.0),
	)
	ax.grid(axis="y", alpha=0.25)
	ax.legend(ncol=3)
	fig.tight_layout()
	fig.savefig(output)
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
	scores_by_k = evaluate_prior(args)
	plot_histograms(scores_by_k, args.output)
	elapsed = time.perf_counter() - started_at
	print(f"Saved histogram to {args.output}")
	print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60.0:.2f} minutes)")


if __name__ == "__main__":
	main()
