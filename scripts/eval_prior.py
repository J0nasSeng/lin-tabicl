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
	parser.add_argument("--max-seq-len", type=int, default=1024)
	parser.add_argument("--min-train-size", type=float, default=0.3)
	parser.add_argument("--max-train-size", type=float, default=0.9)
	parser.add_argument(
		"--prior-type",
		choices=["mlp_scm", "tree_scm", "mix_scm", "nanotabicl", "graph_scm"],
		default="graph_scm",
	)
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--rf-jobs", type=int, default=-1, help="Random-forest parallelism.")
	parser.add_argument(
		"--normalization",
		choices=("none", "std", "robust"),
		default="none",
		help="Optional feature normalization applied before evaluation (default: none).",
	)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("prior_eval/accuracy_histogram.png"),
		help="Path of the output histogram image.",
	)
	parser.add_argument(
		"--basic-stats",
		action="store_true",
		help="Compute and plot basic feature statistics instead of random-forest accuracy.",
	)
	return parser


def _regular_batch(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
	"""Convert nested prior outputs to padded tensors when necessary."""
	return tuple(
		value.to_padded_tensor(padding=0.0) if isinstance(value, torch.Tensor) and value.is_nested else value
		for value in batch
	)


def _evaluate_class_count(
	args: argparse.Namespace, num_classes: int
) -> tuple[list[float], list[float], list[tuple[float, float, float, float, float]]]:
	"""Sample and evaluate datasets for one class count."""
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
		normalization=args.normalization,
		graph_backend=False,
	)

	scores: list[float] = []
	base_frequencies: list[float] = []
	batch_statistics: list[tuple[float, float, float, float, float]] = []
	processed = 0
	while processed < args.num_datasets:
		batch = _regular_batch(prior.get_batch())
		x_batch, y_batch, feature_counts, seq_lens, train_sizes = batch
		batch_scores: list[float] = []
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
				random_state=args.seed + num_classes * args.num_datasets + processed,
				n_jobs=args.rf_jobs,
			)
			classifier.fit(np.nan_to_num(x_train), y_train)
			prediction = classifier.predict(np.nan_to_num(x_test))
			score = float(balanced_accuracy_score(y_test, prediction))
			scores.append(score)
			batch_scores.append(score)

			class_counts = np.bincount(y, minlength=num_classes)
			base_frequencies.append(float(class_counts.max() / class_counts.sum()))
			processed += 1
		if batch_scores:
			batch_statistics.append(
				(
					float(np.mean(batch_scores)),
					float(np.std(batch_scores)),
					float(np.median(batch_scores)),
					float(np.min(batch_scores)),
					float(np.max(batch_scores)),
				)
			)

	return scores, base_frequencies, batch_statistics


def _is_discrete_feature(values: np.ndarray) -> bool:
	"""Classify an observed feature as discrete using its value cardinality.

	The prior returns all features as floating-point tensors and does not expose
	the latent categorical metadata. Consequently, basic-statistics evaluation
	uses a conservative observed-cardinality heuristic: a feature is discrete if
	it has at most 20 distinct finite values.
	"""
	finite_values = values[np.isfinite(values)]
	return np.unique(finite_values).size <= 20


def _skew(values: np.ndarray) -> float:
	"""Compute the moment coefficient of skewness for finite values."""
	values = values[np.isfinite(values)]
	if values.size < 2:
		return float("nan")
	centered = values - np.mean(values)
	std = np.std(values)
	if std == 0:
		return 0.0
	return float(np.mean(centered**3) / std**3)


def evaluate_basic_stats(args: argparse.Namespace) -> dict[str, list[float]]:
	"""Sample prior datasets and summarize their feature distributions."""
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
		normalization=args.normalization,
		graph_backend=False,
	)

	statistics = {
		"continuous features": [],
		"discrete features": [],
		"feature mean": [],
		"feature std": [],
		"feature median": [],
		"feature min": [],
		"feature max": [],
		"continuous feature skew": [],
	}
	grouped_statistics = {
		"continuous features": {
			name: []
			for name in statistics
			if name not in ("continuous features", "discrete features", "continuous feature skew")
		},
		"discrete features": {
			name: []
			for name in statistics
			if name not in ("continuous features", "discrete features", "continuous feature skew")
		},
	}
	processed = 0
	while processed < args.num_datasets:
		x_batch, _, feature_counts, seq_lens, train_sizes = _regular_batch(prior.get_batch())
		for index in range(int(x_batch.shape[0])):
			if processed >= args.num_datasets:
				break

			seq_len = int(seq_lens[index].item())
			n_features = int(feature_counts[index].item())
			x = x_batch[index, :seq_len, :n_features].cpu().numpy().astype(np.float64)
			finite_x = x[np.isfinite(x)]
			discrete_mask = np.asarray([_is_discrete_feature(x[:, feature]) for feature in range(n_features)])
			continuous_mask = ~discrete_mask
			discrete_count = int(discrete_mask.sum())

			statistics["continuous features"].append(float(n_features - discrete_count))
			statistics["discrete features"].append(float(discrete_count))
			statistics["feature mean"].append(float(np.mean(finite_x)))
			statistics["feature std"].append(float(np.std(finite_x)))
			statistics["feature median"].append(float(np.median(finite_x)))
			statistics["feature min"].append(float(np.min(finite_x)))
			statistics["feature max"].append(float(np.max(finite_x)))
			statistics["continuous feature skew"].append(_skew(x[:, continuous_mask]))

			for group_name, feature_mask in (
				("continuous features", continuous_mask),
				("discrete features", discrete_mask),
			):
				group_values = x[:, feature_mask]
				group_values = group_values[np.isfinite(group_values)]
				for statistic_name in grouped_statistics[group_name]:
					if group_values.size == 0:
						value = np.nan
					elif statistic_name == "feature mean":
						value = np.mean(group_values)
					elif statistic_name == "feature std":
						value = np.std(group_values)
					elif statistic_name == "feature median":
						value = np.median(group_values)
					elif statistic_name == "feature min":
						value = np.min(group_values)
					else:
						value = np.max(group_values)
					grouped_statistics[group_name][statistic_name].append(float(value))
			processed += 1

	plot_basic_statistics(statistics, grouped_statistics, args.output)
	for name, values in statistics.items():
		print(
			f"{name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
			f"median={np.median(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}"
		)
	return statistics


def plot_basic_statistics(
	statistics: dict[str, list[float]],
	grouped_statistics: dict[str, dict[str, list[float]]],
	output: Path,
) -> None:
	"""Plot dataset-averaged feature statistics on separate y-axes."""
	output.parent.mkdir(parents=True, exist_ok=True)
	feature_count_names = ["continuous features", "discrete features"]
	distribution_names = [
		"feature mean",
		"feature std",
		"feature median",
		"feature min",
		"feature max",
		"continuous feature skew",
	]
	grouped_distribution_names = [
		"feature mean",
		"feature std",
		"feature median",
		"feature min",
		"feature max",
	]

	fig, (count_ax, distribution_ax, grouped_ax) = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
	for ax, names, title, ylabel in (
		(count_ax, feature_count_names, "Feature-type counts", "Number of features"),
		(distribution_ax, distribution_names, "Feature-value statistics", "Value (dataset-level statistic)"),
	):
		means = np.asarray([np.mean(statistics[name]) for name in names])
		stds = np.asarray([np.std(statistics[name]) for name in names])
		medians = np.asarray([np.median(statistics[name]) for name in names])
		minimums = np.asarray([np.min(statistics[name]) for name in names])
		maximums = np.asarray([np.max(statistics[name]) for name in names])
		positions = np.arange(len(names))

		ax.bar(positions, means, color="tab:blue", alpha=0.7, edgecolor="black")
		ax.errorbar(
			positions,
			means,
			yerr=stds,
			fmt="none",
			ecolor="black",
			capsize=5,
			label="Mean ± std",
		)
		ax.scatter(positions, medians, marker="^", color="tab:green", zorder=3, label="Median")
		for position, minimum, maximum in zip(positions, minimums, maximums):
			ax.vlines(
				position,
				minimum,
				maximum,
				color="tab:red",
				linewidth=1.5,
				label="Min–max" if position == 0 else None,
			)
			ax.plot(position, minimum, "_", color="tab:red", markersize=10)
			ax.plot(position, maximum, "_", color="tab:red", markersize=10)
		ax.set_xticks(positions, names, rotation=25, ha="right")
		ax.set_title(title)
		ax.set_ylabel(ylabel)
		ax.grid(axis="y", alpha=0.25)
		ax.legend()

	positions = np.arange(len(grouped_distribution_names))
	width = 0.36
	for offset, (group_name, color) in zip((-width / 2, width / 2), (("continuous features", "tab:orange"), ("discrete features", "tab:purple"))):
		means = np.asarray([np.nanmean(grouped_statistics[group_name][name]) for name in grouped_distribution_names])
		stds = np.asarray([np.nanstd(grouped_statistics[group_name][name]) for name in grouped_distribution_names])
		medians = np.asarray([np.nanmedian(grouped_statistics[group_name][name]) for name in grouped_distribution_names])
		minimums = np.asarray([np.nanmin(grouped_statistics[group_name][name]) for name in grouped_distribution_names])
		maximums = np.asarray([np.nanmax(grouped_statistics[group_name][name]) for name in grouped_distribution_names])
		bar_positions = positions + offset
		grouped_ax.bar(bar_positions, means, width=width, color=color, alpha=0.7, edgecolor="black", label=group_name.title())
		grouped_ax.errorbar(bar_positions, means, yerr=stds, fmt="none", ecolor="black", capsize=4)
		grouped_ax.scatter(bar_positions, medians, marker="^", color="tab:green", zorder=3)
		for position, minimum, maximum in zip(bar_positions, minimums, maximums):
			grouped_ax.vlines(position, minimum, maximum, color="tab:red", linewidth=1.2)
			grouped_ax.plot(position, minimum, "_", color="tab:red", markersize=8)
			grouped_ax.plot(position, maximum, "_", color="tab:red", markersize=8)
	grouped_ax.set_xticks(positions, grouped_distribution_names, rotation=25, ha="right")
	grouped_ax.set_title("Statistics by feature type")
	grouped_ax.set_ylabel("Value (dataset-level statistic)")
	grouped_ax.grid(axis="y", alpha=0.25)
	grouped_ax.legend()

	fig.tight_layout()
	output_path = output.with_name(f"{output.stem}_basic_stats{output.suffix}")
	fig.savefig(output_path)
	plt.close(fig)



def evaluate_prior(args: argparse.Namespace) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
	"""Evaluate each K sequentially and plot it as soon as it is available."""
	scores_by_k: dict[int, list[float]] = {}
	base_frequencies_by_k: dict[int, list[float]] = {}
	batch_statistics_by_k: dict[int, list[tuple[float, float, float, float, float]]] = {}
	for num_classes in range(10, 11):
		scores, base_frequencies, batch_statistics = _evaluate_class_count(args, num_classes)
		scores_by_k[num_classes] = scores
		base_frequencies_by_k[num_classes] = base_frequencies
		batch_statistics_by_k[num_classes] = batch_statistics
		print(
			f"K={num_classes}: n={len(scores)}, "
			f"mean balanced accuracy={np.mean(scores):.4f}, "
			f"std={np.std(scores):.4f}, "
			f"mean base frequency={np.mean(base_frequencies):.4f}"
		)
		plot_histograms({num_classes: scores}, args.output)
		plot_base_frequency_histograms({num_classes: base_frequencies}, args.output)
		plot_base_frequency_scatter(scores_by_k, base_frequencies_by_k, args.output)
		plot_batch_accuracy_statistics({num_classes: batch_statistics}, args.output)
	return scores_by_k, base_frequencies_by_k


def plot_batch_accuracy_statistics(
	batch_statistics_by_k: dict[int, list[tuple[float, float, float, float, float]]], output: Path
) -> None:
	"""Save batch-wise accuracy statistics with standard-deviation and range bands."""
	output.parent.mkdir(parents=True, exist_ok=True)
	for num_classes, statistics in batch_statistics_by_k.items():
		if not statistics:
			continue
		means = np.asarray([mean for mean, _, _, _, _ in statistics])
		stds = np.asarray([std for _, std, _, _, _ in statistics])
		medians = np.asarray([median for _, _, median, _, _ in statistics])
		minimums = np.asarray([minimum for _, _, _, minimum, _ in statistics])
		maximums = np.asarray([maximum for _, _, _, _, maximum in statistics])
		batch_indices = np.arange(len(statistics))

		fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
		ax.fill_between(
			batch_indices,
			minimums,
			maximums,
			alpha=0.15,
			color="tab:gray",
			label="Min–max",
		)
		ax.errorbar(
			batch_indices,
			means,
			yerr=stds,
			fmt="o-",
			capsize=3,
			linewidth=1,
			label="Mean ± std",
		)
		ax.plot(batch_indices, medians, "s--", linewidth=1, label="Median")
		ax.set(
			title=f"Batch-wise random-forest accuracy (K={num_classes})",
			xlabel="Batch index",
			ylabel="Balanced accuracy (mean ± std)",
			xlim=(-0.5, len(statistics) - 0.5),
			ylim=(0.0, 1.0),
		)
		ax.grid(alpha=0.25)
		ax.legend()
		fig.tight_layout()
		output_path = output.with_name(f"{output.stem}_batch_accuracy_k{num_classes}{output.suffix}")
		fig.savefig(output_path)
		plt.close(fig)


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
	if args.basic_stats:
		evaluate_basic_stats(args)
	else:
		evaluate_prior(args)
	elapsed = time.perf_counter() - started_at
	print(f"Saved evaluation plots with prefix {args.output}")
	print(f"Total runtime: {elapsed:.2f} seconds ({elapsed / 60.0:.2f} minutes)")


if __name__ == "__main__":
	main()
