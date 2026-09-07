"""Analyze and plot TabArena per-split ROC AUC results.

The TabArena CSV stores metric error, so for ROC AUC this script reports
``1 - metric_error`` after averaging over folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHODS = {
	"TabICL graph-1d": "[New] TabICLGraph_c1_default",
	"TabICLv2": "TABICLV2 (default)",
	"TabPFN 2.6": "TABPFN-V2.6 (default)",
	"TabFM": "TABFM (default)",
	"XGBoost": "XGB (default)",
	"Random forest": "RF (default)",
	"Linear": "LR (default)",
	"KNN": "KNN (default)",
}

PROPERTY_COLUMNS = ("num_instances", "num_features", "num_classes")


def load_results(path: Path) -> pd.DataFrame:
	results = pd.read_csv(path, low_memory=False)
	required = {"dataset", "fold", "method", "metric", "metric_error"}
	missing = required - set(results.columns)
	if missing:
		raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

	results = results[results["metric"].eq("roc_auc")].copy()
	results["roc_auc"] = 1.0 - pd.to_numeric(results["metric_error"], errors="coerce")
	return results


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
	selected = results[results["method"].isin(METHODS.values())].copy()
	selected["method_label"] = selected["method"].map(
		{source: label for label, source in METHODS.items()}
	)
	summary = (
		selected.groupby(["dataset", "method_label"], as_index=False)
		.agg(roc_auc=("roc_auc", "mean"), folds=("fold", "nunique"))
	)
	return summary


def load_dataset_metadata(preset: str = "TabArena-v0.1") -> pd.DataFrame:
	"""Load one row of TabArena metadata per dataset, without downloading data."""
	from tabarena.benchmark.task.metadata import TaskMetadataCollection

	metadata = TaskMetadataCollection.from_preset(preset).per_dataset_frame()
	return metadata[["dataset", *PROPERTY_COLUMNS]]


def add_dataset_metadata(summary: pd.DataFrame, preset: str = "TabArena-v0.1") -> pd.DataFrame:
	metadata = load_dataset_metadata(preset)
	merged = summary.merge(metadata, on="dataset", how="left", validate="many_to_one")
	missing = merged.loc[merged[list(PROPERTY_COLUMNS)].isna().all(axis=1), "dataset"].unique()
	if len(missing):
		print(f"Warning: metadata not found for {len(missing)} dataset(s): {', '.join(missing)}")
	return merged


def plot_results(summary: pd.DataFrame, output: Path) -> None:
	wide = summary.pivot(index="dataset", columns="method_label", values="roc_auc")
	order = wide["TabICL graph-1d"].sort_values(na_position="last").index
	wide = wide.reindex(order)

	fig_width = max(12, len(wide.index) * 0.42)
	fig, axis = plt.subplots(figsize=(fig_width, 6.5))
	for label in METHODS:
		if label in wide:
			alpha = 1.0 if label == "TabICL graph-1d" else 0.3
			axis.plot(
				wide.index,
				wide[label],
				marker="o",
				linewidth=1.5,
				markersize=3,
				label=label,
				alpha=alpha,
			)

	axis.set_xlabel("Dataset")
	axis.set_ylabel("ROC AUC")
	axis.set_ylim(0.0, 1.0)
	axis.grid(axis="y", alpha=0.25)
	axis.legend()
	axis.tick_params(axis="x", labelrotation=75)
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=180)
	plt.close(fig)


def plot_results_by_property(summary: pd.DataFrame, property_name: str, output: Path) -> None:
	"""Plot dataset ROC AUC against one numeric dataset property."""
	property_values = pd.to_numeric(summary[property_name], errors="coerce")
	plot_data = summary.loc[property_values.notna()].copy()
	plot_data[property_name] = property_values[property_values.notna()]
	if plot_data.empty:
		print(f"Warning: no values available for {property_name}; skipping {output}")
		return

	fig, axis = plt.subplots(figsize=(8.5, 6.5))
	for label in METHODS:
		method_data = plot_data[plot_data["method_label"].eq(label)]
		if method_data.empty:
			continue
		alpha = 1.0 if label == "TabICL graph-1d" else 0.3
		axis.scatter(
			method_data[property_name],
			method_data["roc_auc"],
			label=label,
			alpha=alpha,
			s=24,
		)

	axis.set_xlabel(property_name.replace("_", " ").title())
	axis.set_ylabel("ROC AUC")
	axis.set_ylim(0.0, 1.0)
	if property_name in {"num_instances", "num_features"} and (plot_data[property_name] > 0).all():
		axis.set_xscale("log")
	axis.grid(alpha=0.25)
	axis.legend()
	fig.tight_layout()
	output.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output, dpi=180)
	plt.close(fig)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--input",
		type=Path,
		default=Path("eval/tabicl_graph/results_per_split.csv"),
		help="Per-split TabArena results CSV.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("eval/tabicl_graph/roc_auc_by_dataset.png"),
		help="Output path for the ROC AUC plot.",
	)
	parser.add_argument(
		"--summary-output",
		type=Path,
		default=Path("eval/tabicl_graph/roc_auc_by_dataset.csv"),
		help="Output path for the aggregated ROC AUC CSV.",
	)
	parser.add_argument(
		"--metadata-preset",
		default="TabArena-v0.1",
		help="TabArena metadata preset used for dataset properties.",
	)
	args = parser.parse_args()

	summary = summarize_results(load_results(args.input))
	if summary.empty:
		raise ValueError("None of the requested methods were found in the input CSV")
	summary = add_dataset_metadata(summary, args.metadata_preset)
	summary.to_csv(args.summary_output, index=False)
	plot_results(summary, args.output)
	for property_name in PROPERTY_COLUMNS:
		property_output = args.output.with_name(f"{args.output.stem}_by_{property_name}{args.output.suffix}")
		plot_results_by_property(summary, property_name, property_output)

	printed = summary.pivot(index="dataset", columns="method_label", values="roc_auc")
	difference = printed["TabICL graph-1d"].sub(printed["TabICLv2"]).abs()
	printed = printed.loc[difference.sort_values(ascending=False, na_position="last").index]
	print(printed.to_string())
	print(f"Plot written to {args.output}")
	print(f"Property plots written next to {args.output}")
	print(f"Summary written to {args.summary_output}")


if __name__ == "__main__":
	main()
