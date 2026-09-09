"""Analyze and plot TabArena per-split ROC AUC results.

The TabArena CSV stores metric error, so for ROC AUC this script reports
``1 - metric_error`` after averaging over folds.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

METADATA_PROPERTY_COLUMNS = ("num_instances", "num_features", "num_classes")
DISCRETE_CARDINALITY_COLUMNS = (
	"discrete_max_cardinality",
	"discrete_average_cardinality",
)
PROPERTY_COLUMNS = (*METADATA_PROPERTY_COLUMNS, "imbalance_factor", *DISCRETE_CARDINALITY_COLUMNS)


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
	grouped = selected.groupby(["dataset", "method_label"], as_index=False)
	summary = grouped.agg(roc_auc=("roc_auc", "mean"), folds=("fold", "nunique"))
	return summary


def load_dataset_metadata(preset: str = "TabArena-v0.1") -> pd.DataFrame:
	"""Load dimensions, class imbalance, and discrete-feature cardinalities."""
	from tabarena.benchmark.task.metadata import TaskMetadataCollection
	from tabarena.benchmark.task.spec import task_spec_from_task_id_str

	metadata = TaskMetadataCollection.from_preset(preset).per_dataset_frame()
	dataset_names = metadata["dataset"].tolist()
	collection = (
		TaskMetadataCollection.from_preset(preset)
		.subset_tasks(dataset_names=dataset_names)
		.materialize()
	)
	imbalance_by_dataset = {}
	discrete_cardinality_by_dataset = {}
	for task_metadata in collection.task_metadata_by_dataset().values():
		task = task_spec_from_task_id_str(task_metadata.task_id_str).with_task_metadata(task_metadata).load()
		X, y = task.get_X_y()
		counts = Counter(y.tolist()).values()
		counts = list(counts)
		imbalance_by_dataset[task_metadata.tabarena_task_name] = (
			max(counts) / min(counts) if len(counts) > 1 else pd.NA
		)
		discrete_columns = X.select_dtypes(
			include=["object", "category", "string", "bool", "boolean"]
		).columns
		cardinalities = X[discrete_columns].nunique(dropna=True)
		discrete_cardinality_by_dataset[task_metadata.tabarena_task_name] = {
			"discrete_max_cardinality": cardinalities.max() if not cardinalities.empty else pd.NA,
			"discrete_average_cardinality": cardinalities.mean() if not cardinalities.empty else pd.NA,
		}
	metadata["imbalance_factor"] = metadata["dataset"].map(imbalance_by_dataset)
	for column in DISCRETE_CARDINALITY_COLUMNS:
		metadata[column] = metadata["dataset"].map(
			{dataset: values[column] for dataset, values in discrete_cardinality_by_dataset.items()}
		)
	return metadata[["dataset", *PROPERTY_COLUMNS]]


def add_dataset_metadata(summary: pd.DataFrame, preset: str = "TabArena-v0.1") -> pd.DataFrame:
	metadata = load_dataset_metadata(preset)
	merged = summary.merge(metadata, on="dataset", how="left", validate="many_to_one")
	missing = merged.loc[merged[list(PROPERTY_COLUMNS)].isna().all(axis=1), "dataset"].unique()
	if len(missing):
		print(f"Warning: metadata not found for {len(missing)} dataset(s): {', '.join(missing)}")
	return merged


def build_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
	"""Build a dataset-level table for comparing the two TabICL models."""
	performance = summary.pivot(index="dataset", columns="method_label", values="roc_auc")
	comparison = performance.reindex(columns=list(METHODS))
	comparison.insert(0, "dataset", comparison.index)
	comparison = comparison.reset_index(drop=True)
	comparison["gat_vs_tabiclv2"] = comparison["TabICL graph-1d"].sub(comparison["TabICLv2"])
	comparison["abs_gat_vs_tabiclv2"] = comparison["gat_vs_tabiclv2"].abs()
	properties = summary[["dataset", *PROPERTY_COLUMNS]].drop_duplicates("dataset")
	return properties.merge(comparison, on="dataset", how="left", validate="one_to_one")


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

	performance_data = plot_data.pivot(
		index=["dataset", property_name], columns="method_label", values="roc_auc"
	).reset_index()
	performance_data["performance_difference"] = performance_data["TabICL graph-1d"].sub(
		performance_data["TabICLv2"]
	)
	performance_data = performance_data.sort_values(property_name)

	fig, (axis, difference_axis) = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
	for label in METHODS:
		method_data = plot_data[plot_data["method_label"].eq(label)]
		if method_data.empty:
			continue
		method_data = method_data.sort_values(property_name)
		alpha = 1.0 if label == "TabICL graph-1d" else 0.3
		axis.plot(
			method_data[property_name],
			method_data["roc_auc"],
			label=label,
			marker="o",
			linewidth=1.2,
			alpha=alpha,
			markersize=4,
		)

	axis.set_xlabel(property_name.replace("_", " ").title())
	axis.set_ylabel("ROC AUC")
	axis.set_ylim(0.0, 1.0)
	if property_name in {
		"num_instances",
		"num_features",
		"imbalance_factor",
		*DISCRETE_CARDINALITY_COLUMNS,
	} and (plot_data[property_name] > 0).all():
		axis.set_xscale("log")
	axis.grid(alpha=0.25)
	axis.legend(loc="best")

	difference_axis.plot(
		performance_data[property_name],
		performance_data["performance_difference"],
		color="tab:purple",
		marker="o",
		linewidth=1.2,
		markersize=4,
	)
	difference_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
	difference_axis.set_xlabel(property_name.replace("_", " ").title())
	difference_axis.set_ylabel("ROC AUC difference (TabICL graph-1d - TabICLv2)")
	difference_axis.set_ylim(-1.0, 1.0)
	difference_axis.grid(alpha=0.25)
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
		"--comparison-output",
		type=Path,
		default=Path("eval/tabicl_graph/roc_auc_model_comparison.csv"),
		help="Output path for the dataset-level model comparison table.",
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
	comparison = build_comparison_table(summary)
	comparison = comparison.sort_values(
		"abs_gat_vs_tabiclv2", ascending=False, na_position="last"
	)
	args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
	comparison.to_csv(args.comparison_output, index=False)
	plot_results(summary, args.output)
	for property_name in PROPERTY_COLUMNS:
		property_output = args.output.with_name(f"{args.output.stem}_by_{property_name}{args.output.suffix}")
		plot_results_by_property(summary, property_name, property_output)

	print(comparison.to_string(index=False))
	print(f"Plot written to {args.output}")
	print(f"Property plots written next to {args.output}")
	print(f"Summary written to {args.summary_output}")
	print(f"Comparison table written to {args.comparison_output}")


if __name__ == "__main__":
	main()
