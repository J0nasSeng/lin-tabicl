"""Run TabICL graph-backend inference on GraphLand node-classification data.

The :class:`~tabicl.eval.benchmarks.graph_land.dataset.Dataset` class owns the
dataset-specific loading and feature preprocessing. This module only adapts
its masked node data to the sklearn-compatible TabICL API and aggregates the
requested classification metrics.
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
	balanced_accuracy_score,
	f1_score,
	precision_score,
	recall_score,
)

from tabicl import TabICLClassifier
from tabicl.eval.benchmarks.graph_land.dataset import Dataset

MAX_FEATURES = 100


def classification_tasks() -> list[tuple[str, str]]:
	"""Return all unique classification datasets and their data sources."""
	tasks: list[tuple[str, str]] = []
	for source, names in (
		("GraphLand", Dataset.graphland_datasets_names),
		("PyG", Dataset.pyg_datasets_names),
		("OGB", Dataset.ogb_datasets_names),
	):
		for name in names:
			if name in Dataset.multiclass_classification_datasets_names or name in Dataset.binary_classification_datasets_names:
				if (name, source) not in tasks:
					tasks.append((name, source))
	return tasks


def _load_dataset(name: str, split: str, data_dir: Path, device: str) -> Dataset:
	"""Construct a Dataset using the supplied directory as its data root."""
	return Dataset(name=name, split=split, device=device, use_pyg=False, data_dir=data_dir, load_graph=True)


def _masked_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Extract finite train/test features and labels from a transductive dataset."""
	if not dataset.transductive:
		raise ValueError("Inductive splits are not supported; use a transductive split such as RL")

	features = dataset.features.detach().cpu().numpy()
	targets = dataset.targets.detach().cpu().numpy()
	train_mask = dataset.train_mask.detach().cpu().numpy().astype(bool)
	test_mask = dataset.test_mask.detach().cpu().numpy().astype(bool)

	train_mask &= np.isfinite(targets)
	test_mask &= np.isfinite(targets)
	if not train_mask.any() or not test_mask.any():
		raise ValueError("Dataset has an empty labelled train or test split")

	X_train = np.asarray(features[train_mask], dtype=np.float32)
	X_test = np.asarray(features[test_mask], dtype=np.float32)
	y_train = np.asarray(targets[train_mask])
	y_test = np.asarray(targets[test_mask])
	if X_train.ndim != 2 or X_train.shape[1] == 0:
		raise ValueError(f"Dataset has invalid feature shape {X_train.shape}")
	if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
		raise ValueError("Dataset contains non-finite feature values")
	return X_train, y_train, X_test, y_test


def _print_dataset_size(name: str, dataset: Dataset) -> None:
	"""Print graph and feature shapes for a loaded dataset."""
	graphs = [
		getattr(dataset, attribute, None)
		for attribute in ("graph", "train_graph", "val_graph", "test_graph")
	]
	graph_shapes = [tuple(graph.shape) for graph in graphs if graph is not None]
	feature_shape = tuple(dataset.features.shape)
	print(
		f"{name}: graph shape(s)={graph_shapes or ['unavailable']}, "
		f"feature shape={feature_shape}"
	)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, binary: bool) -> dict[str, float]:
	"""Calculate the requested classification metrics."""
	result = {
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)) if binary else float("nan"),
	}
	return result


def evaluate_dataset(
	name: str,
	source: str,
	model_path: Path,
	data_dir: Path,
	*,
	split: str = "RL",
	device: str = "cpu",
	n_estimators: int = 8,
	batch_size: int | None = 8,
	gat_mode: str = "ensemble",
	gat_num_iterations: int = 1,
	gat_entry_layer: int | str | None = None,
) -> dict[str, object] | None:
	"""Load, run, and score one node-classification dataset."""
	started = time.perf_counter()
	dataset = _load_dataset(name, split, data_dir, device="cpu")
	_print_dataset_size(name, dataset)
	if dataset.features.shape[1] > MAX_FEATURES:
		print(
			f"skipping {name}: dataset has {dataset.features.shape[1]} features "
			f"(maximum supported by this benchmark is {MAX_FEATURES})"
		)
		return None
	X_train, y_train, X_test, y_test = _masked_arrays(dataset)

	classifier = TabICLClassifier(
		model_path=model_path,
		allow_auto_download=False,
		device=device,
		n_estimators=n_estimators,
		batch_size=batch_size,
		kv_cache=False,
		gat_mode=gat_mode,
		gat_num_iterations=gat_num_iterations,
		gat_entry_layer=gat_entry_layer,
	)
	classifier.fit(X_train, y_train)
	# The sklearn adapter already loads frozen weights in eval mode; make the
	# benchmark invariant explicit for both the adapter and graph engine.
	classifier.model_.eval()
	device_type = torch.device(device).type
	amp_context = (
		torch.autocast(device_type=device_type, dtype=torch.bfloat16)
		if device_type in {"cpu", "cuda"}
		else nullcontext()
	)
	with amp_context:
		y_pred = np.asarray(classifier.predict(X_test))
		probabilities = np.asarray(classifier.predict_proba(X_test))
	if probabilities.shape[0] != y_test.shape[0]:
		raise ValueError("TabICL returned a probability row count different from the test set")

	return {
		"dataset": name,
		"source": source,
		"task": dataset.task,
		"n_nodes": int(dataset.features.shape[0]),
		"n_features": int(dataset.features.shape[1]),
		"n_classes": int(len(np.unique(y_train))),
		"n_train": int(X_train.shape[0]),
		"n_val": int(dataset.val_mask.sum().item()),
		"n_test": int(X_test.shape[0]),
		**_metrics(y_test, y_pred, dataset.task == "binary_classification"),
		"seconds": float(time.perf_counter() - started),
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", required=True, type=Path, help="Trained TabICL graph-backend checkpoint")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("data"),
		help="Dataset data root containing one subdirectory per GraphLand dataset (default: ./data)",
	)
	parser.add_argument("--output", type=Path, default=Path("graph_land_results.csv"), help="Output CSV path")
	parser.add_argument("--split", default="RL", help="Dataset split (default: RL)")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--n-estimators", type=int, default=8)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--gat-mode", choices=("ensemble", "reasoning"), default="ensemble")
	parser.add_argument("--gat-num-iterations", type=int, default=1)
	parser.add_argument("--gat-entry-layer", default=None)
	parser.add_argument("--datasets", nargs="+", help="Optional dataset names; default is all classification tasks")
	parser.add_argument("--continue-on-error", action="store_true")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	model_path = args.model_path.expanduser().resolve()
	data_dir = args.data_dir.expanduser().resolve()
	if not model_path.is_file():
		raise FileNotFoundError(f"Checkpoint not found: {model_path}")
	if not data_dir.is_dir():
		raise FileNotFoundError(f"Data directory not found: {data_dir}")

	tasks = classification_tasks()
	if args.datasets:
		requested = set(args.datasets)
		unknown = requested - {name for name, _ in tasks}
		if unknown:
			raise ValueError(f"Unknown classification dataset(s): {', '.join(sorted(unknown))}")
		tasks = [task for task in tasks if task[0] in requested]

	rows: list[dict[str, object]] = []
	for name, source in tasks:
		try:
			entry_layer: int | str | None = args.gat_entry_layer
			if entry_layer is not None and entry_layer != "last":
				entry_layer = int(entry_layer)
			result = evaluate_dataset(
				name, source, model_path, data_dir, split=args.split, device=args.device,
				n_estimators=args.n_estimators, batch_size=args.batch_size,
				gat_mode=args.gat_mode, gat_num_iterations=args.gat_num_iterations,
				gat_entry_layer=entry_layer,
			)
			if result is not None:
				rows.append(result)
				print(f"completed {name}")
		except Exception as error:
			print(f"failed {name}: {error}")
			if not args.continue_on_error:
				raise

	results = pd.DataFrame(rows)
	output = args.output.expanduser().resolve()
	output.parent.mkdir(parents=True, exist_ok=True)
	results.to_csv(output, index=False)
	print(results.to_string(index=False))
	print(f"Results written to {output}")


if __name__ == "__main__":
	main()
