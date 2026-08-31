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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, recall_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from tabicl.eval._run import EvalSample, _build_model_from_checkpoint, _evaluate_one_dataset
from tabicl.prior._dataset import PriorDataset
from tabicl.prior._prior_config import DEFAULT_FIXED_HP
from tabicl import TabICLClassifier


GRAPH_BACKENDS = {
	"graph",
	"graph-pyg",
	"graph-1d",
	"graph-1d-pyg",
	"graph-2d",
	"graph-2d-pyg",
}

GRAPH_EVALUATION_CONFIG = {
	"graph_min_train_neighbors": 4,
	"graph_max_train_neighbors": 4,
	"graph_train_neighbors_per_test": 2,
	"graph_cross_label_fraction": 0.25,
	"graph_v1_prob": 1.0,
	"graph_v2_prob": 0.0,
	"graph_prob": 0.0,
}


# Registries make adding another baseline or score a local change rather than
# requiring changes to the evaluation loop and plotting code.
BASELINE_FACTORIES: dict[str, Callable[[int, int], object]] = {
	"random_forest": lambda seed, n_estimators: RandomForestClassifier(
		n_estimators=n_estimators, random_state=seed, n_jobs=1
	),
	"linear": lambda seed, _n_estimators: make_pipeline(
		StandardScaler(),
		LogisticRegression(max_iter=1000, random_state=seed),
	),
	"extra_trees": lambda seed, n_estimators: ExtraTreesClassifier(
		n_estimators=n_estimators, random_state=seed, n_jobs=1
	),
	"xgboost": lambda seed, n_estimators: XGBClassifier(
		n_estimators=n_estimators,
		max_depth=6,
		learning_rate=0.05,
		subsample=0.8,
		colsample_bytree=0.8,
		eval_metric="logloss",
		n_jobs=1,
		random_state=seed,
		verbosity=0,
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

	The standard metrics use the sklearn estimators, whose class-label mapping is
	restored before scoring. This helper is also used for optional direct-model
	diagnostic predictions and therefore explicitly supplies the true test labels.
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
	parser.add_argument(
		"--encoder-checkpoint",
		type=Path,
		default=None,
		help="Optional custom encoder-backend TabICL checkpoint to evaluate as a baseline.",
	)
	parser.add_argument(
		"--encoder-n-estimators",
		type=int,
		default=1,
		help="Number of ensemble estimators for the custom encoder baseline.",
	)
	parser.add_argument("--num-datasets", type=int, default=1000, help="Datasets per class count")
	parser.add_argument("--batch-size", type=int, default=1, help="Prior datasets generated per batch")
	parser.add_argument(
		"--batch-size-per-gp",
		type=int,
		default=1,
		help="Datasets sharing sampled prior settings; 5 matches Stage 1 training.",
	)
	parser.add_argument("--min-features", type=int, default=2)
	parser.add_argument("--max-features", type=int, default=256)
	parser.add_argument("--max-seq-len", type=int, default=2048)
	parser.add_argument("--min-seq-len", type=int, default=None)
	parser.add_argument("--min-train-size", type=float, default=0.1)
	parser.add_argument("--max-train-size", type=float, default=0.9)
	parser.add_argument("--prior-type", choices=["mlp_scm", "tree_scm", "mix_scm", "graph_scm"], default="graph_scm")
	parser.add_argument("--prior-device", default="cpu")
	parser.add_argument("--device", default=None, help="Inference device (default: cuda when available)")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument(
		"--normalization",
		choices=("none", "std", "robust"),
		default="std",
		help="Optional feature normalization applied before evaluation (default: std, matching Stage 1 training).",
	)
	parser.add_argument(
		"--normalize-features",
		action="store_true",
		help="Compatibility alias for enabling feature normalization.",
	)
	knn_group = parser.add_mutually_exclusive_group()
	knn_group.add_argument(
		"--knn",
		type=int,
		default=None,
		metavar="K",
		help="Use hard K-nearest-neighbor classification on GAT pre-decoder representations.",
	)
	knn_group.add_argument(
		"--soft-knn",
		type=int,
		default=None,
		metavar="K",
		help="Use a soft top-K nearest-neighbor readout on GAT pre-decoder representations.",
	)
	parser.add_argument(
		"--soft-knn-temperature",
		type=float,
		default=1.0,
		help="Temperature for soft top-K similarity weights (default: 1.0).",
	)
	parser.add_argument(
		"--soft-knn-ablation",
		action="store_true",
		help="Run a GAT-only ablation over soft-KNN K=1..20 and temperatures 0.05..0.5.",
	)
	parser.add_argument(
		"--soft-knn-ablation-num-datasets",
		type=int,
		default=100,
		help="Number of datasets for the soft-KNN ablation (default: 100).",
	)
	parser.add_argument(
		"--soft-knn-ablation-num-classes",
		type=int,
		default=10,
		help="Number of classes for the soft-KNN ablation (default: 10).",
	)
	parser.add_argument(
		"--refinement-ablation",
		action="store_true",
		help="Run a native GAT evaluation ablation over refinement iterations 1..10.",
	)
	parser.add_argument(
		"--graph-mixture-ablation",
		action="store_true",
		help="Compare pure v1, pure v2, and pure graph-prior mixtures.",
	)
	parser.add_argument(
		"--cross-label-fraction-ablation",
		action="store_true",
		help="Compare graph cross-label fractions 0, 0.1, and 0.25.",
	)
	parser.add_argument(
		"--n-estimators-ablation",
		action="store_true",
		help="Compare sklearn ensemble sizes 1, 2, 4, and 8.",
	)
	parser.add_argument(
		"--n-estimators-ablation-values",
		type=int,
		nargs="+",
		default=[1, 2, 4, 8],
		help="Estimator counts for --n-estimators-ablation.",
	)
	parser.add_argument(
		"--discrete-features-ablation",
		action="store_true",
		help="Compare graph TabICL performance across discrete-feature fractions.",
	)
	parser.add_argument(
		"--discrete-features-ablation-values",
		type=float,
		nargs="+",
		default=[0.0, 0.25, 0.5, 0.75, 1.0],
		help="Discrete-feature fractions for --discrete-features-ablation.",
	)
	parser.add_argument(
		"--gat-layers-ablation",
		action="store_true",
		help="Compare GAT-input and post-GAT representations with decoder, UMAP, and silhouette metrics.",
	)
	parser.add_argument(
		"--gat-layers-ablation-num-datasets",
		type=int,
		default=100,
		help="Number of datasets for the GAT-layer ablation (default: 100).",
	)
	parser.add_argument(
		"--gat-layers-ablation-num-classes",
		type=int,
		default=2,
		help="Number of classes for the GAT-layer ablation (default: 2).",
	)
	parser.add_argument(
		"--refinement-ablation-num-datasets",
		type=int,
		default=50,
		help="Number of datasets for the refinement ablation (default: 50).",
	)
	parser.add_argument(
		"--refinement-ablation-num-classes",
		type=int,
		default=2,
		help="Number of classes for the refinement ablation (default: 2).",
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
		help="Number of ensemble members for the pretrained TabICL baseline (default: 1)",
	)
	parser.add_argument(
		"--gat-tabicl-n-estimators",
		type=int,
		default=None,
		help="Number of sklearn ensemble members for the supplied GAT checkpoint (default: pretrained setting)",
	)
	parser.add_argument(
		"--num-refinement-iter",
		type=int,
		default=1,
		help="Number of repeated GAT refinement passes for the supplied checkpoint (default: 1)",
	)
	parser.add_argument(
		"--entry-layer",
		default="last",
		help="GAT layer at which refinement starts; use 'last' for only the final layer (default: last)",
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


def _as_regular_tensors(batch: tuple[object, ...]) -> tuple[object, ...]:
	"""Convert variable-length prior tensors while preserving graph metadata."""
	return tuple(
		value.to_padded_tensor(padding=0.0)
		if isinstance(value, torch.Tensor) and value.is_nested
		else value
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


def _knn_predictions(sample, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Classify test representations using training representations and labels."""
	train_size = sample.train_size
	representations = np.nan_to_num(sample.repr_full.numpy(), nan=0.0, posinf=0.0, neginf=0.0)
	labels = sample.y_full.numpy().astype(int)
	classifier = KNeighborsClassifier(
		n_neighbors=min(n_neighbors, train_size), weights="uniform", metric="cosine"
	)
	classifier.fit(representations[:train_size], labels[:train_size])
	probabilities = classifier.predict_proba(representations[train_size:])
	predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
	return predictions, probabilities, classifier.classes_


def _knn_predictions_from_representation(
	representations: np.ndarray,
	y_train: np.ndarray,
	n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Classify rows using representations returned by sklearn TabICL."""
	representations = np.nan_to_num(representations, nan=0.0, posinf=0.0, neginf=0.0)
	classifier = KNeighborsClassifier(
		n_neighbors=min(n_neighbors, y_train.shape[0]),
		weights="uniform",
		metric="cosine",
	)
	classifier.fit(representations[: y_train.shape[0]], y_train)
	probabilities = classifier.predict_proba(representations[y_train.shape[0] :])
	predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
	return predictions, probabilities, classifier.classes_


def _soft_knn_predictions_from_representation(
	representations: np.ndarray,
	y_train: np.ndarray,
	n_neighbors: int,
	temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Read out labels with a temperature-scaled soft top-K neighbor kernel.

	The similarity is the same scaled dot product used by the model's
	``soft_kmeans`` decoder. Unlike that decoder, only the K largest similarities
	for each test row contribute to the class probabilities.
	"""
	if temperature <= 0:
		raise ValueError("soft_knn_temperature must be positive")
	if y_train.size == 0:
		raise ValueError("soft top-K KNN requires at least one training row")

	representations = np.nan_to_num(
		representations, nan=0.0, posinf=0.0, neginf=0.0
	).astype(np.float32, copy=False)
	train_size = y_train.shape[0]
	train_repr = representations[:train_size]
	test_repr = representations[train_size:]
	classes, inverse_labels = np.unique(y_train, return_inverse=True)
	k = min(int(n_neighbors), train_size)

	# Match the full soft-kmeans readout: scaled dot-product similarities,
	# row-wise softmax, then aggregate neighbor mass by class.
	similarity = test_repr @ train_repr.T
	similarity /= np.sqrt(max(train_repr.shape[1], 1)) * temperature
	if k < train_size:
		top_indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
		top_similarity = np.take_along_axis(similarity, top_indices, axis=1)
	else:
		top_indices = np.broadcast_to(np.arange(train_size), similarity.shape)
		top_similarity = similarity

	top_similarity = top_similarity - top_similarity.max(axis=1, keepdims=True)
	weights = np.exp(top_similarity)
	weights /= weights.sum(axis=1, keepdims=True)
	top_labels = inverse_labels[top_indices]
	probabilities = np.zeros((test_repr.shape[0], classes.shape[0]), dtype=np.float64)
	for class_index in range(classes.shape[0]):
		probabilities[:, class_index] = np.sum(
			weights * (top_labels == class_index), axis=1
		)
	predictions = classes[np.argmax(probabilities, axis=1)].astype(int)
	return predictions, probabilities, classes


def _representation_probe_predictions(
	representations: np.ndarray,
	y_train: np.ndarray,
	probe: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Fit a classifier on graph representations and predict test rows."""
	representations = np.nan_to_num(
		representations, nan=0.0, posinf=0.0, neginf=0.0
	)
	train_size = y_train.shape[0]
	probe.fit(representations[:train_size], y_train)
	probabilities = probe.predict_proba(representations[train_size:])
	classes = np.asarray(probe.classes_)
	predictions = classes[np.argmax(probabilities, axis=1)].astype(int)
	return predictions, probabilities, classes


def _baseline_predictions(
	name: str,
	baseline: object,
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Fit a raw-feature baseline and restore its original class labels."""
	x_train = np.nan_to_num(x_train)
	x_test = np.nan_to_num(x_test)
	if name != "xgboost":
		baseline.fit(x_train, y_train)
		probabilities = baseline.predict_proba(x_test)
		classes = np.asarray(baseline.classes_)
	else:
		# XGBoost requires labels encoded as [0, ..., n_classes - 1], while
		# PriorDataset can produce arbitrary/non-contiguous class IDs.
		encoder = LabelEncoder()
		encoded_labels = encoder.fit_transform(y_train)
		n_classes = len(encoder.classes_)
		if n_classes == 2:
			baseline.set_params(objective="binary:logistic")
		else:
			baseline.set_params(objective="multi:softprob", num_class=n_classes)
		baseline.fit(x_train, encoded_labels)
		probabilities = baseline.predict_proba(x_test)
		classes = encoder.classes_
	predictions = classes[np.argmax(probabilities, axis=1)].astype(int)
	return predictions, probabilities, classes


def _soft_knn_ensemble_probabilities(
	gat_tabicl: TabICLClassifier,
	x_test: np.ndarray,
	y_train: np.ndarray,
	neighbor_counts: tuple[int, ...],
	temperatures: tuple[float, ...],
) -> dict[tuple[int, float], tuple[np.ndarray, np.ndarray]]:
	"""Run soft-KNN independently per GAT ensemble view, then average outputs.

	This mirrors the default sklearn evaluation more closely than decoding the
	mean representation: each view gets its own final predictor LayerNorm and
	readout, and only then are probabilities averaged across views.
	"""
	X_encoded = gat_tabicl.X_encoder_.transform(x_test)
	data = gat_tabicl.ensemble_generator_.transform(X_encoded, mode="both")
	classes = np.unique(y_train)
	probability_sums = {
		(n_neighbors, temperature): np.zeros((x_test.shape[0], classes.shape[0]), dtype=np.float64)
		for n_neighbors in neighbor_counts
		for temperature in temperatures
	}
	view_count = 0
	ln = gat_tabicl.model_.model.icl_predictor.ln

	for norm_method, (Xs, ys) in data.items():
		feature_shuffles = gat_tabicl.ensemble_generator_.feature_shuffles_[norm_method]
		view_representations = gat_tabicl._batch_forward_with_repr(Xs, ys, feature_shuffles)
		with torch.no_grad():
			normalized = ln(
				torch.from_numpy(view_representations).to(gat_tabicl.device_)
			).float().cpu().numpy()
		view_count += len(normalized)

		for view_index, representation in enumerate(normalized):
			view_labels = ys[view_index].astype(int)
			train_size = view_labels.shape[0]
			train_repr = representation[:train_size].astype(np.float32, copy=False)
			test_repr = representation[train_size:].astype(np.float32, copy=False)
			similarity = test_repr @ train_repr.T
			similarity /= np.sqrt(max(train_repr.shape[1], 1))
			view_classes, inverse_labels = np.unique(view_labels, return_inverse=True)
			view_to_global = np.asarray([np.searchsorted(classes, label) for label in view_classes])

			for n_neighbors in neighbor_counts:
				k = min(n_neighbors, train_size)
				top_indices = (
					np.argpartition(similarity, -k, axis=1)[:, -k:]
					if k < train_size
					else np.broadcast_to(np.arange(train_size), similarity.shape)
				)
				top_similarity = np.take_along_axis(similarity, top_indices, axis=1)
				top_labels = inverse_labels[top_indices]
				for temperature in temperatures:
					weights = top_similarity / temperature
					weights -= weights.max(axis=1, keepdims=True)
					weights = np.exp(weights)
					weights /= weights.sum(axis=1, keepdims=True)
					view_probabilities = np.zeros(
						(test_repr.shape[0], view_classes.shape[0]), dtype=np.float64
					)
					for class_index in range(view_classes.shape[0]):
						view_probabilities[:, class_index] = np.sum(
							weights * (top_labels == class_index), axis=1
						)
					probabilities = probability_sums[(n_neighbors, temperature)]
					probabilities[:, view_to_global] += view_probabilities
	if view_count == 0:
		raise ValueError("GAT ensemble produced no representations")
	return {
		key: (classes[np.argmax(probabilities / view_count, axis=1)].astype(int), probabilities / view_count)
		for key, probabilities in probability_sums.items()
	}


def run_refinement_ablation(
	model,
	checkpoint: Path,
	args: argparse.Namespace,
	device: str,
) -> list[dict]:
	"""Evaluate native GAT predictions for refinement iterations 1 through 10.

	Unlike the soft-KNN ablation, this uses ``predict_proba`` directly. Thus it
	keeps the default GAT inference path, including the model decoder and the
	classifier's normal ensemble aggregation.
	"""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	iterations = tuple(range(1, 11))
	num_datasets = args.refinement_ablation_num_datasets
	num_classes = args.refinement_ablation_num_classes
	if num_datasets < 1:
		raise ValueError("--refinement-ablation-num-datasets must be positive")
	if num_classes < 2:
		raise ValueError("--refinement-ablation-num-classes must be at least 2")
	max_classes = int(getattr(model, "max_classes", 0))
	if max_classes > 0 and num_classes > max_classes:
		raise ValueError(
			f"This graph checkpoint supports at most {max_classes} classes, but the refinement "
			f"ablation requested {num_classes}. Use --refinement-ablation-num-classes "
			f"{max_classes} or a compatible checkpoint."
		)

	prior = PriorDataset(
		batch_size=args.batch_size,
		batch_size_per_gp=args.batch_size_per_gp,
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
	gat_n_estimators = args.gat_tabicl_n_estimators or args.pretrained_tabicl_n_estimators
	processed = 0
	batch_index = 0
	while processed < num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= num_datasets:
				break
			seq_len = int(batch[3][index].item())
			train_size = int(batch[4][index].item())
			y = batch[1][index, :seq_len].numpy().astype(int)
			y_train, y_test = y[:train_size], y[train_size:]
			if not _has_complete_label_support(y_train, y_test):
				processed += 1
				continue
			x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
			for refinement_iter in iterations:
				gat_tabicl = _build_gat_tabicl(
					checkpoint,
					gat_n_estimators,
					device,
					args.seed + processed,
					refinement_iter,
					args.entry_layer,
				)
				gat_tabicl.fit(x[:train_size], y_train)
				# Native evaluation path: decoder, softmax/logit handling, and
				# ensemble aggregation are all performed by predict_proba().
				probabilities = gat_tabicl.predict_proba(x[train_size:])
				prediction = gat_tabicl.classes_[np.argmax(probabilities, axis=1)].astype(int)
				rows.append({
					"num_classes": num_classes,
					"dataset": processed,
					"batch": current_batch_index,
					"n_features": x.shape[1],
					"model": "gat_tabicl",
					"score": "balanced_accuracy",
					"refinement_iterations": refinement_iter,
					"value": _balanced_accuracy(y_test, prediction),
				})
			processed += 1

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	results_output = args.results_output or output_dir / "refinement_ablation.json"
	results_output.parent.mkdir(parents=True, exist_ok=True)
	results_output.write_text(json.dumps(rows, indent=2))

	values = [
		[row["value"] for row in rows if row["refinement_iterations"] == refinement_iter]
		for refinement_iter in iterations
	]
	fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
	boxplot = ax.boxplot(
		values,
		labels=[str(refinement_iter) for refinement_iter in iterations],
		patch_artist=True,
		showmeans=True,
	)
	for patch in boxplot["boxes"]:
		patch.set_facecolor("#4472C4")
		patch.set_alpha(0.7)
	ax.set(
		title=f"Native GAT refinement ablation (datasets={num_datasets}, classes={num_classes})",
		xlabel="Refinement iterations",
		ylabel="Balanced accuracy",
		ylim=(0.0, 1.0),
	)
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	plot_output = output_dir / "refinement_ablation_balanced_accuracy.png"
	fig.savefig(plot_output)
	plt.close(fig)
	print(f"Saved refinement ablation results to {results_output}")
	print(f"Saved refinement ablation plot to {plot_output}")
	return rows


def run_soft_knn_ablation(
	model,
	checkpoint: Path,
	args: argparse.Namespace,
	device: str,
) -> list[dict]:
	"""Evaluate only GAT TabICL across the soft-KNN hyperparameter grid.

	The GAT model and representation are computed once per dataset. Each
	parameter pair then replaces only the readout, making the comparison use
	the same datasets and representations for every point in the ablation.
	"""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	temperatures = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
	neighbor_counts = tuple(range(1, 21))
	num_datasets = args.soft_knn_ablation_num_datasets
	if num_datasets < 1:
		raise ValueError("--soft-knn-ablation-num-datasets must be positive")
	if args.soft_knn_ablation_num_classes < 2:
		raise ValueError("--soft-knn-ablation-num-classes must be at least 2")
	if any(value < 1 for value in args.n_estimators_ablation_values):
		raise ValueError("--n-estimators-ablation-values must be positive")
	if args.refinement_ablation_num_datasets < 1:
		raise ValueError("--refinement-ablation-num-datasets must be positive")
	if args.refinement_ablation_num_classes < 2:
		raise ValueError("--refinement-ablation-num-classes must be at least 2")
	max_classes = int(getattr(model, "max_classes", 0))
	if max_classes > 0 and args.soft_knn_ablation_num_classes > max_classes:
		raise ValueError(
			f"This graph checkpoint supports at most {max_classes} classes, but the ablation "
			f"requested {args.soft_knn_ablation_num_classes}. Graph inference does not "
			"support the classifier's many-class mixed-radix path; use "
			f"--soft-knn-ablation-num-classes {max_classes} or a checkpoint trained with "
			"a larger max_classes value."
		)

	prior = PriorDataset(
		batch_size=args.batch_size,
		batch_size_per_gp=args.batch_size_per_gp,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=args.soft_knn_ablation_num_classes,
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
	gat_n_estimators = args.gat_tabicl_n_estimators or args.pretrained_tabicl_n_estimators
	processed = 0
	batch_index = 0
	while processed < num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= num_datasets:
				break

			seq_len = int(batch[3][index].item())
			train_size = int(batch[4][index].item())
			y = batch[1][index, :seq_len].numpy().astype(int)
			y_train, y_test = y[:train_size], y[train_size:]
			if not _has_complete_label_support(y_train, y_test):
				processed += 1
				continue

			x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
			gat_tabicl = _build_gat_tabicl(
				checkpoint,
				gat_n_estimators,
				device,
				args.seed + processed,
				args.num_refinement_iter,
				args.entry_layer,
				graph_config=GRAPH_EVALUATION_CONFIG.copy(),
			)
			gat_tabicl.fit(x[:train_size], y_train)
			ablation_outputs = _soft_knn_ensemble_probabilities(
				gat_tabicl, x[train_size:], y_train, neighbor_counts, temperatures
			)
			for n_neighbors in neighbor_counts:
				for temperature in temperatures:
					prediction, _probabilities = ablation_outputs[(n_neighbors, temperature)]
					rows.append({
						"num_classes": args.soft_knn_ablation_num_classes,
						"dataset": processed,
						"batch": current_batch_index,
						"n_features": x.shape[1],
						"model": "gat_tabicl",
						"score": "balanced_accuracy",
						"soft_knn_k": n_neighbors,
						"soft_knn_temperature": temperature,
						"value": _balanced_accuracy(y_test, prediction),
					})
			processed += 1

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	results_output = args.results_output or output_dir / "soft_knn_ablation.json"
	results_output.parent.mkdir(parents=True, exist_ok=True)
	results_output.write_text(json.dumps(rows, indent=2))

	parameter_pairs = [
		(n_neighbors, temperature)
		for n_neighbors in neighbor_counts
		for temperature in temperatures
	]
	values = [
		[
			row["value"]
			for row in rows
			if row["soft_knn_k"] == n_neighbors
			and row["soft_knn_temperature"] == temperature
		]
		for n_neighbors, temperature in parameter_pairs
	]
	labels = [f"k={n_neighbors}, t={temperature:g}" for n_neighbors, temperature in parameter_pairs]
	valid_values = [
		(np.asarray(parameter_values, dtype=float), parameter_pair)
		for parameter_values, parameter_pair in zip(values, parameter_pairs)
		if parameter_values
	]
	if valid_values:
		statistics = {
			"mean": lambda scores: float(np.mean(scores)),
			"median": lambda scores: float(np.median(scores)),
			"min": lambda scores: float(np.min(scores)),
			"max": lambda scores: float(np.max(scores)),
		}
		print("Best soft-KNN configurations by balanced-accuracy statistic:")
		for statistic_name, statistic_fn in statistics.items():
			best_scores, best_pair = max(
				valid_values,
				key=lambda item: statistic_fn(item[0]),
			)
			best_k, best_temperature = best_pair
			print(
				f"  {statistic_name}: k={best_k}, temperature={best_temperature:g}, "
				f"{statistic_name}={statistic_fn(best_scores):.6f}"
			)
	fig_width = max(18.0, len(labels) * 0.18)
	fig, ax = plt.subplots(figsize=(fig_width, 8), dpi=150)
	boxplot = ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
	for patch in boxplot["boxes"]:
		patch.set_facecolor("#4472C4")
		patch.set_alpha(0.7)
	ax.set(
		title=(
			f"GAT TabICL soft top-K KNN ablation "
			f"(datasets={num_datasets}, classes={args.soft_knn_ablation_num_classes})"
		),
		xlabel="Soft-KNN parameter pair",
		ylabel="Balanced accuracy",
		ylim=(0.0, 1.0),
	)
	ax.tick_params(axis="x", labelrotation=90, labelsize=7)
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	plot_output = output_dir / "soft_knn_ablation_balanced_accuracy.png"
	fig.savefig(plot_output)
	plt.close(fig)
	print(f"Saved soft-KNN ablation results to {results_output}")
	print(f"Saved soft-KNN ablation plot to {plot_output}")
	return rows


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
		norm_methods="none"
	)


def _build_encoder_tabicl(
	checkpoint: Path, n_estimators: int, device: str, seed: int
):
	"""Build a custom encoder-backend TabICL baseline."""
	return TabICLClassifier(
		model_path=checkpoint,
		n_estimators=n_estimators,
		device=device,
		random_state=seed,
		n_jobs=1,
		verbose=False,
		norm_methods="none",
	)


def _build_gat_tabicl(
	checkpoint: Path,
	n_estimators: int,
	device: str,
	seed: int,
	num_refinement_iter: int,
	entry_layer: int | str | None,
	graph_config: dict[str, object] | None = None,
):
	"""Build the supplied graph checkpoint through the sklearn interface."""
	return TabICLClassifier(
		model_path=checkpoint,
		n_estimators=n_estimators,
		device=device,
		random_state=seed,
		n_jobs=1,
		verbose=False,
		gat_mode="reasoning" if num_refinement_iter > 1 else "ensemble",
		gat_num_iterations=num_refinement_iter,
		gat_entry_layer=entry_layer,
		norm_methods="none",
		graph_config=graph_config,
	)


def run_graph_ablation(
	checkpoint: Path,
	args: argparse.Namespace,
	device: str,
	name: str,
	variants: list[tuple[str, dict[str, object], int]],
) -> list[dict]:
	"""Evaluate graph TabICL variants on identical prior datasets.

	Each variant is a ``(label, graph_config, n_estimators)`` tuple. No
	baselines are fitted; the resulting plot contains only graph TabICL.
	"""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	if not variants:
		raise ValueError("At least one ablation variant is required")
	prior = PriorDataset(
		batch_size=args.batch_size,
		batch_size_per_gp=args.batch_size_per_gp,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=2,
		min_seq_len=args.min_seq_len,
		max_seq_len=args.max_seq_len,
		min_train_size=args.min_train_size,
		max_train_size=args.max_train_size,
		prior_type=args.prior_type,
		scm_fixed_hp=DEFAULT_FIXED_HP.copy(),
		device=args.prior_device,
		n_jobs=1,
		normalization=("std" if args.normalize_features and args.normalization == "none" else args.normalization),
	)
	rows: list[dict] = []
	processed = 0
	batch_index = 0
	while processed < args.num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= args.num_datasets:
				break
			seq_len = int(batch[3][index].item())
			train_size = int(batch[4][index].item())
			y = batch[1][index, :seq_len].numpy().astype(int)
			y_train, y_test = y[:train_size], y[train_size:]
			if not _has_complete_label_support(y_train, y_test):
				processed += 1
				continue
			x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
			for label, graph_config, n_estimators in variants:
				classifier = _build_gat_tabicl(
					checkpoint,
					n_estimators,
					device,
					args.seed + processed,
					1,
					"last",
					graph_config=graph_config,
				)
				classifier.fit(x[:train_size], y_train)
				probabilities = classifier.predict_proba(x[train_size:])
				predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
				rows.append({
					"ablation": name,
					"variant": label,
					"dataset": processed,
					"batch": current_batch_index,
					"n_features": x.shape[1],
					"score": "balanced_accuracy",
					"value": _balanced_accuracy(y_test, predictions),
				})
			processed += 1

	args.output_dir.mkdir(parents=True, exist_ok=True)
	results_output = args.output_dir / f"{name}_ablation.json"
	results_output.write_text(json.dumps(rows, indent=2))
	labels = [variant[0] for variant in variants]
	values = [[row["value"] for row in rows if row["variant"] == label] for label in labels]
	fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5), dpi=150)
	boxplot = ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
	for patch in boxplot["boxes"]:
		patch.set_facecolor("#4472C4")
		patch.set_alpha(0.75)
	ax.set(
		title=f"GAT TabICL {name} ablation (datasets={args.num_datasets})",
		ylabel="Balanced accuracy",
		ylim=(0.0, 1.0),
	)
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	plot_output = args.output_dir / f"{name}_ablation_balanced_accuracy.png"
	fig.savefig(plot_output)
	plt.close(fig)
	print(f"Saved {name} ablation results to {results_output}")
	print(f"Saved {name} ablation plot to {plot_output}")
	return rows


def run_discrete_features_ablation(
	checkpoint: Path, args: argparse.Namespace, device: str
) -> list[dict]:
	"""Evaluate graph TabICL while varying the prior categorical-feature rate."""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt

	values = tuple(args.discrete_features_ablation_values)
	if not values or any(value < 0.0 or value > 1.0 for value in values):
		raise ValueError("--discrete-features-ablation-values must be in [0, 1]")

	rows: list[dict] = []
	base = GRAPH_EVALUATION_CONFIG.copy()
	for fraction in values:
		fixed_hp = DEFAULT_FIXED_HP.copy()
		fixed_hp["cat_prob"] = fraction
		prior = PriorDataset(
			batch_size=args.batch_size,
			batch_size_per_gp=args.batch_size_per_gp,
			min_features=args.min_features,
			max_features=args.max_features,
			max_classes=2,
			min_seq_len=args.min_seq_len,
			max_seq_len=args.max_seq_len,
			min_train_size=args.min_train_size,
			max_train_size=args.max_train_size,
			prior_type=args.prior_type,
			scm_fixed_hp=fixed_hp,
			device=args.prior_device,
			n_jobs=1,
			normalization=("std" if args.normalize_features and args.normalization == "none" else args.normalization),
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
				seq_len = int(batch[3][index].item())
				train_size = int(batch[4][index].item())
				y = batch[1][index, :seq_len].numpy().astype(int)
				y_train, y_test = y[:train_size], y[train_size:]
				if not _has_complete_label_support(y_train, y_test):
					processed += 1
					continue
				x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
				classifier = _build_gat_tabicl(
					checkpoint, 1, device, args.seed + processed, 1, "last", graph_config=base
				)
				classifier.fit(x[:train_size], y_train)
				probabilities = classifier.predict_proba(x[train_size:])
				predictions = classifier.classes_[np.argmax(probabilities, axis=1)].astype(int)
				rows.append({
					"ablation": "discrete_features",
					"variant": str(fraction),
					"discrete_feature_fraction": fraction,
					"dataset": processed,
					"batch": current_batch_index,
					"n_features": x.shape[1],
					"score": "balanced_accuracy",
					"value": _balanced_accuracy(y_test, predictions),
				})
				processed += 1

	output_dir = args.output_dir
	output_dir.mkdir(parents=True, exist_ok=True)
	results_output = output_dir / "discrete_features_ablation.json"
	results_output.write_text(json.dumps(rows, indent=2))
	labels = [str(value) for value in values]
	box_values = [[row["value"] for row in rows if row["variant"] == label] for label in labels]
	fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5), dpi=150)
	boxplot = ax.boxplot(box_values, labels=labels, patch_artist=True, showmeans=True)
	for patch in boxplot["boxes"]:
		patch.set_facecolor("#4472C4")
		patch.set_alpha(0.75)
	ax.set(
		title=f"GAT TabICL discrete-feature fraction ablation (datasets={args.num_datasets})",
		xlabel="Fraction of discrete features",
		ylabel="Balanced accuracy",
		ylim=(0.0, 1.0),
	)
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	plot_output = output_dir / "discrete_features_ablation_balanced_accuracy.png"
	fig.savefig(plot_output)
	plt.close(fig)
	print(f"Saved discrete-feature ablation results to {results_output}")
	print(f"Saved discrete-feature ablation plot to {plot_output}")
	return rows


def _safe_silhouette(representation: np.ndarray, labels: np.ndarray) -> float:
	"""Return a label silhouette score or NaN when it is undefined."""
	representation = np.nan_to_num(representation, nan=0.0, posinf=0.0, neginf=0.0)
	labels = np.asarray(labels)
	if representation.shape[0] < 3 or np.unique(labels).size < 2:
		return float("nan")
	try:
		return float(silhouette_score(representation, labels, metric="cosine"))
	except ValueError:
		return float("nan")


def _plot_gat_layer_pair(
	pre: np.ndarray,
	post: np.ndarray,
	labels: np.ndarray,
	train_size: int,
	output_path: Path,
	seed: int,
	n_neighbors: int,
	n_epochs: int,
) -> None:
	"""Write a paired UMAP comparing the same dataset before and after GAT."""
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
	from umap import UMAP

	if pre.shape[0] < 3 or post.shape[0] < 3:
		return
	coords = []
	for representation in (pre, post):
		nn = min(n_neighbors, representation.shape[0] - 1)
		coords.append(UMAP(
			n_components=2,
			n_neighbors=nn,
			min_dist=0.1,
			metric="euclidean",
			n_epochs=n_epochs,
			n_jobs=1,
			random_state=seed,
		).fit_transform(np.nan_to_num(representation)))

	fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
	label_min, label_max = labels.min(), labels.max()
	if label_min == label_max:
		label_min -= 0.5
		label_max += 0.5
	for ax, embedding, title in zip(axes, coords, ("GAT input / skip-GAT", "Post-GAT")):
		ax.scatter(
			embedding[:train_size, 0], embedding[:train_size, 1],
			c=labels[:train_size], cmap="tab10", vmin=label_min, vmax=label_max,
			s=16, alpha=0.75, label="train",
		)
		ax.scatter(
			embedding[train_size:, 0], embedding[train_size:, 1],
			c=labels[train_size:], cmap="tab10", vmin=label_min, vmax=label_max,
			s=22, alpha=0.9, marker="x", label="test",
		)
		ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
		ax.grid(alpha=0.2)
		ax.legend(loc="best")
	fig.suptitle("GAT-layer representation comparison")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path)
	plt.close(fig)


def run_gat_layers_ablation(checkpoint: Path, args: argparse.Namespace, device: str) -> list[dict]:
	"""Compare the exact GAT input with the normal post-GAT representation."""
	if args.gat_layers_ablation_num_datasets < 1:
		raise ValueError("--gat-layers-ablation-num-datasets must be positive")
	if args.gat_layers_ablation_num_classes < 2:
		raise ValueError("--gat-layers-ablation-num-classes must be at least 2")

	prior = PriorDataset(
		batch_size=args.batch_size,
		batch_size_per_gp=args.batch_size_per_gp,
		min_features=args.min_features,
		max_features=args.max_features,
		max_classes=args.gat_layers_ablation_num_classes,
		min_seq_len=args.min_seq_len,
		max_seq_len=args.max_seq_len,
		min_train_size=args.min_train_size,
		max_train_size=args.max_train_size,
		prior_type=args.prior_type,
		device=args.prior_device,
		graph_cross_label_fraction=0.0,
		n_jobs=1,
		normalization=("std" if args.normalize_features and args.normalization == "none" else args.normalization),
	)
	rows: list[dict] = []
	processed = 0
	batch_index = 0
	output_dir = args.output_dir / "gat_layers_ablation"
	while processed < args.gat_layers_ablation_num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= args.gat_layers_ablation_num_datasets:
				break
			seq_len = int(batch[3][index].item())
			train_size = int(batch[4][index].item())
			y = batch[1][index, :seq_len].numpy().astype(int)
			y_train, y_test = y[:train_size], y[train_size:]
			if not _has_complete_label_support(y_train, y_test):
				processed += 1
				continue
			x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
			common = GRAPH_EVALUATION_CONFIG.copy()
			variants = {
				"gat": None,
				"skip_gat": True,
			}
			representations: dict[str, np.ndarray] = {}
			for variant, skip in variants.items():
				config = {**common}
				if skip:
					config["skip_gat"] = True
				classifier = _build_gat_tabicl(
					checkpoint,
					args.gat_tabicl_n_estimators or args.pretrained_tabicl_n_estimators,
					device,
					args.seed + processed,
					args.num_refinement_iter,
					args.entry_layer,
					graph_config=config,
				)
				classifier.fit(x[:train_size], y_train)
				proba = classifier.predict_proba(x[train_size:])
				predictions = classifier.classes_[np.argmax(proba, axis=1)].astype(int)
				# predict_representation() returns the fitted training context followed
				# by the rows supplied here; pass only test rows to avoid duplicating
				# the training context.
				full_repr = classifier.predict_representation(x[train_size:])
				representations[variant] = full_repr
				rows.append({
					"ablation": "gat_layers",
					"variant": variant,
					"decoder": "native",
					"dataset": processed,
					"batch": current_batch_index,
					"n_features": x.shape[1],
					"representation_dim": int(full_repr.shape[1]),
					"score": "balanced_accuracy",
					"value": _balanced_accuracy(y_test, predictions),
				})
				for decoder_name, k in (("soft_knn", args.soft_knn or 10),):
					knn_pred, _, _ = _soft_knn_predictions_from_representation(
						full_repr, y_train, k, args.soft_knn_temperature
					)
					rows.append({
						"ablation": "gat_layers",
						"variant": variant,
						"decoder": decoder_name,
						"dataset": processed,
						"batch": current_batch_index,
						"n_features": x.shape[1],
						"representation_dim": int(full_repr.shape[1]),
						"score": "balanced_accuracy",
						"value": _balanced_accuracy(y_test, knn_pred),
					})
			pre_test = representations["skip_gat"][train_size:]
			post_test = representations["gat"][train_size:]
			for variant, representation in (("skip_gat", pre_test), ("gat", post_test)):
				rows.append({
					"ablation": "gat_layers",
					"variant": variant,
					"decoder": "silhouette",
					"dataset": processed,
					"batch": current_batch_index,
					"n_features": x.shape[1],
					"representation_dim": int(representation.shape[1]),
					"score": "silhouette_cosine_test",
					"value": _safe_silhouette(representation, y_test),
				})
			_plot_gat_layer_pair(
				representations["skip_gat"], representations["gat"], y, train_size,
				output_dir / "umap" / f"dataset{processed:05d}.png",
				args.seed + processed, args.umap_n_neighbors, args.umap_n_epochs,
			)
			processed += 1

	output_dir.mkdir(parents=True, exist_ok=True)
	results_output = output_dir / "gat_layers_ablation.json"
	results_output.write_text(json.dumps(rows, indent=2, allow_nan=True))
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
	for score_name, filename, title, ylabel in (
		("balanced_accuracy", "balanced_accuracy.png", "GAT-layer decoder comparison", "Balanced accuracy"),
		("silhouette_cosine_test", "silhouette.png", "GAT-layer representation separation", "Silhouette score"),
	):
		plot_rows = [row for row in rows if row["score"] == score_name]
		if not plot_rows:
			continue
		labels = sorted({
			f'{row["variant"]} ({row["decoder"]})' for row in plot_rows
		})
		values = [[
			row["value"] for row in plot_rows
			if f'{row["variant"]} ({row["decoder"]})' == label and np.isfinite(row["value"])
		] for label in labels]
		if not all(values):
			continue
		fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.8), 5), dpi=150)
		ax.boxplot(values, labels=labels, patch_artist=True, showmeans=True)
		ax.set(title=title, ylabel=ylabel)
		if score_name == "balanced_accuracy":
			ax.set_ylim(0.0, 1.0)
		ax.grid(axis="y", alpha=0.25)
		fig.tight_layout()
		fig.savefig(output_dir / filename)
		plt.close(fig)
	print(f"Saved GAT-layer ablation results to {results_output}")
	return rows


def run_requested_ablations(checkpoint: Path, args: argparse.Namespace, device: str) -> None:
	"""Run each requested graph-only ablation independently."""
	if args.graph_mixture_ablation:
		base = GRAPH_EVALUATION_CONFIG.copy()
		run_graph_ablation(checkpoint, args, device, "graph_mixture", [
			("v1", {**base, "graph_v1_prob": 1.0, "graph_v2_prob": 0.0, "graph_prob": 0.0}, 1),
			("v2", {**base, "graph_v1_prob": 0.0, "graph_v2_prob": 1.0, "graph_prob": 0.0}, 1),
			("graph", {**base, "graph_v1_prob": 0.0, "graph_v2_prob": 0.0, "graph_prob": 1.0}, 1),
		])
	if args.cross_label_fraction_ablation:
		base = GRAPH_EVALUATION_CONFIG.copy()
		run_graph_ablation(checkpoint, args, device, "cross_label_fraction", [
			(str(fraction), {**base, "graph_cross_label_fraction": fraction}, 1)
			for fraction in (0.0, 0.1, 0.25)
		])
	if args.n_estimators_ablation:
		base = GRAPH_EVALUATION_CONFIG.copy()
		for value in args.n_estimators_ablation_values:
			if value < 1:
				raise ValueError("--n-estimators-ablation-values must be positive")
		run_graph_ablation(checkpoint, args, device, "n_estimators", [
			(str(value), base, value) for value in args.n_estimators_ablation_values
		])
	if args.discrete_features_ablation:
		run_discrete_features_ablation(checkpoint, args, device)
	if args.gat_layers_ablation:
		run_gat_layers_ablation(checkpoint, args, device)


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
	tabicl_prediction: np.ndarray | None = None,
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
	if tabicl_prediction is None:
		tabicl_prediction = sample.y_pred_test.numpy().astype(int)
	tabicl_accuracy = _balanced_accuracy(sample.y_true_test.numpy().astype(int), tabicl_prediction)
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


def _evaluate_class_count(
	model,
	checkpoint: Path,
	args: argparse.Namespace,
	num_classes: int,
	device: str,
) -> list[dict]:
	prior = PriorDataset(
		batch_size=args.batch_size,
		batch_size_per_gp=args.batch_size_per_gp,
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
	gat_n_estimators = args.gat_tabicl_n_estimators or args.pretrained_tabicl_n_estimators
	processed = 0
	batch_index = 0
	while processed < args.num_datasets:
		batch = _as_regular_tensors(prior.get_batch())
		current_batch_index = batch_index
		batch_index += 1
		for index in range(int(batch[0].shape[0])):
			if processed >= args.num_datasets:
				break

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

			x = batch[0][index, :seq_len, : int(batch[2][index].item())].numpy()
			train_size = int(batch[4][index].item())
			x_train, x_test = x[:train_size], x[train_size:]

			print(f"Evaluating K={num_classes}, Shape={x.shape}, dataset={processed}...")

			gat_tabicl = _build_gat_tabicl(
				checkpoint,
				gat_n_estimators,
				device,
				args.seed + processed,
				args.num_refinement_iter,
				args.entry_layer,
				graph_config={
					"graph_v1_prob": 1.0,
					"graph_v2_prob": 0.0,
					"graph_prob": 0.0,
				},
			)
			gat_tabicl.fit(x_train, y_train)
			gat_proba = gat_tabicl.predict_proba(x_test)
			gat_prediction = gat_tabicl.classes_[np.argmax(gat_proba, axis=1)].astype(int)
			gat_representation = gat_tabicl.predict_representation(x_test)

			if args.skip_baselines:
				# Use the fitted GAT estimator's representation for the UMAP plot.
				# Re-evaluating the raw checkpoint here passes the batched prior graph
				# metadata to a shorter, single-dataset input, which can have a
				# different ``num_nodes`` and triggers a graph-dimension mismatch.
				sample = EvalSample(
					x=torch.from_numpy(x),
					y_full=torch.from_numpy(y),
					y_pred_test=torch.from_numpy(gat_prediction),
					y_logits_test=torch.from_numpy(gat_proba),
					y_true_test=torch.from_numpy(y_test),
					repr_full=torch.from_numpy(gat_representation),
					train_size=train_size,
				)
				random_forest = BASELINE_FACTORIES["random_forest"](
					args.seed + processed, args.rf_n_estimators
				)
				random_forest.fit(
					np.nan_to_num(x[:train_size]), y[:train_size]
				)
				rf_prediction = random_forest.predict(np.nan_to_num(x[train_size:])).astype(int)
				rf_accuracy = _balanced_accuracy(y[train_size:], rf_prediction)
				knn_prediction = None
				if args.knn is not None:
					knn_prediction, tabicl_scores, _ = _knn_predictions_from_representation(
						gat_representation, y_train, args.knn
					)
				elif args.soft_knn is not None:
					knn_prediction, tabicl_scores, _ = _soft_knn_predictions_from_representation(
						gat_representation, y_train, args.soft_knn, args.soft_knn_temperature
					)
				else:
					tabicl_scores = gat_proba
				tabicl_entropy = _mean_entropy(tabicl_scores)
				print("Plot UMAP...")
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
						knn_prediction if knn_prediction is not None else gat_prediction,
					)
				except Exception as exc:
					print(f"Warning: UMAP plot failed for K={num_classes}, dataset={processed}: {exc}")
				processed += 1
				continue

			n_features = x.shape[1]
			if args.knn is not None:
				knn_prediction, knn_proba, knn_labels = _knn_predictions_from_representation(
					gat_representation, y_train, args.knn
				)
			elif args.soft_knn is not None:
				knn_prediction, knn_proba, knn_labels = _soft_knn_predictions_from_representation(
					gat_representation, y_train, args.soft_knn, args.soft_knn_temperature
				)
			else:
				knn_prediction = knn_proba = knn_labels = None
			predictions = {
				"gat_tabicl": (gat_prediction, gat_proba, gat_tabicl.classes_),
			}
			# Always include a representation kNN probe. The optional --knn
			# argument remains available for requesting a different k explicitly.
			probe_knn_prediction, probe_knn_proba, probe_knn_labels = (
				_knn_predictions_from_representation(gat_representation, y_train, args.knn or 2)
			)
			predictions["knn_probe"] = (
				probe_knn_prediction,
				probe_knn_proba,
				probe_knn_labels,
			)
			if knn_prediction is not None:
				predictions["knn"] = (knn_prediction, knn_proba, knn_labels)
			linear_probe = make_pipeline(
				StandardScaler(),
				LogisticRegression(max_iter=1000, random_state=args.seed + processed),
			)
			probe_prediction, probe_proba, probe_labels = _representation_probe_predictions(
				gat_representation,
				y_train,
				linear_probe,
			)
			predictions["linear_probe"] = (probe_prediction, probe_proba, probe_labels)
			for name, factory in BASELINE_FACTORIES.items():
				baseline = factory(args.seed + processed, args.rf_n_estimators)
				# RF versions differ in NaN support; replacing non-finite
				# values keeps this baseline portable across sklearn versions.
				predictions[name] = _baseline_predictions(
					name, baseline, x_train, y_train, x_test
				)

			# Use a fresh estimator and the public sklearn-compatible API for the
			# pretrained encoder baseline too.
			pretrained_tabicl = _build_pretrained_tabicl(
				args.pretrained_tabicl_n_estimators,
				device,
				args.seed + processed,
			)
			try:
				pretrained_tabicl.fit(x_train, y_train)
				pretrained_proba = pretrained_tabicl.predict_proba(x_test)
				predictions["pretrained_tabicl"] = (
					pretrained_tabicl.classes_[np.argmax(pretrained_proba, axis=1)].astype(int),
					pretrained_proba,
					pretrained_tabicl.classes_,
				)
			except Exception as exc:
				print(f"Warning: pretrained TabICL failed for K={num_classes}, dataset={processed}: {exc}")
				continue

			if args.encoder_checkpoint is not None:
				encoder_tabicl = _build_encoder_tabicl(
					args.encoder_checkpoint,
					args.encoder_n_estimators,
					device,
					args.seed + processed,
				)
				try:
					encoder_tabicl.fit(x_train, y_train)
					encoder_proba = encoder_tabicl.predict_proba(x_test)
					predictions["encoder_tabicl"] = (
						encoder_tabicl.classes_[np.argmax(encoder_proba, axis=1)].astype(int),
						encoder_proba,
						encoder_tabicl.classes_,
					)
				except Exception as exc:
					print(
						f"Warning: encoder TabICL failed for K={num_classes}, "
						f"dataset={processed}: {exc}"
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

			primary_model = "gat_tabicl"
			primary_values = values[primary_model]
			fig, ax = plt.subplots(figsize=(6, 6))
			print(f"Scatter Plots")
			for baseline in (model for model in models if model != primary_model):
				baseline_values = values[baseline]
				ax.scatter(primary_values, baseline_values, s=18, alpha=0.45, label=baseline)
			ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="equal performance")
			ax.set(
				title=f"K={num_classes}: GAT TabICL vs baselines",
				xlabel="GAT TabICL",
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
				if key[0] == "gat_tabicl" and key in accuracy
			]
			if tabicl_pairs:
				entropy_values, accuracy_values = map(np.asarray, zip(*tabicl_pairs))
				fig, ax = plt.subplots(figsize=(6, 5))
				ax.scatter(entropy_values, accuracy_values, s=18, alpha=0.45, label="gat_tabicl")
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
				if key[0] == "gat_tabicl" and key in accuracy
			]
			if tabicl_pairs:
				cross_entropy_values, accuracy_values = map(np.asarray, zip(*tabicl_pairs))
				fig, ax = plt.subplots(figsize=(6, 5))
				ax.scatter(cross_entropy_values, accuracy_values, s=18, alpha=0.45, label="gat_tabicl")
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
	if args.batch_size < 1 or args.batch_size_per_gp < 1:
		raise ValueError("--batch-size and --batch-size-per-gp must be positive")
	if args.batch_size % args.batch_size_per_gp != 0:
		raise ValueError("--batch-size must be a multiple of --batch-size-per-gp")
	if args.rf_n_estimators < 1:
		raise ValueError("--rf-n-estimators must be positive")
	if args.pretrained_tabicl_n_estimators < 1:
		raise ValueError("--pretrained-tabicl-n-estimators must be positive")
	if args.encoder_n_estimators < 1:
		raise ValueError("--encoder-n-estimators must be positive")
	if args.gat_tabicl_n_estimators is not None and args.gat_tabicl_n_estimators < 1:
		raise ValueError("--gat-tabicl-n-estimators must be positive")
	if args.num_refinement_iter < 1:
		raise ValueError("--num-refinement-iter must be positive")
	if args.entry_layer != "last":
		try:
			if int(args.entry_layer) < 1:
				raise ValueError
			args.entry_layer = int(args.entry_layer)
		except ValueError as exc:
			raise ValueError("--entry-layer must be 'last' or a positive integer") from exc
	if args.umap_n_neighbors < 2:
		raise ValueError("--umap-n-neighbors must be at least 2")
	if args.umap_n_epochs < 1:
		raise ValueError("--umap-n-epochs must be positive")
	if args.knn is not None and args.knn < 1:
		raise ValueError("--knn must be positive")
	if args.soft_knn is not None and args.soft_knn < 1:
		raise ValueError("--soft-knn must be positive")
	if args.soft_knn_temperature <= 0:
		raise ValueError("--soft-knn-temperature must be positive")
	if args.soft_knn_ablation_num_datasets < 1:
		raise ValueError("--soft-knn-ablation-num-datasets must be positive")
	if args.soft_knn_ablation_num_classes < 2:
		raise ValueError("--soft-knn-ablation-num-classes must be at least 2")
	if args.gat_layers_ablation_num_datasets < 1:
		raise ValueError("--gat-layers-ablation-num-datasets must be positive")
	if args.gat_layers_ablation_num_classes < 2:
		raise ValueError("--gat-layers-ablation-num-classes must be at least 2")
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
	if model_type != "tabicl" or getattr(model, "icl_backend", None) not in GRAPH_BACKENDS:
		raise ValueError(
			"The checkpoint must contain a TabICL model with a graph backend "
			"(graph, graph-1d, or graph-2d)."
		)

	checkpoint_config = torch.load(args.checkpoint, map_location="cpu", weights_only=True).get("config", {})
	if str(checkpoint_config.get("model_type", "tabicl")).lower() != "tabicl":
		raise ValueError("The supplied checkpoint must contain a TabICL model.")
	if checkpoint_config.get("icl_backend") not in GRAPH_BACKENDS:
		raise ValueError(
			"The supplied checkpoint must contain a TabICL model with a graph backend "
			"(graph, graph-1d, or graph-2d)."
		)
	if (
		args.graph_mixture_ablation
		or args.cross_label_fraction_ablation
		or args.n_estimators_ablation
		or args.discrete_features_ablation
		or args.gat_layers_ablation
	):
		run_requested_ablations(args.checkpoint, args, device)
		return
	if args.refinement_ablation:
		run_refinement_ablation(model, args.checkpoint, args, device)
		return
	if args.soft_knn_ablation:
		run_soft_knn_ablation(model, args.checkpoint, args, device)
		return

	rows = []
	for num_classes in [2, 5, 10]:
		print(f"Evaluating {args.num_datasets} datasets for K={num_classes}...")
		rows.extend(_evaluate_class_count(model, args.checkpoint, args, num_classes, device))
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
