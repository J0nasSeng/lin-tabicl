"""TabArena model and leaderboard comparison entry point for graph TabICL.

This module follows TabArena's AutoGluon ``AbstractModel`` contract.  The
model-specific implementation remains the sklearn-compatible
``TabICLClassifier`` so that TabICL's preprocessing and graph-backend routing
are used unchanged.
"""

from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

try:
	from autogluon.core.models import AbstractModel  # pyright: ignore[reportMissingImports]
except ImportError as error:  # Keep importing this file possible without TabArena extras.
	AbstractModel = None
	_AUTOGLUON_IMPORT_ERROR = error
else:
	_AUTOGLUON_IMPORT_ERROR = None

from tabicl import TabICLClassifier


GRAPH_BACKENDS = {
	"graph",
	"graph-pyg",
	"graph-1d",
	"graph-1d-pyg",
	"graph-2d",
	"graph-2d-pyg",
}

# These values must match the graph prior used to fine-tune the checkpoint.
# Keep them explicit here so a TabArena run cannot silently evaluate with a
# different graph sampler because of missing or stale checkpoint metadata.
EVALUATION_GRAPH_CONFIG = {
	"graph_min_train_neighbors": 4,
	"graph_max_train_neighbors": 4,
	"graph_train_neighbors_per_test": 2,
	"graph_cross_label_fraction": 0.25,
	"graph_num_graphs": 6,
	"graph_v1_prob": 1.0,
	"graph_v2_prob": 0.0,
	"graph_prob": 0.0,
}


if TYPE_CHECKING:
	import pandas as pd
	from tabarena.utils.config_utils import ConfigGenerator  # pyright: ignore[reportMissingImports]


if AbstractModel is None:
	class _AbstractModelUnavailable:
		"""Placeholder that provides an actionable error outside a TabArena env."""

		def __init__(self, *args: Any, **kwargs: Any) -> None:
			raise ImportError(
				"TabArenaModel requires AutoGluon. Install the TabArena dependencies before running TabArena."
			) from _AUTOGLUON_IMPORT_ERROR

	AbstractModelBase = _AbstractModelUnavailable
else:
	AbstractModelBase = AbstractModel


class TabICLGraphModel(AbstractModelBase):
	"""AutoGluon model wrapper for a trained graph-backend TabICL checkpoint."""

	ag_key = "TabICLGraph"
	ag_name = "TabICLGraph"
	# Tell AutoGluon that this model can use a CUDA device. The actual device
	# ordinal is still selected through the TabICL ``device`` parameter.
	default_num_gpus = 1

	def __init__(self, model_path: str | Path | None = None, **kwargs: Any) -> None:
		# Keep constructor-only settings in AbstractModel.params. AutoGluon
		# reconstructs cloned/bagged fold models from those parameters.
		model_path = model_path or kwargs.get("model_path")
		if model_path is not None:
			model_path = str(model_path)
			kwargs["model_path"] = model_path
		self.model_path = model_path
		self._tabicl: TabICLClassifier | None = None
		self._umap_plotted = False
		self._train_X: Any = None
		self._train_y: Any = None
		super().__init__(**kwargs)

	def _set_default_params(self) -> None:
		"""Set defaults consumed by the TabICL sklearn estimator."""
		for parameter, value in {
			"n_estimators": 8,
			"batch_size": 1,
			"kv_cache": False,
			"use_amp": True,
			"offload_mode": "cpu",
			"gat_mode": "ensemble",
			"gat_num_iterations": 1,
			"gat_entry_layer": None,
			"max_chunk_size": None,
			"decoder_chunk_size": 5000,
			"device": None,
		}.items():
			self._set_default_param_value(parameter, value)

	def _get_default_auxiliary_params(self) -> dict[str, Any]:
		params = super()._get_default_auxiliary_params()
		params.update({"valid_raw_types": ["int", "float", "category", "object"]})
		return params

	@classmethod
	def supported_problem_types(cls) -> list[str]:
		return ["binary", "multiclass"]

	def _validate_checkpoint(self) -> None:
		model_path = self.model_path or self.params.get("model_path")
		if model_path is None:
			raise ValueError("TabICLGraphModel requires model_path in its configuration")
		self.model_path = str(model_path)
		checkpoint_path = Path(self.model_path).expanduser()
		if not checkpoint_path.is_file():
			raise FileNotFoundError(f"TabICL checkpoint not found: {checkpoint_path}")
		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
		if not isinstance(checkpoint, dict) or "config" not in checkpoint or "state_dict" not in checkpoint:
			raise ValueError("TabICL checkpoint must contain 'config' and 'state_dict'")
		config = checkpoint["config"]
		if str(config.get("model_type", "tabicl")).lower() != "tabicl":
			raise ValueError("TabICLGraphModel requires a TabICL checkpoint")
		if config.get("icl_backend", "encoder") not in GRAPH_BACKENDS:
			raise ValueError(
				"TabICLGraphModel requires a checkpoint with a graph backend "
				"(graph, graph-1d, or graph-2d)"
			)
		mismatches = {
			parameter: (config.get(parameter), expected)
			for parameter, expected in EVALUATION_GRAPH_CONFIG.items()
			if parameter in config and config[parameter] != expected
		}
		if mismatches:
			details = ", ".join(
				f"{parameter}={actual!r} (expected {expected!r})"
				for parameter, (actual, expected) in mismatches.items()
			)
			raise ValueError(
				"Checkpoint graph configuration does not match the TabArena evaluation "
				f"configuration: {details}"
			)

	def _tabicl_kwargs(self) -> dict[str, Any]:
		params = self.params.copy()
		params.pop("model_path", None)
		params.pop("problem_type", None)
		params.pop("path", None)
		params.pop("name", None)
		params.pop("ag_name", None)
		params.pop("ag_key", None)
		params.pop("ag_args_fit", None)
		# These are TabArena wrapper settings, not TabICLClassifier arguments.
		params.pop("plot_umap", None)
		params.pop("umap_output_dir", None)
		params["model_path"] = self.model_path or self.params.get("model_path")
		params["allow_auto_download"] = False
		params["kv_cache"] = False
		params["use_amp"] = True
		params["offload_mode"] = "cpu"
		params["batch_size"] = 1
		params["max_chunk_size"] = self.params.get("max_chunk_size")
		params["decoder_chunk_size"] = self.params.get("decoder_chunk_size", 5000)
		params["n_estimators"] = 1
		params["norm_methods"] = "none"
		params["graph_config"] = EVALUATION_GRAPH_CONFIG.copy()
		params["softmax_temperature"] = 0.2
		return params

	def _fit(self, X: "pd.DataFrame", y: "pd.Series", **kwargs: Any) -> None:
		self._validate_checkpoint()
		self._tabicl = TabICLClassifier(**self._tabicl_kwargs())
		self._tabicl.fit(X, y)
		self._train_X = X
		self._train_y = np.asarray(y)
		self.model = self._tabicl

	def _plot_umap(
		self,
		X_test: "pd.DataFrame",
		predicted_labels: np.ndarray | None = None,
	) -> None:
		if not self.params.get("plot_umap", False) or self._umap_plotted:
			return
		if self._tabicl is None or self._train_X is None or self._train_y is None:
			return
		try:
			import matplotlib
			matplotlib.use("Agg", force=True)
			import matplotlib.pyplot as plt
			from umap import UMAP
		except ImportError as error:
			raise ModuleNotFoundError(
				"--plot-umap requires matplotlib and umap-learn"
			) from error

		# Each ensemble view contains the fitted training rows followed by the
		# requested test rows. Averaging these complete views preserves the same
		# representation semantics as TabICLClassifier.predict_representation().
		X_test_array = np.asarray(X_test)
		X_encoded = self._tabicl.X_encoder_.transform(X_test_array)
		if self._tabicl.feature_reducer_ is not None:
			X_encoded = self._tabicl.feature_reducer_.transform(X_encoded)
		outputs = []
		for subset, generator in zip(
			self._tabicl.feature_subsets_, self._tabicl.ensemble_generators_
		):
			data = generator.transform(X_encoded[:, subset], mode="both")
			for norm_method, (Xs, ys) in data.items():
				feature_shuffles = generator.feature_shuffles_[norm_method]
				outputs.append(
					self._tabicl._batch_forward_with_repr(Xs, ys, feature_shuffles)
				)
		if not outputs:
			raise RuntimeError("TabICL produced no representations for the UMAP plot")
		representations = np.mean(np.concatenate(outputs, axis=0), axis=0)
		train_size = len(self._train_y)
		if representations.shape[0] <= train_size:
			raise RuntimeError("TabICL UMAP representation does not contain test rows")

		output_dir = Path(self.params.get("umap_output_dir", "eval"))
		output_dir.mkdir(parents=True, exist_ok=True)
		identity = str(getattr(self, "path", None) or getattr(self, "name", "model"))
		identity = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("_")[-160:]
		if not identity:
			identity = "model"
		classes = np.asarray(self._tabicl.classes_)
		class_indices = {label: index for index, label in enumerate(classes.tolist())}
		try:
			train_labels = np.asarray([class_indices[label] for label in self._train_y])
			if predicted_labels is None:
				predicted_labels = classes[np.argmax(self._tabicl.predict_proba(X_test_array), axis=1)]
			test_labels = np.asarray([class_indices[label] for label in predicted_labels])
		except KeyError as error:
			raise RuntimeError("UMAP labels contain a class unknown to TabICL") from error
		labels = np.concatenate((train_labels, test_labels))
		coordinates = UMAP(
			n_components=2,
			n_neighbors=min(15, representations.shape[0] - 1),
			min_dist=0.1,
			random_state=0,
		).fit_transform(representations)
		fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
		for axis, name, sl in (
			(axes[0], "Train", slice(0, train_size)),
			(axes[1], "Test", slice(train_size, None)),
		):
			axis.scatter(
				coordinates[sl, 0], coordinates[sl, 1], c=labels[sl], cmap="tab10", s=16, alpha=0.8
			)
			axis.set(title=name, xlabel="UMAP-1", ylabel="UMAP-2")
			axis.grid(alpha=0.2)
		fig.suptitle(f"Graph TabICL representations: {identity}")
		# AutoGluon can reuse the same model name/path for separate benchmark
		# jobs, so include a UUID rather than relying on the model identity alone.
		plot_path = output_dir / f"{identity}_umap_{uuid.uuid4().hex[:12]}.png"
		fig.savefig(plot_path)
		plt.close(fig)
		self._umap_plotted = True

	def _predict_proba(self, X: "pd.DataFrame", **kwargs: Any) -> np.ndarray:
		if self._tabicl is None:
			raise RuntimeError("TabICLGraphModel has not been fitted")
		probabilities = np.asarray(self._tabicl.predict_proba(X))
		predicted_labels = self._tabicl.classes_[np.argmax(probabilities, axis=1)]
		self._plot_umap(X, predicted_labels)
		# AutoGluon uses a unified representation internally: binary models must
		# return only the positive-class probability. Passing the full (N, 2)
		# matrix makes bagged OOF accumulation broadcast against its (N,) buffer.
		if self.problem_type == "binary":
			if probabilities.ndim != 2 or probabilities.shape[1] < 2:
				raise ValueError(
					f"TabICL binary predict_proba returned shape {probabilities.shape}; expected (n_samples, 2)."
				)
			return probabilities[:, 1]
		return probabilities

	@classmethod
	def config_generator(
		cls,
		model_path: str | Path | None = None,
		device: str | torch.device | None = None,
		num_cpus: int = 64,
		num_gpus: int = 1,
		max_chunk_size: int | None = None,
		decoder_chunk_size: int = 5000,
		plot_umap: bool = False,
		umap_output_dir: str | Path | None = None,
	) -> "ConfigGenerator":
		"""Return a TabArena config generator for the supplied checkpoint."""
		from tabarena.utils.config_utils import ConfigGenerator  # pyright: ignore[reportMissingImports]

		config: dict[str, Any] = {}
		if model_path is not None:
			config["model_path"] = str(model_path)
		if device is not None:
			config["device"] = str(device)
		if max_chunk_size is not None:
			config["max_chunk_size"] = max_chunk_size
		if decoder_chunk_size <= 0:
			raise ValueError("decoder_chunk_size must be positive")
		config["decoder_chunk_size"] = decoder_chunk_size
		config["ag_args_fit"] = {
			"num_cpus": num_cpus,
			"num_gpus": num_gpus,
		}
		config["plot_umap"] = plot_umap
		if umap_output_dir is not None:
			config["umap_output_dir"] = str(umap_output_dir)
		return ConfigGenerator(
			model_cls=cls,
			manual_configs=[config],
			search_space={},
		)


# Backwards-compatible names for lightweight registry/discovery code.
TabArenaModel = TabICLGraphModel
Model = TabICLGraphModel


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--model-path", required=True, type=Path, help="Trained graph-backend TabICL checkpoint")
	parser.add_argument("--results-dir", type=Path, default=Path("experiments/tabicl_graph"))
	parser.add_argument("--output-dir", type=Path, default=Path("eval/tabicl_graph"))
	parser.add_argument("--subset", default="lite", choices=['all', 'binary', 'classification', 'high_cats', 'high_features', 'lite', 'low_cats', 
														  'low_features', 'medium', 'multiclass', 'numerical', 'regression', 'small', 'tabicl', 
														  'tabpfn', 'tiny'])
	parser.add_argument("--datasets", nargs="+", default=None)
	parser.add_argument("--n-configs", type=int, default=0)
	parser.add_argument(
		"--mode",
		"--benchmark-mode",
		dest="benchmark_mode",
		choices=("default", "tuned", "ensembled", "tuned+ensembled"),
		default="default",
		help=(
			"TabArena evaluation protocol: default/tuned use a single full-data fit, "
			"while ensembled/tuned+ensembled use bagged ensemble folds. "
			"The tuned modes require --n-configs > 0."
		),
	)
	parser.add_argument(
		"--num-cpus",
		type=int,
		default=8,
		help="Maximum CPU cores allocated to each TabICL model fit",
	)
	parser.add_argument(
		"--num-gpus",
		type=float,
		default=1,
		help="GPUs allocated to each TabICL model fit; use 0 for CPU-only execution",
	)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Torch device for TabICL inference, for example cpu, cuda, cuda:0, or mps",
	)
	parser.add_argument(
		"--max-chunk-size",
		type=int,
		default=None,
		help="Maximum number of destination-sorted graph edges processed per attention chunk.",
	)
	parser.add_argument(
		"--decoder-chunk-size",
		type=int,
		default=5000,
		help="Number of query rows processed at once by kernel decoders (default: 5000).",
	)
	parser.add_argument(
		"--debug-mode",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Run jobs in-process; use --no-debug-mode to enable the Ray-backed runner",
	)
	parser.add_argument(
		"--plot-umap",
		action="store_true",
		help="Plot train/test graph TabICL representations as UMAP figures in --output-dir.",
	)
	return parser


def compare_against_leaderboard(args: argparse.Namespace) -> Any:
	"""Run TabICL through TabArena and return its website-format leaderboard."""
	from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle  # pyright: ignore[reportMissingImports]
	from tabarena.contexts import TabArenaContext  # pyright: ignore[reportMissingImports]

	model_path = args.model_path.expanduser().resolve()
	if args.decoder_chunk_size <= 0:
		raise ValueError("--decoder-chunk-size must be positive")
	if not model_path.is_file():
		raise FileNotFoundError(f"TabICL checkpoint not found: {model_path}")
	if args.n_configs < 0:
		raise ValueError("--n-configs must be non-negative")
	tuned = args.benchmark_mode in {"tuned", "tuned+ensembled"}
	if tuned and args.n_configs <= 0:
		raise ValueError(
			f"--benchmark-mode={args.benchmark_mode} requires --n-configs > 0"
		)
	use_outer_experiments = args.benchmark_mode in {"default", "tuned"}
	n_configs = args.n_configs if tuned else 0

	experiments = TabArenaV0pt1ExperimentBundle(
		# Reduce bagging folds on small or imbalanced classification datasets so
		# every validation fold contains every class.
		adapt_num_folds_to_n_classes=True,
		outer_experiments=use_outer_experiments,
		models=[(
			TabICLGraphModel.config_generator(
				model_path,
				device=args.device,
				max_chunk_size=args.max_chunk_size,
				decoder_chunk_size=args.decoder_chunk_size,
				plot_umap=args.plot_umap,
				umap_output_dir=args.output_dir.expanduser().resolve(),
				num_cpus=args.num_cpus,
				num_gpus=args.num_gpus,
			),
			n_configs,
		)]
	).build_experiments(time_limit=1*60*60)

	context = TabArenaContext()
	context.build_and_run_jobs(
		experiments,
		expname=str(args.results_dir.expanduser().resolve()),
		subset=args.subset,
		build_kwargs={"dataset_names": args.datasets} if args.datasets else {},
		new_result_prefix="[New] ",
		debug_mode=args.debug_mode,
	)
	leaderboard = context.compare(output_dir=args.output_dir.expanduser().resolve())
	website_leaderboard = context.leaderboard_to_website_format(leaderboard=leaderboard)
	print(website_leaderboard.to_markdown(index=False))
	return website_leaderboard


if __name__ == "__main__":
	compare_against_leaderboard(build_parser().parse_args())


__all__ = ["Model", "TabArenaModel", "TabICLGraphModel", "build_parser", "compare_against_leaderboard"]