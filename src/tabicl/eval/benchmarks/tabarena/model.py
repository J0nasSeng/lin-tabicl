"""TabArena model and leaderboard comparison entry point for graph TabICL.

This module follows TabArena's AutoGluon ``AbstractModel`` contract.  The
model-specific implementation remains the sklearn-compatible
``TabICLClassifier`` so that TabICL's preprocessing and graph-backend routing
are used unchanged.
"""

from __future__ import annotations

import argparse
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

	def _tabicl_kwargs(self) -> dict[str, Any]:
		params = self.params.copy()
		params.pop("model_path", None)
		params.pop("problem_type", None)
		params.pop("path", None)
		params.pop("name", None)
		params.pop("ag_name", None)
		params.pop("ag_key", None)
		params.pop("ag_args_fit", None)
		params["model_path"] = self.model_path or self.params.get("model_path")
		params["allow_auto_download"] = False
		params["kv_cache"] = False
		params["use_amp"] = True
		params["offload_mode"] = "cpu"
		params["batch_size"] = 1
		params["max_chunk_size"] = self.params.get("max_chunk_size")
		params["decoder_chunk_size"] = self.params.get("decoder_chunk_size", 5000)
		params["n_estimators"] = 1
		params["feature_reduction"] = "ensemble"
		params["n_components"] = 100
		return params

	def _fit(self, X: "pd.DataFrame", y: "pd.Series", **kwargs: Any) -> None:
		self._validate_checkpoint()
		self._tabicl = TabICLClassifier(**self._tabicl_kwargs())
		self._tabicl.fit(X, y)
		self.model = self._tabicl

	def _predict_proba(self, X: "pd.DataFrame", **kwargs: Any) -> np.ndarray:
		if self._tabicl is None:
			raise RuntimeError("TabICLGraphModel has not been fitted")
		probabilities = np.asarray(self._tabicl.predict_proba(X))
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

	experiments = TabArenaV0pt1ExperimentBundle(
		# Reduce bagging folds on small or imbalanced classification datasets so
		# every validation fold contains every class.
		adapt_num_folds_to_n_classes=True,
		models=[(
			TabICLGraphModel.config_generator(
				model_path,
				device=args.device,
				max_chunk_size=args.max_chunk_size,
				decoder_chunk_size=args.decoder_chunk_size,
				num_cpus=args.num_cpus,
				num_gpus=args.num_gpus,
			),
			args.n_configs,
		)]
	).build_experiments(time_limit=1*60*60)

	print(len(experiments), "experiments built; running...")
	print(experiments)
	print("===========================================")

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