"""TabArena adapter for a trained graph-backend TabICL classifier.

The benchmark only needs the usual ``fit``/``predict``/``predict_proba``
interface. The actual preprocessing and inference implementation lives in
``TabICLClassifier``; keeping this module as a thin adapter avoids having a
second implementation of graph construction and checkpoint loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabicl import TabICLClassifier


class TabArenaModel:
	"""TabArena-compatible wrapper around a graph-backend TabICL model."""

	def __init__(
		self,
		model_path: str | Path | None = None,
		*,
		checkpoint_path: str | Path | None = None,
		device: str | torch.device | None = None,
		**kwargs: Any,
	) -> None:
		if model_path is not None and checkpoint_path is not None:
			if Path(model_path).expanduser() != Path(checkpoint_path).expanduser():
				raise ValueError("model_path and checkpoint_path must refer to the same checkpoint")

		self.model_path = model_path if model_path is not None else checkpoint_path
		self.checkpoint_path = self.model_path
		self.device = device
		self.kwargs = dict(kwargs)
		self.estimator_: TabICLClassifier | None = None

	def _validate_checkpoint(self) -> None:
		"""Validate the supplied checkpoint before constructing the estimator."""
		if self.model_path is None:
			raise ValueError("A trained graph-backend checkpoint is required; pass model_path or checkpoint_path")

		checkpoint_path = Path(self.model_path).expanduser()
		if not checkpoint_path.is_file():
			raise FileNotFoundError(f"TabICL checkpoint not found: {checkpoint_path}")

		checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
		if not isinstance(checkpoint, dict):
			raise ValueError("The TabICL checkpoint must contain a mapping")
		if "config" not in checkpoint or "state_dict" not in checkpoint:
			raise ValueError("The TabICL checkpoint must contain 'config' and 'state_dict'")

		config = checkpoint["config"]
		if not isinstance(config, dict):
			raise ValueError("The TabICL checkpoint configuration must be a mapping")
		if str(config.get("model_type", "tabicl")).lower() != "tabicl":
			raise ValueError("TabArenaModel requires a TabICL classification checkpoint")
		if config.get("icl_backend", "encoder") != "graph":
			raise ValueError("TabArenaModel requires a checkpoint with icl_backend='graph'")

	def fit(self, X: Any, y: Any) -> "TabArenaModel":
		"""Prepare the trained TabICL model for in-context predictions."""
		self._validate_checkpoint()

		estimator_kwargs = dict(self.kwargs)
		estimator_kwargs.update(model_path=self.model_path, device=self.device)
		# An explicit checkpoint path makes downloading another model
		# undesirable for a reproducible benchmark run.
		estimator_kwargs.setdefault("allow_auto_download", False)
		self.estimator_ = TabICLClassifier(**estimator_kwargs)
		self.estimator_.fit(X, y)

		for name in ("classes_", "n_classes_", "n_features_in_", "n_samples_in_"):
			if hasattr(self.estimator_, name):
				setattr(self, name, getattr(self.estimator_, name))
		return self

	def _require_fitted(self) -> TabICLClassifier:
		if self.estimator_ is None:
			raise RuntimeError("Call fit(X, y) before making predictions")
		return self.estimator_

	def predict_proba(self, X: Any) -> np.ndarray:
		"""Return class probabilities for benchmark test rows."""
		probabilities = np.asarray(self._require_fitted().predict_proba(X))
		if probabilities.ndim != 2:
			raise ValueError(f"TabICL returned probabilities with shape {probabilities.shape}, expected 2D")
		if not np.isfinite(probabilities).all():
			raise ValueError("TabICL returned non-finite class probabilities")
		return probabilities

	def predict(self, X: Any) -> np.ndarray:
		"""Return predictions using the original target-label representation."""
		return np.asarray(self._require_fitted().predict(X))


# Common names used by lightweight benchmark launchers. Keeping aliases here
# makes discovery possible without changing package exports.
TabICLTabArenaModel = TabArenaModel
Model = TabArenaModel


__all__ = ["Model", "TabArenaModel", "TabICLTabArenaModel"]
