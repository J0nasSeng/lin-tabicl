from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nested import nested_tensor


MAX_DISCRETE = 100.0


def _scale_discrete(values: Tensor, max_discrete: float) -> Tensor:
    """Map discrete IDs to the bounded interval ``[-2, 2]``."""
    return ((values / max_discrete) * 4.0 - 2.0).clamp(-2.0, 2.0)


def infer_feature_types(X: Tensor, d: Tensor, seq_lens: Optional[Tensor] = None) -> Tensor:
    """Infer a legacy batch's discrete-feature mask before normalization.

    This compatibility helper is used only for prior batches that do not yet
    carry explicit feature metadata. The inference must happen on raw values,
    before standardization changes their cardinality.
    """
    if X.ndim != 3 or d.ndim != 1 or d.shape[0] != X.shape[0]:
        raise ValueError("X must be (batch, sequence, features) and d must be (batch,)")
    if seq_lens is None:
        seq_lens = torch.full_like(d, X.shape[1])
    mask = torch.zeros((X.shape[0], X.shape[2]), dtype=torch.bool, device=X.device)
    for dataset_idx in range(X.shape[0]):
        sequence_length = int(seq_lens[dataset_idx].item())
        feature_count = int(d[dataset_idx].item())
        for feature_idx in range(feature_count):
            values = X[dataset_idx, :sequence_length, feature_idx]
            values = values[torch.isfinite(values)]
            mask[dataset_idx, feature_idx] = torch.unique(values).numel() <= 20
    return mask


def build_normalizer(normalization: str) -> Optional["Normalizer"]:
    """Construct the configured normalizer, or ``None`` for ``none``."""
    if normalization == "none":
        return None
    if normalization == "std":
        return Standardizer()
    if normalization == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown normalization method: {normalization!r}")


def normalize_batch(
    X: Tensor,
    d: Tensor,
    seq_lens: Tensor,
    train_sizes: Tensor,
    normalization: str,
) -> Tensor:
    """Normalize a dense or nested prior batch on the worker side."""
    normalizer = build_normalizer(normalization)
    if normalizer is None:
        return X

    if X.is_nested:
        normalized = []
        for index, dataset in enumerate(X.unbind()):
            dataset_batch = dataset.unsqueeze(0)
            metadata = slice(index, index + 1)
            feature_types = infer_feature_types(dataset_batch, d[metadata], seq_lens[metadata])
            normalized.append(
                normalizer(
                    dataset_batch,
                    feature_types,
                    d[metadata],
                    train_sizes[metadata],
                    seq_lens[metadata],
                )[0]
            )
        return nested_tensor(normalized, device=X.device)

    feature_types = infer_feature_types(X, d, seq_lens)
    return normalizer(X, feature_types, d, train_sizes, seq_lens)


class Normalizer(ABC):
    """Base class for per-dataset feature normalization.

    Normalizers operate on a batch of padded datasets. Statistics are fitted
    independently for each dataset using its training rows only, then the
    resulting transform is applied to all active rows in that dataset.
    """

    @abstractmethod
    def transform(
        self,
        X: Tensor,
        feature_types: Tensor,
        d: Tensor,
        train_sizes: Tensor,
        seq_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Normalize a batch of feature tensors and return normalized ``X``."""

    def __call__(
        self,
        X: Tensor,
        feature_types: Tensor,
        d: Tensor,
        train_sizes: Tensor,
        seq_lens: Optional[Tensor] = None,
    ) -> Tensor:
        return self.transform(X, feature_types, d, train_sizes, seq_lens)


class Standardizer(Normalizer):
    """Standardize continuous features and scale discrete features per dataset.

    Continuous features use ``(x - mean) / std``. Discrete features are mapped
    linearly and clipped to ``[-2, 2]`` using :data:`MAX_DISCRETE`; they are not
    centered or standardized.
    Non-finite training values are ignored when estimating continuous
    statistics. Constant or entirely non-finite columns use a unit scale and a
    zero replacement for non-finite values.

    Parameters
    ----------
    max_discrete : float, default=MAX_DISCRETE
        Maximum discrete ID used when mapping discrete features to ``[-2, 2]``.
    eps : float, default=1e-6
        Minimum standard deviation.
    """

    def __init__(self, max_discrete: float = MAX_DISCRETE, eps: float = 1e-6) -> None:
        if max_discrete <= 0:
            raise ValueError("max_discrete must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.max_discrete = float(max_discrete)
        self.eps = float(eps)

    def transform(
        self,
        X: Tensor,
        feature_types: Optional[Tensor],
        d: Tensor,
        train_sizes: Tensor,
        seq_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a normalized copy of a padded ``(B, T, H)`` tensor.

        ``feature_types`` is a boolean tensor with shape ``(B, H)`` where
        ``True`` marks a discrete feature. If it is ``None``, all active
        features are treated as continuous. Inactive/padded feature positions
        are not modified. ``d``, ``train_sizes``, and ``seq_lens`` are per-
        dataset metadata tensors.
        """
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, sequence, features)")
        batch_size, max_seq_len, max_features = X.shape
        if feature_types is not None and feature_types.shape != (batch_size, max_features):
            raise ValueError("feature_types must have shape (batch, features)")
        if d.shape != (batch_size,) or train_sizes.shape != (batch_size,):
            raise ValueError("d and train_sizes must have shape (batch,)")
        if seq_lens is None:
            seq_lens = torch.full_like(d, max_seq_len)
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape (batch,)")

        result = X.clone()
        infer_feature_types = feature_types is None
        if infer_feature_types:
            feature_types = torch.zeros(
                (batch_size, max_features), dtype=torch.bool, device=X.device
            )
        else:
            feature_types = feature_types.to(device=X.device, dtype=torch.bool)
        for dataset_idx in range(batch_size):
            feature_count = int(d[dataset_idx].item())
            sequence_length = int(seq_lens[dataset_idx].item())
            train_size = int(train_sizes[dataset_idx].item())
            if not 0 <= feature_count <= max_features:
                raise ValueError("d contains a feature count outside X")
            if not 0 <= train_size <= sequence_length <= max_seq_len:
                raise ValueError("invalid train_sizes/seq_lens for X")
            if feature_count == 0 or sequence_length == 0 or train_size == 0:
                continue

            active = result[dataset_idx, :sequence_length, :feature_count]
            train = active[:train_size]
            discrete = feature_types[dataset_idx, :feature_count].bool()

            # Scale discrete values before fitting their statistics. The
            # operation is performed on a cloned view so X remains untouched.
            scaled_active = active.clone()
            scaled_train = train.clone()
            if discrete.any():
                scaled_active[:, discrete] = _scale_discrete(
                    active[:, discrete], self.max_discrete
                )
                scaled_train[:, discrete] = _scale_discrete(
                    train[:, discrete], self.max_discrete
                )

            continuous = ~discrete
            finite = torch.isfinite(scaled_train)
            finite_values = torch.where(finite, scaled_train, torch.zeros_like(scaled_train))
            counts = finite.sum(dim=0).clamp_min(1)
            mean = finite_values.sum(dim=0) / counts
            centered = torch.where(finite, scaled_train - mean, torch.zeros_like(scaled_train))
            std = torch.sqrt(centered.square().sum(dim=0) / counts).clamp_min(self.eps)

            normalized = scaled_active.clone()
            normalized[:, continuous] = (scaled_active[:, continuous] - mean[continuous]) / std[continuous]
            result[dataset_idx, :sequence_length, :feature_count] = torch.where(
                torch.isfinite(normalized), normalized, torch.zeros_like(normalized)
            )

        return result


class RobustScaler(Normalizer):
    """Robustly scale continuous features and scale discrete features per dataset.

    Continuous features use ``(x - median) / IQR``. Discrete features are
    mapped linearly and clipped to ``[-2, 2]`` using ``max_discrete`` and are
    not centered or divided by an additional scale. Medians and quartiles are
    fitted from training rows only and the transform is applied to all active
    rows.
    """

    def __init__(self, max_discrete: float = MAX_DISCRETE, eps: float = 1e-2) -> None:
        if max_discrete <= 0:
            raise ValueError("max_discrete must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.max_discrete = float(max_discrete)
        self.eps = float(eps)

    def transform(
        self,
        X: Tensor,
        feature_types: Optional[Tensor],
        d: Tensor,
        train_sizes: Tensor,
        seq_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """Return a robustly scaled copy of a padded ``(B, T, H)`` tensor."""
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, sequence, features)")
        batch_size, max_seq_len, max_features = X.shape
        if feature_types is not None and feature_types.shape != (batch_size, max_features):
            raise ValueError("feature_types must have shape (batch, features)")
        if d.shape != (batch_size,) or train_sizes.shape != (batch_size,):
            raise ValueError("d and train_sizes must have shape (batch,)")
        if seq_lens is None:
            seq_lens = torch.full_like(d, max_seq_len)
        if seq_lens.shape != (batch_size,):
            raise ValueError("seq_lens must have shape (batch,)")

        result = X.clone()
        if feature_types is None:
            feature_types = torch.zeros(
                (batch_size, max_features), dtype=torch.bool, device=X.device
            )
        else:
            feature_types = feature_types.to(device=X.device, dtype=torch.bool)

        for dataset_idx in range(batch_size):
            feature_count = int(d[dataset_idx].item())
            sequence_length = int(seq_lens[dataset_idx].item())
            train_size = int(train_sizes[dataset_idx].item())
            if not 0 <= feature_count <= max_features:
                raise ValueError("d contains a feature count outside X")
            if not 0 <= train_size <= sequence_length <= max_seq_len:
                raise ValueError("invalid train_sizes/seq_lens for X")
            if feature_count == 0 or sequence_length == 0 or train_size == 0:
                continue

            active = result[dataset_idx, :sequence_length, :feature_count]
            train = active[:train_size]
            discrete = feature_types[dataset_idx, :feature_count]
            scaled_active = active.clone()
            scaled_train = train.clone()
            if discrete.any():
                scaled_active[:, discrete] = _scale_discrete(
                    active[:, discrete], self.max_discrete
                )
                scaled_train[:, discrete] = _scale_discrete(
                    train[:, discrete], self.max_discrete
                )

            continuous = ~discrete
            center = torch.zeros(feature_count, device=X.device, dtype=X.dtype)
            spread = torch.ones(feature_count, device=X.device, dtype=X.dtype)
            for feature_idx in torch.where(continuous)[0].tolist():
                values = scaled_train[:, feature_idx]
                values = values[torch.isfinite(values)]
                if values.numel() == 0:
                    continue
                center[feature_idx] = torch.quantile(values, 0.5)
                q25, q75 = torch.quantile(values, torch.tensor([0.25, 0.75], device=X.device))
                spread[feature_idx] = (q75 - q25).clamp_min(self.eps)

            normalized = scaled_active.clone()
            normalized[:, continuous] = (scaled_active[:, continuous] - center[continuous]) / spread[continuous]
            result[dataset_idx, :sequence_length, :feature_count] = torch.where(
                torch.isfinite(normalized), normalized, torch.zeros_like(normalized)
            )

        return result
