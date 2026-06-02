from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

_UMAP_IMPORT_WARNED = False


def _import_umap_and_matplotlib() -> tuple[Any, Any] | tuple[None, None]:
    global _UMAP_IMPORT_WARNED

    try:
        import umap.umap_ as umap_module
        import matplotlib.pyplot as plt
    except ImportError as exc:
        if not _UMAP_IMPORT_WARNED:
            print(
                "Skipping UMAP logging: missing dependency. "
                "Install with `uv pip install umap-learn matplotlib`. "
                f"({exc})"
            )
            _UMAP_IMPORT_WARNED = True
        return None, None

    return umap_module, plt


def _umap_project_to_2d(representations: np.ndarray, seed: int) -> np.ndarray:
    num_points, feature_dim = representations.shape

    if num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_points == 1:
        return np.zeros((1, 2), dtype=np.float32)
    if num_points == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    umap_module, _ = _import_umap_and_matplotlib()
    if umap_module is None:
        # Fallback to first two representation channels when UMAP is unavailable.
        if feature_dim >= 2:
            return representations[:, :2].astype(np.float32)
        one_dim = representations[:, :1].astype(np.float32)
        return np.concatenate([one_dim, np.zeros_like(one_dim)], axis=1)

    n_neighbors = min(15, num_points - 1)
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(representations).astype(np.float32)


def build_test_umap_wandb_images(
    repr_test: Tensor,
    y_test: Tensor,
    wandb_module: Any,
    max_datasets: int = 4,
    start_index: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """Build W&B images for test-only UMAP projections colored by labels.

    Parameters
    ----------
    repr_test : Tensor
        Test representations of shape (B, test_size, D) right before decoder.

    y_test : Tensor
        Test labels of shape (B, test_size).

    wandb_module : Any
        Imported wandb module used to create image payloads.

    max_datasets : int, default=4
        Maximum number of datasets to render from current micro-batch.

    start_index : int, default=0
        Dataset index offset used for deterministic key naming.

    seed : int, default=42
        Random seed used for deterministic UMAP projections.
    """

    if wandb_module is None or max_datasets <= 0:
        return {}

    _, plt = _import_umap_and_matplotlib()
    if plt is None:
        return {}

    if repr_test.ndim != 3 or y_test.ndim != 2:
        raise ValueError("Expected repr_test with shape (B, test_size, D) and y_test with shape (B, test_size)")

    num_datasets = min(max_datasets, repr_test.shape[0], y_test.shape[0])
    payload: dict[str, Any] = {}

    for local_idx in range(num_datasets):
        repr_np = repr_test[local_idx].detach().float().cpu().numpy()
        y_np = y_test[local_idx].detach().long().cpu().numpy()

        if repr_np.shape[0] == 0:
            continue

        embedding = _umap_project_to_2d(repr_np, seed=seed + start_index + local_idx)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        cmap = plt.get_cmap("tab20")
        unique_labels = np.unique(y_np)

        for label in unique_labels:
            mask = y_np == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=18,
                alpha=0.9,
                color=cmap((int(label) % 20) / 20.0),
                label=str(int(label)),
            )

        if len(unique_labels) <= 12:
            ax.legend(title="Label", fontsize=7, title_fontsize=8, loc="best")

        ax.set_title(f"Test UMAP (dataset {start_index + local_idx})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2)

        key = f"umap_test_ds_{start_index + local_idx}"
        payload[key] = wandb_module.Image(fig)
        plt.close(fig)

    return payload
