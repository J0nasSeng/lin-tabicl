from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class SparseGraphBatch:
    """Container for a batch of sparse directed graphs.

    Attributes
    ----------
    edge_index : list[Tensor]
        One edge index tensor per batch element with shape (2, num_edges), where
        ``edge_index[0]`` are source nodes and ``edge_index[1]`` are destination nodes.

    num_nodes : int
        Number of nodes in each graph.
    """

    edge_index: list[Tensor]
    num_nodes: int


def _sample_indices(
    pool: Tensor,
    k: int,
    generator: Optional[torch.Generator],
    allow_replacement: bool = False,
) -> Tensor:
    """Sample up to ``k`` indices from a 1D candidate pool."""

    if k <= 0 or pool.numel() == 0:
        return pool.new_empty((0,), dtype=torch.long)

    if allow_replacement and pool.numel() < k:
        sampled_pos = torch.randint(0, pool.numel(), (k,), generator=generator, device=pool.device)
        return pool[sampled_pos]

    perm = torch.randperm(pool.numel(), generator=generator, device=pool.device)
    return pool[perm[: min(k, pool.numel())]]


def build_class_conditioned_graph(
    y_train: Tensor,
    total_nodes: int,
    min_train_neighbors: int = 8,
    max_train_neighbors: int = 15,
    same_label_ratio: float = 0.9,
    cross_label_ratio: float = 0.1,
    test_k_per_class: int = 3,
    seed: Optional[int] = None,
) -> SparseGraphBatch:
    """Build sparse directed graphs for graph-based in-context learning.

    Parameters
    ----------
    y_train : Tensor
        Training labels of shape (B, train_size).

    total_nodes : int
        Total number of nodes in the graph (train + test).

    min_train_neighbors : int, default=8
        Minimum number of train-to-train incoming neighbors per training node.

    max_train_neighbors : int, default=15
        Maximum number of train-to-train incoming neighbors per training node.

    same_label_ratio : float, default=0.9
        Target ratio of same-label train neighbors for each training node.

    cross_label_ratio : float, default=0.1
        Target ratio of different-label train neighbors for each training node.

    test_k_per_class : int, default=3
        Minimum number of training neighbors per class for each test node (when available).

    seed : Optional[int], default=None
        Optional random seed for deterministic graph sampling.

    Returns
    -------
    SparseGraphBatch
        Batch of sparse directed graphs represented by edge index tensors.
    """

    if y_train.ndim != 2:
        raise ValueError("y_train must be a 2D tensor of shape (B, train_size)")
    if min_train_neighbors <= 0 or max_train_neighbors <= 0:
        raise ValueError("min_train_neighbors and max_train_neighbors must be positive")
    if min_train_neighbors > max_train_neighbors:
        raise ValueError("min_train_neighbors must be <= max_train_neighbors")
    if test_k_per_class <= 0:
        raise ValueError("test_k_per_class must be positive")
    if same_label_ratio < 0.0 or cross_label_ratio < 0.0:
        raise ValueError("same_label_ratio and cross_label_ratio must be non-negative")

    ratio_sum = same_label_ratio + cross_label_ratio
    if ratio_sum == 0.0:
        raise ValueError("same_label_ratio + cross_label_ratio must be > 0")

    same_ratio = same_label_ratio / ratio_sum
    batch_size, train_size = y_train.shape
    if total_nodes < train_size:
        raise ValueError("total_nodes must be >= train_size")

    gen = torch.Generator(device=y_train.device)
    if seed is not None:
        gen.manual_seed(seed)

    edge_index_batch: list[Tensor] = []

    for b in range(batch_size):
        labels = y_train[b].long()
        train_indices = torch.arange(train_size, device=labels.device, dtype=torch.long)
        classes = torch.unique(labels)

        class_to_indices = {int(c.item()): train_indices[labels == c] for c in classes}

        src_edges: list[Tensor] = []
        dst_edges: list[Tensor] = []

        # Train-to-train edges: each train node aggregates messages from sparse train neighbors.
        for dst in range(train_size):
            degree = int(
                torch.randint(
                    min_train_neighbors,
                    max_train_neighbors + 1,
                    (1,),
                    generator=gen,
                    device=labels.device,
                ).item()
            )
            dst_tensor = torch.tensor(dst, device=labels.device, dtype=torch.long)
            label = int(labels[dst].item())

            same_pool = class_to_indices[label]
            same_pool = same_pool[same_pool != dst_tensor]
            cross_pool = train_indices[labels != label]

            target_same = int(round(degree * same_ratio))
            target_cross = degree - target_same

            sampled_same = _sample_indices(same_pool, target_same, generator=gen, allow_replacement=False)
            sampled_cross = _sample_indices(cross_pool, target_cross, generator=gen, allow_replacement=False)

            selected = torch.cat([sampled_same, sampled_cross], dim=0)
            if selected.numel() > 0:
                selected = torch.unique(selected)

            if selected.numel() < degree:
                remaining = degree - selected.numel()
                all_candidates = train_indices[train_indices != dst_tensor]
                if selected.numel() > 0:
                    mask = torch.ones_like(all_candidates, dtype=torch.bool)
                    for s in selected:
                        mask &= all_candidates != s
                    all_candidates = all_candidates[mask]

                refill = _sample_indices(all_candidates, remaining, generator=gen, allow_replacement=True)
                selected = torch.cat([selected, refill], dim=0)

            if selected.numel() == 0:
                selected = dst_tensor.view(1)

            src_edges.append(selected)
            dst_edges.append(torch.full_like(selected, dst))

        # Test-to-train edges: each test node receives messages from at least k train nodes per class.
        for dst in range(train_size, total_nodes):
            selected_per_test = []
            for c in classes:
                class_pool = class_to_indices[int(c.item())]
                k = min(test_k_per_class, class_pool.numel())
                selected_per_test.append(_sample_indices(class_pool, int(k), generator=gen, allow_replacement=False))

            selected = torch.cat(selected_per_test, dim=0)
            if selected.numel() == 0:
                selected = torch.tensor([0], dtype=torch.long, device=labels.device)

            src_edges.append(selected)
            dst_edges.append(torch.full_like(selected, dst))

        # Self loops for all nodes keep isolated-path behavior stable.
        all_nodes = torch.arange(total_nodes, device=labels.device, dtype=torch.long)
        src_edges.append(all_nodes)
        dst_edges.append(all_nodes)

        edge_src = torch.cat(src_edges, dim=0)
        edge_dst = torch.cat(dst_edges, dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        edge_index = torch.unique(edge_index, dim=1)

        edge_index_batch.append(edge_index)

    return SparseGraphBatch(edge_index=edge_index_batch, num_nodes=total_nodes)
