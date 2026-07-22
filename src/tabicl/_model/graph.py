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


@dataclass
class SparseGraphSet:
    """A set of independently sampled sparse graph batches."""

    graphs: list[SparseGraphBatch]

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    @property
    def num_nodes(self) -> int:
        if not self.graphs:
            raise ValueError("A graph set must contain at least one graph")
        return self.graphs[0].num_nodes


def stack_graph_sets(graph_sets: list[SparseGraphSet]) -> SparseGraphSet:
    """Combine per-dataset graph sets into graph batches for model input."""

    if not graph_sets:
        raise ValueError("graph_sets must not be empty")
    num_graphs = graph_sets[0].num_graphs
    if any(graph_set.num_graphs != num_graphs for graph_set in graph_sets):
        raise ValueError("All graph sets must contain the same number of graphs")

    graphs = []
    for graph_idx in range(num_graphs):
        per_dataset = [graph_set.graphs[graph_idx] for graph_set in graph_sets]
        num_nodes = per_dataset[0].num_nodes
        if any(graph.num_nodes != num_nodes for graph in per_dataset):
            raise ValueError("All graph batches must have the same number of nodes")
        graphs.append(SparseGraphBatch(edge_index=[graph.edge_index[0] for graph in per_dataset], num_nodes=num_nodes))
    return SparseGraphSet(graphs=graphs)


def slice_graph_sets(graph_sets: list[SparseGraphSet], indices: slice) -> list[SparseGraphSet]:
    """Slice per-dataset graph sets along the batch dimension."""

    return graph_sets[indices]


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
    share_graph_across_batch: bool = False,
    share_graph_require_identical_labels: bool = True,
) -> SparseGraphBatch:
    """Build sparse directed graphs for graph-based in-context learning.

    Parameters
    ----------
    y_train : Tensor
        Training labels of shape (B, train_size).

    total_nodes : int
        Total number of nodes in the graph (train + test).

    min_train_neighbors : int, default=8
        Lower bound multiplier for same-label train-train sampling per class.
        For class size ``n_c``, the number of sampled same-label pairs is in
        ``[n_c * min_train_neighbors, n_c * max_train_neighbors]`` before adding
        mirrored pairs.

    max_train_neighbors : int, default=15
        Upper bound multiplier for same-label train-train sampling per class.

    same_label_ratio : float, default=0.9
        Target same-label component used to determine cross-label edge budget.

    cross_label_ratio : float, default=0.1
        Target cross-label component used to determine cross-label edge budget.

    test_k_per_class : int, default=3
        Number of sampled train connections per class for each test node before
        adding mirrored pairs.

    seed : Optional[int], default=None
        Optional random seed for deterministic graph sampling.

    share_graph_across_batch : bool, default=False
        If True, build a single graph and reuse it for all batch elements.

    share_graph_require_identical_labels : bool, default=True
        Only used when ``share_graph_across_batch=True``. If True, shared-graph
        mode is activated only when all rows in ``y_train`` are identical.

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

    batch_size, train_size = y_train.shape
    if total_nodes < train_size:
        raise ValueError("total_nodes must be >= train_size")

    gen = torch.Generator(device=y_train.device)
    if seed is not None:
        gen.manual_seed(seed)

    def _sample_pairs_within_pool(pool: Tensor, num_pairs: int) -> tuple[Tensor, Tensor]:
        if num_pairs <= 0 or pool.numel() == 0:
            empty = pool.new_empty((0,), dtype=torch.long)
            return empty, empty

        # Prevent self-connections in random graph generation.
        if pool.numel() == 1:
            empty = pool.new_empty((0,), dtype=torch.long)
            return empty, empty

        src_pos = torch.randint(0, pool.numel(), (num_pairs,), generator=gen, device=pool.device)
        if pool.numel() == 1:
            dst_pos = src_pos.clone()
        else:
            dst_pos = torch.randint(0, pool.numel(), (num_pairs,), generator=gen, device=pool.device)
            # Avoid self-pairs when possible.
            mask_same = src_pos == dst_pos
            while mask_same.any():
                dst_pos[mask_same] = torch.randint(
                    0,
                    pool.numel(),
                    (int(mask_same.sum().item()),),
                    generator=gen,
                    device=pool.device,
                )
                mask_same = src_pos == dst_pos

        return pool[src_pos], pool[dst_pos]

    def _build_single_graph(labels: Tensor) -> Tensor:
        train_indices = torch.arange(train_size, device=labels.device, dtype=torch.long)
        classes = torch.unique(labels)
        class_to_indices = {int(c.item()): train_indices[labels == c] for c in classes}
        class_pools = [class_to_indices[int(c.item())] for c in classes]
        num_classes = len(class_pools)
        num_test = max(0, total_nodes - train_size)

        src_edges: list[Tensor] = []
        dst_edges: list[Tensor] = []
        same_pairs_total = 0

        # 1) Same-label train/train edges: class-level budgets, then mirror.
        for class_pool in class_pools:
            n_c = int(class_pool.numel())
            if n_c == 0:
                continue

            min_pairs = n_c * min_train_neighbors
            max_pairs = n_c * max_train_neighbors
            num_pairs = int(
                torch.randint(min_pairs, max_pairs + 1, (1,), generator=gen, device=labels.device).item()
            )
            same_pairs_total += num_pairs

            src_same, dst_same = _sample_pairs_within_pool(class_pool, num_pairs)
            if src_same.numel() == 0:
                continue

            src_edges.extend([src_same, dst_same])
            dst_edges.extend([dst_same, src_same])

        # 2) Cross-label train/train edges: infer budget from same/cross ratio, then mirror.
        if cross_label_ratio > 0:
            if same_label_ratio > 0:
                cross_pairs_total = int(round(same_pairs_total * (cross_label_ratio / same_label_ratio)))
            else:
                cross_pairs_total = same_pairs_total

            if cross_pairs_total > 0 and num_classes >= 2:
                pair_i: list[int] = []
                pair_j: list[int] = []
                pair_w: list[float] = []
                for i in range(num_classes):
                    for j in range(i + 1, num_classes):
                        w = float(class_pools[i].numel() * class_pools[j].numel())
                        if w > 0:
                            pair_i.append(i)
                            pair_j.append(j)
                            pair_w.append(w)

                if pair_w:
                    weights = torch.tensor(pair_w, dtype=torch.float32, device=labels.device)
                    sampled_pair_ids = torch.multinomial(weights, cross_pairs_total, replacement=True, generator=gen)

                    sampled_src_parts: list[Tensor] = []
                    sampled_dst_parts: list[Tensor] = []
                    unique_pair_ids = torch.unique(sampled_pair_ids)
                    for pid in unique_pair_ids.tolist():
                        count = int((sampled_pair_ids == pid).sum().item())
                        if count == 0:
                            continue
                        left_pool = class_pools[pair_i[pid]]
                        right_pool = class_pools[pair_j[pid]]

                        left = _sample_indices(left_pool, count, generator=gen, allow_replacement=True)
                        right = _sample_indices(right_pool, count, generator=gen, allow_replacement=True)
                        sampled_src_parts.append(left)
                        sampled_dst_parts.append(right)

                    if sampled_src_parts:
                        src_cross = torch.cat(sampled_src_parts, dim=0)
                        dst_cross = torch.cat(sampled_dst_parts, dim=0)
                        src_edges.extend([src_cross, dst_cross])
                        dst_edges.extend([dst_cross, src_cross])

        # 3) Train->test edges only: test nodes are consumers and do not send to train nodes.
        if num_test > 0 and num_classes > 0:
            per_class_count = test_k_per_class * num_test
            dst_template = torch.arange(train_size, total_nodes, device=labels.device, dtype=torch.long).repeat_interleave(
                test_k_per_class
            )

            src_test_parts: list[Tensor] = []
            dst_test_parts: list[Tensor] = []
            for class_pool in class_pools:
                src_c = _sample_indices(class_pool, per_class_count, generator=gen, allow_replacement=True)
                src_test_parts.append(src_c)
                dst_test_parts.append(dst_template)

            src_test = torch.cat(src_test_parts, dim=0)
            dst_test = torch.cat(dst_test_parts, dim=0)
            src_edges.append(src_test)
            dst_edges.append(dst_test)

        edge_src = torch.cat(src_edges, dim=0)
        edge_dst = torch.cat(dst_edges, dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        return torch.unique(edge_index, dim=1)

    if share_graph_across_batch:
        labels_identical = bool((y_train == y_train[0]).all().item()) if batch_size > 1 else True
        if (not share_graph_require_identical_labels) or labels_identical:
            shared_edge_index = _build_single_graph(y_train[0].long())
            edge_index_batch = [shared_edge_index] * batch_size
            return SparseGraphBatch(edge_index=edge_index_batch, num_nodes=total_nodes)

    edge_index_batch: list[Tensor] = []
    for b in range(batch_size):
        edge_index_batch.append(_build_single_graph(y_train[b].long()))

    return SparseGraphBatch(edge_index=edge_index_batch, num_nodes=total_nodes)


def build_class_conditioned_graphs(
    y_train: Tensor,
    total_nodes: int,
    num_graphs: int = 1,
    min_train_neighbors: int = 8,
    max_train_neighbors: int = 15,
    same_label_ratio: float = 0.9,
    cross_label_ratio: float = 0.1,
    test_k_per_class: int = 3,
    seed: Optional[int] = None,
    share_graph_across_batch: bool = False,
    share_graph_require_identical_labels: bool = True,
) -> SparseGraphSet:
    """Build multiple independently sampled class-conditioned graph batches.

    Each graph uses the same sampling rules as
    :func:`build_class_conditioned_graph`. When ``seed`` is supplied, graph
    ``i`` uses ``seed + i`` so the complete graph set is reproducible while
    individual graphs remain independently sampled.
    """

    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive")

    graphs = [
        build_class_conditioned_graph(
            y_train=y_train,
            total_nodes=total_nodes,
            min_train_neighbors=min_train_neighbors,
            max_train_neighbors=max_train_neighbors,
            same_label_ratio=same_label_ratio,
            cross_label_ratio=cross_label_ratio,
            test_k_per_class=test_k_per_class,
            seed=None if seed is None else seed + graph_idx,
            share_graph_across_batch=share_graph_across_batch,
            share_graph_require_identical_labels=share_graph_require_identical_labels,
        )
        for graph_idx in range(num_graphs)
    ]
    return SparseGraphSet(graphs=graphs)
