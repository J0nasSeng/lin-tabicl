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
    cross_label_fraction: float = 0.1,
    train_neighbors_per_test: int = 3,
    seed: int | None = None,
) -> SparseGraphBatch:
    """Build sparse directed graphs for graph-based in-context learning.

    Parameters
    ----------
    y_train : Tensor
        Training labels of shape (B, train_size).

    total_nodes : int
        Total number of nodes in the graph (train + test).

    min_train_neighbors : int, default=8
        Lower bound for the total train-train pair budget, expressed as a
        multiplier of the training-set size.

    max_train_neighbors : int, default=15
        Upper bound for the total train-train pair budget, expressed as a
        multiplier of the training-set size.

    cross_label_fraction : float, default=0.1
        Fraction of train-train pairs that should connect different labels.

    train_neighbors_per_test : int, default=3
        Number of sampled train connections per class for each test node before
        adding mirrored pairs.

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
    if train_neighbors_per_test <= 0:
        raise ValueError("train_neighbors_per_test must be positive")
    if not 0.0 <= cross_label_fraction <= 1.0:
        raise ValueError("cross_label_fraction must be between 0 and 1")

    batch_size, train_size = y_train.shape
    if total_nodes < train_size:
        raise ValueError("total_nodes must be >= train_size")
    if total_nodes > torch.iinfo(torch.uint16).max:
        raise ValueError("total_nodes must fit in an unsigned 16-bit integer")

    gen = torch.Generator(device=y_train.device)
    if seed is not None:
        gen.manual_seed(seed)

    def _build_single_graph(labels: Tensor) -> Tensor:
        train_indices = torch.arange(train_size, device=labels.device, dtype=torch.long)
        classes = torch.unique(labels)
        class_to_indices = {int(c.item()): train_indices[labels == c] for c in classes}
        class_pools = [class_to_indices[int(c.item())] for c in classes]
        num_classes = len(class_pools)
        num_test = max(0, total_nodes - train_size)

        src_edges: list[Tensor] = []
        dst_edges: list[Tensor] = []

        # Build finite unordered candidate pools. Sampling without replacement
        # makes the ratio describe the final unique train-train edges rather
        # than a replacement-based request that is later collapsed by unique().
        same_candidate_parts: list[Tensor] = []
        cross_candidate_parts: list[Tensor] = []
        for class_pool in class_pools:
            if class_pool.numel() >= 2:
                same_candidate_parts.append(torch.combinations(class_pool, r=2).T)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                left_pool = class_pools[i]
                right_pool = class_pools[j]
                if left_pool.numel() == 0 or right_pool.numel() == 0:
                    continue
                cross_candidate_parts.append(
                    torch.stack(
                        [
                            left_pool.repeat_interleave(right_pool.numel()),
                            right_pool.repeat(left_pool.numel()),
                        ],
                        dim=0,
                    )
                )

        def _concat_candidates(parts: list[Tensor]) -> Tensor:
            if not parts:
                return torch.empty((2, 0), dtype=torch.long, device=labels.device)
            return torch.cat(parts, dim=1)

        same_candidates = _concat_candidates(same_candidate_parts)
        cross_candidates = _concat_candidates(cross_candidate_parts)

        total_pair_budget = int(
            torch.randint(
                train_size * min_train_neighbors,
                train_size * max_train_neighbors + 1,
                (1,),
                generator=gen,
                device=labels.device,
            ).item()
        )
        requested_cross = int(round(total_pair_budget * cross_label_fraction))
        requested_same = total_pair_budget - requested_cross

        # If one candidate pool is exhausted, use any remaining budget from
        # the other pool rather than fabricating duplicate edges.
        same_count = min(requested_same, same_candidates.shape[1])
        cross_count = min(requested_cross, cross_candidates.shape[1])
        remaining = total_pair_budget - same_count - cross_count
        if remaining > 0:
            extra_same = min(remaining, same_candidates.shape[1] - same_count)
            same_count += extra_same
            remaining -= extra_same
        if remaining > 0:
            extra_cross = min(remaining, cross_candidates.shape[1] - cross_count)
            cross_count += extra_cross

        same_candidate_indices = torch.arange(same_candidates.shape[1], device=labels.device)
        cross_candidate_indices = torch.arange(cross_candidates.shape[1], device=labels.device)
        same_sample = _sample_indices(same_candidate_indices, same_count, generator=gen)
        cross_sample = _sample_indices(cross_candidate_indices, cross_count, generator=gen)

        if same_count > 0:
            same_edges = same_candidates[:, same_sample]
            src_edges.extend([same_edges[0], same_edges[1]])
            dst_edges.extend([same_edges[1], same_edges[0]])
        if cross_count > 0:
            cross_edges = cross_candidates[:, cross_sample]
            src_edges.extend([cross_edges[0], cross_edges[1]])
            dst_edges.extend([cross_edges[1], cross_edges[0]])

        # 3) Train->test edges only: test nodes are consumers and do not send to train nodes.
        if num_test > 0 and num_classes > 0:
            dst_template = torch.arange(train_size, total_nodes, device=labels.device, dtype=torch.long).repeat_interleave(
                train_neighbors_per_test
            )

            src_test_parts: list[Tensor] = []
            dst_test_parts: list[Tensor] = []
            test_nodes = torch.arange(train_size, total_nodes, device=labels.device, dtype=torch.long)
            for class_pool in class_pools:
                for test_node in test_nodes:
                    src_c = _sample_indices(
                        class_pool,
                        train_neighbors_per_test,
                        generator=gen,
                        allow_replacement=class_pool.numel() < train_neighbors_per_test,
                    )
                    src_test_parts.append(src_c)
                    dst_test_parts.append(test_node.expand(src_c.shape[0]))

            src_test = torch.cat(src_test_parts, dim=0)
            dst_test = torch.cat(dst_test_parts, dim=0)
            src_edges.append(src_test)
            dst_edges.append(dst_test)

        edge_src = torch.cat(src_edges, dim=0)
        edge_dst = torch.cat(dst_edges, dim=0)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)
        return torch.unique(edge_index, dim=1).to(dtype=torch.uint16)

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
    cross_label_fraction: float = 0.1,
    train_neighbors_per_test: int = 3,
    seed: Optional[int] = None,
    share_graph_across_batch: bool = False,
) -> SparseGraphSet:
    """Build multiple independently sampled class-conditioned graph batches.

    Each graph uses the same sampling rules as
    :func:`build_class_conditioned_graph`. When ``seed`` is supplied, graph
    ``i`` uses ``seed + i`` so the complete graph set is reproducible while
    individual graphs remain independently sampled.
    """

    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive")

    if share_graph_across_batch:
        labels_identical = y_train.shape[0] <= 1 or bool((y_train == y_train[0]).all().item())
        if labels_identical:
            shared = [
                build_class_conditioned_graph(
                    y_train=y_train[:1],
                    total_nodes=total_nodes,
                    min_train_neighbors=min_train_neighbors,
                    max_train_neighbors=max_train_neighbors,
                    cross_label_fraction=cross_label_fraction,
                    train_neighbors_per_test=train_neighbors_per_test,
                    seed=None if seed is None else seed + graph_idx,
                )
                for graph_idx in range(num_graphs)
            ]
            return SparseGraphSet(
                graphs=[
                    SparseGraphBatch(edge_index=[graph.edge_index[0]] * y_train.shape[0], num_nodes=total_nodes)
                    for graph in shared
                ]
            )

    graphs = [
        build_class_conditioned_graph(
            y_train=y_train,
            total_nodes=total_nodes,
            min_train_neighbors=min_train_neighbors,
            max_train_neighbors=max_train_neighbors,
            cross_label_fraction=cross_label_fraction,
            train_neighbors_per_test=train_neighbors_per_test,
            seed=None if seed is None else seed + graph_idx,
        )
        for graph_idx in range(num_graphs)
    ]
    return SparseGraphSet(graphs=graphs)
