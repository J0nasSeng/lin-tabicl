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


@dataclass
class CompactGraphSet:
    """Tensor-backed graph set used across DataLoader and training boundaries.

    ``edge_index`` concatenates all edges in graph-major, dataset-major order.
    ``edge_offsets`` has shape ``(num_graphs, num_datasets + 1)`` and identifies
    the slice belonging to each graph/dataset pair.
    """

    edge_index: Tensor
    edge_offsets: Tensor
    num_nodes: int

    @property
    def num_graphs(self) -> int:
        return int(self.edge_offsets.shape[0])

    @property
    def num_datasets(self) -> int:
        return int(self.edge_offsets.shape[1] - 1)

    def __len__(self) -> int:
        return self.num_datasets

    def __iter__(self):
        """Yield single-dataset compact views for legacy callers."""
        for index in range(self.num_datasets):
            yield self.slice(index)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        return self.slice(index)

    def slice(self, index: int | slice) -> "CompactGraphSet":
        selected = [index] if isinstance(index, int) else list(range(self.num_datasets))[index]
        if not selected:
            offsets = torch.zeros((self.num_graphs, 1), dtype=torch.long, device=self.edge_offsets.device)
            return CompactGraphSet(self.edge_index[:, :0], offsets, self.num_nodes)
        parts = []
        rows = []
        running = 0
        for graph_idx in range(self.num_graphs):
            row = [running]
            for dataset_idx in selected:
                start = int(self.edge_offsets[graph_idx, dataset_idx])
                end = int(self.edge_offsets[graph_idx, dataset_idx + 1])
                parts.append(self.edge_index[:, start:end])
                running += end - start
                row.append(running)
            rows.append(row)
        return CompactGraphSet(torch.cat(parts, dim=1), torch.tensor(rows, dtype=torch.long), self.num_nodes)

    def concat(self, other: "CompactGraphSet") -> "CompactGraphSet":
        """Concatenate two already-batched payloads along the dataset axis."""
        if self.num_graphs != other.num_graphs or self.num_nodes != other.num_nodes:
            raise ValueError("Compact graph batches must have matching graph and node dimensions")
        if self.edge_index.device != other.edge_index.device:
            raise ValueError("Compact graph batches must be on the same device")
        left_edges = self.edge_index.shape[1]
        right_offsets = other.edge_offsets[:, 1:] + left_edges
        offsets = torch.cat([self.edge_offsets, right_offsets], dim=1)
        return CompactGraphSet(torch.cat([self.edge_index, other.edge_index], dim=1), offsets, self.num_nodes)

    @property
    def graphs(self) -> list["CompactGraphBatch"]:
        """Compatibility view; the DataLoader payload remains tensor-backed."""
        return [
            CompactGraphBatch(
                edge_index=[self.edge_index[:, start:end] for start, end in zip(
                    offsets[:-1].tolist(), offsets[1:].tolist()
                )],
                num_nodes=self.num_nodes,
            )
            for offsets in self.edge_offsets
        ]


@dataclass
class CompactGraphBatch:
    edge_index: list[Tensor]
    num_nodes: int


def _compact_from_graph_batches(graphs: list[SparseGraphBatch | CompactGraphBatch], num_nodes: int) -> CompactGraphSet:
    parts: list[Tensor] = []
    offset_rows = []
    running = 0
    for graph in graphs:
        row = [running]
        for edge_index in graph.edge_index:
            parts.append(edge_index)
            running += edge_index.shape[1]
            row.append(running)
        offset_rows.append(row)
    edge_index = torch.cat(parts, dim=1) if parts else torch.empty((2, 0), dtype=torch.uint16)
    return CompactGraphSet(
        edge_index=edge_index,
        edge_offsets=torch.tensor(offset_rows, dtype=torch.long, device=edge_index.device),
        num_nodes=num_nodes,
    )


def stack_graph_sets(graph_sets: list[SparseGraphSet | CompactGraphSet]) -> CompactGraphSet:
    """Combine per-dataset graph sets into graph batches for model input."""

    if not graph_sets:
        raise ValueError("graph_sets must not be empty")
    if isinstance(graph_sets[0], CompactGraphSet):
        compact = [graph for graph in graph_sets if isinstance(graph, CompactGraphSet)]
        if len(compact) != len(graph_sets):
            raise ValueError("Cannot mix compact and sparse graph sets")
        num_graphs = compact[0].num_graphs
        if any(graph.num_datasets != 1 for graph in compact):
            raise ValueError("stack_graph_sets expects one compact graph set per dataset")
        if any(graph.num_graphs != num_graphs for graph in compact):
            raise ValueError("All graph sets must contain the same number of graphs")
        parts = []
        offset_rows = []
        running = 0
        for graph_idx in range(num_graphs):
            row = [running]
            for graph in compact:
                start, end = graph.edge_offsets[graph_idx, 0].item(), graph.edge_offsets[graph_idx, 1].item()
                parts.append(graph.edge_index[:, start:end])
                running += end - start
                row.append(running)
            offset_rows.append(row)
        edge_index = torch.cat(parts, dim=1) if parts else compact[0].edge_index[:, :0]
        return CompactGraphSet(edge_index, torch.tensor(offset_rows, dtype=torch.long), compact[0].num_nodes)

    # Legacy input is converted once at the boundary.
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
    return _compact_from_graph_batches(graphs, graphs[0].num_nodes)


def slice_graph_sets(graph_sets: CompactGraphSet, indices: slice) -> CompactGraphSet:
    """Slice a compact graph set along its dataset dimension."""
    return graph_sets.slice(indices)


def concat_compact_graph_sets(first: CompactGraphSet, second: CompactGraphSet) -> CompactGraphSet:
    """Concatenate compact graph payloads without reconstructing graph objects."""
    return first.concat(second)


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

        # Sample pairs directly.  The previous implementation materialized all
        # same-label combinations and cross-label Cartesian products, which is
        # quadratic in the training-set size.
        def _sample_pairs(kind: str, count: int) -> Tensor:
            if count <= 0:
                return torch.empty((2, 0), dtype=torch.long, device=labels.device)
            eligible = [pool for pool in class_pools if pool.numel() >= 2] if kind == "same" else class_pools
            if kind == "cross":
                eligible = [pool for pool in class_pools if pool.numel() > 0]
                if len(eligible) < 2:
                    return torch.empty((2, 0), dtype=torch.long, device=labels.device)
            pairs: set[tuple[int, int]] = set()
            max_attempts = max(100, count * 20)
            for _ in range(max_attempts):
                if len(pairs) >= count:
                    break
                if kind == "same":
                    pool = eligible[int(torch.randint(len(eligible), (1,), generator=gen, device=labels.device))]
                    positions = torch.randperm(pool.numel(), generator=gen, device=labels.device)[:2]
                    left, right = int(pool[positions[0]]), int(pool[positions[1]])
                else:
                    class_ids = torch.randperm(len(eligible), generator=gen, device=labels.device)[:2]
                    left_pool, right_pool = eligible[int(class_ids[0])], eligible[int(class_ids[1])]
                    left = int(left_pool[torch.randint(left_pool.numel(), (1,), generator=gen, device=labels.device)])
                    right = int(right_pool[torch.randint(right_pool.numel(), (1,), generator=gen, device=labels.device)])
                pairs.add((min(left, right), max(left, right)))
            if not pairs:
                return torch.empty((2, 0), dtype=torch.long, device=labels.device)
            return torch.tensor(list(pairs), dtype=torch.long, device=labels.device).T

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

        # Compute capacities analytically so an exhausted category can donate
        # its remaining budget without constructing candidate pairs.
        same_capacity = sum(int(pool.numel()) * (int(pool.numel()) - 1) // 2 for pool in class_pools)
        cross_capacity = sum(
            int(class_pools[i].numel()) * int(class_pools[j].numel())
            for i in range(num_classes)
            for j in range(i + 1, num_classes)
        )
        same_count = min(requested_same, same_capacity)
        cross_count = min(requested_cross, cross_capacity)
        remaining = total_pair_budget - same_count - cross_count
        if remaining:
            extra_same = min(remaining, same_capacity - same_count)
            same_count += extra_same
            remaining -= extra_same
        if remaining:
            cross_count += min(remaining, cross_capacity - cross_count)

        same_edges = _sample_pairs("same", same_count)
        cross_edges = _sample_pairs("cross", cross_count)
        if same_edges.shape[1] > 0:
            src_edges.extend([same_edges[0], same_edges[1]])
            dst_edges.extend([same_edges[1], same_edges[0]])
        if cross_edges.shape[1] > 0:
            src_edges.extend([cross_edges[0], cross_edges[1]])
            dst_edges.extend([cross_edges[1], cross_edges[0]])

        # 3) Train->test edges only: test nodes are consumers and do not send to train nodes.
        if num_test > 0 and num_classes > 0:
            src_test_parts: list[Tensor] = []
            dst_test_parts: list[Tensor] = []
            test_nodes = torch.arange(train_size, total_nodes, device=labels.device, dtype=torch.long)
            for class_pool in class_pools:
                pool_size = class_pool.numel()
                if pool_size < train_neighbors_per_test:
                    positions = torch.randint(
                        pool_size,
                        (num_test, train_neighbors_per_test),
                        generator=gen,
                        device=labels.device,
                    )
                else:
                    # A separate permutation per test node avoids the nested
                    # Python loop while retaining sampling without replacement.
                    positions = torch.argsort(
                        torch.rand((num_test, pool_size), generator=gen, device=labels.device), dim=1
                    )[:, :train_neighbors_per_test]
                src_test_parts.append(class_pool[positions].reshape(-1))
                dst_test_parts.append(test_nodes[:, None].expand(-1, train_neighbors_per_test).reshape(-1))

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
) -> CompactGraphSet:
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
            return _compact_from_graph_batches(
                [
                    SparseGraphBatch(
                        edge_index=[graph.edge_index[0]] * y_train.shape[0],
                        num_nodes=total_nodes,
                    )
                    for graph in shared
                ],
                total_nodes,
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
    return _compact_from_graph_batches(graphs, total_nodes)
