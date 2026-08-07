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


class GraphPrior:
    """Configurable prior for tabular and graph-shaped classification tasks.

    ``y`` passed to :meth:`__call__` contains labels for all nodes and
    ``n_train`` identifies the train/test boundary. ``v1`` delegates to the
    historical class-conditioned sampler; ``v2`` supports tabular and full
    graph sampling modes.
    """

    def __init__(
        self,
        tab_graphs: str = "v1",
        mode_prob: float = 1.0,
        homophily_prob: float = 0.7,
        transition_scope_prob: float = 0.5,
        remain_prob_range: tuple[float, float] = (0.7, 0.99),
        heterophily_structure_probs: Optional[dict[str, float]] = None,
        group_count_range: tuple[int, int] = (2, 4),
        min_train_neighbors: int = 8,
        max_train_neighbors: int = 15,
        cross_label_fraction: float = 0.1,
        train_neighbors_per_test: int = 3,
        seed: Optional[int] = None,
        share_graph_across_batch: bool = False,
    ):
        if tab_graphs not in ("v1", "v2"):
            raise ValueError("tab_graphs must be either 'v1' or 'v2'")
        if not 0.0 <= mode_prob <= 1.0:
            raise ValueError("mode_prob must be between 0 and 1")
        if not 0.0 <= homophily_prob <= 1.0 or not 0.0 <= transition_scope_prob <= 1.0:
            raise ValueError("homophily_prob and transition_scope_prob must be between 0 and 1")
        if len(remain_prob_range) != 2 or not 0.0 <= remain_prob_range[0] <= remain_prob_range[1] <= 1.0:
            raise ValueError("remain_prob_range must be an ordered interval in [0, 1]")
        if min_train_neighbors <= 0 or max_train_neighbors < min_train_neighbors:
            raise ValueError("invalid train-neighbor bounds")
        if train_neighbors_per_test <= 0:
            raise ValueError("train_neighbors_per_test must be positive")
        if not 0.0 <= cross_label_fraction <= 1.0:
            raise ValueError("cross_label_fraction must be between 0 and 1")
        self.tab_graphs = tab_graphs
        self.mode_prob = float(mode_prob)
        self.homophily_prob = float(homophily_prob)
        self.transition_scope_prob = float(transition_scope_prob)
        self.remain_prob_range = tuple(float(value) for value in remain_prob_range)
        self.heterophily_structure_probs = heterophily_structure_probs or {
            "bipartite": 1 / 3,
            "cyclic": 1 / 3,
            "grouped": 1 / 3,
        }
        if not self.heterophily_structure_probs or any(value < 0 for value in self.heterophily_structure_probs.values()):
            raise ValueError("heterophily_structure_probs must contain non-negative weights")
        if sum(self.heterophily_structure_probs.values()) <= 0:
            raise ValueError("heterophily_structure_probs must have positive total weight")
        if len(group_count_range) != 2 or group_count_range[0] <= 0 or group_count_range[0] > group_count_range[1]:
            raise ValueError("group_count_range must be a positive ordered interval")
        self.group_count_range = tuple(int(value) for value in group_count_range)
        self.min_train_neighbors = int(min_train_neighbors)
        self.max_train_neighbors = int(max_train_neighbors)
        self.cross_label_fraction = float(cross_label_fraction)
        self.train_neighbors_per_test = int(train_neighbors_per_test)
        self.seed = seed
        self.share_graph_across_batch = bool(share_graph_across_batch)

    @staticmethod
    def _weighted_choice(names: list[str], weights: list[float], generator: torch.Generator, device: torch.device) -> str:
        probabilities = torch.tensor(weights, dtype=torch.float32, device=device)
        return names[int(torch.multinomial(probabilities, 1, generator=generator).item())]

    def _transition_matrix(self, labels: Tensor, generator: torch.Generator, homophilic: bool) -> Tensor:
        classes = torch.unique(labels)
        num_classes = len(classes)
        matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=labels.device)
        if num_classes == 1:
            matrix.fill_(1.0)
            return matrix
        if homophilic:
            dataset_wide = bool(torch.rand((), generator=generator, device=labels.device) < self.transition_scope_prob)
            remain = torch.empty(num_classes, device=labels.device).uniform_(*self.remain_prob_range, generator=generator)
            if dataset_wide:
                remain.fill_(float(remain[0]))
            matrix[torch.arange(num_classes), torch.arange(num_classes)] = remain
            matrix += (1.0 - remain[:, None]) / (num_classes - 1)
            matrix[torch.arange(num_classes), torch.arange(num_classes)] = remain
            return matrix

        names = list(self.heterophily_structure_probs)
        structure = self._weighted_choice(names, [self.heterophily_structure_probs[name] for name in names], generator, labels.device)
        if structure == "cyclic":
            for index in range(num_classes):
                matrix[index, (index + 1) % num_classes] = 1.0
        elif structure == "bipartite":
            split = max(1, num_classes // 2)
            for index in range(num_classes):
                source_partition = index < split
                targets = list(range(split, num_classes)) if source_partition else list(range(split))
                if not targets:
                    targets = [value for value in range(num_classes) if value != index]
                matrix[index, targets] = 1.0 / len(targets)
        else:
            group_count = min(num_classes, int(torch.randint(self.group_count_range[0], self.group_count_range[1] + 1, (1,), generator=generator, device=labels.device)))
            permutation = torch.randperm(num_classes, generator=generator, device=labels.device)
            groups = torch.tensor_split(permutation, group_count)
            for source in range(num_classes):
                group = next(group for group in groups if bool((group == source).any()))
                group_values = {int(value) for value in group.tolist()}
                same_group = [value for value in group_values if value != source]
                other = [value for value in range(num_classes) if value not in group_values]
                if same_group:
                    matrix[source, same_group] = 0.9 / len(same_group)
                if other:
                    matrix[source, other] = 0.1 / len(other)
                if not same_group and not other:
                    matrix[source, source] = 1.0
        return matrix

    def _sample_transition_edges(self, labels: Tensor, matrix: Tensor, generator: torch.Generator) -> Tensor:
        classes = torch.unique(labels)
        class_indices = [torch.where(labels == value)[0] for value in classes]
        neighbors = int(torch.randint(self.min_train_neighbors, self.max_train_neighbors + 1, (1,), generator=generator, device=labels.device))
        sources: list[int] = []
        destinations: list[int] = []
        for source, label in enumerate(labels.tolist()):
            class_index = int(torch.where(classes == label)[0][0])
            target_classes = torch.multinomial(matrix[class_index], neighbors, replacement=True, generator=generator)
            for target_class in target_classes.tolist():
                candidates = class_indices[target_class][class_indices[target_class] != source]
                if candidates.numel():
                    destination = int(candidates[torch.randint(candidates.numel(), (1,), generator=generator, device=labels.device)])
                    sources.append(source)
                    destinations.append(destination)
        if not sources:
            return torch.empty((2, 0), dtype=torch.uint16, device=labels.device)
        return torch.unique(torch.tensor([sources, destinations], dtype=torch.long, device=labels.device), dim=1).to(torch.uint16)

    def _sample_random_pairs(
        self,
        labels: Tensor,
        count: int,
        same_class: bool,
        generator: torch.Generator,
        class_index: Optional[int] = None,
    ) -> Tensor:
        """Sample unique unordered train pairs without materializing pair pools."""
        if count <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=labels.device)

        classes = torch.unique(labels)
        pools = [torch.where(labels == value)[0] for value in classes]
        if class_index is not None:
            pools = [pools[class_index]]
        pairs: set[tuple[int, int]] = set()
        max_attempts = max(100, count * 30)
        for _ in range(max_attempts):
            if len(pairs) >= count:
                break
            if same_class:
                eligible = [pool for pool in pools if pool.numel() >= 2]
                if not eligible:
                    break
                pool = eligible[int(torch.randint(len(eligible), (1,), generator=generator, device=labels.device))]
                positions = torch.randperm(pool.numel(), generator=generator, device=labels.device)[:2]
                left, right = int(pool[positions[0]]), int(pool[positions[1]])
            else:
                eligible = [pool for pool in pools if pool.numel() > 0]
                if len(eligible) < 2:
                    break
                class_positions = torch.randperm(len(eligible), generator=generator, device=labels.device)[:2]
                left_pool, right_pool = eligible[int(class_positions[0])], eligible[int(class_positions[1])]
                left = int(left_pool[torch.randint(left_pool.numel(), (1,), generator=generator, device=labels.device)])
                right = int(right_pool[torch.randint(right_pool.numel(), (1,), generator=generator, device=labels.device)])
            pairs.add((min(left, right), max(left, right)))

        if not pairs:
            return torch.empty((2, 0), dtype=torch.long, device=labels.device)
        return torch.tensor(list(pairs), dtype=torch.long, device=labels.device).T

    def _sample_tabular_train_edges(self, labels: Tensor, generator: torch.Generator) -> Tensor:
        """Sample v2 tabular train edges from explicit intra/inter budgets.

        Intra-class budgets are proportional to the number of possible pairs in
        each class. This makes the expected pair density comparable across
        classes while using class frequency to determine each class budget.
        """
        train_size = labels.numel()
        classes = torch.unique(labels)
        class_sizes = torch.bincount(
            torch.searchsorted(classes, labels), minlength=classes.numel()
        ).to(dtype=torch.long)
        intra_capacities = class_sizes * (class_sizes - 1) // 2
        inter_capacity = sum(
            int(class_sizes[i]) * int(class_sizes[j])
            for i in range(len(class_sizes))
            for j in range(i + 1, len(class_sizes))
        )

        pair_budget = int(
            torch.randint(
                train_size * self.min_train_neighbors,
                train_size * self.max_train_neighbors + 1,
                (1,),
                generator=generator,
                device=labels.device,
            ).item()
        )
        requested_intra = int(round(pair_budget * (1.0 - self.cross_label_fraction)))
        requested_inter = pair_budget - requested_intra
        intra_budget = min(requested_intra, int(intra_capacities.sum().item()))
        inter_budget = min(requested_inter, inter_capacity)

        # If one category is exhausted, let the other category use its
        # remaining capacity, matching the legacy sampler's behavior.
        remaining = pair_budget - intra_budget - inter_budget
        if remaining:
            extra_intra = min(remaining, int(intra_capacities.sum().item()) - intra_budget)
            intra_budget += extra_intra
            remaining -= extra_intra
        if remaining:
            inter_budget += min(remaining, inter_capacity - inter_budget)

        # Allocate intra-class pairs according to class pair capacity. Since
        # capacity is proportional to frequency squared, this equalizes pair
        # density rather than over-sampling small classes.
        class_budgets = torch.zeros_like(intra_capacities)
        total_capacity = int(intra_capacities.sum().item())
        if intra_budget and total_capacity:
            raw = intra_capacities.to(torch.float64) * intra_budget / total_capacity
            class_budgets = torch.floor(raw).to(torch.long)
            remainder = intra_budget - int(class_budgets.sum().item())
            order = torch.argsort(raw - class_budgets.to(raw.dtype), descending=True)
            while remainder:
                progressed = False
                for index in order.tolist():
                    if class_budgets[index] < intra_capacities[index]:
                        class_budgets[index] += 1
                        remainder -= 1
                        progressed = True
                        if not remainder:
                            break
                if not progressed:
                    break

        intra_parts = []
        for class_index, budget in enumerate(class_budgets.tolist()):
            if budget <= 0:
                continue
            intra_parts.append(
                self._sample_random_pairs(
                    labels,
                    budget,
                    same_class=True,
                    generator=generator,
                    class_index=class_index,
                )
            )

        # Sample inter-class pairs from the full label pool, then mirror all
        # train-train pairs to retain the directed graph convention.
        inter_pairs = self._sample_random_pairs(labels, inter_budget, same_class=False, generator=generator)
        pair_parts = [part for part in intra_parts + [inter_pairs] if part.shape[1] > 0]
        if not pair_parts:
            return torch.empty((2, 0), dtype=torch.long, device=labels.device)
        pairs = torch.cat(pair_parts, dim=1)
        return torch.cat([pairs, pairs.flip(0)], dim=1)

    def _sample_v2_single(self, labels: Tensor, n_train: int, generator: torch.Generator, tabular: bool) -> Tensor:
        if tabular:
            train_labels = labels[:n_train]
            edges = self._sample_tabular_train_edges(train_labels, generator)
            if n_train < labels.numel():
                classes = torch.unique(train_labels)
                test_nodes = torch.arange(n_train, labels.numel(), device=labels.device)
                parts = []
                for class_value in classes:
                    pool = torch.where(train_labels == class_value)[0]
                    positions = torch.randint(pool.numel(), (test_nodes.numel(), self.train_neighbors_per_test), generator=generator, device=labels.device)
                    parts.append(torch.stack([pool[positions].reshape(-1), test_nodes[:, None].expand(-1, self.train_neighbors_per_test).reshape(-1)]))
                if parts:
                    edges = torch.cat([edges, torch.cat(parts, dim=1)], dim=1)
        else:
            homophilic = bool(torch.rand((), generator=generator, device=labels.device) < self.homophily_prob)
            matrix = self._transition_matrix(labels, generator, homophilic=homophilic)
            edges = self._sample_transition_edges(labels, matrix, generator).long()
        return torch.unique(edges, dim=1).to(torch.uint16)

    def __call__(self, y: Tensor, n_train: int, num_graphs: int = 1) -> CompactGraphSet:
        if y.ndim != 2 or not 0 < n_train <= y.shape[1]:
            raise ValueError("y must have shape (batch, total_nodes) and n_train must be valid")
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive")
        if self.tab_graphs == "v1":
            return build_class_conditioned_graphs(
                y_train=y[:, :n_train].long(), total_nodes=y.shape[1], num_graphs=num_graphs,
                min_train_neighbors=self.min_train_neighbors, max_train_neighbors=self.max_train_neighbors,
                cross_label_fraction=self.cross_label_fraction, train_neighbors_per_test=self.train_neighbors_per_test,
                seed=self.seed, share_graph_across_batch=self.share_graph_across_batch,
            )
        per_dataset: list[list[Tensor]] = []
        for batch_index in range(y.shape[0]):
            generator = torch.Generator(device=y.device)
            if self.seed is not None:
                generator.manual_seed(self.seed + batch_index)
            tabular = bool(torch.rand((), generator=generator, device=y.device) < self.mode_prob)
            sampled = [self._sample_v2_single(y[batch_index].long(), n_train, generator, tabular) for _ in range(num_graphs if tabular else 1)]
            if not tabular:
                sampled *= num_graphs
            per_dataset.append(sampled)
        graphs = [
            SparseGraphBatch(
                edge_index=[per_dataset[dataset_index][graph_index] for dataset_index in range(y.shape[0])],
                num_nodes=y.shape[1],
            )
            for graph_index in range(num_graphs)
        ]
        return _compact_from_graph_batches(graphs, y.shape[1])


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
