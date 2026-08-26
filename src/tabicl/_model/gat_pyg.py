from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

if TYPE_CHECKING:
    from .graph import CompactGraphSet


class GraphMultiheadAttention(MessagePassing):
    """PyG implementation of the legacy sparse graph multi-head attention."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        max_parallel_edges: int = 2**12,
        learnable_residual: bool = False,
        max_chunk_size: int | None = None,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if max_parallel_edges <= 0 or (max_chunk_size is not None and max_chunk_size <= 0):
            raise ValueError("edge chunk sizes must be positive")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.learnable_residual = bool(learnable_residual)
        self.max_parallel_edges = int(max_parallel_edges)
        self.max_chunk_size = max_chunk_size
        self._sorted_group_cache: dict[tuple[int, int, int, str], tuple[Tensor, list[int]]] = {}
        if self.learnable_residual:
            self.alpha = nn.Parameter(torch.logit(torch.tensor(0.2, dtype=torch.float32)))

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        logits = (q_i * k_j).sum(dim=-1) * self.scale
        weights = softmax(logits, index=index, ptr=ptr, num_nodes=size_i)
        return v_j * self.dropout(weights).unsqueeze(-1)

    @staticmethod
    def _edge_blocks(
        edge_index_batch: Sequence[Tensor] | Sequence[Sequence[Tensor]],
        columns: int,
        nodes: int,
        batch_size: int,
        device: torch.device,
        max_edges: int,
        edge_index_is_global: bool = False,
        sorted_group_cache: dict[tuple[int, int, int, str], tuple[Tensor, list[int]]] | None = None,
    ):
        """Yield blocks without splitting destination softmax groups.

        Sorted edges and group lengths are cached. Block construction keeps
        contiguous ranges instead of allocating one tensor per destination
        group, matching the optimized benchmark implementation.
        """
        if max_edges <= 0:
            raise ValueError("max_edges must be positive")
        block_ranges: list[tuple[Tensor, int, int, int]] = []
        block_size = 0

        def emit_block() -> Tensor:
            return torch.cat(
                [edges[:, start:end] + offset for edges, start, end, offset in block_ranges],
                dim=1,
            )

        if isinstance(edge_index_batch, Tensor):
            edge_items = [edge_index_batch]
        else:
            edge_items = list(edge_index_batch)
        if edge_items and not isinstance(edge_items[0], Tensor):
            edge_items = [edge for dataset_edges in edge_items for edge in dataset_edges]
        for batch_index, base_edges in enumerate(edge_items):
            edge_index = base_edges.to(device=device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("Each edge index must have shape (2, E)")
            if edge_index.numel() == 0:
                continue
            upper_bound = batch_size * nodes if edge_index_is_global else nodes
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= upper_bound:
                raise ValueError(
                    f"edge_index for batch element {batch_index} must be within [0, {upper_bound - 1}]"
                )
            cache_key = (
                id(base_edges), base_edges.data_ptr(), base_edges.shape[1], str(device)
            )
            cached = sorted_group_cache.get(cache_key) if sorted_group_cache is not None else None
            if cached is None:
                order = torch.argsort(edge_index[1], stable=True)
                sorted_edges = edge_index[:, order]
                starts = torch.cat(
                    [
                        torch.zeros(1, dtype=torch.long, device=device),
                        torch.where(sorted_edges[1, 1:] != sorted_edges[1, :-1])[0] + 1,
                        torch.tensor([sorted_edges.shape[1]], dtype=torch.long, device=device),
                    ]
                ).tolist()
                group_lengths = [end - start for start, end in zip(starts[:-1], starts[1:])]
                if sorted_group_cache is not None:
                    sorted_group_cache[cache_key] = (sorted_edges, group_lengths)
            else:
                sorted_edges, group_lengths = cached
            group_offsets = [0]
            group_offsets.extend(group_offsets[-1] + length for length in group_lengths)
            for column in range(columns):
                offset = (column * batch_size * nodes) if edge_index_is_global else (batch_index * columns + column) * nodes
                group_start = 0
                for group_size in group_lengths:
                    if block_ranges and block_size + group_size > max_edges:
                        yield emit_block()
                        block_ranges = []
                        block_size = 0
                    group_end = group_start + 1
                    start = group_offsets[group_start]
                    end = group_offsets[group_end]
                    if (
                        block_ranges
                        and block_ranges[-1][0] is sorted_edges
                        and block_ranges[-1][3] == offset
                        and block_ranges[-1][2] == start
                    ):
                        previous_edges, previous_start, _, previous_offset = block_ranges[-1]
                        block_ranges[-1] = (previous_edges, previous_start, end, previous_offset)
                    else:
                        block_ranges.append((sorted_edges, start, end, offset))
                    block_size += group_size
                    group_start = group_end
                    if block_size >= max_edges:
                        yield emit_block()
                        block_ranges = []
                        block_size = 0

        if block_ranges:
            yield emit_block()

    def forward(
        self,
        src: Tensor,
        edge_index_batch: Sequence[Tensor] | Tensor,
        residual_src: Tensor | None = None,
        edge_index_is_global: bool = False,
    ) -> Tensor:
        if src.ndim != 4:
            raise ValueError("src must have shape (B, T, C, D)")
        if not isinstance(edge_index_batch, Tensor) and len(edge_index_batch) != src.shape[0]:
            raise ValueError("edge_index_batch length must equal batch size")
        if residual_src is None:
            residual_src = src
        if residual_src.shape != src.shape:
            raise ValueError("residual_src must have the same shape as src")
        batch_size, nodes, columns, _ = src.shape
        q = self.q_proj(src).view(batch_size, nodes, columns, self.nhead, self.head_dim)
        k = self.k_proj(src).view(batch_size, nodes, columns, self.nhead, self.head_dim)
        v = self.v_proj(src).view(batch_size, nodes, columns, self.nhead, self.head_dim)
        # Flatten in the same (batch, column, node) order used by edge offsets.
        q = q.permute(0, 2, 1, 3, 4).reshape(-1, self.nhead, self.head_dim)
        k = k.permute(0, 2, 1, 3, 4).reshape(-1, self.nhead, self.head_dim)
        v = v.permute(0, 2, 1, 3, 4).reshape(-1, self.nhead, self.head_dim)
        aggregated = torch.zeros_like(q)
        chunk_size = self.max_chunk_size or self.max_parallel_edges
        any_edges = False
        for edge_index in self._edge_blocks(
            edge_index_batch, columns, nodes, batch_size, src.device, chunk_size,
            edge_index_is_global, self._sorted_group_cache,
        ):
            any_edges = True
            aggregated.add_(self.propagate(edge_index, q=q, k=k, v=v, size=(q.shape[0], q.shape[0])))
        if not any_edges:
            return residual_src
        attended = aggregated.reshape(batch_size, columns, nodes, self.d_model).permute(0, 2, 1, 3)
        output = self.out_proj(attended)
        if not self.learnable_residual:
            return residual_src + output
        alpha = torch.sigmoid(self.alpha).to(dtype=src.dtype)
        return (1.0 - alpha) * residual_src + alpha * output


class GraphAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        learnable_residual: bool = False,
        max_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.attn = GraphMultiheadAttention(
            d_model, nhead, dropout=dropout,
            learnable_residual=learnable_residual,
            max_chunk_size=max_chunk_size,
        )
        self.norm1 = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.norm2 = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            if activation == "gelu":
                self.activation = nn.GELU()
            elif activation == "relu":
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor, edge_index_batch: Sequence[Tensor], edge_index_is_global: bool = False) -> Tensor:
        if self.norm_first:
            x = self.dropout1(self.attn(self.norm1(src), edge_index_batch, residual_src=src, edge_index_is_global=edge_index_is_global))
            return x + self._ff_block(self.norm2(x))
        x = self.norm1(self.dropout1(self.attn(src, edge_index_batch, edge_index_is_global=edge_index_is_global)))
        return self.norm2(x + self._ff_block(x))


class Graph2DAttentionTransformer(nn.Module):
    """PyG-backed drop-in replacement for ``gat.GraphAttentionTransformer``."""

    def __init__(
        self, num_blocks: int, d_model: int, nhead: int, dim_feedforward: int,
        dropout: float = 0.0, activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True, bias_free_ln: bool = False, recompute: bool = False,
        num_output_cls: int | None = None, out_dim: int | None = None,
        learnable_residual: bool = False, num_graphs: int = 1,
        max_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0 or num_graphs <= 0 or num_blocks % num_graphs != 0:
            raise ValueError("num_blocks must be positive and divisible by num_graphs")
        if (num_output_cls is None) != (out_dim is None):
            raise ValueError("num_output_cls and out_dim must be provided together")
        self.graph_blocks = nn.ModuleList([
            GraphAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first,
                                 bias_free_ln, learnable_residual, max_chunk_size)
            for _ in range(num_blocks)
        ])
        self.col_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_blocks)
        ])
        self.col_attn_ln = nn.ModuleList([nn.LayerNorm(d_model, bias=not bias_free_ln) for _ in range(num_blocks)])
        self.num_output_cls = num_output_cls
        self.out_proj = nn.Linear(num_output_cls * d_model, out_dim) if out_dim is not None and num_output_cls is not None else None
        self.recompute = recompute
        self.num_graphs = num_graphs
        self.layers_per_graph = num_blocks // num_graphs

    @staticmethod
    def _compact_edges(graph_set: "CompactGraphSet", batch_size: int, device: torch.device) -> list[Tensor]:
        compact_edges = graph_set.edge_index.to(device=device, dtype=torch.long)
        result: list[Tensor] = []
        for graph_idx in range(graph_set.num_graphs):
            starts = graph_set.edge_offsets[graph_idx, :-1].to(device=device, dtype=torch.long)
            ends = graph_set.edge_offsets[graph_idx, 1:].to(device=device, dtype=torch.long)
            lengths = ends - starts
            if int(lengths.sum()) == 0:
                result.append(compact_edges[:, :0])
                continue
            dataset_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)
            positions = torch.cat([torch.arange(start, end, device=device) for start, end in zip(starts.tolist(), ends.tolist())])
            result.append(compact_edges[:, positions] + (dataset_ids * graph_set.num_nodes).unsqueeze(0))
        return result

    def forward(self, src: Tensor, edge_index_batch=None, graph_set: "CompactGraphSet | None" = None) -> Tensor:
        if src.ndim != 4:
            raise ValueError("GraphAttentionTransformer expects src with shape (B, T, C, D)")
        if graph_set is not None:
            if edge_index_batch is not None:
                raise ValueError("Provide either graph_set or edge_index_batch, not both")
            if graph_set.num_graphs != self.num_graphs or graph_set.num_datasets != src.shape[0] or graph_set.num_nodes != src.shape[1]:
                raise ValueError("Compact graph dimensions do not match transformer input")
            edge_index_batch = self._compact_edges(graph_set, src.shape[0], src.device)
            edge_index_is_global = True
        else:
            edge_index_is_global = False
        if edge_index_batch is None or len(edge_index_batch) != self.num_graphs:
            raise ValueError(f"Expected {self.num_graphs} graph batches")
        if self.num_output_cls is not None and src.shape[2] < self.num_output_cls:
            raise ValueError("GraphAttentionTransformer expects enough columns for output CLS")
        out = src
        for block_idx, block in enumerate(self.graph_blocks):
            edges = edge_index_batch[block_idx // self.layers_per_graph]
            if self.recompute:
                out = checkpoint(
                    partial(block, edge_index_batch=edges, edge_index_is_global=edge_index_is_global),
                    out, use_reentrant=False,
                )
            else:
                out = block(out, edges, edge_index_is_global=edge_index_is_global)
            flat = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
            attn_in = self.col_attn_ln[block_idx](flat)
            attn_out, _ = self.col_attn[block_idx](attn_in, attn_in, attn_in, need_weights=False)
            out = (flat + attn_out).reshape_as(out)
        if self.num_output_cls is None:
            return out
        cls = out[:, :, -self.num_output_cls:, :].reshape(out.shape[0], out.shape[1], -1)
        return cls if self.out_proj is None else self.out_proj(cls)


class Graph1DAttentionTransformer(nn.Module):
    """PyG sparse graph transformer for compressed ``(B, T, D)`` inputs."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
        recompute: bool = False,
        learnable_residual: bool = False,
        num_graphs: int = 1,
        max_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if num_graphs <= 0 or num_blocks % num_graphs != 0:
            raise ValueError("num_graphs must be positive and divide num_blocks")
        self.graph_blocks = nn.ModuleList([
            GraphAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                bias_free_ln=bias_free_ln,
                learnable_residual=learnable_residual,
                max_chunk_size=max_chunk_size,
            )
            for _ in range(num_blocks)
        ])
        self.recompute = recompute
        self.num_graphs = num_graphs
        self.layers_per_graph = num_blocks // num_graphs

    @staticmethod
    def _compact_edges(graph_set: "CompactGraphSet", batch_size: int, device: torch.device) -> list[Tensor]:
        compact_edges = graph_set.edge_index.to(device=device, dtype=torch.long)
        result: list[Tensor] = []
        for graph_idx in range(graph_set.num_graphs):
            starts = graph_set.edge_offsets[graph_idx, :-1].to(device=device, dtype=torch.long)
            ends = graph_set.edge_offsets[graph_idx, 1:].to(device=device, dtype=torch.long)
            lengths = ends - starts
            if int(lengths.sum().item()) == 0:
                result.append(compact_edges[:, :0])
                continue
            dataset_ids = torch.repeat_interleave(torch.arange(batch_size, device=device), lengths)
            positions = torch.cat([
                torch.arange(start, end, device=device)
                for start, end in zip(starts.tolist(), ends.tolist())
            ])
            result.append(compact_edges[:, positions] + (dataset_ids * graph_set.num_nodes).unsqueeze(0))
        return result

    def forward(
        self,
        src: Tensor,
        edge_index_batch: Sequence[Tensor] | None = None,
        graph_set: "CompactGraphSet" | None = None,
    ) -> Tensor:
        if src.ndim != 3:
            raise ValueError("Graph1DAttentionTransformer expects src with shape (B, T, D)")
        batch_size, nodes, _ = src.shape
        edge_index_is_global = graph_set is not None
        if graph_set is not None:
            if edge_index_batch is not None:
                raise ValueError("Provide either graph_set or edge_index_batch, not both")
            if graph_set.num_graphs != self.num_graphs or graph_set.num_datasets != batch_size:
                raise ValueError("Compact graph dimensions do not match the transformer input")
            if graph_set.num_nodes != nodes:
                raise ValueError("Compact graph node count does not match the transformer input")
            edge_index_batch = self._compact_edges(graph_set, batch_size, src.device)
        if edge_index_batch is None or len(edge_index_batch) != self.num_graphs:
            raise ValueError(f"Expected {self.num_graphs} graph batches")

        out = src.unsqueeze(2)
        for block_idx, block in enumerate(self.graph_blocks):
            edges = edge_index_batch[block_idx // self.layers_per_graph]
            if self.recompute:
                out = checkpoint(
                    partial(block, edge_index_batch=edges, edge_index_is_global=edge_index_is_global),
                    out, use_reentrant=False,
                )
            else:
                out = block(out, edge_index_batch=edges, edge_index_is_global=edge_index_is_global)
        return out.squeeze(2)


# Backwards-compatible name for the original alternating sample/column model.
GraphAttentionTransformer = Graph2DAttentionTransformer
