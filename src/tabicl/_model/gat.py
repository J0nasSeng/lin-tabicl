from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


class GraphMultiheadAttention(nn.Module):
    """Sparse graph multi-head attention with per-destination softmax weights."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, max_parallel_edges: int = 65536):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        if max_parallel_edges <= 0:
            raise ValueError("max_parallel_edges must be > 0")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.logit(torch.tensor(0.05, dtype=torch.float32)))
        self.max_parallel_edges = int(max_parallel_edges)

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, src: Tensor, edge_index_batch: list[Tensor], residual_src: Tensor | None = None) -> Tensor:
        """Apply sparse graph attention.

        Parameters
        ----------
        src : Tensor
            Input node features of shape (B, T, C, D).

        edge_index_batch : list[Tensor]
            List of length B. Each tensor has shape (2, E) where index 0 is source
            node and index 1 is destination node.

        Returns
        -------
        Tensor
            Updated node features of shape (B, T, C, D).
        """

        if src.ndim != 4:
            raise ValueError("src must have shape (B, T, C, D)")
        if len(edge_index_batch) != src.shape[0]:
            raise ValueError("edge_index_batch length must equal batch size")

        if residual_src is None:
            residual_src = src
        if residual_src.shape != src.shape:
            raise ValueError("residual_src must have shape (B, T, C, D)")

        B, T, C, _ = src.shape
        all_edge_src: list[Tensor] = []
        all_edge_dst: list[Tensor] = []

        for b, edge_index in enumerate(edge_index_batch):
            edge_index = edge_index.to(device=src.device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("Each edge_index must have shape (2, E)")

            if edge_index.numel() > 0:
                edge_min = int(edge_index.min().item())
                edge_max = int(edge_index.max().item())
                if edge_min < 0 or edge_max >= T:
                    raise ValueError(
                        f"edge_index for batch element {b} must be within [0, {T - 1}], "
                        f"got min={edge_min}, max={edge_max}"
                    )

            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            if edge_src.numel() == 0:
                continue

            offset = b * T
            all_edge_src.append(edge_src + offset)
            all_edge_dst.append(edge_dst + offset)

        alpha = torch.sigmoid(self.alpha).to(dtype=src.dtype)

        # No incoming edges anywhere: keep only the self-connection part.
        if not all_edge_src:
            return (1.0 - alpha) * residual_src

        global_edge_src = torch.cat(all_edge_src, dim=0)
        global_edge_dst = torch.cat(all_edge_dst, dim=0)

        q = self.q_proj(src).view(B * T, C, self.nhead, self.head_dim)
        k = self.k_proj(src).view(B * T, C, self.nhead, self.head_dim)
        v = self.v_proj(src).view(B * T, C, self.nhead, self.head_dim)

        E = global_edge_dst.numel()
        head_ids = torch.arange(self.nhead, device=src.device, dtype=torch.long).view(1, 1, self.nhead)
        col_ids = torch.arange(C, device=src.device, dtype=torch.long).view(1, C, 1)
        num_groups = B * T * C * self.nhead
        chunk_size = min(E, self.max_parallel_edges)
        denom_floor = torch.finfo(src.dtype).tiny

        def _chunk_group_index(edge_dst_chunk: Tensor) -> Tensor:
            dst_rep = edge_dst_chunk.view(-1, 1, 1)
            return (dst_rep * (C * self.nhead) + col_ids * self.nhead + head_ids).reshape(-1)

        max_per_group = torch.full((num_groups,), float("-inf"), dtype=src.dtype, device=src.device)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            edge_src_chunk = global_edge_src[start:end]
            edge_dst_chunk = global_edge_dst[start:end]

            q_dst_chunk = q[edge_dst_chunk]
            k_src_chunk = k[edge_src_chunk]
            logits_chunk = ((q_dst_chunk * k_src_chunk).sum(dim=-1) * self.scale).reshape(-1)
            group_index_chunk = _chunk_group_index(edge_dst_chunk)
            max_per_group.scatter_reduce_(0, group_index_chunk, logits_chunk, reduce="amax", include_self=True)

        sum_per_group = torch.zeros((num_groups,), dtype=src.dtype, device=src.device)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            edge_src_chunk = global_edge_src[start:end]
            edge_dst_chunk = global_edge_dst[start:end]

            q_dst_chunk = q[edge_dst_chunk]
            k_src_chunk = k[edge_src_chunk]
            logits_chunk = ((q_dst_chunk * k_src_chunk).sum(dim=-1) * self.scale).reshape(-1)
            group_index_chunk = _chunk_group_index(edge_dst_chunk)
            exp_chunk = torch.exp(logits_chunk - max_per_group[group_index_chunk])
            sum_per_group.index_add_(0, group_index_chunk, exp_chunk)

        agg = torch.zeros((B * T, C, self.nhead, self.head_dim), dtype=src.dtype, device=src.device)
        for start in range(0, E, chunk_size):
            end = min(start + chunk_size, E)
            edge_src_chunk = global_edge_src[start:end]
            edge_dst_chunk = global_edge_dst[start:end]

            q_dst_chunk = q[edge_dst_chunk]
            k_src_chunk = k[edge_src_chunk]
            v_src_chunk = v[edge_src_chunk]
            logits_chunk = ((q_dst_chunk * k_src_chunk).sum(dim=-1) * self.scale).reshape(-1)
            group_index_chunk = _chunk_group_index(edge_dst_chunk)
            exp_chunk = torch.exp(logits_chunk - max_per_group[group_index_chunk])

            edge_weight = (exp_chunk / sum_per_group[group_index_chunk].clamp_min(denom_floor)).view(
                end - start, C, self.nhead
            )
            edge_weight = self.dropout(edge_weight)
            messages = v_src_chunk * edge_weight.unsqueeze(-1)
            agg.index_add_(0, edge_dst_chunk, messages)
        attn_out = self.out_proj(agg.view(B, T, C, self.d_model))
        return (1.0 - alpha) * residual_src + alpha * attn_out


class GraphAttentionBlock(nn.Module):
    """Graph attention block with residual connections and FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | Callable[[Tensor], Tensor] = "gelu",
        norm_first: bool = True,
        bias_free_ln: bool = False,
    ):
        super().__init__()
        self.norm_first = norm_first

        self.attn = GraphMultiheadAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model, bias=not bias_free_ln)
        self.norm2 = nn.LayerNorm(d_model, bias=not bias_free_ln)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            if activation == "relu":
                self.activation = nn.ReLU()
            elif activation == "gelu":
                self.activation = nn.GELU()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation

        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor, edge_index_batch: list[Tensor]) -> Tensor:
        if self.norm_first:
            x = self.dropout1(self.attn(self.norm1(src), edge_index_batch=edge_index_batch, residual_src=src))
            x = x + self._ff_block(self.norm2(x))
            return x

        x = self.norm1(self.dropout1(self.attn(src, edge_index_batch=edge_index_batch)))
        x = self.norm2(x + self._ff_block(x))
        return x

class GraphAttentionTransformer(nn.Module):
    """Graph-column transformer with sparse graph attention and intra-row attention."""

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
        num_output_cls: int | None = None,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.graph_blocks = nn.ModuleList(
            [
                GraphAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    bias_free_ln=bias_free_ln,
                )
                for _ in range(num_blocks)
            ]
        )
        self.col_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=nhead,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.col_attn_ln = nn.ModuleList([nn.LayerNorm(d_model, bias=not bias_free_ln) for _ in range(num_blocks)])

        if (num_output_cls is None) != (out_dim is None):
            raise ValueError("num_output_cls and out_dim must be provided together")
        if num_output_cls is not None and num_output_cls <= 0:
            raise ValueError("num_output_cls must be > 0")

        self.num_output_cls = num_output_cls
        self.out_proj = None
        if out_dim is not None and num_output_cls is not None:
            self.out_proj = nn.Linear(num_output_cls * d_model, out_dim)
            nn.init.xavier_uniform_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        self.recompute = recompute

    def forward(self, src: Tensor, edge_index_batch: list[Tensor]) -> Tensor:
        if src.ndim != 4:
            raise ValueError("GraphAttentionTransformer expects src with shape (B, T, C, D)")

        B, T, C, D = src.shape
        if self.num_output_cls is not None and C < self.num_output_cls:
            raise ValueError(
                f"GraphAttentionTransformer expects at least {self.num_output_cls} columns for output CLS, got {C}."
            )

        out = src
        for block_idx, block in enumerate(self.graph_blocks):
            if self.recompute:
                out = checkpoint(partial(block, edge_index_batch=edge_index_batch), out, use_reentrant=False)
            else:
                out = block(out, edge_index_batch=edge_index_batch)

            x_bt = out.reshape(B * T, C, D)
            attn_in = self.col_attn_ln[block_idx](x_bt)
            attn_out, _ = self.col_attn[block_idx](attn_in, attn_in, attn_in, need_weights=False)
            out = (x_bt + attn_out).reshape(B, T, C, D)

        if self.num_output_cls is None:
            return out

        cls_out = out[:, :, -self.num_output_cls :, :].reshape(B, T, self.num_output_cls * D)
        if self.out_proj is None:
            return cls_out
        return self.out_proj(cls_out)
