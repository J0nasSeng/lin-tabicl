from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


class GraphMultiheadAttention(nn.Module):
    """Sparse graph multi-head attention with per-destination softmax weights."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

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

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, src: Tensor, edge_index_batch: list[Tensor], residual_src: Tensor | None = None) -> Tensor:
        """Apply sparse graph attention.

        Parameters
        ----------
        src : Tensor
            Input node features of shape (B, T, D).

        edge_index_batch : list[Tensor]
            List of length B. Each tensor has shape (2, E) where index 0 is source
            node and index 1 is destination node.

        Returns
        -------
        Tensor
            Updated node features of shape (B, T, D).
        """

        if src.ndim != 3:
            raise ValueError("src must have shape (B, T, D)")
        if len(edge_index_batch) != src.shape[0]:
            raise ValueError("edge_index_batch length must equal batch size")

        if residual_src is None:
            residual_src = src
        if residual_src.shape != src.shape:
            raise ValueError("residual_src must have shape (B, T, D)")

        B, T, _ = src.shape
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

        q = self.q_proj(src).view(B * T, self.nhead, self.head_dim)
        k = self.k_proj(src).view(B * T, self.nhead, self.head_dim)
        v = self.v_proj(src).view(B * T, self.nhead, self.head_dim)

        q_dst = q[global_edge_dst]
        k_src = k[global_edge_src]
        v_src = v[global_edge_src]

        attn_logits = (q_dst * k_src).sum(dim=-1) * self.scale

        E = global_edge_dst.numel()
        head_ids = torch.arange(self.nhead, device=src.device, dtype=torch.long).repeat(E)
        dst_rep = global_edge_dst.repeat_interleave(self.nhead)
        group_index = dst_rep * self.nhead + head_ids
        num_groups = B * T * self.nhead

        logits_flat = attn_logits.reshape(-1)
        max_per_group = torch.full((num_groups,), float("-inf"), dtype=src.dtype, device=src.device)
        max_per_group.scatter_reduce_(0, group_index, logits_flat, reduce="amax", include_self=True)

        exp_logits = torch.exp(logits_flat - max_per_group[group_index])
        sum_per_group = torch.zeros((num_groups,), dtype=src.dtype, device=src.device)
        sum_per_group.index_add_(0, group_index, exp_logits)
        edge_weight = (exp_logits / sum_per_group[group_index].clamp_min(1e-12)).view(E, self.nhead)
        edge_weight = self.dropout(edge_weight)

        messages = v_src * edge_weight.unsqueeze(-1)

        agg = torch.zeros((B * T, self.nhead, self.head_dim), dtype=src.dtype, device=src.device)
        agg.index_add_(0, global_edge_dst, messages)
        attn_out = self.out_proj(agg.view(B, T, self.d_model))
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
    """Stack of sparse graph attention blocks."""

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
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
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
        self.recompute = recompute

    def forward(self, src: Tensor, edge_index_batch: list[Tensor]) -> Tensor:
        out = src
        for block in self.blocks:
            if self.recompute:
                out = checkpoint(partial(block, edge_index_batch=edge_index_batch), out, use_reentrant=False)
            else:
                out = block(out, edge_index_batch=edge_index_batch)
        return out
