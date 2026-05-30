from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint


class GraphMultiheadAttention(nn.Module):
    """Sparse graph multi-head attention with sigmoid edge gating."""

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

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, src: Tensor, edge_index_batch: list[Tensor]) -> Tensor:
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

        B, T, _ = src.shape
        out_batch = []

        for b in range(B):
            x = src[b]
            edge_index = edge_index_batch[b].to(device=x.device, dtype=torch.long)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError("Each edge_index must have shape (2, E)")

            edge_src = edge_index[0]
            edge_dst = edge_index[1]

            if edge_src.numel() == 0:
                out_batch.append(self.out_proj(x))
                continue

            q = self.q_proj(x).view(T, self.nhead, self.head_dim)
            k = self.k_proj(x).view(T, self.nhead, self.head_dim)
            v = self.v_proj(x).view(T, self.nhead, self.head_dim)

            q_dst = q[edge_dst]
            k_src = k[edge_src]
            v_src = v[edge_src]

            attn_logits = (q_dst * k_src).sum(dim=-1) * self.scale
            edge_weight = torch.sigmoid(attn_logits)
            edge_weight = self.dropout(edge_weight)

            messages = v_src * edge_weight.unsqueeze(-1)

            agg = torch.zeros((T, self.nhead, self.head_dim), dtype=src.dtype, device=src.device)
            denom = torch.zeros((T, self.nhead), dtype=src.dtype, device=src.device)
            for h in range(self.nhead):
                agg[:, h, :].index_add_(0, edge_dst, messages[:, h, :])
                denom[:, h].index_add_(0, edge_dst, edge_weight[:, h])

            agg = agg / denom.clamp_min(1e-6).unsqueeze(-1)
            out = self.out_proj(agg.reshape(T, self.d_model))
            out_batch.append(out)

        return torch.stack(out_batch, dim=0)


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

        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, src: Tensor, edge_index_batch: list[Tensor]) -> Tensor:
        if self.norm_first:
            x = src + self.dropout1(self.attn(self.norm1(src), edge_index_batch=edge_index_batch))
            x = x + self._ff_block(self.norm2(x))
            return x

        x = self.norm1(src + self.dropout1(self.attn(src, edge_index_batch=edge_index_batch)))
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
