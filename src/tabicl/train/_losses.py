from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Compute supervised contrastive loss over a batch of embeddings.

    Parameters
    ----------
    embeddings : Tensor
        Tensor of shape (N, D).
    labels : Tensor
        Tensor of shape (N,) with integer class labels.
    temperature : float, default=0.1
        Temperature scaling in similarity space.

    Returns
    -------
    Tensor
        Scalar SupCon loss. Returns zero when no valid positive pairs exist.
    """

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must share the same leading dimension")

    num_samples = embeddings.shape[0]
    if num_samples <= 1:
        return embeddings.new_zeros(())

    z = F.normalize(embeddings.float(), p=2, dim=1)
    labels = labels.long().view(-1)

    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(num_samples, device=embeddings.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)
    valid_anchor_mask = positive_mask.any(dim=1)

    if not valid_anchor_mask.any():
        return embeddings.new_zeros(())

    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    pos_log_prob_sum = (log_prob * positive_mask).sum(dim=1)
    pos_count = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = pos_log_prob_sum / pos_count

    loss = -mean_log_prob_pos[valid_anchor_mask].mean()
    return loss.to(dtype=embeddings.dtype)


def entropy_regularizer(logits: Tensor) -> Tensor:
    """Compute mean predictive entropy over a logits tensor.

    The input is flattened to shape (N, C) and entropy is computed as
    ``-sum(p * log(p))`` with softmax probabilities ``p``.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")

    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    log_probs = F.log_softmax(flat_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy.to(dtype=logits.dtype)