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

    # 1. Normalize embeddings
    z = F.normalize(embeddings.float(), p=2, dim=1)
    labels = labels.long().view(-1)

    # 2. Compute similarity matrix / temperature
    logits = torch.matmul(z, z.T) / temperature  # Shape: (N, N)

    # 3. Create masks
    self_mask = torch.eye(num_samples, device=embeddings.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)
    
    valid_anchor_mask = positive_mask.any(dim=1)
    if not valid_anchor_mask.any():
        return embeddings.new_zeros(())

    # 4. Numerically stable Log-Sum-Exp over NON-SELF entries
    # Mask out self-contrast (diagonal) by setting to -infinity before logsumexp
    logits_mask = (~self_mask).float()
    logits_for_denom = logits.masked_fill(self_mask, float('-inf'))
    
    log_denominator = torch.logsumexp(logits_for_denom, dim=1, keepdim=True) # Shape: (N, 1)

    # 5. Compute Log Probabilities
    # log( exp(a) / sum(exp(b)) ) = a - log_denom
    log_prob = logits - log_denominator

    # 6. Compute mean positive log-likelihood per valid anchor
    # Mask out non-positive entries
    pos_log_prob_sum = (log_prob * positive_mask).sum(dim=1)
    pos_count = positive_mask.sum(dim=1)

    # Avoid division by zero for anchors with 0 positives (handled by valid_anchor_mask)
    mean_log_prob_pos = pos_log_prob_sum[valid_anchor_mask] / pos_count[valid_anchor_mask]

    # 7. Final Loss
    loss = -mean_log_prob_pos.mean()
    return loss.to(dtype=embeddings.dtype)


def entropy_regularizer(values: Tensor, input_type: str = "logits") -> Tensor:
    """Compute mean predictive entropy from logits or log probabilities.

    The input is flattened to shape (N, C). ``input_type`` must be ``"logits"``
    or ``"log_probs"``.
    """

    if values.ndim < 2:
        raise ValueError("values must have at least 2 dimensions")
    if input_type not in ("logits", "log_probs"):
        raise ValueError("input_type must be 'logits' or 'log_probs'")

    flat_values = values.reshape(-1, values.shape[-1]).float()
    log_probs = F.log_softmax(flat_values, dim=-1) if input_type == "logits" else flat_values
    probs = torch.exp(log_probs)
    positive = probs > 0
    entropy_terms = torch.zeros_like(probs)
    entropy_terms[positive] = -probs[positive] * log_probs[positive]
    entropy = entropy_terms.sum(dim=-1).mean()
    return entropy.to(dtype=values.dtype)