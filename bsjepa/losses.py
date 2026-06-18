"""Prediction and anti-collapse losses used by BS-JEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def _mean_pairwise_cosine(embeddings: torch.Tensor) -> float:
    """Return the mean cosine similarity over distinct embedding pairs in O(ND)."""
    count = embeddings.shape[0]
    if count < 2:
        return float("nan")
    normalized = F.normalize(embeddings.detach(), dim=-1)
    similarity_sum = normalized.sum(0).square().sum() - count
    return (similarity_sum / (count * (count - 1))).item()


@torch.no_grad()
def representation_diagnostics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, float]:
    """Compute inexpensive scalar diagnostics for representation collapse."""
    metrics: dict[str, float] = {}
    for name, embeddings in (
        ("prediction", predictions),
        ("target", targets),
        ("context", context),
    ):
        norms = embeddings.norm(dim=-1)
        metrics[f"{name}_embedding_std"] = embeddings.std(
            0, unbiased=False
        ).mean().item()
        metrics[f"{name}_pairwise_cosine"] = _mean_pairwise_cosine(embeddings)
        metrics[f"{name}_norm_mean"] = norms.mean().item()
        metrics[f"{name}_norm_std"] = norms.std(unbiased=False).item()
    return metrics


@torch.no_grad()
def per_rsn_prediction_losses(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    row_group_ids: torch.Tensor,
    group_rsn_ids: torch.Tensor,
) -> dict[int, tuple[float, int]]:
    """Return cosine prediction-loss sums and row counts for each target RSN."""
    row_rsn_ids = group_rsn_ids[row_group_ids]
    row_losses = 2 - 2 * (
        F.normalize(predictions, dim=-1) * F.normalize(targets, dim=-1)
    ).sum(-1)
    result: dict[int, tuple[float, int]] = {}
    for rsn_id in row_rsn_ids.unique():
        selected = row_losses[row_rsn_ids == rsn_id]
        result[int(rsn_id.item())] = (selected.sum().item(), selected.numel())
    return result


def jepa_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    context: torch.Tensor,
    *,
    prediction_variance_weight: float,
    context_variance_weight: float,
    covariance_weight: float,
    target_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine prediction loss with VICReg-style anti-collapse terms."""
    similarity = 2 - 2 * (
        F.normalize(predictions, dim=-1) * F.normalize(targets, dim=-1)
    ).sum(-1).mean()
    prediction_variance = F.relu(
        target_std - predictions.std(0, unbiased=False)
    ).mean()
    context_variance = F.relu(target_std - context.std(0, unbiased=False)).mean()
    centered = context - context.mean(0)
    covariance = centered.T @ centered / max(context.shape[0] - 1, 1)
    covariance_penalty = _off_diagonal(covariance).pow(2).sum() / context.shape[1]
    total = (
        similarity
        + prediction_variance_weight * prediction_variance
        + context_variance_weight * context_variance
        + covariance_weight * covariance_penalty
    )
    metrics = {
        "similarity": similarity.item(),
        "prediction_variance": prediction_variance.item(),
        "context_variance": context_variance.item(),
        "context_covariance": covariance_penalty.item(),
        "target_std": targets.std(0, unbiased=False).mean().item(),
    }
    return total, metrics


def rsn_diversity_loss(
    predictions: torch.Tensor,
    row_group_ids: torch.Tensor,
    group_rsn_ids: torch.Tensor,
) -> torch.Tensor:
    """Penalize aligned mean prediction directions across RSNs in a batch."""
    pooled = torch.stack(
        [predictions[row_group_ids == group].mean(0) for group in range(len(group_rsn_ids))]
    )
    rsn_means = torch.stack(
        [pooled[group_rsn_ids == rsn].mean(0) for rsn in group_rsn_ids.unique()]
    )
    if rsn_means.shape[0] < 2:
        return predictions.new_zeros(())
    similarities = F.normalize(rsn_means, dim=-1) @ F.normalize(rsn_means, dim=-1).T
    return _off_diagonal(similarities).pow(2).mean()
