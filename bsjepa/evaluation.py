"""Lightweight frozen-encoder evaluation on held-out PMAT labels."""

from __future__ import annotations

import csv
import math
import re
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch

from .model import BSJEPA
from .data import SubjectSubset

RegionStageActivations = dict[str, dict[int, list[torch.Tensor]]]
SUBJECT_EMBEDDING_ERROR = "Subject embeddings must be a two-dimensional tensor"


def normalize_subject_id(value: Any) -> str:
    """Normalize dataset keys and CSV values without coercing leading zeroes."""
    text = str(value).strip()
    if isinstance(value, Path) or Path(text).suffix.lower() in {".pt", ".npz"}:
        text = Path(text).stem
    integer_float = re.fullmatch(r"([0-9]+)\.0+", text)
    return integer_float.group(1) if integer_float else text


class LabeledGraphDataset(Dataset[tuple[Data, torch.Tensor]]):
    """A labeled index view over the graphs reserved for downstream evaluation."""

    def __init__(
        self,
        dataset: Dataset[Data],
        indices: list[int],
        labels: list[float | int],
        subject_ids: list[str],
        *,
        label_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.labels = torch.tensor(labels, dtype=label_dtype)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Data, torch.Tensor]:
        return self.dataset[self.indices[index]], self.labels[index]


def split_pmat_holdout(
    dataset: Dataset[Data], config: dict[str, Any]
) -> tuple[SubjectSubset, LabeledGraphDataset]:
    """Deterministically reserve labeled subjects and remove them from pretraining."""
    raw_subject_ids = getattr(dataset, "subject_ids", None)
    if raw_subject_ids is None:
        raise TypeError("PMAT evaluation requires a dataset with subject_ids")
    subject_ids = [normalize_subject_id(value) for value in raw_subject_ids]
    if len(subject_ids) != len(dataset) or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Dataset subject IDs must be present and unique")

    csv_path = Path(config["pmat_csv"])
    subject_column = str(config.get("subject_column", "Subject"))
    label_column = str(config.get("label_column", "PMAT24_A_CR"))
    labels: dict[str, float | None] = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (subject_column, label_column):
            if column not in fields:
                raise KeyError(f"Column {column!r} not found in {csv_path}")
        for row in reader:
            subject_id = normalize_subject_id(row[subject_column])
            raw_label = row[label_column].strip()
            try:
                label = float(raw_label) if raw_label else None
            except ValueError:
                label = None
            labels[subject_id] = label if label is not None and math.isfinite(label) else None

    candidates: list[tuple[int, str, float]] = []
    missing_labels = 0
    unmatched = 0
    for index, subject_id in enumerate(subject_ids):
        if subject_id not in labels:
            unmatched += 1
        elif labels[subject_id] is None:
            missing_labels += 1
        else:
            candidates.append((index, subject_id, float(labels[subject_id])))
    if missing_labels:
        warnings.warn(
            f"Excluded {missing_labels} pretraining subjects with missing/invalid PMAT labels "
            "from held-out selection",
            stacklevel=2,
        )
    if unmatched:
        warnings.warn(
            f"No PMAT metadata match for {unmatched} pretraining subjects; they remain in "
            "the self-supervised training set",
            stacklevel=2,
        )

    requested = int(config["heldout_size"])
    if requested < 2:
        raise ValueError("evaluation.heldout_size must be at least 2")
    if len(candidates) < 2:
        raise ValueError("At least two matched subjects with valid PMAT labels are required")
    heldout_size = min(requested, len(candidates))
    if heldout_size < requested:
        warnings.warn(
            f"Requested {requested} held-out subjects but only {heldout_size} have valid labels",
            stacklevel=2,
        )
    generator = torch.Generator().manual_seed(int(config.get("random_seed", 42)))
    selected_positions = torch.randperm(len(candidates), generator=generator)[
        :heldout_size
    ].tolist()
    selected = [candidates[position] for position in selected_positions]
    heldout_indices = {index for index, _, _ in selected}
    pretraining_indices = [
        index for index in range(len(dataset)) if index not in heldout_indices
    ]
    if not pretraining_indices:
        raise ValueError("Held-out selection leaves no subjects for JEPA pretraining")
    heldout = LabeledGraphDataset(
        dataset,
        [index for index, _, _ in selected],
        [label for _, _, label in selected],
        [subject_id for _, subject_id, _ in selected],
    )
    return SubjectSubset(dataset, pretraining_indices), heldout


@torch.no_grad()
def extract_target_encoder_diagnostics(
    model: BSJEPA,
    dataset: Dataset[Data],
    *,
    device: torch.device,
    batch_size: int,
    collect_region_stages: bool = False,
) -> tuple[torch.Tensor, list[str], RegionStageActivations | None]:
    """Extract pooled embeddings and optional aligned EMA encoder stages in batches."""
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive")
    model_was_training = model.training
    target_was_training = model.target_encoder.training
    model.target_encoder.eval()
    embeddings: list[torch.Tensor] = []
    region_stages: RegionStageActivations | None = (
        {} if collect_region_stages else None
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=list, drop_last=False
    )
    try:
        for examples in loader:
            graphs = [
                example[0] if isinstance(example, tuple) else example
                for example in examples
            ]
            batch = Batch.from_data_list(list(graphs)).to(device)
            if collect_region_stages:
                diagnostic_forward = getattr(
                    model.target_encoder, "forward_with_diagnostics", None
                )
                if diagnostic_forward is None:
                    raise TypeError(
                        "Target encoder does not support region-stage diagnostics"
                    )
                node_embeddings, batch_stages = diagnostic_forward(batch)
            else:
                node_embeddings = model.encode(batch)
                batch_stages = None
            embedding_parts = unbatch(node_embeddings, batch.batch)
            embeddings.extend(part.mean(0).cpu() for part in embedding_parts)
            if region_stages is not None and batch_stages is not None:
                for stage, stage_values in batch_stages.items():
                    stage_regions = region_stages.setdefault(stage, {})
                    stage_parts = unbatch(stage_values, batch.batch)
                    for graph, part in zip(graphs, stage_parts, strict=True):
                        region_ids = getattr(graph, "region_ids", None)
                        if region_ids is None:
                            region_ids = torch.arange(graph.num_nodes)
                        region_ids = torch.as_tensor(region_ids).cpu()
                        part = part.detach().cpu()
                        if len(region_ids) != len(part):
                            raise ValueError(
                                "Atlas-region IDs must align with diagnostic activations"
                            )
                        for region_id, value in zip(region_ids, part, strict=True):
                            stage_regions.setdefault(int(region_id), []).append(value)
    finally:
        model.train(model_was_training)
        model.target_encoder.train(target_was_training)
    raw_subject_ids = getattr(dataset, "subject_ids", None)
    subject_ids = (
        [normalize_subject_id(value) for value in raw_subject_ids]
        if raw_subject_ids is not None
        else [str(index) for index in range(len(dataset))]
    )
    if len(subject_ids) != len(embeddings):
        raise ValueError("Dataset subject IDs must align with extracted embeddings")
    return torch.stack(embeddings), subject_ids, region_stages


@torch.no_grad()
def extract_subject_embeddings(
    model: BSJEPA,
    dataset: Dataset[Data],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str]]:
    """Extract one mean-pooled EMA target embedding per subject in batches."""
    embeddings, subject_ids, _ = extract_target_encoder_diagnostics(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        collect_region_stages=False,
    )
    return embeddings, subject_ids


def _subject_embeddings_as_float(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(SUBJECT_EMBEDDING_ERROR)
    return embeddings.float()


def _off_diagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(len(matrix), dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _distribution_metrics(
    values: torch.Tensor, prefix: str
) -> dict[str, float]:
    if values.numel() == 0:
        return {
            f"{prefix}_{name}": float("nan")
            for name in ("mean", "std", "min", "max")
        }
    return {
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_std": values.std(unbiased=False).item(),
        f"{prefix}_min": values.min().item(),
        f"{prefix}_max": values.max().item(),
    }


def subject_similarity_diagnostics(
    embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return pairwise cosine similarities and off-diagonal summary statistics."""
    values = _subject_embeddings_as_float(embeddings)
    normalized = F.normalize(values, p=2, dim=1)
    similarity = normalized @ normalized.T
    off_diagonal = _off_diagonal_values(similarity)
    metrics = _distribution_metrics(off_diagonal, "subject_cosine_similarity")
    return similarity, off_diagonal, metrics


def cohort_centered_cosine_diagnostics(
    embeddings: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute cosine similarity after removing the cohort-mean embedding."""
    values = _subject_embeddings_as_float(embeddings)
    if epsilon <= 0:
        raise ValueError("Centered-cosine epsilon must be positive")
    centered = values - values.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=1, eps=epsilon)
    similarity = normalized @ normalized.T
    off_diagonal = _off_diagonal_values(similarity)
    return (
        similarity,
        off_diagonal,
        _distribution_metrics(off_diagonal, "subject_centered_cosine"),
    )


def subject_variance_rank_diagnostics(
    embeddings: torch.Tensor,
    *,
    near_zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Measure feature variance and effective dimensionality across subjects."""
    values = _subject_embeddings_as_float(embeddings)
    if near_zero_threshold < 0:
        raise ValueError("Near-zero variance threshold must be non-negative")
    feature_variances = values.var(dim=0, unbiased=False)
    variance_metrics = {
        "subject_feature_variance_mean": feature_variances.mean().item(),
        "subject_feature_variance_median": feature_variances.quantile(0.5).item(),
        "subject_feature_variance_min": feature_variances.min().item(),
        "subject_feature_variance_max": feature_variances.max().item(),
        "subject_feature_near_zero_fraction": (
            feature_variances <= near_zero_threshold
        ).float().mean().item(),
    }

    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    total_energy = energy.sum()
    if total_energy <= 0:
        explained_variance = energy
        rank_metrics = {
            "subject_effective_rank": 0.0,
            "subject_matrix_rank": 0.0,
            "subject_largest_singular_energy_fraction": 0.0,
            "subject_components_90pct": 0.0,
        }
    else:
        explained_variance = energy / total_energy
        nonzero_probabilities = explained_variance[explained_variance > 0]
        entropy = -(
            nonzero_probabilities * nonzero_probabilities.log()
        ).sum()
        cumulative = explained_variance.cumsum(0)
        components_90 = int(
            torch.searchsorted(
                cumulative,
                cumulative.new_tensor(0.9),
            ).item()
        ) + 1
        rank_metrics = {
            "subject_effective_rank": entropy.exp().item(),
            "subject_matrix_rank": float(torch.linalg.matrix_rank(centered).item()),
            "subject_largest_singular_energy_fraction": explained_variance.max().item(),
            "subject_components_90pct": float(components_90),
        }
    return feature_variances, explained_variance, {**variance_metrics, **rank_metrics}


def standardized_euclidean_diagnostics(
    embeddings: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    near_zero_threshold: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute pairwise distance after cohort-wise feature standardization."""
    values = _subject_embeddings_as_float(embeddings)
    if epsilon <= 0:
        raise ValueError("Standardization epsilon must be positive")
    if near_zero_threshold < 0:
        raise ValueError("Near-zero variance threshold must be non-negative")
    mean = values.mean(dim=0, keepdim=True)
    variance = values.var(dim=0, unbiased=False)
    standard_deviation = variance.sqrt().clamp_min(epsilon)
    standardized = (values - mean) / standard_deviation
    standardized[:, variance <= near_zero_threshold] = 0
    distances = torch.cdist(standardized, standardized, p=2)
    off_diagonal = _off_diagonal_values(distances)
    return (
        distances,
        off_diagonal,
        _distribution_metrics(off_diagonal, "subject_standardized_distance"),
    )


def region_stage_cross_subject_diagnostics(
    stages: RegionStageActivations,
    *,
    near_zero_threshold: float = 1e-8,
    norm_epsilon: float = 1e-12,
) -> tuple[dict[str, float], dict[str, dict[int, float]]]:
    """Summarize same-region variation across subjects at each encoder stage."""
    if near_zero_threshold < 0:
        raise ValueError("Region-stage near-zero threshold must be non-negative")
    if norm_epsilon <= 0:
        raise ValueError("Region-stage norm epsilon must be positive")
    metrics: dict[str, float] = {}
    per_region_variances: dict[str, dict[int, float]] = {}
    for stage, regions in stages.items():
        region_variances: dict[int, float] = {}
        for region_id, subject_values in regions.items():
            if len(subject_values) < 2:
                continue
            aligned = torch.stack(subject_values).float()
            region_variances[region_id] = aligned.var(
                dim=0, unbiased=False
            ).mean().item()
        per_region_variances[stage] = region_variances
        prefix = f"region_{stage}_cross_subject_variance"
        if region_variances:
            values = torch.tensor(list(region_variances.values()))
            metrics.update(
                {
                    f"{prefix}_mean": values.mean().item(),
                    f"{prefix}_median": values.quantile(0.5).item(),
                    f"{prefix}_min": values.min().item(),
                    f"{prefix}_max": values.max().item(),
                    f"{prefix}_near_zero_fraction": (
                        values <= near_zero_threshold
                    ).float().mean().item(),
                }
            )
        else:
            metrics.update(
                {
                    f"{prefix}_{name}": float("nan")
                    for name in (
                        "mean",
                        "median",
                        "min",
                        "max",
                        "near_zero_fraction",
                    )
                }
            )

    projection = stages.get("projection", {})
    position = stages.get("position", {})
    post_position = stages.get("post_position", {})
    feature_norms: list[torch.Tensor] = []
    position_norms: list[torch.Tensor] = []
    post_position_norms: list[torch.Tensor] = []
    ratios: list[torch.Tensor] = []
    for region_id in projection.keys() & position.keys() & post_position.keys():
        projected_values = projection[region_id]
        position_values = position[region_id]
        post_values = post_position[region_id]
        for projected, positional, post in zip(
            projected_values, position_values, post_values, strict=True
        ):
            feature_norm = projected.float().norm()
            position_norm = positional.float().norm()
            feature_norms.append(feature_norm)
            position_norms.append(position_norm)
            post_position_norms.append(post.float().norm())
            ratios.append(position_norm / feature_norm.clamp_min(norm_epsilon))
    if feature_norms:
        feature_norm_tensor = torch.stack(feature_norms)
        position_norm_tensor = torch.stack(position_norms)
        post_norm_tensor = torch.stack(post_position_norms)
        ratio_tensor = torch.stack(ratios)
        metrics.update(
            {
                "region_projected_feature_norm_mean": feature_norm_tensor.mean().item(),
                "region_position_norm_mean": position_norm_tensor.mean().item(),
                "region_position_to_feature_norm_ratio": ratio_tensor.mean().item(),
                "region_position_to_feature_norm_ratio_median": ratio_tensor.quantile(
                    0.5
                ).item(),
                "region_post_position_norm_mean": post_norm_tensor.mean().item(),
            }
        )

    temporal_variance = metrics.get(
        "region_temporal_cross_subject_variance_mean", float("nan")
    )
    for stage in stages:
        stage_variance = metrics.get(
            f"region_{stage}_cross_subject_variance_mean", float("nan")
        )
        if math.isfinite(temporal_variance) and math.isfinite(stage_variance):
            denominator = max(abs(temporal_variance), norm_epsilon)
            metrics[f"region_{stage}_variance_retention_ratio"] = (
                stage_variance / denominator
            )
        else:
            metrics[f"region_{stage}_variance_retention_ratio"] = float("nan")
    return metrics, per_region_variances


def _mean_region_embeddings(
    stage_values: dict[int, list[torch.Tensor]],
    rsn_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    for region_id, values in sorted(stage_values.items()):
        if not 0 <= region_id < len(rsn_ids) or not values:
            continue
        embeddings.append(torch.stack(values).float().mean(0))
        labels.append(int(rsn_ids[region_id]))
    if not embeddings:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)


def _cosine_group_separation(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    if embeddings.shape[0] < 2:
        return {
            "within_rsn_cosine_mean": float("nan"),
            "between_rsn_cosine_mean": float("nan"),
            "within_minus_between_cosine": float("nan"),
        }
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(labels), dtype=torch.bool)
    same = (labels[:, None] == labels[None, :]) & off_diagonal
    different = (labels[:, None] != labels[None, :]) & off_diagonal
    within = similarity[same]
    between = similarity[different]
    within_mean = within.mean().item() if within.numel() else float("nan")
    between_mean = between.mean().item() if between.numel() else float("nan")
    return {
        "within_rsn_cosine_mean": within_mean,
        "between_rsn_cosine_mean": between_mean,
        "within_minus_between_cosine": (
            within_mean - between_mean
            if math.isfinite(within_mean) and math.isfinite(between_mean)
            else float("nan")
        ),
    }


def _nearest_rsn_centroid_metrics(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    classes = labels.unique(sorted=True)
    predictions: list[int] = []
    targets: list[int] = []
    for index, label in enumerate(labels):
        centroids: list[torch.Tensor] = []
        centroid_labels: list[int] = []
        for rsn_id in classes:
            mask = labels == rsn_id
            if int(rsn_id) == int(label):
                mask[index] = False
            if not bool(mask.any()):
                continue
            centroids.append(embeddings[mask].mean(0))
            centroid_labels.append(int(rsn_id))
        if len(centroids) < 2 or int(label) not in centroid_labels:
            continue
        similarities = (
            F.normalize(embeddings[index].unsqueeze(0), p=2, dim=1)
            @ F.normalize(torch.stack(centroids), p=2, dim=1).T
        ).squeeze(0)
        predictions.append(centroid_labels[int(similarities.argmax())])
        targets.append(int(label))
    if not targets:
        return {
            "rsn_nearest_centroid_accuracy": float("nan"),
            "rsn_nearest_centroid_balanced_accuracy": float("nan"),
            "rsn_nearest_centroid_regions": 0.0,
        }
    prediction_tensor = torch.tensor(predictions)
    target_tensor = torch.tensor(targets)
    accuracy = (prediction_tensor == target_tensor).float().mean().item()
    recalls = [
        (prediction_tensor[target_tensor == rsn_id] == rsn_id).float().mean()
        for rsn_id in target_tensor.unique(sorted=True)
    ]
    return {
        "rsn_nearest_centroid_accuracy": accuracy,
        "rsn_nearest_centroid_balanced_accuracy": torch.stack(recalls).mean().item(),
        "rsn_nearest_centroid_regions": float(len(targets)),
    }


def _rsn_centroid_separation(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    centroids = torch.stack(
        [embeddings[labels == label].mean(0) for label in labels.unique(sorted=True)]
    )
    if len(centroids) < 2:
        return {"rsn_centroid_between_cosine_mean": float("nan")}
    similarity = F.normalize(centroids, p=2, dim=1) @ F.normalize(centroids, p=2, dim=1).T
    return {
        "rsn_centroid_between_cosine_mean": _off_diagonal_values(similarity).mean().item()
    }


def region_network_structure_diagnostics(
    stages: RegionStageActivations,
    rsn_ids: torch.Tensor,
) -> dict[str, float]:
    """Measure whether region embeddings preserve RSN-distinguishable structure."""
    metrics: dict[str, float] = {}
    rsn_ids = rsn_ids.detach().cpu().long()
    for stage, stage_values in stages.items():
        embeddings, labels = _mean_region_embeddings(stage_values, rsn_ids)
        prefix = f"region_network_{stage}"
        if embeddings.numel() == 0:
            metrics.update(
                {
                    f"{prefix}_region_count": 0.0,
                    f"{prefix}_rsn_count": 0.0,
                    f"{prefix}_within_rsn_cosine_mean": float("nan"),
                    f"{prefix}_between_rsn_cosine_mean": float("nan"),
                    f"{prefix}_within_minus_between_cosine": float("nan"),
                    f"{prefix}_rsn_centroid_between_cosine_mean": float("nan"),
                    f"{prefix}_rsn_nearest_centroid_accuracy": float("nan"),
                    f"{prefix}_rsn_nearest_centroid_balanced_accuracy": float("nan"),
                    f"{prefix}_rsn_nearest_centroid_regions": 0.0,
                }
            )
            continue
        stage_metrics = {
            "region_count": float(len(labels)),
            "rsn_count": float(labels.unique().numel()),
            **_cosine_group_separation(embeddings, labels),
            **_rsn_centroid_separation(embeddings, labels),
            **_nearest_rsn_centroid_metrics(embeddings, labels),
        }
        metrics.update({f"{prefix}_{key}": value for key, value in stage_metrics.items()})
    return metrics


@torch.no_grad()
def extract_graph_embeddings(
    model: BSJEPA,
    dataset: LabeledGraphDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract graph embeddings and labels for downstream probes."""
    embeddings, _ = extract_subject_embeddings(
        model, dataset, device=device, batch_size=batch_size
    )
    return embeddings, dataset.labels.detach().cpu().clone()


def _regression_metrics(
    targets: torch.Tensor, predictions: torch.Tensor
) -> dict[str, float]:
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = errors.square().mean().sqrt().item()
    centered_targets = targets - targets.mean()
    centered_predictions = predictions - predictions.mean()
    target_ss = centered_targets.square().sum()
    r2 = (
        1 - errors.square().sum() / target_ss
        if target_ss > 0
        else targets.new_tensor(float("nan"))
    )
    denominator = centered_targets.norm() * centered_predictions.norm()
    pearson = (
        ((centered_targets * centered_predictions).sum() / denominator).clamp(-1, 1)
        if len(targets) > 1 and denominator > 0
        else targets.new_tensor(float("nan"))
    )
    return {
        "pmat_val_mae": mae,
        "pmat_val_rmse": rmse,
        "pmat_val_r2": r2.item(),
        "pmat_val_pearson": pearson.item(),
    }


def evaluate_pmat(
    model: BSJEPA,
    dataset: LabeledGraphDataset,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Fit a fresh linear probe on frozen target-encoder graph embeddings."""
    batch_size = int(config.get("batch_size", 32))
    regressor_epochs = int(config["regressor_epochs"])
    regressor_lr = float(config["regressor_lr"])
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    if regressor_epochs < 1:
        raise ValueError("evaluation.regressor_epochs must be positive")
    if regressor_lr <= 0:
        raise ValueError("evaluation.regressor_lr must be positive")
    embeddings, labels = extract_graph_embeddings(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
    )
    count = len(dataset)
    if count < 2:
        raise ValueError("PMAT evaluation requires at least two held-out subjects")
    validation_fraction = float(config.get("validation_fraction", 0.25))
    if not 0 < validation_fraction < 1:
        raise ValueError("evaluation.validation_fraction must be between 0 and 1")
    validation_count = min(max(round(count * validation_fraction), 1), count - 1)
    seed = int(config.get("random_seed", 42))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    train_x, train_y = embeddings[training_indices], labels[training_indices]
    val_x, val_y = embeddings[validation_indices], labels[validation_indices]

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        regressor = nn.Sequential(
            nn.LayerNorm(embeddings.shape[1]), nn.Linear(embeddings.shape[1], 1)
        )
    optimizer = torch.optim.Adam(regressor.parameters(), lr=regressor_lr)
    label_mean = train_y.mean()
    label_std = train_y.std(unbiased=False).clamp_min(1e-6)
    normalized_labels = (train_y - label_mean) / label_std
    regressor.train()
    for _ in range(regressor_epochs):
        for indices in torch.randperm(len(train_x), generator=generator).split(
            batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            predictions = regressor(train_x[indices]).squeeze(-1)
            loss = F.mse_loss(predictions, normalized_labels[indices])
            loss.backward()
            optimizer.step()
    regressor.eval()
    with torch.no_grad():
        predictions = regressor(val_x).squeeze(-1) * label_std + label_mean
    return _regression_metrics(val_y, predictions)
