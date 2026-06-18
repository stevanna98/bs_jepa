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
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch

from .model import BSJEPA


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
        labels: list[float],
        subject_ids: list[str],
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Data, torch.Tensor]:
        return self.dataset[self.indices[index]], self.labels[index]


def split_pmat_holdout(
    dataset: Dataset[Data], config: dict[str, Any]
) -> tuple[Subset[Data], LabeledGraphDataset]:
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
    return Subset(dataset, pretraining_indices), heldout


@torch.no_grad()
def _extract_embeddings(
    model: BSJEPA,
    dataset: LabeledGraphDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_was_training = model.training
    target_was_training = model.target_encoder.training
    model.target_encoder.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=list, drop_last=False
    )
    try:
        for examples in loader:
            graphs, batch_labels = zip(*examples, strict=True)
            batch = Batch.from_data_list(list(graphs)).to(device)
            node_embeddings = model.encode(batch)
            embeddings.extend(
                part.mean(0).cpu() for part in unbatch(node_embeddings, batch.batch)
            )
            labels.extend(label.detach().cpu() for label in batch_labels)
    finally:
        model.train(model_was_training)
        model.target_encoder.train(target_was_training)
    return torch.stack(embeddings), torch.stack(labels)


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
    embeddings, labels = _extract_embeddings(
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
