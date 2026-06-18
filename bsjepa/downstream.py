"""End-to-end BS-JEPA training for supervised gender classification."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from .evaluation import LabeledGraphDataset, normalize_subject_id
from .linear_probe import _gender_class
from .losses import jepa_loss
from .masking import SubnetworkMaskCollator
from .model import BSJEPA
from .training import build_optimizer, update_target_encoder

Pooling = Literal["mean", "mean_std"]


class GenderClassifier(nn.Module):
    """Layer-normalized linear classifier over pooled graph embeddings."""

    def __init__(self, embed_dim: int, pooling: Pooling = "mean") -> None:
        super().__init__()
        if pooling not in ("mean", "mean_std"):
            raise ValueError("downstream.pooling must be 'mean' or 'mean_std'")
        self.pooling = pooling
        graph_dim = embed_dim * (2 if pooling == "mean_std" else 1)
        self.head = nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, 2))

    def forward(
        self, node_embeddings: torch.Tensor, batch_ids: torch.Tensor
    ) -> torch.Tensor:
        means = global_mean_pool(node_embeddings, batch_ids)
        if self.pooling == "mean":
            graph_embeddings = means
        else:
            mean_squares = global_mean_pool(node_embeddings.square(), batch_ids)
            standard_deviations = (mean_squares - means.square()).clamp_min(0).sqrt()
            graph_embeddings = torch.cat([means, standard_deviations], dim=-1)
        return self.head(graph_embeddings)


def _load_csv_metadata(config: dict[str, Any]) -> dict[str, dict[str, str]] | None:
    raw_path = config.get("metadata_csv")
    if not raw_path:
        return None
    path = Path(raw_path)
    subject_column = str(config.get("subject_column", "Subject"))
    label_column = str(config.get("label_column", "Gender"))
    family_column = str(config.get("family_column", "Family_ID"))
    metadata: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in (subject_column, label_column):
            if column not in fields:
                raise KeyError(f"Column {column!r} not found in {path}")
        for row in reader:
            subject_id = normalize_subject_id(row[subject_column])
            metadata[subject_id] = {
                "label": row[label_column],
                "family": row.get(family_column, ""),
            }
    return metadata


def split_gender_dataset(
    dataset: Dataset[Data], config: dict[str, Any]
) -> tuple[LabeledGraphDataset, LabeledGraphDataset]:
    """Create deterministic gender-stratified train and validation datasets."""
    if str(config.get("task", "gender")).lower() != "gender":
        raise ValueError("Only downstream.task=gender is currently supported")
    subject_ids = getattr(dataset, "subject_ids", None)
    metadata_getter = getattr(dataset, "get_subject_metadata", None)
    if subject_ids is None or metadata_getter is None:
        raise TypeError("Gender training requires subject IDs and subject metadata")
    if len(subject_ids) != len(dataset):
        raise ValueError("Dataset subject IDs must align with dataset indices")

    csv_metadata = _load_csv_metadata(config)
    metadata_key = str(config.get("metadata_key", config.get("label_column", "gender")))
    family_key = str(config.get("family_key", "Family_ID"))
    candidates: dict[int, list[tuple[int, str]]] = {0: [], 1: []}
    skipped = 0
    family_found = False
    for index, raw_subject_id in enumerate(subject_ids):
        subject_id = normalize_subject_id(raw_subject_id)
        if csv_metadata is None:
            metadata = metadata_getter(index)
            raw_label = metadata.get(metadata_key)
            family = metadata.get(family_key)
        else:
            matched = csv_metadata.get(subject_id)
            raw_label = None if matched is None else matched["label"]
            family = None if matched is None else matched["family"]
        label = _gender_class(raw_label)
        if label is None:
            skipped += 1
            continue
        candidates[label].append((index, subject_id))
        family_found |= family is not None and bool(str(family).strip())
    if skipped:
        warnings.warn(
            f"Excluded {skipped} subjects without a matched valid gender label",
            stacklevel=2,
        )
    family_message = (
        "Family identifiers were found, but downstream splitting is subject-level; "
        "related HCP participants may occur in both splits"
        if family_found
        else "Family identifiers were not found; downstream splitting is subject-level "
        "and may leak information between related HCP participants"
    )
    warnings.warn(family_message, stacklevel=2)
    if len(candidates[0]) < 2 or len(candidates[1]) < 2:
        raise ValueError("Gender train/validation splitting requires two subjects per class")

    train_fraction = float(config.get("train_fraction", 0.8))
    if not 0 < train_fraction < 1:
        raise ValueError("downstream.train_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(int(config.get("random_seed", 42)))
    training: list[tuple[int, str, int]] = []
    validation: list[tuple[int, str, int]] = []
    for label in (0, 1):
        values = candidates[label]
        order = torch.randperm(len(values), generator=generator).tolist()
        values = [values[position] for position in order]
        train_count = min(max(round(len(values) * train_fraction), 1), len(values) - 1)
        training.extend((index, subject_id, label) for index, subject_id in values[:train_count])
        validation.extend(
            (index, subject_id, label) for index, subject_id in values[train_count:]
        )
    training.sort(key=lambda item: item[0])
    validation.sort(key=lambda item: item[0])

    def labeled(values: list[tuple[int, str, int]]) -> LabeledGraphDataset:
        return LabeledGraphDataset(
            dataset,
            [index for index, _, _ in values],
            [label for _, _, label in values],
            [subject_id for _, subject_id, _ in values],
            label_dtype=torch.long,
        )

    return labeled(training), labeled(validation)


def _balanced_accuracy(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    recalls = [
        (predictions[targets == label] == label).float().mean()
        for label in (0, 1)
    ]
    return torch.stack(recalls).mean().item()


@torch.no_grad()
def _validate(
    model: BSJEPA,
    classifier: GenderClassifier,
    loader: DataLoader[list],
    device: torch.device,
) -> dict[str, float]:
    model_was_training = model.training
    classifier_was_training = classifier.training
    model.eval()
    classifier.eval()
    loss_sum = 0.0
    targets_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    for examples in loader:
        graphs, labels = zip(*examples, strict=True)
        batch = Batch.from_data_list(list(graphs)).to(device)
        targets = torch.stack(labels).to(device)
        logits = classifier(model.context_encoder(batch), batch.batch)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        targets_all.append(targets.cpu())
        predictions_all.append(logits.argmax(-1).cpu())
    model.train(model_was_training)
    classifier.train(classifier_was_training)
    targets = torch.cat(targets_all)
    predictions = torch.cat(predictions_all)
    return {
        "val_ce_loss": loss_sum / len(targets),
        "val_accuracy": (predictions == targets).float().mean().item(),
        "val_balanced_accuracy": _balanced_accuracy(targets, predictions),
    }


def train_supervised_gender(
    model: BSJEPA,
    classifier: GenderClassifier,
    training_dataset: LabeledGraphDataset,
    validation_dataset: LabeledGraphDataset,
    mask_collator: SubnetworkMaskCollator,
    config: dict[str, Any],
    *,
    full_config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
) -> list[dict[str, float]]:
    """Jointly optimize JEPA L2 and supervised gender classification losses."""
    epochs = int(config["epochs"])
    batch_size = int(config.get("batch_size", 16))
    if epochs < 1 or batch_size < 1:
        raise ValueError("downstream epochs and batch_size must be positive")
    train_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=list,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=list,
        drop_last=False,
    )
    trainable = nn.ModuleDict({"backbone": model, "classifier": classifier})
    optimizer = build_optimizer(
        trainable,
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    jepa_weight = float(config.get("jepa_weight", 1.0))
    classification_weight = float(config.get("classification_weight", 1.0))
    if float(config["lr"]) <= 0:
        raise ValueError("downstream.lr must be positive")
    if jepa_weight < 0 or classification_weight < 0:
        raise ValueError("Downstream loss weights must be non-negative")
    output_path = Path(output_dir)
    checkpoint_path = output_path / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_frequency = int(config.get("checkpoint_frequency", 1))
    history: list[dict[str, float]] = []
    global_step = 0
    total_steps = max(epochs * len(train_loader), 1)
    model.to(device)
    classifier.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        classifier.train()
        totals = {"total_loss": 0.0, "jepa_l2_loss": 0.0, "classification_ce_loss": 0.0}
        example_count = 0
        train_targets: list[torch.Tensor] = []
        train_predictions: list[torch.Tensor] = []
        for examples in train_loader:
            graphs, labels = zip(*examples, strict=True)
            batch, masks = mask_collator(list(graphs))
            batch, masks = batch.to(device), masks.to(device)
            targets = torch.stack(labels).to(device)
            optimizer.zero_grad(set_to_none=True)

            predictions, target_embeddings, context_embeddings = model(batch, masks)[:3]
            jepa_l2, _ = jepa_loss(
                predictions,
                target_embeddings,
                context_embeddings,
                prediction_loss="l2",
                prediction_variance_weight=0.0,
                context_variance_weight=0.0,
                covariance_weight=0.0,
                target_std=0.0,
            )
            # The EMA target encoder is intentionally a teacher. Full-graph context
            # embeddings provide the trainable supervised backbone, so CE gradients
            # update both the context encoder and classifier rather than a frozen copy.
            node_embeddings = model.context_encoder(batch)
            logits = classifier(node_embeddings, batch.batch)
            classification_ce = F.cross_entropy(logits, targets)
            total_loss = jepa_weight * jepa_l2 + classification_weight * classification_ce
            total_loss.backward()
            clip_grad = config.get("clip_grad")
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(trainable.parameters(), float(clip_grad))
            optimizer.step()

            global_step += 1
            progress = global_step / total_steps
            ema_start = float(config.get("ema_start", 0.996))
            ema_end = float(config.get("ema_end", 1.0))
            momentum = ema_start + (ema_end - ema_start) * min(progress, 1.0)
            update_target_encoder(model, momentum)

            count = len(targets)
            example_count += count
            totals["total_loss"] += total_loss.item() * count
            totals["jepa_l2_loss"] += jepa_l2.item() * count
            totals["classification_ce_loss"] += classification_ce.item() * count
            train_targets.append(targets.detach().cpu())
            train_predictions.append(logits.detach().argmax(-1).cpu())

        all_targets = torch.cat(train_targets)
        all_predictions = torch.cat(train_predictions)
        epoch_metrics = {
            "epoch": float(epoch),
            **{key: value / example_count for key, value in totals.items()},
            "train_accuracy": (all_predictions == all_targets).float().mean().item(),
            "train_balanced_accuracy": _balanced_accuracy(all_targets, all_predictions),
            **_validate(model, classifier, validation_loader, device),
            "ema_momentum": momentum,
        }
        history.append(epoch_metrics)
        print(
            " ".join(f"{key}={value:.5g}" for key, value in epoch_metrics.items()),
            flush=True,
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": epoch_metrics,
            "history": history,
            "config": full_config,
        }
        if checkpoint_frequency > 0 and (
            epoch % checkpoint_frequency == 0 or epoch == epochs
        ):
            torch.save(checkpoint, checkpoint_path / f"downstream_checkpoint_{epoch:04d}.pt")
        if bool(config.get("save_plots", True)):
            from .plotting import save_downstream_plots

            save_downstream_plots(history, output_path / "plots")
        with (output_path / "downstream_history.json").open("w") as handle:
            json.dump(history, handle, indent=2)

    torch.save(checkpoint, output_path / "downstream_final.pt")
    return history
