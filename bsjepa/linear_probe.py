"""Periodic frozen-encoder linear probing for subject gender."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .data import SubjectSubset
from .evaluation import (
    LabeledGraphDataset,
    extract_graph_embeddings,
    normalize_subject_id,
)
from .model import BSJEPA


def _gender_class(value: Any) -> int | None:
    normalized = str(value).strip().upper()
    if normalized in {"F", "FEMALE"}:
        return 0
    if normalized in {"M", "MALE"}:
        return 1
    return None


def _shuffled(values: list[Any], generator: torch.Generator) -> list[Any]:
    if not values:
        return []
    return [values[index] for index in torch.randperm(len(values), generator=generator)]


def split_gender_probe_holdout(
    dataset: Dataset[Data], config: dict[str, Any]
) -> tuple[SubjectSubset, LabeledGraphDataset]:
    """Reserve a deterministic gender-stratified probe set from pretraining."""
    if str(config.get("task", "gender")).lower() != "gender":
        raise ValueError("Only linear_probe.task=gender is currently supported")
    subject_ids = getattr(dataset, "subject_ids", None)
    metadata_getter = getattr(dataset, "get_subject_metadata", None)
    if subject_ids is None or metadata_getter is None:
        raise TypeError("Gender probing requires subject IDs and per-subject metadata")
    if len(subject_ids) != len(dataset):
        raise ValueError("Dataset subject IDs must align with dataset indices")

    label_key = str(
        config.get("metadata_key", config.get("label_column", "gender"))
    )
    family_key = str(config.get("family_key", "Family_ID"))
    candidates: dict[int, list[tuple[int, str]]] = {0: [], 1: []}
    invalid_labels = 0
    family_values: list[str] = []
    for index, raw_subject_id in enumerate(subject_ids):
        metadata = metadata_getter(index)
        label = _gender_class(metadata.get(label_key))
        if label is None:
            invalid_labels += 1
            continue
        candidates[label].append((index, normalize_subject_id(raw_subject_id)))
        family = metadata.get(family_key)
        if family is not None and str(family).strip():
            family_values.append(str(family).strip())
    if invalid_labels:
        warnings.warn(
            f"Excluded {invalid_labels} subjects with missing/invalid {label_key!r} labels "
            "from gender-probe holdout selection",
            stacklevel=2,
        )
    if not family_values:
        warnings.warn(
            f"Family metadata key {family_key!r} is unavailable; gender-probe splits are "
            "subject-level and may leak information between related HCP participants",
            stacklevel=2,
        )
    else:
        warnings.warn(
            "Family identifiers were found, but this dataset adapter does not provide a "
            "validated family-level grouping; using subject-level stratification",
            stacklevel=2,
        )

    if len(candidates[0]) < 2 or len(candidates[1]) < 2:
        raise ValueError("At least two valid subjects per gender class are required")
    requested = int(config["heldout_size"])
    if requested < 4:
        raise ValueError("linear_probe.heldout_size must be at least 4")
    available = len(candidates[0]) + len(candidates[1])
    heldout_size = min(requested, available)
    if heldout_size < requested:
        warnings.warn(
            f"Requested {requested} probe subjects but only {heldout_size} have valid labels",
            stacklevel=2,
        )
    lower_female = max(2, heldout_size - len(candidates[1]))
    upper_female = min(len(candidates[0]), heldout_size - 2)
    if lower_female > upper_female:
        raise ValueError("Held-out size cannot preserve both classes in probe splits")
    proportional_female = round(heldout_size * len(candidates[0]) / available)
    female_count = min(max(proportional_female, lower_female), upper_female)
    male_count = heldout_size - female_count

    generator = torch.Generator().manual_seed(int(config.get("random_seed", 42)))
    selected = (
        _shuffled(candidates[0], generator)[:female_count]
        + _shuffled(candidates[1], generator)[:male_count]
    )
    selected = _shuffled(selected, generator)
    heldout_indices = {index for index, _ in selected}
    pretraining_indices = [
        index for index in range(len(dataset)) if index not in heldout_indices
    ]
    if not pretraining_indices:
        raise ValueError("Gender-probe holdout leaves no subjects for JEPA pretraining")
    labels_by_index = {
        index: label for label, values in candidates.items() for index, _ in values
    }
    heldout = LabeledGraphDataset(
        dataset,
        [index for index, _ in selected],
        [labels_by_index[index] for index, _ in selected],
        [subject_id for _, subject_id in selected],
        label_dtype=torch.long,
    )
    return SubjectSubset(dataset, pretraining_indices), heldout


def _stratified_probe_indices(
    labels: torch.Tensor, train_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < train_fraction < 1:
        raise ValueError("linear_probe.probe_train_fraction must be between 0 and 1")
    generator = torch.Generator().manual_seed(seed)
    training: list[int] = []
    validation: list[int] = []
    for label in (0, 1):
        indices = (labels == label).nonzero(as_tuple=True)[0].tolist()
        if len(indices) < 2:
            raise ValueError("Probe train and validation require both gender classes")
        indices = _shuffled(indices, generator)
        training_count = min(
            max(round(len(indices) * train_fraction), 1), len(indices) - 1
        )
        training.extend(indices[:training_count])
        validation.extend(indices[training_count:])
    training = _shuffled(training, generator)
    validation = _shuffled(validation, generator)
    return torch.tensor(training, dtype=torch.long), torch.tensor(validation, dtype=torch.long)


def _classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, *, prefix: str
) -> dict[str, float]:
    validation_loss = F.cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean()
    recalls = torch.stack(
        [
            (predictions[labels == label] == label).float().mean()
            for label in (0, 1)
        ]
    )
    return {
        f"{prefix}_val_accuracy": accuracy.item(),
        f"{prefix}_val_balanced_accuracy": recalls.mean().item(),
        f"{prefix}_val_loss": validation_loss.item(),
    }


def _fit_linear_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    *,
    prefix: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    train_x, train_y = features[train_indices], labels[train_indices]
    validation_x, validation_y = features[validation_indices], labels[validation_indices]

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        classifier = nn.Sequential(
            nn.LayerNorm(features.shape[1]), nn.Linear(features.shape[1], 2)
        )
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=lr, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    classifier.train()
    for _ in range(epochs):
        for indices in torch.randperm(len(train_x), generator=generator).split(
            batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(train_x[indices])
            loss = F.cross_entropy(logits, train_y[indices])
            loss.backward()
            optimizer.step()

    classifier.eval()
    with torch.no_grad():
        return _classification_metrics(
            classifier(validation_x), validation_y, prefix=prefix
        )


def _majority_class_baseline(
    labels: torch.Tensor, validation_indices: torch.Tensor
) -> dict[str, float]:
    validation_y = labels[validation_indices]
    counts = torch.bincount(validation_y, minlength=2)
    majority_class = int(counts.argmax())
    predictions = torch.full_like(validation_y, majority_class)
    accuracy = (predictions == validation_y).float().mean()
    recalls = torch.stack(
        [
            (predictions[validation_y == label] == label).float().mean()
            for label in (0, 1)
        ]
    )
    return {
        "gender_majority_val_accuracy": accuracy.item(),
        "gender_majority_val_balanced_accuracy": recalls.mean().item(),
    }


@torch.no_grad()
def _extract_adjacency_features(dataset: LabeledGraphDataset) -> torch.Tensor:
    features: list[torch.Tensor] = []
    expected_nodes: int | None = None
    upper_indices: tuple[torch.Tensor, torch.Tensor] | None = None
    for index in range(len(dataset)):
        graph, _ = dataset[index]
        node_count = int(graph.num_nodes)
        if expected_nodes is None:
            expected_nodes = node_count
            upper_indices = torch.triu_indices(node_count, node_count, offset=1)
        elif node_count != expected_nodes:
            raise ValueError("Raw adjacency baseline requires a fixed node count")
        if upper_indices is None:
            raise RuntimeError("Adjacency feature indices were not initialized")
        adjacency = torch.zeros(node_count, node_count, dtype=torch.float32)
        edge_values = (
            graph.edge_attr.detach().cpu().float().view(-1)
            if graph.edge_attr is not None
            else torch.ones(graph.edge_index.shape[1], dtype=torch.float32)
        )
        edge_index = graph.edge_index.detach().cpu()
        adjacency[edge_index[0], edge_index[1]] = edge_values
        features.append(adjacency[upper_indices[0], upper_indices[1]])
    if not features:
        raise ValueError("Raw adjacency baseline requires at least one graph")
    return torch.stack(features)


def evaluate_gender_probe(
    model: BSJEPA,
    dataset: LabeledGraphDataset,
    config: dict[str, Any],
    *,
    device: torch.device,
    random_model: BSJEPA | None = None,
) -> dict[str, float]:
    """Train fresh gender probes and simple baselines on the same held-out split."""
    batch_size = int(config.get("batch_size", 32))
    probe_epochs = int(config["probe_epochs"])
    probe_lr = float(config["probe_lr"])
    if batch_size < 1 or probe_epochs < 1 or probe_lr <= 0:
        raise ValueError("Probe batch size, epochs, and learning rate must be positive")
    embeddings, labels = extract_graph_embeddings(
        model, dataset, device=device, batch_size=batch_size
    )
    seed = int(config.get("random_seed", 42))
    train_indices, validation_indices = _stratified_probe_indices(
        labels, float(config.get("probe_train_fraction", 0.7)), seed
    )
    weight_decay = float(config.get("probe_weight_decay", 0.0))
    metrics = _fit_linear_classifier(
        embeddings,
        labels,
        train_indices,
        validation_indices,
        prefix="gender_probe",
        batch_size=batch_size,
        epochs=probe_epochs,
        lr=probe_lr,
        weight_decay=weight_decay,
        seed=seed,
    )

    if bool(config.get("compare_baselines", True)):
        metrics.update(_majority_class_baseline(labels, validation_indices))
        raw_features = _extract_adjacency_features(dataset)
        metrics.update(
            _fit_linear_classifier(
                raw_features,
                labels,
                train_indices,
                validation_indices,
                prefix="gender_raw_adjacency",
                batch_size=batch_size,
                epochs=probe_epochs,
                lr=probe_lr,
                weight_decay=weight_decay,
                seed=seed,
            )
        )
        if random_model is not None:
            random_model.to(device)
            random_embeddings, random_labels = extract_graph_embeddings(
                random_model, dataset, device=device, batch_size=batch_size
            )
            if not torch.equal(random_labels, labels):
                raise ValueError("Random encoder labels do not match probe labels")
            metrics.update(
                _fit_linear_classifier(
                    random_embeddings,
                    labels,
                    train_indices,
                    validation_indices,
                    prefix="gender_random_encoder",
                    batch_size=batch_size,
                    epochs=probe_epochs,
                    lr=probe_lr,
                    weight_decay=weight_decay,
                    seed=seed,
                )
            )
    return metrics
