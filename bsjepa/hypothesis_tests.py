"""Neurobiological hypothesis-test analyses for trained BS-JEPA checkpoints."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "bsjepa_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from .data import (
    Atlas,
    BrainGraphDataset,
    SyntheticBrainDataset,
    load_atlas,
    synthetic_atlas,
)
from .evaluation import (
    LabeledGraphDataset,
    cohort_centered_cosine_diagnostics,
    evaluate_pmat,
    extract_target_encoder_diagnostics,
    region_stage_cross_subject_diagnostics,
    split_pmat_holdout,
    standardized_euclidean_diagnostics,
    subject_similarity_diagnostics,
    subject_variance_rank_diagnostics,
)
from .linear_probe import evaluate_gender_probe, split_gender_probe_holdout
from .model import BSJEPA, build_bsjepa


@dataclass
class AnalysisPaths:
    root: Path
    metrics: Path
    plots: Path
    tables: Path
    embeddings: Path
    reports: Path


@dataclass
class AnalysisContext:
    config: dict[str, Any]
    atlas: Atlas
    dataset: Any
    training_dataset: Any
    evaluation_dataset: LabeledGraphDataset | None
    linear_probe_dataset: LabeledGraphDataset | None
    model_config: dict[str, Any]
    input_feature_dim: int


@dataclass
class AnalysisState:
    paths: AnalysisPaths
    generated_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    hypothesis_status: dict[str, str] = field(default_factory=dict)
    key_results: dict[str, Any] = field(default_factory=dict)

    def add_file(self, path: Path) -> Path:
        if path not in self.generated_files:
            self.generated_files.append(path)
        return path

    def add_note(self, message: str) -> None:
        self.notes.append(message)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply the same SECTION.KEY=VALUE override style as pretrain."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    for override in overrides or []:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"Invalid override: {override!r}")
        if "." not in key:
            if key not in config:
                raise KeyError(f"Unknown config key: {key}")
            config[key] = yaml.safe_load(raw_value)
            continue
        section, field = key.split(".", 1)
        if section not in config or field not in config[section]:
            raise KeyError(f"Unknown config key: {key}")
        config[section][field] = yaml.safe_load(raw_value)
    return config


def prepare_output_dirs(output_dir: str | Path) -> AnalysisPaths:
    """Create the requested results tree without clobbering an existing run."""
    base = Path(output_dir)
    subdirs = ("metrics", "plots", "tables", "embeddings", "reports")
    if base.exists() and any(
        (base / name).is_dir() and any((base / name).iterdir()) for name in subdirs
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = base / f"run_{timestamp}"
    paths = AnalysisPaths(
        root=base,
        metrics=base / "metrics",
        plots=base / "plots",
        tables=base / "tables",
        embeddings=base / "embeddings",
        reports=base / "reports",
    )
    for path in (paths.metrics, paths.plots, paths.tables, paths.embeddings, paths.reports):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> Path:
    with path.open("w") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in fields})
    return path


def _save_figure(path: Path, *, dpi: int = 160, save_pdf: bool = False) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return path


def load_checkpoint_and_history(checkpoint_path: str | Path) -> tuple[dict[str, Any], list[dict[str, float]], Path]:
    """Load either a training checkpoint file or a final-artifact directory/file."""
    path = Path(checkpoint_path)
    if path.is_dir():
        candidates = sorted(path.glob("*_final.pt")) or sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt checkpoint found in artifact directory: {path}")
        path = candidates[0]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history")
    if history is None:
        history_path = path.with_name("training_history.json")
        if history_path.is_file():
            with history_path.open() as handle:
                history = json.load(handle)
        else:
            history = []
    return checkpoint, list(history), path


def build_analysis_context(config: dict[str, Any], checkpoint: dict[str, Any] | None = None) -> AnalysisContext:
    """Rebuild atlas, datasets, and effective model config using pretraining rules."""
    data_config = config["data"]
    seed = int(config["seed"])
    if data_config["source"] == "synthetic":
        atlas = synthetic_atlas(int(data_config["num_regions"]), int(data_config["num_rsns"]))
        dataset = SyntheticBrainDataset(
            atlas,
            int(data_config["num_subjects"]),
            int(data_config["feature_dim"]),
            top_k=int(data_config["top_k"]),
            seed=seed,
        )
    else:
        atlas = load_atlas(data_config["atlas_csv"])
        dataset = BrainGraphDataset(
            data_config["source"],
            atlas,
            node_features=data_config["node_features"],
            bold_key=data_config["bold_key"],
            fc_key=data_config["fc_key"],
            transpose_bold=bool(data_config["transpose_bold"]),
            bold_window_size=data_config.get("bold_window_size"),
            bold_window_start=int(data_config.get("bold_window_start", 0)),
            graph_strategy=data_config["graph_strategy"],
            top_k=int(data_config["top_k"]),
            threshold=float(data_config["threshold"]),
        )

    training_dataset = dataset
    evaluation_dataset = None
    evaluation_config = config.get("evaluation", {})
    if bool(evaluation_config.get("enabled", False)):
        training_dataset, evaluation_dataset = split_pmat_holdout(dataset, evaluation_config)

    linear_probe_dataset = None
    linear_probe_config = config.get("linear_probe", {})
    if bool(linear_probe_config.get("enabled", False)):
        training_dataset, linear_probe_dataset = split_gender_probe_holdout(
            training_dataset, linear_probe_config
        )

    model_config = dict(config["model"])
    if checkpoint is not None:
        model_config = dict(
            checkpoint.get("model_config")
            or checkpoint.get("reconstruction", {}).get("model_config")
            or model_config
        )
    if data_config["source"] == "synthetic":
        model_config["feature_mode"] = "passthrough"

    sample = training_dataset[0]
    graph = sample[0] if isinstance(sample, tuple) else sample
    input_feature_dim = int(
        checkpoint.get("input_feature_dim", graph.x.shape[1]) if checkpoint is not None else graph.x.shape[1]
    )
    return AnalysisContext(
        config=config,
        atlas=atlas,
        dataset=dataset,
        training_dataset=training_dataset,
        evaluation_dataset=evaluation_dataset,
        linear_probe_dataset=linear_probe_dataset,
        model_config=model_config,
        input_feature_dim=input_feature_dim,
    )


def build_model_from_context(context: AnalysisContext, checkpoint: dict[str, Any]) -> BSJEPA:
    model = build_bsjepa(
        in_channels=context.input_feature_dim,
        num_regions=int(checkpoint.get("num_regions", context.atlas.num_regions)),
        **context.model_config,
    )
    model.load_state_dict(checkpoint["model"])
    return model


def summarize_history_metrics(history: list[dict[str, float]]) -> list[dict[str, float | str]]:
    """Summarize scalar history metrics without assuming whether high or low is better."""
    if not history:
        return []
    keys = sorted({key for row in history for key in row if key != "epoch"})
    rows: list[dict[str, float | str]] = []
    for key in keys:
        values = [(float(row["epoch"]), float(row[key])) for row in history if key in row and row[key] is not None]
        values = [(epoch, value) for epoch, value in values if math.isfinite(value)]
        if not values:
            continue
        initial_epoch, initial = values[0]
        final_epoch, final = values[-1]
        best_epoch, best = min(values, key=lambda item: item[1])
        absolute_change = final - initial
        percent_change = absolute_change / abs(initial) * 100 if initial else float("nan")
        rows.append(
            {
                "metric": key,
                "initial_epoch": initial_epoch,
                "initial": initial,
                "final_epoch": final_epoch,
                "final": final,
                "min_epoch": best_epoch,
                "min": best,
                "max": max(value for _, value in values),
                "absolute_change": absolute_change,
                "percent_change": percent_change,
            }
        )
    return rows


def extract_rsn_loss_rows(
    history: list[dict[str, float]], rsn_names: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in history:
        epoch = row.get("epoch")
        for key, value in row.items():
            if not key.startswith("rsn_loss_") or value is None:
                continue
            rsn_id = int(key.rsplit("_", 1)[1])
            rows.append(
                {
                    "epoch": epoch,
                    "rsn_id": rsn_id,
                    "rsn_name": rsn_names[rsn_id] if rsn_id < len(rsn_names) else f"rsn_{rsn_id}",
                    "loss": float(value),
                }
            )
    return rows


def summarize_rsn_losses(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    by_rsn: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rsn.setdefault(int(row["rsn_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    final_losses: list[float] = []
    for rsn_id, values in sorted(by_rsn.items()):
        values = sorted(values, key=lambda item: float(item["epoch"]))
        losses = [float(item["loss"]) for item in values]
        initial = losses[0]
        final = losses[-1]
        final_losses.append(final)
        slope = (final - initial) / max(float(values[-1]["epoch"]) - float(values[0]["epoch"]), 1.0)
        summaries.append(
            {
                "rsn_id": rsn_id,
                "rsn_name": values[-1]["rsn_name"],
                "epochs_observed": len(values),
                "mean_loss": float(np.mean(losses)),
                "initial_loss": initial,
                "final_loss": final,
                "best_loss": min(losses),
                "absolute_improvement": initial - final,
                "percent_improvement": (initial - final) / abs(initial) * 100 if initial else float("nan"),
                "slope_per_epoch": slope,
            }
        )
    stats = {}
    if final_losses:
        mean = float(np.mean(final_losses))
        stats = {
            "final_loss_mean": mean,
            "final_loss_std": float(np.std(final_losses)),
            "final_loss_min": float(np.min(final_losses)),
            "final_loss_max": float(np.max(final_losses)),
            "final_loss_range": float(np.max(final_losses) - np.min(final_losses)),
            "final_loss_coefficient_of_variation": float(np.std(final_losses) / abs(mean)) if mean else float("nan"),
        }
    return summaries, stats


def anova_variance_ratio(groups: dict[str, list[float]]) -> dict[str, float]:
    """Return a dependency-free one-way ANOVA-style F ratio without a p-value."""
    clean = {name: [value for value in values if math.isfinite(value)] for name, values in groups.items()}
    clean = {name: values for name, values in clean.items() if values}
    all_values = [value for values in clean.values() for value in values]
    if len(clean) < 2 or len(all_values) <= len(clean):
        return {"anova_f_ratio": float("nan"), "between_group_df": 0.0, "within_group_df": 0.0}
    grand_mean = float(np.mean(all_values))
    between = sum(len(values) * (float(np.mean(values)) - grand_mean) ** 2 for values in clean.values())
    within = sum(sum((value - float(np.mean(values))) ** 2 for value in values) for values in clean.values())
    between_df = len(clean) - 1
    within_df = len(all_values) - len(clean)
    return {
        "anova_f_ratio": (between / between_df) / (within / within_df) if within > 0 else float("inf"),
        "between_group_df": float(between_df),
        "within_group_df": float(within_df),
    }


def pca_projection(embeddings: torch.Tensor) -> torch.Tensor:
    values = embeddings.float()
    centered = values - values.mean(dim=0, keepdim=True)
    if min(centered.shape) == 0:
        return torch.zeros(len(values), 2)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[: min(2, vh.shape[0])].T
    projected = centered @ components
    if projected.shape[1] == 1:
        projected = torch.cat([projected, projected.new_zeros(len(projected), 1)], dim=1)
    return projected[:, :2]


def _history_values(history: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
    rows = [(float(row["epoch"]), float(row[key])) for row in history if key in row and row[key] is not None]
    rows = [(epoch, value) for epoch, value in rows if math.isfinite(value)]
    return [epoch for epoch, _ in rows], [value for _, value in rows]


def plot_metric_lines(
    history: list[dict[str, float]],
    keys: list[str],
    labels: list[str],
    path: Path,
    *,
    title: str,
    ylabel: str,
    save_pdf: bool,
) -> Path | None:
    available = [(key, label) for key, label in zip(keys, labels, strict=True) if any(key in row for row in history)]
    if not available:
        return None
    plt.figure(figsize=(7, 4))
    for key, label in available:
        epochs, values = _history_values(history, key)
        if values:
            plt.plot(epochs, values, marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    if len(available) > 1:
        plt.legend()
    return _save_figure(path, save_pdf=save_pdf)


def plot_rsn_losses(rows: list[dict[str, Any]], plots_dir: Path, *, save_pdf: bool) -> list[Path]:
    if not rows:
        return []
    paths: list[Path] = []
    by_rsn: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_rsn.setdefault(int(row["rsn_id"]), []).append(row)
    plt.figure(figsize=(8, 5))
    for rsn_id, values in sorted(by_rsn.items()):
        values = sorted(values, key=lambda item: float(item["epoch"]))
        plt.plot(
            [float(item["epoch"]) for item in values],
            [float(item["loss"]) for item in values],
            marker="o",
            label=str(values[-1]["rsn_name"]),
        )
    plt.xlabel("Epoch")
    plt.ylabel("Cosine prediction loss")
    plt.title("Per-RSN Prediction Loss Over Training")
    plt.grid(alpha=0.3)
    plt.legend(fontsize="small", ncol=2)
    paths.append(_save_figure(plots_dir / "rsn_loss_over_time.png", save_pdf=save_pdf))

    latest = [max(values, key=lambda item: float(item["epoch"])) for values in by_rsn.values()]
    latest = sorted(latest, key=lambda item: float(item["loss"]))
    plt.figure(figsize=(8, 5))
    plt.bar([str(item["rsn_name"]) for item in latest], [float(item["loss"]) for item in latest])
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("RSN")
    plt.ylabel("Final cosine prediction loss")
    plt.title("Final Per-RSN Loss Ranked")
    paths.append(_save_figure(plots_dir / "final_rsn_loss_ranked.png", save_pdf=save_pdf))
    return paths


def plot_subject_diagnostics(
    embeddings: torch.Tensor,
    similarity: torch.Tensor,
    off_diagonal: torch.Tensor,
    feature_variances: torch.Tensor,
    spectrum: torch.Tensor,
    projection_rows: list[dict[str, Any]],
    plots_dir: Path,
    *,
    save_pdf: bool,
) -> list[Path]:
    paths: list[Path] = []
    plt.figure(figsize=(7, 6))
    image = plt.imshow(similarity.detach().cpu().numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(image, label="Cosine similarity")
    plt.xlabel("Subject")
    plt.ylabel("Subject")
    plt.title("Subject Embedding Cosine Similarity")
    paths.append(_save_figure(plots_dir / "subject_cosine_similarity_heatmap.png", save_pdf=save_pdf))

    plt.figure(figsize=(7, 4))
    values = off_diagonal.detach().cpu().numpy()
    if values.size:
        plt.hist(values, bins=30, range=(-1, 1), edgecolor="black")
    else:
        plt.text(0.5, 0.5, "Fewer than two subjects", ha="center", va="center")
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
    plt.xlabel("Off-diagonal cosine similarity")
    plt.ylabel("Count")
    plt.title("Between-Subject Cosine Similarity")
    paths.append(_save_figure(plots_dir / "subject_cosine_similarity_histogram.png", save_pdf=save_pdf))

    plt.figure(figsize=(7, 4))
    plt.hist(feature_variances.detach().cpu().numpy(), bins=30, edgecolor="black")
    plt.xlabel("Population variance across subjects")
    plt.ylabel("Feature count")
    plt.title("Subject Embedding Feature Variance")
    paths.append(_save_figure(plots_dir / "subject_embedding_variance_distribution.png", save_pdf=save_pdf))

    plt.figure(figsize=(7, 4))
    spectrum_values = spectrum.detach().cpu().numpy()
    if spectrum_values.size:
        plt.plot(np.arange(1, len(spectrum_values) + 1), spectrum_values, marker="o")
    plt.xlabel("Singular component")
    plt.ylabel("Explained-variance fraction")
    plt.title("Subject Embedding Spectrum")
    plt.grid(alpha=0.3)
    paths.append(_save_figure(plots_dir / "subject_embedding_spectrum.png", save_pdf=save_pdf))

    projection = pca_projection(embeddings)
    colors = [row.get("gender", "") for row in projection_rows]
    unique = sorted({color for color in colors if color})
    plt.figure(figsize=(6, 5))
    if unique:
        for value in unique:
            indices = [index for index, color in enumerate(colors) if color == value]
            plt.scatter(projection[indices, 0], projection[indices, 1], label=value, alpha=0.8)
        plt.legend(title="Gender")
    else:
        plt.scatter(projection[:, 0], projection[:, 1], alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Subject Embedding PCA")
    paths.append(_save_figure(plots_dir / "subject_embedding_pca.png", save_pdf=save_pdf))
    return paths


def _metadata_labels(dataset: Any) -> list[dict[str, Any]]:
    getter = getattr(dataset, "get_subject_metadata", None)
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        metadata = getter(index) if getter is not None else {}
        rows.append(
            {
                "gender": str(metadata.get("gender", metadata.get("Gender", ""))).strip(),
                "diagnosis": str(metadata.get("diagnosis", metadata.get("Diagnosis", ""))).strip(),
            }
        )
    return rows


@torch.no_grad()
def extract_region_embedding_summary(
    model: BSJEPA,
    dataset: Any,
    atlas: Atlas,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model_was_training = model.training
    model.target_encoder.eval()
    region_values: dict[int, list[torch.Tensor]] = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=list, drop_last=False)
    try:
        for examples in loader:
            graphs = [example[0] if isinstance(example, tuple) else example for example in examples]
            batch = Batch.from_data_list(list(graphs)).to(device)
            node_embeddings = model.encode(batch)
            parts = unbatch(node_embeddings, batch.batch)
            for graph, part in zip(graphs, parts, strict=True):
                region_ids = getattr(graph, "region_ids", torch.arange(graph.num_nodes))
                for region_id, value in zip(torch.as_tensor(region_ids).cpu(), part.detach().cpu(), strict=True):
                    region_values.setdefault(int(region_id), []).append(value)
    finally:
        model.train(model_was_training)

    rows: list[dict[str, Any]] = []
    region_means: dict[int, torch.Tensor] = {}
    for region_id, values in sorted(region_values.items()):
        mean_embedding = torch.stack(values).float().mean(0)
        region_means[region_id] = mean_embedding
        rsn_id = int(atlas.rsn_ids[region_id]) if region_id < atlas.num_regions else -1
        rows.append(
            {
                "region_id": region_id,
                "rsn_id": rsn_id,
                "rsn_name": atlas.rsn_names[rsn_id] if 0 <= rsn_id < len(atlas.rsn_names) else "",
                "subjects_observed": len(values),
                "embedding_norm": mean_embedding.norm().item(),
            }
        )

    within: list[float] = []
    between: list[float] = []
    region_ids = sorted(region_means)
    for left_index, left in enumerate(region_ids):
        for right in region_ids[left_index + 1:]:
            similarity = torch.cosine_similarity(region_means[left], region_means[right], dim=0).item()
            if int(atlas.rsn_ids[left]) == int(atlas.rsn_ids[right]):
                within.append(similarity)
            else:
                between.append(similarity)
    stats = {
        "within_rsn_cosine_mean": float(np.mean(within)) if within else float("nan"),
        "between_rsn_cosine_mean": float(np.mean(between)) if between else float("nan"),
        "within_minus_between_cosine": (
            float(np.mean(within) - np.mean(between)) if within and between else float("nan")
        ),
        "region_count": float(len(region_ids)),
    }
    return rows, stats


def latest_metric(history: list[dict[str, float]], key: str) -> float | None:
    for row in reversed(history):
        if key in row and row[key] is not None and math.isfinite(float(row[key])):
            return float(row[key])
    return None


def collect_downstream_comparison(history: list[dict[str, float]], extra_metrics: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    merged_history = list(history)
    if extra_metrics:
        merged_history.append({"epoch": float("nan"), **extra_metrics})
    for prefix, label in (
        ("gender_probe", "BS-JEPA target encoder linear probe"),
        ("gender_random_encoder", "Random encoder linear probe"),
        ("gender_raw_adjacency", "Raw adjacency linear probe"),
        ("gender_majority", "Majority baseline"),
    ):
        accuracy = latest_metric(merged_history, f"{prefix}_val_accuracy")
        balanced = latest_metric(merged_history, f"{prefix}_val_balanced_accuracy")
        loss = latest_metric(merged_history, f"{prefix}_val_loss")
        if accuracy is not None or balanced is not None or loss is not None:
            rows.append(
                {
                    "analysis": "gender",
                    "model": label,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced,
                    "validation_loss": loss,
                }
            )
    for key, label in (
        ("pmat_val_mae", "PMAT MAE"),
        ("pmat_val_rmse", "PMAT RMSE"),
        ("pmat_val_r2", "PMAT R2"),
        ("pmat_val_pearson", "PMAT Pearson"),
    ):
        value = latest_metric(merged_history, key)
        if value is not None:
            rows.append({"analysis": "pmat", "model": "BS-JEPA target encoder", "metric": label, "value": value})
    return rows


def plot_downstream(history: list[dict[str, float]], comparison_rows: list[dict[str, Any]], plots_dir: Path, *, save_pdf: bool) -> list[Path]:
    paths: list[Path] = []
    gender_path = plot_metric_lines(
        history,
        [
            "gender_probe_val_accuracy",
            "gender_raw_adjacency_val_accuracy",
            "gender_random_encoder_val_accuracy",
            "gender_majority_val_accuracy",
            "gender_probe_val_balanced_accuracy",
        ],
        [
            "BS-JEPA accuracy",
            "Raw adjacency accuracy",
            "Random encoder accuracy",
            "Majority accuracy",
            "BS-JEPA balanced accuracy",
        ],
        plots_dir / "gender_probe_metrics.png",
        title="Gender Probe Metrics",
        ylabel="Score",
        save_pdf=save_pdf,
    )
    if gender_path is not None:
        paths.append(gender_path)

    pmat_path = plot_metric_lines(
        history,
        ["pmat_val_mae", "pmat_val_rmse", "pmat_val_r2", "pmat_val_pearson"],
        ["MAE", "RMSE", "R2", "Pearson"],
        plots_dir / "pmat_downstream_metrics.png",
        title="PMAT Evaluation Metrics",
        ylabel="Metric value",
        save_pdf=save_pdf,
    )
    if pmat_path is not None:
        paths.append(pmat_path)

    gender_rows = [row for row in comparison_rows if row.get("analysis") == "gender" and row.get("balanced_accuracy") is not None]
    if gender_rows:
        plt.figure(figsize=(8, 4))
        plt.bar([row["model"] for row in gender_rows], [float(row["balanced_accuracy"]) for row in gender_rows])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Balanced accuracy")
        plt.title("Downstream Baseline Comparison")
        paths.append(_save_figure(plots_dir / "downstream_baseline_comparison.png", save_pdf=save_pdf))
    return paths


def plot_gender_embedding_cosine(
    embeddings: torch.Tensor,
    metadata_rows: list[dict[str, Any]],
    plots_dir: Path,
    *,
    save_pdf: bool,
) -> Path | None:
    genders = [row.get("gender", "").upper() for row in metadata_rows]
    groups = sorted({gender for gender in genders if gender})
    if len(groups) < 2:
        return None
    normalized = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
    similarity = normalized @ normalized.T
    within: list[float] = []
    between: list[float] = []
    for left in range(len(genders)):
        for right in range(left + 1, len(genders)):
            if not genders[left] or not genders[right]:
                continue
            if genders[left] == genders[right]:
                within.append(similarity[left, right].item())
            else:
                between.append(similarity[left, right].item())
    if not within or not between:
        return None
    plt.figure(figsize=(5, 4))
    plt.bar(["Within gender", "Between gender"], [float(np.mean(within)), float(np.mean(between))])
    plt.ylabel("Mean cosine similarity")
    plt.title("Subject Embedding Cosine by Gender")
    return _save_figure(plots_dir / "gender_probe_embedding_cosine.png", save_pdf=save_pdf)


def _support_from_subject_metrics(metrics: dict[str, float]) -> str:
    rank = metrics.get("subject_effective_rank", 0.0)
    mean_cosine = metrics.get("subject_cosine_similarity_mean", float("nan"))
    variance = metrics.get("subject_feature_variance_mean", 0.0)
    if rank > 1 and variance > 0 and (not math.isfinite(mean_cosine) or mean_cosine < 0.95):
        return "Supported"
    if rank > 1 or variance > 0:
        return "Partially supported"
    return "Not supported"


def _support_from_downstream(rows: list[dict[str, Any]]) -> str:
    by_model = {row["model"]: row for row in rows if row.get("analysis") == "gender"}
    bsjepa = by_model.get("BS-JEPA target encoder linear probe", {}).get("balanced_accuracy")
    majority = by_model.get("Majority baseline", {}).get("balanced_accuracy")
    random = by_model.get("Random encoder linear probe", {}).get("balanced_accuracy")
    raw = by_model.get("Raw adjacency linear probe", {}).get("balanced_accuracy")
    if bsjepa is None:
        return "Not testable with available outputs"
    baselines = [value for value in (majority, random, raw) if value is not None]
    if baselines and all(float(bsjepa) > float(value) for value in baselines):
        return "Supported"
    if baselines and any(float(bsjepa) > float(value) for value in baselines):
        return "Partially supported"
    return "Not supported"


def write_report(
    state: AnalysisState,
    *,
    config_path: Path,
    checkpoint_path: Path,
    context: AnalysisContext,
    history: list[dict[str, float]],
) -> Path:
    lines = [
        "# Neurobiological Hypothesis-Test Report",
        "",
        f"Analysis time: {datetime.now(timezone.utc).isoformat()}",
        f"Config path: `{config_path}`",
        f"Checkpoint/artifact path: `{checkpoint_path}`",
        f"Output directory: `{state.paths.root}`",
        "",
        "## Dataset Summary",
        "",
        f"- Total subjects in source dataset: {len(context.dataset)}",
        f"- Subjects used for pretraining/embedding analyses: {len(context.training_dataset)}",
        f"- Atlas regions: {context.atlas.num_regions}",
        f"- Atlas RSNs: {context.atlas.num_rsns}",
        f"- Evaluation holdout subjects: {len(context.evaluation_dataset) if context.evaluation_dataset is not None else 0}",
        f"- Gender probe holdout subjects: {len(context.linear_probe_dataset) if context.linear_probe_dataset is not None else 0}",
        "",
        "## Model Summary",
        "",
        f"- Input feature dimension: {context.input_feature_dim}",
        f"- Model config: `{json.dumps(_json_safe(context.model_config), sort_keys=True)}`",
        f"- Training history rows available: {len(history)}",
        "",
    ]
    descriptions = {
        "rsn_structure": "RSN Structure Hypothesis",
        "inter_network_dependency": "Inter-Network Dependency Hypothesis",
        "region_specific_representation": "Region-Specific Representation Hypothesis",
        "subject_specific_variation": "Subject-Specific Neurobiological Variation Hypothesis",
        "compact_fingerprint": "Compact Functional Fingerprint Hypothesis",
    }
    tested = {
        "rsn_structure": "Per-RSN prediction-loss trajectories, final ranked RSN losses, RSN loss variability, and RSN grouping of mean region embeddings.",
        "inter_network_dependency": "Training loss, prediction-target alignment proxies, variance diagnostics, covariance diagnostics, and representation non-collapse controls.",
        "region_specific_representation": "Region-stage cross-subject variance, variance retention, positional-feature norm ratios, and per-region stage variance.",
        "subject_specific_variation": "Frozen target-encoder subject embeddings, pairwise subject cosine similarity, feature variance, effective rank, standardized distances, and PCA projection.",
        "compact_fingerprint": "Logged or rerun downstream probe metrics for BS-JEPA embeddings and available baselines.",
    }
    generated = {
        "rsn_structure": "rsn_loss_history.csv, rsn_loss_summaries.csv, rsn_loss_over_time.png, final_rsn_loss_ranked.png, and region_embedding_rsn_summary.csv when available.",
        "inter_network_dependency": "training_metric_summaries.csv, training_loss_over_time.png, representation_variance_over_time.png, prediction_target_alignment_over_time.png, and collapse_diagnostics_over_time.png when available.",
        "region_specific_representation": "region_stage_summaries.csv, region_stage_per_region_variance.csv, region_stage_variance_over_time.png, region_stage_variance_retention.png, and region_stage_position_norms.png when available.",
        "subject_specific_variation": "subject_embeddings.pt, subject_embeddings.csv, subject_embedding_diagnostics.csv, subject cosine plots, feature-variance plot, spectrum plot, and PCA plot.",
        "compact_fingerprint": "downstream_baseline_comparison.csv, gender_probe_metrics.png, gender_probe_embedding_cosine.png, pmat_downstream_metrics.png, and downstream_baseline_comparison.png when available.",
    }
    for key, title in descriptions.items():
        lines.extend(
            [
                f"## {title}",
                "",
                f"What was tested: {tested[key]}",
                "",
                f"Generated metrics/plots: {generated[key]}",
                "",
                f"Evidence classification: **{state.hypothesis_status.get(key, 'Not testable with available outputs')}**",
                "",
            ]
        )
        result = state.key_results.get(key)
        if result is not None:
            lines.append("Key quantitative results:")
            lines.append("")
            for metric, value in _json_safe(result).items():
                lines.append(f"- {metric}: {value}")
            lines.append("")
        lines.append(
            "Interpretation is conservative: prediction performance or non-collapse alone is not treated as proof of neurobiological meaning."
        )
        lines.append("")
    if state.notes:
        lines.extend(["## Caveats And Unavailable Analyses", ""])
        for note in state.notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.extend(
        [
            "## Final Conclusion",
            "",
            "These analyses test whether the saved BS-JEPA representations are structured, non-collapsed, and useful for available downstream signals. Claims should be treated as descriptive unless matched baselines and leakage-controlled splits support them.",
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in sorted(state.generated_files):
        lines.append(f"- `{path.relative_to(state.paths.root)}`")
    report_path = state.add_file(state.paths.reports / "neurobiological_hypothesis_report.md")
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def run_neurobiological_hypothesis_tests(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    overrides: list[str] | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    save_pdf: bool = True,
    run_downstream_probes: bool = True,
) -> AnalysisState:
    paths = prepare_output_dirs(output_dir)
    state = AnalysisState(paths)
    config = load_config(config_path, overrides)
    checkpoint, history, resolved_checkpoint = load_checkpoint_and_history(checkpoint_path)
    context = build_analysis_context(config, checkpoint)
    model = build_model_from_context(context, checkpoint)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    effective_batch_size = batch_size or int(config.get("training", {}).get("subject_embedding_batch_size", 32))

    state.add_file(write_json(paths.metrics / "training_history.json", history))
    metric_summary = summarize_history_metrics(history)
    state.add_file(write_rows_csv(paths.tables / "training_metric_summaries.csv", metric_summary))
    state.add_file(write_json(paths.metrics / "training_metric_summaries.json", metric_summary))

    for path in (
        plot_metric_lines(history, ["loss"], ["Total loss"], paths.plots / "training_loss_over_time.png", title="Training Loss Over Time", ylabel="Loss", save_pdf=save_pdf),
        plot_metric_lines(
            history,
            ["prediction_variance", "context_variance", "target_std", "prediction_embedding_std", "target_embedding_std", "context_embedding_std"],
            ["Prediction variance penalty", "Context variance penalty", "Target std", "Prediction embedding std", "Target embedding std", "Context embedding std"],
            paths.plots / "representation_variance_over_time.png",
            title="Representation Variance Diagnostics",
            ylabel="Metric value",
            save_pdf=save_pdf,
        ),
        plot_metric_lines(
            history,
            ["similarity", "prediction_pairwise_cosine", "target_pairwise_cosine", "context_pairwise_cosine"],
            ["Prediction-target loss", "Prediction pairwise cosine", "Target pairwise cosine", "Context pairwise cosine"],
            paths.plots / "prediction_target_alignment_over_time.png",
            title="Prediction-Target Alignment And Cosine Diagnostics",
            ylabel="Metric value",
            save_pdf=save_pdf,
        ),
        plot_metric_lines(
            history,
            ["context_covariance", "prediction_embedding_std", "target_embedding_std", "context_embedding_std"],
            ["Context covariance", "Prediction std", "Target std", "Context std"],
            paths.plots / "collapse_diagnostics_over_time.png",
            title="Collapse Diagnostics",
            ylabel="Metric value",
            save_pdf=save_pdf,
        ),
    ):
        if path is not None:
            state.add_file(path)

    loss_initial = latest_metric(history[:1], "loss") if history else None
    loss_final = latest_metric(history, "loss")
    pred_std_final = latest_metric(history, "prediction_embedding_std")
    if loss_initial is not None and loss_final is not None and loss_final < loss_initial and (pred_std_final is None or pred_std_final > 0):
        state.hypothesis_status["inter_network_dependency"] = "Supported"
    elif loss_final is not None:
        state.hypothesis_status["inter_network_dependency"] = "Partially supported"
    else:
        state.hypothesis_status["inter_network_dependency"] = "Not testable with available outputs"
    state.key_results["inter_network_dependency"] = {
        "initial_loss": loss_initial,
        "final_loss": loss_final,
        "final_prediction_embedding_std": pred_std_final,
    }

    rsn_rows = extract_rsn_loss_rows(history, context.atlas.rsn_names)
    if rsn_rows:
        rsn_summary, rsn_stats = summarize_rsn_losses(rsn_rows)
        state.add_file(write_rows_csv(paths.tables / "rsn_loss_history.csv", rsn_rows))
        state.add_file(write_rows_csv(paths.tables / "rsn_loss_summaries.csv", rsn_summary))
        groups: dict[str, list[float]] = {}
        for row in rsn_rows:
            groups.setdefault(str(row["rsn_name"]), []).append(float(row["loss"]))
        rsn_stats.update(anova_variance_ratio(groups))
        state.add_file(write_json(paths.metrics / "rsn_loss_statistics.json", rsn_stats))
        state.add_file(write_rows_csv(paths.tables / "rsn_loss_statistics.csv", [rsn_stats]))
        for path in plot_rsn_losses(rsn_rows, paths.plots, save_pdf=save_pdf):
            state.add_file(path)
        improved = sum(1 for row in rsn_summary if float(row["absolute_improvement"]) > 0)
        state.hypothesis_status["rsn_structure"] = "Partially supported" if improved else "Not supported"
        state.key_results["rsn_structure"] = {**rsn_stats, "rsns_with_loss_improvement": improved}
    else:
        state.add_note("Per-RSN loss metrics were not found in training history; RSN loss plots and tables were skipped.")
        state.hypothesis_status["rsn_structure"] = "Not testable with available outputs"

    embeddings, subject_ids, region_stages = extract_target_encoder_diagnostics(
        model,
        context.training_dataset,
        device=torch_device,
        batch_size=effective_batch_size,
        collect_region_stages=True,
    )
    state.add_file(paths.embeddings / "subject_embeddings.pt")
    torch.save({"embeddings": embeddings, "subject_ids": subject_ids}, paths.embeddings / "subject_embeddings.pt")
    metadata_rows = _metadata_labels(context.training_dataset)
    projection = pca_projection(embeddings)
    embedding_rows = [
        {
            "subject_id": subject_id,
            "pc1": projection[index, 0].item(),
            "pc2": projection[index, 1].item(),
            **metadata_rows[index],
            **{f"embedding_{dim}": embeddings[index, dim].item() for dim in range(embeddings.shape[1])},
        }
        for index, subject_id in enumerate(subject_ids)
    ]
    state.add_file(write_rows_csv(paths.embeddings / "subject_embeddings.csv", embedding_rows))

    similarity, off_diagonal, similarity_metrics = subject_similarity_diagnostics(embeddings)
    _, _, centered_metrics = cohort_centered_cosine_diagnostics(embeddings)
    feature_variances, spectrum, rank_metrics = subject_variance_rank_diagnostics(embeddings)
    _, _, distance_metrics = standardized_euclidean_diagnostics(embeddings)
    subject_metrics = {**similarity_metrics, **centered_metrics, **rank_metrics, **distance_metrics}
    subject_metrics["subject_feature_variance_mean"] = feature_variances.mean().item()
    state.add_file(write_json(paths.metrics / "subject_embedding_diagnostics.json", subject_metrics))
    state.add_file(write_rows_csv(paths.tables / "subject_embedding_diagnostics.csv", [subject_metrics]))
    state.add_file(write_rows_csv(paths.tables / "subject_embedding_projection.csv", embedding_rows))
    for path in plot_subject_diagnostics(
        embeddings,
        similarity,
        off_diagonal,
        feature_variances,
        spectrum,
        metadata_rows,
        paths.plots,
        save_pdf=save_pdf,
    ):
        state.add_file(path)
    gender_cosine_path = plot_gender_embedding_cosine(embeddings, metadata_rows, paths.plots, save_pdf=save_pdf)
    if gender_cosine_path is not None:
        state.add_file(gender_cosine_path)
    else:
        state.add_note("Gender labels were unavailable or insufficient for gender_probe_embedding_cosine.png.")
    state.hypothesis_status["subject_specific_variation"] = _support_from_subject_metrics(subject_metrics)
    state.key_results["subject_specific_variation"] = subject_metrics

    if region_stages:
        region_metrics, per_region = region_stage_cross_subject_diagnostics(region_stages)
        state.add_file(write_json(paths.metrics / "region_stage_summaries.json", region_metrics))
        state.add_file(write_rows_csv(paths.tables / "region_stage_summaries.csv", [region_metrics]))
        region_rows = [
            {"stage": stage, "region_id": region_id, "variance": value}
            for stage, values in per_region.items()
            for region_id, value in values.items()
        ]
        state.add_file(write_rows_csv(paths.tables / "region_stage_per_region_variance.csv", region_rows))
        region_history = (
            history
            if any(
                key.startswith("region_")
                and (
                    key.endswith("_cross_subject_variance_mean")
                    or key.endswith("_variance_retention_ratio")
                    or key in {
                        "region_projected_feature_norm_mean",
                        "region_position_norm_mean",
                        "region_post_position_norm_mean",
                        "region_position_to_feature_norm_ratio",
                    }
                )
                for row in history
                for key in row
            )
            else [{"epoch": 1.0, **region_metrics}]
        )
        for path in (
            plot_metric_lines(
                region_history,
                [key for key in region_metrics if key.endswith("_cross_subject_variance_mean")],
                [key.removeprefix("region_").removesuffix("_cross_subject_variance_mean") for key in region_metrics if key.endswith("_cross_subject_variance_mean")],
                paths.plots / "region_stage_variance_over_time.png",
                title="Region-Stage Cross-Subject Variance",
                ylabel="Mean variance",
                save_pdf=save_pdf,
            ),
            plot_metric_lines(
                region_history,
                [key for key in region_metrics if key.endswith("_variance_retention_ratio")],
                [key.removeprefix("region_").removesuffix("_variance_retention_ratio") for key in region_metrics if key.endswith("_variance_retention_ratio")],
                paths.plots / "region_stage_variance_retention.png",
                title="Region-Stage Variance Retention",
                ylabel="Variance / temporal variance",
                save_pdf=save_pdf,
            ),
            plot_metric_lines(
                region_history,
                ["region_projected_feature_norm_mean", "region_position_norm_mean", "region_post_position_norm_mean", "region_position_to_feature_norm_ratio"],
                ["Projected feature norm", "Position norm", "Post-position norm", "Position / feature norm"],
                paths.plots / "region_stage_position_norms.png",
                title="Region Position And Feature Norms",
                ylabel="Norm or ratio",
                save_pdf=save_pdf,
            ),
        ):
            if path is not None:
                state.add_file(path)
        nonzero_final = region_metrics.get("region_final_cross_subject_variance_mean", 0.0) > 0
        ratio = region_metrics.get("region_position_to_feature_norm_ratio", float("nan"))
        state.hypothesis_status["region_specific_representation"] = (
            "Partially supported" if nonzero_final and (not math.isfinite(ratio) or ratio < 10) else "Not supported"
        )
        state.key_results["region_specific_representation"] = region_metrics
    else:
        state.add_note("Region-stage diagnostics could not be extracted from the target encoder.")
        state.hypothesis_status["region_specific_representation"] = "Not testable with available outputs"

    region_embedding_rows, region_embedding_stats = extract_region_embedding_summary(
        model,
        context.training_dataset,
        context.atlas,
        device=torch_device,
        batch_size=effective_batch_size,
    )
    state.add_file(write_rows_csv(paths.tables / "region_embedding_rsn_summary.csv", region_embedding_rows))
    state.add_file(write_json(paths.metrics / "region_embedding_rsn_statistics.json", region_embedding_stats))
    state.key_results.setdefault("rsn_structure", {}).update(region_embedding_stats)

    downstream_metrics: dict[str, float] = {}
    if run_downstream_probes and context.linear_probe_dataset is not None:
        random_model = build_bsjepa(
            in_channels=context.input_feature_dim,
            num_regions=context.atlas.num_regions,
            **context.model_config,
        )
        downstream_metrics.update(
            evaluate_gender_probe(
                model,
                context.linear_probe_dataset,
                context.config["linear_probe"],
                device=torch_device,
                random_model=random_model,
            )
        )
    elif context.linear_probe_dataset is None:
        state.add_note("No gender probe holdout was available; downstream gender probes were not rerun.")
    if run_downstream_probes and context.evaluation_dataset is not None:
        downstream_metrics.update(
            evaluate_pmat(
                model,
                context.evaluation_dataset,
                context.config["evaluation"],
                device=torch_device,
            )
        )
    elif context.evaluation_dataset is None:
        state.add_note("No PMAT evaluation holdout was available; PMAT downstream evaluation was not rerun.")
    if downstream_metrics:
        state.add_file(write_json(paths.metrics / "rerun_downstream_probe_metrics.json", downstream_metrics))

    comparison_rows = collect_downstream_comparison(history, downstream_metrics)
    if comparison_rows:
        state.add_file(write_rows_csv(paths.tables / "downstream_baseline_comparison.csv", comparison_rows))
        state.add_file(write_json(paths.metrics / "downstream_baseline_comparison.json", comparison_rows))
    else:
        state.add_note("No downstream baseline metrics were found or rerun.")
    for path in plot_downstream(history + ([{"epoch": history[-1]["epoch"] + 1 if history else 1.0, **downstream_metrics}] if downstream_metrics else []), comparison_rows, paths.plots, save_pdf=save_pdf):
        state.add_file(path)
    if not any(path.name == "gender_probe_metrics.png" for path in state.generated_files):
        state.add_note("gender_probe_metrics.png was not generated because no gender probe metrics were available.")
    if not any(path.name == "downstream_baseline_comparison.png" for path in state.generated_files):
        state.add_note("downstream_baseline_comparison.png was not generated because no comparable downstream baselines were available.")
    state.hypothesis_status["compact_fingerprint"] = _support_from_downstream(comparison_rows)
    state.key_results["compact_fingerprint"] = {"downstream_rows": comparison_rows}

    write_report(
        state,
        config_path=Path(config_path),
        checkpoint_path=resolved_checkpoint,
        context=context,
        history=history,
    )
    manifest_rows = [
        {"path": str(path.relative_to(paths.root)), "size_bytes": path.stat().st_size if path.exists() else 0}
        for path in sorted(state.generated_files)
    ]
    state.add_file(write_rows_csv(paths.tables / "generated_file_manifest.csv", manifest_rows))
    state.add_file(write_json(paths.metrics / "generated_file_manifest.json", manifest_rows))
    write_report(
        state,
        config_path=Path(config_path),
        checkpoint_path=resolved_checkpoint,
        context=context,
        history=history,
    )
    return state
