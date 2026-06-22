"""Headless diagnostic plotting for BS-JEPA pretraining."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _save_figure(path: Path, *, dpi: int, save_pdf: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def save_subject_similarity_plots(
    similarity: torch.Tensor,
    off_diagonal: torch.Tensor,
    subject_ids: list[str],
    plot_dir: str | Path,
    *,
    epoch: int,
    dpi: int = 150,
    save_pdf: bool = False,
    max_tick_labels: int = 40,
    histogram_bins: int = 30,
) -> None:
    """Save the latest subject cosine-similarity heatmap and histogram."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Subject similarity matrix must be square")
    if len(subject_ids) != similarity.shape[0]:
        raise ValueError("Subject IDs must align with the similarity matrix")
    if dpi < 1 or max_tick_labels < 0 or histogram_bins < 1:
        raise ValueError("Plot settings must be positive (max_tick_labels may be zero)")
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    matrix = similarity.detach().cpu().numpy()

    plt.figure(figsize=(7, 6))
    image = plt.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(image, label="Cosine similarity")
    subject_count = len(subject_ids)
    if max_tick_labels > 0 and subject_count:
        step = max(1, int(np.ceil(subject_count / max_tick_labels)))
        positions = np.arange(0, subject_count, step)
        labels = [subject_ids[index] for index in positions]
        plt.xticks(positions, labels, rotation=90, fontsize=7)
        plt.yticks(positions, labels, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.xlabel("Subject")
    plt.ylabel("Subject")
    plt.title(f"EMA Target Subject Similarity (Epoch {epoch})")
    _save_figure(
        path / "subject_similarity_heatmap.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 4))
    values = off_diagonal.detach().cpu().numpy()
    if values.size:
        plt.hist(values, bins=histogram_bins, range=(-1, 1), edgecolor="black")
    else:
        plt.text(0.5, 0.5, "Fewer than two subjects", ha="center", va="center")
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
    plt.xlabel("Off-diagonal cosine similarity")
    plt.ylabel("Count")
    plt.title(f"Between-Subject Similarity (Epoch {epoch})")
    _save_figure(
        path / "subject_similarity_histogram.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )


def _set_subject_ticks(subject_ids: list[str], max_tick_labels: int) -> None:
    subject_count = len(subject_ids)
    if max_tick_labels > 0 and subject_count:
        step = max(1, int(np.ceil(subject_count / max_tick_labels)))
        positions = np.arange(0, subject_count, step)
        labels = [subject_ids[index] for index in positions]
        plt.xticks(positions, labels, rotation=90, fontsize=7)
        plt.yticks(positions, labels, fontsize=7)
    else:
        plt.xticks([])
        plt.yticks([])


def save_extended_subject_diagnostic_plots(
    centered_similarity: torch.Tensor,
    centered_off_diagonal: torch.Tensor,
    feature_variances: torch.Tensor,
    explained_variance: torch.Tensor,
    standardized_distances: torch.Tensor,
    distance_off_diagonal: torch.Tensor,
    subject_ids: list[str],
    plot_dir: str | Path,
    *,
    epoch: int,
    dpi: int = 150,
    save_pdf: bool = False,
    max_tick_labels: int = 40,
    histogram_bins: int = 30,
) -> None:
    """Save the latest centered, spectral, and standardized-distance plots."""
    subject_count = len(subject_ids)
    for name, matrix in (
        ("centered similarity", centered_similarity),
        ("standardized distance", standardized_distances),
    ):
        if matrix.ndim != 2 or matrix.shape != (subject_count, subject_count):
            raise ValueError(f"Subject {name} matrix must align with subject IDs")
    if dpi < 1 or max_tick_labels < 0 or histogram_bins < 1:
        raise ValueError("Plot settings must be positive (max_tick_labels may be zero)")
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    image = plt.imshow(
        centered_similarity.detach().cpu().numpy(),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        aspect="auto",
    )
    plt.colorbar(image, label="Centered cosine similarity")
    _set_subject_ticks(subject_ids, max_tick_labels)
    plt.xlabel("Subject")
    plt.ylabel("Subject")
    plt.title(f"Cohort-Centered Subject Similarity (Epoch {epoch})")
    _save_figure(
        path / "subject_centered_cosine_heatmap.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 4))
    centered_values = centered_off_diagonal.detach().cpu().numpy()
    if centered_values.size:
        plt.hist(
            centered_values,
            bins=histogram_bins,
            range=(-1, 1),
            edgecolor="black",
        )
    else:
        plt.text(0.5, 0.5, "Fewer than two subjects", ha="center", va="center")
        plt.xlim(-1, 1)
        plt.ylim(0, 1)
    plt.xlabel("Off-diagonal centered cosine similarity")
    plt.ylabel("Count")
    plt.title(f"Cohort-Centered Similarity (Epoch {epoch})")
    _save_figure(
        path / "subject_centered_cosine_histogram.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 4))
    plt.hist(
        feature_variances.detach().cpu().numpy(),
        bins=histogram_bins,
        edgecolor="black",
    )
    plt.xlabel("Population variance across subjects")
    plt.ylabel("Feature count")
    plt.title(f"Subject-Embedding Feature Variance (Epoch {epoch})")
    _save_figure(
        path / "subject_feature_variance_histogram.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 4))
    spectrum = explained_variance.detach().cpu().numpy()
    if spectrum.size:
        plt.plot(np.arange(1, len(spectrum) + 1), spectrum, marker="o")
    plt.xlabel("Singular component")
    plt.ylabel("Explained-variance fraction")
    plt.title(f"Centered Subject-Embedding Spectrum (Epoch {epoch})")
    plt.grid(alpha=0.3)
    _save_figure(
        path / "subject_effective_rank_spectrum.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 6))
    image = plt.imshow(
        standardized_distances.detach().cpu().numpy(),
        vmin=0,
        cmap="viridis",
        aspect="auto",
    )
    plt.colorbar(image, label="Standardized Euclidean distance")
    _set_subject_ticks(subject_ids, max_tick_labels)
    plt.xlabel("Subject")
    plt.ylabel("Subject")
    plt.title(f"Standardized Subject Distance (Epoch {epoch})")
    _save_figure(
        path / "subject_standardized_distance_heatmap.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )

    plt.figure(figsize=(7, 4))
    distance_values = distance_off_diagonal.detach().cpu().numpy()
    if distance_values.size:
        plt.hist(distance_values, bins=histogram_bins, edgecolor="black")
    else:
        plt.text(0.5, 0.5, "Fewer than two subjects", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    plt.xlabel("Off-diagonal standardized Euclidean distance")
    plt.ylabel("Count")
    plt.title(f"Standardized Subject Distances (Epoch {epoch})")
    _save_figure(
        path / "subject_standardized_distance_histogram.png",
        dpi=dpi,
        save_pdf=save_pdf,
    )


def _save_training_plots(
    history: list[dict[str, float]],
    plot_dir: str | Path,
    *,
    dpi: int = 150,
    save_pdf: bool = False,
) -> None:
    """Write loss, per-RSN loss, and collapse diagnostic curves."""
    if not history:
        return
    if dpi < 1:
        raise ValueError("Plot DPI must be positive")
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["loss"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    _save_figure(path / "training_loss.png", dpi=dpi, save_pdf=save_pdf)

    if any("similarity" in row for row in history):
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, [row.get("similarity", float("nan")) for row in history])
        plt.xlabel("Epoch")
        plt.ylabel("Cosine prediction loss")
        plt.title("JEPA Prediction Loss")
        plt.grid(alpha=0.3)
        _save_figure(path / "prediction_loss.png", dpi=dpi, save_pdf=save_pdf)

    rsn_keys = sorted(
        {key for row in history for key in row if key.startswith("rsn_loss_")},
        key=lambda key: int(key.rsplit("_", 1)[1]),
    )
    if rsn_keys:
        plt.figure(figsize=(8, 5))
        for key in rsn_keys:
            values = [row.get(key, float("nan")) for row in history]
            plt.plot(epochs, values, label=f"RSN {key.rsplit('_', 1)[1]}")
        plt.xlabel("Epoch")
        plt.ylabel("Cosine prediction loss")
        plt.title("Per-RSN prediction loss")
        plt.grid(alpha=0.3)
        plt.legend(fontsize="small", ncol=2)
        _save_figure(
            path / "rsn_prediction_losses.png", dpi=dpi, save_pdf=save_pdf
        )

    collapse_groups = {
        "anti_collapse_losses.png": [
            "prediction_variance",
            "context_variance",
            "context_covariance",
            "target_std",
            "rsn_diversity",
        ],
        "embedding_standard_deviations.png": [
            "context_embedding_std",
            "target_embedding_std",
            "prediction_embedding_std",
        ],
        "embedding_pairwise_cosine.png": [
            "context_pairwise_cosine",
            "target_pairwise_cosine",
            "prediction_pairwise_cosine",
        ],
        "embedding_norms.png": [
            "context_norm_mean",
            "context_norm_std",
            "target_norm_mean",
            "target_norm_std",
            "prediction_norm_mean",
            "prediction_norm_std",
        ],
    }
    for filename, keys in collapse_groups.items():
        available = [key for key in keys if any(key in row for row in history)]
        if not available:
            continue
        plt.figure(figsize=(8, 5))
        for key in available:
            plt.plot(
                epochs,
                [row.get(key, float("nan")) for row in history],
                label=key.replace("_", " "),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(filename.removesuffix(".png").replace("_", " ").title())
        plt.grid(alpha=0.3)
        plt.legend(fontsize="small")
        _save_figure(path / filename, dpi=dpi, save_pdf=save_pdf)

    downstream_rows = [row for row in history if "pmat_val_mae" in row]
    if downstream_rows:
        evaluation_epochs = [row["epoch"] for row in downstream_rows]
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for key in ("pmat_val_mae", "pmat_val_rmse"):
            axes[0].plot(
                evaluation_epochs,
                [row[key] for row in downstream_rows],
                label=key.removeprefix("pmat_val_").upper(),
            )
        axes[0].set_ylabel("PMAT score error")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        for key in ("pmat_val_r2", "pmat_val_pearson"):
            axes[1].plot(
                evaluation_epochs,
                [row[key] for row in downstream_rows],
                label=key.removeprefix("pmat_val_").replace("r2", "R²").title(),
            )
        axes[1].set_xlabel("Pretraining epoch")
        axes[1].set_ylabel("Validation score")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        figure.suptitle("Frozen Target Encoder PMAT Evaluation")
        _save_figure(
            path / "pmat_downstream_metrics.png", dpi=dpi, save_pdf=save_pdf
        )

    gender_rows = [row for row in history if "gender_probe_val_accuracy" in row]
    if gender_rows:
        probe_epochs = [row["epoch"] for row in gender_rows]
        figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(
            probe_epochs,
            [row["gender_probe_val_accuracy"] for row in gender_rows],
            label="Accuracy",
        )
        axes[0].plot(
            probe_epochs,
            [row["gender_probe_val_balanced_accuracy"] for row in gender_rows],
            label="Balanced accuracy",
        )
        axes[0].set_ylabel("Validation score")
        axes[0].set_ylim(0, 1)
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].plot(
            probe_epochs,
            [row["gender_probe_val_loss"] for row in gender_rows],
            label="Cross-entropy loss",
        )
        axes[1].set_xlabel("Pretraining epoch")
        axes[1].set_ylabel("Validation loss")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        figure.suptitle("Frozen Target Encoder Gender Probe")
        _save_figure(path / "gender_probe_metrics.png", dpi=dpi, save_pdf=save_pdf)


def save_training_plots(
    history: list[dict[str, float]],
    plot_dir: str | Path,
    *,
    dpi: int = 150,
    save_pdf: bool = False,
) -> None:
    """Save consistently styled, headless diagnostic plots."""
    style = {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with matplotlib.rc_context(style):
        _save_training_plots(
            history, plot_dir, dpi=dpi, save_pdf=save_pdf
        )
