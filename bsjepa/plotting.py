"""Headless diagnostic plotting for BS-JEPA pretraining."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save_figure(path: Path, *, dpi: int, save_pdf: bool) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


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
        "region_subject_specificity.png": [
            "prediction_same_region_across_subject_cosine",
            "prediction_across_region_within_subject_cosine",
        ],
        "subject_graph_embedding_variance.png": [
            "context_graph_embedding_subject_std",
            "target_graph_embedding_subject_std",
            "prediction_graph_embedding_subject_std",
        ],
        "subject_graph_embedding_similarity.png": [
            "context_graph_embedding_subject_pairwise_cosine",
            "target_graph_embedding_subject_pairwise_cosine",
            "prediction_graph_embedding_subject_pairwise_cosine",
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
