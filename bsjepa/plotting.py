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
    _set_subject_ticks(subject_ids, max_tick_labels)
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


def _region_stage_sort_key(stage: str) -> tuple[int, int | str]:
    fixed_order = {
        "raw": 0,
        "temporal": 1,
        "projection": 2,
        "position": 3,
        "post_position": 4,
        "final": 1000,
    }
    if stage in fixed_order:
        return fixed_order[stage], stage
    if "_layer_" in stage:
        return 5, int(stage.rsplit("_", 1)[1])
    return 999, stage


def save_region_stage_diagnostic_plots(
    history: list[dict[str, float]],
    per_region_variances: dict[str, dict[int, float]],
    plot_dir: str | Path,
    *,
    epoch: int,
    dpi: int = 150,
    save_pdf: bool = False,
) -> None:
    """Save cross-subject region-stage variance and positional-norm diagnostics."""
    if dpi < 1:
        raise ValueError("Plot DPI must be positive")
    path = Path(plot_dir)
    path.mkdir(parents=True, exist_ok=True)
    variance_suffix = "_cross_subject_variance_mean"
    stage_keys = {
        key.removeprefix("region_").removesuffix(variance_suffix)
        for row in history
        for key in row
        if key.startswith("region_") and key.endswith(variance_suffix)
    }
    stages = sorted(stage_keys, key=_region_stage_sort_key)
    diagnostic_rows = [
        row
        for row in history
        if any(f"region_{stage}{variance_suffix}" in row for stage in stages)
    ]
    if diagnostic_rows and stages:
        plt.figure(figsize=(9, 5))
        epochs = [row["epoch"] for row in diagnostic_rows]
        for stage in stages:
            plt.plot(
                epochs,
                [
                    row.get(f"region_{stage}{variance_suffix}", float("nan"))
                    for row in diagnostic_rows
                ],
                marker="o",
                label=stage.replace("_", " "),
            )
        plt.xlabel("Epoch")
        plt.ylabel("Mean same-region variance")
        plt.title("Cross-Subject Variance by Encoder Stage")
        plt.grid(alpha=0.3)
        plt.legend(fontsize="small", ncol=2)
        _save_figure(
            path / "region_stage_cross_subject_variance.png",
            dpi=dpi,
            save_pdf=save_pdf,
        )

    current_stages = sorted(per_region_variances, key=_region_stage_sort_key)
    region_ids = sorted(
        {
            region_id
            for values in per_region_variances.values()
            for region_id in values
        }
    )
    if current_stages and region_ids:
        matrix = np.full((len(current_stages), len(region_ids)), np.nan)
        region_positions = {region_id: index for index, region_id in enumerate(region_ids)}
        for stage_index, stage in enumerate(current_stages):
            for region_id, value in per_region_variances[stage].items():
                matrix[stage_index, region_positions[region_id]] = value
        plt.figure(figsize=(max(9, len(region_ids) * 0.04), 5))
        image = plt.imshow(matrix, cmap="viridis", aspect="auto")
        plt.colorbar(image, label="Mean feature-wise variance")
        step = max(1, int(np.ceil(len(region_ids) / 40)))
        positions = np.arange(0, len(region_ids), step)
        plt.xticks(
            positions,
            [str(region_ids[index]) for index in positions],
            rotation=90,
            fontsize=7,
        )
        plt.yticks(
            np.arange(len(current_stages)),
            [stage.replace("_", " ") for stage in current_stages],
        )
        plt.xlabel("Atlas region ID")
        plt.ylabel("Encoder stage")
        plt.title(f"Same-Region Cross-Subject Variance (Epoch {epoch})")
        _save_figure(
            path / "region_stage_per_region_variance_heatmap.png",
            dpi=dpi,
            save_pdf=save_pdf,
        )

    norm_rows = [
        row for row in history if "region_projected_feature_norm_mean" in row
    ]
    if norm_rows:
        epochs = [row["epoch"] for row in norm_rows]
        plt.figure(figsize=(7, 4))
        for key, label in (
            ("region_projected_feature_norm_mean", "Projected feature"),
            ("region_position_norm_mean", "Position"),
            ("region_post_position_norm_mean", "After addition"),
        ):
            plt.plot(epochs, [row[key] for row in norm_rows], marker="o", label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Mean node norm")
        plt.title("Feature and Atlas-Position Norms")
        plt.grid(alpha=0.3)
        plt.legend()
        _save_figure(
            path / "region_stage_feature_position_norms.png",
            dpi=dpi,
            save_pdf=save_pdf,
        )

        plt.figure(figsize=(7, 4))
        plt.plot(
            epochs,
            [row["region_position_to_feature_norm_ratio"] for row in norm_rows],
            marker="o",
        )
        plt.axhline(1, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Position norm / projected-feature norm")
        plt.title("Atlas-Position Dominance Ratio")
        plt.grid(alpha=0.3)
        _save_figure(
            path / "region_stage_position_feature_ratio.png",
            dpi=dpi,
            save_pdf=save_pdf,
        )

    retention_suffix = "_variance_retention_ratio"
    retention_stages = sorted(
        {
            key.removeprefix("region_").removesuffix(retention_suffix)
            for row in history
            for key in row
            if key.startswith("region_") and key.endswith(retention_suffix)
        },
        key=_region_stage_sort_key,
    )
    retention_rows = [
        row
        for row in history
        if any(f"region_{stage}{retention_suffix}" in row for stage in retention_stages)
    ]
    if retention_rows and retention_stages:
        epochs = [row["epoch"] for row in retention_rows]
        plt.figure(figsize=(9, 5))
        for stage in retention_stages:
            plt.plot(
                epochs,
                [
                    row.get(f"region_{stage}{retention_suffix}", float("nan"))
                    for row in retention_rows
                ],
                marker="o",
                label=stage.replace("_", " "),
            )
        plt.axhline(1, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Variance / temporal-feature variance")
        plt.title("Layer-Wise Cross-Subject Variance Retention")
        plt.grid(alpha=0.3)
        plt.legend(fontsize="small", ncol=2)
        _save_figure(
            path / "region_stage_variance_retention.png",
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

    rank_rows = [row for row in history if "subject_effective_rank" in row]
    if rank_rows:
        plt.figure(figsize=(7, 4))
        plt.plot(
            [row["epoch"] for row in rank_rows],
            [row["subject_effective_rank"] for row in rank_rows],
            marker="o",
        )
        plt.xlabel("Pretraining epoch")
        plt.ylabel("Effective rank")
        plt.title("EMA Subject-Embedding Effective Rank")
        plt.grid(alpha=0.3)
        _save_figure(
            path / "subject_effective_rank_over_time.png",
            dpi=dpi,
            save_pdf=save_pdf,
        )

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
        figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        comparisons = (
            ("gender_majority", "Majority class", "--", "tab:blue"),
            ("gender_raw_adjacency", "Raw adjacency linear", "-.", "tab:orange"),
            ("gender_random_encoder", "Random encoder linear", ":", "tab:green"),
            ("gender_probe", "Trained BS-JEPA linear", "-", "tab:red"),
        )
        for prefix, label, linestyle, color in comparisons:
            accuracy_key = f"{prefix}_val_accuracy"
            if accuracy_key in gender_rows[0]:
                axes[0].plot(
                    probe_epochs,
                    [row[accuracy_key] for row in gender_rows],
                    label=label,
                    linestyle=linestyle,
                    marker="o",
                    color=color,
                )
            balanced_key = f"{prefix}_val_balanced_accuracy"
            if balanced_key in gender_rows[0]:
                axes[1].plot(
                    probe_epochs,
                    [row[balanced_key] for row in gender_rows],
                    label=label,
                    linestyle=linestyle,
                    marker="o",
                    color=color,
                )
            loss_key = f"{prefix}_val_loss"
            if loss_key in gender_rows[0]:
                axes[2].plot(
                    probe_epochs,
                    [row[loss_key] for row in gender_rows],
                    label=label,
                    linestyle=linestyle,
                    marker="o",
                    color=color,
                )
        axes[0].set_ylabel("Accuracy")
        axes[1].set_ylabel("Balanced accuracy")
        for axis in axes[:2]:
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.3)
            axis.legend()
        axes[2].set_xlabel("Pretraining epoch")
        axes[2].set_ylabel("Validation loss")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
        figure.suptitle("Gender Probe Baseline Comparison")
        _save_figure(path / "gender_probe_metrics.png", dpi=dpi, save_pdf=save_pdf)

        similarity_keys = (
            ("gender_probe_all_embedding_cosine_mean", "All held-out"),
            ("gender_probe_train_embedding_cosine_mean", "Probe train"),
            ("gender_probe_val_embedding_cosine_mean", "Probe validation"),
        )
        if any(key in row for row in gender_rows for key, _ in similarity_keys):
            figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            for key, label in similarity_keys:
                rows = [row for row in gender_rows if key in row]
                if rows:
                    axes[0].plot(
                        [row["epoch"] for row in rows],
                        [row[key] for row in rows],
                        label=label,
                        marker="o",
                    )
                std_key = key.replace("_mean", "_std")
                rows = [row for row in gender_rows if std_key in row]
                if rows:
                    axes[1].plot(
                        [row["epoch"] for row in rows],
                        [row[std_key] for row in rows],
                        label=label,
                        marker="o",
                    )
            axes[0].set_ylabel("Mean cosine")
            axes[1].set_ylabel("Cosine std")
            axes[1].set_xlabel("Pretraining epoch")
            for axis in axes:
                axis.grid(alpha=0.3)
                axis.legend()
            figure.suptitle("Gender Probe Embedding Cosine Similarity")
            _save_figure(
                path / "gender_probe_embedding_cosine.png",
                dpi=dpi,
                save_pdf=save_pdf,
            )


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
