"""Minimal BS-JEPA optimization loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import (
    LabeledGraphDataset,
    cohort_centered_cosine_diagnostics,
    evaluate_pmat,
    extract_subject_embeddings,
    extract_target_encoder_diagnostics,
    region_stage_cross_subject_diagnostics,
    standardized_euclidean_diagnostics,
    subject_similarity_diagnostics,
    subject_variance_rank_diagnostics,
)
from .linear_probe import evaluate_gender_probe
from .losses import (
    jepa_loss,
    per_rsn_prediction_losses,
    representation_diagnostics,
    rsn_diversity_loss,
)
from .masking import SubnetworkMaskCollator
from .model import BSJEPA


def build_optimizer(model: nn.Module, *, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Build AdamW with bias and normalization parameters excluded from decay."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(
            parameter
        )
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0, "no_weight_decay": True},
        ],
        lr=lr,
    )


@torch.no_grad()
def update_target_encoder(model: BSJEPA, momentum: float) -> None:
    for context, target in zip(
        model.context_encoder.parameters(), model.target_encoder.parameters(), strict=True
    ):
        target.mul_(momentum).add_(context, alpha=1 - momentum)


def _cosine_value(start: float, end: float, progress: float) -> float:
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def pretrain(
    model: BSJEPA,
    loader: DataLoader[list],
    mask_collator: SubnetworkMaskCollator,
    config: dict[str, Any],
    *,
    device: torch.device,
    output_dir: str | Path,
    evaluation_dataset: LabeledGraphDataset | None = None,
    evaluation_config: dict[str, Any] | None = None,
    linear_probe_dataset: LabeledGraphDataset | None = None,
    linear_probe_config: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Run BS-JEPA pretraining and write checkpoints and diagnostic plots."""
    epochs = int(config["epochs"])
    total_steps = max(epochs * len(loader), 1)
    warmup_steps = int(config.get("warmup_epochs", 0)) * len(loader)
    optimizer = build_optimizer(
        model, lr=float(config["lr"]), weight_decay=float(config["weight_decay_start"])
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_frequency = int(config.get("checkpoint_frequency", 1))
    diversity_weight = float(config.get("rsn_diversity_weight", 0.0))
    save_plots = bool(config.get("save_plots", True))
    plot_frequency = int(config.get("plot_frequency", 1))
    subject_similarity_enabled = bool(
        config.get("subject_similarity_diagnostics", False)
    )
    extended_subject_diagnostics = bool(
        config.get("subject_extended_diagnostics", False)
    )
    subject_diagnostics = subject_similarity_enabled or extended_subject_diagnostics
    subject_embedding_batch_size = int(
        config.get("subject_embedding_batch_size", loader.batch_size or 1)
    )
    if subject_embedding_batch_size < 1:
        raise ValueError("training.subject_embedding_batch_size must be positive")
    centered_cosine_epsilon = float(
        config.get("subject_centered_cosine_epsilon", 1e-12)
    )
    near_zero_variance_threshold = float(
        config.get("subject_near_zero_variance_threshold", 1e-8)
    )
    standardization_epsilon = float(
        config.get("subject_standardization_epsilon", 1e-6)
    )
    if centered_cosine_epsilon <= 0 or standardization_epsilon <= 0:
        raise ValueError("Subject diagnostic epsilons must be positive")
    if near_zero_variance_threshold < 0:
        raise ValueError("Subject near-zero variance threshold must be non-negative")
    region_stage_enabled = bool(config.get("region_stage_diagnostics", False))
    region_stage_frequency = int(
        config.get("region_stage_diagnostics_frequency", plot_frequency)
    )
    region_stage_batch_size = int(
        config.get("region_stage_diagnostics_batch_size", loader.batch_size or 1)
    )
    region_stage_near_zero_threshold = float(
        config.get("region_stage_near_zero_threshold", 1e-8)
    )
    region_stage_norm_epsilon = float(
        config.get("region_stage_norm_epsilon", 1e-12)
    )
    if region_stage_enabled and region_stage_frequency <= 0:
        raise ValueError("training.region_stage_diagnostics_frequency must be positive")
    if region_stage_batch_size < 1:
        raise ValueError("training.region_stage_diagnostics_batch_size must be positive")
    if region_stage_near_zero_threshold < 0 or region_stage_norm_epsilon <= 0:
        raise ValueError("Invalid region-stage diagnostic numerical threshold")
    collapse_metrics = bool(config.get("collapse_metrics", True))
    evaluation_frequency = (
        int(evaluation_config["frequency_epochs"])
        if evaluation_config is not None and evaluation_dataset is not None
        else 0
    )
    if evaluation_dataset is not None and evaluation_frequency <= 0:
        raise ValueError("evaluation.frequency_epochs must be positive")
    linear_probe_frequency = (
        int(linear_probe_config["eval_frequency_epochs"])
        if linear_probe_config is not None and linear_probe_dataset is not None
        else 0
    )
    if linear_probe_dataset is not None and linear_probe_frequency <= 0:
        raise ValueError("linear_probe.eval_frequency_epochs must be positive")
    history: list[dict[str, float]] = []
    global_step = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        rsn_loss_sums: dict[int, float] = {}
        rsn_row_counts: dict[int, int] = {}
        batch_count = 0
        for raw_graphs in loader:
            batch, masks = mask_collator(raw_graphs)
            batch, masks = batch.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, masks, return_groups=True)
            predictions, targets, context = outputs[:3]
            loss, metrics = jepa_loss(
                predictions,
                targets,
                context,
                prediction_variance_weight=float(config["prediction_variance_weight"]),
                context_variance_weight=float(config["context_variance_weight"]),
                covariance_weight=float(config["covariance_weight"]),
                target_std=float(config["target_std"]),
            )
            if collapse_metrics:
                metrics.update(representation_diagnostics(predictions, targets, context))
            row_group_ids, group_rsn_ids = outputs[3], outputs[4]
            rsn_losses = per_rsn_prediction_losses(
                predictions, targets, row_group_ids, group_rsn_ids
            )
            for rsn_id, (loss_sum, row_count) in rsn_losses.items():
                rsn_loss_sums[rsn_id] = rsn_loss_sums.get(rsn_id, 0.0) + loss_sum
                rsn_row_counts[rsn_id] = rsn_row_counts.get(rsn_id, 0) + row_count
            if diversity_weight > 0:
                diversity = rsn_diversity_loss(
                    predictions, row_group_ids, group_rsn_ids
                )
                # loss = loss + diversity_weight * diversity
                loss = loss
                metrics["rsn_diversity"] = diversity.item()
            loss.backward()
            clip_grad = config.get("clip_grad")
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optimizer.step()

            global_step += 1
            if warmup_steps and global_step <= warmup_steps:
                learning_rate = float(config["start_lr"]) + (
                    float(config["lr"]) - float(config["start_lr"])
                ) * global_step / warmup_steps
            else:
                cosine_progress = (global_step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                learning_rate = _cosine_value(
                    float(config["lr"]), float(config["final_lr"]), cosine_progress
                )
            weight_decay = float(config["weight_decay_start"]) + (
                float(config["weight_decay_end"]) - float(config["weight_decay_start"])
            ) * global_step / total_steps
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                if not group.get("no_weight_decay", False):
                    group["weight_decay"] = weight_decay
            momentum = float(config["ema_start"]) + (
                float(config["ema_end"]) - float(config["ema_start"])
            ) * global_step / total_steps
            update_target_encoder(model, momentum)

            batch_count += 1
            totals["loss"] = totals.get("loss", 0.0) + loss.item()
            metric_counts["loss"] = metric_counts.get("loss", 0) + 1
            for name, value in metrics.items():
                if not math.isfinite(value):
                    continue
                totals[name] = totals.get(name, 0.0) + value
                metric_counts[name] = metric_counts.get(name, 0) + 1

        epoch_metrics = {
            "epoch": float(epoch),
            **{
                name: value / max(metric_counts.get(name, batch_count), 1)
                for name, value in totals.items()
            },
            **{
                f"rsn_loss_{rsn_id}": loss_sum / rsn_row_counts[rsn_id]
                for rsn_id, loss_sum in sorted(rsn_loss_sums.items())
                if rsn_row_counts[rsn_id] > 0
            },
            "learning_rate": learning_rate,
            "ema_momentum": momentum,
        }
        plot_interval = save_plots and plot_frequency > 0 and (
            epoch % plot_frequency == 0 or epoch == epochs
        )
        similarity_plot_data = None
        extended_plot_data = None
        region_plot_data = None
        subject_due = subject_diagnostics and plot_interval
        region_due = region_stage_enabled and (
            epoch % region_stage_frequency == 0 or epoch == epochs
        )
        subject_embeddings = None
        subject_ids = None
        region_stages = None
        if subject_due and region_due:
            subject_embeddings, subject_ids, region_stages = (
                extract_target_encoder_diagnostics(
                    model,
                    loader.dataset,
                    device=device,
                    batch_size=min(
                        subject_embedding_batch_size, region_stage_batch_size
                    ),
                    collect_region_stages=True,
                )
            )
        elif subject_due:
            subject_embeddings, subject_ids = extract_subject_embeddings(
                model,
                loader.dataset,
                device=device,
                batch_size=subject_embedding_batch_size,
            )
        elif region_due:
            _, _, region_stages = extract_target_encoder_diagnostics(
                model,
                loader.dataset,
                device=device,
                batch_size=region_stage_batch_size,
                collect_region_stages=True,
            )
        if subject_due:
            if subject_embeddings is None or subject_ids is None:
                raise RuntimeError("Subject diagnostic extraction did not return data")
            if subject_similarity_enabled:
                similarity, off_diagonal, similarity_metrics = (
                    subject_similarity_diagnostics(subject_embeddings)
                )
                epoch_metrics.update(similarity_metrics)
                similarity_plot_data = (
                    similarity,
                    off_diagonal,
                    subject_ids,
                )
            if extended_subject_diagnostics:
                centered_similarity, centered_off_diagonal, centered_metrics = (
                    cohort_centered_cosine_diagnostics(
                        subject_embeddings,
                        epsilon=centered_cosine_epsilon,
                    )
                )
                feature_variances, explained_variance, variance_rank_metrics = (
                    subject_variance_rank_diagnostics(
                        subject_embeddings,
                        near_zero_threshold=near_zero_variance_threshold,
                    )
                )
                distances, distance_off_diagonal, distance_metrics = (
                    standardized_euclidean_diagnostics(
                        subject_embeddings,
                        epsilon=standardization_epsilon,
                        near_zero_threshold=near_zero_variance_threshold,
                    )
                )
                epoch_metrics.update(
                    {
                        **centered_metrics,
                        **variance_rank_metrics,
                        **distance_metrics,
                    }
                )
                extended_plot_data = (
                    centered_similarity,
                    centered_off_diagonal,
                    feature_variances,
                    explained_variance,
                    distances,
                    distance_off_diagonal,
                    subject_ids,
                )
        if region_due:
            if region_stages is None:
                raise RuntimeError("Region-stage diagnostic extraction did not return data")
            region_metrics, per_region_variances = (
                region_stage_cross_subject_diagnostics(
                    region_stages,
                    near_zero_threshold=region_stage_near_zero_threshold,
                    norm_epsilon=region_stage_norm_epsilon,
                )
            )
            epoch_metrics.update(region_metrics)
            region_plot_data = per_region_variances
        downstream_evaluated = False
        if evaluation_frequency > 0 and (
            epoch % evaluation_frequency == 0 or epoch == epochs
        ):
            epoch_metrics.update(
                evaluate_pmat(
                    model,
                    evaluation_dataset,
                    evaluation_config,
                    device=device,
                )
            )
            downstream_evaluated = True
        if linear_probe_frequency > 0 and (
            epoch % linear_probe_frequency == 0 or epoch == epochs
        ):
            epoch_metrics.update(
                evaluate_gender_probe(
                    model,
                    linear_probe_dataset,
                    linear_probe_config,
                    device=device,
                )
            )
            downstream_evaluated = True
        history.append(epoch_metrics)
        summary = " ".join(f"{key}={value:.5g}" for key, value in epoch_metrics.items())
        print(summary, flush=True)
        if checkpoint_frequency > 0 and (
            epoch % checkpoint_frequency == 0 or epoch == epochs
        ):
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": epoch_metrics,
                    "history": history,
                },
                output_path / f"checkpoint_{epoch:04d}.pt",
            )
        if save_plots and (
            downstream_evaluated
            or plot_interval
            or region_due
        ):
            from .plotting import (
                save_extended_subject_diagnostic_plots,
                save_region_stage_diagnostic_plots,
                save_subject_similarity_plots,
                save_training_plots,
            )

            save_training_plots(history, output_path / "plots")
            if similarity_plot_data is not None:
                save_subject_similarity_plots(
                    *similarity_plot_data,
                    output_path / "plots",
                    epoch=epoch,
                    dpi=int(config.get("publication_plot_dpi", 150)),
                    save_pdf=bool(config.get("save_plot_pdf", False)),
                    max_tick_labels=int(config.get("subject_similarity_max_ticks", 40)),
                    histogram_bins=int(config.get("subject_similarity_histogram_bins", 30)),
                )
            if region_plot_data is not None:
                save_region_stage_diagnostic_plots(
                    history,
                    region_plot_data,
                    output_path / "plots",
                    epoch=epoch,
                    dpi=int(config.get("publication_plot_dpi", 150)),
                    save_pdf=bool(config.get("save_plot_pdf", False)),
                )
            if extended_plot_data is not None:
                save_extended_subject_diagnostic_plots(
                    *extended_plot_data,
                    output_path / "plots",
                    epoch=epoch,
                    dpi=int(config.get("publication_plot_dpi", 150)),
                    save_pdf=bool(config.get("save_plot_pdf", False)),
                    max_tick_labels=int(config.get("subject_similarity_max_ticks", 40)),
                    histogram_bins=int(config.get("subject_similarity_histogram_bins", 30)),
                )
    return history
