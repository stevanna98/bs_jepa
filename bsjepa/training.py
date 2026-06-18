"""Minimal BS-JEPA optimization loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluation import LabeledGraphDataset, evaluate_pmat
from .linear_probe import evaluate_gender_probe
from .losses import (
    jepa_loss,
    per_rsn_prediction_losses,
    representation_diagnostics,
    rsn_diversity_loss,
    subject_specificity_diagnostics,
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
    prediction_loss = str(config.get("prediction_loss", "cosine"))
    save_plots = bool(config.get("save_plots", True))
    plot_frequency = int(config.get("plot_frequency", 1))
    collapse_metrics = bool(config.get("collapse_metrics", True))
    subject_specificity_metrics = bool(
        config.get("subject_specificity_metrics", True)
    )
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
            outputs = model(
                batch,
                masks,
                return_groups=True,
                return_metadata=subject_specificity_metrics,
            )
            predictions, targets, context = outputs[:3]
            loss, metrics = jepa_loss(
                predictions,
                targets,
                context,
                prediction_loss=prediction_loss,
                prediction_variance_weight=float(config["prediction_variance_weight"]),
                context_variance_weight=float(config["context_variance_weight"]),
                covariance_weight=float(config["covariance_weight"]),
                target_std=float(config["target_std"]),
            )
            if collapse_metrics:
                metrics.update(representation_diagnostics(predictions, targets, context))
            row_group_ids, group_rsn_ids = outputs[3], outputs[4]
            if subject_specificity_metrics:
                metrics.update(
                    subject_specificity_diagnostics(
                        predictions,
                        outputs[5],
                        outputs[6],
                        context,
                        outputs[7],
                        outputs[8],
                    )
                )
            rsn_losses = per_rsn_prediction_losses(
                predictions,
                targets,
                row_group_ids,
                group_rsn_ids,
                prediction_loss=prediction_loss,
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
            or (
                plot_frequency > 0
                and (epoch % plot_frequency == 0 or epoch == epochs)
            )
        ):
            from .plotting import save_training_plots

            save_training_plots(history, output_path / "plots")
    return history
