"""Final, self-contained artifact export for completed BS-JEPA runs."""

from __future__ import annotations

import csv
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch_geometric

from .model import BSJEPA
from .plotting import save_training_plots


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    fields = ["epoch"] + sorted(
        {key for row in history for key in row if key != "epoch"}
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    key: "" if _json_safe(row.get(key)) is None else row.get(key, "")
                    for key in fields
                }
            )


def _artifact_base_name(model_config: dict[str, Any], configured_name: Any) -> str:
    raw_name = configured_name or (
        f"bsjepa_{model_config['encoder_type']}_encoder_"
        f"{model_config['predictor_type']}_predictor"
    )
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_name)).strip("._-")
    if not name:
        raise ValueError("training.artifact_name must contain a valid filename character")
    return name


def _unique_artifact_path(output_dir: Path, base_name: str, timestamp: str) -> Path:
    candidate = output_dir / f"{base_name}_{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = output_dir / f"{base_name}_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def export_final_artifact(
    model: BSJEPA,
    history: list[dict[str, float]],
    config: dict[str, Any],
    *,
    effective_model_config: dict[str, Any],
    input_feature_dim: int,
    num_regions: int,
    num_rsns: int,
    total_subjects: int,
    pretraining_subjects: int,
    heldout_subjects: int,
) -> Path:
    """Save final weights, reconstruction metadata, history, and final plots."""
    if not history:
        raise ValueError("Cannot export a final artifact without training history")
    training_config = config["training"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_at = datetime.now(timezone.utc).isoformat()
    base_name = _artifact_base_name(
        effective_model_config, training_config.get("artifact_name")
    )
    artifact_path = _unique_artifact_path(Path(config["output_dir"]), base_name, timestamp)
    model_filename = f"{base_name}_final.pt"
    final_epoch = int(history[-1]["epoch"])
    final_metrics = history[-1]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    reconstruction = {
        "in_channels": input_feature_dim,
        "num_regions": num_regions,
        "model_config": effective_model_config,
    }
    checkpoint = {
        "epoch": final_epoch,
        "model": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "model_config": effective_model_config,
        "reconstruction": reconstruction,
        "masking_config": config["masking"],
        "num_regions": num_regions,
        "num_rsns": num_rsns,
        "input_feature_dim": input_feature_dim,
        "final_metrics": final_metrics,
    }
    torch.save(checkpoint, artifact_path / model_filename)

    metadata = {
        "artifact_format_version": 1,
        "created_at_utc": created_at,
        "model_filename": model_filename,
        "configuration": config,
        "architecture": effective_model_config,
        "reconstruction": reconstruction,
        "atlas": {"num_regions": num_regions, "num_rsns": num_rsns},
        "masking": config["masking"],
        "num_target_subnetworks": int(config["masking"]["num_targets"]),
        "training": training_config,
        "evaluation": config.get("evaluation", {}),
        "data": {
            **config["data"],
            "total_subjects": total_subjects,
            "pretraining_subjects": pretraining_subjects,
            "heldout_pmat_subjects": heldout_subjects,
        },
        "random_seed": int(config["seed"]),
        "parameters": {
            "total": parameter_count,
            "trainable": trainable_parameter_count,
        },
        "final_epoch": final_epoch,
        "final_metrics": final_metrics,
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    _write_json(artifact_path / "model_metadata.json", metadata)
    _write_json(artifact_path / "training_history.json", history)
    _write_history_csv(artifact_path / "training_history.csv", history)
    save_training_plots(
        history,
        artifact_path / "plots",
        dpi=int(training_config.get("publication_plot_dpi", 300)),
        save_pdf=bool(training_config.get("save_plot_pdf", True)),
    )
    return artifact_path
