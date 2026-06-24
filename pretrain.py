#!/usr/bin/env python
"""Command-line entry point for minimal BS-JEPA pretraining."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bsjepa import (
    BrainGraphDataset,
    SubnetworkMaskCollator,
    SyntheticBrainDataset,
    build_bsjepa,
    load_atlas,
    pretrain,
    split_pmat_holdout,
)
from bsjepa.data import synthetic_atlas
from bsjepa.linear_probe import split_gender_probe_holdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain BS-JEPA")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="Override a YAML value; may be repeated",
    )
    return parser.parse_args()


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    for override in overrides:
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    if data_config["source"] == "synthetic":
        atlas = synthetic_atlas(
            int(data_config["num_regions"]), int(data_config["num_rsns"])
        )
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
    evaluation_config = config.get("evaluation", {})
    evaluation_dataset = None
    training_dataset = dataset
    if bool(evaluation_config.get("enabled", False)):
        training_dataset, evaluation_dataset = split_pmat_holdout(
            dataset, evaluation_config
        )
        print(
            f"PMAT split: pretraining_subjects={len(training_dataset)} "
            f"heldout_subjects={len(evaluation_dataset)}"
        )
    linear_probe_config = config.get("linear_probe", {})
    linear_probe_dataset = None
    if bool(linear_probe_config.get("enabled", False)):
        training_dataset, linear_probe_dataset = split_gender_probe_holdout(
            training_dataset, linear_probe_config
        )
        print(
            f"Gender probe split: pretraining_subjects={len(training_dataset)} "
            f"heldout_subjects={len(linear_probe_dataset)} "
            f"female={int((linear_probe_dataset.labels == 0).sum())} "
            f"male={int((linear_probe_dataset.labels == 1).sum())}"
        )
    loader = DataLoader(
        training_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        collate_fn=list,
        drop_last=False,
    )
    sample = training_dataset[0]
    model_config = dict(config["model"])
    if data_config["source"] == "synthetic":
        model_config["feature_mode"] = "passthrough"
    model = build_bsjepa(
        in_channels=sample.x.shape[1],
        num_regions=atlas.num_regions,
        num_rsns=atlas.num_rsns,
        **model_config,
    )
    random_probe_model = (
        copy.deepcopy(model)
        if linear_probe_dataset is not None
        and bool(linear_probe_config.get("compare_baselines", True))
        else None
    )
    collator = SubnetworkMaskCollator(
        atlas.num_rsns, int(config["masking"]["num_targets"])
    )
    print(
        f"device={device} subjects={len(training_dataset)} regions={atlas.num_regions} "
        f"trainable_parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    history = pretrain(
        model,
        loader,
        collator,
        config["training"],
        device=device,
        output_dir=config["output_dir"],
        evaluation_dataset=evaluation_dataset,
        evaluation_config=evaluation_config if evaluation_dataset is not None else None,
        linear_probe_dataset=linear_probe_dataset,
        linear_probe_config=(
            linear_probe_config if linear_probe_dataset is not None else None
        ),
        random_probe_model=random_probe_model,
        rsn_names=atlas.rsn_names,
    )
    if bool(config["training"].get("save_final_artifact", True)):
        from bsjepa.artifacts import export_final_artifact

        artifact_path = export_final_artifact(
            model,
            history,
            config,
            effective_model_config=model_config,
            input_feature_dim=int(sample.x.shape[1]),
            num_regions=atlas.num_regions,
            num_rsns=atlas.num_rsns,
            total_subjects=len(dataset),
            pretraining_subjects=len(training_dataset),
            heldout_subjects=(
                len(evaluation_dataset) if evaluation_dataset is not None else 0
            ),
            gender_probe_heldout_subjects=(
                len(linear_probe_dataset) if linear_probe_dataset is not None else 0
            ),
        )
        print(f"final_artifact={artifact_path}", flush=True)


if __name__ == "__main__":
    main()
