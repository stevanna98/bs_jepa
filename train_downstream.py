#!/usr/bin/env python
"""Train BS-JEPA end-to-end with a supervised gender objective."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from bsjepa import BrainGraphDataset, SyntheticBrainDataset, build_bsjepa, load_atlas
from bsjepa.data import load_gradient_features, synthetic_atlas
from bsjepa.downstream import GenderClassifier, split_gender_dataset, train_supervised_gender
from bsjepa.masking import SubnetworkMaskCollator
from bsjepa.model import resolve_positional_encoding_config
from pretrain import load_config, parse_args


def main() -> None:
    args = parse_args("Train BS-JEPA end-to-end for gender classification")
    config = load_config(args.config, args.set)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    masking_config = config["masking"]
    downstream_config = config["downstream"]
    strategy = str(masking_config.get("strategy", "atlas_rsn"))
    num_subnetworks = int(masking_config.get("num_subnetworks", data_config["num_rsns"]))
    options = {
        "subnetwork_strategy": strategy,
        "num_subnetworks": num_subnetworks,
        "subnetwork_seed": int(masking_config.get("random_seed", seed)),
        "community_method": str(masking_config.get("community_method", "fc_kmeans")),
    }
    if data_config["source"] == "synthetic":
        atlas = synthetic_atlas(int(data_config["num_regions"]), int(data_config["num_rsns"]))
        dataset = SyntheticBrainDataset(
            atlas,
            int(data_config["num_subjects"]),
            int(data_config["feature_dim"]),
            top_k=int(data_config["top_k"]),
            seed=seed,
            **options,
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
            graph_strategy=data_config["graph_strategy"],
            top_k=int(data_config["top_k"]),
            threshold=float(data_config["threshold"]),
            gradient_key=data_config.get("gradient_key"),
            **options,
        )
    training_dataset, validation_dataset = split_gender_dataset(dataset, downstream_config)
    sample, _ = training_dataset[0]
    model_config = dict(config["model"])
    if data_config["source"] == "synthetic":
        model_config["feature_mode"] = "passthrough"
    positional_config = resolve_positional_encoding_config(
        model_config.get("positional_encoding"),
        model_config.get("region_positional_encoding"),
    )
    fixed_gradients = None
    if positional_config["type"] == "fixed_gradient":
        gradient_file = positional_config.get("gradient_file")
        if not gradient_file:
            raise ValueError("fixed_gradient requires model.positional_encoding.gradient_file")
        fixed_gradients = load_gradient_features(
            gradient_file,
            atlas.num_regions,
            gradient_columns=positional_config.get("gradient_columns"),
            region_column=positional_config.get("region_column"),
        )
        positional_config["gradient_dim"] = fixed_gradients.shape[1]
    elif positional_config["type"] == "subject_gradient":
        positional_features = getattr(sample, "positional_features", None)
        if positional_features is None:
            raise ValueError("subject_gradient requires data.gradient_key")
        positional_config["gradient_dim"] = positional_features.shape[1]
    model_config["positional_encoding"] = positional_config
    model = build_bsjepa(
        in_channels=sample.x.shape[1],
        num_regions=atlas.num_regions,
        fixed_gradient_features=fixed_gradients,
        **model_config,
    )
    classifier = GenderClassifier(
        int(model_config["embed_dim"]), str(downstream_config.get("pooling", "mean"))
    )
    mask_collator = SubnetworkMaskCollator(
        num_subnetworks, int(masking_config["num_targets"])
    )
    female_train = int((training_dataset.labels == 0).sum())
    male_train = int((training_dataset.labels == 1).sum())
    female_val = int((validation_dataset.labels == 0).sum())
    male_val = int((validation_dataset.labels == 1).sum())
    print(
        f"device={device} train={len(training_dataset)} validation={len(validation_dataset)} "
        f"train_female={female_train} train_male={male_train} "
        f"val_female={female_val} val_male={male_val} "
        f"pooling={classifier.pooling}",
        flush=True,
    )
    train_supervised_gender(
        model,
        classifier,
        training_dataset,
        validation_dataset,
        mask_collator,
        downstream_config,
        full_config=config,
        device=device,
        output_dir=config["output_dir"],
    )


if __name__ == "__main__":
    main()
