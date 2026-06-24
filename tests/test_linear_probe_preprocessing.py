from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data

from bsjepa.evaluation import LabeledGraphDataset
from bsjepa.linear_probe import evaluate_gender_probe


class _ProbeDataset(Dataset[Data]):
    def __init__(self) -> None:
        self.graphs = [
            self._graph([0.0, 0.0]),
            self._graph([0.2, 0.1]),
            self._graph([0.4, 0.2]),
            self._graph([0.6, 0.3]),
            self._graph([2.0, 2.0]),
            self._graph([2.2, 2.1]),
            self._graph([2.4, 2.2]),
            self._graph([2.6, 2.3]),
        ]

    @staticmethod
    def _graph(values: list[float]) -> Data:
        x = torch.tensor([values, values], dtype=torch.float32)
        return Data(
            x=x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            num_nodes=2,
        )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]


class _IdentityTarget(nn.Module):
    def forward(self, batch: Data) -> torch.Tensor:
        return batch.x


class _IdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_encoder = _IdentityTarget()

    def encode(self, batch: Data) -> torch.Tensor:
        return self.target_encoder(batch)


def test_gender_probe_logs_preprocessing_variant_and_legacy_raw_metrics() -> None:
    dataset = _ProbeDataset()
    labeled = LabeledGraphDataset(
        dataset,
        list(range(len(dataset))),
        [0, 0, 0, 0, 1, 1, 1, 1],
        [f"subject-{index}" for index in range(len(dataset))],
        label_dtype=torch.long,
    )

    metrics = evaluate_gender_probe(
        _IdentityModel(),
        labeled,
        {
            "probe_epochs": 1,
            "probe_lr": 0.01,
            "probe_train_fraction": 0.5,
            "probe_weight_decay": 0.0,
            "batch_size": 4,
            "random_seed": 3,
            "compare_baselines": False,
            "embedding_preprocessing": {
                "variants": [
                    "raw",
                    "centered",
                    "standardized",
                    "centered_pc_removed",
                    "standardized_pc_removed",
                ],
                "pc_remove_components": [1],
                "standardize_epsilon": 1e-6,
            },
        },
        device=torch.device("cpu"),
    )

    assert metrics["gender_probe_val_accuracy"] == metrics[
        "gender_probe_raw_val_accuracy"
    ]
    assert metrics["gender_probe_val_balanced_accuracy"] == metrics[
        "gender_probe_raw_val_balanced_accuracy"
    ]
    assert "gender_probe_centered_val_accuracy" in metrics
    assert "gender_probe_standardized_val_accuracy" in metrics
    assert "gender_probe_centered_pc1_val_accuracy" in metrics
    assert "gender_probe_standardized_pc1_val_accuracy" in metrics
    assert "gender_probe_raw_all_embedding_cosine_mean" in metrics
    assert "gender_probe_centered_all_embedding_cosine_mean" in metrics
    assert "gender_probe_centered_pc1_train_embedding_cosine_std" in metrics
    assert "gender_probe_all_embedding_cosine_mean" in metrics
