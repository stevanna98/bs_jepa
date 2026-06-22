from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data

from bsjepa.evaluation import (
    extract_target_encoder_diagnostics,
    region_stage_cross_subject_diagnostics,
)
from bsjepa.model import GraphNetwork
from bsjepa.plotting import save_region_stage_diagnostic_plots


class _RegionDataset(Dataset[Data]):
    def __init__(self) -> None:
        self.subject_ids = ["a", "b", "c"]
        self.graphs = [
            self._graph([0, 1, 2], [[0.0], [100.0], [50.0]]),
            self._graph([0, 1], [[2.0], [100.0]]),
            self._graph([0], [[4.0]]),
        ]

    @staticmethod
    def _graph(region_ids: list[int], values: list[list[float]]) -> Data:
        return Data(
            x=torch.tensor(values),
            region_ids=torch.tensor(region_ids),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            num_nodes=len(region_ids),
        )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]


class _DiagnosticIdentityEncoder(nn.Module):
    def forward(self, batch: Data) -> torch.Tensor:
        return batch.x

    def forward_with_diagnostics(
        self, batch: Data
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = batch.x
        projected = torch.cat([raw, raw], dim=-1)
        position = torch.ones_like(projected)
        final = projected + position
        return final, {
            "raw": raw.detach(),
            "temporal": (2 * raw).detach(),
            "projection": projected.detach(),
            "position": position.detach(),
            "post_position": final.detach(),
            "gcn_layer_0": final.detach(),
            "final": final.detach(),
        }


class _DiagnosticIdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_encoder = _DiagnosticIdentityEncoder()

    def encode(self, batch: Data) -> torch.Tensor:
        return self.target_encoder(batch)


def test_batched_collection_aligns_same_regions_and_handles_missing_regions() -> None:
    model = _DiagnosticIdentityModel()

    embeddings, subject_ids, stages = extract_target_encoder_diagnostics(
        model,
        _RegionDataset(),
        device=torch.device("cpu"),
        batch_size=2,
        collect_region_stages=True,
    )

    assert embeddings.shape == (3, 2)
    assert subject_ids == ["a", "b", "c"]
    assert stages is not None
    assert [value.item() for value in stages["raw"][0]] == [0.0, 2.0, 4.0]
    assert [value.item() for value in stages["raw"][1]] == [100.0, 100.0]
    assert [value.item() for value in stages["raw"][2]] == [50.0]

    metrics, per_region = region_stage_cross_subject_diagnostics(stages)

    assert per_region["raw"][0] == pytest.approx(8 / 3)
    assert per_region["raw"][1] == 0
    assert 2 not in per_region["raw"]
    assert metrics["region_raw_cross_subject_variance_mean"] == pytest.approx(4 / 3)
    assert metrics["region_raw_cross_subject_variance_near_zero_fraction"] == 0.5


def test_known_feature_and_position_norms() -> None:
    stages = {
        "temporal": {0: [torch.tensor([0.0]), torch.tensor([2.0])]},
        "projection": {
            0: [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]
        },
        "position": {
            0: [torch.tensor([0.0, 5.0]), torch.tensor([0.0, 1.0])]
        },
        "post_position": {
            0: [torch.tensor([3.0, 9.0]), torch.tensor([0.0, 3.0])]
        },
    }

    metrics, _ = region_stage_cross_subject_diagnostics(stages)

    assert metrics["region_projected_feature_norm_mean"] == pytest.approx(3.5)
    assert metrics["region_position_norm_mean"] == pytest.approx(3.0)
    assert metrics["region_position_to_feature_norm_ratio"] == pytest.approx(0.75)
    assert metrics["region_position_to_feature_norm_ratio_median"] == pytest.approx(
        0.75
    )
    assert metrics["region_post_position_norm_mean"] == pytest.approx(
        (math.sqrt(90) + 3) / 2
    )


def test_zero_feature_norm_ratio_is_finite() -> None:
    stages = {
        "temporal": {0: [torch.zeros(1), torch.zeros(1)]},
        "projection": {0: [torch.zeros(2), torch.zeros(2)]},
        "position": {0: [torch.tensor([2.0, 0.0]), torch.tensor([2.0, 0.0])]},
        "post_position": {
            0: [torch.tensor([2.0, 0.0]), torch.tensor([2.0, 0.0])]
        },
    }

    metrics, _ = region_stage_cross_subject_diagnostics(
        stages, norm_epsilon=1e-3
    )

    ratio = metrics["region_position_to_feature_norm_ratio"]
    assert math.isfinite(ratio)
    assert ratio == pytest.approx(2000.0)


@pytest.mark.parametrize("kind", ["gcn", "gat"])
def test_graph_encoder_collects_every_stage_without_changing_standard_forward(
    kind: str,
) -> None:
    encoder = GraphNetwork(
        in_channels=12,
        hidden_channels=8,
        out_channels=4,
        kind=kind,
        num_layers=2,
        heads=1,
        dropout=0.0,
        num_regions=3,
        feature_mode="conv1d",
        feature_dim=6,
    ).eval()
    graph = Data(
        x=torch.randn(3, 12),
        region_ids=torch.arange(3),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        num_nodes=3,
    )

    standard_output = encoder(graph)
    diagnostic_output, stages = encoder.forward_with_diagnostics(graph)
    internal_output, ordinary_activations = encoder._forward_impl(
        graph, collect_diagnostics=False
    )

    assert torch.equal(standard_output, diagnostic_output)
    assert torch.equal(standard_output, internal_output)
    assert ordinary_activations is None
    assert set(stages) == {
        "raw",
        "temporal",
        "projection",
        "position",
        "post_position",
        f"{kind}_layer_0",
        f"{kind}_layer_1",
        "final",
    }
    assert stages["raw"].shape == (3, 12)
    assert stages["temporal"].shape == (3, 6)
    assert stages["projection"].shape == (3, 8)
    assert stages[f"{kind}_layer_0"].shape == (3, 8)
    assert stages[f"{kind}_layer_1"].shape == (3, 4)
    assert stages["final"].shape == (3, 4)
    assert all(not activation.requires_grad for activation in stages.values())
    assert not hasattr(encoder, "diagnostic_activations")


def test_region_stage_plots_created_from_synthetic_history(tmp_path) -> None:
    history = [
        {
            "epoch": 1.0,
            "region_temporal_cross_subject_variance_mean": 2.0,
            "region_projection_cross_subject_variance_mean": 1.0,
            "region_gcn_layer_0_cross_subject_variance_mean": 0.5,
            "region_final_cross_subject_variance_mean": 0.25,
            "region_projected_feature_norm_mean": 2.0,
            "region_position_norm_mean": 3.0,
            "region_post_position_norm_mean": 4.0,
            "region_position_to_feature_norm_ratio": 1.5,
            "region_temporal_variance_retention_ratio": 1.0,
            "region_projection_variance_retention_ratio": 0.5,
            "region_gcn_layer_0_variance_retention_ratio": 0.25,
            "region_final_variance_retention_ratio": 0.125,
        }
    ]
    per_region = {
        "temporal": {0: 1.0, 1: 2.0},
        "projection": {0: 0.5, 1: 1.0},
        "gcn_layer_0": {0: 0.2, 1: 0.4},
        "final": {0: 0.1, 1: 0.2},
    }

    save_region_stage_diagnostic_plots(
        history, per_region, tmp_path, epoch=1
    )

    expected = {
        "region_stage_cross_subject_variance.png",
        "region_stage_per_region_variance_heatmap.png",
        "region_stage_feature_position_norms.png",
        "region_stage_position_feature_ratio.png",
        "region_stage_variance_retention.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
