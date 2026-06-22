from __future__ import annotations

import matplotlib
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data

from bsjepa.evaluation import (
    extract_subject_embeddings,
    subject_similarity_diagnostics,
)
from bsjepa.plotting import save_subject_similarity_plots


class _GraphDataset(Dataset[Data]):
    def __init__(self, node_embeddings: list[torch.Tensor]) -> None:
        self.node_embeddings = node_embeddings
        self.subject_ids = [f"subject-{index}" for index in range(len(node_embeddings))]

    def __len__(self) -> int:
        return len(self.node_embeddings)

    def __getitem__(self, index: int) -> Data:
        x = self.node_embeddings[index]
        return Data(
            x=x,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            num_nodes=len(x),
        )


class _IdentityTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_training_states: list[bool] = []
        self.forward_grad_states: list[bool] = []

    def forward(self, batch: Data) -> torch.Tensor:
        self.forward_training_states.append(self.training)
        self.forward_grad_states.append(torch.is_grad_enabled())
        return batch.x


class _IdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_encoder = _IdentityTarget()

    def encode(self, batch: Data) -> torch.Tensor:
        return self.target_encoder(batch)


def test_subject_embeddings_mean_pool_nodes() -> None:
    dataset = _GraphDataset(
        [
            torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
            torch.tensor([[0.0, 2.0], [3.0, 2.0], [6.0, 2.0]]),
        ]
    )
    model = _IdentityModel()

    embeddings, subject_ids = extract_subject_embeddings(
        model, dataset, device=torch.device("cpu"), batch_size=1
    )

    assert torch.equal(embeddings, torch.tensor([[2.0, 4.0], [3.0, 2.0]]))
    assert subject_ids == dataset.subject_ids
    assert model.target_encoder.forward_training_states == [False, False]
    assert model.target_encoder.forward_grad_states == [False, False]
    assert model.training
    assert model.target_encoder.training


def test_similarity_excludes_diagonal() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    similarity, off_diagonal, metrics = subject_similarity_diagnostics(embeddings)

    expected = similarity[~torch.eye(3, dtype=torch.bool)]
    assert torch.equal(off_diagonal, expected)
    assert off_diagonal.numel() == 6
    assert not torch.any(off_diagonal == similarity.diagonal()[0])
    assert metrics["subject_cosine_similarity_mean"] == pytest.approx(-1 / 3)


@pytest.mark.parametrize(
    ("embeddings", "expected"),
    [
        (torch.tensor([[1.0, 2.0], [1.0, 2.0]]), 1.0),
        (torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 0.0),
    ],
)
def test_identical_and_orthogonal_embeddings(
    embeddings: torch.Tensor, expected: float
) -> None:
    _, off_diagonal, metrics = subject_similarity_diagnostics(embeddings)

    assert torch.allclose(off_diagonal, torch.full((2,), expected))
    for key in ("mean", "min", "max"):
        assert metrics[f"subject_cosine_similarity_{key}"] == pytest.approx(expected)
    assert metrics["subject_cosine_similarity_std"] == pytest.approx(0.0)


def test_similarity_plots_created_for_small_synthetic_dataset(tmp_path) -> None:
    dataset = _GraphDataset(
        [torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    )
    embeddings, subject_ids = extract_subject_embeddings(
        _IdentityModel(), dataset, device=torch.device("cpu"), batch_size=1
    )
    similarity, off_diagonal, _ = subject_similarity_diagnostics(embeddings)

    save_subject_similarity_plots(
        similarity, off_diagonal, subject_ids, tmp_path, epoch=3
    )

    heatmap = tmp_path / "subject_similarity_heatmap.png"
    histogram = tmp_path / "subject_similarity_histogram.png"
    assert heatmap.is_file()
    assert histogram.is_file()
    original_files = set(tmp_path.iterdir())

    save_subject_similarity_plots(
        similarity, off_diagonal, subject_ids, tmp_path, epoch=4
    )

    assert set(tmp_path.iterdir()) == original_files
    assert matplotlib.get_backend().lower() == "agg"


def test_single_subject_has_no_off_diagonal_values() -> None:
    similarity, off_diagonal, metrics = subject_similarity_diagnostics(
        torch.tensor([[1.0, 0.0]])
    )

    assert similarity.shape == (1, 1)
    assert off_diagonal.numel() == 0
    assert all(torch.isnan(torch.tensor(value)) for value in metrics.values())
