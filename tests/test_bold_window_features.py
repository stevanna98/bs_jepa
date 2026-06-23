from __future__ import annotations

import pickle

import pytest
import torch

from bsjepa.data import Atlas, BrainGraphDataset, pearson_correlation


def _dataset_path(tmp_path, bold: torch.Tensor):
    path = tmp_path / "subjects.pkl"
    with path.open("wb") as handle:
        pickle.dump({"subject-1": {"BOLD": bold.numpy()}}, handle)
    return path


def test_bold_window_slices_node_features_but_keeps_full_signal_graph(tmp_path) -> None:
    bold = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 3.0, 2.0, 5.0, 7.0],
            [2.0, 0.0, 1.0, 6.0, 8.0],
        ]
    )
    atlas = Atlas(torch.tensor([0, 1, 2]), ["a", "b", "c"])
    dataset = BrainGraphDataset(
        _dataset_path(tmp_path, bold),
        atlas,
        node_features="bold",
        graph_strategy="dense",
        bold_window_size=2,
        bold_window_start=1,
    )

    graph = dataset[0]

    expected_features = torch.tensor(
        [
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
    full_fc = pearson_correlation(
        (bold - bold.mean(1, keepdim=True))
        / bold.std(1, keepdim=True, unbiased=False).clamp_min(1e-8)
    )
    expected_adjacency = full_fc.clone()
    expected_adjacency.fill_diagonal_(0)
    observed_adjacency = torch.zeros_like(expected_adjacency)
    observed_adjacency[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr.squeeze(-1)

    assert torch.allclose(graph.x, expected_features)
    assert torch.allclose(observed_adjacency, expected_adjacency)


def test_bold_window_rejects_out_of_range_window(tmp_path) -> None:
    atlas = Atlas(torch.tensor([0, 1]), ["a", "b"])
    dataset = BrainGraphDataset(
        _dataset_path(tmp_path, torch.ones(2, 4)),
        atlas,
        bold_window_size=3,
        bold_window_start=2,
    )

    with pytest.raises(ValueError, match="requested window"):
        dataset[0]
