from __future__ import annotations

import pytest
import torch

from bsjepa.data import build_graph


def _adjacency(graph, node_count: int) -> torch.Tensor:
    result = torch.zeros(node_count, node_count)
    result[graph.edge_index[0], graph.edge_index[1]] = graph.edge_attr.squeeze(-1)
    return result


def test_positive_proportional_keeps_strongest_fraction_of_positive_edges() -> None:
    fc = torch.tensor(
        [
            [1.0, 0.9, -0.95, 0.2],
            [0.9, 1.0, 0.8, -0.4],
            [-0.95, 0.8, 1.0, 0.5],
            [0.2, -0.4, 0.5, 1.0],
        ]
    )

    graph = build_graph(fc, strategy="positive_proportional", threshold=0.5)
    adjacency = _adjacency(graph, 4)

    # Four positive undirected candidates yield two retained edges (four arcs).
    assert graph.edge_index.shape[1] == 4
    assert torch.equal(adjacency, adjacency.T)
    assert torch.all(graph.edge_attr > 0)
    assert adjacency[0, 1] == pytest.approx(0.9)
    assert adjacency[1, 2] == pytest.approx(0.8)
    assert adjacency[2, 3] == 0
    assert adjacency[0, 2] == 0


def test_positive_proportional_rounds_up_and_handles_no_positive_edges() -> None:
    positive_fc = torch.tensor(
        [[1.0, 0.5, 0.4], [0.5, 1.0, 0.3], [0.4, 0.3, 1.0]]
    )
    graph = build_graph(
        positive_fc, strategy="positive_proportional", threshold=0.01
    )
    assert graph.edge_index.shape[1] == 2

    non_positive_fc = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
    empty_graph = build_graph(
        non_positive_fc, strategy="positive_proportional", threshold=0.5
    )
    assert empty_graph.edge_index.shape == (2, 0)
    assert empty_graph.edge_attr.shape == (0, 1)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
def test_positive_proportional_rejects_invalid_proportions(threshold: float) -> None:
    with pytest.raises(ValueError, match="interval"):
        build_graph(
            torch.eye(2),
            strategy="positive_proportional",
            threshold=threshold,
        )
