from __future__ import annotations

import pytest
import torch

from bsjepa.linear_probe import _embedding_similarity_metrics


def test_embedding_similarity_metrics_track_probe_splits() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    metrics = _embedding_similarity_metrics(
        embeddings,
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
    )

    assert metrics["gender_probe_all_embedding_cosine_mean"] == pytest.approx(-1 / 3)
    assert metrics["gender_probe_train_embedding_cosine_mean"] == pytest.approx(0.0)
    assert metrics["gender_probe_val_embedding_cosine_mean"] == pytest.approx(0.0)
    assert "gender_probe_all_embedding_cosine_std" in metrics
