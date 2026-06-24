from __future__ import annotations

import pytest
import torch

from bsjepa.embedding_preprocessing import (
    EmbeddingPreprocessor,
    build_preprocessing_specs,
)


def test_raw_preprocessing_returns_embeddings_unchanged() -> None:
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    transformed = EmbeddingPreprocessor("raw").fit_transform(embeddings)

    assert torch.equal(transformed, embeddings)


def test_centered_subtracts_training_mean() -> None:
    train = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

    transformed = EmbeddingPreprocessor("centered").fit_transform(train)

    assert torch.allclose(transformed.mean(0), torch.zeros(2))
    assert torch.allclose(transformed, torch.tensor([[-1.0, -2.0], [1.0, 2.0]]))


def test_standardized_train_features_have_zero_mean_and_unit_variance() -> None:
    train = torch.tensor([[1.0, 2.0, 5.0], [3.0, 6.0, 5.0]])

    transformed = EmbeddingPreprocessor("standardized").fit_transform(train)

    assert torch.allclose(transformed.mean(0), torch.zeros(3))
    assert torch.allclose(transformed[:, :2].std(0, unbiased=False), torch.ones(2))
    assert torch.count_nonzero(transformed[:, 2]) == 0


def test_pc_removal_reduces_projection_on_removed_component() -> None:
    train = torch.tensor([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
    preprocessor = EmbeddingPreprocessor("centered_pc_removed", pc_components=1)

    transformed = preprocessor.fit_transform(train)

    assert preprocessor.components_ is not None
    projection = transformed @ preprocessor.components_.T
    assert torch.allclose(projection, torch.zeros_like(projection), atol=1e-6)


def test_validation_transform_uses_training_statistics_only() -> None:
    train = torch.tensor([[10.0], [12.0]])
    validation = torch.tensor([[14.0]])
    preprocessor = EmbeddingPreprocessor("centered").fit(train)

    transformed = preprocessor.transform(validation)

    assert torch.equal(transformed, torch.tensor([[3.0]]))


def test_pc_removal_k_zero_behaves_like_base_preprocessing() -> None:
    train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    centered = EmbeddingPreprocessor("centered").fit_transform(train)
    removed = EmbeddingPreprocessor(
        "centered_pc_removed", pc_components=0
    ).fit_transform(train)

    assert torch.equal(removed, centered)


def test_excessive_pc_count_clamps_with_warning() -> None:
    train = torch.randn(2, 3)
    preprocessor = EmbeddingPreprocessor("standardized_pc_removed", pc_components=10)

    with pytest.warns(UserWarning, match="clamping"):
        preprocessor.fit(train)

    assert preprocessor.fitted_pc_components_ == 2


def test_build_preprocessing_specs_expands_pc_variants() -> None:
    specs = build_preprocessing_specs(
        {
            "variants": ["raw", "centered_pc_removed", "standardized_pc_removed"],
            "pc_remove_components": [1, 3],
        }
    )

    assert [spec.metric_suffix for spec in specs] == [
        "raw",
        "centered_pc1",
        "centered_pc3",
        "standardized_pc1",
        "standardized_pc3",
    ]
