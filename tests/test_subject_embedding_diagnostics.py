from __future__ import annotations

import math

import pytest
import torch

from bsjepa.evaluation import (
    cohort_centered_cosine_diagnostics,
    standardized_euclidean_diagnostics,
    subject_variance_rank_diagnostics,
)
from bsjepa.plotting import save_extended_subject_diagnostic_plots


def test_cohort_centering_and_zero_norm_rows() -> None:
    embeddings = torch.tensor([[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])

    similarity, off_diagonal, metrics = cohort_centered_cosine_diagnostics(
        embeddings
    )

    assert torch.allclose(
        similarity,
        torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]),
    )
    assert torch.equal(off_diagonal, similarity[~torch.eye(3, dtype=torch.bool)])
    assert off_diagonal.numel() == 6
    assert metrics["subject_centered_cosine_mean"] == pytest.approx(-1 / 3)


def test_identical_embeddings_are_handled_as_degenerate() -> None:
    embeddings = torch.tensor([[3.0, -2.0], [3.0, -2.0], [3.0, -2.0]])

    _, centered_values, centered_metrics = cohort_centered_cosine_diagnostics(
        embeddings
    )
    variances, spectrum, rank_metrics = subject_variance_rank_diagnostics(
        embeddings
    )
    distances, distance_values, distance_metrics = (
        standardized_euclidean_diagnostics(embeddings)
    )

    assert torch.count_nonzero(centered_values) == 0
    assert centered_metrics["subject_centered_cosine_mean"] == 0
    assert torch.count_nonzero(variances) == 0
    assert torch.count_nonzero(spectrum) == 0
    assert rank_metrics["subject_effective_rank"] == 0
    assert rank_metrics["subject_matrix_rank"] == 0
    assert rank_metrics["subject_components_90pct"] == 0
    assert torch.count_nonzero(distances) == 0
    assert torch.count_nonzero(distance_values) == 0
    assert distance_metrics["subject_standardized_distance_max"] == 0


def test_feature_variance_and_zero_variance_fraction() -> None:
    embeddings = torch.tensor([[0.0, 5.0, 2.0], [2.0, 5.0, 4.0]])

    variances, _, metrics = subject_variance_rank_diagnostics(
        embeddings, near_zero_threshold=1e-9
    )

    assert torch.allclose(variances, torch.tensor([1.0, 0.0, 1.0]))
    assert metrics["subject_feature_variance_mean"] == pytest.approx(2 / 3)
    assert metrics["subject_feature_variance_median"] == pytest.approx(1.0)
    assert metrics["subject_feature_variance_min"] == 0
    assert metrics["subject_feature_variance_max"] == 1
    assert metrics["subject_feature_near_zero_fraction"] == pytest.approx(1 / 3)


def test_effective_rank_for_rank_one_matrix() -> None:
    embeddings = torch.tensor([[-1.0, -2.0], [0.0, 0.0], [1.0, 2.0]])

    _, spectrum, metrics = subject_variance_rank_diagnostics(embeddings)

    assert spectrum.sum() == pytest.approx(1.0)
    assert metrics["subject_effective_rank"] == pytest.approx(1.0)
    assert metrics["subject_matrix_rank"] == 1
    assert metrics["subject_largest_singular_energy_fraction"] == pytest.approx(1.0)
    assert metrics["subject_components_90pct"] == 1


def test_effective_rank_for_full_rank_equal_energy_matrix() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )

    _, spectrum, metrics = subject_variance_rank_diagnostics(embeddings)

    assert torch.allclose(spectrum, torch.tensor([0.5, 0.5]))
    assert metrics["subject_effective_rank"] == pytest.approx(2.0)
    assert metrics["subject_matrix_rank"] == 2
    assert metrics["subject_largest_singular_energy_fraction"] == pytest.approx(0.5)
    assert metrics["subject_components_90pct"] == 2


def test_standardized_euclidean_distance_neutralizes_constant_features() -> None:
    embeddings = torch.tensor([[0.0, 1000.0], [2.0, 1000.0]])

    distances, off_diagonal, metrics = standardized_euclidean_diagnostics(
        embeddings, epsilon=1e-6, near_zero_threshold=1e-8
    )

    assert torch.allclose(distances, torch.tensor([[0.0, 2.0], [2.0, 0.0]]))
    assert torch.equal(off_diagonal, torch.tensor([2.0, 2.0]))
    assert metrics["subject_standardized_distance_mean"] == pytest.approx(2.0)
    assert metrics["subject_standardized_distance_std"] == pytest.approx(0.0)


def test_clearly_separated_embeddings_have_nonzero_distances() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])

    distances, off_diagonal, _ = standardized_euclidean_diagnostics(embeddings)

    assert distances[0, 2] == pytest.approx(math.sqrt(6))
    assert off_diagonal.min() > 0


def test_extended_plots_created_for_small_synthetic_embeddings(tmp_path) -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [-1.0, 0.0, 2.0]]
    )
    subject_ids = ["a", "b", "c"]
    centered, centered_off_diagonal, _ = cohort_centered_cosine_diagnostics(
        embeddings
    )
    variances, spectrum, _ = subject_variance_rank_diagnostics(embeddings)
    distances, distance_off_diagonal, _ = standardized_euclidean_diagnostics(
        embeddings
    )

    save_extended_subject_diagnostic_plots(
        centered,
        centered_off_diagonal,
        variances,
        spectrum,
        distances,
        distance_off_diagonal,
        subject_ids,
        tmp_path,
        epoch=2,
    )

    expected = {
        "subject_centered_cosine_heatmap.png",
        "subject_centered_cosine_histogram.png",
        "subject_feature_variance_histogram.png",
        "subject_effective_rank_spectrum.png",
        "subject_standardized_distance_heatmap.png",
        "subject_standardized_distance_histogram.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}


def test_single_subject_diagnostics() -> None:
    embeddings = torch.tensor([[1.0, 2.0, 3.0]])

    _, centered_off_diagonal, centered_metrics = (
        cohort_centered_cosine_diagnostics(embeddings)
    )
    variances, _, rank_metrics = subject_variance_rank_diagnostics(embeddings)
    distances, distance_off_diagonal, distance_metrics = (
        standardized_euclidean_diagnostics(embeddings)
    )

    assert centered_off_diagonal.numel() == 0
    assert distance_off_diagonal.numel() == 0
    assert torch.count_nonzero(variances) == 0
    assert torch.equal(distances, torch.zeros(1, 1))
    assert rank_metrics["subject_effective_rank"] == 0
    assert all(math.isnan(value) for value in centered_metrics.values())
    assert all(math.isnan(value) for value in distance_metrics.values())
