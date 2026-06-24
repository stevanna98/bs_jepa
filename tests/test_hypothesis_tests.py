from __future__ import annotations

import math

import pytest
import torch

from bsjepa.hypothesis_tests import (
    anova_variance_ratio,
    collect_downstream_comparison,
    extract_rsn_loss_rows,
    pca_projection,
    prepare_output_dirs,
    summarize_history_metrics,
    summarize_rsn_losses,
)


def test_prepare_output_dirs_uses_timestamped_child_for_existing_results(tmp_path) -> None:
    base = tmp_path / "neurobiological_hypothesis_tests"
    first = prepare_output_dirs(base)
    (first.metrics / "existing.json").write_text("{}\n")

    second = prepare_output_dirs(base)

    assert first.root == base
    assert second.root.parent == base
    assert second.root.name.startswith("run_")
    assert (base / "metrics" / "existing.json").read_text() == "{}\n"


def test_training_metric_summary_tracks_initial_final_and_change() -> None:
    history = [
        {"epoch": 1.0, "loss": 4.0, "prediction_variance": 0.5},
        {"epoch": 2.0, "loss": 3.0},
        {"epoch": 3.0, "loss": 2.0, "prediction_variance": 0.25},
    ]

    rows = summarize_history_metrics(history)
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["loss"]["initial"] == 4.0
    assert by_metric["loss"]["final"] == 2.0
    assert by_metric["loss"]["percent_change"] == pytest.approx(-50.0)
    assert by_metric["prediction_variance"]["final"] == 0.25


def test_rsn_loss_rows_and_summaries_map_names_and_improvements() -> None:
    history = [
        {"epoch": 1.0, "rsn_loss_0": 2.0, "rsn_loss_1": 1.0},
        {"epoch": 2.0, "rsn_loss_0": 1.5, "rsn_loss_1": 1.25},
    ]

    rows = extract_rsn_loss_rows(history, ["visual", "default"])
    summaries, stats = summarize_rsn_losses(rows)

    assert rows[0]["rsn_name"] == "visual"
    assert summaries[0]["absolute_improvement"] == pytest.approx(0.5)
    assert summaries[1]["absolute_improvement"] == pytest.approx(-0.25)
    assert stats["final_loss_range"] == pytest.approx(0.25)


def test_anova_variance_ratio_reports_group_degrees_of_freedom() -> None:
    result = anova_variance_ratio({"a": [1.0, 1.2], "b": [3.0, 3.2]})

    assert result["between_group_df"] == 1.0
    assert result["within_group_df"] == 2.0
    assert result["anova_f_ratio"] > 1.0


def test_anova_variance_ratio_handles_insufficient_groups() -> None:
    result = anova_variance_ratio({"a": [1.0, 2.0]})

    assert math.isnan(result["anova_f_ratio"])
    assert result["between_group_df"] == 0.0


def test_pca_projection_returns_two_columns_even_for_rank_one_data() -> None:
    projection = pca_projection(torch.tensor([[1.0], [2.0], [3.0]]))

    assert projection.shape == (3, 2)
    assert torch.count_nonzero(projection[:, 1]) == 0


def test_downstream_comparison_collects_logged_and_rerun_metrics() -> None:
    history = [
        {
            "epoch": 1.0,
            "gender_probe_val_balanced_accuracy": 0.6,
            "gender_majority_val_balanced_accuracy": 0.5,
        }
    ]
    rows = collect_downstream_comparison(
        history,
        {"gender_random_encoder_val_balanced_accuracy": 0.55},
    )

    by_model = {row["model"]: row for row in rows}
    assert by_model["BS-JEPA target encoder linear probe"]["balanced_accuracy"] == 0.6
    assert by_model["Majority baseline"]["balanced_accuracy"] == 0.5
    assert by_model["Random encoder linear probe"]["balanced_accuracy"] == 0.55
