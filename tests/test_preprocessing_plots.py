from __future__ import annotations

from bsjepa.plotting import save_training_plots


def test_gender_preprocessing_plots_created(tmp_path) -> None:
    history = [
        {
            "epoch": 1.0,
            "loss": 1.0,
            "gender_probe_val_accuracy": 0.5,
            "gender_probe_val_balanced_accuracy": 0.5,
            "gender_probe_val_loss": 0.8,
            "gender_probe_raw_val_accuracy": 0.5,
            "gender_probe_raw_val_balanced_accuracy": 0.5,
            "gender_probe_raw_val_loss": 0.8,
            "gender_probe_raw_all_embedding_cosine_mean": 0.9,
            "gender_probe_raw_all_embedding_cosine_std": 0.1,
            "gender_probe_centered_val_accuracy": 0.6,
            "gender_probe_centered_val_balanced_accuracy": 0.6,
            "gender_probe_centered_val_loss": 0.7,
            "gender_probe_centered_all_embedding_cosine_mean": 0.0,
            "gender_probe_centered_all_embedding_cosine_std": 0.4,
        },
        {
            "epoch": 2.0,
            "loss": 0.9,
            "gender_probe_val_accuracy": 0.55,
            "gender_probe_val_balanced_accuracy": 0.55,
            "gender_probe_val_loss": 0.75,
            "gender_probe_raw_val_accuracy": 0.55,
            "gender_probe_raw_val_balanced_accuracy": 0.55,
            "gender_probe_raw_val_loss": 0.75,
            "gender_probe_raw_all_embedding_cosine_mean": 0.85,
            "gender_probe_raw_all_embedding_cosine_std": 0.12,
            "gender_probe_centered_val_accuracy": 0.65,
            "gender_probe_centered_val_balanced_accuracy": 0.65,
            "gender_probe_centered_val_loss": 0.65,
            "gender_probe_centered_all_embedding_cosine_mean": -0.05,
            "gender_probe_centered_all_embedding_cosine_std": 0.45,
        },
    ]

    save_training_plots(history, tmp_path)

    expected = {
        "gender_probe_preprocessing_accuracy.png",
        "gender_probe_preprocessing_balanced_accuracy.png",
        "gender_probe_preprocessing_loss.png",
        "gender_probe_preprocessing_cosine_mean.png",
        "gender_probe_preprocessing_cosine_std.png",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
