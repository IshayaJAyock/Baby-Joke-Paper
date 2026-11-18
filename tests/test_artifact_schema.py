"""Schema tests for notebook-compatible CV artifacts (no GPU / no video required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact_io import (  # noqa: E402
    PERF_COLUMNS,
    append_performance_row,
    save_fold_predictions,
)


def test_performance_csv_single_schema(tmp_path: Path):
    append_performance_row(
        tmp_path,
        {
            "model": "Multi-cue",
            "fold": 1,
            "validation_accuracy": 0.95,
            "auc_roc": 0.99,
            "train_loss": 0.1,
            "validation_loss": 0.2,
            "epoch_cap": 150,
            "epochs_ran": 40,
            "selection_epochs_ran": 40,
            "best_epoch": 25,
            "training_time": 1.2,
            "n_correct": 95,
            "n_train": 1000,
            "n_validation": 100,
            "n_train_children": 70,
            "n_validation_children": 8,
        },
    )
    append_performance_row(
        tmp_path,
        {
            "model": "Multi-cue",
            "fold": 2,
            "validation_accuracy": 0.96,
            "auc_roc": 0.99,
            "epoch_cap": 150,
            "epochs_ran": 50,
            "best_epoch": 30,
            "training_time": 1.3,
            "n_correct": 96,
            "n_train": 1000,
            "n_validation": 100,
            "n_train_children": 70,
            "n_validation_children": 8,
        },
    )
    df = pd.read_csv(tmp_path / "all_models_performance.csv")
    assert list(df.columns) == PERF_COLUMNS
    assert len(df) == 2
    assert set(df["model"]) == {"Multi-cue"}
    # early-stopping compatible: best_epoch <= epochs_ran
    assert (df["best_epoch"] <= df["epochs_ran"]).all()


def test_fold_npz_schema(tmp_path: Path):
    y_true = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 3, 3], dtype=np.int64)
    y_proba = np.eye(5, dtype=np.float32)
    val_idx = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    out = save_fold_predictions(
        tmp_path, "Multi-cue", 1, y_true, y_pred, y_proba, val_idx
    )
    z = np.load(out)
    assert set(z.files) >= {"y_true", "y_pred", "y_proba", "val_idx"}
    assert z["y_proba"].shape == (5, 5)
    assert out.parent.name == "multi-cue"


def test_reference_results_package_present():
    ref = ROOT / "analysis" / "reference_results"
    assert (ref / "all_models_performance.csv").exists()
    assert (ref / "shared_fold_indices.json").exists()
    df = pd.read_csv(ref / "all_models_performance.csv")
    assert list(df.columns) == PERF_COLUMNS
    assert len(df) == 40
    abl = pd.read_csv(ref / "ablation" / "ablation_summary.csv")
    assert "full" in set(abl["variant"])
    full = float(abl.loc[abl["variant"] == "full", "mean_accuracy"].iloc[0])
    assert abs(full - 0.967582) < 1e-5


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        test_performance_csv_single_schema(Path(td))
        test_fold_npz_schema(Path(td))
    test_reference_results_package_present()
    print("OK")
