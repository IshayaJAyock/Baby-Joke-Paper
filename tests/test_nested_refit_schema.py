"""Verify committed performance rows match the nested-then-refit protocol schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_reference_performance_matches_refit_schema():
    perf = pd.read_csv(ROOT / "analysis" / "reference_results" / "all_models_performance.csv")
    shared = json.loads((ROOT / "config" / "shared_fold_indices.json").read_text())
    assert shared.get("early_stopping") == "nested_on_training_children_then_refit"

    assert (perf["epochs_ran"] == perf["best_epoch"]).all()
    # Selection run continues patience epochs past the best checkpoint.
    assert (perf["selection_epochs_ran"] == perf["best_epoch"] + 25).all()

    folds = {int(f["fold"]): f for f in shared["folds"]}
    for _, row in perf.iterrows():
        meta = folds[int(row["fold"])]
        assert int(row["n_train"]) == len(meta["train_idx"])
        assert int(row["n_validation"]) == len(meta["val_idx"])


def test_cv_defaults_use_patience_25():
    from config.config import StandardConfig

    cfg = StandardConfig()
    assert cfg.early_stopping_patience == 25
    assert cfg.early_stopping_start_epoch == 15


if __name__ == "__main__":
    test_reference_performance_matches_refit_schema()
    test_cv_defaults_use_patience_25()
    print("OK")
