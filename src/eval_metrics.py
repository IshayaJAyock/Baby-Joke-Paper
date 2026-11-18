"""Shared classification metrics (macro + weighted)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Return accuracy plus macro and weighted precision/recall/F1 (and AUC if possible)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Primary labels match the manuscript (macro-averaged).
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["auc_roc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except Exception:
            out["auc_roc"] = float("nan")
    return out
