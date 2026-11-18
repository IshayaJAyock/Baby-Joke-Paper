"""
Notebook-compatible result artifacts
====================================

Writes the same layout that ``analysis/cross_validation_analysis.ipynb`` expects:

  outputs/
    shared_fold_indices.json
    all_models_performance.csv
    multi-cue|3d_cnn|tan|cnn_lstm/
      fold_{k}_predictions.npz   # y_true, y_pred, y_proba, val_idx

Model folder names follow the frozen cohort convention (hyphenated multi-cue).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Display name -> on-disk folder (matches notebook: model_name.lower())
MODEL_DIR = {
    "Multi-cue": "multi-cue",
    "3D_CNN": "3d_cnn",
    "TAN": "tan",
    "CNN_LSTM": "cnn_lstm",
    "Multimodal": "multi-cue",
    "Multimodal_full": "multi-cue",
}

PERF_COLUMNS = [
    "model",
    "fold",
    "validation_accuracy",
    "auc_roc",
    "train_loss",
    "validation_loss",
    "epoch_cap",
    "epochs_ran",
    "selection_epochs_ran",
    "best_epoch",
    "training_time",
    "inference_ms_per_clip",
    "inference_ms_per_clip_sd",
    "n_correct",
    "n_train",
    "n_validation",
    "n_train_children",
    "n_validation_children",
    "timestamp",
]



def folds_index_hash(folds: List[Dict[str, Any]]) -> str:
    """Stable fingerprint of outer train/val index lists."""
    import hashlib
    payload = []
    for fold_meta in sorted(folds, key=lambda f: int(f["fold"])):
        payload.append(
            {
                "fold": int(fold_meta["fold"]),
                "train_idx": [int(i) for i in fold_meta["train_idx"]],
                "val_idx": [int(i) for i in fold_meta["val_idx"]],
            }
        )
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def canonical_model_name(name: str) -> str:
    n = str(name)
    if n.startswith("Multimodal"):
        return "Multi-cue"
    if n in ("3D_CNN", "3d_cnn", "3D-CNN"):
        return "3D_CNN"
    if n in ("CNN_LSTM", "cnn_lstm", "CNN-LSTM"):
        return "CNN_LSTM"
    if n in ("TAN", "tan"):
        return "TAN"
    if n in ("Multi-cue", "multi-cue", "Multimodal"):
        return "Multi-cue"
    return n


def model_folder(name: str) -> str:
    return MODEL_DIR.get(canonical_model_name(name), canonical_model_name(name).lower())


def ensure_shared_folds(results_dir: Path, frozen_path: Optional[Path] = None) -> Path:
    """Copy frozen ChildID splits into results_dir if missing."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "shared_fold_indices.json"
    if dest.exists():
        return dest
    if frozen_path is None:
        frozen_path = Path(__file__).resolve().parents[1] / "config" / "shared_fold_indices.json"
    if not Path(frozen_path).exists():
        raise FileNotFoundError(
            f"Frozen split file not found: {frozen_path}. "
            "Expected config/shared_fold_indices.json"
        )
    shutil.copy2(frozen_path, dest)
    return dest


def load_shared_folds(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_fold_predictions(
    results_dir: Path,
    model_name: str,
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    val_idx: np.ndarray,
) -> Path:
    folder = Path(results_dir) / model_folder(model_name)
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"fold_{int(fold)}_predictions.npz"
    np.savez_compressed(
        out,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_proba=np.asarray(y_proba, dtype=np.float32),
        val_idx=np.asarray(val_idx, dtype=np.int64),
    )
    return out


def append_performance_row(results_dir: Path, row: Dict[str, Any]) -> None:
    path = Path(results_dir) / "all_models_performance.csv"
    payload = {c: row.get(c, np.nan) for c in PERF_COLUMNS}
    if "timestamp" not in row or not row.get("timestamp"):
        payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["model"] = canonical_model_name(str(payload["model"]))
    df = pd.DataFrame([payload], columns=PERF_COLUMNS)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def write_fold_indices_sidecar(
    results_dir: Path,
    model_name: str,
    folds: List[Dict[str, Any]],
) -> Path:
    folder = Path(results_dir) / model_folder(model_name)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fold_indices.json"
    path.write_text(json.dumps({"folds": folds}, indent=2))
    return path


def cohort_row_fingerprint(
    video_ids: List[Any],
    child_ids: List[Any],
    labels: List[Any],
) -> Dict[str, Any]:
    """Immutable cohort order hash: (VideoID, ChildID, label) per row."""
    import hashlib

    rows = [
        {
            "i": i,
            "video_id": str(v),
            "child_id": str(c),
            "label": int(y),
        }
        for i, (v, c, y) in enumerate(zip(video_ids, child_ids, labels))
    ]
    raw = json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {
        "n_rows": len(rows),
        "row_hash": hashlib.sha256(raw).hexdigest(),
        "schema": ["index", "video_id", "child_id", "label"],
    }


def save_inner_split_indices(
    results_dir: Path,
    model_name: str,
    fold: int,
    fit_idx: np.ndarray,
    mon_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Path:
    """Persist inner-fit / inner-monitor / outer-test indices for one fold."""
    folder = Path(results_dir) / model_folder(model_name)
    folder.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "fold": int(fold),
        "fit_idx": [int(i) for i in np.asarray(fit_idx)],
        "monitor_idx": [int(i) for i in np.asarray(mon_idx)],
        "test_idx": [int(i) for i in np.asarray(test_idx)],
        "n_fit": int(len(fit_idx)),
        "n_monitor": int(len(mon_idx)),
        "n_test": int(len(test_idx)),
    }
    if groups is not None:
        g = np.asarray(groups)
        payload["fit_children"] = sorted({str(x) for x in g[np.asarray(fit_idx)]})
        payload["monitor_children"] = sorted({str(x) for x in g[np.asarray(mon_idx)]})
        payload["test_children"] = sorted({str(x) for x in g[np.asarray(test_idx)]})
    out = folder / f"fold_{int(fold)}_inner_splits.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def write_run_manifest(
    results_dir: Path,
    *,
    model_name: str,
    config: Dict[str, Any],
    protocol: str,
    cohort_fingerprint: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write provenance manifest (commit, config slice, protocol, hashes)."""
    import hashlib
    import subprocess

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    commit = "unknown"
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    cfg_keys = [
        "num_epochs",
        "early_stopping_patience",
        "early_stopping_start_epoch",
        "reduce_lr_patience",
        "learning_rate",
        "batch_size",
        "n_splits",
        "random_state",
        "ablation_variant",
        "max_frames",
    ]
    cfg_slice = {k: config.get(k) for k in cfg_keys if k in config}
    cfg_hash = hashlib.sha256(
        json.dumps(cfg_slice, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    payload = {
        "model": canonical_model_name(model_name),
        "protocol": protocol,
        "git_commit": commit,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_slice": cfg_slice,
        "config_hash": cfg_hash,
        "cohort_fingerprint": cohort_fingerprint or {},
        "field_definitions": {
            "best_epoch": "Epoch of best inner-monitor val_accuracy during selection",
            "selection_epochs_ran": "Epochs executed in selection (typically best_epoch + patience)",
            "epochs_ran": "Refit length on full outer-train (= best_epoch)",
            "n_train": "Number of clips in the full outer-training fold (after refit)",
            "inference_ms_per_clip": "Forward-pass only on prepared tensors; excludes decode/extract",
        },
    }
    if extra:
        payload["extra"] = extra
    path = results_dir / model_folder(model_name) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    # Also keep a top-level copy for the latest model run in this results_dir.
    (results_dir / "run_manifest_latest.json").write_text(json.dumps(payload, indent=2))
    return path
