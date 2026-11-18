"""
Pure TensorFlow/Keras Cross-Validation Framework
==============================================

ChildID-grouped nested evaluation (``nested_on_training_children_then_refit``):

1. Outer ``StratifiedGroupKFold`` (frozen ``config/shared_fold_indices.json`` when present)
   holds out validation infants for **final scoring only**.
2. Inner ``GroupShuffleSplit`` on the outer-train infants supplies the callback /
   ``ReduceLROnPlateau`` / early-stopping monitor set (patience 25).
3. After selection, **refit** on all outer-train children for ``best_epoch`` epochs
   at constant LR (no callbacks; outer test never monitored).
4. Artifact schema: ``epochs_ran == best_epoch``, ``selection_epochs_ran`` = selection
   length, ``n_train`` = full outer-train fold size.

Artifacts match ``analysis/cross_validation_analysis.ipynb``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold

from .artifact_io import (
    append_performance_row,
    canonical_model_name,
    ensure_shared_folds,
    folds_index_hash,
    load_shared_folds,
    save_fold_predictions,
    write_fold_indices_sidecar,
)
from .results_manager import ResultsManager

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)


def _index_data(X: Any, idx: np.ndarray, multimodal: bool):
    if multimodal:
        return [modality[idx] for modality in X]
    return X[idx]


def _inner_child_split(
    y_train: np.ndarray,
    groups_train: np.ndarray,
    fold_id: int,
    random_state: int,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split outer-train into fit / monitor with no ChildID overlap."""
    for attempt in range(40):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state + fold_id * 100 + attempt,
        )
        fit_rel, mon_rel = next(splitter.split(np.zeros(len(y_train)), y_train, groups_train))
        fit_rel = np.asarray(fit_rel, dtype=np.int64)
        mon_rel = np.asarray(mon_rel, dtype=np.int64)
        if (
            len(np.unique(y_train[fit_rel])) == len(np.unique(y_train))
            and len(np.unique(y_train[mon_rel])) == len(np.unique(y_train))
            and not (set(groups_train[fit_rel]) & set(groups_train[mon_rel]))
        ):
            return fit_rel, mon_rel
    raise RuntimeError(f"Inner ChildID split failed for fold {fold_id}")




class PureTFCrossValidation:
    """Pure TensorFlow/Keras nested ChildID cross-validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_splits = config.get("n_splits", 10)
        self.random_state = config.get("random_state", 42)
        self.results_dir = Path(config.get("results_dir", "outputs"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_manager = ResultsManager(str(self.results_dir))
        self._configure_tf_m3()

    def _configure_tf_m3(self):
        try:
            tf.config.optimizer.set_jit(True)
            try:
                tf.config.threading.set_inter_op_parallelism_threads(0)
                tf.config.threading.set_intra_op_parallelism_threads(0)
            except RuntimeError:
                pass
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"GPU memory growth failed: {e}")
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            logger.info("TensorFlow configured for M3 MPS")
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")

    def _validate_frozen_folds(
        self,
        shared: Dict[str, Any],
        y: np.ndarray,
        groups: Optional[np.ndarray],
        n_samples: int,
    ) -> None:
        """Assert frozen indices cover the cohort and keep ChildIDs disjoint."""
        folds = list(shared.get("folds", []))
        if len(folds) != self.n_splits:
            raise ValueError(f"Frozen folds={len(folds)} != n_splits={self.n_splits}")
        all_val = []
        for fold_meta in folds:
            tr = np.asarray(fold_meta["train_idx"], dtype=np.int64)
            va = np.asarray(fold_meta["val_idx"], dtype=np.int64)
            if tr.min() < 0 or va.min() < 0 or tr.max() >= n_samples or va.max() >= n_samples:
                raise ValueError(f"Fold {fold_meta.get('fold')} indices out of range for n={n_samples}")
            if set(tr) & set(va):
                raise ValueError(f"Fold {fold_meta.get('fold')} train/val index overlap")
            if groups is not None:
                g_tr = set(np.asarray(groups)[tr].tolist())
                g_va = set(np.asarray(groups)[va].tolist())
                leak = g_tr & g_va
                if leak:
                    raise ValueError(
                        f"Fold {fold_meta.get('fold')} ChildID leakage across outer split: {sorted(leak)[:5]}"
                    )
            all_val.extend(va.tolist())
        if len(set(all_val)) != n_samples:
            raise ValueError(
                f"Outer val indices cover {len(set(all_val))} unique rows, expected {n_samples}"
            )
        fp = shared.get("cohort_fingerprint") or {}
        if fp.get("n_rows") not in (None, n_samples):
            raise ValueError(f"cohort_fingerprint.n_rows={fp.get('n_rows')} != {n_samples}")
        if fp.get("random_state") not in (None, self.random_state):
            logger.warning(
                f"Frozen fingerprint random_state={fp.get('random_state')} "
                f"vs config random_state={self.random_state}"
            )
        expected_hash = folds_index_hash(folds)
        stored_hash = fp.get("index_hash")
        if stored_hash in (None, "", "null"):
            logger.warning(
                "Frozen folds lack cohort_fingerprint.index_hash; "
                f"computed={expected_hash[:16]}…"
            )
        elif stored_hash != expected_hash:
            raise ValueError(
                "Frozen fold index_hash mismatch "
                f"(stored={stored_hash[:16]}… computed={expected_hash[:16]}…). "
                "Cohort row order may have changed."
            )

    def _resolve_outer_folds(
        self, y: np.ndarray, groups: Optional[np.ndarray], n_samples: int
    ) -> List[Dict[str, Any]]:
        """Prefer frozen shared folds; else build StratifiedGroupKFold."""
        frozen_src = Path(self.config.get("shared_folds_path", "config/shared_fold_indices.json"))
        use_frozen = bool(self.config.get("use_frozen_folds", True))
        shared_path = ensure_shared_folds(self.results_dir, frozen_src if frozen_src.exists() else None)

        if use_frozen and shared_path.exists():
            shared = load_shared_folds(shared_path)
            if int(shared.get("n_rows", -1)) == n_samples and int(shared.get("n_splits", -1)) == self.n_splits:
                self._validate_frozen_folds(shared, y, groups, n_samples)
                logger.info(f"Using frozen ChildID folds from {shared_path}")
                return list(shared["folds"])
            logger.warning(
                f"Frozen folds n_rows={shared.get('n_rows')} / n_splits={shared.get('n_splits')} "
                f"do not match current data ({n_samples}, {self.n_splits}); regenerating."
            )

        use_group_cv = bool(self.config.get("use_group_cv", True))
        if use_group_cv:
            if groups is None:
                raise ValueError("use_group_cv=True requires groups (ChildID)")
            kfold = StratifiedGroupKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            split_iter = kfold.split(np.zeros(n_samples), y, groups)
        else:
            kfold = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )
            split_iter = kfold.split(np.zeros(n_samples), y)

        folds = []
        for fold_i, (train_idx, val_idx) in enumerate(split_iter, start=1):
            folds.append(
                {
                    "fold": fold_i,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                }
            )
        payload = {
            "split": "StratifiedGroupKFold" if use_group_cv else "StratifiedKFold",
            "group_column": "ChildID",
            "n_splits": self.n_splits,
            "shuffle": True,
            "random_state": self.random_state,
            "n_rows": n_samples,
            "epoch_cap": int(self.config.get("num_epochs", 150)),
            "early_stopping": "nested_on_training_children_then_refit",
            "protocol_note": (
                "Phase 1: train on inner-fit children; monitor early-stopping/LR on "
                "inner-monitor children (patience=25). Phase 2: refit on all outer-train "
                "children for best_epoch epochs at constant LR. Outer test is scored once "
                "and never used for callbacks."
            ),
            "start_from_epoch": int(self.config.get("early_stopping_start_epoch", 15)),
            "early_stopping_patience": int(self.config.get("early_stopping_patience", 25)),
            "cohort_fingerprint": {
                "n_rows": n_samples,
                "n_splits": self.n_splits,
                "random_state": self.random_state,
                "group_column": "ChildID",
                "index_hash": folds_index_hash(folds),
            },
            "folds": folds,
        }
        out = self.results_dir / "shared_fold_indices.json"
        out.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote regenerated shared folds → {out}")
        return folds

    def run_cross_validation(
        self,
        model_name: str,
        model_trainer: Callable,
        X: Any,
        y: np.ndarray,
        config: Dict[str, Any],
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        try:
            display_name = canonical_model_name(model_name)
            logger.info(f"Starting nested {self.n_splits}-fold CV for {display_name}")

            is_multimodal = isinstance(X, (list, tuple))
            if is_multimodal:
                if len(X) == 0:
                    raise ValueError("Received empty modalities list for X")
                n_mods = [len(m) for m in X]
                if len(set(n_mods)) != 1:
                    raise ValueError(f"All modalities must have same length, got {n_mods}")
                n_samples = n_mods[0]
            else:
                n_samples = len(X)

            if n_samples != len(y):
                raise ValueError(f"X ({n_samples}) and y ({len(y)}) length mismatch")

            if groups is not None:
                groups = np.asarray(groups)
                if len(groups) != n_samples:
                    raise ValueError(f"groups length {len(groups)} != n_samples {n_samples}")

            folds = self._resolve_outer_folds(y, groups, n_samples)
            write_fold_indices_sidecar(self.results_dir, display_name, folds)

            fold_results: List[Dict[str, Any]] = []
            all_predictions: List[Any] = []
            all_true_labels: List[Any] = []

            for fold_meta in folds:
                fold = int(fold_meta["fold"])
                train_idx = np.asarray(fold_meta["train_idx"], dtype=np.int64)
                test_idx = np.asarray(fold_meta["val_idx"], dtype=np.int64)  # outer test
                try:
                    logger.info(f"Processing fold {fold}/{self.n_splits}")

                    y_train_outer = y[train_idx]
                    if groups is None:
                        raise ValueError("Nested ChildID CV requires groups")
                    groups_train = groups[train_idx]
                    fit_rel, mon_rel = _inner_child_split(
                        y_train_outer, groups_train, fold, self.random_state
                    )
                    fit_idx = train_idx[fit_rel]
                    mon_idx = train_idx[mon_rel]

                    # Runtime ChildID disjointness (outer + inner)
                    g_fit = set(groups[fit_idx].tolist())
                    g_mon = set(groups[mon_idx].tolist())
                    g_test = set(groups[test_idx].tolist())
                    if g_fit & g_mon:
                        raise ValueError(f"Fold {fold}: ChildID leak fit∩monitor")
                    if g_fit & g_test or g_mon & g_test:
                        raise ValueError(f"Fold {fold}: ChildID leak into outer test")
                    if set(fit_idx) & set(mon_idx) or set(fit_idx) & set(test_idx) or set(mon_idx) & set(test_idx):
                        raise ValueError(f"Fold {fold}: index overlap across fit/monitor/test")

                    X_fit = _index_data(X, fit_idx, is_multimodal)
                    X_mon = _index_data(X, mon_idx, is_multimodal)
                    X_outer = _index_data(X, train_idx, is_multimodal)
                    X_test = _index_data(X, test_idx, is_multimodal)
                    y_fit, y_mon = y[fit_idx], y[mon_idx]
                    y_outer, y_test = y[train_idx], y[test_idx]

                    def _balanced_weights(y_arr: np.ndarray) -> Optional[Dict[int, float]]:
                        try:
                            from sklearn.utils.class_weight import compute_class_weight

                            classes = np.unique(y_arr)
                            cw = compute_class_weight("balanced", classes=classes, y=y_arr)
                            return {int(c): float(w) for c, w in zip(classes, cw)}
                        except Exception:
                            return None

                    patience = int(config.get("early_stopping_patience", 25))
                    fold_dir = (
                        self.results_dir
                        / "checkpoints"
                        / model_folder_safe(display_name)
                        / f"fold_{fold}"
                    )
                    fold_dir.mkdir(parents=True, exist_ok=True)

                    t0 = time.time()
                    # Phase 1 — selection on inner-fit / inner-monitor (outer test unused).
                    sel_cfg = {
                        **config,
                        "class_weight": _balanced_weights(y_fit),
                        "fold": fold,
                        "outer_test_idx": test_idx,
                        "training_phase": "selection",
                        "early_stopping_patience": patience,
                        "early_stopping_start_epoch": int(
                            config.get("early_stopping_start_epoch", 15)
                        ),
                        "enable_callbacks": True,
                    }
                    sel_out = model_trainer(
                        X_fit,
                        y_fit,
                        X_mon,
                        y_mon,
                        fold_dir / "selection",
                        sel_cfg,
                        X_test=None,
                        y_test=None,
                    )
                    best_epoch = int(sel_out.get("best_epoch", 0) or 0)
                    selection_epochs_ran = int(sel_out.get("epochs_ran", best_epoch) or best_epoch)
                    if best_epoch < 1:
                        raise RuntimeError(f"{display_name} fold {fold}: invalid best_epoch={best_epoch}")

                    # Phase 2 — refit on all outer-train children for best_epoch epochs.
                    refit_cfg = {
                        **config,
                        "class_weight": _balanced_weights(y_outer),
                        "fold": fold,
                        "outer_test_idx": test_idx,
                        "training_phase": "refit",
                        "num_epochs": best_epoch,
                        "enable_callbacks": False,
                    }
                    fold_out = model_trainer(
                        X_outer,
                        y_outer,
                        X_mon,
                        y_mon,
                        fold_dir / "refit",
                        refit_cfg,
                        X_test=X_test,
                        y_test=y_test,
                    )
                    fold_time = time.time() - t0
                    # Artifact schema matches committed results: epochs_ran == best_epoch;
                    # selection_epochs_ran records the inner early-stopping run length.
                    fold_out["best_epoch"] = best_epoch
                    fold_out["epochs_ran"] = best_epoch
                    fold_out["selection_epochs_ran"] = selection_epochs_ran
                    fold_out["selection_train_time"] = float(sel_out.get("training_time", 0.0))
                    fold_out["training_time"] = float(
                        float(sel_out.get("training_time", 0.0))
                        + float(fold_out.get("training_time", 0.0))
                    )

                    y_pred = np.asarray(fold_out.get("y_pred", []))
                    y_true = np.asarray(fold_out.get("y_true", y_test))
                    y_proba = fold_out.get("y_proba")
                    if y_proba is None:
                        raise RuntimeError(
                            f"{display_name} fold {fold}: trainer must return y_proba "
                            "(softmax) for artifact export"
                        )
                    y_proba = np.asarray(y_proba)

                    acc = float(fold_out.get("accuracy", np.mean(y_pred == y_true) if len(y_pred) else 0.0))
                    n_correct = int((y_pred == y_true).sum()) if len(y_pred) else 0

                    save_fold_predictions(
                        self.results_dir,
                        display_name,
                        fold,
                        y_true=y_true,
                        y_pred=y_pred,
                        y_proba=y_proba,
                        val_idx=test_idx,
                    )

                    append_performance_row(
                        self.results_dir,
                        {
                            "model": display_name,
                            "fold": fold,
                            "validation_accuracy": acc,
                            "auc_roc": fold_out.get("auc_roc", np.nan),
                            "train_loss": fold_out.get("train_loss", np.nan),
                            "validation_loss": fold_out.get("validation_loss", np.nan),
                            "epoch_cap": int(config.get("num_epochs", 150)),
                            "epochs_ran": best_epoch,
                            "selection_epochs_ran": selection_epochs_ran,
                            "best_epoch": best_epoch,
                            "training_time": float(fold_out.get("training_time", fold_time)),
                            "inference_ms_per_clip": fold_out.get("inference_ms_per_clip", np.nan),
                            "inference_ms_per_clip_sd": fold_out.get("inference_ms_per_clip_sd", np.nan),
                            "n_correct": n_correct,
                            "n_train": int(len(train_idx)),
                            "n_validation": int(len(test_idx)),
                            "n_train_children": int(len(np.unique(groups[train_idx]))),
                            "n_validation_children": int(len(np.unique(groups[test_idx]))),
                        },
                    )

                    fold_out["fold"] = fold
                    fold_out["fold_time"] = fold_time
                    fold_out["train_size"] = len(train_idx)
                    fold_out["val_size"] = len(test_idx)
                    fold_out["monitor_size"] = len(mon_idx)
                    fold_out["fit_size"] = len(fit_idx)
                    fold_out["validation_accuracy"] = acc
                    fold_out["y_true"] = y_true
                    fold_out["y_pred"] = y_pred

                    self.results_manager.add_fold_result(display_name, fold, fold_out)
                    fold_results.append(fold_out)
                    all_predictions.extend(list(y_pred))
                    all_true_labels.extend(list(y_true))
                    logger.info(f"Fold {fold} done in {fold_time:.1f}s acc={acc:.4f}")
                    tf.keras.backend.clear_session()
                except Exception as e:
                    logger.error(f"Fold {fold} failed: {e}")
                    raise

            overall_metrics = self._calculate_overall_metrics(
                fold_results, all_predictions, all_true_labels
            )
            results = {
                "model_name": display_name,
                "n_splits": self.n_splits,
                "fold_results": fold_results,
                "overall_metrics": overall_metrics,
                "config": config,
                "protocol": "nested_on_training_children_then_refit",
            }
            results_file = self.results_dir / f"{display_name}_cv_results.json"
            self._save_results(results, results_file)

            if all_predictions and all_true_labels:
                try:
                    from config.config import JOKE_TYPE_NAMES

                    class_names = JOKE_TYPE_NAMES
                except Exception:
                    class_names = [f"Class_{i}" for i in range(len(np.unique(all_true_labels)))]
                self.results_manager.save_model_classification_report(
                    display_name, np.array(all_true_labels), np.array(all_predictions), class_names
                )
                self.results_manager.save_model_confusion_matrix(
                    display_name, np.array(all_true_labels), np.array(all_predictions), class_names
                )

            logger.info(f"Cross-validation completed for {display_name}")
            return results
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise

    def _calculate_overall_metrics(
        self, fold_results: List[Dict], all_predictions: List, all_true_labels: List
    ) -> Dict[str, Any]:
        try:
            if not fold_results:
                return {}
            metrics = ["accuracy", "precision", "recall", "f1_score"]
            overall: Dict[str, Any] = {}
            for metric in metrics:
                values = [fold.get(metric, 0) for fold in fold_results if metric in fold]
                if values:
                    overall[f"{metric}_mean"] = float(np.mean(values))
                    overall[f"{metric}_std"] = float(np.std(values))
                    overall[f"{metric}_min"] = float(np.min(values))
                    overall[f"{metric}_max"] = float(np.max(values))
            if all_predictions and all_true_labels:
                overall["confusion_matrix"] = confusion_matrix(
                    all_true_labels, all_predictions
                ).tolist()
                overall["classification_report"] = classification_report(
                    all_true_labels, all_predictions, output_dict=True, zero_division=0
                )
            times = [fold.get("training_time", 0) for fold in fold_results if "training_time" in fold]
            if times:
                overall["total_training_time"] = float(np.sum(times))
                overall["avg_training_time"] = float(np.mean(times))
                overall["std_training_time"] = float(np.std(times))
            return overall
        except Exception as e:
            logger.warning(f"Overall metrics calculation failed: {e}")
            return {}

    def _save_results(self, results: Dict[str, Any], results_file: Path):
        try:

            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy(i) for i in obj]
                return obj

            with open(results_file, "w") as f:
                json.dump(convert_numpy(results), f, indent=2)
            logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            logger.warning(f"Results saving failed: {e}")

    def create_summary_report(self, results: Dict[str, Any]) -> str:
        try:
            lines = [
                "=" * 60,
                f"CROSS-VALIDATION RESULTS: {results['model_name']}",
                "=" * 60,
                f"Protocol: {results.get('protocol', 'nested_childid_grouped')}",
                f"Number of folds: {results['n_splits']}",
                "",
            ]
            overall = results.get("overall_metrics", {})
            if overall:
                lines.append("OVERALL METRICS:")
                lines.append("-" * 20)
                for metric in ["accuracy", "precision", "recall", "f1_score"]:
                    mean_key, std_key = f"{metric}_mean", f"{metric}_std"
                    if mean_key in overall and std_key in overall:
                        lines.append(
                            f"{metric.capitalize()}: {overall[mean_key]:.4f} ± {overall[std_key]:.4f}"
                        )
            lines.append("")
            lines.append("FOLD DETAILS:")
            lines.append("-" * 20)
            for fold in results.get("fold_results", []):
                lines.append(
                    f"Fold {fold.get('fold')}: acc={fold.get('accuracy', fold.get('validation_accuracy', 0)):.4f} "
                    f"macro-F1={fold.get('f1_score', 0):.4f} time={fold.get('training_time', 0):.1f}s"
                )
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Summary report creation failed: {e}")
            return "Summary report not available"

    def clear_session(self):
        try:
            tf.keras.backend.clear_session()
        except Exception as e:
            logger.warning(f"Session clearing failed: {e}")


def model_folder_safe(name: str) -> str:
    from .artifact_io import model_folder

    return model_folder(name)
