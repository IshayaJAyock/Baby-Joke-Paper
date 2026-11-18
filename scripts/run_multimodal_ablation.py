#!/usr/bin/env python3
"""
Multi-cue fusion ablation on real videos
========================================

Runs ChildID-grouped 10-fold CV for each ablation variant of the multi-cue model:

  (i)   cues alone   → cue_motion, cue_face, cue_hands
  (ii)  concat       → pool + concat + project
  (iii) attention_only → cross-attn + concat + project (no gate)
  (iv)  gating_only  → pool + concat + gate + project (no cross-attn)
  (v)   full         → cross-attn + concat + gate + project

Usage (from repo root):
  python scripts/run_multimodal_ablation.py
  python scripts/run_multimodal_ablation.py --variants full,concat,gating_only
  python scripts/run_multimodal_ablation.py --max-videos 80   # smoke test

Outputs:
  outputs/ablation_multicue/ablation_summary.csv
  outputs/ablation_multicue/<variant>_cv_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("multicue_ablation")

from models.multimodal import ABLATION_VARIANTS  # noqa: E402


VARIANT_META = {
    "cue_motion": {
        "family": "cues_alone",
        "description": "(i) motion cue alone (Farneback 8-d)",
    },
    "cue_face": {
        "family": "cues_alone",
        "description": "(i) face cue alone (MediaPipe Face Mesh 15-d)",
    },
    "cue_hands": {
        "family": "cues_alone",
        "description": "(i) hands/gesture cue alone (MediaPipe Hands 45-d)",
    },
    "concat": {
        "family": "fusion",
        "description": "(ii) concat: GAP each encoder → concat → project",
    },
    "attention_only": {
        "family": "fusion",
        "description": "(iii) attention only: cross-attn → concat → project (no gate)",
    },
    "gating_only": {
        "family": "fusion",
        "description": "(iv) gating only: GAP → concat → sigmoid gate → project (no attn)",
    },
    "full": {
        "family": "fusion",
        "description": "(v) full: cross-attn → concat → gate → project",
    },
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-cue ablation CV on real videos")
    p.add_argument(
        "--variants",
        type=str,
        default=",".join(ABLATION_VARIANTS),
        help=f"Comma-separated variants. Default: all of {ABLATION_VARIANTS}",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="outputs/ablation_multicue",
        help="Directory for ablation CSVs / JSON",
    )
    p.add_argument(
        "--feature-cache",
        type=str,
        default="data/cache/multimodal_features.npz",
        help="NPZ cache for extracted motion/face/hands features",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="If >0, use only the first N clips after dedup (smoke test)",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=0,
        help="Override CV folds (0 = use config.cv_folds, usually 10)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Override epoch cap (0 = use config.num_epochs)",
    )
    p.add_argument(
        "--force-reextract",
        action="store_true",
        help="Ignore feature cache and re-extract from videos",
    )
    return p.parse_args()


def _build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    from config.config import StandardConfig

    std = StandardConfig()
    cfg = {
        "metadata_path": std.metadata_path,
        "cache_dir": "data/cache",
        "results_dir": args.results_dir,
        "max_frames": std.num_frames,
        "frame_height": std.frame_size[0],
        "frame_width": std.frame_size[1],
        "motion_dim": 8,
        "facial_dim": 15,
        "gesture_dim": 45,
        "fusion_method": "full",
        "attention_heads": std.multimodal["attention_heads"],
        "fusion_units": std.multimodal["fusion_units"],
        "dropout_rate": std.multimodal["dropout_rate"],
        "learning_rate": std.learning_rate,
        "batch_size": std.batch_size,
        "num_epochs": args.epochs or std.num_epochs,
        "n_splits": args.n_splits or std.cv_folds,
        "random_state": std.cv_random_state,
        "use_group_cv": getattr(std, "use_group_cv", True),
        "num_classes": std.num_classes,
    }
    return cfg


def _load_or_extract_features(
    config: Dict[str, Any],
    cache_path: Path,
    max_videos: int,
    force_reextract: bool,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Return X=[motion,face,hands], y, groups. Extract once; reuse across variants."""
    from scripts.run_multimodal import load_data, process_multimodal_data

    video_paths, labels, groups = load_data(config)
    if max_videos and max_videos > 0:
        video_paths = video_paths[:max_videos]
        labels = labels[:max_videos]
        groups = groups[:max_videos]
        logger.info(f"Smoke subset: first {len(video_paths)} videos")

    cache_path = Path(cache_path)
    if cache_path.exists() and not force_reextract:
        logger.info(f"Loading cached multimodal features from {cache_path}")
        z = np.load(cache_path, allow_pickle=True)
        # Validate length match
        if int(z["y"].shape[0]) == len(labels) and np.array_equal(z["y"], labels):
            X = [z["motion"], z["facial"], z["gesture"]]
            groups = z["groups"] if "groups" in z.files else groups
            return X, labels, groups
        logger.warning("Feature cache mismatch with current labels; re-extracting")

    X, y = process_multimodal_data(video_paths.tolist(), labels.tolist(), config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        motion=X[0],
        facial=X[1],
        gesture=X[2],
        y=y,
        groups=np.asarray(groups),
        video_paths=np.asarray(video_paths),
    )
    logger.info(f"Saved feature cache → {cache_path}")
    return X, y, groups


def _mean_fold_accuracy(results: Dict[str, Any]) -> Tuple[float, float]:
    folds = results.get("fold_results", [])
    accs = [float(f["accuracy"]) for f in folds if "accuracy" in f]
    if not accs:
        return float("nan"), float("nan")
    return float(np.mean(accs)), float(np.std(accs, ddof=1) if len(accs) > 1 else 0.0)


def run_one_variant(
    variant: str,
    X: List[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    from scripts.run_multimodal import train_single_fold, configure_tf_m3
    from src.cross_validation import PureTFCrossValidation

    configure_tf_m3()
    cfg = dict(base_config)
    cfg["ablation_variant"] = variant
    cfg["fusion_method"] = variant
    cfg["results_dir"] = str(Path(base_config["results_dir"]) / variant)
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(f"Ablation variant: {variant} — {VARIANT_META[variant]['description']}")
    logger.info("=" * 72)

    cv = PureTFCrossValidation(cfg)
    results = cv.run_cross_validation(
        model_name=f"Multimodal_{variant}",
        model_trainer=train_single_fold,
        X=X,
        y=y,
        config=cfg,
        groups=groups,
    )
    mean_acc, sd_acc = _mean_fold_accuracy(results)
    meta = VARIANT_META[variant]
    row = {
        "variant": variant,
        "family": meta["family"],
        "description": meta["description"],
        "mean_accuracy": mean_acc,
        "sd_accuracy": sd_acc,
        "n_folds": cfg["n_splits"],
        "n_clips": int(len(y)),
        "n_children": int(len(np.unique(groups))),
    }
    # Persist per-variant summary next to CV JSON
    out_dir = Path(base_config["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{variant}_ablation_row.json", "w") as f:
        json.dump(row, f, indent=2)
    return row


def main() -> None:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ABLATION_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}. Choose from {ABLATION_VARIANTS}")

    base = _build_base_config(args)
    out_dir = Path(base["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, groups = _load_or_extract_features(
        base,
        Path(args.feature_cache),
        max_videos=args.max_videos,
        force_reextract=args.force_reextract,
    )

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        try:
            rows.append(run_one_variant(variant, X, y, groups, base))
        except Exception as e:
            logger.exception(f"Variant {variant} failed: {e}")
            rows.append(
                {
                    "variant": variant,
                    "family": VARIANT_META[variant]["family"],
                    "description": VARIANT_META[variant]["description"],
                    "mean_accuracy": np.nan,
                    "sd_accuracy": np.nan,
                    "n_folds": base["n_splits"],
                    "n_clips": int(len(y)),
                    "n_children": int(len(np.unique(groups))),
                    "error": str(e),
                }
            )

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "ablation_summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info(f"Ablation summary → {csv_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
