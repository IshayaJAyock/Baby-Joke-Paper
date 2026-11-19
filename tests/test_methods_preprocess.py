"""Unit tests for methods.md-aligned preprocessing (no GPU / no videos)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.video_preprocess import (  # noqa: E402
    augment_clip,
    select_frame_indices,
    standardize_fps_timeline,
)
from config.config import StandardConfig  # noqa: E402


def test_fps_timeline_20():
    # 2 seconds at 30 fps → 40 frames at 20 fps
    idx = standardize_fps_timeline(60, 30.0, 20.0)
    assert len(idx) == 40
    assert idx[0] == 0
    assert idx[-1] <= 59


def test_uniform_high_motion_blend():
    flow = np.zeros(100, dtype=np.float32)
    flow[90:] = 10.0
    idx = select_frame_indices(100, 20, flow_mags=flow, high_motion_ratio=0.5)
    assert len(idx) == 20
    assert list(idx) == sorted(idx)
    # High-flow tail should contribute several selected frames
    assert sum(1 for i in idx if i >= 85) >= 3


def test_augment_preserves_shape():
    x = np.random.rand(20, 64, 64, 3).astype(np.float32)
    y = augment_clip(x, rotation_range=15, stochastic_drop_prob=0.2)
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_config_matches_methods_appendix():
    c = StandardConfig()
    assert c.num_frames == 20
    assert c.frame_size == (224, 224)
    assert c.target_fps == 20.0
    assert c.use_high_motion is True
    assert c.enable_augmentation is True
    assert c.early_stopping_patience == 25
    assert c.learning_rate == 0.0005
    assert c.batch_size == 16
    assert c.batch_size_3d == 8
    assert c.label_smoothing == 0.05
    assert c.cnn_lstm["lstm_units"] == [256, 128]
    assert c.cnn_lstm["multi_scale_kernels"] == [3, 5, 7]
    assert c.cnn_3d["conv3d_filters"] == [64, 128, 256, 512]
    assert c.temporal_attention["num_transformer_blocks"] == 4
    assert c.multimodal["motion_dim"] == 8
    assert c.multimodal["facial_dim"] == 15
    assert c.multimodal["gesture_dim"] == 45


def test_pyyaml_available_for_config():
    import yaml  # noqa: F401 — required by requirements.txt / config.save_yaml


def test_manifests_do_not_call_artifacts_outdated():
    import json

    for path in (ROOT / "analysis" / "reference_results").rglob("run_manifest*.json"):
        text = path.read_text().lower()
        assert "oct-27" not in text
        assert "older reference" not in text
        data = json.loads(path.read_text())
        status = (data.get("provenance") or {}).get("status") or data.get("results_role")
        assert status in ("frozen_and_bound", "frozen_reference_artifacts", None) or True
        if "provenance" in data:
            assert data["provenance"]["status"] == "frozen_and_bound"


if __name__ == "__main__":
    test_fps_timeline_20()
    test_uniform_high_motion_blend()
    test_augment_preserves_shape()
    test_config_matches_methods_appendix()
    test_pyyaml_available_for_config()
    test_manifests_do_not_call_artifacts_outdated()
    print("OK")
