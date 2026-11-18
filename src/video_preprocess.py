"""
Methods-aligned video preprocessing (methods.md §Preprocessing).

Implements:
  - Standardisation to target FPS (default 20) and 224×224 RGB
  - Fixed-length T=20 sequences with intelligent edge padding
  - Uniform temporal sampling blended with high optical-flow frame selection
  - Spatial augmentations (rotation, brightness, horizontal flip) and
    temporal perturbations (jitter, stochastic frame dropping)

Audio is never decoded. This module is shared by RGB and multi-cue runners.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _pad_to_length(frames: List[np.ndarray], length: int) -> List[np.ndarray]:
    """Intelligent padding: repeat edge frames to reach fixed length."""
    if not frames:
        raise ValueError("Cannot pad an empty frame list")
    if len(frames) >= length:
        return frames[:length]
    out = list(frames)
    # Alternate forward/backward edge fill to avoid a single frozen frame.
    while len(out) < length:
        out.append(frames[-1].copy())
        if len(out) >= length:
            break
        out.insert(0, frames[0].copy())
    return out[:length]


def _read_all_bgr(video_path: str) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames in video: {video_path}")
    return frames, fps


def standardize_fps_timeline(
    n_src: int,
    src_fps: float,
    target_fps: float = 20.0,
) -> np.ndarray:
    """Map a target-FPS timeline onto source frame indices (methods: 20 fps)."""
    if n_src <= 0:
        return np.zeros(0, dtype=np.int64)
    duration = n_src / max(src_fps, 1e-6)
    n_tgt = max(1, int(round(duration * target_fps)))
    # Continuous times on the target grid → nearest source index.
    t = np.linspace(0.0, duration, n_tgt, endpoint=False)
    idx = np.clip(np.round(t * src_fps).astype(np.int64), 0, n_src - 1)
    return idx


def optical_flow_magnitudes(frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """Per-frame mean Farneback flow magnitude (frame 0 = 0)."""
    if len(frames_bgr) == 0:
        return np.zeros(0, dtype=np.float32)
    mags = np.zeros(len(frames_bgr), dtype=np.float32)
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames_bgr)):
        cur = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags[i] = float(np.mean(mag))
        prev = cur
    return mags


def select_frame_indices(
    n_frames: int,
    max_frames: int,
    flow_mags: Optional[np.ndarray] = None,
    high_motion_ratio: float = 0.5,
) -> np.ndarray:
    """
    Blend uniform sampling with high-motion selection, then sort temporally.

    methods.md: uniform fixed-length sequences + complementary high-flow frames.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)

    n_motion = int(np.clip(round(max_frames * float(high_motion_ratio)), 0, max_frames))
    n_uniform = max_frames - n_motion
    uniform = {
        int(i)
        for i in np.linspace(0, n_frames - 1, max(1, n_uniform), dtype=np.int64)
    }
    chosen = set(uniform)

    if flow_mags is not None and len(flow_mags) == n_frames and n_motion > 0:
        for i in np.argsort(-np.asarray(flow_mags)):
            ii = int(i)
            if ii in chosen:
                continue
            chosen.add(ii)
            if len(chosen) >= max_frames:
                break

    # Top up from a dense uniform grid if still short.
    if len(chosen) < max_frames:
        for i in np.linspace(0, n_frames - 1, max_frames * 3, dtype=np.int64):
            chosen.add(int(i))
            if len(chosen) >= max_frames:
                break

    ordered = sorted(chosen)
    if len(ordered) > max_frames:
        pick = np.linspace(0, len(ordered) - 1, max_frames, dtype=np.int64)
        ordered = [ordered[int(i)] for i in pick]
    return np.asarray(ordered, dtype=np.int64)


def sample_video_frames(
    video_path: str,
    max_frames: int = 20,
    target_size: Tuple[int, int] = (224, 224),
    target_fps: float = 20.0,
    use_high_motion: bool = True,
    high_motion_ratio: float = 0.5,
    to_rgb: bool = True,
    normalize: bool = False,
) -> Optional[np.ndarray]:
    """
    Decode → FPS timeline → (uniform ∪ high-motion) selection → resize → pad.

    Returns array (T, H, W, 3) uint8 RGB (or float32 [0,1] if normalize).
    """
    try:
        raw, src_fps = _read_all_bgr(video_path)
        # Build a target-fps proxy sequence by index remapping, then select T frames.
        timeline = standardize_fps_timeline(len(raw), src_fps, target_fps=target_fps)
        timed = [raw[i] for i in timeline]
        flow = optical_flow_magnitudes(timed) if use_high_motion else None
        idx = select_frame_indices(
            len(timed), max_frames, flow_mags=flow, high_motion_ratio=high_motion_ratio
        )
        frames = []
        for i in idx:
            fr = timed[int(i)]
            fr = cv2.resize(fr, (target_size[1], target_size[0]))
            if to_rgb:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(fr)
        frames = _pad_to_length(frames, max_frames)
        arr = np.asarray(frames)
        if normalize:
            return arr.astype(np.float32) / 255.0
        return arr
    except Exception as e:
        logger.error(f"sample_video_frames failed for {video_path}: {e}")
        return None


def augment_clip(
    frames: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    rotation_range: float = 15.0,
    brightness_delta: float = 0.2,
    horizontal_flip: bool = True,
    temporal_jitter: bool = True,
    stochastic_drop_prob: float = 0.1,
) -> np.ndarray:
    """
    Spatial + temporal augmentation (methods.md).

    ``frames``: (T,H,W,C) float[0,1] or uint8.
    """
    rng = rng or np.random.default_rng()
    x = frames.astype(np.float32, copy=True)
    scale = 255.0 if x.max() > 1.5 else 1.0
    x = x / scale  # work in [0,1]

    t, h, w, c = x.shape

    # Horizontal flip (same for whole clip)
    if horizontal_flip and rng.random() < 0.5:
        x = x[:, :, ::-1, :]

    # Small rotation (degrees)
    if rotation_range and rng.random() < 0.5:
        angle = float(rng.uniform(-rotation_range, rotation_range))
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        rot = []
        for i in range(t):
            fr = cv2.warpAffine(
                x[i], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            if fr.ndim == 2:
                fr = fr[..., None]
            rot.append(fr)
        x = np.stack(rot, axis=0)

    # Brightness
    if brightness_delta and rng.random() < 0.5:
        delta = float(rng.uniform(-brightness_delta, brightness_delta))
        x = np.clip(x + delta, 0.0, 1.0)

    # Temporal jitter: circular shift by ±1–2 frames
    if temporal_jitter and t > 2 and rng.random() < 0.5:
        shift = int(rng.integers(-2, 3))
        if shift:
            x = np.roll(x, shift=shift, axis=0)

    # Stochastic frame dropping: replace random frames with neighbours
    if stochastic_drop_prob > 0 and t > 2:
        for i in range(1, t - 1):
            if rng.random() < stochastic_drop_prob:
                x[i] = x[i - 1] if rng.random() < 0.5 else x[i + 1]

    if scale > 1.5:
        return np.clip(x * 255.0, 0, 255).astype(frames.dtype)
    return x.astype(np.float32)


def methods_preprocess_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Defaults matching methods.md Appendix A.3 / preprocessing section."""
    cfg = dict(config or {})
    return {
        "max_frames": int(cfg.get("max_frames", cfg.get("num_frames", 20))),
        "frame_height": int(cfg.get("frame_height", cfg.get("frame_size", (224, 224))[0] if isinstance(cfg.get("frame_size"), tuple) else cfg.get("frame_height", 224))),
        "frame_width": int(cfg.get("frame_width", cfg.get("frame_size", (224, 224))[1] if isinstance(cfg.get("frame_size"), tuple) else cfg.get("frame_width", 224))),
        "target_fps": float(cfg.get("target_fps", 20.0)),
        "use_high_motion": bool(cfg.get("use_high_motion", True)),
        "high_motion_ratio": float(cfg.get("high_motion_ratio", 0.5)),
        "enable_augmentation": bool(cfg.get("enable_augmentation", True)),
        "rotation_range": float(cfg.get("rotation_range", 15.0)),
        "brightness_delta": float(cfg.get("brightness_range", cfg.get("brightness_delta", 0.2))),
        "horizontal_flip": bool(cfg.get("horizontal_flip", True)),
        "stochastic_drop_prob": float(cfg.get("stochastic_drop_prob", 0.1)),
        "label_smoothing": float(cfg.get("label_smoothing", 0.05)),
        "early_stopping_patience": int(cfg.get("early_stopping_patience", 25)),
        "batch_size_3d": int(cfg.get("batch_size_3d", 8)),
    }
