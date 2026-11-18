Analysis
=======

Statistical notebooks and helpers for child-held-out runs produced by `scripts/run_*.py`.

## Video training entrypoints (repo root)

```bash
python scripts/run_multimodal.py                 # full multi-cue (attention + gating)
python scripts/run_multimodal.py --ablation concat
python scripts/run_multimodal_ablation.py        # all cue / fusion ablations
python scripts/run_3d_cnn.py
python scripts/run_tan.py
python scripts/run_cnn_lstm.py
```

Features come from `data/videos/` (Farneback / MediaPipe / ResNet / RGB). Results default to `outputs/`.

## Notebooks in this folder

- `multicue_ablation.ipynb` — runs / plots multi-cue ablations via `scripts/run_multimodal_ablation.py`. Writes under `outputs/ablation_multicue/`.
- `cross_validation_analysis.ipynb` — does **not** train; reads CV scores from `../outputs`.

## Reproduce

1. Train models with the scripts above (videos under `data/videos/`).
2. From `analysis/`, run `multicue_ablation.ipynb` and/or `cross_validation_analysis.ipynb`.

Note: Large figures and CSVs under `figures/` / `outputs/` may be gitignored; regenerate locally.
