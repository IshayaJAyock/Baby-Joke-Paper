# Multi-cue ablation artifacts

## Present (aggregate)
- `ablation_summary.csv` / `T13_multicue_ablation.csv` — mean±SD over 10 outer folds
- Full row locked to Multi-cue mean **0.967582**

## Produced by a video ablation re-run
`python scripts/run_multimodal_ablation.py` writes, for each variant under
`outputs/ablation_multicue/<variant>/`:

- `fold_*_predictions.npz` (y_true, y_pred, y_proba, val_idx)
- `fold_*_inner_splits.json` (fit / monitor / test indices + ChildIDs)
- `all_models_performance.csv` (same 19-column schema)
- `run_manifest.json`

Until that re-run is archived here, ablation **fold-level** predictions are not
independently auditable from this folder alone—only the aggregate table is.
