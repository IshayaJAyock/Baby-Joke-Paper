# Reference results protocol

These artifacts match the nested ChildID-grouped evaluation contract used by
`src/cross_validation.py` and `analysis/cross_validation_analysis.ipynb`:

- Frozen splits: `shared_fold_indices.json` (1,404 clips, 88 infants, seed 3443)
- Protocol: `nested_inner_monitor_no_outer_refit`
  (inner monitor for callbacks; outer test scored once; **no** full outer-train refit)
- Outer fold predictions: `{multi-cue,3d_cnn,tan,cnn_lstm}/fold_*_predictions.npz`
  with `y_true`, `y_pred`, `y_proba`, `val_idx`
- Fold metrics: `all_models_performance.csv` (**single** 19-column schema via `src/artifact_io.py`)
- Multi-cue ablation: `ablation/ablation_summary.csv` and `T13_multicue_ablation.csv`
  (Full row = Multi-cue mean 0.9676)

Re-running `scripts/run_*.py` on `data/videos/` writes the same layout under `outputs/`.

Schema guard:

```bash
python tests/test_artifact_schema.py
```

Derived manuscript tables (alignment test): `manuscript_tables/`  
(T1–T13 / appendix CSVs + PNG figures). Checked against the fold CSV above — see `manuscript_tables/ALIGNMENT.md`.
