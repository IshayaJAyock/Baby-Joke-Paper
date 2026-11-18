# Analysis

## Reference results (committed)

`reference_results/` is the default input for notebooks:

- `shared_fold_indices.json` + `cohort/` (row-order fingerprint)
- `all_models_performance.csv`
- `{multi-cue,3d_cnn,tan,cnn_lstm}/fold_*_predictions.npz`
- `{…}/fold_*_inner_splits.json` (inner-fit / monitor / outer-test)
- `{…}/run_manifest.json`
- `ablation/` aggregate tables (see `ablation/README.md`)
- `manuscript_tables/` derived T1–T13 CSVs (alignment-checked)

Protocol: **nested_on_training_children_then_refit** (inner ES patience 25 → refit on full outer-train for `best_epoch`).

## Notebooks

- `cross_validation_analysis.ipynb` — prefers `reference_results/`, else `../outputs/`
- `multicue_ablation.ipynb` — loads reference ablation aggregates

## Commands

```bash
make test
make analysis
python scripts/run_multimodal.py
python scripts/run_multimodal_ablation.py
python scripts/run_3d_cnn.py
python scripts/run_tan.py
python scripts/run_cnn_lstm.py
```
