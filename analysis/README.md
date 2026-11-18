Analysis
=======

## Reference results (committed)

`reference_results/` holds the nested ChildID-fold package the notebooks read by default:

- `shared_fold_indices.json`, `all_models_performance.csv`
- `{multi-cue,3d_cnn,tan,cnn_lstm}/fold_*_predictions.npz`
- `ablation/ablation_summary.csv` (Full = Multi-cue 0.9676)

Protocol: see `reference_results/PROTOCOL.md`.

## Notebooks

- `cross_validation_analysis.ipynb` — prefers `reference_results/`, else `../outputs/` after a local retrain
- `multicue_ablation.ipynb` — loads reference ablation; optional video re-run

## Training entrypoints (repo root)

```bash
python scripts/run_multimodal.py
python scripts/run_multimodal_ablation.py
python scripts/run_3d_cnn.py
python scripts/run_tan.py
python scripts/run_cnn_lstm.py
```
