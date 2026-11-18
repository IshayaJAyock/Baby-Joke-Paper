Analysis
=======

Statistical notebooks for child-held-out runs produced by `scripts/run_*.py`.

## Training entrypoints (repo root)

```bash
python scripts/run_multimodal.py
python scripts/run_multimodal.py --ablation concat
python scripts/run_multimodal_ablation.py
python scripts/run_3d_cnn.py
python scripts/run_tan.py
python scripts/run_cnn_lstm.py
```

Features come from `data/videos/`. Nested ChildID folds are frozen in `config/shared_fold_indices.json` and copied into `outputs/` on run.

## Notebooks

- `multicue_ablation.ipynb` — ablation via `scripts/run_multimodal_ablation.py`
- `cross_validation_analysis.ipynb` — reads `../outputs` (`all_models_performance.csv`, `shared_fold_indices.json`, `*/fold_*_predictions.npz`)

## Fixtures

`fixtures/` stores CSV **schemas** the pipeline must match (not result numbers).
