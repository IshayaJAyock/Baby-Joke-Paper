# Analysis fixtures

Column schemas the training pipeline must emit under `outputs/`:

- `all_models_performance.schema.csv` — fold-level metrics CSV
- `ablation_summary.schema.csv` — multi-cue ablation summary

Frozen ChildID splits live in `config/shared_fold_indices.json` (copied from the audited cohort split; 1,404 clips, seed 3443).
