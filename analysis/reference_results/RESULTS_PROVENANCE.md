# Results provenance (audit note)

## Are PDF / notebook numbers tied to the public commit and frozen artifacts?

**Yes, for the numerical package in this folder.**

| Binding | Location |
|---|---|
| Outer ChildID folds | `config/shared_fold_indices.json` (`cohort_fingerprint.index_hash`) |
| Cohort row order (VideoID, ChildID, label) | `cohort/cohort_fingerprint.json` (`row_hash`) |
| Fold predictions (40 NPZs) | `{multi-cue,3d_cnn,tan,cnn_lstm}/fold_*_predictions.npz` |
| Fold metrics | `all_models_performance.csv` |
| SHA-256 of the above | each model’s `run_manifest.json` → `artifact_hashes` |
| Protocol encoded by the CSV | `nested_on_training_children_then_refit` (ES patience 25 → refit; `epochs_ran == best_epoch`; `n_train` = full outer-train) |

`analysis/cross_validation_analysis.ipynb` and `manuscript_tables/` read **these** files by default.

## Why did older manifests mention an “Oct-27 reference package”?

That wording was a temporary backfill note and was **misleading**. It is removed.

Clarification:

- The prediction NPZs / performance CSV **are** the frozen, committed reference artifacts for this repository.
- They are not an external orphaned dump; they are bound to the frozen fold indices and cohort `row_hash` above.
- Inner-split JSON sidecars document the nested fit/monitor/test partition under the current protocol.
- A full video retrain with `scripts/run_*.py` on the same commit regenerates the **same schema**; until that retrain is archived, these frozen files remain the canonical numerical evidence.

## Protocol vs PDF

Implemented training protocol (code + manifests):

1. Outer `StratifiedGroupKFold` on ChildID (seed 3443).
2. Inner ChildID monitor for early stopping / LR (patience **25**).
3. Refit on **all** outer-train children for `best_epoch` epochs.
4. Score outer test once (never as Keras `validation_data`).

Implemented preprocessing (code + README):

- 20 fps timeline, T=20, 224², uniform∪high-motion sampling, augmentation **enabled**.

Manuscript PDF text must match these statements; if the PDF still says “augmentation off” or “uniform linspace only,” that is a PDF edit, not a repository protocol.
