# Manuscript tables alignment check

Source: local `f2-result/tables` + `appendix_tables` (+ PNG figures).
Checked against `analysis/reference_results/all_models_performance.csv` (same 40 fold rows / npz package).

| Check | Result |
|---|---|
| T2 fold_mean_acc vs ref model means | PASS |
| Appendix fold_accuracy_pct vs ref folds | PASS (max |Δ|=4.93151e-05 pp) |
| T13 Full == Multi-cue mean | PASS |
| T9 best_epoch_mean vs ref | PASS |

This package is a **derived table test** for notebook/manuscript numbers against the committed fold artifacts.
