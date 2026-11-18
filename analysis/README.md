Analysis
=======

This folder contains the statistical analysis of the child-held-out run, the figures, and a note on how to present them.

`presenting_results.md` is the guide for the chapter: one accuracy endpoint, Friedman then Nemenyi, and confusion matrices on a shared count scale.

`cross_validation_analysis.ipynb` reads `../main-output` and does not train. Run it from this directory after the training notebook.

`_extended_meaning.py` recomputes complementarity (oracle), ECE calibration, accuracy-vs-epochs, and per-child accuracy from the saved `fold_*_predictions.npz` files. It writes extra CSVs under `outputs/` and figures under `figures/`, and self-checks against pooled counts and McNemar discordant counts.

Contents:
- `cross_validation_analysis.ipynb`: Main analysis notebook (publication-quality plots)
- `_extended_meaning.py`: Extra meaning analyses without retraining
- `figures/`: Saved PNG/PDF figures
- `outputs/`: CSV summaries (bootstrap CIs, ranks, Wilcoxon, McNemar, ECE, per-child, …)
- `presenting_results.md`: How to put the results on the page

Reproduce:
1. Execute `cross_validation_analysis.ipynb` with kernel `ml` and working directory `analysis/`.
2. Run `python _extended_meaning.py` from `analysis/` (or via the repo-root relative path).

Note: Large figures and CSVs may be ignored by git; re-generate locally.
