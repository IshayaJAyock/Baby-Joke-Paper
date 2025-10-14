Analysis
=======

This folder contains the statistical analysis notebook and generated figures/outputs for the cross-validation results.

Contents:
- `cross_validation_analysis.ipynb`: Main analysis notebook (publication-quality plots)
- `figures/`: Saved PNG figures (boxplots, scatter plots, etc.)
- `outputs/`: CSV summaries (bootstrap CIs, ranks, Wilcoxon, etc.)

Reproduce analysis:
```
make analysis
```
or run inside Jupyter and execute all cells.

Note: Large figures and CSVs are ignored by git; re-generate locally.

