# Adaptive versus fixed-neighborhood ablation

Only neighborhood membership changes: adaptive bounded-radius selection versus fixed counts (5,10,20). The base count is 10. Fixed selection keeps the original distance-weight formula, multi-scale assessment, reference radius for expansion adequacy, thresholds, reciprocal support, allocation, endpoint criteria and original/synthetic weights. Thus this is a membership-selection ablation, not removal of every adaptive-radius dependency.

Synthetic: same 63 grid settings and 17 stress settings, 30 replications, seeds/data/corruption as the previous expanded experiments. Both variants rerun; adaptive scores checked against original raw results.
Benchmark: same six datasets, 15 outer folds, two conditions, three classifiers; fixed variant repeats complete inner classifier selection. Original adaptive benchmark results reused. Same preprocessing, folds, corruption and fitting seeds.

Synthetic CIs: 5000 paired bootstrap resamples within a setting; individual 95% intervals, no simultaneous coverage claim. Benchmark inference: dataset-level paired differences from complete 15-fold means, two-sided Wilcoxon, Holm adjustment over the six new ablation comparisons (a supplementary family separate from earlier baseline tests). Folds are not independent datasets.

Run from this experiment workspace: python neighborhood_ablation/run.py then python neighborhood_ablation/analyze.py. Original expanded/data and runner modules required. Checkpoints allow resume. No parameters selected using test outcomes.
