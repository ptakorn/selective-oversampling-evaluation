# Adaptive versus fixed-neighborhood ablation

| study | settings | adaptive | fixed | difference | wins | losses | individual_ci_positive | individual_ci_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid | 63 | 0.796029 | 0.794939 | 0.001090 | 41 | 22 | 11 | 8 |
| stress | 17 | 0.867748 | 0.866100 | 0.001648 | 14 | 3 | 3 | 1 |

These aggregate means weight each setting equally; there is no pooled-replication CI or claim of simultaneous significance. Full individual paired intervals are in synthetic_paired_summary.csv.

## Benchmark dataset-level comparisons

| noise_rate | classifier | datasets | mean_difference | median_difference | raw_p | holm_p |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | lr | 6 | -0.009367 | 0.000654 | 0.843750 | 1.000000 |
| 0.000000 | rf | 6 | 0.008399 | 0.001394 | 0.031250 | 0.187500 |
| 0.000000 | svm | 6 | 0.000194 | 0.000221 | 0.875000 | 1.000000 |
| 0.100000 | lr | 6 | -0.001132 | 0.001916 | 0.562500 | 1.000000 |
| 0.100000 | rf | 6 | 0.004157 | 0.001378 | 0.218750 | 1.000000 |
| 0.100000 | svm | 6 | -0.001284 | -0.000685 | 0.437500 | 1.000000 |

The supplementary Holm family includes these six new comparisons. Previously reported 42 baseline comparisons are unchanged.

# Adaptive versus fixed-neighborhood ablation

Only neighborhood membership changes: adaptive bounded-radius selection versus fixed counts (5,10,20). The base count is 10. Fixed selection keeps the original distance-weight formula, multi-scale assessment, reference radius for expansion adequacy, thresholds, reciprocal support, allocation, endpoint criteria and original/synthetic weights. Thus this is a membership-selection ablation, not removal of every adaptive-radius dependency.

Synthetic: same 63 grid settings and 17 stress settings, 30 replications, seeds/data/corruption as the previous expanded experiments. Both variants rerun; adaptive scores checked against original raw results.
Benchmark: same six datasets, 15 outer folds, two conditions, three classifiers; fixed variant repeats complete inner classifier selection. Original adaptive benchmark results reused. Same preprocessing, folds, corruption and fitting seeds.

Synthetic CIs: 5000 paired bootstrap resamples within a setting; individual 95% intervals, no simultaneous coverage claim. Benchmark inference: dataset-level paired differences from complete 15-fold means, two-sided Wilcoxon, Holm adjustment over the six new ablation comparisons (a supplementary family separate from earlier baseline tests). Folds are not independent datasets.

Run from this experiment workspace: python neighborhood_ablation/run.py then python neighborhood_ablation/analyze.py. Original expanded/data and runner modules required. Checkpoints allow resume. No parameters selected using test outcomes.


## Verification

{
  "grid": {
    "rows": 1890,
    "max_gmean_difference": 0.0
  },
  "stress": {
    "rows": 510,
    "max_gmean_difference": 0.0
  }
}