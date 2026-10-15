> Historical mapping for the earlier snapshot. See NEW_EVIDENCE_MAPPING.md for the latest follow-up and supplementary S1--S23. Main table numbers may have shifted.

# Manuscript evidence mapping

Use table titles rather than numbers, which change when tables are inserted.

| Evidence | Raw source | Derived statistics |
|---|---|---|
| Initial development classification/intervention | pilot/pilot_raw.csv | pilot/pilot_summary.csv; pilot/pilot_paired_intervals.csv |
| Weighting ablation | ablation/ablation_raw.csv; combined_raw.csv | ablation/summary.csv; paired_intervals.csv |
| Synthetic grid | expanded/synthetic_grid_raw.csv | synthetic_grid_summary.csv; synthetic_paired_intervals.csv |
| Structural stress | expanded/stress_raw.csv | stress_summary.csv |
| Benchmark SMOTE/complete paired means | expanded/benchmark_raw.csv | benchmark_dataset_means.csv; benchmark_paired_statistics.csv |
| Gaussian threshold control | expanded/threshold_control_raw.csv | threshold_control_intervals.csv |
| Adaptive versus fixed | neighborhood_ablation/synthetic_raw.csv; benchmark_raw.csv | synthetic_paired_summary.csv; benchmark_dataset_means.csv; benchmark_statistics.csv |
| Benchmark threshold control | benchmark_threshold/raw.csv | dataset_means.csv; summary.csv; paired_statistics.csv |

Raw differences are computed before rounding. G-mean combines minority recall and majority specificity. AP is average precision, not trapezoidal PR area. Endpoint contamination measures corrupt endpoints, not known true labels of generated points. Cases with no eligible sources can have zero augmentation. Failure records remain in the original baseline raw data.
