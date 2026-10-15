# Benchmark decision-threshold control

# Protocol

Six original benchmarks; three repeated five-fold outer splits with exact-feature duplicates grouped; three inner folds. No intervention, SMOTE and complete framework; LR, RBF SVM and RF; supplied labels and 10% dispersed fitting corruption. Original classifier parameters (selected using default inner G-mean) are reused, not retuned using thresholds.

For each outer task and method, refit the three inner models at the previously selected classifier parameters. Use their held-out scores and supplied labels to choose a common threshold maximizing the unweighted mean of three fold-specific G-means. Candidates: distinct concatenated inner validation scores, descending; predictions positive for score >= threshold; first maximum wins. All-negative rule has G-mean zero and cannot improve a positive maximum. LR/SVM use decision scores, RF positive-class probabilities. The refitted outer model uses both default and selected rules on identical test observations. All splits, corruption, preprocessing, generation, and classifier seeds match prior experiments. Original fixed intervention parameters unchanged.

Threshold search is an additional model-selection stage within inner validation. Selected inner scores are optimistic and not performance estimates. Only outer scores assess performance. Threshold transfer across inner models and the final refit is a limitation, especially with uncalibrated scores. No clean-label guarantee is made for supplied benchmark labels. No injected corruption is applied to validation or test labels.

Statistical unit: dataset mean over all 15 folds. Two-sided Wilcoxon signed-rank with zero differences excluded; supplementary Holm family contains 12 selected-threshold comparisons (two baselines x three classifiers x two conditions). Default comparisons are descriptive here; earlier baseline-family adjustment remains unchanged. Six datasets offer limited power. No equivalence claim. Dataset-averaged performance weights each dataset equally.

Files: raw.csv, dataset_means.csv, summary.csv, paired_statistics.csv, verification.json; per-task checkpoints include OOF scores/fold identities and outer scores/labels for auditing. run.py imports original expanded modules; analyze.py generates report and LaTeX. Python dependencies match expanded/requirements.txt.


## Equal-dataset mean results

| noise_rate | classifier | method | rule | gmean |
| --- | --- | --- | --- | --- |
| 0.000000 | lr | none | default | 0.875982 |
| 0.000000 | lr | none | selected | 0.910513 |
| 0.000000 | lr | proposed | default | 0.895319 |
| 0.000000 | lr | proposed | selected | 0.905485 |
| 0.000000 | lr | smote | default | 0.916389 |
| 0.000000 | lr | smote | selected | 0.910819 |
| 0.000000 | rf | none | default | 0.902885 |
| 0.000000 | rf | none | selected | 0.930522 |
| 0.000000 | rf | proposed | default | 0.922773 |
| 0.000000 | rf | proposed | selected | 0.929411 |
| 0.000000 | rf | smote | default | 0.930343 |
| 0.000000 | rf | smote | selected | 0.932199 |
| 0.000000 | svm | none | default | 0.905253 |
| 0.000000 | svm | none | selected | 0.933180 |
| 0.000000 | svm | proposed | default | 0.925157 |
| 0.000000 | svm | proposed | selected | 0.932883 |
| 0.000000 | svm | smote | default | 0.934202 |
| 0.000000 | svm | smote | selected | 0.940940 |
| 0.100000 | lr | none | default | 0.846680 |
| 0.100000 | lr | none | selected | 0.898462 |
| 0.100000 | lr | proposed | default | 0.908134 |
| 0.100000 | lr | proposed | selected | 0.906243 |
| 0.100000 | lr | smote | default | 0.901161 |
| 0.100000 | lr | smote | selected | 0.909113 |
| 0.100000 | rf | none | default | 0.897067 |
| 0.100000 | rf | none | selected | 0.915592 |
| 0.100000 | rf | proposed | default | 0.923299 |
| 0.100000 | rf | proposed | selected | 0.924080 |
| 0.100000 | rf | smote | default | 0.921353 |
| 0.100000 | rf | smote | selected | 0.911001 |
| 0.100000 | svm | none | default | 0.901025 |
| 0.100000 | svm | none | selected | 0.925565 |
| 0.100000 | svm | proposed | default | 0.925930 |
| 0.100000 | svm | proposed | selected | 0.930612 |
| 0.100000 | svm | smote | default | 0.931987 |
| 0.100000 | svm | smote | selected | 0.934105 |

## Dataset-level comparisons

| noise_rate | classifier | rule | baseline | datasets | mean_difference | median_difference | wins | ties | raw_p | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | lr | default | none | 6 | 0.019337 | 0.005815 | 4 | 0 | 0.562500 | nan |
| 0.000000 | lr | default | smote | 6 | -0.021070 | 0.001235 | 3 | 0 | 0.843750 | nan |
| 0.000000 | lr | selected | none | 6 | -0.005028 | 0.003691 | 4 | 0 | 0.562500 | 1.000000 |
| 0.000000 | lr | selected | smote | 6 | -0.005334 | 0.001633 | 3 | 0 | 0.843750 | 1.000000 |
| 0.000000 | rf | default | none | 6 | 0.019888 | 0.001809 | 4 | 0 | 0.312500 | nan |
| 0.000000 | rf | default | smote | 6 | -0.007569 | -0.001764 | 2 | 0 | 0.437500 | nan |
| 0.000000 | rf | selected | none | 6 | -0.001111 | 0.000083 | 3 | 0 | 1.000000 | 1.000000 |
| 0.000000 | rf | selected | smote | 6 | -0.002788 | -0.000870 | 3 | 0 | 0.562500 | 1.000000 |
| 0.000000 | svm | default | none | 6 | 0.019905 | 0.000452 | 3 | 2 | 0.625000 | nan |
| 0.000000 | svm | default | smote | 6 | -0.009045 | -0.003691 | 1 | 1 | 0.125000 | nan |
| 0.000000 | svm | selected | none | 6 | -0.000297 | -0.000206 | 2 | 0 | 0.687500 | 1.000000 |
| 0.000000 | svm | selected | smote | 6 | -0.008057 | -0.001872 | 2 | 0 | 0.218750 | 1.000000 |
| 0.100000 | lr | default | none | 6 | 0.061454 | 0.021078 | 6 | 0 | 0.031250 | nan |
| 0.100000 | lr | default | smote | 6 | 0.006973 | 0.003043 | 4 | 0 | 0.156250 | nan |
| 0.100000 | lr | selected | none | 6 | 0.007781 | 0.003732 | 4 | 0 | 0.218750 | 1.000000 |
| 0.100000 | lr | selected | smote | 6 | -0.002870 | 0.002274 | 4 | 0 | 1.000000 | 1.000000 |
| 0.100000 | rf | default | none | 6 | 0.026231 | 0.003675 | 5 | 0 | 0.156250 | nan |
| 0.100000 | rf | default | smote | 6 | 0.001945 | 0.002348 | 4 | 0 | 0.687500 | nan |
| 0.100000 | rf | selected | none | 6 | 0.008488 | 0.000938 | 4 | 0 | 0.843750 | 1.000000 |
| 0.100000 | rf | selected | smote | 6 | 0.013079 | 0.004117 | 5 | 0 | 0.062500 | 0.750000 |
| 0.100000 | svm | default | none | 6 | 0.024905 | 0.003371 | 4 | 0 | 0.437500 | nan |
| 0.100000 | svm | default | smote | 6 | -0.006057 | -0.001550 | 3 | 0 | 0.437500 | nan |
| 0.100000 | svm | selected | none | 6 | 0.005047 | -0.000564 | 3 | 0 | 0.843750 | 1.000000 |
| 0.100000 | svm | selected | smote | 6 | -0.003493 | -0.000861 | 2 | 0 | 0.562500 | 1.000000 |

## Verification

{
  "default_models_checked": 1620,
  "max_gmean_difference": 0.0,
  "ranking_metrics_unchanged": true
}