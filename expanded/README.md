# Expanded synthetic and benchmark evaluation

This package reports actual runs, not the entire proposed Section IV. Read config.json for the executed scope and pending work.

## Reproduction
1. Install requirements.txt in a new Python environment (Python 3.12 used here).
2. Run `python setup_wrnd.py` to fetch pinned author baseline code. Third-party source is NOT redistributed in this package.
3. Datasets are provided as NPZ files with attribution in data/manifest.json. To retrieve original UCI files again, run `python download_data.py`.
4. Run `python run_synthetic_grid.py`, `python run_stress.py`, `python run_threshold_control.py`, and `python run_benchmarks.py`.
5. Run `python analyze_results.py` after benchmark completion.

Benchmark runner resumes from results_v2/*.json. Remove those completed-result files only if you intentionally want a full rerun or change a configuration; cached results are for exactly this version. The runner uses 8 process workers; reduce n_jobs for a smaller computer.

## Executed protocols
Synthetic grid: 63 settings, 30 paired repetitions, 2000 training/10000 independent test observations, separation 1/2/3, true IR 5/10/20, clean or .05/.10/.20 dispersed/localized training-label corruption; fixed LR C=1. 11340 final evaluations across six methods. No intervention parameter tuning.
Stress study: 17 settings, 30 repetitions, 3060 final evaluations. Minority subgroup, valid isolation, training-feature contamination, and 2/10/50 dimensions. Fixed LR C=1; no injected label corruption in this study.
Threshold control: 30 independent repetitions, three conditions and four methods, fixed LR C=1; separate trusted validation set chooses a G-mean threshold. 720 evaluated decision rules from 360 model fits. Test labels and scores never choose the threshold.
Benchmarks: six preselected datasets, 5 outer folds x 3 repetitions, 3 inner folds, three classifiers, supplied labels or 10% fitting-label corruption. Intervention parameters fixed; only classifier parameters selected using inner G-mean. Exact-feature duplicates remain in the same inner/outer folds. Scaling and variance exclusion fit only on fitting data. Nominally 4320 final models, with failures reported explicitly.

## Benchmark definitions and attribution
- WDBC, UCI 17: malignant positive; 569 x 30, 212 positive. Data from scikit-learn's packaged copy. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic ; DOI 10.24432/C5DW2B.
- Banknote Authentication, UCI 267: original class 1 positive; 1372 x 4, 610 positive. https://archive.ics.uci.edu/dataset/267/banknote+authentication ; DOI 10.24432/C55P57.
- Spambase, UCI 94: spam positive; 4601 x 57, 1813 positive. https://archive.ics.uci.edu/dataset/94/spambase .
- Ionosphere, UCI 52: bad return positive; 351 x 32 after removing first binary and second constant features; 126 positive. https://archive.ics.uci.edu/dataset/52/ionosphere .
- Ecoli-imU (five continuous features): imU vs rest; 336 x 5, 35 positive. ID and binary lip/chg removed. Derived binary task, NOT the full original multiclass dataset. https://archive.ics.uci.edu/dataset/39/ecoli ; DOI 10.24432/C5388M.
- Glass-7: class 7 headlamps vs rest; 214 x 9, 29 positive. ID removed. Derived binary task. https://archive.ics.uci.edu/dataset/42/glass+identification .
UCI dataset licenses and creator attribution should accompany manuscript citations. Selection and binary mappings were fixed before results were examined. Several datasets are only mildly imbalanced; their original proportions were preserved.

## Baselines and limitations
SMOTE, Borderline-SMOTE1, ADASYN, ROS: imbalanced-learn 0.14.2 with default neighbor settings (generation k=5 bounded by minority count), target ratio 1. Proposed and no_downweight use original fixed development parameters in methods.py. No original observations are relabeled or removed by those methods.
WRND author-code adapter: retrieves Natural_Neighbor and unmodified generate_samples_Lm from dream-lm/WRND_framework commit f69a5c71fdff7dd3ff9b52c9d96801c3a931cc99. Adapter reproduces author SMOTE _fit_resample logic without importing obsolete imbalanced-learn internals. It retains original observations and filters augmentation sources through positive relative density, as the code does. It is identified as an adapted author-code baseline; equivalence to every paper detail is not assumed. Known zero-distance/empty-generation issues are reported as failures; there is no silent fallback.
The prior IEEE ambiguity framework is not yet included: an unverified substitute would not be a faithful baseline.
No 50-configuration joint intervention search, corrupted-validation sensitivity, complete ablations, or runtime scaling study has been completed.

## Interpretation
The original fixed-threshold Gaussian gains shrink when every method selects its threshold from clean validation. All six individual paired 95% intervals for proposed-minus-none/SMOTE in that supplementary control include zero. This undermines a claim that the original Gaussian gains demonstrate improved discrimination. Stress cases also show that the method does not dominate on clean data or irrelevant dimensions. Read detailed per-setting CSVs; do not use pooled means as proof of universal superiority.

Bootstrap intervals are individual, not simultaneous. Benchmark statistical comparisons use one mean per dataset, require all 15 folds for each paired method, and Holm-adjust the entire set of reported primary pairwise tests. Failed runs are not assigned artificial scores.
