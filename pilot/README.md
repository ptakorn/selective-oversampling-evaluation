# Synthetic fixed-parameter pilot

Run: `OPENBLAS_NUM_THREADS=1 python run_pilot.py`
Dependencies: Python, NumPy, pandas, SciPy, scikit-learn. Tested with scikit-learn 1.8.0.

30 paired replications, 2000 training observations, independent 10000-observation test samples; two Gaussian classes, separation 2, true imbalance approximately 10:1. Clean, dispersed and localized 10% class-conditional corruption. Same original datasets across methods and scenarios. Fixed a priori parameters, LogisticRegression C=1, no validation tuning. This is a pilot, NOT the complete nested-CV study described in the draft manuscript. Numeric continuous features only. SMOTE is directly implemented with five minority neighbors and uniform source sampling.

Files: run_pilot.py, config.json, pilot_raw.csv (270 runs), pilot_summary.csv, pilot_paired_intervals.csv, run.log.

Important findings: G-mean improves over SMOTE in both corrupted scenarios, but falls below SMOTE on clean data. Clean-data false downweighting affects about 11.4% of valid minority observations. Localized corruption downweighting recall is only 0.27%, while about 17.3% of valid minority observations are downweighted. Thus results do NOT demonstrate reliable separation of noise and overlap. Endpoint contamination describes involvement of deliberately flipped endpoints; it is not a synthetic-label correctness measure. Bootstrap intervals are individual paired intervals, not simultaneous confidence intervals. No WRND or prior-paper baseline included yet.

The test set retains clean labels and original class prevalence. Corruption changes observed training prevalence (181 minority points become 344); this is part of the specified symmetric class-conditional corruption experiment and affects the balancing budget.
