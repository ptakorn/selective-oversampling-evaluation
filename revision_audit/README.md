# Revision evidence audit — 2026-10-15

This package extends the existing experiments without modifying their raw results or the public repository. It is an audit plus targeted confirmation study, not a completed rewritten article or a claim of preregistration.

## Start here

- report.md: existing benchmark/synthetic ROC-AUC/AP, all baseline means, budget audit and seed caveats.
- DECISIONS.md: interpretation and recommended manuscript direction.
- all_baselines_tables.tex: six supplementary LaTeX tables (three classifiers x two label conditions). D is the count of datasets with all 15 folds; different D values must not be ranked as though evaluated on identical data.
- confirmation/config.json: design frozen before the new model evaluations.
- confirmation/raw.csv, summary.csv, paired_intervals.csv: new complete results.
- confirmation/results_tables.tex: selected-threshold G-mean for every new scenario and method.

## Reproduce

Use Python 3.12 and the package versions in expanded/requirements.txt. The archive includes original code/data-result dependencies under expanded/ and pilot/; it does not redistribute benchmark feature arrays or third-party WRND sources. Run from the archive's root:

```
python revision_audit/analyze.py
python revision_audit/confirmation/run.py
python revision_audit/confirmation/analyze.py
```

The fixed-parameter new study contains 3 geometries x 3 corruption conditions x 30 independent replications x 5 methods x 2 classifiers = 2700 fitted models, each assessed with default and trusted-validation-selected thresholds. Original and new results are separate. Generated fitting/validation/test data use separate roles under NumPy SeedSequence [20261015,901,geometry,replication,role]. Data are shared across noise conditions for paired comparisons, not treated as independent conditions.

## New generators and corruption

N_train=1000, N_validation=2000, N_test=5000; pre-corruption minority count floor(N/11). Gaussian has identity covariance and minority shift (2,0). Moons uses class-conditional angles uniform [0,pi], coordinates (cos(a),sin(a)) and (1-cos(a),0.5-sin(a)); additive coordinate noise N(0,0.12^2). Circles uses angles uniform [0,2pi], radii 1 for majority and 0.45 for minority, with the same coordinate noise. LR is intentionally a simple linear comparison; its circle-model misspecification must not be hidden.

Corruption is clean, dispersed, or boundary-near. Both noisy conditions flip floor(0.1*n_c) within each true class. Boundary-near selects observations closest to any opposite-true-class fitting observation, using raw feature distance. True labels are used only by the controlled data generator and evaluation, not the detector. This is a stylized stress mechanism, not evidence about prevalence of noise in applications. Validation/test labels are uncorrupted.

LR: C=1, max_iter=2000. SVM: C=1, RBF gamma=1/d. Intervention parameters unchanged. No classifier tuning. Threshold maximizes validation G-mean over distinct scores descending, first maximum; no model refit after validation selection. All methods receive the same data and threshold-selection opportunity.

## Count-and-weight control

smote_matched_weights uses SMOTE with exactly the realized synthetic count of complete, identical original weights, and exactly the same multiset of synthetic weights assigned by an independent random permutation to its generated points. Count and both weight sums are asserted equal. This controls overall training mass without requiring identical spatial assignment. Differences concern generation/selection geometry and geometry-weight association together; they cannot isolate one internal score or establish noise-label recovery. If complete generates zero samples, the control also generates zero.

Bootstrap intervals: 5000 paired resamples of 30 replications within each setting. Individual, exploratory, not simultaneous. Reused data across conditions must not be pooled as independent replications. This small fixed-parameter confirmation is not a replacement for larger benchmark validation.
