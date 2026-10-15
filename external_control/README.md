# External hybrid-method follow-up — 15 October 2026

This package adds an independent implementation family to the selective-oversampling evaluation. It evaluates the explicitly configured SMOTE-ENN implementation in imbalanced-learn 0.14.2, not a verified WRND reproduction and not all modern noise-aware oversamplers.

## Completed work

- Three geometries × three corruption conditions × 30 replications.
- LR and RBF-SVM, with fixed settings; independent trusted validation selects thresholds.
- Three new configurations: SMOTE-ENN, synthetic-count-matched SMOTE, and synthetic-count-plus-original-retention-matched SMOTE.
- **1620 new model fits, 3240 decision-rule evaluations, zero failures or fitting warnings.**
- Reuses 2160 no-intervention / ordinary-SMOTE evaluations from the identical earlier targeted experiment.
- 432 individual paired bootstrap intervals, including 72 selected-threshold G-mean intervals.
- All 270 count/retention checks pass. Independent copied generator exactly matches original generator in 270 data-role comparisons.

## Files to read

- `REPORT.md`: findings, scope, caveats and implementation gate.
- `config.json`: protocol frozen after fitting-only preflight and before comparative held-out results.
- `preflight.json`, `verification.json`: structural and numerical checks.
- `new_raw.csv`, `all_raw.csv`: new results and results with reused baselines.
- `summary.csv`, `summary_with_sd.csv`: every setting and metric.
- `paired_intervals.csv`: all metrics, rules and four comparators.
- `selected_gmean_intervals.csv`: selected-threshold G-mean subset.
- `intervention_audit.csv`, `intervention_summary.csv`: original removal, retained synthetics and endpoint contamination.
- `provenance/`: original and synthetic survival indices and SMOTE endpoint indices for all 270 resampling cases.
- `external_supplement.pdf`: eight E-numbered tables, kept separate from prior S1--S15.
- `protocol_insert.tex`, `results_insert.tex`, `discussion_insert.tex`, `bibitem_insert.tex`: manuscript insert drafts.
- `external_control_intervals.png`: figure; copy to the manuscript's `figures` directory if using the optional figure environment in results_insert.tex.

## Reproduce

Use Python 3.12 and the versions in `requirements.txt`.

```bash
cd external_control
python preflight.py
python run.py
python analyze.py
python make_deliverables.py
pdflatex external_supplement.tex
pdflatex external_supplement.tex
```

`run.py` reuses completed checkpoint files. For a fresh rerun, remove only checkpoint `*.json` files while retaining the `checkpoints/` directory and `.gitkeep`; the frozen runner expects that directory. No alternative method or artificial scores are substituted on failure. Full 30-replication pairing is required for reported interval comparisons. The plotting/table script regenerates numerical tables; the rendered supplement includes additional explanatory introductory prose.

## Manuscript placement

- Protocol: append to Section IV after the targeted-extension protocol.
- Results: append to Section V, before Discussion.
- Discussion: integrate into the evidence-scope discussion; do not leave an older claim saying hybrid methods were never evaluated.
- Bibliography: add the supplied Batista entry in first-citation order, retaining existing citation keys.
- If adding this study, also update abstract, evaluation coverage and code availability. This package alone does not perform those full-manuscript edits.
- E1--E8 are a separate addendum. They may be renumbered S16--S23 if combined with the existing supplementary document, with references updated consistently.

## Interpretation

The study was designed after earlier results. It uses the same nine synthetic conditions, not new independent benchmark datasets. Cross-setting averages are descriptive. Bootstrap intervals are exploratory individual intervals, not simultaneous intervals and not multiplicity-adjusted significance tests. Fixed synthetic classifier settings do not establish performance after joint tuning. The external SMOTE-ENN configuration cleans both classes with ENN kind_sel='all'; it is not presented as exact reproduction of the 2004 paper's experiments.

Count matching alone does not match original deletion. The retention-matched control retains exactly the same originals and uses the same count-matched synthetic observations as its corresponding count-only control. It does not re-run ENN on those synthetics. This matches original and synthetic training mass while changing synthetic coordinates and endpoint composition together, and therefore is not a pure causal estimate of location.

No updates to the public GitHub repository have been made. Do not cite the older public commit as containing this follow-up until the package is archived there and a real updated commit/release exists.
