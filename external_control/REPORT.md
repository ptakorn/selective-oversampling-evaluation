# External-method control: results and interpretation

## Why SMOTE-ENN rather than WRND
The earlier WRND adapter is not sufficiently reconciled with the full published algorithm to label it a verified external-method confirmation. The pinned code has a rank-2 start and rank-5 cap in natural-neighbor search, and wrapper seed/neighbor/filtering semantics still require reconciliation. These observations are unresolved fidelity issues, not confirmed defects in the published method. No patched WRND results are substituted.

The replacement is a standard external hybrid method: imbalanced-learn SMOTE-ENN, version 0.14.2, explicitly configured to SMOTE ratio 1/k=5 followed by ENN all classes/3 neighbors/kind_sel=all. The public implementation, not a bespoke replacement, generates the data. Instrumentation observes generation indices without changing random draws or coordinates. Nine fitting-only preflight cases reproduce the uninstrumented output exactly.

## Completed experiment
1620 new models, 3240 evaluations; all successful without fitting warnings. 2160 prior no-intervention/SMOTE evaluations are reused. The copied generator agrees with the original across 270 role checks. All 270 resampling cases satisfy exact count and retention matching.

## Descriptive SMOTE-ENN minus comparator G-mean
Each number weights nine setting means equally. It is not an independent-dataset pooled effect estimate.

| Classifier | Comparator | Default | Validation-selected |
|---|---|---:|---:|
| LR | none | +0.243571 | -0.002618 |
| LR | smote | -0.010971 | -0.000305 |
| LR | smote_matched_count | +0.041038 | -0.000309 |
| LR | smote_matched_retention | +0.010309 | +0.000402 |
| SVM | none | +0.055477 | -0.003008 |
| SVM | smote | -0.004731 | -0.002569 |
| SVM | smote_matched_count | -0.001980 | -0.004759 |
| SVM | smote_matched_retention | +0.011457 | +0.005824 |

## What the experiment adds
1. Default-rule gains over no intervention are much larger than selected-threshold differences for this external method too. This corroborates operating-point assessment beyond the proposed framework within these conditions.
2. Matching synthetic counts and matching original retention answer different questions. The SVM mean difference changes from -0.004759 against count-only SMOTE to +0.005824 against count-and-retention-matched SMOTE. This does not show superiority over ordinary SMOTE: the corresponding selected-threshold difference is -0.002569.
3. With dispersed corruption, SVM differences against the retention-matched control are +0.010177 Gaussian, +0.016010 moons and +0.030624 circles; individual 95% intervals are above zero. Boundary-near cases include adverse outcomes. Thus the results reject a uniform narrative that all hybrid effects are merely threshold/count artifacts.
4. Editing can reduce corrupted-endpoint use while removing valid minority originals. Under dispersed corruption, mean endpoint contamination changes from 0.574085 before editing to 0.365621 after editing; valid minority removal averages 0.141564. These are separate outcomes from predictive utility.

## What it does not establish
No general superiority, universal noise identification, equivalence or broad claim across modern noise-aware oversamplers is established. Only one additional hybrid method, two classifiers and nine synthetic settings are evaluated. The experiment was designed after earlier results. All intervals are individual/exploratory and no multiplicity-adjusted significance claim is made. Count-and-retention matching changes retained synthetic coordinates and endpoint composition together. Endpoint contamination does not label every synthetic observation as correct/incorrect.

## Implementation and reproducibility sources
- https://imbalanced-learn.org/stable/references/generated/imblearn.combine.SMOTEENN.html
- https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html
- Batista, Prati and Monard (2004), https://www.kdd.org/exploration_files/batista.pdf

See config.json for the frozen protocol, implementation_manifest.json for software/source hashes, and verification.json for structural checks. No public GitHub update has been performed.
