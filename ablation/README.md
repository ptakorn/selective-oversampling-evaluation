# Fixed-parameter development ablation

Run `OPENBLAS_NUM_THREADS=1 python run_ablation.py` from this folder's parent or any directory. Requires NumPy, pandas, SciPy and scikit-learn (tested 1.8.0). The parent `pilot` folder contains original results required for reproduction checks.

30 paired replications per condition; same data and seeds as the first pilot, 2000 training and 10000 independent test observations. Fixed parameters and LogisticRegression C=1; no tuning. 450 newly fitted models; baseline results reused, not rerun. This is DEVELOPMENT evidence, not independent confirmation of a selected new method.

Variants:
- full: original A-E formulation.
- no_downweight: original weights all one, synthetic weights unchanged.
- unit_synthetic: original downweighting retained, all synthetic weights one.
- unit_all: no original downweighting and unit synthetic weights.
- ambiguity_only: adaptive entropy source selection; no reliability/adequacy/reciprocity partner restrictions; no downweighting; unit synthetic weights; priority proportional to entropy.

Assertions verify the full implementation reproduces original pilot metrics to 1e-12, and weight ablations preserve exact synthetic coordinates and endpoints. All variants generated identical counts within a condition (1638 clean, 1312 corrupted). Ambiguity-only is a bundled comparison, not an isolated single-component test.

G-mean averages (clean, dispersed, localized):
full: .8205, .8390, .8350
no_downweight: .8165, .8391, .8368
unit_synthetic: .8290, .8397, .8162
unit_all: .8265, .8394, .8194
ambiguity_only: .8372, .8154, .8221
SMOTE: .8410, .7965, .7937

Removing original downweighting produces no clear G-mean change for dispersed noise (paired delta .00017, individual 95% bootstrap interval [-.00024,.00058]); localized noise improves slightly (+.00179,[.00112,.00245]). Unit synthetic weights reduce localized G-mean (-.01877,[-.02043,-.01718]). Do not interpret these intervals as simultaneous or multiplicity-adjusted.

Localized endpoint contamination is LOWER in ambiguity-only (.3229) than full (.4208), despite lower G-mean. Thus reliability-aware endpoint selection is not uniformly superior at avoiding known corrupted endpoints. F1 and other metrics may rank variants differently. Original-noise discrimination remains weak; no noise-detection claim is justified.

Next: characterize an augmentation-only candidate without original downweighting; freeze its specification using development evidence, then evaluate on new seeds/scenarios and additional classifiers with appropriate tuning and baselines. Do not call it validated or Q1-ready from these results.
