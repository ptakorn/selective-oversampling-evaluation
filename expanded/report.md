# Expanded experiment report

This is an expanded evaluation of the original fixed intervention, not the entire planned Section IV. All failures are reported; test data were not used to select parameters or thresholds.

## Selected benchmarks

|Dataset|N|Features|Minority|IR|
|---|---:|---:|---:|---:|
|wdbc|569|30|212|1.68|
|banknote|1372|4|610|1.25|
|spambase|4601|57|1813|1.54|
|ionosphere|351|32|126|1.79|
|ecoli_imu|336|5|35|8.60|
|glass7|214|9|29|6.38|

## Execution counts

- Synthetic grid: 11,340 successful method evaluations (63 settings x 30 repetitions x 6 methods).
- Stress study: 3,060 successful method evaluations (17 settings x 30 repetitions x 6 methods).
- Threshold control: 360 model fits, 720 decision-rule evaluations.
- Benchmarks: 4320 attempted final fits; 4107 successful and 213 failed. Inner-validation fits are additional.

## Benchmark proposed-minus-SMOTE G-mean

Each entry uses the mean of all 15 outer folds for the dataset. Values are absolute differences; no folds are treated as independent datasets.

|Dataset|Classifier|Supplied labels|10% fitting-label corruption|
|---|---|---:|---:|
|banknote|lr|-0.0009|+0.0053|
|banknote|rf|+0.0015|+0.0032|
|banknote|svm|+0.0000|+0.0038|
|ecoli_imu|lr|+0.0035|+0.0325|
|ecoli_imu|rf|-0.0119|-0.0125|
|ecoli_imu|svm|+0.0009|-0.0038|
|glass7|lr|-0.1464|+0.0030|
|glass7|rf|-0.0369|+0.0190|
|glass7|svm|-0.0273|-0.0195|
|ionosphere|lr|+0.0162|-0.0020|
|ionosphere|rf|+0.0054|+0.0042|
|ionosphere|svm|-0.0205|-0.0182|
|spambase|lr|+0.0034|+0.0031|
|spambase|rf|-0.0022|-0.0038|
|spambase|svm|-0.0035|+0.0007|
|wdbc|lr|-0.0022|-0.0001|
|wdbc|rf|-0.0013|+0.0015|
|wdbc|svm|-0.0039|+0.0007|

## Failed final fits

|Method|Failures|
|---|---:|
|adasyn|39|
|wrnd_author|174|

Failures propagate from resampling to all three classifiers for an affected fold. No artificial zero score or alternative resampler was substituted. WRND is an author-code adapter and requires further reconciliation with the full paper before being represented as a fully verified published-method baseline.

## Threshold control

On independent Gaussian data with separation 2 and IR approximately 10:1, selecting a threshold on a separate trusted validation set makes proposed, SMOTE and untreated LR G-mean very similar. All six individual proposed-minus-SMOTE/none 95% paired bootstrap intervals include zero. This does not establish equivalence, but does not support a positive advantage in this control.

## Remaining Section IV work

- joint intervention search
- prior framework baseline
- synthetic multi-classifier evaluation
- corrupted-validation sensitivity
- full component ablations
- formal cross-dataset inference
- runtime scaling
- subgroup and label-noise interactions
