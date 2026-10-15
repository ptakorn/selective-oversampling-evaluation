# Evidence additions for items 4–5

These are analyses and supplementary presentation of existing fitted models. New model fits: zero. Original raw results are unchanged.

## Use in manuscript

1. Copy paired_gmean_intervals.png and the contents of main_text_insert.tex into the main TeX project. Place the insertion after the targeted-extension results (current Tables 13–14). It adds one full-width figure. The graphic covers all nine settings and both comparators for each classifier; no favorable subset was selected.
2. Upload supplement.pdf as supplementary material. Tables S1–S2 give the 36 requested numerical intervals, S3–S8 all eight baseline summaries, S9–S14 differences on common complete datasets, and S15 completion/failure counts.
3. Keep supplementary tables outside the main 19-page manuscript. supplement.tex and supplement_body.tex are editable sources. The standalone supplement compiled to seven pages without overfull-box warnings.

## Statistical interpretation

- Complete minus comparator: positive favors Complete. Each targeted setting uses 30 paired replications and 5000 percentile-bootstrap resamples.
- Intervals exactly reproduce the previous calculations. They are exploratory individual 95% intervals, NOT simultaneous intervals and NOT multiplicity-adjusted significance tests. Counts of intervals above/below zero do not establish a population-wide win rate.
- All_paired_intervals.csv includes default and selected rules, G-mean/ROC-AUC/AP, and four comparators (432 intervals). selected_gmean_intervals.csv contains the 36 requested main comparisons.
- Benchmark means use dataset-level averages over 15 outer folds, not folds as independent datasets. D varies across methods due to failures. Summary means with different D must not be directly ranked. benchmark_common_dataset_pairs.csv and Tables S9–S14 restrict both methods to identical complete datasets.
- WRND is the previously documented compatibility adapter; its failures are not proof of inferiority of the published method. No failed observation is assigned a fabricated score.

## Results

For selected-threshold G-mean against count/weight-matched SMOTE:
LR: 1/9 intervals entirely positive, 2/9 entirely negative, 6/9 include zero.
RBF-SVM: 1/9 entirely positive, 6/9 entirely negative, 2/9 include zero.
The results do not support a general Complete advantage. They identify favorable and adverse conditions and uncertainty. The inference remains limited to the recorded scenarios/configurations.

## Reproduce

Original input files are included under inputs/. Run python build.py from this folder. Dependencies are numpy, pandas and matplotlib in addition to the recorded experiment environment. The bootstrap seed and resample count are unchanged. PDF source compiles with pdflatex supplement.tex from this folder. validation.json records source checksums and verification counts.
