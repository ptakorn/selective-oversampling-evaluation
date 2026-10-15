# Third-party notices

## WRND

Source: https://github.com/dream-lm/WRND_framework
Pinned commit: f69a5c71fdff7dd3ff9b52c9d96801c3a931cc99
Associated paper: Li et al., Expert Systems with Applications, DOI 10.1016/j.eswa.2023.122593.

No LICENSE file was found in the local pinned source snapshot. Do not infer redistribution permission from public accessibility. WRND source, its extracted generation routine, figures and datasets are excluded. `expanded/setup_wrnd.py` obtains source locally from its author repository for reproducibility; the external material retains its authors' rights and is excluded from this repository's MIT grant. Check upstream terms/permissions for your intended use. The included adapter documents compatibility changes; its outputs do not establish faithful reproduction of all published results.

## Datasets

UCI source IDs: 17, 267, 94, 52, 39, 42. Original feature arrays and source ZIPs are excluded. Refer to expanded/data/manifest.json and the individual UCI pages for dataset attribution and terms. WDBC is obtained through scikit-learn. Included result CSVs and score-audit files are experiment outputs, not original feature datasets.

## Software dependencies

NumPy, pandas, SciPy, scikit-learn, imbalanced-learn and joblib are installed separately under their respective licenses. They are not vendored. See requirements.txt for the recorded versions.

## Added external hybrid method

SMOTE-ENN uses the separately installed imbalanced-learn 0.14.2 implementation. Original analysis/tracing wrappers are included; imbalanced-learn source is not vendored. The configured version uses SMOTE k=5/ratio=1 and ENN n_neighbors=3, kind_sel=all, sampling_strategy=all. It is not a verified WRND replacement or a claim of exact reproduction of all experiments in Batista et al. (2004). Matplotlib is also installed separately for figures.
