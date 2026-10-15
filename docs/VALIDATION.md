# Packaging validation — 15 October 2026

Validated in Python 3.12 with the pinned dependencies. No full experimental refit was performed while packaging.

- 32 Python source files compile.
- Existing verify_results.py passes: counts/failures; 1620 reproduced benchmark default rules; 2400 adaptive reruns; 3240 test G-means and validation objectives reconstructed from saved scores.
- verify_new_results.py passes: 2700 targeted models and 1620 external model result groups; 2160 reused evaluations; 864 paired bootstrap intervals recomputed; 270 endpoint/retention provenance cases checked against regenerated fitting data; canonical raw-input copies match.
- Existing adaptive/fixed/SMOTE smoke checks pass.
- Nine external fitting-only preflight cases reproduce uninstrumented imbalanced-learn SMOTE-ENN exactly.
- New frozen experimental runners remain byte-identical to the completed experiment sources. Runtime output directories are retained with tracked .gitkeep placeholders.
- Original benchmark feature arrays and WRND sources are excluded. Generated endpoint-provenance arrays are archived in external_control/provenance.zip.
- Combined supplementary source contains 23 tables.

These checks establish internal consistency and packaging behavior, not independent replication of every model fit, generalization, novelty or journal acceptance. The previous dataset-hash verification is retained historically; datasets were not downloaded again in this packaging pass.

The four added audit/analysis entry points were executed successfully from this staged repository; the 23-table combined supplement was rebuilt and rendered (15 pages).
