# Added evidence and manuscript mapping

| Manuscript evidence | Sources |
|---|---|
| Original ranking/count/seed audit | revision_audit/analyze.py and resulting CSV/JSON |
| Targeted geometry and count/weight controls | revision_audit/confirmation/run.py; raw.csv |
| Targeted paired intervals, Supplementary S1--S2 | evidence_supplement/build.py; selected_gmean_intervals.csv |
| All benchmark means, Supplementary S3--S8 | evidence_supplement/benchmark_all_methods_summary.csv |
| Common complete-dataset comparisons, S9--S14 | evidence_supplement/benchmark_common_dataset_pairs.csv |
| Benchmark completion, S15 | evidence_supplement/benchmark_completion.csv |
| External-method intervals, S16--S19 | external_control/paired_intervals.csv; external_tables.tex Tables E1--E4 |
| External-method metric means, S20--S23 | external_control/summary.csv; external_tables.tex Tables E5--E8 |
| External original/synthetic intervention quality | external_control/intervention_audit.csv; provenance.zip |

`reports/` contains the combined S1--S23 supplement. The experiment-local external document intentionally keeps E1--E8 for standalone use. They are the same numerical tables with different document numbering. Main-manuscript table numbers can shift during editing; use labels and evidence descriptions rather than older table numbers in TABLE_MAPPING.md.

Figures are available in evidence_supplement/paired_gmean_intervals.png and external_control/external_control_intervals.png. `*_insert.tex` files are drafting aids, not a second independent evaluation.
