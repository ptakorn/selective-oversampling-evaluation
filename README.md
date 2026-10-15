# Ambiguity-Guided Selective Oversampling

Reproducibility package for **Selective Oversampling under Label Noise: Separating Ranking, Decision-Threshold, and Intervention Effects**, by Takorn Prexawanprasut, Sukhothai Thammathirat Open University.

This is a research snapshot, not a claim of general superiority or verified label-noise identification. Intervention parameters are fixed. Results include adverse outcomes, baseline failures, and nonsignificant comparisons. Manuscript metadata/DOI are not assigned in this package.

## Quick start (Python 3.12)

Run commands from the repository root. Install Git separately if using the WRND setup.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_files.py
python verify_results.py
python verify_new_results.py
python smoke_test.py
python external_control/preflight.py
```

Verification uses the included results and saved validation/test scores; it does not download datasets or rerun the full experiment suite. A successful smoke test is not a complete replication. Fitting all experiments can take substantially longer and depends on hardware. `EXPERIMENT_JOBS` controls workers in the earlier packaged runners that support it. The frozen targeted and external follow-up runners use four workers. These worker choices are separate from the scientific configuration.

## Repository contents

| Directory | Experiment | Main result file |
|---|---|---|
| `pilot/` | Initial development, 30 replications, 3 conditions, 3 methods | `pilot_raw.csv` |
| `ablation/` | Original/synthetic weighting and ambiguity-only variants | `ablation_raw.csv`, `combined_raw.csv` |
| `expanded/` | 63 synthetic settings; 17 stress settings; 6 benchmarks, 3 classifiers; Gaussian threshold control | `synthetic_grid_raw.csv`, `stress_raw.csv`, `benchmark_raw.csv`, `threshold_control_raw.csv` |
| `neighborhood_ablation/` | Adaptive membership vs fixed counts (5,10,20) | `synthetic_raw.csv`, `benchmark_fixed_raw.csv` |
| `benchmark_threshold/` | Default vs inner-validation threshold, 3 methods, 3 classifiers | `raw.csv`, `audit_scores.zip` |
| `revision_audit/` | Ranking, count and seed audits | `report.md`, ranking/budget CSVs |
| `revision_audit/confirmation/` | Targeted Gaussian/moons/circles extension, 2700 fits | `raw.csv`, `paired_intervals.csv` |
| `evidence_supplement/` | 432 paired intervals and complete baseline reporting | `all_paired_intervals.csv`, baseline tables |
| `external_control/` | SMOTE-ENN, count and original-retention controls, 1620 new fits | `new_raw.csv`, `paired_intervals.csv`, `provenance.zip` |
| `reports/` | Latest combined supplementary tables S1--S23 | `supplement_body.tex`, `supplement_updated.pdf` |
| `docs/` | Provenance, table mapping and update instructions | Markdown documentation |

Raw files and numeric summaries are the evidence; older per-experiment prose reports may predate the supplementary studies. The latest root protocol and study READMEs explain their scope. `*_insert.tex` files are drafting aids and may duplicate methods between inserts; place protocol in Experimental Design and numerical outcomes in Results.

## Download benchmark datasets

```bash
python expanded/download_data.py
python verify_data.py
```

The six UCI tasks are WDBC, Banknote Authentication, Spambase, Ionosphere, Ecoli-imU versus rest, and Glass-7 versus rest. Dataset manifests document IDs, label mappings, excluded features and source checksums. Feature arrays and source dataset archives are not redistributed here. WDBC is loaded from scikit-learn; other datasets are fetched from UCI. `verify_data.py` checks normalized X/y array hashes against the arrays actually used in the experiments. A mismatch must be investigated before claiming exact replication. Downloading requires internet access.

## WRND dependency

```bash
python expanded/setup_wrnd.py
```

This fetches the authors' WRND repository at commit `f69a5c71fdff7dd3ff9b52c9d96801c3a931cc99` and extracts its generation routine locally for compatibility. Those third-party sources are **not included** in this distribution; see `THIRD_PARTY_NOTICES.md`. The adapter is not a fully verified reproduction of the published paper. Recorded WRND failures are retained, not assigned invented scores. No inference of published-method inferiority follows from adapter failures.

Other experiments do not need WRND downloaded. The default full benchmark run includes WRND, so fetch it first. For deliberate subsets, `BENCHMARK_METHODS` accepts comma-separated method names; label outputs as a subset and do not confuse them with the included full comparison.

## Reproduce experiment results

Commands below fit models and overwrite their respective result CSVs. Back up the snapshot or run in a separate checkout before recomputing. Some runners resume from local per-task checkpoints; use a fresh checkout when changing configuration.

```bash
python pilot/run_pilot.py
python ablation/run_ablation.py
python expanded/run_synthetic_grid.py
python expanded/run_stress.py
python expanded/run_benchmarks.py
python expanded/run_threshold_control.py
python neighborhood_ablation/run.py
python benchmark_threshold/run.py
python revision_audit/confirmation/run.py
python external_control/run.py
```

The supplementary benchmark threshold runner reuses classifier parameters selected in `expanded/benchmark_raw.csv`. It refits inner models at those parameters to choose the threshold; it does not retune classifier parameters using threshold optimization.

To regenerate the main statistical summaries from raw results:

```bash
python expanded/analyze_results.py
python neighborhood_ablation/analyze.py
python benchmark_threshold/analyze.py
python revision_audit/analyze.py
python revision_audit/confirmation/analyze.py
python evidence_supplement/build.py
python external_control/analyze.py
python external_control/make_deliverables.py
python build_supplement.py
```

Pilot/weighting summaries are generated by their experiment runners. See `docs/TABLE_MAPPING.md` for source files. Per-study report-generation scripts that packaged workspace files have been omitted; this repository itself is the deliverable.

## Protocol and interpretation

- Synthetic settings have 30 paired replications. Benchmark folds are grouped by identical selected feature vectors: 5 outer folds x 3 repetitions, with 3 inner folds. Fitting corruption is reused across methods. Validation/test observations never enter training neighborhoods or resampling.
- Benchmark classifier parameters are selected by default-rule inner G-mean. Intervention parameters are not tuned. No default primary experiment uses automatic class weighting.
- Adaptive ablation changes membership selection to counts 5/10/20, retaining the reference radius for expansion adequacy. It does not remove all adaptive-radius dependencies. Adaptive reruns match all 2400 prior synthetic G-mean values exactly.
- Benchmark threshold selection uses a common threshold maximizing the unweighted mean of three inner-fold G-means. Scores are not pooled into a single confusion matrix. Thresholds transfer from inner models to the final refit without calibration; supplied validation labels are not verified ground truth.
- Statistical unit on benchmarks is the dataset mean over 15 folds, not an individual fold. Holm families are separate: 42 original baseline comparisons; 6 adaptive/fixed comparisons; 12 selected-threshold baseline comparisons. Individual synthetic bootstrap intervals are not simultaneous intervals. Nonsignificance is not equivalence.
- Original benchmark attempts: 4320; successful 4107; failed 213 (ADASYN 39, WRND adapter 174). Supplementary fixed-neighborhood benchmark evaluations: 540 successful. Threshold control: 1620 final models, 3240 decision-rule evaluations, all completed.

## Added follow-up evidence

The targeted extension contains 2700 model fits / 5400 decision-rule evaluations across nine geometry--corruption settings, two classifiers and 30 replications. Its Complete/SMOTE matched control holds realized counts, original weights and the synthetic-weight multiset fixed, while changing geometry and weight assignment together.

The external SMOTE-ENN study adds 1620 fits / 3240 evaluations and reuses 2160 earlier no-intervention / SMOTE evaluations. It is an explicitly configured imbalanced-learn hybrid cleaning method, not a verified WRND reproduction. Count-only and count-plus-original-retention controls are distinct. All external fits completed without fitting warnings. This does not establish general superiority or generalization across modern noise-aware methods.

Both follow-ups were designed after earlier results. Settings are not independent benchmark datasets. Bootstrap intervals remain exploratory individual intervals. No new model fits were performed merely to package this version.

`external_control/config.json` and `implementation_manifest.json` record the original experiment configuration and local source hashes; those paths/hashes are historical provenance, not assertions that every third-party file is bundled. `verify_new_results.py` independently checks included raw data, interval calculations and endpoint/retention provenance. It does not verify science by itself.

## Updating the existing GitHub repository

See [docs/UPLOAD.md](docs/UPLOAD.md). This folder is an update for the existing repository, not a request to create a new project. It contains no Git credentials or Git history and has not been pushed. Version `2026.10.15` labels this prepared snapshot; it is not a publication date or an already-created public release. Preserve the existing `.git` directory when copying the files into a local checkout, review changes, then commit and push yourself.

After upload, record the actual public commit in the manuscript. Commit `5c1850b` identifies the earlier snapshot and must not be represented as containing this update.

## AI assistance and responsibility

OpenAI ChatGPT and Codex assisted manuscript drafting/revision, methodological development, implementation, experiment execution and analysis. AI assistance is not limited to language polishing. The author is responsible for independently checking the final code, data provenance, reported results and interpretations before submission. Automated consistency checks do not replace scientific review.

## Licensing and citation

The original software in this repository is provided under the MIT license; see `LICENSE`. This license does not grant rights to external WRND code or datasets. Attribution and source instructions appear in `THIRD_PARTY_NOTICES.md`. Cite the article when published; until then identify the author, repository URL and exact release/commit used. Do not invent a publication DOI. Create a GitHub release for the manuscript snapshot after upload; no public release has been created by this package.
