> **This is the no-patient-data variant of the deposit.** The PREDIX
> individual-level input matrix `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt`
> is not in this repository, and the passages below that describe it, verify
> its SHA-256 or count its columns describe the full deposit. See
> `data/README.md` for what runs without it and `docs/NO_PATIENT_DATA.md` for
> what individual-level information the deposited artefacts still carry.

# PREDIX HER2 — reproducibility package (Nature Cancer revision)

This package lets a reviewer verify, on a laptop, every number and figure in the
revised manuscript of the PREDIX HER2 multimodal pCR-prediction study — from the
deposited model artefacts down to the exact cell values of the deposited tables.
The centrepiece is `PREDIX_HER2_reproducibility.ipynb`, an **executed** notebook
whose code cells recompute each deposited quantity and assert equality
(tolerance 1e-9 on point estimates and CI bounds, with one stated exception —
see "Comparison tolerances" below).

## Folder map

| path | contents |
|---|---|
| `PREDIX_HER2_reproducibility.ipynb` | the executed verification notebook (outputs embedded) |
| `run_notebook.py` | re-executes the notebook headless (`python run_notebook.py`) |
| `code/` | the eleven analysis scripts (`apply_locked_external_validation.py`, `apply_locked_signatures.py`, `compute_monte_carlo_error.py`, `cv_estimands.py`, `external_validation.py`, `feature_deduplication.py`, `generate_report.py`, `make_fig_feature_ranking.py`, `multimodal_pcr_pipeline.py`, `preflight.py`, `revision_analyses.py`), `tests/test_statistics.py`, `requirements.txt`, and the production run scripts (`production_run_ubuntu.sh`, `production_run.ps1`) |
| `data/` | the canonical PREDIX input file (SHA-256 verified against provenance) and the two external cohort files (I-SPY2, NCT02326974) |
| `results/` | production model artefacts: discovery + consensus PKLs per scenario, CV splits, `run_provenance.json`, `methods_cv_statement.txt`, per-classifier signature CSVs |
| `report/` | the **deposited** figures and tables the notebook reproduces (`report/tables/revision/` is the citable set) |
| `environment/` | `requirements.txt`, `pip_freeze_windows.txt` (notebook machine), `production_environment.json` (model-run machine, transcribed from `results/run_provenance.json`) |
| `MANIFEST_SHA256.txt` | SHA-256 of every deposited file except the notebook and the manifest itself. Written when the deposit is assembled, and rewritten by notebook Section 13 under the same rule — so re-executing the notebook reproduces it byte for byte, and a changed manifest means a changed package |
| `_regenerated/` | **scratch, not part of the deposit.** Notebook Section 11 and Section 12 create it on demand and write a full second copy of the report into it for the cell-by-cell comparison. It is excluded from `MANIFEST_SHA256.txt` and from the distributed archive, so a clean package does not contain it. Nothing needs it to pre-exist: the analysis scripts create it, and an empty one left behind by an earlier run is harmless. Delete it freely. |

## Two reproduction levels

**Level A — the post-modelling computation, exactly reproducible from the
deposited fold-level outputs (this notebook; ≈ 15 min).**
All randomness downstream of the model PKLs (the cluster bootstraps) is seeded
(base 20240517, deterministic CRC32 offset per quantity), so every table and
figure reproduces exactly from `results/`. Run `python run_notebook.py` or open
the notebook and *Run All*. Indicative runtimes on the notebook machine
(Windows 11, Python 3.14, consumer laptop) are **15–17 minutes end to end**:
test suite ≈ 13 s; Section 5 bootstrap CIs ≈ 2 min; Section 7 paired comparisons
≈ 1.7 min; Section 8 calibration ≈ 13 s; Section 11 locked external-validation
re-runs ≈ 1.1 min; Section 12 full
regeneration ≈ 4.3 min (`generate_report.py`) plus ≈ 5 min
(`revision_analyses.py`), followed by the cell-by-cell comparison of ≈ 179,000
workbook cells in about 6 s. These are wall-clock timings and will vary with the
machine; **the executed notebook shipped here prints its own exact total in the
last cell**, which is the figure to quote.

**Level B — full pipeline re-run (many CPU-hours).**
`code/production_run_ubuntu.sh` drives the whole thing: step −1 `preflight.py`
gates the input file, step 0 runs the test suite, step 1 re-trains everything
(5-fold × 200 repeats global, 5-fold × 100 per arm; seed 42;
`--training_data cc_only --dedup_per_scenario`; six classifier families), and
steps 2–3 regenerate the report. The external
validation is produced separately and is described in
`external_validation/README.md`. CV partitions
are fully determined by seed 42, but
classifier-internal randomness of the tree models is deliberately **unseeded**
(seeding it would correlate the CV repeats and understate variance — see
`reproducibility_note` in `results/run_provenance.json`). A Level B re-run
therefore reproduces linear-model numbers exactly and tree-model numbers
statistically; the deposited PKLs in `results/` are the archival record, and
Level A reproduces every published quantity from them exactly.

**What has been demonstrated, stated precisely.** The **post-modelling
computation** — consensus aggregation, metric computation, the seeded
patient-level cluster bootstrap, signature ranking and fusion summarisation —
reproduces bit-identically from the deposited fold-level outputs, verified
across Linux → Windows and a much newer software stack (≈ 179,000 workbook
cells, zero mismatches; both external validations at |diff| = 0). This is
**not** a
demonstration that re-fitting the models reproduces bit-identically. The
pipeline deliberately leaves classifier-internal randomness unseeded
(`random_state=None`), and its own `reproducibility_note` records that the last
digit of a per-fold metric may vary between runs. Solver defaults, tie-breaking
and RNG consumption also change between scikit-learn majors, and this analysis
reports selection frequencies over 1,000 folds, which are sensitive to exactly
that. Say "exactly reproducible from the deposited fold-level outputs"; never
"bit-reproducible end to end". Treat a re-run on a newer stack as a re-analysis,
and report it as one.

## Software

`pip install -r environment/requirements.txt`, which **pins the exact versions
that produced the published results** (numpy 1.26.4, pandas 2.3.3,
scikit-learn 1.7.2, scipy 1.10.0, shap 0.49.1, joblib 1.5.3 on Python 3.10.12);
matplotlib, openpyxl and threadpoolctl stay as lower bounds because they affect
rendering and file I/O, not any computed value. Add `pymupdf` for inline figure
rendering and `nbformat nbclient ipykernel` for headless execution. The
production model run
used Python 3.10.12 / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.7.2 /
scipy 1.10.0 / shap 0.49.1 / joblib 1.5.3 on Ubuntu 6.8.0 — those versions are
transcribed verbatim from the `environment` block of
`results/run_provenance.json` into `environment/production_environment.json`,
so the two can never disagree. The notebook was executed on
Windows 11 / Python 3.14.7 / numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0 /
scipy 1.18.0 (`environment/pip_freeze_windows.txt`) — the post-modelling results
are identical across the two environments, which is the cross-platform claim
this package supports and the only one it supports (see "What has been
demonstrated" above).

> Do not relax these to `numpy>=2.0` or `scipy>=1.14`. Those bounds
> **exclude** the versions production used (numpy 1.26.4, scipy 1.10.0), so
> anyone following them installs an environment the results were not produced
> in, from a file that says it describes the published run.
> `results/run_provenance.json` records the real versions independently, and the
> notebook checks them against it.

## The cohort and the feature panel

The input file is **197 patients × 112 columns** (`patientID`, `pCR`, and 110
curated features: Clin 5, RNA 42, DNA 41, Prot 19, WSI 3).

- **Evaluation cohort: n = 110, 46 pCR events (DHP 59/24, T-DM1 51/22).** These
  are the patients complete across RNA, DNA, proteomics and WSI. Clinical
  covariates are recorded for everyone and imputed in-fold, so `Clin_` does not
  enter the completeness rule (`get_complete_case()`). Every model — unimodal or
  fused — is scored on exactly these 110, which is what makes every comparison
  paired.
- **Training and scoring use the same 110 patients.** Every model, unimodal
  or fused, is fitted and evaluated on the complete-case cohort. The 87
  patients who carry at least one assay but are not complete across all four
  molecular modalities do not enter the analysis.
- **Feature panel: 110 metrics.** The outcome-blind deduplication list
  `TIER1_REMOVE` names **21** features, of which **18** are present in this
  delivery and are therefore actually removed, leaving **92 candidate features**.
  A second, per-scenario deduplication (`--dedup_per_scenario`,
  `code/feature_deduplication.py`) then removes within-modality near-duplicates
  on each scenario's own complete-case cohort — 11 in the pooled cohort, 12 in
  the DHP arm and 6 in the T-DM1 arm — so **81, 80 and 86 candidates** enter the
  in-fold univariate screen in the three scenarios; every removal is recorded in
  `results/<scenario>/<scenario>_deduplication_audit.csv`. The three that the
  list names but the
  file does not carry are `DNA_TMB_uniform`, `DNA_TMB_clone` and `DNA_pTMB`, so
  on this delivery the panel carries clonal oncogenic TMB only — never write
  "total TMB". Separately, `RNA_ADC_trafficking` was withdrawn upstream by the
  authors before round 2 and is absent from the file altogether; it is not on the
  `TIER1_REMOVE` list. `RNA_FCGR3B` is present in the file and removed by
  `TIER1_REMOVE`, not by the delivery.
- The locked consensus signature is aggregated over the outer folds won by the
  modal classifier (`SIGNATURE_SOURCE = "winner_folds"`), so the reported
  classifier and the reported signature are one model rather than two.

One trap worth stating once: `Clin_TUMSIZE` and `Clin_prolifvalu` encode missing
as the **string** `"Unknown"` rather than as `NaN`. Both encode to `NaN` and are
median-imputed per fold, so modelling is unaffected — but only 104 rows are
literally complete on all 110 columns, and a script that reaches for `dropna()`
across every feature column is not computing the pipeline's cohort. Notebook
Section 2 uses the RNA/DNA/Prot/WSI rule and checks the token count explicitly.

## What each notebook section verifies

| § | reproduces | checked against |
|---|---|---|
| 1 | environment, input-file SHA-256, provenance, canonical CV statement | `results/run_provenance.json`, `methods_cv_statement.txt` |
| 2 | cohort: 197×112 file; evaluation cohort n = 110 (46 events; DHP 59/24, T-DM1 51/22); per-modality training cohorts; 110-metric panel with `TIER1_REMOVE` 21 listed / 18 present → 92 candidates | data file, `code/multimodal_pcr_pipeline.py` |
| 3 | statistics test suite (vs scipy / sklearn / R references) | `code/tests/test_statistics.py` exit 0 |
| 4 | CV design from the artefacts: 5×200/100 repeats, every patient predicted once per repeat | `results/*.pkl` |
| 5 | headline AUROC/AUPRC/Brier + cluster-bootstrap 95% CIs, both sources | `revision_performance_CI.xlsx`, `PREDIX_HER2_results.xlsx`, fig01 |
| 6 | why the estimand: per-repeat mean vs ensemble-mean artefact vs per-fold mean | (didactic; artefact asserted) |
| 7 | paired fused-vs-unimodal Δ, CIs, bootstrap p, per-repeat DeLong, verdicts | `revision_model_comparisons.xlsx`, revfig07 |
| 8 | calibration slope, recalibration intercept, calibration-in-the-large, Brier, ECE + CIs | `revision_calibration.xlsx`, revfig01 |
| 9 | selection stability, fusion-weight stability, per-fold EPV (3 spot recomputations) | `revision_stability.xlsx`, `revision_epv_per_fold.xlsx`, revfig02/08/03 |
| 10 | consensus signatures (K, winner classifiers) | `PREDIX_HER2_results.xlsx` Signatures, fig02/fig05 |
| 11 | locked external validation, **arm-matched**: the deposited table for all three applicable cohorts, the internal comparator of each locked arm model, and the full script re-run | `external_validation.xlsx`, `revfig06_external_validation.pdf` |
| 12 | **entire deposited report regenerated** from the PKLs and compared cell-by-cell, strictly (one declared and printed exception: the `locked_from` CLI-path string) | every workbook under `report/tables/` |
| 13 | package manifest | `MANIFEST_SHA256.txt` |

### Comparison tolerances

Every recomputed value is compared with its deposited counterpart at **1e-9**,
the tolerance of `close_to()` in the notebook's setup cell, with one exception
that is stated in the notebook's own source and again here.

Section 11 checks the external-validation workbook's `internal_AUROC` against
`revision_performance_CI.xlsx` at **1e-6**, because `external_validation.xlsx`
stores that value rounded to six decimal places. At 1e-9 the check would be
testing the storage format rather than the agreement of the two figures, and
would fail on a deposit in which nothing was wrong. The observed `|diff|` is
printed beside the check, so the margin is visible rather than asserted.

Section 12's cell-by-cell comparison of the regenerated report is exact for
strings and 1e-9 for numbers, with one declared and printed exception, the
`locked_from` CLI-path string (see the caveats below).

## The estimand (statement of record)

AUROC/AUPRC/Brier = in each CV repeat every patient has exactly one out-of-fold
prediction; the metric is computed on that complete out-of-fold vector and
averaged over the repeats (200 global / 100 arm). 95% CI = patient-level CLUSTER
bootstrap: 2,000 stratified resamples of PATIENTS, a resampled patient carrying
all its repeat predictions. Predictions are never averaged across repeats or
models. Paired Δ: same patient resample applied to both models and all repeats
(primary); DeLong per repeat summarised (descriptive). Verdict "not
distinguishable" whenever the paired 95% CI includes 0. The "±" values in the
submitted manuscript are the SD of per-fold AUROC and are NOT a CI.

Multiplicity: the Benjamini–Hochberg correction on the paired comparisons is
applied across the family that is actually published — all **15** comparisons
within a `source` (3 scenarios × 5 comparators) — not per scenario. Correcting
inside each scenario would ship six independent families of five while
presenting 30 comparisons; the published family is the wider one, and widening
it only ever makes q larger. `revision_model_comparisons.xlsx` records the
family size in `BH_family_size` and keeps the narrower per-scenario values in
`q_within_call_BH_m5` for comparison.

Every bootstrap in the package draws its seed from one function,
`cv_estimands.shared_seed(tag) = 20240517 + crc32(tag) % 2**31`.
`generate_report.py`, `revision_analyses.py`, `external_validation.py` and the
notebook all call it, so the same quantity gets the same resample stream
wherever it is computed and re-running reproduces the CI endpoints exactly. The
offset uses the full crc32 range rather than a `% 10000` reduction, and it has
to: 10,000 slots for the ~126 tags these scripts generate produces three real
collisions — pairs of analyses silently sharing a resample stream. Each CI would
still be individually valid, but their Monte-Carlo errors would be correlated
and nothing would say so.

## External validation (statement of record)

The locked transcriptomic models are **arm-matched**: each is the model locked
in the PREDIX arm corresponding to the external cohort's regimen. Each is frozen
on PREDIX and applied once to its external cohort — nothing refitted, no feature
dropped — with a SHA-256 provenance guard confirming that the locking run and
the validation run saw the same input file. Every applicable cohort's result is
reported whichever way it falls, and none was chosen after seeing its result.

The table below is generated from
`report/tables/revision/external_validation.xlsx` at build time, so it cannot
disagree with the workbook it tells you to quote.

| cohort | PREDIX model | n / events | AUROC (95% CI) | one-sided P vs chance | calibration slope (95% CI) |
|---|---|---:|---|---:|---|
| I-SPY2 | DHP (arm-matched) | 44 / 26 | **0.795 (0.630–0.932)** | < 0.001 | 0.77 (0.28–1.74) |
| TransNEO | DHP (arm-matched) | 60 / 19 | **0.665 (0.525–0.798)** | 0.0125 | 0.38 (0.02–0.92) |
| NCT02326974 | T-DM1 (arm-matched) | 129 / 64 | **0.709 (0.618–0.797)** | < 0.001 | 0.64 (0.37–1.02) |

**All three transfer**: every row in the table is above chance on the one-sided test.

**The genomic model has no external result, and that is reported rather than
omitted.** three of its four signature features are not measured in the TransNEO metric file, so it cannot be scored in any cohort available:
a signature with features removed is a different model, and scoring it would
answer a different question. The per-feature account is in the
`Feature_provenance` sheet of the same workbook.

The arm-matched design is used throughout because it is the only one applicable
to every external cohort — a signature locked on the pooled cohort can require
features a given cohort does not measure, and the same reduced-signature
objection then applies. The rule is uniform applicability, fixed in advance.

Features were standardised within each cohort by z-score (`zscore`
harmonisation). That step is unsupervised but transductive, and the workbook
says so in its own header.

## The deposited report is the production report

Everything under `report/` and `results/` is byte-identical to
the production run that produced it. Nothing was regenerated or edited for this
deposit. Notebook Section 12 re-runs `generate_report.py` and
`revision_analyses.py` on the deposited artefacts with the deposited code and
compares every workbook cell by cell, **strictly, with no sheet exempt**, so the
agreement between this package's code and this package's tables is shown rather
than assumed. The `Methodology` sheet of
`supp_PREDIX_HER2_feature_pruning_report.xlsx` reads every threshold and the
`TIER1_REMOVE` list from `code/multimodal_pcr_pipeline.py` at generation time,
so it describes the pipeline that ran; and the univariate screen's five-feature
floor, which the response letter discloses, is recorded per fold in the
`floor_used` flag of the deposited fold records.

## Known caveats

- **Classifier randomness is unseeded by design**, so Level B reproduces the
  tree-model numbers statistically, not bit-for-bit; Level A (from the deposited
  PKLs) is exact.
- **Training and evaluation are both complete-case.** The data file carries 197
  patients, of whom 110 are complete across RNA, DNA,
  proteomics and WSI. Every model, unimodal or fused, is *fitted* as well as
  *scored* on those 110 — so every paired comparison is
  paired by construction, and the 87 patients who carry some assays but not all
  four molecular modalities do not enter the analysis at all. The one-cell check
  is `median_n_events_train` in `revision_epv_per_fold.xlsx`: it is **identical
  across modalities** within each scenario (37 pooled, 19 DHP, 18 T-DM1), which is the
  signature of complete-case training. Values that differed between modalities
  would mean the unimodal models had been fitted on different, larger cohorts;
  they do not differ.
- **The pooled models meet five events per variable; the arm-level models do
  not.** In the pooled cohort `pct_folds_epv_below_5` is 0.0 for every model
  and median realised EPV runs 7.4–12.3 (`revision_epv_per_fold.xlsx`).
  Within the arms a fold trains on 18–19 pCR events, the five-feature floor
  overrides the cap, and eight of the ten arm unimodal models fall below five
  in a large fraction of folds — from about half of the DHP genomic folds to
  every T-DM1 clinical fold (`pct_folds_epv_below_5`). The arm-level fusion
  layer is below five in about 42% of DHP folds and 50% of T-DM1 folds. This is
  *structural*
  rather than fixable: the second-stage combiner needs all five modality
  predictions for a patient, so it is complete-case by construction and trains
  on ~19 (DHP) or ~18 (T-DM1) events. Arm-level *integrated* results should be
  read accordingly; the pooled fusion layer is unaffected (0% of folds below 5).
- **The weakest model is arm-level imaging, and it is below chance.** The
  lowest deposited AUROC is T-DM1 imaging 0.428 (95% CI 0.397–0.459), an interval that
  EXCLUDES 0.5. Three imaging features cannot be fitted informatively in
  51 patients with 22 events; we report
  this rather than suppress it, and it is one reason the arm-level results are
  exploratory. It is the **only** point estimate below chance in the consensus table: every other consensus model is above it, the next lowest being DHP clinical 0.522.
- The `locked_from` column of `external_validation.xlsx` records the `--locked_*`
  path as typed on the production command line; re-runs from this package record
  the package-relative path instead. Printed in the Section 12 comparison, not
  skipped silently.
- **Row order is part of the record, and is fixed deterministically.**
  `selection_frequency()` sorts by `(selection_freq_eligible, selection_freq,
  feature)` with a stable sort, so the `Feature_selection_stability` sheet of
  `revision_stability.xlsx` comes out in the same order on every run. Without
  that, features tied at the same frequency would permute with the per-process
  string hash seed and Section 12's comparison would report a difference where
  no value had changed. `tests/test_statistics.py` checks the order, and every
  sheet is compared in its natural row order with no special handling.

## How to cite numbers

Quote numbers **only** from `report/tables/revision/` (and
`report/tables/PREDIX_HER2_results.xlsx` for the headline consensus table) — the
patient-level cluster-bootstrap set. Discovery-phase
diagnostics under `report/tables/supplementary/` are per-fold descriptive
values, and any "±" found there is a fold SD, not a confidence interval.
