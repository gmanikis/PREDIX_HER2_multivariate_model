# External validation of the locked transcriptomic models

Every model here was **frozen first** — signature, classifier and
hyperparameters taken from the locked consensus — then refit **once** on the
corresponding PREDIX cohort with no grid search and no cross-validation,
standardised within each cohort by z-score, and applied **once**. Nothing was
refitted on external data, and no feature was dropped to make a model fit: a
signature with features removed is a different model.

## Results

| Cohort | Modality | PREDIX model | Internal | External | P vs chance |
|---|---|---|---|---:|---:|
| I-SPY2 (GSE194040) | transcriptomic | DHP, linear SVM, K = 5, fit on 59 patients / 24 events | 0.808 | **0.795 (0.630–0.932)** | < 0.001 |
| TransNEO | transcriptomic | DHP, linear SVM, K = 5, fit on 59 / 24 | 0.808 | **0.665 (0.525–0.798)** | 0.012 |
| NCT02326974 (GSE243375) | transcriptomic | T-DM1, linear SVM, K = 4, fit on 51 / 22 | 0.788 | **0.709 (0.618–0.797)** | < 0.001 |
| TransNEO | **genomic** | DHP, elastic-net logistic regression, K = 4 | — | **not validatable** | — |

Four evaluations were attempted and three are applicable. All three
transcriptomic validations clear chance. The internal-to-external drop is 0.013,
0.143 and 0.079.

Each cohort is matched to the PREDIX arm whose regimen it resembles: the DHP
model to the two chemotherapy-plus-HER2-targeted cohorts, the T-DM1 model to the
T-DM1 cohort.

**How to read the P column.** It is a one-sided bootstrap tail probability that
the external AUROC is at or below 0.5, computed as (draws at or below chance + 1)
/ (draws + 1) over 2000 stratified draws. Its smallest attainable value is
1/2001, about 0.0005, so a row that reaches the floor means no draw fell at or
below chance — a limit of the resampling, not a measured probability. Such rows
are reported as `< 0.001` rather than as a point estimate, in this table, in both
figures and in the workbook.

**Calibration.** Probabilities are Platt-scaled using the locked model's
cross-validated predictions in the PREDIX fit cohort, so that the external
probabilities describe the same pipeline the internal ones do. Which route each
row actually used is recorded per row in the `probability_calibration` column of
the workbook and the CSV; it affects the Brier score and the calibration slope
and intercept only. **AUROC and AUPRC are the same either way**, Platt scaling
being monotone. The external calibration slopes are 0.77 (0.28–1.74), 0.38
(0.02–0.92) and 0.64 (0.37–1.02): all below one, and in TransNEO the interval
excludes one. Discrimination transfers; the probabilities would need
recalibration before use outside the trial.

## Why there is no genomic validation

The genomic signature is `DNA_ERBB2_CNA`, `DNA_HRD`, `DNA_NCOR1_CNA`,
`DNA_COSMIC.Signature.2`. `DNA_curated_metrics_TransNEO.txt` carries five genomic
metrics: COSMIC signatures 6 and 13, *TP53* coding mutation, *ERBB2* copy number
and LOH/deletion burden. **Three of the four signature features are not measured
there**, so the model cannot be scored.

That is a limitation of the metric file available to us, not of the cohort:
re-extracting the fuller genomic panel from TransNEO would make a genomic
validation possible. The row is kept in
`locked_signatures_external_validation.csv` with `applicable = False` and its
missing features named, it is stated on both figures, and it is carried in the
`Feature_provenance` sheet of the workbook — so the absence is legible rather
than silent.

## Feature portability

`feature_portability.csv` gives, per signature feature, the directional
univariate AUROC in PREDIX and in each external cohort, and the Spearman
correlation with `RNA_mRNA-ESR1` as a common anchor. The PREDIX side is measured
on the **complete-case cohort of the matching arm** — 59 DHP, 51 T-DM1, the
population the locked models were fitted on — and that population is carried in
the file itself, in the `PREDIX_population` and `n_PREDIX` columns.

**Two transcriptomic features flip direction, both in TransNEO**: `RNA_Th2 cells`
and `RNA_Fatty_acid_metabolism`. All five DHP-signature features hold their
direction in I-SPY2, and all four T-DM1-signature features hold theirs in
NCT02326974 — 0 of 5 flips in I-SPY2, 2 of 5 in TransNEO, 0 of 4 in
NCT02326974. The two HER2DX scores hold their direction in every cohort.

That the flips sit on an immune-deconvolution term and a metabolic programme
rather than on a HER2DX score is consistent with such estimates being platform-
and pipeline-dependent by construction. Direction is the more portable property
here than correlation structure: the largest shifts in correlation with ESR1 are
in the T-DM1 signature (up to Δρ = 0.45, on the exosome programme), where
nevertheless no feature flips.

The genomic signature cannot be assessed on the same terms: three of its four
features are absent, and `DNA_ERBB2_CNA`, the one feature present, flips.

**An unresolved scale problem.** `DNA_ERBB2_CNA` spans −0.315 to 3.657 in PREDIX
and 1 to 79 in TransNEO, consistent with a log-ratio against an absolute copy
number. Z-scoring corrects location and scale but not a different estimator.
This should be settled before any genomic transfer is attempted.

## The files

Results, beside this README:

| File | What it is |
|---|---|
| `locked_signatures_external_validation.csv` | one row per evaluation attempted, applicable or not; the source every other artefact here reads |
| `external_validation_all_cohorts.xlsx` | the same rows as a table, with the design notes |
| `fig_external_validation_all_cohorts.pdf` / `.png` | the figure |
| `feature_portability.csv` | the portability diagnostic |

Code, in the `code/` subdirectory:

| File | What it is |
|---|---|
| `apply_locked_external_validation.py` | **the driver.** Reads each locked signature, classifier and hyper-parameters from `results/`, applies the model once to every external cohort, and writes both `locked_signatures_external_validation.csv` and `report/tables/revision/external_validation.xlsx`. A copy also sits in the top-level `code/`, where the production runner calls it. |
| `apply_locked_signatures.py` | the two shared primitives it imports: the external-cohort encoding, and the bootstrap test against chance |
| `make_fig_external_validation_all.py` | draws the figure and writes the all-cohorts workbook, from the CSV alone |
| `feature_portability.py` | computes the diagnostic; needs the locked `results/{dhp,tdm1}` and the curated matrices as well |

## What is not here

The TransNEO metric files are **third-party and are not redistributed**; the
I-SPY2 and NCT02326974 derived metrics are included because they come from
public GEO series (GSE194040, GSE243375) and carry those studies' terms.

That is the one thing that stops `apply_locked_external_validation.py` running
to completion from this repository alone: it will score I-SPY2 and NCT02326974
and stop at TransNEO with the missing input named. Its complete output is
deposited, and `make_fig_external_validation_all.py` re-derives the figure and
the all-cohorts workbook from that output alone.

`FEATURES_NEEDED_FOR_VALIDATION.md`, which lists feature by feature what each
locked model requires of an external cohort, is likewise part of the delivery
rather than of this repository; the same information for the one model that
could not be scored is in *Why there is no genomic validation* above and in the
`Feature_provenance` sheet of the workbook.

## Regenerating

`make_fig_external_validation_all.py` re-derives the figure and the all-cohorts
workbook from `locked_signatures_external_validation.csv` alone; it recomputes no
reported number. It resolves every path relative to its own file — it reads the
CSV beside itself and writes the figure and the workbook back to the same place
— so run it with the script and the CSV in one directory:

```bash
python make_fig_external_validation_all.py
```

`feature_portability.py` is the record of how `feature_portability.csv` was
computed. It needs the locked `results/{dhp,tdm1}` model artefacts and the
curated PREDIX and external matrices, the undistributable TransNEO files among
them, so it documents the diagnostic rather than re-running from this repository
alone. Nothing in the results above depends on it.
