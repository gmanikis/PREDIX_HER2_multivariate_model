# `data/` — what is here, and what is not

## The file that is not here

**`clin_multiomics_curated_metrics_PREDIX_HER2_new.txt`** is **not deposited in this
variant of the repository.**

| | |
|---|---|
| what it is | the PREDIX HER2 trial's curated analysis matrix — one row per trial patient |
| size | 197 patients × 112 columns |
| columns | `patientID`, `pCR`, and 110 curated metrics |
| metrics by modality | clinical 5, transcriptomic 42, genomic 41, proteomic 19, whole-slide image 3 |
| format | tab-separated text, UTF-8, 272,100 bytes |
| SHA-256 | `64dd2f3ff1c99170c70a27685c7d9d5633c5ae2edb23b45dbabc1b88a575cef0` |
| where it belongs | `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt` |

`patientID` is the trial's own patient number. `pCR` is that patient's
pathological complete response (0/1). The remaining 110
columns are the per-patient clinical, transcriptomic, genomic, proteomic and
whole-slide-image measurements described in
`docs/CANDIDATE_FEATURE_CURATION.md` and listed in
`supplementary/S-ML8_candidate_panel.xlsx`. It is, in other words, the trial
cohort itself.

## Why it is not here

Releasing individual-level data from the trial requires an ethics and consent
decision that has not been taken. This variant of the deposit exists so that
the analysis code and the deposited cross-validation artefacts can be published
and reviewed while that decision is pending. The full deposit, identical to
this one except that it contains the file, is what should be published once —
and only if — the release is approved.

**This does not make the repository free of individual-level information.**
`docs/NO_PATIENT_DATA.md` sets out exactly what the deposited `results/`
artefacts still carry, per patient, and how much of it. Read it before
treating this variant as anonymous.

## What still works without the file

Everything downstream of the deposited model artefacts, which is almost
everything:

| | |
|---|---|
| `report/` | deposited unchanged: all 31 figures and 13 workbooks, every number quoted in the article |
| `code/generate_report.py` | reads only `results/`. Regenerates every figure and table |
| `code/revision_analyses.py` | reads only `results/`. `--data_path` is optional and feeds only the exploratory survival section, which is inert here (the matrix carries no follow-up columns) |
| `code/compute_monte_carlo_error.py` | recovers the per-repeat AUROCs from the deposited fold predictions |
| `code/cv_estimands.py`, `code/tests/test_statistics.py` | self-contained; the test suite runs unchanged |
| `PREDIX_HER2_reproducibility.ipynb` | runs end to end. Four cells stand down and say so — see below |
| `MANIFEST_SHA256.txt` | verifies this tree as deposited |

## What does not work without the file, and what it says instead

Each of these stops with a message naming the file, its SHA-256 and this
document. None of them fails with a bare `FileNotFoundError`, and none of them
produces a partial or silently different result.

| | |
|---|---|
| `code/preflight.py` | stops immediately. It is the gate that checks the delivery before a production run; with no delivery there is nothing to gate |
| `code/multimodal_pcr_pipeline.py` | stops as soon as its arguments are parsed. This is "Level B" in the README — retraining the models from scratch — and it is the only thing in the deposit that needs the matrix in order to produce results rather than to check them |
| `code/external_validation.py` | stops at once. `--predix` is how it rebuilds the locked training frame; the **deposited** external-validation workbooks and figures are unaffected and are still read and checked by the notebook |
| `code/production_run_ubuntu.sh`, `code/production_run.ps1` | drive the two scripts above, so they stop where those stop |
| `supplementary/build_S-ML8_candidate_panel.py` | stops without writing a partial table. It recomputes the candidate panel and its correlations from the matrix; the **deposited** `supplementary/S-ML8_candidate_panel.xlsx` is unaffected |
| notebook Section 1 | keeps the whole provenance dump; only the two checks that hash the input file stand down. The expected SHA-256 is in the table above |
| notebook Section 2 | skipped in full: the 197×112 shape, the modality panel count, and the derivation of the n = 110 complete-case evaluation cohort |
| notebook Section 11 | the cell that **re-runs** the two locked external validations is skipped. The cell before it, which checks the deposited external-validation workbooks, still runs |
| notebook Section 12 | complete, except that the two artefacts Section 11 could not regenerate are not in the cell-by-cell comparison; the cell prints which |

One further script cannot run here, but not because of this file:
`docs/build_ED_Fig11a_schematic.py` resolves `run_provenance.json`, the input
matrix and the pipeline source against the authors' working-tree layout rather
than this repository's. It stops with `missing input:` and a path. The figure
it draws is deposited as `docs/ED_Fig11a_CV_schematic.pdf`.

## The two files that ARE here

`RNA_curated_metrics_ISPY2.txt` and `RNA_curated_metrics_NCT02326974.txt` are
the external validation cohorts. They are curated metrics computed from the
**public** GEO series **GSE194040** (I-SPY2) and **GSE243375**
(NCT02326974); their identifier column is `sampleID` and carries those series'
own sample names. They contain no PREDIX patient identifier and no PREDIX
patient — the build script re-checks both facts and refuses to assemble this
variant if either fails. Attribution to the original studies is owed and is
noted in `LICENSE`.

## Requesting the withheld file

[AUTHOR NOTE: name here the person or office that receives requests for the individual-level data, the address to write to, and the procedure and conditions that apply (data-access committee, data-transfer agreement, any embargo). Do not publish this repository with this note still in it.]

If you obtain the file, put it at `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt`, check its SHA-256 against the
value above, and re-run anything in this repository unchanged: every guard
described here is an existence test that falls through once the file is
present.

---
Generated 2026-09-06 by `build_github_repo.py --no-patient-data`.
