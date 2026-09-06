# PREDIX HER2 — multimodal prediction of pathological complete response

Analysis code, deposited cross-validation artefacts, and an executed
verification notebook for the multimodal pCR-prediction analysis of the
PREDIX HER2 randomised trial (clinical, transcriptomic, genomic, proteomic and
whole-slide-image data).

> ### This is the no-patient-data variant of the deposit
>
> The PREDIX individual-level input matrix
> **`data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt`** —
> 197 patients x 112 columns (patientID, pCR and 110 curated metrics), Clin 5 / RNA 42 / DNA 41 / Prot 19 / WSI 3 —
> **is not in this repository.** It is withheld pending the ethics and consent
> decision on releasing individual-level trial data.
>
> * **[`data/README.md`](data/README.md)** — what the file contained, what
>   still works without it, what does not, and how to request it.
> * **[`docs/NO_PATIENT_DATA.md`](docs/NO_PATIENT_DATA.md)** — what
>   individual-level information the deposited `results/` artefacts **still
>   carry**. Removing the matrix removed the trial's `patientID` values; it did
>   not make this repository anonymous. Read this before treating it as such.
>
> Everything computed from `results/` — every table, every figure, every number
> quoted in the article — is here and reproduces exactly. The parts that do not
> run are listed in `data/README.md`; each of them says so when you run it,
> naming the file and where to ask for it.

Everything reported can be re-derived from this repository. The centrepiece is
**`PREDIX_HER2_reproducibility.ipynb`**, shipped executed: it recomputes each
deposited quantity from the model artefacts, asserts equality, and finally
re-runs both post-processing scripts end to end and compares every regenerated
workbook with its deposited counterpart cell by cell.

The notebook prints its own totals — checks passed, workbook cells compared,
mismatches, figures regenerated — and this tree ships the executed copy, so read
the count there rather than here. Re-execute `run_notebook.py` against this
tree and let its stored output be the record.

## Quick start

```bash
pip install -r requirements.txt

python code/tests/test_statistics.py     # ~190 statistical checks, ~25 s
python run_notebook.py                   # re-executes the whole verification, ~20-40 min
                                         # (four cells stand down without the
                                         #  input matrix and say so; see
                                         #  data/README.md)
```

Run both from the repository root — the notebook asserts that `code/` and
`results/` are in the working directory. To read the results without running
anything, open `PREDIX_HER2_reproducibility.ipynb`: the outputs are stored.

**[Read all the results here](RESULTS.md)** — every table and every figure, on
one page, generated from the deposited workbooks.

**Reviewing this work?** [FOR_REVIEWERS.md](FOR_REVIEWERS.md) is the shorter
route: what the one command proves, which section answers which question, what
this deposit does *not* establish, and the four things worth knowing before
reading the figures.

## Headline results

Cross-validated AUROC (95 % patient-level cluster-bootstrap CI), consensus
models. Source: `report/tables/revision/revision_performance_CI.xlsx`.

| Model | Pooled (n = 110, 46 pCR) | DHP (n = 59, 24 pCR) | T-DM1 (n = 51, 22 pCR) |
|---|---|---|---|
| Clinical | 0.61 (0.51–0.70) | 0.52 (0.41–0.62) | 0.70 (0.56–0.84) |
| Transcriptomic | 0.81 (0.73–0.88) | 0.81 (0.70–0.90) | 0.79 (0.65–0.90) |
| Genomic | 0.59 (0.52–0.67) | 0.73 (0.61–0.84) | 0.73 (0.59–0.86) |
| Proteomic | 0.73 (0.64–0.81) | 0.83 (0.71–0.92) | 0.67 (0.54–0.79) |
| Whole-slide image | 0.60 (0.51–0.69) | 0.62 (0.51–0.73) | 0.43 (0.40–0.46) |
| **Integrated (late fusion)** | **0.79 (0.72–0.86)** | **0.83 (0.73–0.91)** | **0.79 (0.69–0.88)** |

Integration is **not distinguishable** from the best single modality in any
scenario: every paired interval for ΔAUROC includes zero, so the comparison is
undetermined in **both** directions. It is not a finding that integration adds
nothing, and it should not be read as one.

Where the integrated model ranks, on the point estimates in the table above (six
models per cohort): **second of six in the pooled cohort**, below the transcriptomic (0.81) model; **first of six in the DHP arm**; **first of six in the T-DM1 arm**. Ranks are descriptive orderings of overlapping
intervals and carry no inferential claim; `RESULTS.md` section 3 tabulates the
full ordering, under the consensus and the fully nested discovery source, next
to every paired contrast.

External validation of the locked transcriptomic models, arm-matched, in the
three applicable cohorts —
I-SPY2, TransNEO and NCT02326974 (`report/tables/revision/external_validation.xlsx`):
I-SPY2 AUROC 0.79 (0.63–0.93), slope 0.77, P < 0.001; TransNEO AUROC 0.66 (0.53–0.80), slope 0.38, P = 0.012; NCT02326974 AUROC 0.71 (0.62–0.80), slope 0.64, P < 0.001. Features were standardised within each cohort by **z-score**
before the locked models were applied. The genomic model is **not** validatable
in the cohorts available and is reported as such rather than omitted:
three of its four signature features are not measured in the TransNEO metric file, and scoring a reduced signature would evaluate a different
model.

The **arm-matched** design — the model locked in the PREDIX arm corresponding to
the external regimen — is used for every external result, and it is the only
design applicable to every cohort: a signature locked on the pooled cohort can
contain features a given external cohort does not measure, and scoring a reduced
signature evaluates a different model. The rule is uniform applicability, fixed
in advance.

## Repository map

| path | contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **every table and figure of the analysis, rendered for reading here** — generated from the workbooks, nothing typed by hand |
| `PREDIX_HER2_reproducibility.ipynb` | the executed verification notebook (outputs embedded) |
| `run_notebook.py` | re-executes the notebook headless |
| `code/` | the eleven analysis scripts (`apply_locked_external_validation.py`, `apply_locked_signatures.py`, `compute_monte_carlo_error.py`, `cv_estimands.py`, `external_validation.py`, `feature_deduplication.py`, `generate_report.py`, `make_fig_feature_ranking.py`, `multimodal_pcr_pipeline.py`, `preflight.py`, `revision_analyses.py`), the test suite, and the production run scripts |
| `data/` | **the PREDIX input file is NOT deposited in this variant** (see [`data/README.md`](data/README.md)); the two external cohort files, derived from public GEO series, are |
| `results/` | production model artefacts: discovery and consensus PKLs per scenario, CV splits, `run_provenance.json`, `methods_cv_statement.txt` |
| `report/figures/`, `report/tables/` | the deposited figures and tables the notebook regenerates |
| `supplementary/` | the candidate feature panel (Table S-ML8) and the script that builds it |
| `docs/` | reproducibility guide, candidate-feature curation, the cross-validation statement and schematic, and [`NO_PATIENT_DATA.md`](docs/NO_PATIENT_DATA.md) |
| `environment/` | pinned requirements and the two environment records |
| `MANIFEST_SHA256.txt` | SHA-256 of every deposited file except the notebook and the manifest itself |

## The analysis in one page

`docs/ED_Fig11a_CV_schematic.pdf` draws the whole design; it is generated from
`results/run_provenance.json` by `docs/build_ED_Fig11a_schematic.py`, so it
cannot drift away from what was run.

Pipeline order (`code/production_run_ubuntu.sh` drives all four steps):

1. `multimodal_pcr_pipeline.py` — trains the models, writes the PKLs in `results/`
2. `generate_report.py` — figures and tables
3. `revision_analyses.py` — confidence intervals, calibration, stability, EPV
4. `external_validation.py` — the arm-matched locked-model validation in
   I-SPY2, TransNEO and NCT02326974, into `report/`

`cv_estimands.py` is imported by steps 2–4 and is the single definition of the
performance estimand.

## The estimand (statement of record)

In each cross-validation repeat every patient has exactly one out-of-fold
prediction; AUROC, AUPRC and the Brier score are computed on that complete
out-of-fold vector and averaged over the repeats (200 pooled, 100 per arm). The
95 % interval is a **patient-level cluster bootstrap** — 2,000 stratified
resamples of patients, a resampled patient carrying all of its repeat
predictions. Paired comparisons use the same patient resample for both models
and all repeats. **Predictions are never averaged across repeats or across
models**: doing so scores a 200-model ensemble instead of the model, and is
badly biased for weak models.

The "±" values in the submitted manuscript are the standard deviation of
per-fold AUROC, not a confidence interval.

## Two levels of reproduction

**Level A — post-processing, bit-for-bit (the notebook).** Everything downstream
of the model artefacts is seeded, so all tables and figures reproduce exactly
from `results/`.

**Level B — full pipeline re-run (many CPU-hours).** Step 1 of
`code/production_run_ubuntu.sh` re-trains everything (5-fold × 200 repeats
pooled, 5-fold × 100 per arm, seed 42). Cross-validation partitions are fully
determined by the seed, but classifier-internal randomness of the tree models is
deliberately **not** seeded — seeding it would correlate the repeats and
understate variance (see `reproducibility_note` in `results/run_provenance.json`).
Level B therefore reproduces linear-model numbers exactly and tree-model numbers
statistically; the deposited PKLs are the archival record that Level A verifies
exactly.

See `docs/REPRODUCIBILITY.md` for the section-by-section account of what the
notebook checks, measured runtimes, and the known caveats.

## Feature selection and leakage

The candidate panel and the leakage-free in-fold screen are documented in
`docs/CANDIDATE_FEATURE_CURATION.md`, with the complete 110-metric panel in
`supplementary/S-ML8_candidate_panel.xlsx`. In short: the a-priori biological
curation uses no outcome; the univariate association step, which in the original
submission had been applied across the whole cohort, is now performed inside
every training fold.

## Software

Python ≥ 3.10, `pip install -r requirements.txt`. The production model run used
Python 3.10.12 / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.7.2 / scipy 1.10.0
/ shap 0.49.1 on Ubuntu — recorded verbatim under
`environment` in `results/run_provenance.json`. The notebook was executed on
Windows 11 / Python 3.14.7 /
numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0
(`environment/pip_freeze_windows.txt`). Post-processing results are identical
across the two environments.

## Repository size

dominated by the deposited PKLs; the largest single file is under 50 MB, so
a plain `git push` works and Git LFS is not required. If
you prefer LFS, track `*.pkl` **before** the first commit:

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

## Verifying integrity

```bash
python - <<'PY'
import hashlib, pathlib
root = pathlib.Path('.')
bad = []
for line in (root / 'MANIFEST_SHA256.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    digest, rel = line.split(None, 1)
    p = root / rel.strip()
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'MISSING'
    if h != digest:
        bad.append(rel.strip())
print('mismatched or missing:', bad or 'none')
PY
```

## Before making this repository public

- [ ] Decide, and record, whether ethics approval and patient consent permit
      releasing `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt`
      (197 patients). Until they do, publish **this**
      variant rather than the full deposit.
- [ ] Decide the same question for what this variant *does* contain. The
      deposited `results/` artefacts carry per-patient outcomes, predicted
      probabilities and standardised feature values for
      110 of the 197 patients;
      `docs/NO_PATIENT_DATA.md` states exactly what and how much. This is a
      decision to take deliberately, not by omission.
- [ ] Confirm the redistribution terms of the two external cohort files
      (GEO GSE194040 and GSE243375) and add attribution.
- [x] Choose and write `LICENSE` — **done: MIT for the software, with a scope
      notice excluding all patient-derived material.** Read that notice; it is
      the part that does the work, not the MIT text above it.
- [ ] Fill in the data-access contact in `data/README.md`, under
      "Requesting the withheld file". **That file states that this repository
      must not be published while the placeholder remains** — a hard blocker,
      not a formality.
- [ ] Complete `CITATION.cff` — author list, repository URL, article title, year
      and DOI — and add the release DOI (Zenodo) to the article's Code
      availability statement. Its `license:` field is already set to MIT. The
      build prints every field still outstanding at the end of every run.

## Known caveats

- Completeness is defined on the molecular modalities only — the clinical block
  never enters it — so the complete-case cohort is **n = 110** (46 pCR; DHP 59/24, T-DM1 51/22), matching the
  submitted manuscript. `Clin_TUMSIZE` and `Clin_prolifvalu` encode missing as
  the **string** `Unknown` rather than as `NaN`, so a plain `dropna()` over every
  feature column returns 104 rows and is not the pipeline's cohort; the rule that
  is, is `get_complete_case()`.
- Models are trained and evaluated on the same 110 complete
  cases. Patients carrying only some assays do not enter the analysis.
- The `locked_from` column of `external_validation.xlsx` records the command-line
  path as typed in the production run; a re-run from this repository records the
  local path instead. The notebook prints this difference rather than skipping it.

Generated 2026-09-06 from pipeline version 2.0.0-revision1, seed 42.
