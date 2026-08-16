# PREDIX HER2 — multimodal prediction of pathological complete response

Analysis code, deposited cross-validation artefacts, and an executed
verification notebook for the multimodal pCR-prediction analysis of the
PREDIX HER2 randomised trial (clinical, transcriptomic, genomic, proteomic and
whole-slide-image data).

Everything reported can be re-derived from this repository. The centrepiece is
**`PREDIX_HER2_reproducibility.ipynb`**, shipped executed: it recomputes each
deposited quantity from the model artefacts, asserts equality, and finally
re-runs both post-processing scripts end to end and compares every regenerated
workbook with its deposited counterpart cell by cell.

Last full verification: **36 / 36 checks passed**, **180,749 workbook cells
compared, 0 mismatches**, all 31 figures regenerated, in ≈ 20 minutes.

## Quick start

```bash
pip install -r requirements.txt

python code/tests/test_statistics.py     # ~190 statistical checks, ~25 s
python run_notebook.py                   # re-executes the whole verification, ~20-40 min
```

Run both from the repository root — the notebook asserts that `code/` and
`results/` are in the working directory. To read the results without running
anything, open `PREDIX_HER2_reproducibility.ipynb`: the outputs are stored.

**[Read all the results here](RESULTS.md)** — every table and every figure, on
one page, generated from the deposited workbooks.

## Headline results

Cross-validated AUROC (95 % patient-level cluster-bootstrap CI), consensus
models. Source: `report/tables/revision/revision_performance_CI.xlsx`.

| Model | Pooled (n = 109, 46 pCR) | DHP (n = 58, 24) | T-DM1 (n = 51, 22) |
|---|---|---|---|
| Clinical | 0.61 (0.51–0.70) | 0.57 (0.44–0.70) | 0.58 (0.47–0.69) |
| Transcriptomic | 0.78 (0.70–0.86) | 0.79 (0.67–0.89) | 0.81 (0.69–0.91) |
| Genomic | 0.52 (0.45–0.58) | 0.64 (0.53–0.75) | 0.62 (0.51–0.72) |
| Proteomic | 0.75 (0.66–0.84) | 0.78 (0.66–0.88) | 0.69 (0.55–0.82) |
| Whole-slide image | 0.60 (0.49–0.70) | 0.62 (0.49–0.75) | 0.54 (0.43–0.65) |
| **Integrated (late fusion)** | **0.78 (0.71–0.86)** | **0.78 (0.67–0.88)** | **0.77 (0.66–0.87)** |

Integration is **not distinguishable** from the best single modality in any
scenario (pooled ΔAUROC vs transcriptomic −0.00, 95 % CI −0.04 to 0.04,
P = 0.99); it is higher than the clinical, genomic and WSI models. External
validation of the locked transcriptomic models: I-SPY2 (GSE194040) AUROC 0.77
(0.62–0.90), calibration slope 1.07 — transfers; NCT02326974 (GSE243375) AUROC
0.55 (0.44–0.64), slope 0.24 — does not transfer.

## Repository map

| path | contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **every table and figure of the analysis, rendered for reading here** — generated from the workbooks, nothing typed by hand |
| `PREDIX_HER2_reproducibility.ipynb` | the executed verification notebook (outputs embedded) |
| `run_notebook.py` | re-executes the notebook headless |
| `code/` | the five analysis scripts, the test suite, and the production run scripts |
| `data/` | the PREDIX input file and the two external cohort files |
| `results/` | production model artefacts: discovery and consensus PKLs per scenario, CV splits, `run_provenance.json`, `methods_cv_statement.txt` |
| `results_rna_ispy2/`, `results_rna_nct/` | the RNA-only locked-model runs behind the external validation |
| `report/figures/`, `report/tables/` | the deposited figures and tables the notebook regenerates |
| `supplementary/` | the candidate feature panel (Table S-ML9) and the script that builds it |
| `docs/` | reproducibility guide, candidate-feature curation, the cross-validation statement and schematic |
| `environment/` | pinned requirements and the two environment records |
| `MANIFEST_SHA256.txt` | SHA-256 of every file in the repository |

## The analysis in one page

`docs/ED_Fig11a_CV_schematic.pdf` draws the whole design; it is generated from
`results/run_provenance.json` by `docs/build_ED_Fig11a_schematic.py`, so it
cannot drift away from what was run.

Pipeline order (`code/production_run_ubuntu.sh` drives all four steps):

1. `multimodal_pcr_pipeline.py` — trains the models, writes the PKLs in `results/`
2. `generate_report.py` — figures and tables
3. `revision_analyses.py` — confidence intervals, calibration, stability, EPV, biomarker groups
4. `external_validation.py` — the locked-model validation in I-SPY2 and NCT02326974

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

Any "±" value in an earlier version of this work is a standard deviation of
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
`docs/CANDIDATE_FEATURE_CURATION.md`, with the complete 112-metric panel in
`supplementary/S-ML9_candidate_panel.xlsx`. In short: the a-priori biological
curation uses no outcome; the univariate association step, which in the original
submission had been applied across the whole cohort, is now performed inside
every training fold.

## Software

Python ≥ 3.10, `pip install -r requirements.txt`. The production model run used
Python 3.10.12 / numpy 2.2.6 / pandas 2.3.3 / scikit-learn 1.7.2 on Ubuntu
(`environment/production_environment.json`); the notebook was executed on
Windows 11 / Python 3.14.7 / numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0
(`environment/pip_freeze_windows.txt`). Post-processing results are identical
across the two environments.

## Repository size

≈ 95 MB, dominated by the deposited PKLs; the largest single file is ≈ 31 MB, so
a plain `git push` works and Git LFS is not required. If you prefer LFS, track
`*.pkl` **before** the first commit:

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

- [ ] Confirm that ethics approval and patient consent permit releasing
      `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt` (197 patients).
- [ ] Confirm the redistribution terms of the two external cohort files
      (GEO GSE194040 and GSE243375) and add attribution.
- [ ] Choose and write `LICENSE` (see the placeholder for the usual arrangement).
- [ ] Complete `CITATION.cff` and add the release DOI (Zenodo) to the article's
      Code availability statement.

## Known caveats

- The complete-case cohort is **n = 109** here against n = 110 in the submitted
  manuscript (DHP 58 vs 59): one DHP patient lacks complete molecular data in the
  canonical data delivery. Under reconciliation with the data provider.
- The S1/S2/S3 biomarker groups use the default driver specification in
  `code/revision_analyses.py`; the exact driver columns are still to be confirmed.
  These groups are exploratory and the quotable interaction P value is the
  permutation-corrected one.
- The `locked_from` column of `external_validation.xlsx` records the command-line
  path as typed in the production run; a re-run from this repository records the
  local path instead. The notebook prints this difference rather than skipping it.

Generated 2026-08-16 from pipeline version 2.0.0-revision1, seed 42.
