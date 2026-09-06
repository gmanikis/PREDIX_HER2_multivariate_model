# For reviewers — checking that the manuscript's numbers come from this code

One command answers most of it:

```bash
pip install -r requirements.txt
python run_notebook.py          # ~20-40 min
```

**26 of the notebook's checks run.** Sections 1, 2 and part of 11 stand down because they need the withheld input matrix, so this variant cannot confirm the cohort construction: the input file's SHA-256, its 197x112 shape, the Clin 5 / RNA 42 / DNA 41 / Prot 19 / WSI 3 panel, and the derivation of the n = 110 evaluation cohort are all asserted here and verifiable only in the full deposit.

You do not have to run anything to read the answers. The notebook ships
**executed**, with every output stored, and [RESULTS.md](RESULTS.md) presents
every table and figure on one page.

## What the run actually establishes

| Question | Where it is answered |
|---|---|
| Do the reported AUROCs come from the deposited cross-validation? | Section 5 — `revision_performance_CI.xlsx` recomputed from `results/*.pkl`, 360 values, tolerance 1e-9 |
| Is the cross-validation what the Methods describe? | Section 4 — all 18 (scenario, model) pairs: 5 folds per repeat, 200/100/100 complete repeats, **every patient predicted exactly once per repeat**, cohorts 110/59/51 with 46/24/22 events |
| Are the paired comparisons and their corrections right? | Section 7 — 300 values including verdicts, and the Benjamini-Hochberg family size (m = 15 per source) |
| Is the calibration as reported? | Section 8 |
| Are the signatures, weights and EPV as reported? | Sections 9 and 10 |
| Does every deposited table and figure follow from `results/`? | **Section 12 — the whole report is regenerated and compared cell by cell: ~180,000 cells across 12 workbooks, and every figure redrawn** |
| Did anything change after deposit? | Section 13 — SHA-256 of every file, against `MANIFEST_SHA256.txt` |

Section 12 is the one that speaks to "does the manuscript come from this code":
it rebuilds the report from the model artefacts using the shipped scripts and
requires **zero** mismatching cells.

## Two levels, and which one this is

**Level A — post-processing, bit-for-bit.** Everything downstream of the model
artefacts is seeded and reproduces exactly from `results/`. This is what
`run_notebook.py` verifies.

**Level B — full pipeline re-run** (many CPU-hours) via
`code/production_run_ubuntu.sh`. Cross-validation partitions are fixed by
seed 42, but tree-model internal randomness is deliberately **not** seeded —
seeding it would correlate the repeats and understate variance. Level B
therefore reproduces linear-model numbers exactly and tree-model numbers
statistically. Level B cannot be run in this variant: it needs the withheld input matrix.

## Where each display item comes from

`RESULTS.md` names the source workbook under every table and figure, and
`report/` mirrors the manuscript's structure:

- `report/tables/revision/` — **the tables the manuscript quotes.** Quote from
  here, not from the archive.
- `report/figures/` and `report/figures/revision/` — the deposited artwork
- `report/figures_png/` — the same figures as PNG, for reading on GitHub
- `external_validation/` — the three applicable validation sets
  (I-SPY2, TransNEO and NCT02326974) under one procedure, the model that could not be scored
  and why, and the code that produced all of it

## Things worth knowing before you read the figures

- **Direction in `fig02` and `supp_fig06` is drawn with a discredited
  statistic.** Mean SHAP averages to about zero and does not carry direction; it
  agreed with the SHAP dependence slope on only 29 of 63 consensus features.
  Read direction from `fig_feature_ranking_by_scope` and its workbook. The full
  account is in `docs/FEATURE_DIRECTION_CORRECTION.md`.
- **The genomic model is not externally validated**, and that is reported
  rather than omitted: three of its four signature features are not measured in the TransNEO metric file, the only external cohort carrying
  genomic metrics. See `external_validation/README.md`.
- **Every external result uses the arm-matched design** — the model locked in
  the PREDIX arm corresponding to the external regimen. It is the only design
  applicable to every cohort, because a signature locked on the pooled cohort
  can require features a given external cohort does not measure, and scoring a
  reduced signature evaluates a different model.
- **The candidate panel was curated with knowledge of the outcome literature**;
  this is stated rather than glossed. See `docs/CANDIDATE_FEATURE_CURATION.md`.

## If something does not reproduce

That is worth reporting, and the deposit is built to make it diagnosable: every
check prints what it compared and to what tolerance, and
`docs/REPRODUCIBILITY.md` gives the section-by-section account with measured
runtimes and known caveats.
