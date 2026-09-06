# What individual-level information this repository still contains

This is the no-patient-data variant of the PREDIX HER2 deposit. The trial's
individual-level input matrix is not in it (see `data/README.md`). This page
states, factually and without argument in either direction, what per-patient
information the rest of the deposit still carries, so that the authors and
their ethics committee can decide about it deliberately rather than discover it
later.

Every count on this page was measured from the artefacts in this repository
when it was assembled, by the build script that wrote the page. Nothing here is
typed from memory.

## 1. The short version

Removing the input matrix removed the trial's `patientID` values and the
authoritative feature matrix. It did not remove individual-level data. The
deposited cross-validation artefacts under `results/` contain, for
**110 of the 197 trial patients**:

* the patient's pCR outcome;
* 450,941 predicted probabilities in total across the deposit;
* which cross-validation fold, in each of the repeats, the patient was held out in;
* the treatment arm (derivable — see §4);
* and, through `oof_shap['X_test_scaled']`, the **within-fold standardised
  values of the curated metrics themselves**: 74 of the
  110 metrics appear, a median of 63 per patient
  (range 57–69), 6,902 patient-metric cells in all.

The last item is the one that is easy to miss. §5 shows that those standardised
values are enough to reconstruct the withheld columns.

## 2. The identifier

The pipeline never puts the trial's `patientID` into any artefact. In
`code/multimodal_pcr_pipeline.py`:

```python
df["patient_id"] = range(len(df))   # a stable integer id, assigned before any filtering
```

so the `test_pids` arrays in every fold record, and the "Patient global ID"
column of `report/tables/supplementary/supp_PREDIX_HER2_results.xlsx`, are
**0-based row positions in the input matrix**, not trial patient numbers. Grep
this repository for a PREDIX `patientID` value and you will find none.

Row position is nevertheless a stable pseudonym, and it is not a strong one:
the withheld matrix is sorted by ascending `patientID`, so row position *i* is
the *i*-th smallest patient number in the 197-patient
cohort. Anyone holding the cohort's patient-number list — the trial team, a
participating site, anyone who is later given the matrix — can invert the
mapping exactly. The pseudonym protects against a reader who has no other
PREDIX information; it does not protect against one who has.

## 3. Which patients appear, and where

110 distinct row positions appear across the deposited fold
records; 87 of the 197 rows appear
nowhere in this repository at all.

| artefact | patients appearing | fold records |
|---|---:|---:|
| `results/dhp/dhp_consensus_eval.pkl` | 59 | 500 |
| `results/dhp/dhp_elasticnet_results.pkl` | 59 | 3,000 |
| `results/global/global_consensus_eval.pkl` | 110 | 1,000 |
| `results/global/global_elasticnet_results.pkl` | 110 | 6,000 |
| `results/tdm1/tdm1_consensus_eval.pkl` | 51 | 500 |
| `results/tdm1/tdm1_elasticnet_results.pkl` | 51 | 3,000 |

`results/` covers the n = 110 complete-case cohort, which is
the whole of this analysis: every model is fitted and scored on those patients.

## 4. Exactly what survives, per patient

**Survives.**

* **Outcome.** `y_test` in every fold record: the patient's pCR, 0 or 1.
  Because a patient recurs in every repeat, the outcome is present many times
  over; it is not recoverable "on balance", it is simply present.
* **Predicted probabilities.** `y_pred` (per modality), `unimodal_y_pred`,
  `fused_y_pred` and `cross_arm_preds` — 450,941 numbers over
  the whole deposit, up to a few thousand per patient. Per-patient means with
  confidence intervals are also published in readable form in the
  `Counterfactual` sheet of
  `report/tables/supplementary/supp_PREDIX_HER2_results.xlsx`, one row per
  patient.
* **Fold membership.** `test_idx`/`test_pids` per fold and repeat, and
  `results/shared_splits/*.pkl`, which fix the partitions exactly.
* **Treatment arm.** Not stored as such, but determined by which directory a
  patient appears in: `results/dhp/` and `results/tdm1/` partition the
  complete-case cohort.
* **Modality completeness.** Also determined by membership: appearing anywhere
  under `results/` means the patient had complete transcriptomics, genomics,
  proteomics and imaging, because that is the cohort the analysis is fitted and
  scored on.
* **Feature values, standardised within fold.** `oof_shap['X_test_scaled']` is
  stored for every fold record and is row-aligned with `test_pids`. It holds
  the held-out patients' own values for the features that survived that fold's
  screen, after median imputation and standardisation on that fold's training
  patients. Coverage: 74 of the 110 curated
  metrics, 6,902 patient-metric cells, a median of 63 metrics
  per patient (range 57–69).
  By modality: Clin 5, DNA 26, Prot 16, RNA 24, WSI 3.
* **SHAP attributions.** `oof_shap['shap_values']`, the same shape as the
  above: one attribution per patient per feature.

**Does not survive.**

* The trial's `patientID` values, in any form.
* The input matrix itself, and with it the authoritative, unstandardised values
  in their original units.
* The 36 curated metrics that never entered any fold's
  screen — mostly the outcome-blind deduplication list, which is removed before
  the fold loop: `DNA_CDK12_CNA`, `DNA_CTTN_CNA`, `DNA_FADD_CNA`, `DNA_GRB7_CNA`, `DNA_MED1_CNA`, `DNA_MIEN1_CNA`, `DNA_PPP1R1B_CNA`, `DNA_RPL19_CNA`, `DNA_coding_mutation_ERBB2`, `DNA_coding_mutation_ERBB2_oncokb`, `DNA_coding_mutation_GATA3_oncokb`, `DNA_coding_mutation_HER_pathway`, `DNA_coding_mutation_PIK3CA_oncokb`, `DNA_coding_mutation_PIK3_AKT_pathway`, `DNA_coding_mutation_TP53_oncokb`, `Prot_ERBB2`, `Prot_GRB7`, `Prot_MIEN1`, `RNA_B-cells`, `RNA_BCR_clonality`, `RNA_CD45`, `RNA_CD8-T-cells`, `RNA_Cytotoxic-cells`, `RNA_DC`, `RNA_Dysfunction`, `RNA_FCGR3B`, `RNA_HER2DX_IGG`, `RNA_HER2DX_pCR_likelihood_score`, `RNA_Macrophages`, `RNA_Oxidative_phosphorylation`, `RNA_T-cells`, `RNA_TILs`, `RNA_mRNA-CD8A`, `RNA_mRNA-ERBB2`, `RNA_mRNA-ESR1`, `RNA_mRNA-MKI67`.
* Any measurement of the 87 patients who appear in no fold
  record.
* Anything that was never in the matrix in the first place: age, dates, site,
  histopathology reports, follow-up, survival, free text, images. The
  whole-slide-image modality is present only as three summary metrics per
  patient; no slide, tile or image-derived map is deposited.

## 5. Whether the standardised values can be turned back into the real ones

Yes, essentially exactly, and it does not take an unusual amount of work, so
the possibility should be assumed rather than weighed.

Standardisation is an affine map with constants that depend on the fold, and
each patient is held out once per repeat, so any two folds of different repeats
share several patients. That is enough to solve for a common scale across all
folds, and hence to recover each column up to a single affine transform of the
whole column — which preserves every rank, every ratio of differences, and
every standardised value.

The build script verifies this on one metric each time it assembles this
repository, using only files that are deposited here:

* metric: `WSI_Immune_Cell_prop`
* reconstructed from 1,000 fold records covering 110 patients
* Pearson correlation with the withheld column: **1.000000**
* largest residual after a single affine map of the whole column: **0.0000 % of the column's range**

The reconstruction script is deliberately not shipped. Its absence is not a
control: the method is three lines of least squares.

## 6. So what is the disclosure, stated plainly

For 110 pseudonymous individuals — identifiable by name only
to someone who also holds the trial's patient-number list — this repository
discloses the pCR outcome, the treatment arm, which molecular assays succeeded,
and a median of 63 of the 110 curated clinical,
transcriptomic, genomic, proteomic and imaging metrics, recoverable up to a
per-metric affine transform. It does not disclose any direct identifier, and it
does not disclose anything about the 87 patients who appear in
no fold record.

Whether that is acceptable to release is a question for the trial's ethics
approval and the participants' consent, not for this document.

## 7. Why the artefacts were kept

They are the deposit. The reproducibility argument this repository exists to
make — that every published number can be recomputed from what is deposited —
rests on the fold records: the notebook's cell-by-cell verification, the
regeneration of every table and figure by `generate_report.py` and
`revision_analyses.py`, and `compute_monte_carlo_error.py` all read them and
nothing else. Removing them would leave a repository of scripts and PDFs that
verifies nothing.

The alternative, if what §4 and §5 describe cannot be released, is not to
publish the fold records with the feature values stripped and hope the rest
still verifies. It is to decide first, and then to rebuild: dropping
`oof_shap['X_test_scaled']` alone would remove the feature-value disclosure
while leaving outcomes, predictions and fold membership in place, at the cost
of the SHAP figures (`fig05_*`, `supp_fig05_*`, `supp_fig08_*`) and of the
deposited artefacts no longer being the ones the manuscript's SHA-256 refers
to. That change is **not** made by this build script, deliberately: it is an
authors' decision, not a build option.

## 8. How this variant's code differs from the full deposit's

The full deposit ships `code/` byte-identical to the scripts that produced the
deposited results. This variant cannot, because scripts that read the withheld
matrix have to be able to say so. The complete list of differences, and nothing
else, is below; every insertion is bracketed by a comment containing
`NO-PATIENT-DATA VARIANT`, so `grep -rn "NO-PATIENT-DATA VARIANT" .` finds all
of them. Each is an existence test that falls through when the matrix is
present.

| file | SHA-256 in the full deposit | SHA-256 here | what changed |
|---|---|---|---|
| `PREDIX_HER2_reproducibility.ipynb` | `826776aa550e…` | `0566cf2737d7…` | four code cells guarded so they explain themselves instead of raising FileNotFoundError, plus one step inside Section 12 that guards itself in both variants; one markdown cell added; stored outputs untouched |
| `code/external_validation.py` | `f75931bc8b58…` | `daaa2404ab02…` | a missing --predix stops main() with the explanation instead of a pandas FileNotFoundError |
| `code/multimodal_pcr_pipeline.py` | `59e9733ae76a…` | `cab128b54b7d…` | a missing or absent --data_path stops main() at once with the explanation, instead of raising FileNotFoundError from the provenance hash deeper in the run |
| `code/preflight.py` | `1d8a6d05c9b6…` | `f0af4c33a1f0…` | looks for the input in this repository's data/ as well as in the working directory, and its MISSING-file exit message now names the variant, the file's fingerprint and where to ask for it |
| `code/revision_analyses.py` | `92332b2431f3…` | `a927a3ba17ac…` | prints a one-line notice when --data_path points at nothing; the script still produces every table and figure it always did |
| `supplementary/build_S-ML8_candidate_panel.py` | `06d424f1d1ee…` | `6f7042651cd8…` | explains the missing input with the standard banner instead of naming a bare path |

The `.py` digests are of the files as they sit on disk. The notebook's two
digests are of its **cell sources only**, because `run_notebook.py` re-executes
it in place and rewrites every output and execution count: a file digest quoted
here would be stale the moment the deposit's own verification step ran, and a
stale digest presented as a checked value is worse than none. The guard being
attested to lives in the cell source, which execution does not touch.
`verify_github_repo.py` recomputes both digests the same way.

---
Generated 2026-09-06 by `build_github_repo.py --no-patient-data`, from the
artefacts in this repository.
