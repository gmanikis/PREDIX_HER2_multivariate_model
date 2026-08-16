# Results

Every table on this page is generated directly from the deposited workbooks under [`report/tables/`](report/tables) by [`docs/build_RESULTS_md.py`](docs/build_RESULTS_md.py), and every figure is the PNG rendering of the corresponding PDF in [`report/figures/`](report/figures). Nothing here is typed by hand.

Pipeline `2.0.0-revision1`, seed 42. Regenerated 2026-08-16.

> **How to read every number below.** In each cross-validation repeat every patient has exactly one out-of-fold prediction; the metric is computed on that complete out-of-fold vector and averaged over the repeats (200 pooled, 100 per arm). The 95% interval is a patient-level **cluster** bootstrap — 2,000 stratified resamples of patients, a resampled patient carrying all of its repeat predictions. Predictions are never averaged across repeats or across models. A comparison whose interval for ΔAUROC includes zero is reported as *not distinguishable*, however large the point difference.

## Contents

1. [Design and cohort](#1-design-and-cohort)
2. [Cross-validated performance](#2-cross-validated-performance)
3. [Is integration better than the best single modality?](#3-is-integration-better-than-the-best-single-modality)
4. [Calibration](#4-calibration)
5. [Events per variable](#5-events-per-variable)
6. [Feature-selection stability](#6-feature-selection-stability)
7. [Consensus signatures and fusion weights](#7-consensus-signatures-and-fusion-weights)
8. [Exploratory biomarker groups](#8-exploratory-biomarker-groups)
9. [External validation](#9-external-validation)
10. [Figures](#10-figures)

## 1. Design and cohort

| Cohort | Patients | pCR events | pCR rate | CV repeats | Outer evaluations |
|---|---|---|---|---|---|
| Pooled cohort | 109 | 46 | 42.2% | 200 | 1,000 |
| DHP arm | 58 | 24 | 41.4% | 100 | 500 |
| T-DM1 arm | 51 | 22 | 43.1% | 100 | 500 |

| Design element | Value |
|---|---|
| Outer resampling | stratified 5-fold `RepeatedStratifiedKFold` (no shuffle-split) |
| Inner resampling | 5-fold (pooled), 3-fold (per arm) |
| Candidate panel | 112 pre-defined metrics → 101 after the outcome-blind biological deduplication |
| Feature screen | in-fold Mann–Whitney AUROC, BH q ≤ 0.25, keep 5–40 |
| Classifier families | `ElasticNet_LR`, `RandomForest`, `ExtraTrees`, `HistGradBoost`, `SVM_Linear` |
| Signature size cap | at least 5 pCR events per selected variable |
| Fusion | elastic-net logistic regression (l1_ratio 0.5) over the five Platt-calibrated modality probability streams |
| Consensus finalisation | features above the stability threshold (0.6 pooled, 0.5 per arm); modal classifier |
| Random seed | 42 |

The full design is stated in [`docs/methods_cv_statement.txt`](docs/methods_cv_statement.txt), both generated from the run's own parameters.

## 2. Cross-validated performance

Consensus models — the frozen signature and classifier re-evaluated on the same outer splits. Source: `report/tables/revision/revision_performance_CI.xlsx`.

### Pooled cohort

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.609 | 0.509–0.700 | 0.531 | 0.456–0.637 | 0.226 | 0.199–0.254 |
| Transcriptomic | 0.785 | 0.701–0.861 | 0.758 | 0.669–0.849 | 0.183 | 0.150–0.217 |
| Genomic | 0.519 | 0.448–0.585 | 0.452 | 0.412–0.527 | 0.247 | 0.240–0.255 |
| Proteomic | 0.752 | 0.660–0.836 | 0.654 | 0.567–0.775 | 0.198 | 0.166–0.233 |
| Whole-slide image | 0.595 | 0.485–0.701 | 0.571 | 0.480–0.679 | 0.237 | 0.219–0.257 |
| **Integrated (late fusion)** | **0.784** | 0.710–0.858 | 0.736 | 0.651–0.834 | 0.186 | 0.159–0.216 |

### DHP arm

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.568 | 0.445–0.696 | 0.479 | 0.399–0.628 | 0.246 | 0.221–0.273 |
| Transcriptomic | 0.793 | 0.674–0.895 | 0.702 | 0.576–0.871 | 0.188 | 0.134–0.241 |
| Genomic | 0.644 | 0.534–0.751 | 0.542 | 0.473–0.662 | 0.231 | 0.205–0.258 |
| Proteomic | 0.777 | 0.663–0.878 | 0.644 | 0.540–0.805 | 0.171 | 0.126–0.218 |
| Whole-slide image | 0.621 | 0.493–0.746 | 0.542 | 0.455–0.697 | 0.236 | 0.207–0.267 |
| **Integrated (late fusion)** | **0.778** | 0.667–0.877 | 0.665 | 0.571–0.810 | 0.191 | 0.160–0.227 |

### T-DM1 arm

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.583 | 0.472–0.688 | 0.509 | 0.436–0.658 | 0.256 | 0.222–0.292 |
| Transcriptomic | 0.808 | 0.688–0.914 | 0.765 | 0.641–0.902 | 0.179 | 0.128–0.237 |
| Genomic | 0.617 | 0.512–0.718 | 0.590 | 0.507–0.712 | 0.243 | 0.214–0.275 |
| Proteomic | 0.689 | 0.550–0.816 | 0.679 | 0.564–0.818 | 0.222 | 0.181–0.267 |
| Whole-slide image | 0.541 | 0.429–0.648 | 0.505 | 0.434–0.635 | 0.255 | 0.230–0.281 |
| **Integrated (late fusion)** | **0.771** | 0.658–0.870 | 0.724 | 0.619–0.852 | 0.198 | 0.164–0.238 |

### Discovery phase (fully nested)

The signature and classifier are re-selected independently inside every fold, so these estimates carry no consensus selection optimism. They are the conservative reading of the same data.

| Cohort | Model | Discovery AUROC | 95% CI | Consensus − discovery |
|---|---|---|---|---|
| Pooled cohort | Clinical | 0.605 | 0.513–0.700 | +0.004 |
| Pooled cohort | Transcriptomic | 0.759 | 0.675–0.840 | +0.026 |
| Pooled cohort | Genomic | 0.522 | 0.456–0.589 | -0.003 |
| Pooled cohort | Proteomic | 0.694 | 0.606–0.776 | +0.058 |
| Pooled cohort | Whole-slide image | 0.569 | 0.480–0.655 | +0.026 |
| Pooled cohort | Integrated (late fusion) | 0.750 | 0.663–0.829 | +0.035 |
| DHP arm | Clinical | 0.542 | 0.424–0.663 | +0.026 |
| DHP arm | Transcriptomic | 0.765 | 0.641–0.874 | +0.027 |
| DHP arm | Genomic | 0.596 | 0.492–0.698 | +0.048 |
| DHP arm | Proteomic | 0.766 | 0.658–0.865 | +0.011 |
| DHP arm | Whole-slide image | 0.581 | 0.460–0.698 | +0.040 |
| DHP arm | Integrated (late fusion) | 0.748 | 0.640–0.850 | +0.030 |
| T-DM1 arm | Clinical | 0.615 | 0.496–0.723 | -0.033 |
| T-DM1 arm | Transcriptomic | 0.737 | 0.620–0.846 | +0.071 |
| T-DM1 arm | Genomic | 0.619 | 0.504–0.727 | -0.002 |
| T-DM1 arm | Proteomic | 0.642 | 0.502–0.774 | +0.047 |
| T-DM1 arm | Whole-slide image | 0.517 | 0.405–0.634 | +0.024 |
| T-DM1 arm | Integrated (late fusion) | 0.719 | 0.606–0.824 | +0.052 |

## 3. Is integration better than the best single modality?

**No.** Paired patient-level cluster bootstrap, the same patient resample applied to both models and all repeats.

| Cohort | Integrated AUROC | Best single modality | Δ AUROC | 95% CI | P | Verdict |
|---|---|---|---|---|---|---|
| Pooled cohort | 0.784 | Transcriptomic 0.785 | -0.000 | -0.036 to 0.037 | 0.990 | not distinguishable |
| DHP arm | 0.778 | Transcriptomic 0.793 | -0.015 | -0.065 to 0.039 | 0.561 | not distinguishable |
| T-DM1 arm | 0.771 | Transcriptomic 0.808 | -0.037 | -0.086 to 0.014 | 0.160 | not distinguishable |

Against every comparator:

| Cohort | Integrated vs | Δ AUROC | 95% CI | P | BH q | Verdict |
|---|---|---|---|---|---|---|
| Pooled cohort | Clinical | +0.176 | 0.071–0.274 | 0.001 | 0.003 | integrated higher |
| Pooled cohort | Transcriptomic | -0.000 | -0.036 to 0.037 | 0.990 | 0.990 | not distinguishable |
| Pooled cohort | Genomic | +0.266 | 0.165–0.367 | < 0.001 | 0.003 | integrated higher |
| Pooled cohort | Proteomic | +0.033 | -0.031 to 0.097 | 0.292 | 0.365 | not distinguishable |
| Pooled cohort | Whole-slide image | +0.189 | 0.060–0.315 | 0.006 | 0.010 | integrated higher |
| DHP arm | Clinical | +0.210 | 0.058–0.352 | 0.006 | 0.015 | integrated higher |
| DHP arm | Transcriptomic | -0.015 | -0.065 to 0.039 | 0.561 | 0.701 | not distinguishable |
| DHP arm | Genomic | +0.134 | 0.046–0.221 | 0.004 | 0.015 | integrated higher |
| DHP arm | Proteomic | +0.001 | -0.049 to 0.050 | 0.926 | 0.926 | not distinguishable |
| DHP arm | Whole-slide image | +0.157 | -0.009 to 0.312 | 0.074 | 0.123 | not distinguishable |
| T-DM1 arm | Clinical | +0.189 | 0.037–0.340 | 0.014 | 0.035 | integrated higher |
| T-DM1 arm | Transcriptomic | -0.037 | -0.086 to 0.014 | 0.160 | 0.197 | not distinguishable |
| T-DM1 arm | Genomic | +0.154 | 0.021–0.278 | 0.024 | 0.040 | integrated higher |
| T-DM1 arm | Proteomic | +0.082 | -0.044 to 0.205 | 0.197 | 0.197 | not distinguishable |
| T-DM1 arm | Whole-slide image | +0.231 | 0.084–0.374 | 0.004 | 0.020 | integrated higher |

DeLong's test computed per repeat and summarised is reported in the workbook as a descriptive secondary analysis; the bootstrap is the primary comparison.

## 4. Calibration

Slope and intercept of `logit(pCR) = a + b · logit(p̂)`, fitted on each repeat's out-of-fold vector and averaged. Slope 1 and intercept 0 are perfect; slope below 1 means the predictions are too extreme (the classic overfitting signature), above 1 that they are compressed toward the base rate.

| Cohort | Slope | 95% CI | Intercept | 95% CI | Brier | ECE | Observed vs mean predicted |
|---|---|---|---|---|---|---|---|
| Pooled cohort | 1.16 | 0.80–1.69 | 0.09 | -0.12 to 0.38 | 0.186 | 0.099 | 0.422 vs 0.415 |
| DHP arm | 1.49 | 0.86–2.71 | 0.14 | -0.16 to 0.70 | 0.191 | 0.133 | 0.414 vs 0.412 |
| T-DM1 arm | 1.21 | 0.74–2.66 | 0.15 | -0.06 to 0.76 | 0.198 | 0.123 | 0.431 vs 0.416 |

Every slope interval covers 1 and every intercept interval covers 0.

<details><summary>Reliability bins (equal-count bins over all (patient, repeat) out-of-fold predictions)</summary>

| Cohort | Bin | Predictions | Distinct patients | Mean predicted | Observed | 95% CI |
|---|---|---|---|---|---|---|
| Pooled cohort | 1 | 2,180 | 47 | 0.089 | 0.063 | 0.015–0.136 |
| Pooled cohort | 2 | 2,180 | 63 | 0.171 | 0.173 | 0.072–0.285 |
| Pooled cohort | 3 | 2,180 | 71 | 0.234 | 0.214 | 0.129–0.317 |
| Pooled cohort | 4 | 2,180 | 78 | 0.294 | 0.299 | 0.200–0.406 |
| Pooled cohort | 5 | 2,180 | 78 | 0.357 | 0.368 | 0.269–0.481 |
| Pooled cohort | 6 | 2,180 | 85 | 0.418 | 0.460 | 0.341–0.549 |
| Pooled cohort | 7 | 2,180 | 77 | 0.492 | 0.450 | 0.353–0.580 |
| Pooled cohort | 8 | 2,180 | 69 | 0.586 | 0.583 | 0.436–0.739 |
| Pooled cohort | 9 | 2,180 | 51 | 0.689 | 0.756 | 0.610–0.857 |
| Pooled cohort | 10 | 2,180 | 40 | 0.821 | 0.855 | 0.708–0.971 |
| DHP arm | 1 | 829 | 25 | 0.117 | 0.054 | 0.000–0.151 |
| DHP arm | 2 | 829 | 31 | 0.238 | 0.094 | 0.013–0.211 |
| DHP arm | 3 | 828 | 39 | 0.325 | 0.273 | 0.085–0.588 |
| DHP arm | 4 | 829 | 39 | 0.435 | 0.607 | 0.422–0.750 |
| DHP arm | 5 | 828 | 37 | 0.503 | 0.575 | 0.461–0.732 |
| DHP arm | 6 | 829 | 36 | 0.569 | 0.632 | 0.497–0.776 |
| DHP arm | 7 | 828 | 32 | 0.695 | 0.662 | 0.481–0.835 |
| T-DM1 arm | 1 | 850 | 38 | 0.135 | 0.169 | 0.052–0.297 |
| T-DM1 arm | 2 | 850 | 42 | 0.274 | 0.215 | 0.116–0.324 |
| T-DM1 arm | 3 | 850 | 48 | 0.357 | 0.282 | 0.196–0.388 |
| T-DM1 arm | 4 | 850 | 48 | 0.431 | 0.466 | 0.353–0.600 |
| T-DM1 arm | 5 | 850 | 42 | 0.537 | 0.674 | 0.526–0.791 |
| T-DM1 arm | 6 | 850 | 32 | 0.761 | 0.781 | 0.592–0.926 |

</details>

## 5. Events per variable

The design caps signature size at five pCR events per selected variable. This table reports what was actually realised in each fold.

| Cohort | Model | Folds | Test-fold events (median, range) | Median signature size | Median EPV | Min EPV | Folds below EPV 5 |
|---|---|---|---|---|---|---|---|
| DHP arm | Clinical | 500 | 5 (4–5) | 4 | 10.00 | 10.00 | 0.0% |
| DHP arm | Genomic | 500 | 5 (4–5) | 4 | 9.75 | 5.57 | 0.0% |
| DHP arm | Integrated (late fusion) | 500 | 5 (4–5) | 3 | 6.33 | 3.80 | 33.4% |
| DHP arm | Proteomic | 500 | 5 (4–5) | 5 | 5.20 | 5.20 | 0.0% |
| DHP arm | Transcriptomic | 500 | 5 (4–5) | 5 | 7.80 | 5.71 | 0.0% |
| DHP arm | Whole-slide image | 500 | 5 (4–5) | 3 | 10.33 | 10.33 | 0.0% |
| Pooled cohort | Clinical | 1,000 | 9 (9–10) | 5 | 15.80 | 15.60 | 0.0% |
| Pooled cohort | Genomic | 1,000 | 9 (9–10) | 5 | 14.80 | 6.25 | 0.0% |
| Pooled cohort | Integrated (late fusion) | 1,000 | 9 (9–10) | 4 | 9.25 | 7.20 | 0.0% |
| Pooled cohort | Proteomic | 1,000 | 9 (9–10) | 7 | 7.43 | 6.50 | 0.0% |
| Pooled cohort | Transcriptomic | 1,000 | 9 (9–10) | 11 | 6.82 | 6.82 | 0.0% |
| Pooled cohort | Whole-slide image | 1,000 | 9 (9–10) | 3 | 20.67 | 20.33 | 0.0% |
| T-DM1 arm | Clinical | 500 | 4 (4–5) | 4 | 9.75 | 9.50 | 0.0% |
| T-DM1 arm | Genomic | 500 | 4 (4–5) | 4 | 8.75 | 5.83 | 0.0% |
| T-DM1 arm | Integrated (late fusion) | 500 | 4 (4–5) | 3 | 5.67 | 3.40 | 47.8% |
| T-DM1 arm | Proteomic | 500 | 4 (4–5) | 5 | 5.20 | 5.00 | 0.0% |
| T-DM1 arm | Transcriptomic | 500 | 4 (4–5) | 5 | 7.20 | 6.00 | 0.0% |
| T-DM1 arm | Whole-slide image | 500 | 4 (4–5) | 3 | 10.33 | 10.00 | 0.0% |

The arm-level fusion layer is the most exposed component: it takes five modality inputs by design and cannot be capped, so a third of DHP folds and almost half of T-DM1 folds run below five events per variable.

## 6. Feature-selection stability

How often each candidate feature was selected across the outer folds. Features above the pre-specified threshold (0.6 pooled, 0.5 per arm) are the consensus signature; 90 of 181 candidate rows clear it.

**The threshold is applied to the *eligible-fold* frequency** — the fraction of the folds in which the feature survived preprocessing and the in-fold screen at all. A feature can therefore be stable on that denominator while its all-fold frequency is low: it was rarely eligible, but was chosen almost whenever it was. Both columns are given below, with the Wilson interval on the eligible-fold proportion.

<details><summary><b>Pooled cohort</b> — 38 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_Arm` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.13` | 0.969 | 1.000 | 0.996–1.000 | 969 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.6` | 0.925 | 1.000 | 0.996–1.000 | 925 / 1,000 |
| Genomic | `DNA_PIK3CA_CNA` | 0.736 | 0.983 | 0.971–0.990 | 736 / 1,000 |
| Genomic | `DNA_meanHED` | 0.132 | 0.936 | 0.883–0.966 | 132 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.7` | 0.695 | 0.934 | 0.914–0.950 | 695 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.2` | 0.701 | 0.902 | 0.879–0.921 | 701 / 1,000 |
| Genomic | `DNA_TCRA.tcell.fraction.adj` | 0.217 | 0.865 | 0.817–0.901 | 217 / 1,000 |
| Genomic | `DNA_ERBB2_CNA` | 0.402 | 0.791 | 0.754–0.824 | 402 / 1,000 |
| Genomic | `DNA_CNV_burden` | 0.028 | 0.757 | 0.599–0.866 | 28 / 1,000 |
| Genomic | `DNA_BRCA2_CNA` | 0.090 | 0.750 | 0.666–0.819 | 90 / 1,000 |
| Genomic | `DNA_coding_mutation_TP53` | 0.434 | 0.729 | 0.692–0.764 | 434 / 1,000 |
| Genomic | `DNA_MED1_CNA` | 0.128 | 0.615 | 0.548–0.679 | 128 / 1,000 |
| Proteomic | `Prot_RPL19` | 0.977 | 0.977 | 0.966–0.985 | 977 / 1,000 |
| Proteomic | `Prot_ERBB2` | 0.972 | 0.972 | 0.960–0.981 | 972 / 1,000 |
| Proteomic | `Prot_VAMP3` | 0.843 | 0.862 | 0.839–0.882 | 843 / 1,000 |
| Proteomic | `Prot_CDK12` | 0.828 | 0.828 | 0.803–0.850 | 828 / 1,000 |
| Proteomic | `Prot_HER2_amplicon` | 0.692 | 0.692 | 0.663–0.720 | 692 / 1,000 |
| Proteomic | `Prot_SLC12A2` | 0.638 | 0.639 | 0.609–0.668 | 638 / 1,000 |
| Proteomic | `Prot_ERBB2_PG` | 0.619 | 0.619 | 0.588–0.649 | 619 / 1,000 |
| Transcriptomic | `RNA_HER2DX_HER2_amplicon` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Transcriptomic | `RNA_HER2DX_luminal` | 0.999 | 0.999 | 0.994–1.000 | 999 / 1,000 |
| Transcriptomic | `RNA_ADC_trafficking` | 0.998 | 0.998 | 0.993–0.999 | 998 / 1,000 |
| Transcriptomic | `RNA_mRNA-ESR1` | 0.997 | 0.997 | 0.991–0.999 | 997 / 1,000 |
| Transcriptomic | `RNA_FCGR3B` | 0.920 | 0.997 | 0.990–0.999 | 920 / 1,000 |
| Transcriptomic | `RNA_Exosome` | 0.980 | 0.980 | 0.969–0.987 | 980 / 1,000 |
| Transcriptomic | `RNA_Mast-cells` | 0.825 | 0.840 | 0.816–0.862 | 825 / 1,000 |
| Transcriptomic | `RNA_HER2DX_pCR_likelihood_score` | 0.821 | 0.821 | 0.796–0.844 | 821 / 1,000 |
| Transcriptomic | `RNA_mRNA-PGR` | 0.751 | 0.751 | 0.723–0.777 | 751 / 1,000 |
| Transcriptomic | `RNA_sspbc_LumB` | 0.723 | 0.723 | 0.694–0.750 | 723 / 1,000 |
| Transcriptomic | `RNA_B-cells` | 0.441 | 0.609 | 0.573–0.644 | 441 / 1,000 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |

</details>

<details><summary><b>DHP arm</b> — 29 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Genomic | `DNA_FADD_CNA` | 0.002 | 1.000 | 0.207–1.000 | 1 / 500 |
| Genomic | `DNA_ERBB2_CNA` | 0.970 | 0.970 | 0.951–0.982 | 485 / 500 |
| Genomic | `DNA_COSMIC.Signature.13` | 0.718 | 0.950 | 0.923–0.968 | 359 / 500 |
| Genomic | `DNA_COSMIC.Signature.6` | 0.500 | 0.874 | 0.831–0.908 | 250 / 500 |
| Genomic | `DNA_LOH_Del_burden` | 0.272 | 0.829 | 0.764–0.879 | 136 / 500 |
| Genomic | `DNA_RAB11FIP1_CNA` | 0.010 | 0.714 | 0.359–0.918 | 5 / 500 |
| Genomic | `DNA_coding_mutation_TP53` | 0.532 | 0.686 | 0.638–0.730 | 266 / 500 |
| Genomic | `DNA_meanHED` | 0.096 | 0.667 | 0.552–0.765 | 48 / 500 |
| Genomic | `DNA_coding_mutation_TP53_oncokb` | 0.478 | 0.592 | 0.543–0.638 | 239 / 500 |
| Genomic | `DNA_COSMIC.Signature.2` | 0.332 | 0.587 | 0.528–0.642 | 166 / 500 |
| Proteomic | `Prot_ERBB2_PG` | 0.988 | 0.988 | 0.974–0.994 | 494 / 500 |
| Proteomic | `Prot_MIEN1` | 0.888 | 0.888 | 0.857–0.913 | 444 / 500 |
| Proteomic | `Prot_HER2_amplicon` | 0.808 | 0.808 | 0.771–0.840 | 404 / 500 |
| Proteomic | `Prot_ERBB2` | 0.746 | 0.746 | 0.706–0.782 | 373 / 500 |
| Proteomic | `Prot_RPL19` | 0.732 | 0.733 | 0.693–0.770 | 366 / 500 |
| Proteomic | `Prot_CDK12` | 0.544 | 0.557 | 0.513–0.601 | 272 / 500 |
| Transcriptomic | `RNA_HER2DX_HER2_amplicon` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Transcriptomic | `RNA_pik3ca_sig` | 0.810 | 0.955 | 0.931–0.971 | 405 / 500 |
| Transcriptomic | `RNA_HER2DX_pCR_likelihood_score` | 0.896 | 0.896 | 0.866–0.920 | 448 / 500 |
| Transcriptomic | `RNA_mRNA-ESR1` | 0.818 | 0.895 | 0.863–0.920 | 409 / 500 |
| Transcriptomic | `RNA_HER2DX_luminal` | 0.698 | 0.739 | 0.698–0.777 | 349 / 500 |
| Transcriptomic | `RNA_sspbc_LumB` | 0.260 | 0.549 | 0.485–0.611 | 130 / 500 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |

</details>

<details><summary><b>T-DM1 arm</b> — 23 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Genomic | `DNA_PIK3CA_CNA` | 0.992 | 0.992 | 0.980–0.997 | 496 / 500 |
| Genomic | `DNA_NCOR1_CNA` | 0.882 | 0.938 | 0.913–0.957 | 441 / 500 |
| Genomic | `DNA_LOH_Del_burden` | 0.442 | 0.929 | 0.889–0.955 | 221 / 500 |
| Genomic | `DNA_BRCA2_CNA` | 0.814 | 0.839 | 0.804–0.869 | 407 / 500 |
| Genomic | `DNA_HLA_Supertype_A01` | 0.484 | 0.659 | 0.609–0.706 | 242 / 500 |
| Genomic | `DNA_CNV_burden` | 0.306 | 0.567 | 0.507–0.624 | 153 / 500 |
| Proteomic | `Prot_SLC12A2` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Proteomic | `Prot_CTTN` | 0.106 | 0.841 | 0.732–0.911 | 53 / 500 |
| Proteomic | `Prot_VAMP3` | 0.744 | 0.800 | 0.761–0.834 | 372 / 500 |
| Proteomic | `Prot_RPL19` | 0.544 | 0.756 | 0.709–0.797 | 272 / 500 |
| Proteomic | `Prot_FLOT1` | 0.552 | 0.633 | 0.587–0.677 | 276 / 500 |
| Transcriptomic | `RNA_FCGR3B` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Transcriptomic | `RNA_Exosome` | 0.928 | 0.928 | 0.902–0.948 | 464 / 500 |
| Transcriptomic | `RNA_ADC_trafficking` | 0.896 | 0.896 | 0.866–0.920 | 448 / 500 |
| Transcriptomic | `RNA_Mast-cells` | 0.844 | 0.844 | 0.810–0.873 | 422 / 500 |
| Transcriptomic | `RNA_mRNA-ESR1` | 0.812 | 0.812 | 0.775–0.844 | 406 / 500 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |

</details>

### Stability of the fusion weights

| Cohort | Modality | Mean weight | Median weight | Selection rate | 95% CI | Sign consistency |
|---|---|---|---|---|---|---|
| Pooled cohort | Transcriptomic | 2.47 | 2.45 | 100.0% | 1.00–1.00 | 1.00 |
| Pooled cohort | Proteomic | 2.03 | 1.97 | 96.9% | 0.96–0.98 | 1.00 |
| Pooled cohort | Whole-slide image | 1.59 | 1.24 | 74.8% | 0.72–0.77 | 1.00 |
| Pooled cohort | Clinical | 0.78 | 0.56 | 74.4% | 0.72–0.77 | 0.99 |
| Pooled cohort | Genomic | 0.71 | 0.00 | 50.3% | 0.47–0.53 | 1.00 |
| DHP arm | Transcriptomic | 1.09 | 0.95 | 98.2% | 0.97–0.99 | 1.00 |
| DHP arm | Proteomic | 1.75 | 1.52 | 96.2% | 0.94–0.98 | 1.00 |
| DHP arm | Genomic | 0.44 | 0.07 | 58.0% | 0.54–0.62 | 0.95 |
| DHP arm | Whole-slide image | 0.72 | 0.00 | 46.8% | 0.42–0.51 | 1.00 |
| DHP arm | Clinical | 0.17 | 0.00 | 22.4% | 0.19–0.26 | 0.92 |
| T-DM1 arm | Transcriptomic | 2.33 | 2.22 | 98.8% | 0.97–0.99 | 1.00 |
| T-DM1 arm | Proteomic | 1.33 | 1.13 | 82.4% | 0.79–0.85 | 1.00 |
| T-DM1 arm | Genomic | 1.54 | 1.30 | 81.6% | 0.78–0.85 | 1.00 |
| T-DM1 arm | Whole-slide image | 0.53 | 0.00 | 38.6% | 0.34–0.43 | 1.00 |
| T-DM1 arm | Clinical | 0.39 | 0.00 | 37.6% | 0.33–0.42 | 0.99 |

## 7. Consensus signatures and fusion weights

### Pooled cohort

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 5 | `SVM_Linear` | 55% | `Clin_ER`, `Clin_ANYNODES`, `Clin_TUMSIZE`, `Clin_prolifvalu`, `Clin_Arm` |
| Transcriptomic | 11 | `ExtraTrees` | 35% | `RNA_HER2DX_HER2_amplicon`, `RNA_mRNA-ESR1`, `RNA_ADC_trafficking`, `RNA_FCGR3B`, `RNA_Exosome`, `RNA_HER2DX_luminal`, `RNA_HER2DX_pCR_likelihood_score`, `RNA_mRNA-PGR`, `RNA_Mast-cells`, `RNA_sspbc_LumB`, `RNA_B-cells` |
| Genomic | 5 | `HistGradBoost` | 30% | `DNA_COSMIC.Signature.13`, `DNA_COSMIC.Signature.6`, `DNA_PIK3CA_CNA`, `DNA_COSMIC.Signature.7`, `DNA_COSMIC.Signature.2` |
| Proteomic | 7 | `SVM_Linear` | 34% | `Prot_RPL19`, `Prot_ERBB2`, `Prot_VAMP3`, `Prot_CDK12`, `Prot_HER2_amplicon`, `Prot_ERBB2_PG`, `Prot_SLC12A2` |
| Whole-slide image | 3 | `ElasticNet_LR` | 31% | `WSI_Cell_Interaction`, `WSI_Immune_Cell_prop`, `WSI_Distance_tumor_immune` |

### DHP arm

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 4 | `ElasticNet_LR` | 47% | `Clin_ER`, `Clin_ANYNODES`, `Clin_prolifvalu`, `Clin_TUMSIZE` |
| Transcriptomic | 5 | `ElasticNet_LR` | 57% | `RNA_HER2DX_HER2_amplicon`, `RNA_pik3ca_sig`, `RNA_mRNA-ESR1`, `RNA_HER2DX_pCR_likelihood_score`, `RNA_HER2DX_luminal` |
| Genomic | 4 | `SVM_Linear` | 25% | `DNA_ERBB2_CNA`, `DNA_COSMIC.Signature.13`, `DNA_coding_mutation_TP53`, `DNA_COSMIC.Signature.6` |
| Proteomic | 5 | `SVM_Linear` | 52% | `Prot_ERBB2_PG`, `Prot_MIEN1`, `Prot_HER2_amplicon`, `Prot_RPL19`, `Prot_ERBB2` |
| Whole-slide image | 3 | `RandomForest` | 54% | `WSI_Immune_Cell_prop`, `WSI_Distance_tumor_immune`, `WSI_Cell_Interaction` |

### T-DM1 arm

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 4 | `ElasticNet_LR` | 74% | `Clin_ER`, `Clin_ANYNODES`, `Clin_TUMSIZE`, `Clin_prolifvalu` |
| Transcriptomic | 5 | `SVM_Linear` | 32% | `RNA_FCGR3B`, `RNA_Exosome`, `RNA_ADC_trafficking`, `RNA_Mast-cells`, `RNA_mRNA-ESR1` |
| Genomic | 4 | `SVM_Linear` | 46% | `DNA_PIK3CA_CNA`, `DNA_NCOR1_CNA`, `DNA_BRCA2_CNA`, `DNA_HLA_Supertype_A01` |
| Proteomic | 5 | `SVM_Linear` | 34% | `Prot_SLC12A2`, `Prot_VAMP3`, `Prot_RPL19`, `Prot_FLOT1`, `Prot_GRB7` |
| Whole-slide image | 3 | `ElasticNet_LR` | 76% | `WSI_Cell_Interaction`, `WSI_Distance_tumor_immune`, `WSI_Immune_Cell_prop` |

### Late-fusion modality weights

| Cohort | Modality | Mean coefficient | SD | Selection rate |
|---|---|---|---|---|
| Pooled cohort | Transcriptomic | 2.39 | 0.70 | 100.0% |
| Pooled cohort | Proteomic | 1.96 | 0.92 | 97.1% |
| Pooled cohort | Whole-slide image | 1.51 | 1.42 | 68.7% |
| Pooled cohort | Clinical | 0.92 | 0.81 | 80.3% |
| Pooled cohort | Genomic | 0.47 | 0.84 | 36.4% |
| DHP arm | Proteomic | 1.76 | 0.96 | 98.4% |
| DHP arm | Transcriptomic | 1.15 | 0.59 | 99.4% |
| DHP arm | Whole-slide image | 0.70 | 1.07 | 46.2% |
| DHP arm | Genomic | 0.20 | 0.50 | 40.6% |
| DHP arm | Clinical | 0.05 | 0.26 | 13.2% |
| T-DM1 arm | Transcriptomic | 2.21 | 1.12 | 97.0% |
| T-DM1 arm | Proteomic | 1.14 | 1.10 | 73.2% |
| T-DM1 arm | Genomic | 0.90 | 1.19 | 52.6% |
| T-DM1 arm | Whole-slide image | 0.50 | 0.87 | 37.4% |
| T-DM1 arm | Clinical | 0.47 | 0.77 | 41.4% |

## 8. Exploratory biomarker groups

**Exploratory.** The group definition involves a cut-point that was searched, so the quotable P value is the permutation-corrected one, which accounts for that search.

| Step | Group | Rule (with the realised threshold) |
|---|---|---|
| 1 | S1 | `DNA_ERBB2_CNA >= 2.785 (q=0.5) AND DNA_CNV_burden >= 0.2151 (q=0.5)` |
| 2 | S3 | `NOT S1 AND RNA_ADC_trafficking >= -0.06558 (q=0.5) AND RNA_Th2 cells <= 0.07757 (q=0.5)` |
| 3 | S2 | `all remaining classifiable patients` |
| 4 | Unclassifiable | `any driver variable missing` |

| Group | Arm | n | pCR | pCR rate | 95% CI |
|---|---|---|---|---|---|
| S1 | Overall | 50 | 31 | 62.0% | 0.48–0.74 |
| S1 | 0.0 | 27 | 19 | 70.4% | 0.52–0.84 |
| S1 | 1.0 | 23 | 12 | 52.2% | 0.33–0.71 |
| S2 | Overall | 87 | 30 | 34.5% | 0.25–0.45 |
| S2 | 0.0 | 44 | 15 | 34.1% | 0.22–0.49 |
| S2 | 1.0 | 43 | 15 | 34.9% | 0.22–0.50 |
| S3 | Overall | 44 | 21 | 47.7% | 0.34–0.62 |
| S3 | 0.0 | 23 | 10 | 43.5% | 0.26–0.63 |
| S3 | 1.0 | 21 | 11 | 52.4% | 0.32–0.72 |

| Interaction test | Value |
|---|---|
| Test | treatment-by-group interaction (likelihood-ratio) |
| Adjusted for | Clin_ER |
| Likelihood-ratio statistic | 1.54 on 2 df |
| Nominal P at the median split | 0.464 |
| Cut-points searched | 13 (best 0.55, minimum P 0.035) |
| **Permutation-corrected P** | **0.254** (1,000 permutations) |
| Patients / events | 181 / 82 |

| Group | DHP pCR/n | T-DM1 pCR/n | OR (T-DM1 vs DHP) | 95% CI |
|---|---|---|---|---|
| S1 | 19/27 | 12/23 | 0.47 | 0.14–1.53 |
| S2 | 15/44 | 15/43 | 0.86 | 0.31–2.33 |
| S3 | 10/23 | 11/21 | 1.44 | 0.44–4.75 |

<details><summary>Cut-point sweep</summary>

| Quantile | Classified | S1 | S2 | S3 | LRT | Interaction P |
|---|---|---|---|---|---|---|
| 0.20 | 181 | 130 | 27 | 24 | 3.19 | 0.203 |
| 0.25 | 181 | 121 | 33 | 27 | 4.49 | 0.106 |
| 0.30 | 181 | 109 | 44 | 28 | 5.46 | 0.065 |
| 0.35 | 181 | 93 | 53 | 35 | 2.94 | 0.230 |
| 0.40 | 181 | 80 | 63 | 38 | 3.73 | 0.155 |
| 0.45 | 181 | 62 | 79 | 40 | 3.75 | 0.154 |
| 0.50 | 181 | 50 | 87 | 44 | 1.54 | 0.464 |
| 0.55 | 181 | 39 | 97 | 45 | 6.68 | 0.035 |
| 0.60 | 181 | 33 | 105 | 43 | 6.68 | 0.035 |
| 0.65 | 181 | 29 | 114 | 38 | 5.46 | 0.065 |
| 0.70 | 181 | 23 | 125 | 33 | 4.18 | 0.124 |
| 0.75 | 181 | 17 | 133 | 31 | 0.40 | 0.817 |
| 0.80 | 181 | 12 | 145 | 24 | 1.10 | 0.578 |

</details>

## 9. External validation

The pipeline's own transcriptomic consensus model was **frozen** — signature, classifier and hyper-parameters — refit once on the matching PREDIX arm with no grid search, and applied to the external cohort. Nothing was refitted on external data. Both harmonisation schemes are reported so that a result present under only one would be identified as an artefact of that scheme.

| Cohort | Locked model | Harmonisation | n | pCR | Internal AUROC | External AUROC | AUPRC | Brier | Calibration slope | P vs chance |
|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY2 | DHP | zscore | 44 | 26 | 0.748 [0.653–0.839] | **0.774 [0.622–0.904]** | 0.824 [0.715–0.932] | 0.1964 [0.1448–0.2491] | 1.07 (0.59–1.81) | 0.001 |
| I-SPY2 | DHP | rank | 44 | 26 | 0.748 [0.653–0.839] | **0.793 [0.645–0.915]** | 0.837 [0.736–0.939] | 0.1958 [0.1467–0.2441] | 1.21 (0.70–2.05) | < 0.001 |
| NCT02326974 | T-DM1 | zscore | 129 | 64 | 0.787 [0.695–0.875] | **0.546 [0.444–0.641]** | 0.593 [0.510–0.693] | 0.2835 [0.2443–0.3219] | 0.24 (-0.05–0.51) | 0.167 |
| NCT02326974 | T-DM1 | rank | 129 | 64 | 0.787 [0.695–0.875] | **0.545 [0.442–0.641]** | 0.591 [0.507–0.696] | 0.2764 [0.2393–0.3127] | 0.25 (-0.06–0.57) | 0.171 |

Locked specifications:

| External cohort | PREDIX arm | Frozen classifier | Features | Refit on (n / events) |
|---|---|---|---|---|
| I-SPY2 (GSE194040) — trastuzumab/pertuzumab + chemotherapy | DHP | `ElasticNet_LR` {'C': 0.1} | 5 | 95 / 44 |
| NCT02326974 (GSE243375) — T-DM1 + pertuzumab | T-DM1 | `SVM_Linear` {'C': 0.1} | 5 | 90 / 40 |

The DHP transcriptomic model transfers to I-SPY2 with its calibration intact. The T-DM1 model does not transfer: its external discrimination is close to chance and its calibration slope far below 1, meaning the probabilities it produces are far too extreme for that cohort. Both results are reported as they stand.

## 10. Figures

PNG renderings at 150 dpi; the citable vector versions are the PDFs in [`report/figures/`](report/figures).

### Main figures

#### fig01_consensus_performance

![fig01_consensus_performance](report/figures_png/fig01_consensus_performance.png)

*Cross-validated AUROC of every consensus model with its 95% patient-level cluster-bootstrap interval, in the pooled cohort and each arm.*

#### fig02_consensus_signatures

![fig02_consensus_signatures](report/figures_png/fig02_consensus_signatures.png)

*The frozen consensus signature of each modality and scenario: mean absolute SHAP importance per feature, coloured by the direction of the association, with the winning classifier family above each panel.*

#### fig03_consensus_roc

![fig03_consensus_roc](report/figures_png/fig03_consensus_roc.png)

*Out-of-fold ROC curves of the integrated model and of the best single modality, drawn on all pooled (patient, repeat) predictions.*

#### fig04_consensus_modality_weights

![fig04_consensus_modality_weights](report/figures_png/fig04_consensus_modality_weights.png)

*Late-fusion modality weights of the consensus models: mean elastic-net coefficient and the fraction of folds in which each modality received a non-zero weight.*

#### fig05_consensus_feature_shap_Global

![fig05_consensus_feature_shap_Global](report/figures_png/fig05_consensus_feature_shap_Global.png)

*Feature-level SHAP attribution for the pooled-cohort consensus models, restricted to the consensus signature.*

#### fig05_consensus_feature_shap_DHP

![fig05_consensus_feature_shap_DHP](report/figures_png/fig05_consensus_feature_shap_DHP.png)

*Feature-level SHAP attribution for the DHP-arm consensus models.*

#### fig05_consensus_feature_shap_T_DM1

![fig05_consensus_feature_shap_T_DM1](report/figures_png/fig05_consensus_feature_shap_T_DM1.png)

*Feature-level SHAP attribution for the T-DM1-arm consensus models.*

#### fig06_counterfactual_summary

![fig06_counterfactual_summary](report/figures_png/fig06_counterfactual_summary.png)

*Counterfactual summary: predicted response under each treatment assignment.*

### Revision figures

Calibration, stability, events per variable, biomarker groups, external validation, paired comparisons and fusion weights — the diagnostics added in this revision.

#### revfig01_calibration

![revfig01_calibration](report/figures_png/revfig01_calibration.png)

*Calibration of the consensus integrated model: reliability curves over ten equal-count bins of all out-of-fold predictions, with patient-level cluster-bootstrap intervals, and the slope, intercept and Brier score of each scenario.*

#### revfig02_selection_stability

![revfig02_selection_stability](report/figures_png/revfig02_selection_stability.png)

*Feature-selection frequency across the outer folds, with Wilson intervals and the pre-specified stability threshold (0.60 pooled, 0.50 per arm).*

#### revfig03_epv_per_fold

![revfig03_epv_per_fold](report/figures_png/revfig03_epv_per_fold.png)

*Per-fold pCR event counts and realised events-per-variable for every model.*

#### revfig04_biomarker_groups

![revfig04_biomarker_groups](report/figures_png/revfig04_biomarker_groups.png)

*Exploratory biomarker groups S1–S3: group sizes and pCR rates by arm, the interaction P value across the cut-point sweep, and the permutation null.*

#### revfig06_external_validation

![revfig06_external_validation](report/figures_png/revfig06_external_validation.png)

*Locked-model external validation: ROC and precision–recall curves and reliability of the frozen DHP and T-DM1 transcriptomic models in I-SPY2 and NCT02326974.*

#### revfig07_model_comparisons

![revfig07_model_comparisons](report/figures_png/revfig07_model_comparisons.png)

*AUROC forest and paired ΔAUROC of the integrated model against every single-modality comparator, with 95% paired cluster-bootstrap intervals.*

#### revfig08_fusion_weights

![revfig08_fusion_weights](report/figures_png/revfig08_fusion_weights.png)

*Fold-wise distribution of the late-fusion modality weights and each modality's selection rate.*

### Supplementary figures — discovery phase

Diagnostics of the fully nested discovery phase, before consensus finalisation.

#### supp_fig01_roc_curves

![supp_fig01_roc_curves](report/figures_png/supp_fig01_roc_curves.png)

*Discovery-phase ROC curves.*

#### supp_fig02_performance_distributions

![supp_fig02_performance_distributions](report/figures_png/supp_fig02_performance_distributions.png)

*Discovery-phase distribution of per-fold performance for every model.*

#### supp_fig03_fusion_benefit

![supp_fig03_fusion_benefit](report/figures_png/supp_fig03_fusion_benefit.png)

*Discovery-phase fusion benefit against the best single modality.*

#### supp_fig04_forest_plot

![supp_fig04_forest_plot](report/figures_png/supp_fig04_forest_plot.png)

*Discovery-phase forest plot of per-fold AUROC.*

#### supp_fig05_feature_shap_Global

![supp_fig05_feature_shap_Global](report/figures_png/supp_fig05_feature_shap_Global.png)

*Discovery-phase SHAP attribution, pooled cohort.*

#### supp_fig05_feature_shap_DHP

![supp_fig05_feature_shap_DHP](report/figures_png/supp_fig05_feature_shap_DHP.png)

*Discovery-phase SHAP attribution, DHP arm.*

#### supp_fig05_feature_shap_T_DM1

![supp_fig05_feature_shap_T_DM1](report/figures_png/supp_fig05_feature_shap_T_DM1.png)

*Discovery-phase SHAP attribution, T-DM1 arm.*

#### supp_fig06_feature_selection_frequency

![supp_fig06_feature_selection_frequency](report/figures_png/supp_fig06_feature_selection_frequency.png)

*Discovery-phase selection frequency of every candidate feature.*

#### supp_fig07_cross_scenario_features

![supp_fig07_cross_scenario_features](report/figures_png/supp_fig07_cross_scenario_features.png)

*Features shared between the pooled and arm-specific signatures.*

#### supp_fig08_fusion_shap

![supp_fig08_fusion_shap](report/figures_png/supp_fig08_fusion_shap.png)

*SHAP attribution of the five modality streams inside the fusion layer.*

#### supp_fig09_modality_weights

![supp_fig09_modality_weights](report/figures_png/supp_fig09_modality_weights.png)

*Discovery-phase modality weights.*

#### supp_fig10_winner_classifier_heatmap

![supp_fig10_winner_classifier_heatmap](report/figures_png/supp_fig10_winner_classifier_heatmap.png)

*Which classifier family won each fold, by modality and scenario.*

#### supp_fig11_inner_auroc_comparison

![supp_fig11_inner_auroc_comparison](report/figures_png/supp_fig11_inner_auroc_comparison.png)

*Inner-cross-validation AUROC of each classifier family, the basis of the Stage A choice.*

#### supp_fig12_calibration_profile

![supp_fig12_calibration_profile](report/figures_png/supp_fig12_calibration_profile.png)

*Discovery-phase calibration profile.*

#### supp_fig13_signature_sizes

![supp_fig13_signature_sizes](report/figures_png/supp_fig13_signature_sizes.png)

*Distribution of discovered signature sizes across folds.*

#### supp_fig14_performance_CI

![supp_fig14_performance_CI](report/figures_png/supp_fig14_performance_CI.png)

*Discovery-phase AUROC with patient-level cluster-bootstrap intervals — the fully nested estimates, free of consensus selection optimism.*

---

Regenerate this page with `python docs/build_RESULTS_md.py`.
