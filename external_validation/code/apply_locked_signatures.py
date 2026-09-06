#!/usr/bin/env python3
"""The two primitives the locked-signature application shares.

WHAT THIS IS
------------
A module, not a script: it is imported, never run. It holds the two pieces of
the external-validation procedure that more than one caller needs, so that they
exist once and cannot drift into two versions that disagree.

  encode_external(path)
      Read an external cohort's curated metric file and put it in the frame the
      locked model expects - the pipeline's own one-hot encoding of the sspbc
      subtype call, and the sample-identifier column under the pipeline's name.
      A cohort that carries the raw subtype column therefore carries the derived
      dummy columns too, which is what decides whether a signature feature counts
      as measured there.

  p_vs_chance(y, p)
      One-sided bootstrap tail probability that the external AUROC is at or
      below 0.5. Stratified resampling of cases and controls, 2000 draws, and
      the (hits + 1) / (draws + 1) correction - so the smallest value it can
      return is 1/2001, about 0.0005. A row at that value means no draw fell at
      or below chance; it is a resolution floor, not a measured probability, and
      the reporting layer renders it as an inequality for that reason.

WHERE THE ANALYSIS ITSELF LIVES
-------------------------------
`apply_locked_external_validation.py` is the driver. It reads the frozen
signature, classifier and hyperparameters from the locked `results/{dhp,tdm1}`,
fits that specification ONCE on the corresponding PREDIX arm cohort with no
grid search and no cross-validation, standardises within each cohort by z-score,
applies the model once, and writes both the workbook and
`locked_signatures_external_validation.csv`. Nothing is refitted on external
data and no feature is dropped to make a model fit: a signature with features
removed is a different model.

`make_fig_external_validation_all.py` and `feature_portability.py` read what the
driver wrote. Neither recomputes a reported number.
"""
import numpy as np
import pandas as pd

N_BOOT, SEED = 2000, 42


def encode_external(path):
    """Load an external frame and apply the pipeline's sspbc one-hot encoding."""
    df = pd.read_csv(path, sep="\t")
    if "Donor.ID" in df.columns and "sampleID" not in df.columns:
        df = df.rename(columns={"Donor.ID": "sampleID"})
    if "RNA_sspbc.subtype" in df.columns:
        d = pd.get_dummies(df["RNA_sspbc.subtype"], prefix="RNA_sspbc")
        d = d.drop(columns=["RNA_sspbc_Her2"], errors="ignore")
        df = pd.concat([df.drop(columns=["RNA_sspbc.subtype"]),
                        d.astype(float)], axis=1)
    return df


def p_vs_chance(y_ext, p_ext, n_boot=N_BOOT, seed=SEED):
    """One-sided bootstrap tail probability that the external AUROC is <= 0.5.

    The same stratified resampling, rng seed offset and (hits + 1) / (draws + 1)
    correction as the pipeline's own external-validation module, so this column
    is comparable with the p_vs_chance_one_sided the pipeline reports.

    Draws in which the resampled labels are single-class carry no AUROC and are
    discarded, so the denominator is the number of usable draws rather than
    n_boot.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed + 5)
    idx_pos = np.where(y_ext == 1)[0]
    idx_neg = np.where(y_ext == 0)[0]
    boots = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(idx_pos, size=len(idx_pos), replace=True),
                               rng.choice(idx_neg, size=len(idx_neg), replace=True)])
        if len(np.unique(y_ext[take])) < 2:
            continue
        boots.append(roc_auc_score(y_ext[take], p_ext[take]))
    boots = np.asarray(boots)
    return (float((np.sum(boots <= 0.5) + 1) / (len(boots) + 1))
            if len(boots) else np.nan)
