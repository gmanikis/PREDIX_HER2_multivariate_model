#!/usr/bin/env python3
"""Do the signature features mean the same thing in the external cohorts?

DIAGNOSTIC ONLY. Nothing is fitted, nothing is selected, no model is touched.
For each feature of each locked signature this measures, separately in PREDIX
and in the external cohort:

THE PREDIX SIDE IS THE COMPLETE-CASE COHORT of the matching arm (59 DHP, 51
T-DM1) - the population the locked models were fitted on, and the population
every other document in this package reports. Measuring it on the full arm
instead would compare a feature's behaviour in one set of patients against a
model fitted in another, and would put a different denominator in this file from
the one beside it.

  * its DIRECTIONAL univariate association with pCR (AUROC of the raw feature;
    0.5 = none, <0.5 = higher value means less pCR), and
  * its Spearman correlation with mRNA ESR1 within that cohort.

ESR1 is the anchor because it is a single transcript with an unambiguous
measurement — no deconvolution, no composite definition — so it should sit in
the same place in the correlation structure of any RNA-seq cohort. A feature
that correlates with ESR1 quite differently in two cohorts is not measuring the
same quantity in both, whatever its name.

WHY THIS IS NOT POST-HOC ANALYSIS. It asks why an already-reported result came
out as it did; it does not change the model, the signature or any reported
number, and it is reportable whichever way it falls. The three signatures are
examined together — including the two that DO transfer — so the comparison is
not built only around the null.

USAGE
  python feature_portability.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# This is a delivery folder, not a self-contained package: the pipeline module
# (used only for load_and_encode_data) and the curated input matrices live in
# one shared tree rather than being duplicated per folder. The matrix is checked
# against the locked analysis's own provenance record below rather than assumed.
ASSETS = next(
    (d / "7_external_validation_transneo"
     for d in sorted(ROOT.glob("revision_delivery*"))
     if (d / "7_external_validation_transneo" / "data").is_dir()),
    ROOT / "assets")
sys.path.insert(0, str(ASSETS / "code"))
sys.path.insert(0, str(ROOT / "revision_deliverables" / "3_reproducibility_package" / "code"))
from multimodal_pcr_pipeline import load_and_encode_data  # noqa: E402

# Resolved for both layouts this file lives in. In the deposit the artefacts sit
# at `results/` beside this script's grandparent; in the working tree they sit in
# the run directory. Naming the run directory outright made the deposited copy
# point at a path no reader has, and published the run's name with it.
# NAMED, NOT DISCOVERED. A results directory picked up by pattern can be any
# tree that happens to sit beside the right one, and a portability table
# computed on the wrong tree's signatures would look entirely plausible. The
# production run is named in exactly one place; in the deposit layout the
# artefacts sit at results/.
if (ROOT / "results" / "run_provenance.json").is_file():
    RUN = ROOT                                         # deposit layout
else:
    sys.path.insert(0, str(ROOT / "revision_deliverables"))
    from locked_run import require as _require_locked  # noqa: E402
    RUN = _require_locked()                            # working tree
PREDIX = ASSETS / "data" / "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"

import hashlib as _hl                                       # noqa: E402
import json as _json                                        # noqa: E402
_want = _json.loads((RUN / "results" / "run_provenance.json")
                    .read_text(encoding="utf-8-sig"))["input_data"]["sha256"]
_got = _hl.sha256(PREDIX.read_bytes()).hexdigest()
if _want != _got:
    sys.exit("input matrix does not match the locked analysis's provenance:\n"
             f"  provenance {_want}\n  file       {_got}")
ANCHOR = "RNA_mRNA-ESR1"

# signature -> (experiment, modality, PREDIX arm code, [(cohort, file), ...])
CASES = [
    ("DHP transcriptomic", "dhp", "RNA", 0, [
        ("I-SPY2", ROOT / "RNA_curated_metrics_ISPY2.txt"),
        ("TransNEO", ASSETS / "data" / "RNA_curated_metrics_TransNEO_rawcounts.txt")]),
    ("T-DM1 transcriptomic", "tdm1", "RNA", 1, [
        ("NCT02326974", ROOT / "RNA_curated_metrics_NCT02326974.txt")]),
    ("DHP genomic", "dhp", "DNA", 0, [
        ("TransNEO", ASSETS / "data" / "DNA_curated_metrics_TransNEO.txt")]),
]


def signature(exp, modality):
    with open(RUN / "results" / exp / f"{exp}_consensus_eval.pkl", "rb") as f:
        blob = pickle.load(f)
    return list(blob["consensus"][modality]["signature"])


def complete_case(df):
    """The pipeline's rule: complete on the four molecular blocks. Clin never
    enters - clinical covariates are imputed in-fold. This is the population the
    locked models were fitted and evaluated on."""
    mol = [c for c in df.columns
           if c.split("_", 1)[0] in ("RNA", "DNA", "Prot", "WSI")]
    return df.dropna(subset=mol)


def dir_auroc(x, y):
    """Directional univariate AUROC; NaN if the feature is constant."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or len(np.unique(x[ok])) < 2 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], x[ok]))


def anchor_rho(df, feat):
    if ANCHOR not in df.columns or feat == ANCHOR or feat not in df.columns:
        return np.nan
    a = pd.to_numeric(df[ANCHOR], errors="coerce")
    b = pd.to_numeric(df[feat], errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 10 or b[ok].nunique() < 2:
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def main():
    predix = load_and_encode_data(PREDIX)
    if isinstance(predix, tuple):
        predix = predix[0]
    predix = complete_case(predix)
    print(f"PREDIX side: complete-case cohort, {len(predix)} patients, "
          f"{int(predix['pCR'].sum())} events")
    rows = []
    for label, exp, modality, arm, cohorts in CASES:
        sig = signature(exp, modality)
        dp = predix[predix["Clin_Arm"] == arm]
        y_int = dp["pCR"].astype(float).values
        for cname, cpath in cohorts:
            ext = pd.read_csv(cpath, sep="\t")
            if "Donor.ID" in ext.columns:
                ext = ext.rename(columns={"Donor.ID": "sampleID"})
            y_ext = ext["pCR"].astype(float).values
            for f in sig:
                a_int = dir_auroc(dp[f].values, y_int)
                a_ext = dir_auroc(ext[f].values, y_ext) if f in ext.columns else np.nan
                r_int, r_ext = anchor_rho(dp, f), anchor_rho(ext, f)
                rows.append({
                    "signature": label, "external_cohort": cname, "feature": f,
                    # The population is carried in the file, not only in the
                    # prose: "in PREDIX" alone does not say which patients.
                    "PREDIX_population": "complete-case arm cohort",
                    "n_PREDIX": int(len(dp)), "n_external": int(len(ext)),
                    "AUROC_PREDIX": round(a_int, 3),
                    "AUROC_external": round(a_ext, 3) if a_ext == a_ext else np.nan,
                    "delta_from_0.5_PREDIX": round(a_int - 0.5, 3),
                    "delta_from_0.5_external": (round(a_ext - 0.5, 3)
                                                if a_ext == a_ext else np.nan),
                    "direction_flips": (bool((a_int - 0.5) * (a_ext - 0.5) < 0)
                                        if a_ext == a_ext else None),
                    "rho_with_ESR1_PREDIX": (round(r_int, 3) if r_int == r_int
                                             else np.nan),
                    "rho_with_ESR1_external": (round(r_ext, 3) if r_ext == r_ext
                                               else np.nan),
                    "delta_rho": (round(abs(r_int - r_ext), 3)
                                  if r_int == r_int and r_ext == r_ext else np.nan),
                })

    d = pd.DataFrame(rows)
    out = HERE / "feature_portability.csv"
    d.to_csv(out, index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    for label in d["signature"].unique():
        for coh in d[d["signature"] == label]["external_cohort"].unique():
            sub = d[(d["signature"] == label) & (d["external_cohort"] == coh)]
            print("\n" + "=" * 96)
            print(f"{label}  ->  {coh}")
            print("=" * 96)
            print(sub[["feature", "AUROC_PREDIX", "AUROC_external",
                       "direction_flips", "rho_with_ESR1_PREDIX",
                       "rho_with_ESR1_external", "delta_rho"]]
                  .to_string(index=False))
            n_flip = int(sub["direction_flips"].fillna(False).sum())
            big = sub["delta_rho"].dropna()
            print(f"  direction flips: {n_flip} of {len(sub)}"
                  + (f" | max |Δρ with ESR1| = {big.max():.2f}"
                     if len(big) else ""))
    print(f"\n[WRITE] {out}")


if __name__ == "__main__":
    main()
