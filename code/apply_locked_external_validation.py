#!/usr/bin/env python
"""Apply the locked signatures to the external cohorts. One procedure, applied once.

WHAT THIS PRODUCES
------------------
Two artefacts, both under `<run>/report/tables/revision/`:
`external_validation.xlsx` (three sheets) and
`locked_signatures_external_validation.csv`. Both are read downstream - by the
results page, by the verification notebook and by
`make_fig_external_validation_all.py` - so they are written in ONE pass from ONE
set of row objects and cannot disagree about a number.

THE PROCEDURE
-------------
Direct application of the locked model. Take the frozen signature, classifier and
hyperparameters from `<run>/results/{dhp,tdm1}`; fit that specification ONCE on
the corresponding PREDIX cohort - the complete-case patients of the matching arm,
which is the population the model was trained on - with no grid search and no
cross-validation; standardise within each cohort by z-score; apply once. Nothing
is refitted on external data, and no external patient influences feature
selection, hyper-parameters or any threshold.

WHICH EVALUATIONS ARE ATTEMPTED, AND WHICH CAN BE SCORED
--------------------------------------------------------
Four evaluations are attempted, each arm-matched to the regimen the cohort
received:

    I-SPY2       <- the DHP transcriptomic model
    TransNEO     <- the DHP transcriptomic model
    NCT02326974  <- the T-DM1 transcriptomic model
    TransNEO     <- the DHP genomic model

A model is scored in a cohort ONLY IF EVERY FEATURE OF ITS SIGNATURE IS MEASURED
THERE. A signature with features removed is a different model, so an evaluation
whose cohort lacks a signature member is not scored on what remains: it is
recorded - in the Feature_provenance sheet, and in the CSV with
`applicable = False` and the missing features named - rather than dropped.
Which evaluations that rule admits is decided by the locked signatures and the
cohort files at run time, and is reported, not assumed here.

Only the z-score harmonisation is reported.

Usage:
    python apply_locked_external_validation.py --run <dir holding results/ and report/>
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EV = HERE
# The shared pipeline modules and the curated input matrices live in one tree and
# are not duplicated per delivery folder.
ASSETS = HERE          # self-contained: every input sits beside this script
CODE = HERE

sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(CODE))
# Inserted last, so it is searched FIRST: the helpers that ship beside the
# results are the ones that run, and a stale sibling elsewhere on the path
# cannot quietly substitute itself.
sys.path.insert(0, str(EV))

from external_validation import build_model, standardise_within_cohort  # noqa: E402
from revision_analyses import bootstrap_metric_ci, calibration_metrics  # noqa: E402
from multimodal_pcr_pipeline import load_and_encode_data  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # noqa: E402
from apply_locked_signatures import encode_external, p_vs_chance  # noqa: E402

PREDIX = HERE / "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
N_BOOT, SEED = 2000, 42
# p_vs_chance returns (draws at or below chance + 1) / (draws + 1), so this is
# the smallest value it can return. A row sitting here means no draw fell at or
# below chance - it is a resolution floor, not a measurement.
P_FLOOR = 1.0 / (N_BOOT + 1)

SETS = [
    ("I-SPY2", HERE / "RNA_curated_metrics_ISPY2.txt", "dhp", 0, "RNA",
     "GSE194040, trastuzumab/pertuzumab + chemotherapy"),
    ("TransNEO", HERE / "RNA_curated_metrics_TransNEO_rawcounts.txt",
     "dhp", 0, "RNA", "chemotherapy + HER2-targeted therapy (DHP-like)"),
    ("NCT02326974", HERE / "RNA_curated_metrics_NCT02326974.txt", "tdm1", 1, "RNA",
     "GSE243375, T-DM1 + pertuzumab"),
]
# Checked and reported as NOT applicable, not silently dropped.
NOT_APPLICABLE = [
    ("TransNEO", HERE / "DNA_curated_metrics_TransNEO.txt", "dhp", 0,
     "DNA", "chemotherapy + HER2-targeted therapy (DHP-like)"),
]
ARM_LABEL = {"dhp": "DHP", "tdm1": "T-DM1"}


def format_p(p):
    """Render the one-sided bootstrap p for a figure or a table.

    A value at P_FLOOR is not an estimate of the tail probability, so it is
    rendered as an inequality rather than as a point. The thresholds match the
    ones `build_RESULTS_md.py` applies, so every artefact - both figures, the
    CSV and the rendered results page - shows the same string for the same row.
    """
    if p is None or p != p:
        return "P = not estimable"
    p = float(p)
    return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"


def spec(run, exp, modality):
    with open(run / "results" / exp / f"{exp}_consensus_eval.pkl", "rb") as f:
        blob = pickle.load(f)
    c = blob["consensus"][modality]
    return list(c["signature"]), c["winner_clf"], (c.get("params") or {})


def internal_auroc(run, exp, modality):
    """The locked model's own internal consensus AUROC, from its workbook."""
    p = run / "report" / "tables" / "revision" / "revision_performance_CI.xlsx"
    raw = pd.read_excel(p, sheet_name="Performance_patient_CI", header=None)
    hdr = next(i for i in range(len(raw))
               if str(raw.iat[i, 0]).strip().lower() == "scenario")
    df = pd.read_excel(p, sheet_name="Performance_patient_CI", header=hdr)
    scen = {"dhp": "DHP", "tdm1": "T-DM1", "global": "Global"}[exp]
    r = df[(df["scenario"].astype(str).str.strip() == scen)
           & (df["source"].astype(str).str.strip() == "consensus")
           & (df["model"].astype(str).str.strip() == modality)]
    if r.empty:
        return None, None, None, None
    r = r.iloc[0]
    return (float(r["AUROC"]), str(r["AUROC_formatted"]),
            int(r["n_patients"]), int(r["n_events"]))


def complete_case(df):
    """The pipeline's rule: complete on the four molecular blocks. Clin never
    enters - clinical covariates are imputed in-fold."""
    mol = [c for c in df.columns
           if c.split("_", 1)[0] in ("RNA", "DNA", "Prot", "WSI")]
    return df.dropna(subset=mol)


def run_one(run, label, path, exp, arm_code, modality, note, cc):
    sig, clf, params = spec(run, exp, modality)
    ext = encode_external(path)
    missing = [f for f in sig if f not in ext.columns]
    present = [f for f in sig if f in ext.columns]
    prov = [{"cohort": label, "feature": f,
             "status": "present" if f in present else "ABSENT",
             "reason": ("measured in this cohort" if f in present else
                        "not measured in this cohort; the model cannot be "
                        "scored without it")}
            for f in sig]

    print("-" * 78)
    print(f"{label} | {modality} | locked {ARM_LABEL[exp]} model | {clf} | K={len(sig)}")
    if missing:
        print(f"  NOT APPLICABLE - {len(missing)}/{len(sig)} features absent: {missing}")
        return None, prov, None

    dp = cc[cc["Clin_Arm"] == arm_code].dropna(subset=sig)
    y_int = dp["pCR"].astype(float).values
    X_int = np.nan_to_num(standardise_within_cohort(
        dp[sig].apply(pd.to_numeric, errors="coerce").values, "zscore"),
        nan=0.0, posinf=0.0, neginf=0.0)
    y_ext = ext["pCR"].astype(float).values
    X_ext = np.nan_to_num(standardise_within_cohort(
        ext[sig].apply(pd.to_numeric, errors="coerce").values, "zscore"),
        nan=0.0, posinf=0.0, neginf=0.0)

    model = build_model(clf, params).fit(X_int, y_int)
    p_ext = model.predict_proba(X_ext)[:, 1]
    # The pipeline's internal probabilities are Platt-calibrated, so raw external
    # probabilities would describe a pipeline the locked model never uses. If the
    # layer cannot be fitted the raw probabilities are used instead - and WHICH
    # was used is recorded in the row, because a reader cannot otherwise tell
    # whether a Brier score or a calibration slope came from calibrated or raw
    # probabilities. AUROC and AUPRC are unaffected either way: Platt scaling is
    # monotone, so it cannot reorder patients.
    try:
        p_cv = cross_val_predict(
            build_model(clf, params), X_int, y_int,
            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            method="predict_proba")[:, 1]
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(
            p_cv.reshape(-1, 1), y_int)
        p_cal = platt.predict_proba(p_ext.reshape(-1, 1))[:, 1]
        calibration = "Platt, fitted on 5-fold CV predictions in the PREDIX fit cohort"
    except Exception as e:                                    # pragma: no cover
        print(f"  Platt layer skipped ({e})")
        p_cal = p_ext
        calibration = (f"none - raw model probabilities "
                       f"(Platt layer unavailable: {type(e).__name__})")

    au = bootstrap_metric_ci(y_ext, p_cal, "AUROC", n_boot=N_BOOT, seed=SEED + 2)
    ap = bootstrap_metric_ci(y_ext, p_cal, "AUPRC", n_boot=N_BOOT, seed=SEED + 3)
    br = bootstrap_metric_ci(y_ext, p_cal, "Brier", n_boot=N_BOOT, seed=SEED + 4)
    cal = calibration_metrics(y_ext, p_cal, n_boot=N_BOOT, seed=SEED + 5)
    pch = p_vs_chance(y_ext, p_cal)
    ia, ifmt, i_n, i_ev = internal_auroc(run, exp, modality)

    def ci(d, lo="ci_low", hi="ci_high", dp=3):
        """Format as "estimate [lo-hi]", which is what the schema means.

        These `*_CI` columns are rendered directly into RESULTS.md under headings
        like "External AUROC". Bare bounds in this column put an interval where
        the point estimate belongs and leave the estimate off the page - a
        well-formed string in the right column, so no checker can see it.
        """
        v = d.get(lo), d.get(hi)
        if not all(x is not None and x == x for x in v):
            return "not estimable"
        return (f"{float(d['estimate']):.{dp}f} "
                f"[{float(v[0]):.{dp}f}–{float(v[1]):.{dp}f}]")

    def pair(key):
        v = cal.get(key + "_ci")
        return (f"{float(v[0]):.2f}-{float(v[1]):.2f}"
                if v is not None and all(x == x for x in v) else "not estimable")

    row = {
        "cohort": label, "cohort_description": note,
        "cohort_resembles_PREDIX_arm": ARM_LABEL[exp],
        "model_refit_population": f"PREDIX {ARM_LABEL[exp]} arm, complete cases only",
        "arm_matched": True, "harmonisation": "zscore",
        "probability_calibration": calibration,
        "model_source": "locked consensus (complete-case training)",
        "locked_from": "results/{dhp,tdm1} of the locked analysis",
        "classifier": clf, "hyperparameters": str(params),
        "n_model_features": len(sig),
        "n_PREDIX_train": len(dp), "events_PREDIX_train": int(y_int.sum()),
        "internal_harmonisation": "within-fold standardisation",
        "internal_AUROC": (round(ia, 6) if ia is not None else ""),
        "internal_AUROC_CI": (ifmt or ""),
        "n_external": len(y_ext), "events_external": int(y_ext.sum()),
        "external_AUROC": round(float(au["estimate"]), 6),
        "external_AUROC_CI": ci(au),
        "external_AUPRC": round(float(ap["estimate"]), 6),
        "external_AUPRC_CI": ci(ap),
        "external_event_rate": round(float(y_ext.mean()), 6),
        "external_Brier": round(float(br["estimate"]), 6),
        # Brier is quoted at four decimals in this schema, unlike AUROC/AUPRC.
        "external_Brier_CI": ci(br, dp=4),
        "calibration_slope": round(float(cal["slope"]), 6),
        "calibration_slope_CI": pair("slope"),
        "calibration_intercept": round(float(cal["intercept"]), 6),
        "calibration_intercept_CI": pair("intercept"),
        "AUROC_drop_internal_to_external": (
            round(float(ia) - float(au["estimate"]), 6) if ia is not None else ""),
        "p_vs_chance_one_sided": round(float(pch), 6),
        "p_vs_chance_display": format_p(pch),
        "p_vs_chance_at_resolution_floor": bool(float(pch) <= P_FLOOR + 1e-12),
    }
    print(f"  fit on {len(dp)} PREDIX patients ({int(y_int.sum())} events); "
          f"external n={len(y_ext)}, {int(y_ext.sum())} events")
    print(f"  AUROC {row['external_AUROC_CI']}  {row['p_vs_chance_display']}  "
          f"slope {row['calibration_slope']:.2f}  [{calibration}]")

    return row, prov, reliability(label, y_ext, p_cal)


def reliability(label, y, p, n_bins=10):
    """Observed pCR rate by bin of predicted probability, with Wilson intervals.

    calibration_metrics() does not return the bin table, so it is computed here
    rather than left as a stub sheet - a sheet whose first column is not `cohort`
    breaks the deposit's own header locator, which searches for that literal.

    Bins are equal-width on [0,1] and EMPTY BINS ARE DROPPED: a bin with n = 0
    has no observed rate, and carrying it as a NaN row invites a reader to treat
    the gap as a measurement.
    """
    import numpy as _np
    y = _np.asarray(y, dtype=float)
    p = _np.asarray(p, dtype=float)
    edges = _np.linspace(0.0, 1.0, n_bins + 1)
    idx = _np.clip(_np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = float(y[m].sum())
        phat = k / n
        # Wilson score interval, the same construction the stability tables use.
        z = 1.959963984540054
        den = 1.0 + z * z / n
        centre = (phat + z * z / (2 * n)) / den
        half = (z / den) * _np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        rows.append({
            "cohort": label, "harmonisation": "zscore", "bin": b + 1,
            "n": n, "n_events": int(k),
            "mean_predicted": round(float(p[m].mean()), 6),
            "observed": round(float(phat), 6),
            "obs_ci_low": round(float(max(0.0, centre - half)), 6),
            "obs_ci_high": round(float(min(1.0, centre + half)), 6)})
    return pd.DataFrame(rows)


def write_book(out, rows, prov, rel):
    TITLE_EV = [
        "EXTERNAL VALIDATION OF THE LOCKED TRANSCRIPTOMIC MODELS.",
        "Procedure: the frozen signature, classifier and hyperparameters are refit ONCE on the",
        "corresponding PREDIX cohort - the COMPLETE-CASE patients of the matching arm, which is",
        "the population the model was trained on - with no grid search and no cross-validation,",
        "then applied once to the external cohort. Nothing is refitted on external data and no",
        "feature is dropped: a signature with features removed is a different model.",
        "Harmonisation: features are standardised within each cohort by z-score.",
        "probability_calibration records whether the Brier score and the calibration slope and",
        "intercept were computed on Platt-scaled or on raw probabilities. AUROC and AUPRC are the",
        "same either way, Platt scaling being monotone.",
        "p_vs_chance_one_sided is a bootstrap tail probability, (draws at or below chance + 1) /",
        "(draws + 1) over 2000 stratified draws, so the smallest value it can take is 1/2001,",
        "about 0.0005. p_vs_chance_display renders a row at that floor as an inequality rather",
        "than as a point estimate; p_vs_chance_at_resolution_floor flags which rows those are.",
        "The genomic model could not be validated - see Feature_provenance.",
    ]
    TITLE_FP = [
        "FEATURE PROVENANCE: which signature features each external cohort measures.",
        "A model is scored only where EVERY feature of its signature is present.",
        "The DHP genomic signature is listed here as not validatable: the external genomic metric",
        "file carries five metrics, and three of the signature's four features are not among them.",
    ]
    TITLE_RL = [
        "RELIABILITY: observed pCR rate by bin of predicted probability, per cohort.",
        "Wilson intervals on the observed rate.",
        "",
    ]
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for sheet, title, df in (("External_validation", TITLE_EV, pd.DataFrame(rows)),
                                 ("Feature_provenance", TITLE_FP, pd.DataFrame(prov)),
                                 ("External_reliability", TITLE_RL, rel)):
            if df is None or len(df) == 0:
                df = pd.DataFrame({"note": ["not produced in this run"]})
            pd.DataFrame({0: title}).to_excel(xl, sheet_name=sheet, index=False,
                                              header=False, startrow=0)
            df.to_excel(xl, sheet_name=sheet, index=False, startrow=len(title) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="Results tree of ONE arm: the directory "
                         "holding results/ and report/.")
    args = ap.parse_args()
    run = Path(args.run).resolve()

    print("=" * 78)
    print(f"EXTERNAL VALIDATION - locked model source {run}")
    print("=" * 78)
    predix = load_and_encode_data(str(PREDIX))
    if isinstance(predix, tuple):
        predix = predix[0]
    cc = complete_case(predix)
    print(f"PREDIX complete-case cohort: {len(cc)} patients, "
          f"{int(cc['pCR'].sum())} events\n")

    rows, prov, rels = [], [], []
    for s in SETS + NOT_APPLICABLE:
        row, pr, rel = run_one(run, *s, cc)
        if row:
            rows.append(row)
        prov.extend(pr)
        if rel is not None:
            rels.append(rel)

    out_dir = run / "report" / "tables" / "revision"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "external_validation.xlsx"
    write_book(out, rows, prov,
               pd.concat(rels, ignore_index=True) if rels else None)

    # The CSV is the flat, per-evaluation form of the same result, with its own
    # column names; the figure script and the portability diagnostic read it.
    # It is emitted here, in the same pass and from the same row objects, so the
    # workbook and the CSV cannot disagree.
    #
    # The workbook schema has no signature column; look the signature up per row
    # rather than widening the workbook, which is read positionally downstream.
    _sig_for = {ARM_LABEL[exp]: "; ".join(spec(run, exp, mod)[0])
                for _, _, exp, _, mod, _ in SETS}

    def bounds(s):
        """Bare "lo-hi" from the workbook's "estimate [lo-hi]" string."""
        t = str(s)
        if "[" in t:
            t = t.split("[", 1)[1].rstrip("]")
        return t.replace("–", "-").strip()
    csv_rows = []
    for r in rows:
        csv_rows.append({
            # The workbook schema has no `modality`/`model` columns; the CSV
            # schema does. Derive rather than duplicate the fields in run_one.
            "cohort": r["cohort"], "cohort_description": r["cohort_description"],
            "modality": "Transcriptomic",
            "PREDIX_model": f"{r['cohort_resembles_PREDIX_arm']} (arm-matched)",
            "classifier": r["classifier"], "K": r["n_model_features"],
            "hyperparameters": r["hyperparameters"], "applicable": True,
            "missing_features": "",
            "n_PREDIX_fit": r["n_PREDIX_train"],
            "events_PREDIX_fit": r["events_PREDIX_train"],
            "internal_AUROC": r["internal_AUROC"],
            "internal_AUROC_CI": r["internal_AUROC_CI"],
            # NOTE: the two schemas differ deliberately. The WORKBOOK stores
            # "estimate [lo-hi]" because RESULTS.md renders that string straight
            # into a results column. This CSV stores BARE BOUNDS, because
            # make_fig_external_validation_all.py parses them to draw error bars.
            # Both are written from the same value here; do not retype either.
            "internal_n": r["n_PREDIX_train"],
            "internal_events": r["events_PREDIX_train"],
            "n_external": r["n_external"], "events_external": r["events_external"],
            "external_AUROC": r["external_AUROC"],
            "external_AUROC_CI": bounds(r["external_AUROC_CI"]),
            "external_AUROC_p_vs_chance_one_sided": r["p_vs_chance_one_sided"],
            "external_AUROC_p_vs_chance_display": r["p_vs_chance_display"],
            "external_AUROC_p_at_resolution_floor":
                r["p_vs_chance_at_resolution_floor"],
            "external_AUPRC": r["external_AUPRC"],
            "external_Brier": r["external_Brier"],
            "probability_calibration": r["probability_calibration"],
            "calibration_slope": r["calibration_slope"],
            "calibration_slope_CI": r["calibration_slope_CI"],
            "signature": _sig_for[r["cohort_resembles_PREDIX_arm"]]})
    # The genomic model is carried as a row with applicable=False rather than
    # dropped: a reader who finds three results in a four-row design should be
    # able to see WHY, in the same file.
    for lbl, path, exp, arm, mod, note in NOT_APPLICABLE:
        sig, clf, params = spec(run, exp, mod)
        ext = encode_external(path)
        miss = [f for f in sig if f not in ext.columns]
        csv_rows.append({
            "cohort": lbl, "cohort_description": note,
            "modality": "Genomic" if mod == "DNA" else "Transcriptomic",
            "PREDIX_model": f"{ARM_LABEL[exp]} (arm-matched)",
            "classifier": clf, "K": len(sig), "hyperparameters": str(params),
            "applicable": False, "missing_features": "; ".join(miss),
            "signature": "; ".join(sig)})
    # Both artefacts follow --run. The message names the file that was actually
    # written: an earlier version printed one path while writing another, which
    # is how a locked deliverable came to be overwritten without anyone noticing.
    _csv = out_dir / "locked_signatures_external_validation.csv"
    pd.DataFrame(csv_rows).to_csv(_csv, index=False)
    print(f"wrote {_csv} "
          f"({len(csv_rows)} rows: {len(rows)} validated, "
          f"{len(csv_rows) - len(rows)} not applicable)")
    print("\n" + "=" * 78)
    print(pd.DataFrame(rows)[["cohort", "classifier", "n_external",
                              "events_external", "internal_AUROC",
                              "external_AUROC", "external_AUROC_CI",
                              "p_vs_chance_display",
                              "probability_calibration"]].to_string(index=False))
    print(f"\nwrote {out}  ({len(rows)} validated, "
          f"{len(NOT_APPLICABLE)} not applicable)")


if __name__ == "__main__":
    main()
