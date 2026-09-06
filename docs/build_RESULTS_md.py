#!/usr/bin/env python3
"""
Build RESULTS.md — every result of the analysis, rendered so that it can be read
on GitHub without downloading or running anything.

Tables are generated from the deposited workbooks, never typed, so the page
cannot drift away from the analysis. Figures are embedded as PNG (GitHub does
not render PDF inline); the PDFs remain the citable artefacts.

    python revision_deliverables/build_RESULTS_md.py

Writes  predix-her2-multimodal/RESULTS.md
        predix-her2-multimodal/report/figures_png/*.png
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# --out mirrors build_github_repo.py's flag so this script can finish a deposit
# built somewhere other than the default tree. Parsed by hand rather than with
# argparse because this module does its work at import time, and adding a
# parser would mean restructuring the whole file. Anything unrecognised is
# rejected outright, so that a mistyped flag cannot silently rebuild the
# default deposit instead of printing usage.
_argv = sys.argv[1:]
if _argv in (["-h"], ["--help"]):
    sys.exit("usage: build_RESULTS_md.py [--out DIR]\n"
             "  --out DIR   deposit tree to write RESULTS.md into "
             "(default: predix-her2-multimodal/)")
if _argv[:1] == ["--out"]:
    if len(_argv) != 2:
        sys.exit("--out needs exactly one directory")
    REPO = Path(_argv[1]).resolve()
elif _argv:
    sys.exit(f"unrecognised arguments: {' '.join(_argv)}\n"
             "usage: build_RESULTS_md.py [--out DIR]")
else:
    REPO = ROOT / "predix-her2-multimodal"

TAB = REPO / "report" / "tables"
PNG_SRC = ROOT / "revision_deliverables" / "figures_png"
PNG_DST = REPO / "report" / "figures_png"
OUT = REPO / "RESULTS.md"

if not REPO.exists():
    sys.exit("run build_github_repo.py first")


# --------------------------------------------------------------- reading ----
def sheet(path, name, first_header):
    """Read one styled worksheet; the header row is the row whose first cell
    equals `first_header` (explanatory note rows precede it)."""
    ws = openpyxl.load_workbook(path, data_only=True)[name]
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    i = next(k for k, r in enumerate(rows) if r and r[0] == first_header)
    hdr = [h for h in rows[i] if h is not None]
    body = [list(r[:len(hdr)]) + [None] * max(0, len(hdr) - len(r))
            for r in rows[i + 1:] if any(v is not None for v in r)]
    return pd.DataFrame(body, columns=hdr)


def col(row, *names):
    """The first of `names` that exists in the row.

    A few workbook columns carry a long descriptive name and a short one —
    `recalibration_intercept` / `calibration_intercept`,
    `cohort_resembles_PREDIX_arm` / `matched_PREDIX_arm`,
    `delta_vs_best_p_marginal_selected_comparator` /
    `delta_vs_best_p_bootstrap`. Those cells are read through here, descriptive
    name first, so the page builds from whichever the workbook carries and
    fails loudly only when none of them is present.
    """
    for n in names:
        if n in row.index:
            return row[n]
    raise KeyError(f"none of {names} present; row has {list(row.index)}")


def _f(v):
    """Coerce a worksheet cell to float, or None. numpy integer types are not
    instances of `int` on Windows, so isinstance checks are not usable here."""
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def num(v, nd=3):
    """A measurement: always nd decimals."""
    if isinstance(v, str):
        return v
    f = _f(v)
    return "—" if f is None else f"{f:.{nd}f}"


def cnt(v):
    """A count: integer, with thousands separators."""
    if isinstance(v, str):
        return v
    f = _f(v)
    return "—" if f is None else f"{int(round(f)):,}"


def nword(n):
    """A small count spelled out, for sentences that count their own rows.
    Nothing on this page states a count as a word without passing it through
    here, so a table that gains or loses a row cannot leave the prose saying
    'both' about three of them."""
    return {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n, str(n))


def pct(v, nd=1):
    f = _f(v)
    return "—" if f is None else f"{f:.{nd}%}"


def pval(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    v = float(v)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def ci(lo, hi, nd=3):
    """An en-dash reads as a minus sign when a bound is negative, so switch to
    'a to b' whenever either bound is below zero."""
    a, b = _f(lo), _f(hi)
    sep = " to " if (a is not None and a < 0) or (b is not None and b < 0) else "–"
    return f"{num(lo, nd)}{sep}{num(hi, nd)}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


MOD = {"Clin": "Clinical", "RNA": "Transcriptomic", "DNA": "Genomic",
       "Prot": "Proteomic", "WSI": "Whole-slide image",
       "Fused_ElasticNet": "**Integrated (late fusion)**"}
ORDER = ["Clin", "RNA", "DNA", "Prot", "WSI", "Fused_ElasticNet"]
SC = ["Global", "DHP", "T-DM1"]
SCL = {"Global": "Pooled cohort", "DHP": "DHP arm", "T-DM1": "T-DM1 arm"}

# Two counts quoted in the captions below describe the bars of the
# feature-ranking figure: how many of them the mean-SHAP sign gets right, and
# how many carry a sign too unstable across folds to quote. Both are read from
# the figure's own workbook rather than typed, because a count that no longer
# matches the figure it captions is a false statement about a figure the reader
# is looking at.
_frank = sheet(TAB / "revision" / "fig_feature_ranking_by_scope.xlsx",
               "Feature_ranking", "scope")
N_RANK_BARS = len(_frank)
N_LOW_STABILITY = sum(str(v).strip().lower() == "true"
                      for v in _frank["direction_low_stability"])
# agreement in SIGN between the two statistics, over the bars where both exist
N_MEANSHAP_AGREE = sum(
    _f(a) is not None and _f(b) is not None and _f(a) * _f(b) > 0
    for a, b in zip(_frank["mean_signed_shap_NOT_a_direction"],
                    _frank["shap_dependence_r"]))

FIG_CAPTION = {
    "fig01_consensus_performance": "Cross-validated AUROC of every consensus model with its 95% patient-level cluster-bootstrap interval, in the pooled cohort and each arm.",
    "fig02_consensus_signatures": f"The frozen consensus signature of each modality and scenario, with the winning classifier family above each panel. Two cautions. The bar is the mean cross-classifier selection RANK, not a SHAP magnitude, so lengths are comparable within a panel and not between panels (generate_report.py:1728-1746). And the green/red colouring marks the sign of each feature's mean SHAP value, which averages to ~0 by construction and therefore does not carry direction: it agrees with the SHAP dependence slope on {N_MEANSHAP_AGREE} of {N_RANK_BARS} consensus features. Read direction from fig_feature_ranking_by_scope and its workbook instead; see docs/FEATURE_DIRECTION_CORRECTION.md.",
    "fig03_consensus_roc": "Out-of-fold ROC curves of the integrated model and of the best single modality, drawn on all pooled (patient, repeat) predictions.",
    "fig04_consensus_modality_weights": "Late-fusion modality weights of the consensus models: mean elastic-net coefficient and the fraction of folds in which each modality received a non-zero weight.",
    "fig05_consensus_feature_shap_Global": "Feature-level SHAP attribution for the pooled-cohort consensus models, restricted to the consensus signature.",
    "fig05_consensus_feature_shap_DHP": "Feature-level SHAP attribution for the DHP-arm consensus models.",
    "fig05_consensus_feature_shap_T_DM1": "Feature-level SHAP attribution for the T-DM1-arm consensus models.",
    "fig06_counterfactual_summary": "Counterfactual summary: predicted response under each treatment assignment.",
    "revfig01_calibration": "Calibration of the consensus integrated model: reliability curves over ten equal-count bins of all out-of-fold predictions, with patient-level cluster-bootstrap intervals, and the slope, intercept and Brier score of each scenario.",
    "revfig02_selection_stability": "Feature-selection frequency across the outer folds, with Wilson intervals and the pre-specified stability threshold (0.60 pooled, 0.50 per arm).",
    "revfig03_epv_per_fold": "Per-fold pCR event counts and realised events-per-variable for every model.",
    # The cohort-specific half of this caption is appended in Section 8, where
    # the external workbook is read, so that the figure is described by the same
    # rows the table above it prints.
    "revfig06_external_validation": "External validation of the locked transcriptomic models, drawn as an AUROC forest: for every cohort a pair of intervals, the internal cross-validated AUROC of the model and the external AUROC of that same frozen model applied once, each with its 95% confidence interval and a chance reference at 0.5. Nothing was refitted on external data.",
    # Drawn by code/make_fig_feature_ranking.py from results/<arm>/*.pkl alone.
    "fig_feature_ranking_by_scope": f"Consensus-signature feature ranking by model scope: every feature in the frozen signature of each modality, for the pooled model and for each arm-specific model, with the direction of the pCR association marked by the sign of the SHAP dependence slope — the correlation between a feature's standardised value and its own SHAP value over the folds that selected it, so that + means a higher value of the feature pushes the prediction towards pCR. A bar whose per-fold slope agrees with that pooled slope in fewer than 75% of folds keeps its sign but is drawn in parentheses and greyed, because a direction that reverses between folds should not be quoted; {N_LOW_STABILITY} of {N_RANK_BARS} bars are bracketed on this basis. The bar length is the mean cross-classifier selection rank across discovery folds, so it is comparable within a modality block and not between blocks — a rank of 1.0 means top of its own panel, not the strongest predictor in the study. Values, slopes and per-bar sign stability are tabulated in report/tables/revision/fig_feature_ranking_by_scope.xlsx. NOTE: fig02 and supp_fig06 below mark direction with a different and discredited statistic; see docs/FEATURE_DIRECTION_CORRECTION.md.",
    "revfig07_model_comparisons": "AUROC forest and paired ΔAUROC of the integrated model against every single-modality comparator, with 95% paired cluster-bootstrap intervals.",
    "revfig08_fusion_weights": "Fold-wise distribution of the late-fusion modality weights and each modality's selection rate.",
    "supp_fig01_roc_curves": "Discovery-phase ROC curves.",
    "supp_fig02_performance_distributions": "Discovery-phase distribution of per-fold performance for every model.",
    "supp_fig03_fusion_benefit": "Discovery-phase fusion benefit against the best single modality.",
    "supp_fig04_forest_plot": "Discovery-phase forest plot of per-fold AUROC.",
    "supp_fig05_feature_shap_Global": "Discovery-phase SHAP attribution, pooled cohort.",
    "supp_fig05_feature_shap_DHP": "Discovery-phase SHAP attribution, DHP arm.",
    "supp_fig05_feature_shap_T_DM1": "Discovery-phase SHAP attribution, T-DM1 arm.",
    "supp_fig06_feature_selection_frequency": "Discovery-phase selection frequency of every candidate feature. Its green/red colouring uses the same mean-SHAP sign as fig02 and carries the same caveat: it does not reliably indicate the direction of the pCR association. See docs/FEATURE_DIRECTION_CORRECTION.md.",
    "supp_fig07_cross_scenario_features": "Features shared between the pooled and arm-specific signatures.",
    "supp_fig08_fusion_shap": "SHAP attribution of the five modality streams inside the fusion layer.",
    "supp_fig09_modality_weights": "Discovery-phase modality weights.",
    "supp_fig10_winner_classifier_heatmap": "Which classifier family won each fold, by modality and scenario.",
    "supp_fig11_inner_auroc_comparison": "Inner-cross-validation AUROC of each classifier family, the basis of the Stage A choice.",
    "supp_fig12_calibration_profile": "Discovery-phase calibration profile.",
    "supp_fig13_signature_sizes": "Distribution of discovered signature sizes across folds.",
    "supp_fig14_performance_CI": "Discovery-phase AUROC with patient-level cluster-bootstrap intervals — the fully nested estimates, free of consensus selection optimism.",
}

M = []          # markdown accumulator
def w(s=""):
    M.append(s)


# =============================================================== header =====
prov = json.loads((REPO / "results" / "run_provenance.json").read_text(encoding="utf-8-sig"))
P = prov["parameters"]

w("# Results")
w()
w("Every table on this page is generated directly from the deposited workbooks "
  "under [`report/tables/`](report/tables) by "
  "[`docs/build_RESULTS_md.py`](docs/build_RESULTS_md.py), and every figure is the "
  "PNG rendering of the corresponding PDF in [`report/figures/`](report/figures). "
  "Nothing here is typed by hand.")
w()
w(f"Pipeline `{prov['pipeline_version']}`, seed {prov['random_seed']}. "
  f"Regenerated {date.today().isoformat()}.")
w()
w("> **How to read every number below.** In each cross-validation repeat every "
  "patient has exactly one out-of-fold prediction; the metric is computed on "
  "that complete out-of-fold vector and averaged over the repeats (200 pooled, "
  "100 per arm). The 95% interval is a patient-level **cluster** bootstrap — "
  "2,000 stratified resamples of patients, a resampled patient carrying all of "
  "its repeat predictions. Predictions are never averaged across repeats or "
  "across models. A comparison whose interval for ΔAUROC includes zero is "
  "reported as *not distinguishable*, however large the point difference.")
w()
w("## Contents")
w()
for i, t in enumerate([
        "Design and cohort", "Cross-validated performance",
        "Integration versus the best single modality",
        "Calibration", "Events per variable",
        "Feature-selection stability", "Consensus signatures and fusion weights",
        "External validation", "Figures"], 1):
    w(f"{i}. [{t}](#{i}-{t.lower().replace(' ', '-').replace('?', '').replace(',', '')})")
w()

# ===================================================== 1. design & cohort ====
perf = sheet(TAB / "revision" / "revision_performance_CI.xlsx",
             "Performance_patient_CI", "scenario")
cons = perf[perf["source"] == "consensus"]
w("## 1. Design and cohort")
w()
rows = []
for sc in SC:
    r = cons[(cons["scenario"] == sc) & (cons["model"] == "Fused_ElasticNet")].iloc[0]
    rows.append([SCL[sc], cnt(r["n_patients"]), cnt(r["n_events"]),
                 f"{float(r['n_events']) / float(r['n_patients']):.1%}",
                 cnt(r["n_cv_repeats"]), cnt(r["n_outer_folds"])])
w(table(rows, ["Cohort", "Patients evaluated", "pCR events", "pCR rate",
               "CV repeats", "Outer evaluations"]))
w()
# The training cohort is the central design point of the analysis, so it is
# stated here with the run's own per-fold numbers underneath it rather than left
# to be inferred from the one-line design note further down. Both the sentence
# and the numbers come from the run: the training population from
# run_provenance.json, the per-fold sizes from the EPV workbook.
_TRAINING = P.get("training_data", "cc_only")
assert _TRAINING == "cc_only", (
    f"run_provenance.json records training_data={_TRAINING!r}, but the design "
    "prose in Section 1 describes complete-case training. Rewrite the prose "
    "before rebuilding this page.")
_epvf = sheet(TAB / "revision" / "revision_epv_per_fold.xlsx",
              "EPV_per_fold", "scenario")
_trows = []
_sizes = {}
_mods_shown = set()
for sc in SC:
    _d = _epvf[(_epvf["scenario"] == sc) & (_epvf["model"] != "Fused_ElasticNet")]
    if _d.empty:
        continue
    _cells, _seen_sizes = [], set()
    for _m in ORDER[:-1]:
        _md = _d[_d["model"] == _m]
        if _md.empty:
            continue
        # complete-case training: `n_train_cc` IS the training set of every
        # model, unimodal and integrated alike
        _n, _e = (int(_md["n_train_cc"].median()),
                  int(_md["n_events_train_cc"].median()))
        _seen_sizes.add((_n, _e))
        _mods_shown.add(_m)
        _cells.append(f"{MOD[_m]} {_n}/{_e}")
    _sizes[sc] = _seen_sizes
    _n_eval = cons[(cons["scenario"] == sc)
                   & (cons["model"] == "Fused_ElasticNet")].iloc[0]["n_patients"]
    _trows.append([SCL[sc], cnt(_n_eval), "; ".join(_cells)])
# derived, not typed: if a modality ever trained on a different set of patients
# the sentence below would have to say so, and it would say so
_same_train = all(len(s) == 1 for s in _sizes.values())
_n_pooled = cnt(cons[(cons["scenario"] == "Global")
                     & (cons["model"] == "Fused_ElasticNet")].iloc[0]["n_patients"])
w(f"**The {_n_pooled} patients of the pooled cohort are the training cohort as "
  "well as the evaluation cohort.** Both the training folds and the outer test "
  "folds are drawn from the patients complete on all four omics modalities, so "
  "that every model sees the same patients and the comparisons between them are "
  "paired. No model is fitted on a patient that another model cannot be fitted "
  "on, and no patient enters a training set on the strength of carrying one "
  "modality alone. The per-fold training cohorts "
  + (f"are therefore identical across the {nword(len(_mods_shown))} modalities, "
     "and smaller than the cohort only by that fold's held-out patients:"
     if _same_train else "are given by modality below:"))
w()
w(table(_trows, ["Cohort", "Evaluated on",
                 "Median per-fold training cohort, patients/events, by modality"]))
w()
w("The fusion layer is the component most exposed by that choice: it takes five "
  "modality inputs by design, so its variable count cannot be capped the way a "
  "single-modality signature can. That is why its events-per-variable falls "
  "short inside the arms — reported in Section 5 rather than hidden.")
w()
# The per-scenario deduplication is a product of the run, so its counts are read
# from the deposited audits rather than typed.
_DEDUP_NOTE = ""
if P.get("dedup_per_scenario"):
    _n_removed = {}
    for _exp in ("global", "dhp", "tdm1"):
        _a = REPO / "results" / _exp / f"{_exp}_deduplication_audit.csv"
        if not _a.is_file():
            sys.exit(f"dedup_per_scenario is set but the audit is missing: {_a}")
        _n_removed[_exp] = len(pd.read_csv(_a))
    _DEDUP_NOTE = (f" → {92 - _n_removed['global']} (pooled), "
                   f"{92 - _n_removed['dhp']} (DHP) and {92 - _n_removed['tdm1']} (T-DM1) "
                   "after the per-scenario, within-modality deduplication "
                   "(`results/<scenario>/<scenario>_deduplication_audit.csv`)")
w(table([
    ["Outer resampling", f"stratified {P['outer_folds_global']}-fold "
                         f"`RepeatedStratifiedKFold` (no shuffle-split)"],
    ["Inner resampling", f"{P['inner_folds_global']}-fold (pooled), "
                         f"{P['inner_folds_arm']}-fold (per arm)"],
    # the curated panel and the deduplication list are fixed inputs of the
    # pipeline, not products of the run, so they are not in any workbook
    ["Candidate panel", "110 pre-defined metrics → 92 after the outcome-blind "
                        "biological deduplication" + _DEDUP_NOTE],
    ["Feature screen", f"in-fold Mann–Whitney AUROC, BH q ≤ {P['univ_fdr_q']}, "
                       f"keep {P['univ_min_k']}–{P['univ_max_k']}"],
    ["Classifier families", ", ".join(f"`{c}`" for c in P["classifiers"])],
    ["Signature size cap", "at least 5 pCR events per selected variable"],
    ["Fusion", "elastic-net logistic regression (l1_ratio 0.5) over the five "
               "Platt-calibrated modality probability streams"],
    ["Consensus finalisation", f"features above the stability threshold "
                               f"({P['stability_thresh_global']} pooled, "
                               f"{P['stability_thresh_arm']} per arm); modal classifier"],
    # read from provenance rather than typed
    ["Signature aggregation", f"`{P.get('signature_source', 'all_folds')}`"
                              + (" — aggregated only over the outer folds the "
                                 "modal classifier won, so the reported "
                                 "classifier and signature are one model"
                                 if P.get("signature_source") == "winner_folds"
                                 else "")],
    ["Training cohort", f"`{_TRAINING}` — the patients complete on all four "
                        "omics modalities; the same patients the models are "
                        "evaluated on"],
    ["Random seed", str(prov["random_seed"])],
], ["Design element", "Value"]))
w()
w("The full design is drawn in "
  "[`docs/ED_Fig11a_CV_schematic.pdf`](docs/ED_Fig11a_CV_schematic.pdf) and stated "
  "in [`docs/methods_cv_statement.txt`](docs/methods_cv_statement.txt), both "
  "generated from the run's own parameters.")
w()

# ===================================================== 2. performance ========
w("## 2. Cross-validated performance")
w()
w("Consensus models — the frozen signature and classifier re-evaluated on the "
  "same outer splits. Source: `report/tables/revision/revision_performance_CI.xlsx`.")
w()
for sc in SC:
    d = cons[cons["scenario"] == sc]
    w(f"### {SCL[sc]}")
    w()
    rows = []
    for m in ORDER:
        r = d[d["model"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([MOD[m],
                     f"**{num(r['AUROC'])}**" if m == "Fused_ElasticNet" else num(r["AUROC"]),
                     ci(r["AUROC_CI_low"], r["AUROC_CI_high"]),
                     num(r["AUPRC"]), ci(r["AUPRC_CI_low"], r["AUPRC_CI_high"]),
                     num(r["Brier"]), ci(r["Brier_CI_low"], r["Brier_CI_high"])])
    w(table(rows, ["Model", "AUROC", "95% CI", "AUPRC", "95% CI", "Brier", "95% CI"]))
    w()

disc = perf[perf["source"] == "discovery"]
w("### Discovery phase (fully nested)")
w()
w("The signature and classifier are re-selected independently inside every fold, "
  "so these estimates carry no consensus selection optimism. They are the "
  "conservative reading of the same data.")
w()
rows = []
for sc in SC:
    d = disc[disc["scenario"] == sc]
    for m in ORDER:
        r = d[d["model"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        c = cons[(cons["scenario"] == sc) & (cons["model"] == m)]
        gap = (float(c.iloc[0]["AUROC"]) - float(r["AUROC"])) if not c.empty else None
        rows.append([SCL[sc], MOD[m].replace("**", ""), num(r["AUROC"]),
                     ci(r["AUROC_CI_low"], r["AUROC_CI_high"]),
                     f"{gap:+.3f}" if gap is not None else "—"])
w(table(rows, ["Cohort", "Model", "Discovery AUROC", "95% CI",
               "Consensus − discovery"]))
w()

# ===================================================== 3. comparisons =======
cmp_ = sheet(TAB / "revision" / "revision_model_comparisons.xlsx",
             "Model_comparisons", "scenario")
fb = sheet(TAB / "revision" / "revision_model_comparisons.xlsx",
           "Fusion_benefit", "scenario")
cc = cmp_[cmp_["source"] == "consensus"]
w("## 3. Integration versus the best single modality")
w()
# This section must not be reduced to a yes/no answer. Every row of the verdict
# column reads "not distinguishable", i.e. the 95% interval for ΔAUROC includes
# zero, and that is an ABSENCE OF EVIDENCE for a difference in either direction,
# not evidence of no benefit — while in the pooled cohort the integrated model
# still carries the highest point estimate of all six models. Headline, verdict
# and ranking are all derived from the deposited workbooks, so none of them can
# drift away from the tables printed directly beneath them.
_fbc = fb[fb["source"] == "consensus"]
_verdicts = {str(v).strip().lower() for v in _fbc["verdict_vs_best_unimodal"]}
_all_nd = _verdicts <= {"not distinguishable"}


def _auroc_ranking(src, sc):
    """Every model of one scenario ordered by point-estimate AUROC, best first.

    Returns `(pairs, rank_of_fused, n_models)` where `pairs` is
    [(model, AUROC), ...] descending and `rank_of_fused` is 1-based.

    Derived from `revision_performance_CI.xlsx`, already loaded above. Nothing
    here is a typed constant: the ordering is recomputed on every build from the
    same table the page prints, so the two cannot disagree.
    """
    d = src[src["scenario"] == sc]
    pairs = [(m, _f(d[d["model"] == m].iloc[0]["AUROC"]))
             for m in ORDER if not d[d["model"] == m].empty]
    pairs = sorted([(m, a) for m, a in pairs if a is not None],
                   key=lambda t: -t[1])
    names = [m for m, _ in pairs]
    rk = names.index("Fused_ElasticNet") + 1 if "Fused_ElasticNet" in names else None
    return pairs, rk, len(pairs)


_cons_rank = {sc: _auroc_ranking(cons, sc) for sc in SC}
_disc_rank = {sc: _auroc_ranking(disc, sc) for sc in SC}
_headline = ("**Not established in either direction.**" if _all_nd
             else "**Mixed — read the verdict column.**")
w(f"{_headline} "
  + ("Every paired comparison of the integrated model against the best single "
     "modality returns *not distinguishable*: the interval for ΔAUROC includes "
     "zero in all of them. The data do not establish that integration beats the "
     "best single modality, and they equally do not establish that it falls "
     "short of one. This is not a finding that integration adds nothing, and it "
     "must not be quoted as one."
     if _all_nd else
     "The verdicts differ between scenarios — read the verdict column rather "
     "than this sentence.")
  + " Paired patient-level cluster bootstrap, the same patient resample applied "
    "to both models and all repeats.")
w()
# The ranking is reported alongside the verdict because a null result and a
# point-estimate ordering are different statements, and suppressing the second
# is how "not distinguishable" gets read as "fusion adds nothing".
_rk_bits = "; ".join(f"{SCL[sc]} **{_cons_rank[sc][1]} of {_cons_rank[sc][2]}**"
                     for sc in SC if _cons_rank[sc][1] is not None)
_dk_bits = "; ".join(f"{SCL[sc]} {_disc_rank[sc][1]} of {_disc_rank[sc][2]}"
                     for sc in SC if _disc_rank[sc][1] is not None)
_agree = all(_cons_rank[sc][1] == _disc_rank[sc][1] for sc in SC)
w("Point estimates still order the models, and that ordering is part of the "
  "result — it is not a substitute for the test above, and a rank is not a "
  f"significant difference. On consensus AUROC the integrated model ranks "
  f"{_rk_bits}. Under the fully nested discovery source it ranks {_dk_bits}"
  + ("." if _agree else
     " — so the two sources do not agree on the ordering, which is a further "
     "reason to report the comparison as undetermined rather than to assert a "
     "direction from the point estimates.")
  + " The full ordering under both sources is tabulated below.")
w()
rows = []
for sc in SC:
    r = fb[(fb["scenario"] == sc) & (fb["source"] == "consensus")]
    if r.empty:
        continue
    r = r.iloc[0]
    rows.append([SCL[sc], num(r["fused_AUROC"]),
                 f"{MOD.get(r['best_unimodal'], r['best_unimodal'])} "
                 f"{num(r['best_unimodal_AUROC'])}".replace("**", ""),
                 f"{float(r['delta_vs_best_unimodal']):+.3f}",
                 str(r["delta_vs_best_CI"]),
                 # the P value is marginal on the comparator that was
                 # *selected* as best, which is what the longer column name
                 # records
                 pval(col(r, "delta_vs_best_p_marginal_selected_comparator",
                          "delta_vs_best_p_bootstrap")),
                 str(r["verdict_vs_best_unimodal"])])
w(table(rows, ["Cohort", "Integrated AUROC", "Best single modality", "Δ AUROC",
               "95% CI", "P", "Verdict"]))
w()
# The `Fusion_benefit` sheet takes an argmax over the five comparators, so its
# `best_unimodal` column structurally reports exactly ONE modality per row. In
# the DHP arm that hides the fact that a second modality also sits above the
# integrated point estimate. Say so, and print the full ordering underneath.
w("The **Best single modality** column names one comparator only. The "
  "workbook's `Fusion_benefit` sheet selects it by taking the highest-scoring "
  "modality, so by construction the column cannot show whether a *second* "
  "modality also scores above the integrated model. The ordering below shows "
  "every modality, and the per-comparator table after it gives every paired "
  "contrast individually.")
w()
w("### Where the integrated model ranks")
w()
w("All six models of each scenario ordered by point-estimate AUROC, best first, "
  "derived from `report/tables/revision/revision_performance_CI.xlsx`. The "
  "intervals in section 2 overlap heavily; these ranks are descriptive and "
  "carry no inferential claim.")
w()
rows = []
for _slbl, _srank in [("consensus", _cons_rank), ("discovery", _disc_rank)]:
    for sc in SC:
        _pairs, _rk, _n = _srank[sc]
        if _rk is None:
            continue
        _order = " > ".join(
            (f"**{MOD[m].replace('**', '')} {a:.3f}**" if m == "Fused_ElasticNet"
             else f"{MOD.get(m, m)} {a:.3f}")
            for m, a in _pairs)
        rows.append([SCL[sc], _slbl, f"**{_rk} of {_n}**", _order])
w(table(rows, ["Cohort", "Source", "Integrated rank",
               "Ordering by AUROC, best first"]))
w()
w("Against every comparator:")
w()
rows = []
for sc in SC:
    d = cc[cc["scenario"] == sc]
    for _, r in d.iterrows():
        rows.append([SCL[sc], MOD.get(r["comparator"], r["comparator"]).replace("**", ""),
                     f"{float(r['delta_AUROC']):+.3f}",
                     ci(r["delta_CI_low"], r["delta_CI_high"]),
                     pval(r["p_bootstrap"]), pval(r["q_bootstrap_BH"]),
                     str(r["verdict"]).replace("Fused_ElasticNet", "integrated")])
w(table(rows, ["Cohort", "Integrated vs", "Δ AUROC", "95% CI", "P", "BH q", "Verdict"]))
w()
w("DeLong's test computed per repeat and summarised is reported in the workbook "
  "as a descriptive secondary analysis; the bootstrap is the primary comparison.")
w()

# ======================================================= 4. calibration =====
cal = sheet(TAB / "revision" / "revision_calibration.xlsx",
            "Calibration_summary", "scenario")
rel = sheet(TAB / "revision" / "revision_calibration.xlsx",
            "Reliability_bins", "scenario")
w("## 4. Calibration")
w()
w("Slope and intercept of `logit(pCR) = a + b · logit(p̂)`, fitted on each "
  "repeat's out-of-fold vector and averaged. Slope 1 and intercept 0 are perfect; "
  "slope below 1 means the predictions are too extreme (the classic overfitting "
  "signature), above 1 that they are compressed toward the base rate.")
w()
rows = []
_covers = True
for sc in SC:
    r = cal[cal["scenario"] == sc].iloc[0]
    _b = col(r, "recalibration_intercept", "calibration_intercept")
    _blo = col(r, "recalibration_intercept_CI_low", "intercept_CI_low")
    _bhi = col(r, "recalibration_intercept_CI_high", "intercept_CI_high")
    if not (_f(r["slope_CI_low"]) <= 1 <= _f(r["slope_CI_high"])
            and _f(_blo) <= 0 <= _f(_bhi)):
        _covers = False
    rows.append([SCL[sc], num(r["calibration_slope"], 2),
                 ci(r["slope_CI_low"], r["slope_CI_high"], 2),
                 num(_b, 2), ci(_blo, _bhi, 2),
                 num(r["brier"]), num(r["ECE"]),
                 f"{float(r['observed_pCR_rate']):.3f} vs {float(r['mean_predicted']):.3f}"])
w(table(rows, ["Cohort", "Slope", "95% CI", "Intercept", "95% CI", "Brier", "ECE",
               "Observed vs mean predicted"]))
w()
# derived, not typed: an interval that stopped covering its null would otherwise
# leave a sentence asserting the opposite of the table above it
w("Every slope interval covers 1 and every intercept interval covers 0."
  if _covers else
  "Not every slope interval covers 1 or every intercept interval covers 0 — "
  "read the two CI columns above.")
w()
w("<details><summary>Reliability bins (equal-count bins over all "
  "(patient, repeat) out-of-fold predictions)</summary>")
w()
rows = [[SCL[r["scenario"]], cnt(r["bin"]), cnt(r["n_rows"]),
         cnt(r["n_patients_distinct"]), num(r["mean_predicted"]),
         num(r["observed"]), ci(r["obs_ci_low"], r["obs_ci_high"])]
        for _, r in rel.iterrows()]
w(table(rows, ["Cohort", "Bin", "Predictions", "Distinct patients",
               "Mean predicted", "Observed", "95% CI"]))
w()
w("</details>")
w()

# ============================================================== 5. EPV ======
epv = sheet(TAB / "revision" / "revision_epv_per_fold.xlsx", "EPV_summary", "scenario")
w("## 5. Events per variable")
w()
w("The design caps signature size at five pCR events per selected variable. "
  "This table reports what was actually realised in each fold.")
w()
rows = []
for _, r in epv.iterrows():
    rows.append([SCL.get(r["scenario"], r["scenario"]),
                 MOD.get(r["model"], r["model"]).replace("**", ""),
                 cnt(r["n_folds"]),
                 f"{cnt(r['median_n_events_test'])} ({cnt(r['min_n_events_test'])}–{cnt(r['max_n_events_test'])})",
                 cnt(r["median_signature_size"]), num(r["median_epv_realized"], 2),
                 num(r["min_epv_realized"], 2),
                 f"{float(r['pct_folds_epv_below_5']):.1f}%"])
w(table(rows, ["Cohort", "Model", "Folds", "Test-fold events (median, range)",
               "Median signature size", "Median EPV", "Min EPV", "Folds below EPV 5"]))
w()
# the two percentages are read from the table above, so the sentence cannot
# describe a spread the table does not show
_fus = epv[epv["model"] == "Fused_ElasticNet"].set_index("scenario")
_pdhp = _f(_fus.loc["DHP", "pct_folds_epv_below_5"])
_ptdm = _f(_fus.loc["T-DM1", "pct_folds_epv_below_5"])
_others = epv[(epv["model"] != "Fused_ElasticNet")
              & (epv["pct_folds_epv_below_5"].map(_f) > 0)]
w("The arm-level fusion layer is the most exposed component: it takes five "
  "modality inputs by design and cannot be capped, so "
  f"{_pdhp:.0f}% of DHP folds and {_ptdm:.0f}% of T-DM1 folds run below five "
  "events per variable."
  + (" Every single-modality model, in every scenario, stays at or above the "
     "cap in every fold." if _others.empty else
     f" {len(_others)} single-modality rows also fall below it; see the table."))
w()

# ======================================================== 6. stability ======
stab = sheet(TAB / "revision" / "revision_stability.xlsx",
             "Feature_selection_stability", "scenario")
mws = sheet(TAB / "revision" / "revision_stability.xlsx",
            "Modality_weight_stability", "scenario")
stable = stab[stab["stable"].astype(str).str.lower().isin(["true", "yes", "1"])]
w("## 6. Feature-selection stability")
w()
w(f"How often each candidate feature was selected across the outer folds. "
  f"Features above the pre-specified threshold ({P['stability_thresh_global']} "
  f"pooled, {P['stability_thresh_arm']} per arm) are the consensus signature; "
  f"{len(stable)} of {len(stab)} candidate rows clear it.")
w()
w("**The threshold is applied to the *eligible-fold* frequency** — the fraction "
  "of the folds in which the feature survived preprocessing and the in-fold "
  "screen at all. A feature can therefore be stable on that denominator while "
  "its all-fold frequency is low: it was rarely eligible, but was chosen almost "
  "whenever it was. Both columns are given below, with the Wilson interval on "
  "the eligible-fold proportion.")
w()
for sc in SC:
    d = stable[stable["scenario"] == sc]
    if d.empty:
        continue
    w(f"<details><summary><b>{SCL[sc]}</b> — {len(d)} stable features</summary>")
    w()
    rows = [[MOD.get(r["modality"], r["modality"]).replace("**", ""),
             f"`{r['feature']}`", num(r["selection_freq"]),
             num(r["selection_freq_eligible"]),
             ci(r["wilson_low"], r["wilson_high"]),
             cnt(r["n_selected"]) + " / " + cnt(r["n_folds_total"])]
            for _, r in d.sort_values(
                ["modality", "selection_freq_eligible", "selection_freq"],
                ascending=[True, False, False]).iterrows()]
    w(table(rows, ["Modality", "Feature", "All-fold frequency",
                   "Eligible-fold frequency", "95% Wilson CI", "Folds selected"]))
    w()
    w("</details>")
    w()
w("### Stability of the fusion weights")
w()
rows = []
for _, r in mws.iterrows():
    rows.append([SCL.get(r["scenario"], r["scenario"]),
                 MOD.get(r["modality"], r["modality"]).replace("**", ""),
                 num(r["mean_weight"], 2), num(r["median_weight"], 2),
                 pct(r["selection_rate"]),
                 ci(r["selection_rate_ci_low"], r["selection_rate_ci_high"], 2),
                 num(r["sign_consistency"], 2)])
w(table(rows, ["Cohort", "Modality", "Mean weight", "Median weight",
               "Selection rate", "95% CI", "Sign consistency"]))
w()

# ===================================================== 7. signatures ========
sig = sheet(TAB / "PREDIX_HER2_results.xlsx", "Signatures", "Scenario")
fus = sheet(TAB / "PREDIX_HER2_results.xlsx", "Fusion", "Scenario")
w("## 7. Consensus signatures and fusion weights")
w()
for sc in SC:
    d = sig[sig["Scenario"] == sc]
    if d.empty:
        continue
    w(f"### {SCL[sc]}")
    w()
    rows = []
    for m in ORDER[:-1]:
        dm = d[d["Modality"] == m].sort_values("Rank")
        if dm.empty:
            continue
        feats = ", ".join(f"`{f}`" for f in dm["Feature"])
        r0 = dm.iloc[0]
        rows.append([MOD[m], cnt(r0["Signature size (K)"]),
                     f"`{r0['Winner classifier']}`",
                     f"{float(r0['Classifier support (%)']):.0f}%", feats])
    w(table(rows, ["Modality", "K", "Winning classifier", "Fold support",
                   "Signature (in rank order)"]))
    w()
w("### Late-fusion modality weights")
w()
rows = []
for sc in SC:
    d = fus[fus["Scenario"] == sc].sort_values("Mean fusion coefficient",
                                               ascending=False)
    for _, r in d.iterrows():
        rows.append([SCL[sc], MOD.get(r["Modality"], r["Modality"]).replace("**", ""),
                     num(r["Mean fusion coefficient"], 2),
                     num(r["SD fusion coefficient"], 2),
                     pct(r["Selection rate (|coef| > 1e-6)"])])
w(table(rows, ["Cohort", "Modality", "Mean coefficient", "SD", "Selection rate"]))
w()

# ================================================== 8. external validation ==
ext = sheet(TAB / "revision" / "external_validation.xlsx",
            "External_validation", "cohort")
# Which signature features each external cohort measures. This sheet is what
# decides whether a locked model could be carried to a cohort at all, so the
# models that could not be scored are read from it and printed here rather than
# left in the workbook for a reader to find.
extprov = sheet(TAB / "revision" / "external_validation.xlsx",
                "Feature_provenance", "cohort")
w("## 8. External validation")
w()
w("The pipeline's own transcriptomic consensus model was **frozen** — signature, "
  "classifier and hyper-parameters — refit once on PREDIX with no grid search, "
  "and applied to the external cohort. Nothing was refitted on external data. "
  "Because the cohorts are on incompatible measurement scales, features were "
  "standardised within each cohort by z-score, computed independently within "
  "each cohort and without reference to outcome; the `Harmonisation` column "
  "below records it on every row.")
w()
w("The refit population is the PREDIX arm whose regimen the external cohort "
  "resembles, restricted to the same complete-case patients the model was "
  "trained on. Every feature of the frozen signature has to be present in the "
  "external cohort: scoring a signature with features removed evaluates a "
  "different model, so a model whose signature the cohort does not fully "
  "measure is reported as not scoreable instead of being scored on what "
  "remains.")
w()


def ext_table(df, heading, source):
    w(f"### {heading}")
    w()
    w(f"Source: `{source}`.")
    w()
    rows = []
    for _, r in df.iterrows():
        rows.append([str(r["cohort"]),
                     str(col(r, "model_refit_population", "matched_PREDIX_arm")),
                     str(r["harmonisation"]),
                     cnt(r["n_external"]), cnt(r["events_external"]),
                     str(r["internal_AUROC_CI"]),
                     f"**{str(r['external_AUROC_CI'])}**",
                     str(r["external_AUPRC_CI"]), str(r["external_Brier_CI"]),
                     f"{num(r['calibration_slope'], 2)} ({r['calibration_slope_CI']})",
                     pval(r["p_vs_chance_one_sided"])])
    w(table(rows, ["Cohort", "Refit on", "Harmonisation", "n", "pCR",
                   "Internal AUROC", "External AUROC", "AUPRC", "Brier",
                   "Calibration slope", "P vs chance"]))
    w()
    w("Locked specifications:")
    w()
    seen = set()
    rows = []
    for _, r in df.iterrows():
        k = str(r["cohort"])
        if k in seen:
            continue
        seen.add(k)
        rows.append([str(r["cohort_description"]),
                     str(col(r, "cohort_resembles_PREDIX_arm",
                             "matched_PREDIX_arm")),
                     str(col(r, "model_refit_population", "matched_PREDIX_arm")),
                     f"`{r['classifier']}` {r['hyperparameters']}",
                     cnt(r["n_model_features"]),
                     f"{cnt(r['n_PREDIX_train'])} / {cnt(r['events_PREDIX_train'])}"])
    w(table(rows, ["External cohort", "Resembles PREDIX arm", "Refit on",
                   "Frozen classifier", "Features", "Refit on (n / events)"]))
    w()


ext_table(ext, "Scored models",
          "report/tables/revision/external_validation.xlsx")


# Every (cohort, modality) pair in the provenance sheet is one locked model
# carried to one external cohort. A pair with an absent feature is a model that
# could not be scored, and it belongs on this page beside the ones that could:
# a reader quoting the scored results has to be able to see how many
# evaluations were attempted to produce them.
def _mod_of(feature):
    """The modality of a signature feature, from its `Modality_name` prefix."""
    return str(feature).split("_", 1)[0]


def _modname(m):
    """`RNA` → `transcriptomic`, for use inside a sentence."""
    return MOD.get(m, m).replace("**", "").lower()


_absent = extprov[extprov["status"].astype(str).str.strip().str.upper() == "ABSENT"]


def _absent_features(cohort, modality):
    return sorted(f"`{r['feature']}`" for _, r in _absent.iterrows()
                  if str(r["cohort"]) == cohort
                  and _mod_of(r["feature"]) == modality)


_attempted = sorted({(str(r["cohort"]), _mod_of(r["feature"]))
                     for _, r in extprov.iterrows()})
_blocked = sorted({(str(r["cohort"]), _mod_of(r["feature"]))
                   for _, r in _absent.iterrows()})
_scored = [k for k in _attempted if k not in _blocked]
# the scored models and the rows of the results table are the same set of
# evaluations counted two ways; if they ever disagree one of the two is wrong
assert len(_scored) == len(ext), (
    f"{len(_scored)} scoreable model(s) in Feature_provenance but {len(ext)} "
    "row(s) in External_validation")
_arm_of = {str(r["cohort"]): str(col(r, "cohort_resembles_PREDIX_arm",
                                     "matched_PREDIX_arm"))
           for _, r in ext.iterrows()}
if _blocked:
    w("### Attempted and not scoreable")
    w()
    w(f"{nword(len(_attempted)).capitalize()} locked models were carried to "
      f"external data and {nword(len(_scored))} could be scored. The "
      f"{'others are' if len(_blocked) > 1 else 'remaining one is'} recorded "
      "here rather than dropped. Source: the `Feature_provenance` sheet of the "
      "same workbook.")
    w()
    rows = []
    for _c, _m in _blocked:
        _grp = [str(r["feature"]) for _, r in extprov.iterrows()
                if str(r["cohort"]) == _c and _mod_of(r["feature"]) == _m]
        rows.append([_c, MOD.get(_m, _m).replace("**", ""),
                     _arm_of.get(_c, "—"), cnt(len(_grp)),
                     ", ".join(_absent_features(_c, _m)),
                     "not scoreable — the locked model cannot be applied "
                     "without them, and a reduced signature is a different "
                     "model"])
    w(table(rows, ["External cohort", "Locked model", "Resembles PREDIX arm",
                   "Signature features", "Absent from the cohort", "Outcome"]))
    w()

# The verdict sentence is assembled from the workbooks: the count of cohorts,
# the AUROC of each, and the harmonisation scheme are all read off the table
# printed above rather than typed, so a cohort added to or removed from the
# analysis cannot leave this line miscounting the ones it describes.
_bits = []
for _c in ext["cohort"].unique():
    _d = ext[ext["cohort"] == _c]
    _lo = min(_f(v) for v in _d["external_AUROC"])
    _hi = max(_f(v) for v in _d["external_AUROC"])
    _p = max(_f(v) for v in _d["p_vs_chance_one_sided"])
    _rng = f"{_lo:.3f}" if abs(_hi - _lo) < 5e-4 else f"{_lo:.3f}–{_hi:.3f}"
    # "worst-case" is earned only when a cohort contributes more than one row.
    # With one row per cohort the maximum IS the P value, and calling it
    # worst-case would imply a spread that is not there.
    _plbl = "worst-case P" if len(_d) > 1 else "P"
    _bits.append(f"{_c} {_rng} ({_plbl} {pval(_p)})")
_all = [_f(v) for v in ext["p_vs_chance_one_sided"]]
# The harmonisation clause is READ OFF the workbook's own `harmonisation` column
# rather than typed, so that removing or restoring a scheme in
# external_validation.py can never leave this line claiming a scheme that was
# not run.
_schemes = sorted({str(v) for v in ext["harmonisation"] if v})
_HNAME = {"zscore": "z-score"}
_harm = (f"the {_HNAME.get(_schemes[0], _schemes[0])} harmonisation scheme"
         if len(_schemes) == 1 else
         ("both" if len(_schemes) == 2 else "all") + " harmonisation schemes")
_scope = "under " + _harm
_ncoh = len(ext["cohort"].unique())
_subject = ("The external cohort discriminates" if _ncoh == 1 else
            ("Both external cohorts discriminate" if _ncoh == 2 else
             f"All {nword(_ncoh)} external cohorts discriminate"))
if max(_all) < 0.05:
    _verdict = f"**{_subject} above chance, {_scope}.**"
    # the scheme is already named in the clause above; do not repeat it
    _pref = (" AUROC under that scheme: " if len(_schemes) == 1
             else " AUROC across those schemes: ")
else:
    _verdict = "**Not every external estimate is distinguishable from chance.**"
    _pref = (f" AUROC under {_harm}: " if len(_schemes) == 1
             else f" AUROC across {_harm}: ")
w(_verdict + _pref + "; ".join(_bits) + ".")
w()
w("Calibration is the honest qualifier, and it is reported separately from "
  "discrimination for exactly that reason: a frozen model can rank patients "
  "usefully in a cohort whose base rate and spread it mis-states. Read the "
  "calibration-slope column above — below 1 means the probabilities are more "
  "extreme than the cohort warrants, above 1 that they are compressed toward "
  "the base rate. No result is withheld on calibration grounds and none is "
  "presented as though calibration were settled.")
w()

# The forest plot draws exactly the rows tabulated above, so the cohort half of
# its caption is written from the same table instead of a typed list.
_scored_mod = dict(_scored)
FIG_CAPTION["revfig06_external_validation"] += (
    " Cohorts, with the locked model applied to each: "
    + "; ".join(f"{c}, {_modname(_scored_mod[c])} model of the "
                f"{_arm_of.get(c, '—')} arm"
                for c in ext["cohort"].unique() if c in _scored_mod)
    + ".")
if _blocked:
    FIG_CAPTION["revfig06_external_validation"] += (
        " The footnote records the model that could not be scored: "
        + "; ".join(f"the {_modname(_m)} model in {_c}, which does not measure "
                    + ", ".join(_absent_features(_c, _m))
                    for _c, _m in _blocked)
        + ".")

# ============================================================= 9. figures ===
# FIG_CAPTION decides what ships. The rendering loop below iterates it, so a PNG
# copied without a caption entry would sit in the deposit without ever appearing
# on this page — present, unexplained and counted by verify_github_repo.py. The
# copy is therefore caption-driven, and the destination is pruned first, because
# a rebuild overwrites files but does not remove the ones it no longer writes.
PNG_DST.mkdir(parents=True, exist_ok=True)
for _stale in sorted(PNG_DST.glob("*.png")):
    if _stale.stem not in FIG_CAPTION:
        _stale.unlink()
copied = 0
uncaptioned = []
for p in sorted(PNG_SRC.glob("*.png")):
    if p.stem not in FIG_CAPTION:
        uncaptioned.append(p.name)
        continue
    shutil.copy2(p, PNG_DST / p.name)
    copied += 1

w("## 9. Figures")
w()
w("PNG renderings at 170 dpi; the citable vector versions are the PDFs in "
  "[`report/figures/`](report/figures).")
w()


def fig_block(title, stems, note=""):
    w(f"### {title}")
    w()
    if note:
        w(note)
        w()
    for s in stems:
        f = PNG_DST / f"{s}.png"
        if not f.exists():
            continue
        w(f"#### {s}")
        w()
        w(f"![{s}](report/figures_png/{s}.png)")
        w()
        w(f"*{FIG_CAPTION.get(s, '')}*")
        w()


main = [s for s in FIG_CAPTION if s.startswith("fig")]
rev = [s for s in FIG_CAPTION if s.startswith("revfig")]
supp = [s for s in FIG_CAPTION if s.startswith("supp_fig")]
fig_block("Main figures", main)
fig_block("Revision figures", rev,
          "Calibration, stability, events per variable, external validation, "
          "paired comparisons and fusion weights.")
fig_block("Supplementary figures — discovery phase", supp,
          "Diagnostics of the fully nested discovery phase, before consensus "
          "finalisation.")

# A caption whose PNG never arrived renders as a heading with no image, and the
# build cannot tell whether the figure was renamed or simply not drawn. It is
# not fatal, so report it loudly instead of failing. The mirror case — a PNG
# with no caption — cannot happen: the copy above is caption-driven, and any
# such file is listed in `uncaptioned` and left out of the deposit.
dead_caption = sorted(s for s in FIG_CAPTION if not (PNG_DST / f"{s}.png").exists())

w("---")
w()
w("Regenerate this page with `python docs/build_RESULTS_md.py`.")

OUT.write_text("\n".join(M) + "\n", encoding="utf-8", newline="\n")
shutil.copy2(Path(__file__), REPO / "docs" / "build_RESULTS_md.py")

# The manifest must cover the files just added. Same rule as Section 14 of the
# notebook (the notebook and the manifest itself excluded, no header lines), so
# that re-running the notebook regenerates it byte-identically.
import hashlib
man = sorted((p for p in REPO.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and "_regenerated" not in p.parts and p.suffix != ".pyc"
              and not p.name.startswith((".~lock", "~$"))
              and p.name not in ("MANIFEST_SHA256.txt",
                                 "PREDIX_HER2_reproducibility.ipynb")),
             key=lambda p: p.relative_to(REPO).as_posix())


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


(REPO / "MANIFEST_SHA256.txt").write_text(
    "\n".join(f"{_sha(p)}  {p.relative_to(REPO).as_posix()}" for p in man) + "\n",
    encoding="utf-8", newline="\n")
print(f"  MANIFEST_SHA256.txt regenerated: {len(man)} files")
text = "\n".join(M)
n_tables = sum(1 for line in text.splitlines() if line.startswith("|---"))
n_figs = sum(1 for line in text.splitlines() if line.startswith("!["))
n_rows = sum(1 for line in text.splitlines()
             if line.startswith("|") and not line.startswith("|---"))
# --out may point outside this tree, in which case the path cannot be shown
# relative to it; print it whole rather than raising after the page is written
try:
    _shown = OUT.relative_to(ROOT)
except ValueError:
    _shown = OUT
print(f"wrote {_shown}  "
      f"{OUT.stat().st_size / 1024:.0f} KB, {len(M)} lines")
print(f"  {n_tables} tables ({n_rows - n_tables} data rows), "
      f"{n_figs} figures embedded, {copied} PNGs copied")
if uncaptioned:
    print(f"  {len(uncaptioned)} PNG(s) in {PNG_SRC.name}/ have no FIG_CAPTION "
          f"entry and were not copied: {uncaptioned}")
if dead_caption:
    print(f"  WARNING  {len(dead_caption)} FIG_CAPTION entr(ies) with no PNG: "
          f"{dead_caption}")
print(f"  external validation: {len(_scored)} model(s) scored, "
      f"{len(_blocked)} not scoreable, {len(_attempted)} attempted")
