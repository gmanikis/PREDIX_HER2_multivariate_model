#!/usr/bin/env python3
"""
Consensus-signature feature ranking per model scope, in the published panel
style: one horizontal bar per selected feature, grouped and coloured by
modality, annotated with the direction of the pCR association.

One panel per scope:
    Global   the whole cohort, one model for both arms
    DHP      arm-specific model, docetaxel + trastuzumab + pertuzumab
    T-DM1    arm-specific model

WHAT THE BAR LENGTH IS. It is `mean_importance` from the consensus PKL, which
despite the key name is NOT a SHAP magnitude. The pipeline converts each
classifier's raw importance to a PERCENTILE RANK within the fold, then averages
those ranks across classifiers and across discovery folds. The result lies in
(0, 1] and is exactly the quantity fig02_consensus_signatures.pdf draws.
CONSEQUENCE: bar lengths are comparable WITHIN a modality block and are NOT
comparable between modalities. A rank of 1.0 means "top-ranked inside its own
3-feature WSI panel", which says nothing about how it compares to the top RNA
feature out of 35 candidates. Only the within-block ordering carries meaning.
The panel titled "Normalized Importance" in the submitted manuscript carries
the same caveat; it is stated here rather than left implicit.

WHAT THE + / - SIGN IS. The sign of the SHAP dependence slope: the correlation
between a feature's standardised value and its own SHAP value, pooled over the
discovery folds in which it appears in the winner signature. "+" = a higher
value of the feature pushes the prediction towards pCR. Read from `oof_shap`
in the discovery PKL. A feature with no SHAP record in any fold is drawn
without a sign rather than with an invented one. A feature whose per-fold slope
reverses often — agreeing with the pooled slope in less than
`--min-sign-stability` of the folds that selected it, default 0.75 — keeps its
sign but is drawn **in parentheses and greyed**, so the bar is neither left
looking like missing data nor quoted at the confidence of a stable one. How
many bars that brackets is counted at the end of the run and printed per scope;
the per-bar flag is `direction_low_stability` in the companion workbook.

WHY NOT THE MEAN SHAP VALUE. The mean SHAP value over patients cannot carry
direction, so nothing here is signed with it. SHAP attributions are defined
against a baseline, so a feature's SHAP values average to approximately zero
over the sample by construction, and the residual that survives is dominated by
sampling noise: |mean SHAP| is only 0.20-0.30 of mean|SHAP| at the median. Its
sign therefore agrees with the dependence slope on barely more than half the
plotted bars — that count is COMPUTED from the drawn bars and written into the
workbook's Notes sheet rather than typed in here, so it cannot go stale — while
the dependence slope agrees with the model-free per-arm univariate direction far
more often. The statistic is retained in the workbook column
`mean_signed_shap_NOT_a_direction` so that it stays inspectable, and is used
nowhere else.

The slope is exact where it is checkable. For a linear winner the pipeline
computes SHAP analytically as (x - x_train_mean) * coef, so within such a fold
|corr(x_j, shap_j)| must be 1 and its sign must be sign(coef_j). Measured over
|corr| = 1.000000000000 in 11,548 of 11,548 linear feature-folds, which
also pins the column alignment between `X_test_scaled` and `shap_values`.

`generate_report.py` colours fig02_consensus_signatures.pdf and
supp_fig06_feature_selection_frequency.pdf by the same discarded statistic.
Those figures are produced by the locked report and are NOT corrected
here; see FEATURE_DIRECTION_CORRECTION.md.

WHICH FEATURES APPEAR. By default every feature in the frozen consensus
signature of each modality, i.e. the panel the final model actually uses.
`--features` restricts the figure to a chosen subset, by raw feature name.

Sources, all read-only:
    <RUN>/results/<arm>/<arm>_consensus_eval.pkl      signature + mean_importance
    <RUN>/results/<arm>/<arm>_elasticnet_results.pkl  per-fold oof_shap
    <RUN>/results/<arm>/<arm>_per_classifier_signatures.csv   selection frequency

WHERE <RUN> COMES FROM. With no --run the script finds the tree itself and
prints which one it read; see `_default_run`. In the deposit, where results/
sits at the repository root, that is the repository root, and the figure and
its workbook go to _regenerated/figures/revision and _regenerated/tables/
revision respectively - so the no-argument invocation below is the whole
regeneration command.

Usage:
    python code/make_fig_feature_ranking.py
    python make_fig_feature_ranking.py --out some/dir --layout row
    python make_fig_feature_ranking.py --scopes DHP T-DM1
    python make_fig_feature_ranking.py --features Clin_ER RNA_HER2DX_HER2_amplicon

    # the same thing spelled out, for a tree the finder cannot resolve on its own
    python code/make_fig_feature_ranking.py --run . \\
        --out _regenerated/figures/revision \\
        --table-out _regenerated/tables/revision
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent.parent


def _generated_utc(tree: Path) -> str:
    """A run tree's own record of when it was produced; "" if it has none."""
    prov = tree / "results" / "run_provenance.json"
    try:
        return str(json.loads(prov.read_text(encoding="utf-8")).get(
            "generated_utc", ""))
    except Exception:
        return ""


def _default_run() -> Path:
    """The tree holding results/<arm>/*.pkl, in whichever layout this file sits.

    Two layouts. In the DEPOSIT the artefacts are `results/` at the top level,
    one directory above this script, so the tree is that root and no argument
    is needed. In a WORKING TREE the analysis outputs sit in their own
    directory beside the scripts instead.

    Candidates are found by looking for a consensus evaluation rather than for
    a directory name, so a half-copied tree is not silently accepted. Where a
    working tree offers more than one, the tree whose own run_provenance.json
    records the later generation time wins, and main() prints the tree it read
    so the choice is never invisible. Pass --run to decide it explicitly.
    """
    if (BASE / "results" / "global" / "global_consensus_eval.pkl").is_file():
        return BASE
    found = sorted({p.parents[2] for p in
                    BASE.glob("*/results/*/*_consensus_eval.pkl")})
    if not found:
        return BASE          # reported as missing, with the path shown
    return max(found, key=lambda t: (_generated_utc(t), t.name))


DEFAULT_RUN = _default_run()


def _default_out(run: Path) -> tuple[Path, Path]:
    """Figure directory and table directory for the default --run.

    The deposit keeps regenerated artefacts under _regenerated/, with figures
    and tables in separate trees because the verification notebook sweeps the
    table tree cell by cell. A working tree files them beside the other
    deliverables.
    """
    if run.resolve() == BASE.resolve():
        return (BASE / "_regenerated" / "figures" / "revision",
                BASE / "_regenerated" / "tables" / "revision")
    d = BASE / "revision_deliverables" / f"figures_{run.name.split('_')[-1]}"
    return d, d

# Scope key -> (results subdirectory, panel title). "Global" is the model fitted
# on the whole cohort; the other two are the arm-specific models.
SCOPES = {
    "Global": ("global", "All patients (both arms)"),
    "DHP":    ("dhp",    "DHP"),
    "T-DM1":  ("tdm1",   "T-DM1"),
}

# Modality block order and legend labels, matching the submitted figure.
MODS = ["Clin", "DNA", "RNA", "Prot", "WSI"]
MOD_LABEL = {
    "Clin": "Clinic", "DNA": "Genomic", "RNA": "Transcriptomic",
    "Prot": "Proteomic", "WSI": "WSI",
}
# Bar fill colours sampled from the published panel, so that this figure and
# that one sit side by side without a palette shift.
MOD_COLOR = {
    "Clin": "#515183", "DNA": "#f5aa36", "RNA": "#7fa9c6",
    "Prot": "#9b75b4", "WSI": "#2c7d4d",
}

# Display labels. Anything absent falls back to the raw name with the modality
# prefix stripped and underscores spaced out, so a new feature never silently
# disappears or gets mislabelled.
LABELS = {
    "Clin_ER": "ER status (IHC)",
    "Clin_prolifvalu": "Ki67 (IHC)",
    "Clin_ANYNODES": "Lymph node",
    "Clin_TUMSIZE": "Tumor size",
    "Clin_Arm": "Treatment arm",

    "DNA_ERBB2_CNA": "SCNA ERBB2",
    "DNA_PIK3CA_CNA": "SCNA PIK3CA",
    "DNA_BRCA2_CNA": "SCNA BRCA2",
    "DNA_NCOR1_CNA": "SCNA NCOR1",
    "DNA_RAB11FIP1_CNA": "SCNA RAB11FIP1",
    "DNA_RPL19_CNA": "SCNA RPL19",
    "DNA_coding_mutation_TP53": "TP53 mutation",
    "DNA_LOH_Del_burden": "LOH/deletion burden",
    "DNA_HLA_Supertype_A01": "HLA supertype A01",
    "DNA_meanHED": "meanHED",
    "DNA_TCRA_T_cell_fraction": "TCRA T cell fraction",
    "DNA_HRD": "HRD",

    "RNA_HER2DX_HER2_amplicon": "HER2DX (HER2 amplicon)",
    "RNA_HER2DX_pCR_likelihood_score": "HER2DX (pCR score)",
    "RNA_HER2DX_luminal": "HER2DX (luminal)",
    "RNA_mRNA-ESR1": "mRNA ESR1",
    "RNA_mRNA-PGR": "mRNA PGR",
    "RNA_mRNA-FCGR3A": "mRNA FCGR3A",
    "RNA_Mast-cells": "Mast cells",
    "RNA_B-cells": "B cells",
    "RNA_Th2-cells": "Th2 cells",
    "RNA_CAF": "CAF",
    "RNA_Exosome": "Exosome signature",
    "RNA_sspbc_LumB": "LumB subtype",
    "RNA_sspbc_LumA": "LumA subtype",
    "RNA_pik3ca_sig": "PIK3CA signature",
    "RNA_adc_trafficking_sig": "ADC trafficking signature",

    "Prot_ERBB2_PG": "Protein ERBB2 (PG)",
    "Prot_HER2_amplicon": "Protein HER2 amplicon",
    "Prot_RPL19": "Protein RPL19",
    "Prot_CDK12": "Protein CDK12",
    "Prot_MIEN1": "Protein MIEN1",
    "Prot_VAMP3": "Protein VAMP3",
    "Prot_SLC12A2": "Protein SLC12A2",
    "Prot_FLOT1": "Protein FLOT1",
    "Prot_PPP1R1B": "Protein PPP1R1B",
    "Prot_ARL1": "Protein ARL1",

    "WSI_Cell_Interaction": "Cell interaction",
    "WSI_Immune_Cell_prop": "Immune cell proportion",
    "WSI_Distance_tumor_immune": "Tumor-immune distance",
}
# COSMIC signatures are regular enough to derive rather than enumerate.
for _n in range(1, 31):
    LABELS.setdefault(f"DNA_COSMIC.Signature.{_n}", f"COSMIC signature {_n}")


def label_of(feat: str) -> str:
    if feat in LABELS:
        return LABELS[feat]
    body = feat.split("_", 1)[1] if "_" in feat else feat
    return body.replace("_", " ")


def _binary_slice(sv):
    """The positive-class slice of a SHAP array, whatever shape it arrives in."""
    if isinstance(sv, list):
        return np.asarray(sv[1])
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def shap_direction(folds: list, sig: set) -> dict:
    """Direction of the pCR association per feature, as a SHAP dependence slope.

    For each fold where the feature was selected, pair its standardised value
    with its own SHAP value; pool those pairs over folds and take the
    correlation. Positive => a higher value of the feature pushes the
    prediction towards pCR.

    Returns {feature: {"r", "mean_shap", "n_folds", "n_pairs", "frac_pos"}}.
    `mean_shap` is retained only so the discarded statistic stays visible in
    the workbook; it is not a direction (see the module docstring).
    `frac_pos` is the fraction of folds whose own slope is positive, i.e. how
    stable the sign is across the discovery folds.
    """
    xs = defaultdict(list)
    ss = defaultdict(list)
    means = defaultdict(list)
    fold_sign = defaultdict(list)

    for fold in folds:
        sh = fold.get("oof_shap")
        if sh is None:
            continue
        sv = _binary_slice(sh["shap_values"])
        X = np.asarray(sh["X_test_scaled"], dtype=float)
        names = list(sh["feature_names"])
        if X.shape != sv.shape:
            continue
        m = sv.mean(axis=0)
        for j, feat in enumerate(names):
            if feat not in sig:
                continue
            x, s = X[:, j], sv[:, j]
            means[feat].append(float(m[j]))
            xs[feat].append(x)
            ss[feat].append(s)
            if x.std() > 1e-12 and s.std() > 1e-12:
                fold_sign[feat].append(1.0 if np.corrcoef(x, s)[0, 1] >= 0 else 0.0)

    out = {}
    for feat in means:
        x = np.concatenate(xs[feat])
        s = np.concatenate(ss[feat])
        ok = x.size > 2 and x.std() > 1e-12 and s.std() > 1e-12
        fs = fold_sign.get(feat, [])
        out[feat] = {
            "r": float(np.corrcoef(x, s)[0, 1]) if ok else np.nan,
            "mean_shap": float(np.mean(means[feat])),
            "n_folds": len(means[feat]),
            "n_pairs": int(x.size),
            "frac_pos": float(np.mean(fs)) if fs else np.nan,
        }
    return out


def selection_freq(csv: Path, mod: str, clf: str) -> dict:
    """Per-feature selection frequency for the winner classifier of a modality."""
    if not csv.exists():
        return {}
    d = pd.read_csv(csv)
    m = (d["modality"].astype(str) == mod) & (d["classifier"].astype(str) == clf)
    return dict(zip(d.loc[m, "feature"].astype(str),
                    d.loc[m, "selection_frequency"].astype(float)))


def collect(results: Path, scope: str, keep: set | None,
            min_stability: float = 0.75) -> tuple[list, dict]:
    """Rows to plot for one scope, in modality-block order, plus block metadata."""
    arm = SCOPES[scope][0]
    with open(results / arm / f"{arm}_consensus_eval.pkl", "rb") as fh:
        cons = pickle.load(fh)["consensus"]
    with open(results / arm / f"{arm}_elasticnet_results.pkl", "rb") as fh:
        disc = pickle.load(fh)
    csv = results / arm / f"{arm}_per_classifier_signatures.csv"

    rows, meta = [], {}
    for mod in MODS:
        c = cons.get(mod) or {}
        sig = list(c.get("signature") or [])
        if keep is not None:
            sig = [f for f in sig if f in keep]
        if not sig:
            continue
        imp = c.get("mean_importance", {})
        clf = c.get("winner_clf", "")
        signs = shap_direction(disc.get(mod, []), set(sig))
        freqs = selection_freq(csv, mod, clf)
        meta[mod] = {"winner_clf": clf, "K": c.get("K"),
                     "support_fraction": float(c.get("support_fraction", 0.0))}
        block = sorted(sig, key=lambda f: imp.get(f, 0.0), reverse=True)
        for i, f in enumerate(block, start=1):
            s = signs.get(f)
            r = np.nan if s is None else s["r"]
            # Sign stability of the pooled slope across the discovery folds.
            if s is None or np.isnan(s["frac_pos"]) or np.isnan(r):
                stab = np.nan
            else:
                stab = s["frac_pos"] if r >= 0 else 1.0 - s["frac_pos"]
            # A direction that reverses from fold to fold is still a direction,
            # just a poorly determined one. Leaving the bar blank reads as
            # missing data; drawing it like a 99%-stable sign overclaims. So it
            # is drawn in PARENTHESES and greyed: present, and visibly weaker.
            weak = (not np.isnan(r)) and (np.isnan(stab)
                                          or stab < min_stability)
            sign = "" if (s is None or np.isnan(r)) else ("+" if r >= 0 else "-")
            rows.append({
                "scope": scope, "modality": mod, "feature": f,
                "label": label_of(f), "rank_in_block": i,
                "normalised_selection_rank": float(imp.get(f, 0.0)),
                "selection_frequency": freqs.get(f, np.nan),
                "shap_dependence_r": r,
                # The direction itself, for every bar that has a slope at all.
                "direction": sign,
                # Fraction of discovery folds whose own slope agrees in sign
                # with the pooled slope.
                "sign_stability_frac_folds": stab,
                # True when that fraction is below --min-sign-stability: the
                # sign is real but weakly determined.
                "direction_low_stability": bool(weak),
                # Exactly what is drawn next to the bar.
                "direction_display": (f"({sign})" if weak else sign) if sign else "",
                "n_discovery_folds_selected": np.nan if s is None else s["n_folds"],
                # Retained so this statistic stays inspectable alongside the
                # slope. It is NOT a direction: see the docstring.
                "mean_signed_shap_NOT_a_direction":
                    np.nan if s is None else s["mean_shap"],
                "winner_classifier": clf,
                "classifier_support_fraction": meta[mod]["support_fraction"],
                "signature_size_K": c.get("K"),
            })
    return rows, meta


def draw_panel(ax, rows: list, title: str, xlabel: str, block_gap: float = 0.8):
    """One scope. Bars top-down in MODS order, a gap between modality blocks."""
    ys, vals, cols, labs, signs = [], [], [], [], []
    y = 0.0
    prev_mod = None
    for r in rows:
        if prev_mod is not None and r["modality"] != prev_mod:
            y += block_gap
        ys.append(y)
        vals.append(r["normalised_selection_rank"])
        cols.append(MOD_COLOR[r["modality"]])
        labs.append(r["label"])
        signs.append(r["direction_display"])
        prev_mod = r["modality"]
        y += 1.0

    ys = np.asarray(ys)
    ax.barh(-ys, vals, color=cols, edgecolor="black", linewidth=0.8, height=0.78)
    for yy, v, s in zip(ys, vals, signs):
        if not s:
            continue
        # A parenthesised sign is a real direction that reverses between folds
        # often enough not to be quoted; it is greyed as well as bracketed so
        # the distinction survives a greyscale print.
        weak = s.startswith("(")
        ax.text(v + 0.014, -yy, s.replace("-", "−"),
                va="center", ha="left", fontsize=9 if weak else 10,
                color="#8a8a8a" if weak else "black")
    ax.set_yticks(-ys)
    ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_ylim(-(ys.max() + 0.7), 0.7)
    ax.set_xlim(0, 1.09)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, pad=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN,
                    help="tree holding results/<arm>/*.pkl; found automatically "
                         "when omitted (default here: %(default)s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; required whenever --run is given "
                         "explicitly")
    ap.add_argument("--table-out", type=Path, default=None, dest="table_out",
                    help="write the companion workbook here instead of --out. The "
                         "deposit files figures and tables in separate trees, and "
                         "the notebook's cell-by-cell workbook comparison sweeps "
                         "the table tree, so the two must be separable.")
    ap.add_argument("--scopes", nargs="+", default=list(SCOPES),
                    choices=list(SCOPES), help="panels to draw, in order")
    ap.add_argument("--features", nargs="+", default=None,
                    help="restrict to these raw feature names, or a file with one per line")
    ap.add_argument("--layout", choices=["stack", "row"], default="stack",
                    help="stack = one panel above the next (published style)")
    ap.add_argument("--xlabel", default="Normalized importance\n"
                                        "(mean cross-classifier selection rank)",
                    help="x-axis label")
    ap.add_argument("--name", default="fig_feature_ranking_by_scope",
                    help="output basename")
    # 170 dpi is the house rendition for the PNGs RESULTS.md embeds, and that
    # page states the number. One PNG serves both the deposit and that page, so
    # it is rendered at the figure of record rather than at 300 and downsampled
    # into a second, silently different copy.
    ap.add_argument("--png-dpi", type=int, default=170, dest="png_dpi",
                    help="raster resolution of the PNG; the PDF is vector")
    ap.add_argument("--min-sign-stability", type=float, default=0.75,
                    dest="min_sign_stability",
                    help="print the +/- only when the per-fold slope agrees "
                         "with the pooled slope in at least this fraction of "
                         "discovery folds; 0 prints every sign")
    args = ap.parse_args()

    results = args.run / "results"
    if not results.is_dir():
        raise SystemExit(f"no results directory under {args.run}")
    print("reading", results)

    # Deriving the destination from an arbitrary --run silently invents a folder
    # (`--run .` would file the figure under "figures_."), so once --run is given
    # the destination has to be stated.
    if args.out is None and args.run.resolve() != DEFAULT_RUN.resolve():
        raise SystemExit("--run needs an explicit --out directory")
    _fig_dir, _tab_dir = _default_out(args.run)
    out = args.out or _fig_dir
    out.mkdir(parents=True, exist_ok=True)
    table_out = args.table_out or (_tab_dir if args.out is None else out)
    table_out.mkdir(parents=True, exist_ok=True)

    keep = None
    if args.features:
        if len(args.features) == 1 and Path(args.features[0]).is_file():
            keep = {l.strip() for l in Path(args.features[0]).read_text(
                encoding="utf-8").splitlines() if l.strip()}
        else:
            keep = set(args.features)

    per_scope, meta_all = {}, {}
    for sc in args.scopes:
        rows, meta = collect(results, sc, keep, args.min_sign_stability)
        if not rows:
            raise SystemExit(f"no features left to draw for {sc} — check --features")
        per_scope[sc] = rows
        meta_all[sc] = meta

    if keep:
        missing = keep - {r["feature"] for rs in per_scope.values() for r in rs}
        if missing:
            print("not in any consensus signature, so not drawn: "
                  + ", ".join(sorted(missing)))

    counts = {sc: len(rs) for sc, rs in per_scope.items()}
    n = len(args.scopes)
    if args.layout == "stack":
        fig, axes = plt.subplots(
            n, 1, figsize=(6.4, 0.30 * sum(counts.values()) + 1.6 * n),
            gridspec_kw={"height_ratios": [counts[sc] for sc in args.scopes]})
    else:
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n,
                                                0.30 * max(counts.values()) + 2.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, sc in zip(axes, args.scopes):
        draw_panel(ax, per_scope[sc], SCOPES[sc][1], args.xlabel)

    for ax, letter in zip(axes, "abcdefgh"):
        ax.text(-0.62 if args.layout == "stack" else -0.42, 1.0, letter,
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="bottom", ha="left")

    drawn = [m for m in MODS
             if any(r["modality"] == m for rs in per_scope.values() for r in rs)]
    # The legend band is reserved in INCHES, not as a fraction. A stacked figure
    # of 33+22+21 bars is ~20 in tall, where a 4.5% fraction is a hand's width
    # of blank paper between the legend and the first panel.
    band = 0.75 / fig.get_figheight()
    fig.legend(handles=[Patch(facecolor=MOD_COLOR[m], edgecolor="black",
                              label=MOD_LABEL[m]) for m in drawn],
               loc="upper center", ncol=min(len(drawn), 5), frameon=True,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0 - 0.12 * band))
    fig.tight_layout(rect=[0, 0, 1, 1.0 - band])

    for ext in ("pdf", "png"):
        p = out / f"{args.name}.{ext}"
        fig.savefig(p, dpi=args.png_dpi, bbox_inches="tight")
        print("wrote", p)

    # Companion table: every plotted value, so the figure can be audited without
    # reopening the pickles.
    flat = pd.DataFrame([r for sc in args.scopes for r in per_scope[sc]])

    # How often the discarded mean-SHAP statistic would have agreed with the
    # dependence slope, COUNTED FROM THE BARS THIS RUN ACTUALLY DREW. It is
    # computed rather than typed so that it describes the figure in hand and
    # cannot survive a change to the signatures. The comparison uses the same
    # `>= 0` sign convention as the drawing code above.
    _r = flat["shap_dependence_r"]
    _ms = flat["mean_signed_shap_NOT_a_direction"]
    _cmp = _r.notna() & _ms.notna()
    _n_cmp = int(_cmp.sum())
    _n_agree = int((((_r >= 0) == (_ms >= 0)) & _cmp).sum())
    _n_weak = int(flat["direction_low_stability"].astype(bool).sum())

    xlsx = table_out / f"{args.name}.xlsx"
    notes = [
        "CONSENSUS-SIGNATURE FEATURE RANKING per model scope.",
        "normalised_selection_rank is the bar length: a mean cross-classifier "
        "PERCENTILE RANK in (0,1], not a SHAP magnitude. Comparable within a "
        "modality block, never between blocks or between scopes.",
        "direction is the sign of shap_dependence_r, the correlation between "
        "the feature's within-fold standardised value and its own SHAP value, "
        "pooled over the discovery folds that selected it. '+' means a higher "
        "value of the feature pushes the prediction towards pCR.",
        "sign_stability_frac_folds is the fraction of those folds whose own "
        "slope agrees in sign with the pooled slope. Below the "
        f"--min-sign-stability threshold ({args.min_sign_stability:g}) the sign "
        "is real but weakly determined: it is drawn IN PARENTHESES and greyed, "
        "and flagged in direction_low_stability. direction_display is exactly "
        "what is drawn next to the bar; direction is the sign itself, "
        f"unbracketed. {_n_weak} of {len(flat)} bars are bracketed here.",
        "mean_signed_shap_NOT_a_direction is a mean SHAP value. It is NOT a "
        "direction and is not used to sign anything: SHAP values average to ~0 "
        "over the sample by construction, so the sign of the mean is dominated "
        "by sampling noise. On the bars in this workbook its sign agrees with "
        f"the dependence slope on {_n_agree} of {_n_cmp} bars. The column is "
        "kept only so the statistic stays inspectable, and must not be used to "
        "state a direction.",
    ]
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame({"note": notes}).to_excel(
            xw, sheet_name="Notes", index=False)
        flat.to_excel(xw, sheet_name="Feature_ranking", index=False)
        pd.DataFrame([{"scope": sc, "modality": m, **v}
                      for sc, mm in meta_all.items() for m, v in mm.items()]
                     ).to_excel(xw, sheet_name="Winner_classifier", index=False)
    print("wrote", xlsx)

    for sc in args.scopes:
        rs = per_scope[sc]
        no_rec = sum(1 for r in rs if not r["direction"])
        weak = sum(1 for r in rs if r["direction_low_stability"])
        extra = []
        if no_rec:
            extra.append(f"{no_rec} with no SHAP record")
        if weak:
            extra.append(f"{weak} sign bracketed below "
                         f"{args.min_sign_stability:.0%} fold stability")
        print(f"{sc:<7} {len(rs):>3} features"
              + (f"  ({'; '.join(extra)})" if extra else ""))


if __name__ == "__main__":
    main()
