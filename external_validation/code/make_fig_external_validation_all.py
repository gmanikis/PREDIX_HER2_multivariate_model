#!/usr/bin/env python3
"""The external validation, all validation sets, one figure and one table.

Every row is the SAME procedure: the locked signature, classifier and
hyper-parameters from the locked run's results/{dhp,tdm1}, fitted once on the
corresponding PREDIX arm cohort and applied to the external set. No feature
selection, no tuning, no cross-validation of the model.

    I-SPY2         transcriptomic   DHP model
    NCT02326974    transcriptomic   T-DM1 model
    TransNEO       transcriptomic   DHP model
    TransNEO       genomic          DHP model      <- not validatable; see README

Everything is read from locked_signatures_external_validation.csv, which
apply_locked_external_validation.py writes. Nothing is recomputed or retyped here
- including the rendering of the p-values, which is stored in that CSV so that
this figure, revfig06 and the results page cannot show three different strings
for one number.

THE INTERNAL COMPARATOR is the locked model's own cross-validated AUROC
model, read from its performance table. It is measured on the complete-case
EVALUATION cohort (59 DHP, 51 T-DM1) — where the pipeline draws its outer test
sets so the modality comparisons stay paired — while the external figure is that
model on the external cohort. The two differ in population as well as in setting.
That is stated rather than glossed, and it is uniform across all four rows, so
the rows remain comparable with each other.

USAGE
  python make_fig_external_validation_all.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE / "locked_signatures_external_validation.csv"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})

C_EXT = "#2166ac"      # external estimate, transcriptomic
C_DNA = "#1b7837"      # genomic, so it cannot be mistaken for a transcriptomic row
C_INT = "#9ecae1"      # internal comparator


def ci_pair(s):
    """'0.481-0.678' or '0.774 [0.622–0.904]' -> (lo, hi); NaN if unparseable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (np.nan, np.nan)
    t = str(s).replace("–", "-").replace("—", "-")
    if "[" in t:
        t = t[t.index("[") + 1:t.index("]")]
    t = t.strip()
    neg = t.startswith("-")
    parts = [p for p in (t[1:] if neg else t).replace(" ", "").split("-") if p]
    try:
        lo = float(parts[0]) * (-1 if neg else 1)
        return (lo, float(parts[1]))
    except (ValueError, IndexError):
        return (np.nan, np.nan)


def skipped_note(skipped):
    """The sentence describing every evaluation that was attempted and skipped.

    It is built from the CSV's own applicable=False rows, so it cannot describe
    a row that is not there or miss one that is. Until this existed, the count
    of skipped evaluations went to stdout and reached no artefact at all: the
    figure showed three rows and a reader had no way to tell whether a fourth
    had been attempted, had failed, or had never been considered.
    """
    lines = []
    for _, s in skipped.iterrows():
        miss = [m.strip() for m in str(s["missing_features"]).split(";")
                if m.strip()]
        lines.append(
            f"{s['cohort']} {str(s['modality']).lower()} "
            f"({str(s['PREDIX_model']).split(' ')[0]} model): NOT VALIDATABLE — "
            f"{len(miss)} of the {int(s['K'])} signature features are not "
            f"measured in that cohort ({', '.join(miss)}).")
    return "\n".join(lines)


def draw(d, skipped, out_stem):
    n = len(d)
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * n + 2.3))
    ypos = np.arange(n)[::-1]

    for y, (_, r) in zip(ypos, d.iterrows()):
        col = C_DNA if r["modality"] == "Genomic" else C_EXT
        lo, hi = ci_pair(r["external_AUROC_CI"])
        ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([r["external_AUROC"]], [y], "o", color=col, ms=7.5, zorder=4)

        ilo, ihi = ci_pair(r["internal_AUROC_CI"])
        if ilo == ilo:
            ax.plot([ilo, ihi], [y + 0.24, y + 0.24], color=C_INT, lw=1.8,
                    solid_capstyle="round", zorder=2)
            ax.plot([r["internal_AUROC"]], [y + 0.24], "s", color=C_INT,
                    ms=5.5, zorder=2)

        # The p rendering is READ from the CSV, not formatted here: it is written
        # once by the builder, so this figure, revfig06 and the results page show
        # one string per row. Values at the bootstrap resolution floor are
        # inequalities there rather than point estimates.
        p = float(r["external_AUROC_p_vs_chance_one_sided"])
        ptxt = str(r["external_AUROC_p_vs_chance_display"])
        ax.text(1.012, y, f"{r['external_AUROC']:.3f} "
                          f"[{lo:.3f}–{hi:.3f}]   {ptxt}"
                          f"{'' if p < 0.05 else '   n.s.'}",
                va="center", ha="left", fontsize=8,
                color="black" if p < 0.05 else "#7a7a7a")

    ax.axvline(0.5, color="#999999", ls=":", lw=1.1, zorder=1)
    ax.text(0.5, n - 0.35, "chance", ha="center", va="bottom", fontsize=7.5,
            color="#777777")

    # int(), not the raw cell: the non-applicable row leaves these blank, which
    # types the whole column float64, and the axis then reads "n=44.0, 26.0 pCR".
    labels = [f"{r['cohort']}\n{r['modality']} · "
              f"{r['PREDIX_model'].split(' ')[0]} model · "
              f"n={int(r['n_external'])}, {int(r['events_external'])} pCR"
              for _, r in d.iterrows()]
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    # The lower bound reserves an empty band under the last row for the legend.
    ax.set_ylim(-1.9, n - 0.15)
    ax.set_xlim(0.35, 1.0)
    ax.set_xlabel("AUROC in the external cohort (95% CI)")
    ax.set_title("Locked-model external validation, arm-matched design\n"
                 f"{len(d) + len(skipped)} evaluations attempted; "
                 f"the {len(d)} that can be scored",
                 loc="left")

    # The legend describes what is DRAWN. A permanent genomic handle put a green
    # swatch beside a figure with no green row and no explanation of the absence;
    # the genomic evaluation is now stated in the note below the axes instead,
    # in the same colour, so nothing is silently dropped and nothing is claimed.
    h = [plt.Line2D([], [], color=C_EXT, lw=2.4, marker="o", ms=7,
                    label="External (transcriptomic)")]
    if (d["modality"] == "Genomic").any():
        h.append(plt.Line2D([], [], color=C_DNA, lw=2.4, marker="o", ms=7,
                            label="External (genomic)"))
    h.append(plt.Line2D([], [], color=C_INT, lw=1.8, marker="s", ms=5.5,
                        label="Internal: cross-validated AUROC of the same model"))
    ax.legend(handles=h, loc="lower left", frameon=False, ncol=1, fontsize=7.5,
              handlelength=1.8, borderaxespad=0.3)

    y_note = -0.03
    if len(skipped):
        fig.text(0.0, y_note, skipped_note(skipped), fontsize=7.5,
                 color=C_DNA, va="top")
        y_note -= 0.035 * (1 + skipped_note(skipped).count("\n"))

    fig.text(0.0, y_note - 0.02,
             "Every row applies the locked signature, classifier and "
             "hyper-parameters, fitted once on the corresponding PREDIX arm "
             "cohort. No feature selection, no tuning,\nno cross-validation of "
             "the model. The internal comparator is measured on the PREDIX "
             "complete-case evaluation cohort (59 DHP, 51 T-DM1), a different "
             "population from\nthe external one. Features are standardised "
             "within each cohort by z-score. P is a one-sided bootstrap tail "
             "probability against chance; its\nresolution floor over 2000 draws "
             "is about 0.0005, so rows at the floor are shown as an inequality.",
             fontsize=7, color="#555555", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}")
    plt.close(fig)


def main():
    if not CSV.exists():
        sys.exit(f"{CSV} not found — run apply_locked_external_validation.py first")
    d = pd.read_csv(CSV)
    for col in ("external_AUROC_p_vs_chance_display",):
        if col not in d.columns:
            sys.exit(f"{CSV} carries no {col} column. The p-value rendering is "
                     "written once, by apply_locked_external_validation.py, so "
                     "that every artefact shows the same string; rebuild the "
                     "CSV rather than reformatting the number here.")
    _all = d.copy()
    d = d[d["applicable"]].reset_index(drop=True)
    # Three applicable sets, not four. The genomic validation is not possible
    # because the DHP genomic signature (ERBB2/MED1/NCOR1/RPL19 CNA) is not
    # measurable in TransNEO, whose genomic file carries a different set of
    # genomic metrics. It is carried in the CSV as applicable=False with its
    # missing features named, so the count is asserted against BOTH parts: three
    # drawn, and every non-drawn row must say why it was not drawn. A bare
    # "== 3" would pass just as happily on a row silently lost.
    _skipped = _all[~_all["applicable"].astype(bool)].reset_index(drop=True)
    if len(d) != 3:
        sys.exit(f"expected 3 applicable validation sets, got {len(d)}")
    if len(_skipped) and not _skipped["missing_features"].astype(str).str.strip().all():
        sys.exit("a non-applicable row does not name its missing features")
    print(f"drawing {len(d)} validation sets; "
          f"{len(_skipped)} recorded as not applicable")
    d["internal_to_external_drop"] = (d["internal_AUROC"]
                                      - d["external_AUROC"]).round(4)

    # The non-validatable evaluation, written from the CSV rather than typed, so
    # the workbook states what the CSV holds.
    _not_val = "; ".join(
        f"{s['cohort']} {str(s['modality']).lower()} model, missing "
        f"{str(s['missing_features']).replace('; ', ', ')}"
        for _, s in _skipped.iterrows()) or "none"

    xlsx = HERE / "external_validation_all_cohorts.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        note = pd.DataFrame({"External validation — all validation sets": [
            "Every row is the ARM-MATCHED locking design and the SAME procedure: "
            "the locked signature, classifier and hyper-parameters, fitted "
            "once on the corresponding PREDIX arm cohort, applied to the external "
            "set. No feature selection, no tuning, no cross-validation of the model.",
            "The model is the one the manuscript reports, taken from "
            "the locked run's results/{dhp,tdm1}: its signature, classifier and "
            "hyper-parameters are read from the locked artefacts and used "
            "unchanged. Each external cohort is scored once by the arm-matched "
            "locked model. No signature is re-derived inside an external cohort, "
            "no feature is dropped to make a model fit, and nothing is refitted "
            "on external data.",
            f"{len(_all)} evaluations were attempted and {len(d)} are applicable. "
            f"The rest are recorded in locked_signatures_external_validation.csv "
            f"with applicable = FALSE and their missing features named, not "
            f"dropped: {_not_val}.",
            "internal_AUROC is the locked model's cross-validated AUROC of the same locked "
            "model, measured on the complete-case EVALUATION cohort (59 DHP, 51 "
            "T-DM1). The external figure is that model on the external cohort. "
            "The two differ in population as well as in setting; the difference "
            "is uniform across rows, so the rows compare with each other.",
            "Features are standardised within each cohort by z-score before the "
            "locked model is applied. That step is unsupervised but transductive.",
            "TransNEO received chemotherapy plus HER2-targeted therapy: DHP-like, "
            "not a pertuzumab-matched replicate of the PREDIX DHP arm.",
            "Only the transcriptomic models can be validated. No external cohort "
            "we hold carries clinical, proteomic or whole-slide-image metrics, "
            "and the genomic model cannot be scored either: three of its four "
            "signature features are absent from the one external genomic metric "
            "file we hold.",
            "probability_calibration records whether the Brier score and the "
            "calibration slope came from Platt-scaled or from raw probabilities. "
            "AUROC is the same either way, Platt scaling being monotone.",
            "external_AUROC_p_vs_chance_one_sided is a bootstrap tail probability "
            "with a resolution floor of about 0.0005 over 2000 draws; the "
            "_display column renders a row at that floor as an inequality rather "
            "than as a point estimate, and the figures print that same string.",
        ]})
        note.to_excel(xw, sheet_name="External_validation", index=False)
        d.to_excel(xw, sheet_name="External_validation", index=False,
                   startrow=len(note) + 2)
    draw(d, _skipped, str(HERE / "fig_external_validation_all_cohorts"))

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(d[["cohort", "modality", "PREDIX_model", "classifier", "K",
             "n_PREDIX_fit", "internal_AUROC", "n_external", "events_external",
             "external_AUROC", "external_AUROC_CI",
             "external_AUROC_p_vs_chance_display",
             "internal_to_external_drop",
             "calibration_slope"]].to_string(index=False))
    if len(_skipped):
        print("\nnot applicable:")
        print(_skipped[["cohort", "modality", "K",
                        "missing_features"]].to_string(index=False))
    print(f"\n[WRITE] {xlsx}")
    print(f"[WRITE] {HERE / 'fig_external_validation_all_cohorts'}.pdf / .png")


if __name__ == "__main__":
    main()
