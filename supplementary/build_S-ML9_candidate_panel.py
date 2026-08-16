"""Build Supplementary Table S-ML9: the complete candidate feature panel and the
fixed biological deduplication, with the |r| values recomputed from the data
rather than copied from the code comments."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\georg\Documents\claude_kang_multimodal_natcancer")
DATA = ROOT / "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
OUT = ROOT / "revision_deliverables" / "supp_table_S-ML9_candidate_panel.xlsx"

# (removed, stated counterpart, stated r, reason)
TIER1 = [
    ("DNA_PPP1R1B_CNA", "DNA_ERBB2_CNA", 1.000,
     "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    ("DNA_MIEN1_CNA", "DNA_ERBB2_CNA", 1.000,
     "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    ("DNA_GRB7_CNA", "DNA_ERBB2_CNA", 1.000,
     "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    ("DNA_CDK12_CNA", "DNA_ERBB2_CNA", 0.904, "17q12 amplicon"),
    ("DNA_CTTN_CNA", "DNA_PPFIA1_CNA", 1.000,
     "11q13 amplicon: co-amplifies with PPFIA1"),
    ("DNA_TMB_uniform", "DNA_TMB_clone_oncogenic", 0.975,
     "alternative parameterisation of the same mutational burden"),
    ("DNA_TMB_clone", "DNA_TMB_clone_oncogenic", 0.908,
     "duplicate column: the same values are carried by the retained metric"),
    ("DNA_pTMB", "DNA_TMB_clone_oncogenic", 0.903,
     "alternative parameterisation of the same mutational burden"),
    ("RNA_CD8-T-cells", "RNA_mRNA-CD8A", 0.984,
     "immune deconvolution near-identical to the CD8A transcript"),
    ("RNA_T-cells", "RNA_TILs", 0.972, "near-identical to the TIL score"),
    ("RNA_CD45", "RNA_TILs", 0.948, "near-identical to the TIL score"),
    ("RNA_Cytotoxic-cells", "RNA_mRNA-CD8A", 0.940,
     "near-identical to the CD8A transcript"),
    ("RNA_mRNA-ERBB2", "RNA_HER2DX_HER2_amplicon", 0.959,
     "subsumed by the validated HER2DX HER2-amplicon composite score"),
]

# Presentation categories, assigned from the feature names. Prefix match, first
# hit wins; the catch-all per modality is last.
CATEGORY_RULES = [
    ("Clin_", None, "Clinical"),
    ("RNA_HER2DX", None, "Transcriptomics — validated composite signature"),
    ("RNA_sspbc", None, "Transcriptomics — validated composite signature"),
    ("RNA_pik3ca_sig", None, "Transcriptomics — validated composite signature"),
    ("RNA_Taxane", None, "Transcriptomics — validated composite signature"),
    ("RNA_ADC_traffick", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Endocytosis", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Lysosome", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Exosome", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Oxidative", None, "Transcriptomics — metabolic programme"),
    ("RNA_Glycolysis", None, "Transcriptomics — metabolic programme"),
    ("RNA_Fatty_acid", None, "Transcriptomics — metabolic programme"),
    ("RNA_Glutathione", None, "Transcriptomics — metabolic programme"),
    ("RNA_mRNA-", None, "Transcriptomics — single transcript"),
    ("RNA_FCGR3", None, "Transcriptomics — single transcript"),
    ("RNA_", None, "Transcriptomics — immune microenvironment"),
    ("DNA_coding_mutation", None, "Genomics — coding mutation (gene or pathway)"),
    ("DNA_COSMIC", None, "Genomics — COSMIC mutational signature"),
    (None, "_CNA", "Genomics — copy number at a recurrently altered locus"),
    ("DNA_", None, "Genomics — burden / immunogenomic metric"),
    ("Prot_", None, "Proteomics"),
    ("WSI_", None, "Whole-slide image — spatial metric"),
]

PROT_TRAFFICKING = {"Prot_RAB11FIP1", "Prot_RAB11B", "Prot_RAB5C", "Prot_EEA1",
                    "Prot_ARL1", "Prot_FLOT1", "Prot_VAMP3", "Prot_SLC12A2"}


def category(col):
    if col in PROT_TRAFFICKING:
        return "Proteomics — endosomal / vesicular trafficking machinery"
    if col.startswith("Prot_"):
        return "Proteomics — 17q12 / 11q13 amplicon protein"
    for pre, suf, cat in CATEGORY_RULES:
        if pre is not None and col.startswith(pre):
            return cat
        if suf is not None and col.endswith(suf) and col.startswith("DNA_"):
            return cat
    return "unclassified"


df = pd.read_csv(DATA, sep="\t")
cols = [c for c in df.columns if c not in ("patientID", "pCR")]
cc = df.dropna(subset=cols + ["pCR"])
print(f"file: {df.shape[0]} patients x {df.shape[1]} columns; "
      f"{len(cols)} features; complete case n = {len(cc)}")

removed = {t[0] for t in TIER1}

# ---- sheet 1: the panel ------------------------------------------------------
rows = []
for c in cols:
    mod = c.split("_", 1)[0]
    rows.append({
        "modality": {"Clin": "Clinical", "RNA": "Transcriptomics",
                     "DNA": "Genomics", "Prot": "Proteomics",
                     "WSI": "Whole-slide image"}.get(mod, mod),
        "feature": c,
        "category (assigned for presentation)": category(c),
        "removed by the fixed biological deduplication": "yes" if c in removed else "no",
        "enters the cross-validation fold loop": "no" if c in removed else "yes",
    })
panel = pd.DataFrame(rows)
assert (panel["category (assigned for presentation)"] != "unclassified").all(), \
    panel[panel["category (assigned for presentation)"] == "unclassified"]

# ---- sheet 2: the deduplication, with r recomputed ---------------------------
ded = []
for feat, keep, stated_r, reason in TIER1:
    present = feat in df.columns
    kpres = keep in df.columns
    r_cc = r_all = np.nan
    if present and kpres:
        a, b = cc[feat].astype(float), cc[keep].astype(float)
        if a.std() > 0 and b.std() > 0:
            r_cc = abs(float(np.corrcoef(a, b)[0, 1]))
        sub = df[[feat, keep]].dropna().astype(float)
        if len(sub) > 2 and sub[feat].std() > 0 and sub[keep].std() > 0:
            r_all = abs(float(np.corrcoef(sub[feat], sub[keep])[0, 1]))
    ded.append({
        "removed feature": feat,
        "reason": reason,
        "retained instead": keep,
        "|r| recorded when the list was written (earlier data release)": stated_r,
        "|r| recomputed, complete case (n=%d)" % len(cc): round(r_cc, 3) if np.isfinite(r_cc) else "not computable",
        "|r| recomputed, all patients with both measured": round(r_all, 3) if np.isfinite(r_all) else "not computable",
        "present in the analysis file": "yes" if present else "no (already absent)",
    })
ded = pd.DataFrame(ded)

print("\nDEDUPLICATION — stated vs recomputed |r|:")
print(ded.to_string(index=False))

n_removed_present = sum(1 for t in TIER1 if t[0] in df.columns)
print(f"\npanel: {len(panel)} metrics; deduplication removes "
      f"{n_removed_present} present of {len(TIER1)} listed -> "
      f"{len(panel) - n_removed_present} candidates")

by_mod = panel.groupby("modality").size()
print("\nby modality:\n", by_mod.to_string())

hdr1 = ("SUPPLEMENTARY TABLE S-ML9a. The complete candidate feature panel "
        "(a-priori biological curation). No outcome information was used to "
        "assemble this panel. The 'category' column is assigned for "
        "presentation from the feature naming and should be checked by the "
        "authors against the assay documentation.")
hdr2 = ("SUPPLEMENTARY TABLE S-ML9b. The fixed, pre-specified biological "
        "deduplication applied before any train/test split (TIER1_REMOVE in "
        "multimodal_pcr_pipeline.py; disabled by --feature_pool full). "
        "Correlations are between features only; the pCR label is never used.")

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    pd.DataFrame({hdr1: []}).to_excel(xw, sheet_name="Candidate_panel", index=False)
    panel.to_excel(xw, sheet_name="Candidate_panel", index=False, startrow=2)
    pd.DataFrame({hdr2: []}).to_excel(xw, sheet_name="Deduplication", index=False)
    ded.to_excel(xw, sheet_name="Deduplication", index=False, startrow=2)
print("\nwrote", OUT)
