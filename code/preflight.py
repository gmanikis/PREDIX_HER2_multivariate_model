#!/usr/bin/env python3
"""Pre-flight check on the input file. Runs in seconds; stops the pipeline
before the multi-hour modelling step if the data is not what the analysis
assumes.

WHAT THE ANALYSIS ASSUMES. Every model — the five unimodal models and the
prediction-level fusion layer — is trained AND evaluated on the complete
multimodal cases, so the input file has to carry that cohort intact: 110
patients, 46 pCR events, 59 DHP and 51 T-DM1. The pipeline is pointed at that
cohort with `--training_data cc_only`, which must be passed explicitly because
the pipeline's own default for the flag is `expanded`. This script checks the
DATA, not the command line, so it cannot tell you the flag was passed; the
command line that defines the analysis is in production_run_ubuntu.sh.

WHICH CHECKS BLOCK AND WHICH ONLY REPORT. `check()` appends to `fail` and the
script exits 1; `note()` appends to `warn`, prints "PRE-FLIGHT OK WITH NOTES"
and exits 0, so a note does NOT stop the run. The one gate that neither routes
through is the outcome-derived-feature gate below: it calls sys.exit(1) on the
spot, because if that column is present nothing measured afterwards means
anything.

THE COLLINEARITY GATE. The pipeline applies no per-fold correlation filter: it
relies on the fixed, outcome-blind TIER1_REMOVE list, backed by a
consensus-stage deduplication. That is only sound if Tier 1 leaves no
correlated pair the pipeline is still blind to, so this script recomputes every
within-modality pairwise correlation on the complete case AFTER applying
TIER1_REMOVE — pooled, DHP and T-DM1 — and FAILS the run if a pair exceeds
|r| = 0.90 in a modality the consensus-stage dedup does NOT cover. A pair above
the gate inside CONSENSUS_DEDUP_MODS (RNA, DNA) is printed as a NOTE and does
not block, because that stage removes it once on the full cohort, without the
fold-to-fold rotation of representatives that a per-fold filter introduces; see
the block above the `_covered` / `_uncovered` split for the full reasoning. If
the check fires, either add the offending feature to TIER1_REMOVE or reinstate
a per-fold filter.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA = Path("clin_multiomics_curated_metrics_PREDIX_HER2_new.txt")
# ---- NO-PATIENT-DATA VARIANT (inserted by build_github_repo.py) ----
_deposit_copy = Path(__file__).resolve().parent.parent / "data" / DATA.name
if not DATA.exists() and _deposit_copy.exists():
    DATA = _deposit_copy
# ---- end NO-PATIENT-DATA VARIANT ----

# The complete case is what every model is evaluated on; it must reproduce the
# cohort the manuscript reports.
EXPECT_CC = 110
EXPECT_CC_EVENTS = 46
EXPECT_CC_ARMS = {"DHP": 59, "T-DM1": 51}

# ---------------------------------------------------------------------------
# FEATURES WITHDRAWN UPSTREAM BY THE AUTHORS
# ---------------------------------------------------------------------------
# Two of them, and they are NOT the same kind of problem, so they are not
# enforced the same way. Read this before merging the two tuples back together.
#
# *** RNA_ADC_trafficking MUST NEVER BE RESTORED. THIS IS A HARD CHECK. ***
# It is not a predictor. Per the bioinformatics lead, the signature
# is constructed from a mixture of pCR and residual disease — i.e. it is derived
# from the OUTCOME. Including it would regress pCR partly on a transformed copy
# of pCR, which is circular by construction and would inflate every metric that
# touches it. It is NOT in TIER1_REMOVE, so if the column ever reappears in the
# input file it enters the candidate pool and the whole run is contaminated:
# hence an immediate sys.exit(1) rather than a note.
#
# Two observations are worth recording so the question is not reopened:
#   * it is the reason the column was withdrawn upstream;
#   * it scored AUROC 0.740 (q = 0.025, rank 6/40) in the T-DM1 arm and 0.513
#     (q = 0.97) in DHP — a "predictor" that tracks the outcome will look strong
#     wherever the outcome is being tracked, not where biology says it should
#     be.
#
# Restoring it was considered and REJECTED on the grounds above. Any
# external-cohort column of that name is equally unusable; external_validation.py
# only applies the alias when the PREDIX target exists, so with the feature
# absent from PREDIX the alias is inert and reports itself as not applied. BOTH
# spellings are gated here — the external cohorts spell it 'RNA_ADC_traficking',
# with one 'f' — because FEATURE_ALIASES would map the external one onto the
# PREDIX name the moment a merged file was built.
OUTCOME_DERIVED = ("RNA_ADC_trafficking", "RNA_ADC_traficking")

# DNA_TMB_clone is a SOFT check — reported, does not block. It was withdrawn on
# measurement grounds, not because it is outcome-derived, and unlike
# RNA_ADC_trafficking it IS in TIER1_REMOVE, so even a stale file that carries
# the column cannot get it into the candidate pool. Presence is a signal that
# the wrong delivery was dropped in, which is what the note says.
WITHDRAWN = ("RNA_ADC_trafficking", "DNA_TMB_clone")

MODALITIES = ("Clin", "RNA", "DNA", "Prot", "WSI")
COMPLETENESS = ("RNA", "DNA", "Prot", "WSI")   # Clin is imputed in-fold

fail, warn = [], []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fail.append(name)


def note(name, detail=""):
    print(f"NOTE  {name}  {detail}")
    warn.append(name)


if not DATA.exists():
    # ---- NO-PATIENT-DATA VARIANT (inserted by build_github_repo.py) ----
    sys.exit(
        "==============================================================================\n"
        "PREDIX HER2 - NO-PATIENT-DATA VARIANT OF THE DEPOSIT\n"
        "==============================================================================\n"
        "preflight.py\n"
        "cannot run here. It reads the PREDIX individual-level input matrix, and this\n"
        "variant of the repository does not contain it:\n"
        "\n"
        "    data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt\n"
        "    197 patients x 112 columns (patientID, pCR and 110 curated metrics)\n"
        "    Clin 5 / RNA 42 / DNA 41 / Prot 19 / WSI 3\n"
        "    SHA-256 64dd2f3ff1c99170c70a27685c7d9d5633c5ae2edb23b45dbabc1b88a575cef0\n"
        "\n"
        "THE DEPOSIT IS NOT BROKEN. The file is withheld on purpose, pending the\n"
        "ethics and consent decision on releasing individual-level trial data.\n"
        "\n"
        "  * what still runs without it, and what does not : data/README.md\n"
        "  * how to request the file                       : data/README.md\n"
        "  * what individual-level information the deposited results/ artefacts\n"
        "    still carry                                   : docs/NO_PATIENT_DATA.md\n"
        "\n"
        "If you have obtained the file, put it back at exactly the path above, check\n"
        "its SHA-256 against the value printed here, and re-run this command unchanged.\n"
        "==============================================================================\n"
    )
    # ---- end NO-PATIENT-DATA VARIANT ----

sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
print(f"input SHA-256: {sha}\n")

df = pd.read_csv(DATA, sep="\t")
check("patientID and pCR present", {"patientID", "pCR"} <= set(df.columns))

feat = [c for c in df.columns if c not in ("patientID", "pCR")]
by = {m: sum(1 for c in feat if c.startswith(m + "_")) for m in MODALITIES}
check("all five modalities present", all(v > 0 for v in by.values()), str(by))
print(f"      {len(df)} rows x {len(df.columns)} columns; {len(feat)} features\n")

# ---- the complete case must reproduce the reported cohort -------------------
molecular = [c for c in feat if c.split("_")[0] in COMPLETENESS]
cc = df.dropna(subset=molecular)
check(f"complete case = {EXPECT_CC} patients", len(cc) == EXPECT_CC, str(len(cc)))
check(f"complete case = {EXPECT_CC_EVENTS} pCR events",
      int(cc["pCR"].sum()) == EXPECT_CC_EVENTS, str(int(cc["pCR"].sum())))

arm = cc["Clin_Arm"]
if pd.api.types.is_numeric_dtype(arm):
    got = {("DHP" if k == 0 else "T-DM1"): int(v)
           for k, v in arm.value_counts().to_dict().items()}
else:
    got = {str(k): int(v) for k, v in arm.value_counts().to_dict().items()}
check("complete-case arm sizes", got == EXPECT_CC_ARMS, str(got))

# ---- what the complete-case design needs beyond the counts above ------------
# The three checks above are the substance of it: the cohort exists, it is the
# size the results describe, and it splits into the two arms as reported.
# One more thing has to hold before an arm-level model can be fitted or scored
# at all — each arm must contain both outcome classes. Without that, stratified
# cross-validation cannot be constructed for the arm and its AUROC is
# undefined, which is a data problem, not a modelling result.
print()
_arm_events = {}
for _code, _lab in ((0, "DHP"), (1, "T-DM1")):
    _sub = cc[arm == _code] if pd.api.types.is_numeric_dtype(arm) else cc[arm == _lab]
    _ev = int(_sub["pCR"].sum())
    _arm_events[_lab] = _ev
    check(f"{_lab} complete case carries both outcome classes",
          0 < _ev < len(_sub), f"n={len(_sub)}, events={_ev}")
check("arm events sum to the cohort event count",
      sum(_arm_events.values()) == int(cc["pCR"].sum()),
      f"{_arm_events} vs {int(cc['pCR'].sum())}")

# Reported, not asserted. RNA/DNA/Prot/WSI are complete on this cohort by
# construction — that is what makes it the complete case — so those four rows
# are a restatement of the cohort definition and only the feature counts carry
# information. Clin is not part of that definition: it is checked here, and any
# shortfall on the Clin row is handled by the in-fold imputer rather than by
# dropping patients. Note that a placeholder token such as 'Unknown' counts as
# observed at this stage; it becomes NaN only when the pipeline encodes the
# frame, which is where the imputer picks it up.
print("\nfeatures per modality, and complete-case patients fully observed on them:")
for m in MODALITIES:
    cols = [c for c in feat if c.startswith(m + "_")]
    n_obs = int(cc[cols].notna().all(axis=1).sum()) if cols else 0
    print(f"      {m:<5} {len(cols):3d} features   "
          f"{n_obs:3d}/{len(cc)} patients fully observed")

# ---- features withdrawn upstream -------------------------------------------
print()

# HARD GATE. Not a note: an outcome-derived column in the input file invalidates
# every number the run would go on to produce, so stop here and stop loudly.
_contaminated = [c for c in OUTCOME_DERIVED if c in df.columns]
if _contaminated:
    print("!" * 78)
    print("!!  PRE-FLIGHT FAILED — OUTCOME-DERIVED FEATURE PRESENT IN THE INPUT")
    print("!" * 78)
    for c in _contaminated:
        _n = int(df[c].notna().sum())
        print(f"!!  {c} is a column of {DATA} ({_n}/{len(df)} observed).")
    print("!!")
    print("!!  The ADC-trafficking signature is CONSTRUCTED FROM THE OUTCOME: it")
    print("!!  is built from a mixture of pCR and residual disease. Regressing")
    print("!!  pCR on it is circular by construction, and every AUROC, every")
    print("!!  confidence interval and every external-validation number produced")
    print("!!  from a file containing it is inflated and unpublishable.")
    print("!!")
    print("!!  It is NOT in TIER1_REMOVE, so nothing downstream would remove it.")
    print("!!  It was withdrawn upstream by the bioinformatics lead, and")
    print("!!  restoring it was considered and REJECTED on those grounds.")
    print("!!  'RNA_ADC_traficking' (one 'f') is the external-cohort spelling")
    print("!!  of the same signature and is equally unusable.")
    print("!!")
    print("!!  Do NOT 'fix' this by editing preflight.py. Fix it by supplying the")
    print("!!  curated input matrix, which does not contain the column:")
    print("!!      clin_multiomics_curated_metrics_PREDIX_HER2_new.txt")
    print("!!      sha256 64dd2f3ff1c99170c70a27685c7d9d5633c5ae2edb23b45dbabc1b88a575cef0")
    print("!" * 78)
    print("\nThe pipeline was NOT started.")
    sys.exit(1)
print(f"PASS  no outcome-derived ADC-trafficking column "
      f"({', '.join(OUTCOME_DERIVED)}) is present  [HARD CHECK]")

for c in WITHDRAWN:
    if c in df.columns:
        note(f"{c} is PRESENT in this file",
             "it was withdrawn upstream — confirm this is intended")
    else:
        print(f"PASS  {c} absent (withdrawn upstream)")

# ---- Tier 1 must leave nothing for a per-fold filter to do ------------------
print()
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal_pcr_pipeline import (TIER1_REMOVE, CORR_FILTER_MODS,  # noqa: E402
                                     CONSENSUS_DEDUP_MODS, SIGNATURE_SOURCE)

CORR_GATE = 0.90
t1_present = [c for c in TIER1_REMOVE if c in df.columns]
candidates = [c for c in feat if c not in TIER1_REMOVE]
print(f"TIER1_REMOVE lists {len(TIER1_REMOVE)}, {len(t1_present)} present "
      f"-> {len(candidates)} candidates enter the fold loop")

check("Tier 3 (per-fold correlation filter) is disabled",
      len(CORR_FILTER_MODS) == 0, f"CORR_FILTER_MODS={CORR_FILTER_MODS or '{}'}")
check("the consensus-stage dedup is enabled as the safety net",
      CONSENSUS_DEDUP_MODS == {"RNA", "DNA"}, str(CONSENSUS_DEDUP_MODS))

# ---- the locked model must be one object, not two ---------------------------
check("the locked signature belongs to the locked classifier",
      SIGNATURE_SOURCE in ("winner_all_folds", "winner_folds"),
      f"SIGNATURE_SOURCE={SIGNATURE_SOURCE!r} — 'winner_folds' (the pipeline "
      f"default, and what this analysis uses) aggregates the folds the modal "
      f"classifier won; 'winner_all_folds' uses that classifier's own signature "
      f"from every fold; 'all_folds' mixes classifier families and would leave "
      f"the locked classifier and the locked signature describing different "
      f"objects")

check("RNA_FCGR3B is excluded from the candidate pool",
      "RNA_FCGR3B" in TIER1_REMOVE,
      "the Methods must carry the OUTCOME-BLIND justification for this one; "
      "it is not a collinearity removal like the others")
if "RNA_FCGR3B" in df.columns:
    _obs = int(df["RNA_FCGR3B"].notna().sum())
    print(f"      RNA_FCGR3B was observed in {_obs}/{len(df)} patients and is "
          f"excluded from the candidate pool before any outcome is examined")

# THE GATE MUST SEE THE CATEGORICAL FEATURES.
#
# Correlating the raw frame with `.apply(pd.to_numeric, errors="coerce")` turns
# every non-numeric column into NaN, and the `np.isfinite(v)` test below then
# skips those pairs SILENTLY — leaving 21 of the candidates untested: every
# DNA_coding_mutation_* (stored as the strings True/False), Clin_Arm / Clin_ER /
# Clin_ANYNODES / Clin_TUMSIZE, RNA_sspbc.subtype, and Prot_ERBB2_PG
# (Positive/Negative), which coerces to a 100%-NaN column. Prot_ERBB2_PG is not
# a minor case: it carries the DHP proteomic signature's top-ranked feature, so
# a gate that cannot see it is not testing what it claims to test. The frame is
# therefore ENCODED FIRST, and the encoded maxima are printed per modality so an
# all-NaN failure cannot look like a pass.
#
# Use the PIPELINE'S OWN encoder, not a local re-implementation. A hand-rolled
# encoder gets two things wrong:
# Clin_TUMSIZE was alphabetised ('21-50' < '<=20' < '>50') instead of ordered
# (<=20 < 21-50 < >50) — Pearson between the two codings is only 0.464, so for
# a >2-level ordinal the gate's |r| is not a monotone function of the modelled
# |r| and a collinear pair could slip through — and RNA_sspbc.subtype was
# integer-coded 0-3 as ONE column while the pipeline one-hots it into three
# dummies, so the gate tested a column the model never sees and never tested
# the three it does (RNA_sspbc_LumB is in the global RNA signature).
# load_and_encode_data also applies TIER1_REMOVE, so its columns ARE the
# candidate pool the fold loop will see.
from multimodal_pcr_pipeline import load_and_encode_data  # noqa: E402

_enc = load_and_encode_data(DATA)
_enc_feats = [c for c in _enc.columns
              if c.split("_")[0] in MODALITIES and c not in ("patient_id", "pCR")]

# The gate must cover every cohort the consensus dedup will actually run on.
# finalize_consensus receives df_cc_exp — the ARM frame for dhp/tdm1, not the
# pooled complete case — and there is no per-fold filter in any of the three
# experiments. A pair can be far more correlated inside one arm than pooled:
# DNA_coding_mutation_HER_pathway ~ DNA_coding_mutation_ERBB2_oncokb is 0.834
# pooled but 1.000 on the 59 DHP complete cases. Checking the pooled cohort
# alone would establish inertness for only one of the three experiments.
# Restrict to the ALREADY-VERIFIED complete case (cc, n=110 above) rather than
# re-deriving it. Encoding maps the 'Unknown' tokens in Clin_TUMSIZE and
# Clin_prolifvalu to NaN, so a dropna() over the encoded columns would silently
# shrink the cohort to 104 and the gate would then describe a population the
# pipeline never evaluates. .corr() uses pairwise-complete deletion, which is
# the right treatment for those few missing cells.
_enc_cc = _enc.loc[cc.index]
_arm_col = "Clin_Arm"
_cohorts = [("pooled", _enc_cc)]
if _arm_col in _enc_cc.columns:
    _cohorts += [("DHP", _enc_cc[_enc_cc[_arm_col] == 0]),
                 ("T-DM1", _enc_cc[_enc_cc[_arm_col] == 1])]

residual = []
for _label, _cc in _cohorts:
    print(f"      {_label:6s} complete case n={len(_cc):3d} — strongest "
          f"retained pair per modality:")
    if len(_cc) < 3:
        print(f"        too few complete cases to correlate — INVESTIGATE")
        continue
    for m in MODALITIES:
        cols = [c for c in _enc_feats if c.startswith(m + "_")]
        if len(cols) < 2:
            continue
        R = _cc[cols].corr().abs()
        pr = [(R.loc[a, b], a, b) for i, a in enumerate(cols)
              for b in cols[i + 1:] if np.isfinite(R.loc[a, b])]
        if not pr:
            print(f"        {m:5s} no finite pair — INVESTIGATE")
            continue
        r, a, b = max(pr)
        flag = "  <-- ABOVE GATE" if r > CORR_GATE else ""
        print(f"        {m:5s} {r:.3f}  {a} ~ {b}{flag}")
        for rr, aa, bb in pr:
            if rr > CORR_GATE:
                residual.append((f"{_label}/{m}", aa, bb, float(rr)))

# Deleting Tier 3 is safe for a modality if EITHER no pair exceeds the gate,
# OR the consensus-stage dedup covers that modality (it operates on the full
# cohort, so unlike Tier 3 it does not rotate representatives between folds).
# Only an uncovered modality is a genuine blocker.
_covered = [t for t in residual if t[0].split("/")[1] in CONSENSUS_DEDUP_MODS]
_uncovered = [t for t in residual if t[0].split("/")[1] not in CONSENSUS_DEDUP_MODS]

if _covered:
    print(f"      NOTE {len(_covered)} correlated pair(s) in "
          f"{sorted(CONSENSUS_DEDUP_MODS)} — these are handled by the "
          f"consensus-stage dedup, which is the reason that stage exists. "
          f"Expect [CONSENSUS-POOL] lines in the run log:")
    for m, a, b, v in sorted(_covered, key=lambda t: -t[3]):
        print(f"        {m}  {v:.4f}  {a} ~ {b}")

check(f"no UNPROTECTED candidate pair exceeds |r| = {CORR_GATE} after Tier 1",
      not _uncovered,
      "every modality is either free of correlated pairs or covered by the "
      "consensus dedup, so no per-fold filter is needed" if not _uncovered
      else f"{len(_uncovered)} pair(s) exceed the gate in a modality the "
           f"consensus dedup does NOT cover ({sorted(set(t[0] for t in _uncovered))}) "
           f"— add them to TIER1_REMOVE or enable a per-fold filter")
for m, a, b, v in sorted(_uncovered, key=lambda t: -t[3]):
    print(f"        {m}  {v:.4f}  {a} ~ {b}")

# Report the identical-column pairs explicitly. Two columns that are literally
# equal wherever both are observed cannot be told apart by any model, and they
# would enter a consensus signature as two separate entries describing one
# measurement.
print("\nexact-duplicate check on the retained candidates:")
dupes = []
for m in MODALITIES:
    cols = [c for c in candidates if c.startswith(m + "_")]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = df[a].notna() & df[b].notna()
            if int(both.sum()) > 0 and bool((df.loc[both, a] == df.loc[both, b]).all()):
                dupes.append((a, b, int(both.sum())))
check("no two retained candidates are byte-identical", not dupes,
      "none" if not dupes else str(dupes))

# ---- missingness profile, reported not asserted -----------------------------
n_missing = int(df[feat].isna().sum().sum())
print(f"\nmissing feature cells in the full file: {n_missing:,} "
      f"({n_missing / (len(df) * len(feat)):.1%}) — expected and fine; the "
      f"complete case above is what is evaluated")

# ---- the per-scenario deduplication contract ---------------------------------
# Applied inside each scenario's own cohort, within modality, before any split.
# Three statistics because one cannot serve three data types: correlation for
# continuous features, exact agreement for segment-called copy number (where
# genes in one segment carry one number and correlation is blind to it), and
# Cohen's kappa for sparse binary indicators (where raw agreement is inflated by
# rarity -- two mutations each present in 5% of patients agree in ~90% of rows
# without ever co-occurring).
#
# Nothing here reads pCR. That is what makes a pre-split step legitimate.
try:
    from feature_deduplication import (deduplicate, surviving_violations,
                                       CORR_T, IDENT_T, KAPPA_T)
except ImportError as exc:                                   # pragma: no cover
    print(f"\nFAIL  feature_deduplication.py is not importable: {exc}")
    fail.append("feature_deduplication.py missing")
else:
    print(f"\nper-scenario deduplication contract "
          f"(|r|>={CORR_T}, identical>={IDENT_T}, kappa>={KAPPA_T}), "
          f"within modality:")
    _cand = [c for c in feat if c not in TIER1_REMOVE]
    # Arm labels, not codes. The raw file stores "DHP"/"T-DM1" as strings; the
    # pipeline encodes them to 0/1 later. Testing dtype == object silently
    # matched nothing under pandas' str dtype and produced two empty cohorts,
    # so both spellings are accepted and the result is asserted non-empty.
    _armcol = cc["Clin_Arm"].astype(str)
    _scen = {"global": cc,
             "dhp": cc[_armcol.isin(["DHP", "0"])],
             "tdm1": cc[_armcol.isin(["T-DM1", "1"])]}
    for _name, _pop in _scen.items():
        if not len(_pop):
            check(f"scenario {_name} has patients", False, "empty cohort")
            continue
        _res = deduplicate(_pop, _cand, scenario=_name)
        _left = surviving_violations(_pop, _res.keep)
        print(f"  {_name:7} n={len(_pop):>3}  removes {len(_res.drop):>2} of "
              f"{len(_cand)}  -> pool {len(_res.keep)}")
        for _d in _res.decisions:
            print(f"      {_d.dropped:36} -> {_d.kept:30} "
                  f"{_d.statistic}={_d.value:.3f}")
        # The rule has to be idempotent: one pass must leave nothing behind, or
        # the pool the pipeline models on is not the pool this describes.
        check(f"deduplication leaves no violation in {_name}",
              not _left,
              "clean" if not _left
              else f"{len(_left)} surviving: {[(a, b) for a, b, _, _ in _left[:3]]}")

print()
if fail:
    print("PRE-FLIGHT FAILED:", fail)
    print("The pipeline was NOT started. Fix the input or update preflight.py.")
    sys.exit(1)
if warn:
    print("PRE-FLIGHT OK WITH NOTES:", warn)
else:
    print("PRE-FLIGHT OK", end=" ")
print("— starting the run.")
