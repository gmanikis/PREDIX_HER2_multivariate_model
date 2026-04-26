#!/usr/bin/env python3
"""
MULTIMODAL pCR PREDICTION PIPELINE — PREDIX HER2
=================================================
Primary analysis mode (elasticnet) implements multi-classifier signature
discovery with leakage-safe stacking and Platt calibration.

PRIMARY ANALYSIS — elasticnet mode
  Stage A Pass 1: All classifiers compared with fixed STAGE_A_PARAMS using inner CV.
           Feature importance converted to cross-classifier percentile ranks and
           averaged across inner folds. EPV=5 cap + 25th-percentile filter + floor=5
           derive signature per classifier. Clin and WSI keep all features.
           Calibration slope estimated from inner-loop OOF predictions.
  Stage A Pass 2: Pruned signature evaluated on each cached inner val fold.
           Winner selected by mean pruned inner AUROC (not all-feature AUROC).
  Stage B: Winner tuned with GridSearchCV on expanded training.
           Outer refit on winner signature features + Platt calibration if needed.
           OOF scores via make_oof_signature (expanded inner training, same config).
  Fusion:  Single Fused_ElasticNet (L1+L2, l1_ratio=0.5) trained on calibrated
           5-column OOF matrix. L1 zeroes non-contributing modalities for
           interpretable sparse modality weighting.

SUPPLEMENTARY MODES (best_per_fold, ensemble_weighted)
  Standard CC-only training, no signature discovery.
  Used for robustness comparison only.

EXPANDED TRAINING (elasticnet mode only)
  Each unimodal model trains on ALL patients with that modality minus test patients.
  Outer test sets always drawn from complete-case (n=110) for paired comparisons.

USAGE
-----
  # Primary analysis:
  python3 multimodal_pcr_pipeline.py --data_path /data/predix.txt \\
      --classifiers ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear \\
      --repeats_global 200 --repeats_arm 100

OUTPUT PKL FORMAT  ({results_dir}/{exp}/{exp}_elasticnet_results.pkl)
  {
    "Clin" / "RNA" / "DNA" / "Prot" / "WSI":  [fold_dict, ...]
    "Fused_ElasticNet":                        [fold_dict, ...]
  }

  Unimodal fold_dict keys (elasticnet / primary mode):
    fold_idx, metrics (AUROC/AUPRC/Brier/Sensitivity/Specificity/Threshold)
    y_test, y_pred
    winner_clf, winner_signature, signature_size, n_events_inner
    inner_cv_aurocs_A  {clf: mean_auroc}   — Stage A fixed-params AUROC
    inner_cv_auroc_B   float               — Stage B tuned AUROC
    inner_cv_params    dict                — winner best hyperparams
    inner_importance   {clf: {feat: val}}  — normalised importance per clf
    signatures_all     {clf: [feats]}      — EPV-capped signature per clf
    calibration        {clf: {slope, needs_platt}}
    platt_applied      bool
    features           [feature names in winner signature]
    oof_shap           {feature_names, shap_values, X_test_scaled}

  Fusion fold_dict keys:
    fold_idx, metrics, y_test, y_pred
    tuned_C, modality_weights, selected_modalities
    oof_shap  {feature_names: [Clin,RNA,DNA,Prot,WSI], shap_values, X_test_scaled}
"""

import argparse, pickle, warnings, os
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# THREAD-POOL DISCIPLINE (must be set BEFORE numpy / sklearn are imported so
# BLAS/OpenMP libraries pick them up at init). Setting os.environ[...] later
# from inside joblib workers does NOT work because BLAS has already started
# its thread pool by then. Workers additionally use threadpool_limits() as a
# runtime belt-and-braces guard (see _process_single_fold).
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("BLIS_NUM_THREADS",     "1")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from threadpoolctl import threadpool_limits
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     RepeatedStratifiedKFold)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap

# ==============================================================================
# SECTION 1 — MODULE-LEVEL PLACEHOLDERS  (all assigned in main)
# ==============================================================================
DATA_PATH = RESULTS_DIR = RANDOM_SEED = None
GLOBAL_N_OUTER_FOLDS = GLOBAL_N_REPEATS = GLOBAL_N_INNER_FOLDS = None
ARM_N_OUTER_FOLDS    = ARM_N_REPEATS    = ARM_N_INNER_FOLDS    = None
CORR_THRESHOLD = NZV_RATIO_THRESHOLD = None
NZV_FREQ_GLOBAL = NZV_FREQ_ARM = NZV_FREQ_THRESHOLD = None
STABILITY_THRESHOLD_GLOBAL = STABILITY_THRESHOLD_ARM           = None
N_JOBS                                                         = None


def _resolve_parallel_budget(n_folds, n_jobs):
    """
    Allocate the available CPU budget between outer-fold parallelism and
    inner-fit parallelism.

    Three regimes:
      1. n_jobs == 1                 → sequential (debugging)
      2. n_jobs >= n_folds           → outer=n_folds, inner = n_jobs // n_folds
                                       (nested parallelism — the "CPU-rich" case
                                       the user reports when folds < CPUs)
      3. n_jobs <  n_folds           → outer=n_jobs, inner=1
                                       (outer-only — the classic case)

    Returns (n_outer_workers, n_inner_jobs).
    """
    if n_jobs is None or n_jobs == 1:
        return 1, 1
    total_cpus = n_jobs if n_jobs > 0 else joblib.cpu_count()
    n_outer    = min(total_cpus, n_folds)
    n_inner    = max(1, total_cpus // max(n_outer, 1))
    return n_outer, n_inner


# Fixed algorithmic constants
L1_RATIO            = 0.5
ELASTICNET_C_GRID   = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
FUSION_C_GRID       = ELASTICNET_C_GRID
# CORR_FILTER_MODS: modalities where Tier 3 correlation filter is applied.
# All modalities use elastic net; Tier 3 is restricted to high-dimensional
# RNA and DNA because WSI has only 3 features and Clin/Prot have 5.
CORR_FILTER_MODS = {"RNA", "DNA"}

# Populated in main() after seed is set
CLASSIFIERS = {}

# Fixed hyperparameters for Stage A (classifier comparison + feature ranking).
# These are applied uniformly to all classifiers so that the AUROC
# comparison is fair: no classifier benefits from extra tuning time.
# The winner classifier is then fully tuned in Stage B (GridSearchCV).
STAGE_A_PARAMS = {
    "ElasticNet_LR": {"C": 0.1},
    "RandomForest":  {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "ExtraTrees":    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "HistGradBoost": {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200},
    "SVM_Linear":    {"C": 0.1},
}



# ==============================================================================
# SECTION 2 — CLASSIFIER FACTORY
# ==============================================================================

def build_classifiers(seed):
    """
    Return full classifier config dict. Called once in main() after seed set.

    RANDOMNESS DESIGN
    -----------------
    Classifier internal randomness is DECOUPLED from the reproducibility seed.
    RepeatedStratifiedKFold uses `seed` so the train/test partitions are
    reproducible across runs — that is what anchors reproducibility. The
    classifiers themselves use `random_state=None`, i.e. fresh NumPy
    randomness on every fit. Without this, every RandomForest in every
    repeat sees the same bootstrap sample for a given fold and every
    saga-solver uses the same shuffle order — this makes the 200 repeats
    artificially correlated and produces overly narrow CIs on fold-averaged
    metrics. With random_state=None the repeats are genuinely independent
    for fixed fold assignments, so reported variance is honest.

    The `seed` argument is retained in the signature for backwards
    compatibility but is no longer propagated into classifier constructors.
    """
    return {
        "ElasticNet_LR": {
            "build":     lambda: LogisticRegression(
                             penalty="elasticnet", solver="saga",
                             l1_ratio=L1_RATIO, max_iter=2000,
                             random_state=None),
            "grid":      {"C": ELASTICNET_C_GRID},
            "shap_type": "linear",
        },
        "RandomForest": {
            "build":     lambda: RandomForestClassifier(
                             random_state=None, n_jobs=1),
            "grid":      {"n_estimators": [100, 300],
                          "max_depth": [None, 5, 10],
                          "min_samples_leaf": [1, 5]},
            "shap_type": "tree",
        },
        "ExtraTrees": {
            "build":     lambda: ExtraTreesClassifier(
                             random_state=None, n_jobs=1),
            "grid":      {"n_estimators": [100, 300],
                          "max_depth": [None, 5, 10],
                          "min_samples_leaf": [1, 5]},
            "shap_type": "tree",
        },
        "HistGradBoost": {
            "build":     lambda: HistGradientBoostingClassifier(
                             random_state=None),
            "grid":      {"learning_rate": [0.05, 0.1, 0.2],
                          "max_depth": [3, 5, None],
                          "max_iter": [100, 300]},
            "shap_type": "tree",
        },
        "SVM_RBF": {
            "build":     lambda: SVC(kernel="rbf", probability=True,
                                     random_state=None, cache_size=500),
            "grid":      {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
            "shap_type": "none",
        },
        "SVM_Linear": {
            "build":     lambda: SVC(kernel="linear", probability=True,
                                     random_state=None, cache_size=500),
            "grid":      {"C": [0.01, 0.1, 1.0, 10.0]},
            "shap_type": "linear_svm",
        },
    }


# SECTION 2b — TIER 1 BIOLOGICAL DEDUPLICATION (FIXED, PRE-SPECIFIED)
# ==============================================================================
# These features are removed BEFORE any analysis based purely on domain knowledge
# of the PREDIX HER2 genomic architecture. This is NOT a data-driven decision —
# it is data quality management. Each removal is individually justified below.
#
# Empirical verification: all removed features had r >= 0.90 with their retained
# counterpart in the complete-case cohort (r = 1.000 for exact duplicates).

TIER1_REMOVE = [
    # ------------------------------------------------------------------
    # DNA: 17q12 chromosomal amplicon co-amplification
    # ERBB2, GRB7, PPP1R1B, MIEN1 and CDK12 all reside on 17q12 and
    # co-amplify as a single genomic segment. Their CNA values are
    # therefore identical by construction (r = 1.000 with ERBB2_CNA).
    # Decision: keep DNA_ERBB2_CNA (the oncogenic driver of HER2+ BC).
    # ------------------------------------------------------------------
    "DNA_PPP1R1B_CNA",   # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_MIEN1_CNA",     # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_GRB7_CNA",      # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_CDK12_CNA",     # r=0.904 with DNA_ERBB2_CNA  (17q12 amplicon)

    # ------------------------------------------------------------------
    # DNA: 11q13 amplicon co-amplification
    # PPFIA1 and CTTN co-amplify on 11q13 (r = 1.000).
    # Decision: keep DNA_PPFIA1_CNA.
    # ------------------------------------------------------------------
    "DNA_CTTN_CNA",      # r=1.000 with DNA_PPFIA1_CNA (11q13 amplicon)

    # ------------------------------------------------------------------
    # DNA: TMB metric cluster
    # totalTMB, TMB_uniform, TMB_clone, and pTMB measure the same
    # underlying mutational burden at different granularities (r=0.903–0.975).
    # Decision: keep DNA_totalTMB (most comprehensive aggregate) and
    # DNA_TMB_subclone (captures subclonal-specific burden — the least
    # redundant metric among the group, complementary information).
    # ------------------------------------------------------------------
    "DNA_TMB_uniform",   # r=0.975 with DNA_totalTMB
    "DNA_TMB_clone",     # r=0.908 with DNA_totalTMB
    "DNA_pTMB",          # r=0.903 with DNA_totalTMB

    # ------------------------------------------------------------------
    # RNA: immune infiltration cluster
    # CD8-T-cells, T-cells, CD45, and Cytotoxic-cells are near-identical
    # to RNA_TILs and RNA_mRNA-CD8A (r = 0.926–0.984). Retaining all
    # causes the 'rotating basis' instability in elastic net.
    # Decision: keep RNA_TILs  (global immune composite, widely used)
    #                RNA_mRNA-CD8A (cytotoxic-specific signal)
    #                RNA_NK-cells  (innate immunity, distinct biology)
    # ------------------------------------------------------------------
    "RNA_CD8-T-cells",    # r=0.984 with RNA_mRNA-CD8A
    "RNA_T-cells",        # r=0.972 with RNA_TILs
    "RNA_CD45",           # r=0.948 with RNA_TILs
    "RNA_Cytotoxic-cells",# r=0.940 with RNA_mRNA-CD8A

    # ------------------------------------------------------------------
    # RNA: HER2 expression redundancy
    # RNA_HER2DX_HER2_amplicon is a validated composite score that
    # subsumes raw RNA_mRNA-ERBB2 (r = 0.959). Keep the composite score
    # (HER2DX_HER2_amplicon) as it is the curated, clinically validated
    # representation.
    # ------------------------------------------------------------------
    "RNA_mRNA-ERBB2",    # r=0.959 with RNA_HER2DX_HER2_amplicon
]

# ==============================================================================
# SECTION 3 — DATA LOADING AND BASE ENCODING
# ==============================================================================

def load_and_encode_data(path: Path) -> pd.DataFrame:
    """
    Load the dataset and apply all fixed categorical/ordinal encodings.

    This function performs ONLY transformations that are non-data-dependent
    (i.e., they map known fixed categories to numbers). No imputation, scaling,
    feature selection, or any fold-dependent operation is performed here.

    Encoding decisions:
      - Boolean strings ('True'/'False') → 0/1
      - Clin_ER: 'positive'=1, 'negative'=0
      - Clin_ANYNODES: 'N+'=1, 'N0'=0
      - Clin_TUMSIZE: ordinal (<=20=1, 21-50=2, >50=3); NaN preserved → imputed in CV
      - Clin_Arm: 'DHP'=0, 'T-DM1'=1
      - Prot_ERBB2_PG: 'Positive'=1, 'Negative'=0
      - RNA_sspbc.subtype: one-hot with Her2 as reference category

    Parameters
    ----------
    path : Path
        Location of the tab-separated dataset file.

    Returns
    -------
    pd.DataFrame
        Encoded dataset with Tier 1 redundant features removed.
    """
    df = pd.read_csv(path, sep="\t")
    # Assign a stable integer patient ID before any filtering or reindexing.
    # This ID is used to match patients across modality-specific subsets and
    # to exclude test-set patients from training sets without index confusion.
    df["patient_id"] = range(len(df))
    print(f"[LOAD] Raw data: {df.shape[0]} patients, {df.shape[1]} columns")

    # --- 1. Convert boolean columns to 0/1 -----------------------------------
    # DNA mutation and genomic flag columns are stored as Python bool objects
    # (True/False) inside object-dtype arrays — NOT as strings 'True'/'False'.
    # We identify them by checking that all non-null unique values are a subset
    # of {True, False} (Python booleans), then cast with astype(int).
    bool_cols = [
        c for c in df.columns
        if df[c].dtype == object
        and set(df[c].dropna().unique()).issubset({True, False})
    ]
    for col in bool_cols:
        # Cast to float (not int) to preserve NaN — these are imputed inside each CV fold
        df[col] = df[col].astype(float)
    print(f"[ENCODE] {len(bool_cols)} boolean columns → 0/1: {bool_cols}")

    # --- 2. Clinical categorical encodings ------------------------------------
    df["Clin_ER"]       = df["Clin_ER"].map({"positive": 1, "negative": 0})
    df["Clin_ANYNODES"] = df["Clin_ANYNODES"].map({"N+": 1, "N0": 0})
    # Ordinal encoding: tumour size categories map to 1, 2, 3.
    # Missing TUMSIZE (n=6) kept as NaN and imputed within each CV fold.
    df["Clin_TUMSIZE"]  = df["Clin_TUMSIZE"].map({"<=20": 1, "21-50": 2, ">50": 3})
    df["Clin_Arm"]      = df["Clin_Arm"].map({"DHP": 0, "T-DM1": 1})
    print("[ENCODE] Clin_ER, Clin_ANYNODES, Clin_TUMSIZE, Clin_Arm encoded")

    # --- 3. Proteomics categorical encoding -----------------------------------
    df["Prot_ERBB2_PG"] = df["Prot_ERBB2_PG"].map({"Positive": 1, "Negative": 0})
    print("[ENCODE] Prot_ERBB2_PG: Positive=1, Negative=0")

    # --- 4. RNA_sspbc.subtype: one-hot encoding (Her2 as reference) ----------
    # This is a multiclass variable (Her2=104, LumA=36, LumB=35, Basal=10).
    # Her2 is used as reference because it is the most common category in this
    # HER2-enriched cohort. Rare categories (Basal, n=5 per arm) may be removed
    # by the NZV filter in arm-specific folds — this is correct and expected.
    subtype_dummies = pd.get_dummies(df["RNA_sspbc.subtype"], prefix="RNA_sspbc")
    # Drop Her2 dummy to avoid perfect multicollinearity (Her2 is the reference)
    subtype_dummies = subtype_dummies.drop(
        columns=["RNA_sspbc_Her2"], errors="ignore"
    )
    df = df.drop(columns=["RNA_sspbc.subtype"])
    df = pd.concat([df, subtype_dummies.astype(float)], axis=1)
    print(f"[ENCODE] RNA_sspbc.subtype → dummies: {subtype_dummies.columns.tolist()}")

    # --- 5. Apply Tier 1 biological deduplication ----------------------------
    # Features removed here are BIOLOGICALLY redundant (co-amplicons or
    # near-identical composite scores). This is a domain decision, not a
    # statistical one, and is applied before any train/test splitting.
    present = [c for c in TIER1_REMOVE if c in df.columns]
    df = df.drop(columns=present)
    print(f"[TIER1] Removed {len(present)} biologically redundant features")
    for feat in present:
        print(f"         - {feat}")

    return df


def define_modality_features(df: pd.DataFrame) -> dict:
    """
    Define the column sets for each modality after Tier 1 deduplication.

    Returns a dict with keys:
      'Clin_global' : all Clin_ columns including Clin_Arm
      'Clin_arm'    : Clin_ columns excluding Clin_Arm (for arm-specific models
                      where treatment arm is constant and non-informative)
      'RNA'         : all RNA_ columns
      'DNA'         : all DNA_ columns
      'Prot'        : all Prot_ columns
      'WSI'         : all WSI_ columns
    """
    features = {}
    features["Clin_global"] = [c for c in df.columns if c.startswith("Clin_")]
    features["Clin_arm"]    = [c for c in features["Clin_global"]
                                if c != "Clin_Arm"]
    features["RNA"]         = [c for c in df.columns if c.startswith("RNA_")]
    features["DNA"]         = [c for c in df.columns if c.startswith("DNA_")]
    features["Prot"]        = [c for c in df.columns if c.startswith("Prot_")]
    features["WSI"]         = [c for c in df.columns if c.startswith("WSI_")]

    print("\n[FEATURE SETS after Tier 1]")
    for name, cols in features.items():
        print(f"  {name:15s}: {len(cols):3d} features")

    return features


def get_complete_case(df: pd.DataFrame, features: dict) -> pd.DataFrame:
    """
    Restrict to patients who have ALL five modalities measured.

    Rationale: using the same 110 patients across all unimodal AND fused models
    ensures every performance comparison is perfectly paired on the same test
    patients. This avoids the confound of 'model A was evaluated on more or
    different patients than model B'.

    The 87 excluded patients are missing data primarily due to Proteomics
    (60 missing; modality-level missingness, not random feature dropout).
    This is stated as a design decision in the manuscript, not a limitation.

    Parameters
    ----------
    df       : encoded DataFrame
    features : dict from define_modality_features()

    Returns
    -------
    pd.DataFrame (index reset to 0..n-1 for clean positional indexing)
    """
    all_modality_cols = (
        features["RNA"] + features["DNA"] + features["Prot"] + features["WSI"]
    )
    df_complete = df.dropna(subset=all_modality_cols).reset_index(drop=True)

    arm0 = (df_complete["Clin_Arm"] == 0).sum()
    arm1 = (df_complete["Clin_Arm"] == 1).sum()
    pcr_global = df_complete["pCR"].mean()
    pcr_arm0   = df_complete.loc[df_complete["Clin_Arm"] == 0, "pCR"].mean()
    pcr_arm1   = df_complete.loc[df_complete["Clin_Arm"] == 1, "pCR"].mean()

    print(f"\n[COMPLETE CASE] n={len(df_complete)} "
          f"(DHP={arm0}, T-DM1={arm1})")
    print(f"[COMPLETE CASE] pCR rate: "
          f"overall={pcr_global:.3f}, DHP={pcr_arm0:.3f}, T-DM1={pcr_arm1:.3f}")

    return df_complete



# ==============================================================================
# SECTION 3b — PER-MODALITY PATIENT DATASETS
# ==============================================================================

def get_modality_datasets(df_enc: pd.DataFrame, features: dict) -> dict:
    """
    For each modality, return ALL patients who have complete data for that
    modality (not just the 110 complete-case patients).

    These expanded datasets are used as training sets: each unimodal model
    trains on ALL patients with that modality minus the current outer test
    patients, rather than being restricted to the 110 complete-case patients.
    The outer TEST sets remain fixed to complete-case patients so that:
      (a) All five modality predictions are always available at test time,
          allowing the fusion model to always run.
      (b) All pairwise performance comparisons remain fully paired on
          identical test patients across all models.

    For arm-specific experiments, modality datasets are filtered to the
    appropriate treatment arm (Clin_Arm == 0 for DHP, == 1 for T-DM1) so
    that T-DM1 patients never enter DHP unimodal training and vice versa.

    Returns
    -------
    dict mapping modality key → pd.DataFrame with patient_id, pCR, and
    all modality feature columns. Only patients with non-null data for that
    modality are included.
    """
    datasets = {}
    for mod_key in ["Clin_global", "Clin_arm", "RNA", "DNA", "Prot", "WSI"]:
        cols = [c for c in features.get(mod_key, []) if c in df_enc.columns]
        feat_cols = [c for c in cols if c not in ["patient_id", "pCR"]]
        if feat_cols:
            mask = df_enc[feat_cols].notna().all(axis=1) & df_enc["pCR"].notna()
        else:
            mask = df_enc["pCR"].notna()
        # Keep patient_id and pCR alongside features (needed for set operations)
        keep_cols = ["patient_id", "pCR"] + cols
        keep_cols = [c for c in keep_cols if c in df_enc.columns]
        datasets[mod_key] = df_enc.loc[mask, keep_cols].copy().reset_index(drop=True)
    return datasets


# ==============================================================================
# SECTION 4 — WITHIN-FOLD PREPROCESSING UTILITIES
# ==============================================================================

def remove_near_zero_variance(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    freq_threshold:  float = None,
    ratio_threshold: float = None,
) -> tuple:
    """
    Remove near-zero variance (NZV) features — FITTED ON TRAINING SET ONLY.

    A feature is flagged as NZV if either condition holds on the training set:
      (a) The most common value occupies > freq_threshold of training samples.
      (b) The ratio of most-common-value frequency to second-most-common > ratio_threshold.

    The same features are then dropped from the test set.

    Why this matters: binary genomic features (e.g., rare mutation indicators)
    may have near-zero variance in small training folds, making them unstable
    predictors. For arm-specific models (n≈44 training), even features with 5%
    global prevalence may appear in 0–2 cases per inner training fold.

    IMPORTANT — default argument design:
    freq_threshold and ratio_threshold default to None and are resolved inside
    the function body by reading the module-level globals. This avoids the
    Python default-argument capture issue where default values are evaluated at
    function definition time rather than call time. CLI overrides of
    --nzv_freq and --nzv_ratio therefore take effect correctly.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame — must share the same column set.
    freq_threshold  : fraction of training samples occupied by most common value.
                      Defaults to NZV_FREQ_THRESHOLD (read at call time).
    ratio_threshold : freq(top1) / freq(top2) ratio above which a feature is NZV.
                      Defaults to NZV_RATIO_THRESHOLD (read at call time).

    Returns
    -------
    X_train_filtered, X_test_filtered : pd.DataFrame
    removed_features : list of column names removed
    """
    # Resolve defaults at call time so CLI overrides propagate correctly
    if freq_threshold  is None: freq_threshold  = NZV_FREQ_THRESHOLD
    if ratio_threshold is None: ratio_threshold = NZV_RATIO_THRESHOLD

    to_remove = []
    n_train   = len(X_train)

    for col in X_train.columns:
        counts = X_train[col].value_counts(dropna=True)

        # If all values are NaN, remove the feature
        if len(counts) == 0:
            to_remove.append(col)
            continue

        # Condition (a): dominant value frequency
        if counts.iloc[0] / n_train >= freq_threshold:
            to_remove.append(col)
            continue

        # Condition (b): top-1 to top-2 ratio
        if len(counts) >= 2 and (counts.iloc[0] / counts.iloc[1]) >= ratio_threshold:
            to_remove.append(col)

    X_train_f = X_train.drop(columns=to_remove)
    X_test_f  = X_test.drop(columns=to_remove)
    return X_train_f, X_test_f, to_remove


def remove_high_correlation(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train=None,
    threshold: float = None,
) -> tuple:
    """
    Remove highly correlated features — FITTED ON TRAINING SET ONLY.

    Algorithm:
      1. Compute the absolute Pearson correlation matrix on X_train for
         features with >2 unique values (continuous/ordinal only).
         Binary features are excluded because their correlations are usually
         lower and they carry distinct biological meaning per category.
      2. For each feature not yet assigned to a cluster, find all features
         correlated with it at |r| >= threshold. This defines a cluster.
      3. Within the cluster, KEEP the feature with the highest univariate
         signal and drop the rest (see "Cluster keeper criterion" below).

    What this function IS: a redundancy-removal step. The correlation-based
    clustering is the selector; it decides which features are drops.
    What this function is NOT: a global feature filter. Features not in any
    correlation cluster pass through untouched. Downstream elastic net + the
    signature-discovery logic in Stage A still handle multivariate selection
    across the full surviving feature set.

    Rationale for redundancy removal: correlated features cause the
    'rotating basis' problem in elastic net — the model selects feature A
    in fold 1 and feature B in fold 2 (both with ~r=0.95), making each
    appear only 50% stable, even though the biological signal is stable
    at 100%. Keeping one representative per cluster resolves this.

    Cluster keeper criterion
    -------------------------
    When y_train is provided AND has both classes, the keeper is the
    cluster member with the highest univariate discrimination measured
    as |AUROC - 0.5| + 0.5 (so an inverse-oriented predictor, AUROC=0.2,
    scores the same as a correctly-oriented AUROC=0.8; we only care
    about magnitude of signal, not direction). This replaces an older
    variance-based keeper which had three problems:
      1. Variance is scale-dependent and the correlation filter runs
         BEFORE standardization, so features on wider raw scales won
         regardless of informativeness.
      2. Variance is outlier-dominated — a single extreme value in a
         gene's training-fold expression can flip the keeper, and a
         different fold picks a different keeper, reintroducing the
         same instability the filter was meant to fix.
      3. Variance is y-blind, so the "representative" feature kept for
         downstream SHAP and interpretation was not selected for
         predictive content.

    Univariate AUROC fixes all three: it is rank-based (scale-invariant
    and outlier-robust) and y-aware. It is computed leakage-free on
    training data only.

    Tiebreaker: when y_train is missing, has one class, or AUROC cannot
    be computed, the function falls back to variance (the original
    behaviour). When both members of a cluster have AUROC exactly 0.5
    (both uninformative), variance is used as the secondary tiebreaker.

    The Tier 3 correlation filter is only applied to RNA and DNA.
    Clin, Prot, and WSI (3–5 features each) are too low-dimensional for
    aggressive correlation pruning — removing any feature from a 3-feature
    space (WSI) risks a degenerate model. Elastic net's L2 component handles
    residual correlation in small modalities adequately.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame — must share the same column set.
    y_train         : np.array or None — pCR labels for training patients,
                      same row order as X_train. Used only to pick the
                      keeper within correlation clusters.
    threshold       : |r| above which two features are considered redundant.

    Returns
    -------
    X_train_filtered, X_test_filtered : pd.DataFrame
    removed_features : list of column names removed
    """
    # Apply only to continuous/ordinal features (>2 unique values)
    candidate_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]

    # Resolve default at call time so --corr_threshold CLI override takes effect
    if threshold is None:
        threshold = CORR_THRESHOLD

    if len(candidate_cols) < 2:
        return X_train, X_test, []

    corr_matrix = X_train[candidate_cols].corr().abs()

    # Validate y_train for use in the keeper criterion
    use_auroc = False
    if y_train is not None:
        y_arr = np.asarray(y_train)
        # Strip any NaN labels defensively (should not occur on CC training)
        y_mask = ~pd.isna(y_arr)
        if y_mask.sum() >= 3 and len(np.unique(y_arr[y_mask])) >= 2:
            y_arr     = y_arr[y_mask].astype(float)
            y_mask_np = y_mask
            use_auroc = True

    def _univariate_auroc_score(feat_col):
        """|AUROC - 0.5| + 0.5, with median imputation for any NaN feature values."""
        x = X_train[feat_col].values.astype(float)
        if use_auroc:
            x = x[y_mask_np]
        if np.isnan(x).any():
            med = np.nanmedian(x)
            x = np.where(np.isnan(x), med if not np.isnan(med) else 0.0, x)
        if len(np.unique(x)) < 2:
            return 0.5   # constant feature: no signal
        try:
            return abs(float(roc_auc_score(y_arr, x)) - 0.5) + 0.5
        except Exception:
            return 0.5

    # ── Connected components via BFS ──────────────────────────────────────
    # A greedy "star" approach (for each feature, find all features directly
    # correlated with it) misses transitive chains:
    #   A-B >= threshold, B-C >= threshold, A-C < threshold
    # → star clustering creates {A,B} and {C}, losing the B-C relationship.
    # BFS connected components correctly groups {A, B, C} in this case.
    adj = defaultdict(set)
    for i, fa in enumerate(candidate_cols):
        for fb in candidate_cols[i+1:]:
            if corr_matrix.loc[fa, fb] >= threshold:
                adj[fa].add(fb)
                adj[fb].add(fa)

    visited  = set()
    to_remove = []
    decided   = set()

    for start in candidate_cols:
        if start in visited:
            continue
        # BFS to find connected component
        cluster = []
        queue   = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(adj[node] - visited)

        decided.update(cluster)

        if len(cluster) <= 1:
            continue

        # Pick the keeper within the cluster
        if use_auroc:
            scores    = {c: _univariate_auroc_score(c) for c in cluster}
            max_score = max(scores.values())
            top       = [c for c, s in scores.items() if s == max_score]
            if len(top) == 1:
                keeper = top[0]
            else:
                variances = X_train[top].var()
                keeper    = variances.idxmax()
        else:
            variances = X_train[cluster].var()
            keeper    = variances.idxmax()

        removals = [c for c in cluster if c != keeper]
        to_remove.extend(removals)

    to_remove = list(set(to_remove))

    X_train_f = X_train.drop(columns=to_remove)
    X_test_f  = X_test.drop(columns=to_remove)
    return X_train_f, X_test_f, to_remove


def fit_imputer_scaler(X_train: pd.DataFrame) -> dict:
    """
    Fit median imputer and StandardScaler on the training set.

    Median imputation is used because:
      - Clinical ordinal variables (Clin_TUMSIZE) have a few missing values.
      - Median is robust to outliers common in biomedical data.

    StandardScaler ensures all features are on the same scale before
    logistic regression, regardless of original units (mRNA counts, CNA ratios,
    cell proportions, etc.).

    CRITICAL: Both transformers are fitted on training data only.
    Applying test-set statistics to imputation or scaling would constitute
    data leakage.

    Returns
    -------
    preprocessor : dict with keys 'imputer', 'scaler', 'columns'
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_arr = imputer.fit_transform(X_train)
    scaler.fit(X_arr)

    return {
        "imputer": imputer,
        "scaler":  scaler,
        "columns": X_train.columns.tolist(),
    }


def apply_imputer_scaler(X: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    """
    Apply a fitted imputer + scaler to a new dataset.

    Columns not present in the fitted preprocessor are silently set to NaN
    (imputer handles them). This guards against edge cases where test set
    columns differ after preprocessing-step filtering.
    """
    X_aligned = X.reindex(columns=preprocessor["columns"], fill_value=np.nan)
    X_imputed = preprocessor["imputer"].transform(X_aligned)
    X_scaled  = preprocessor["scaler"].transform(X_imputed)
    return X_scaled


def preprocess_fold(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    apply_corr_filter: bool = True,
    y_train=None,
) -> tuple:
    """
    Full preprocessing pipeline for a single (train, test) fold pair.

    Order of operations (all fitted on training, applied to test):
      1. Tier 2: Near-zero variance removal
      2. Tier 3: High-correlation filter (only if apply_corr_filter=True).
         When y_train is supplied, the per-cluster keeper is chosen by
         univariate AUROC (rank-based, y-aware, outlier-robust) instead
         of variance. See remove_high_correlation docstring.
      3. Median imputation + StandardScaling

    The apply_corr_filter flag is False for Clin, Prot, and WSI because:
      - They have ≤5 features; elastic net's L2 component handles residual
        correlation adequately without explicit pruning.
      - Removing a feature from a 3-feature space (WSI) risks a degenerate model.

    Returns
    -------
    X_train_proc : np.ndarray  (processed training features)
    X_test_proc  : np.ndarray  (processed test features)
    preprocessor : dict        (fitted imputer + scaler for reproducibility)
    removed_nzv  : list        (Tier 2 removed features)
    removed_corr : list        (Tier 3 removed features)
    final_cols   : list        (feature names after all preprocessing)
    """
    # Step 1: NZV removal (Tier 2)
    X_tr2, X_te2, removed_nzv = remove_near_zero_variance(X_train, X_test)

    # Step 2: Correlation filter (Tier 3) — only for high-dimensional modalities
    if apply_corr_filter and X_tr2.shape[1] > 3:
        X_tr3, X_te3, removed_corr = remove_high_correlation(
            X_tr2, X_te2, y_train=y_train)
    else:
        X_tr3, X_te3, removed_corr = X_tr2, X_te2, []

    # Step 3: Imputation + scaling (fitted on training, applied to test)
    preprocessor = fit_imputer_scaler(X_tr3)
    X_train_proc = apply_imputer_scaler(X_tr3, preprocessor)
    X_test_proc  = apply_imputer_scaler(X_te3, preprocessor)

    return (
        X_train_proc,
        X_test_proc,
        preprocessor,
        removed_nzv,
        removed_corr,
        preprocessor["columns"],  # final feature names
    )





# ==============================================================================
# SECTION 3d — FOLD-LEVEL METRICS
# ==============================================================================

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute discrimination and calibration metrics for a single outer fold.

    Metrics
    -------
    AUROC       : Area under the ROC curve. Primary discrimination metric.
    AUPRC       : Area under the precision-recall curve. Complementary to
                  AUROC; more sensitive to performance on the positive class.
    Brier       : Proper scoring rule (calibration + discrimination jointly).
    Sensitivity : True positive rate at the Youden-optimal threshold.
                  Youden index = Sensitivity + Specificity - 1 (maximised).
                  Using the fold-specific optimal threshold avoids committing
                  to a fixed operating point and gives the best-achievable
                  sensitivity/specificity pair for the model in that fold.
    Specificity : True negative rate at the same Youden-optimal threshold.

    Note: Sensitivity and Specificity are threshold-dependent. The Youden-
    optimal threshold maximises their sum and is the standard reporting choice
    for binary classifiers when no clinical cost asymmetry is specified.
    """
    if len(np.unique(y_true)) < 2:
        print("  [WARN] Degenerate fold: only one class in y_true")
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                "Sensitivity": np.nan, "Specificity": np.nan,
                "Threshold": np.nan}

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Youden index: maximise TPR + TNR - 1  ≡  maximise TPR - FPR
    youden_idx  = np.argmax(tpr - fpr)
    best_thresh = float(thresholds[youden_idx])
    sensitivity = float(tpr[youden_idx])
    specificity = float(1.0 - fpr[youden_idx])

    return {
        "AUROC":       float(roc_auc_score(y_true, y_pred)),
        "AUPRC":       float(average_precision_score(y_true, y_pred)),
        "Brier":       float(brier_score_loss(y_true, y_pred)),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold":   best_thresh,
    }


def compute_pooled_metrics(fold_list) -> dict:
    """
    Compute AUROC / Sensitivity / Specificity / Threshold on predictions
    POOLED across all folds (concatenated y_test, y_pred) rather than as
    the mean of per-fold values.

    Why this matters for Sens/Spec specifically
    -------------------------------------------
    Per-fold Sens/Spec are computed at the Youden-optimal threshold of
    that fold. With only ~30 test patients per fold, the Youden threshold
    is high-variance and optimistically chosen: it's the pair of
    (TPR, TNR) that happens to maximise TPR+TNR-1 on that specific fold's
    ROC curve. Averaging per-fold Sens/Spec therefore reports the
    *upper envelope* of achievable operating points, not the operating
    point you'd get if you deployed the model.

    Pooling y_test and y_pred across folds first, then picking ONE Youden
    threshold on the aggregated data, gives:
      - a single, reproducible threshold ("deploy this one")
      - a single Sens/Spec pair with honest variance (~N_total patients
        worth of statistics rather than mean of 1000 × 30-patient estimates)
      - a meaningful "Threshold" to report in the paper's operating-point
        table

    AUROC is threshold-free and is essentially unchanged between per-fold
    mean and pooled computation. AUPRC and Brier shift slightly; the
    pooled values are closer to what a real deployment would achieve.

    Returns same keys as compute_fold_metrics. Returns NaN dict if pooled
    data is degenerate.
    """
    if not fold_list:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                "Sensitivity": np.nan, "Specificity": np.nan,
                "Threshold": np.nan, "N_pooled": 0}

    y_true = np.concatenate([np.asarray(f["y_test"], dtype=float)
                             for f in fold_list])
    y_pred = np.concatenate([np.asarray(f["y_pred"], dtype=float)
                             for f in fold_list])

    pooled = compute_fold_metrics(y_true, y_pred)
    pooled["N_pooled"] = int(len(y_true))
    return pooled

# ==============================================================================
# SECTION 3c — SIGNATURE DISCOVERY: HELPER FUNCTIONS
# ==============================================================================

def _map_cc_splits_to_expanded(inner_splits, cc_train_pids, df_mod_train):
    """
    Convert inner CV splits from cc-relative indices to positions in the
    expanded training array (all modality patients minus outer test patients).

    The inner splits (i_tr_rel, i_va_rel) index into the cc training set
    of size len(cc_train_pids). We convert i_va_rel to absolute positions
    in df_mod_train (= X_tr_p row order) so that GridSearchCV receives
    validation indices that correctly identify cc patients in the expanded
    array, while all other expanded patients form the inner training set.

    This ensures:
      - Inner validation = cc patients only  → OOF scores always available
        for fusion regardless of which modality is being trained.
      - Inner training = full expanded set minus cc val → maximum data.

    Parameters
    ----------
    inner_splits  : list of (i_tr_rel, i_va_rel) — cc-relative int arrays
    cc_train_pids : np.array — patient IDs for cc training patients (ordered)
    df_mod_train  : pd.DataFrame with 'patient_id'; row order matches X_tr_p

    Returns
    -------
    list of (i_tr_exp, i_va_exp) as np.int64 arrays into X_tr_p rows.
    Empty list if no cc validation patient is found in df_mod_train.
    """
    mod_pids      = df_mod_train["patient_id"].values
    n_expanded    = len(mod_pids)
    pid_to_exppos = {int(p): pos for pos, p in enumerate(mod_pids)}

    mapped = []
    for i_tr_rel, i_va_rel in inner_splits:
        # Convert cc inner val relative indices → expanded positions
        i_va_exp = np.array(
            [pid_to_exppos[int(cc_train_pids[i])]
             for i in i_va_rel
             if int(cc_train_pids[i]) in pid_to_exppos],
            dtype=np.int64)

        if len(i_va_exp) == 0:
            continue  # no cc val patient in this modality (should not occur for cc)

        # Inner training = all expanded rows except cc inner val
        val_set  = set(i_va_exp.tolist())
        i_tr_exp = np.fromiter(
            (i for i in range(n_expanded) if i not in val_set),
            dtype=np.int64, count=n_expanded - len(val_set))

        mapped.append((i_tr_exp, i_va_exp))

    return mapped


def _compute_inner_importance(clf_name, model, X_train_p, fcols_inner_list):
    """
    Compute per-feature importance for one fitted classifier on inner
    TRAINING data (NOT validation — this avoids signature-selection optimism
    where the signature is tuned to the val-fold feature distribution and
    then scored on that same val fold in Stage A Pass 2).

    Linear classifiers (ElasticNet_LR, SVM_Linear):
      importance[f] = 1.0 if |coef[f]| > 1e-8, else 0.0
      Aggregated across K inner folds this gives selection frequency ∈ [0,1].

    Tree classifiers (RandomForest, ExtraTrees, HistGradBoost):
      importance[f] = mean |SHAP| over inner TRAINING patients
      Uses shap.TreeExplainer with model_output='probability'.
      Computed on inner-training rather than inner-val; this is slightly more
      expensive but removes the structural dependence of the derived signature
      on the val-fold feature distribution. Pass 2's scoring on inner-val is
      therefore a clean hold-out for the signature, not a re-use of the same
      set that drove signature construction.

    SVM_RBF (shap_type='none'):
      Returns {} — classifier excluded from signature discovery.

    Returns
    -------
    dict {feature_name: importance_value}  (empty on failure)
    """
def _compute_inner_importance(clf_name, model, X_train_p, fcols_inner_list):
    """
    Compute per-feature importance for one fitted classifier on inner
    TRAINING data (NOT validation — this avoids signature-selection optimism
    where the signature is tuned to the val-fold feature distribution and
    then scored on that same val fold in Stage A Pass 2).

    Linear classifiers (ElasticNet_LR, SVM_Linear):
      importance[f] = |coef[f]| / max(|coef|)   — normalised coefficient magnitude.

      Rationale for normalised magnitude rather than binary selection (0/1):
        Binary selection discards coefficient magnitude completely — every
        selected feature receives importance=1.0 regardless of whether its
        coefficient is 0.80 or 0.005. When converted to percentile ranks in
        Stage A Pass 1, every selected feature gets an identical rank, making
        the 25th-percentile filter in _derive_signature entirely ineffective
        for linear classifiers: it cannot distinguish a strongly weighted
        feature from a weakly weighted one.

        Normalised |coef| (divided by this fold's maximum |coef|) preserves
        relative magnitude in a scale-invariant way. Features with larger
        regularised weights consistently receive higher ranks across folds.
        Zero-coefficient features map to importance=0.0 and rank at the
        bottom, preserving the implicit selection behaviour of L1
        regularisation. The normalisation is fold-local, consistent with
        tree classifiers where SHAP values are also fold-local and
        immediately converted to percentile ranks.

    Tree classifiers (RandomForest, ExtraTrees, HistGradBoost):
      importance[f] = mean |SHAP| over inner TRAINING patients.
      Uses shap.TreeExplainer with model_output='probability'.
      Computed on inner-training rather than inner-val to remove the
      structural dependence of the derived signature on the val-fold feature
      distribution. Pass 2's scoring on inner-val is therefore a clean
      hold-out for the signature.

    SVM_RBF (shap_type='none'):
      Returns {} — classifier excluded from signature discovery.

    Returns
    -------
    dict {feature_name: importance_value}  (empty on failure)
    """
    stype = CLASSIFIERS.get(clf_name, {}).get("shap_type", "none")
    imp   = {}
    try:
        if stype in ("linear", "linear_svm"):
            abs_coefs = np.abs(model.coef_[0].astype(float))
            max_coef  = float(abs_coefs.max())
            # Normalise by fold-local maximum so scale is invariant to
            # regularisation strength. Degenerate all-zero case → uniform 0.
            denom = max_coef if max_coef > 1e-12 else 1.0
            for feat, ac in zip(fcols_inner_list, abs_coefs):
                imp[feat] = float(ac) / denom

        elif stype == "tree":
            exp = shap.TreeExplainer(model, data=X_train_p,
                                     feature_names=fcols_inner_list,
                                     model_output="probability")
            sv = exp.shap_values(X_train_p)
            if isinstance(sv, list): sv = sv[1]
            elif sv.ndim == 3:       sv = sv[:, :, 1]
            mean_abs = np.abs(sv).mean(axis=0)
            for feat, val in zip(fcols_inner_list, mean_abs):
                imp[feat] = float(val)
        # stype == "none": return {} → feature never selected → clf never wins
    except Exception as e:
        print(f"  [WARN] _compute_inner_importance failed for {clf_name}: "
              f"{type(e).__name__}: {e}")
    return imp


def _derive_signature(importance_dict, mod, n_events_expanded):
    """
    Select the feature signature from cross-classifier percentile-rank importance.

    Rules by modality:

    ── Small modalities: Clin, WSI ─────────────────────────────────────────────
    Keep ALL features. These modalities have ≤5 features; the elastic net's
    L1/L2 regularisation handles non-informative features by shrinking their
    coefficients toward zero. Removing any feature from a 3-feature space (WSI)
    risks a degenerate model and provides no methodological benefit.

    ── High-dimensional modalities: RNA, DNA, Prot ─────────────────────────────
    Three constraints applied in sequence:

    1. EPV ceiling: max_k = max(floor(n_pCR_events / EPV=5), FLOOR=5).
       Hard upper bound grounded in the events-per-variable literature, adjusted
       for regularised models (EPV=5 vs the classical EPV=10 for OLS/unregularised
       logistic regression). FLOOR=5 ensures a minimum of 5 features regardless
       of the EPV cap, including the T-DM1/Prot case where EPV=5 gives 4.

    2. 25th-percentile filter within cap: among the top max_k features by
       mean cross-classifier percentile-rank importance, drop those whose score
       falls below the 25th percentile of the retained set. This removes the
       bottom quartile — features the classifiers consistently ranked as least
       informative — while retaining all features with meaningful cross-classifier
       consensus.

    3. Floor protection: if the percentile filter would reduce the set below
       FLOOR=5, restore the top FLOOR features by importance rank regardless
       of the percentile threshold. This prevents over-pruning in arm scenarios
       where the importance distribution is compressed.

    Parameters
    ----------
    importance_dict   : {feature_name: mean_cross_classifier_percentile_rank}
                        Keys are features surviving outer NZV/corr preprocessing.
    mod               : str — modality name (Clin, RNA, DNA, Prot, WSI).
    n_events_expanded : int — pCR=1 count in the expanded outer training set.

    Returns
    -------
    list of feature names in descending importance order.
    """
    if not importance_dict:
        return []

    SMALL_MODS = {"Clin", "WSI"}
    EPV   = 5
    FLOOR = 5

    ranked = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)

    if mod in SMALL_MODS:
        return [f for f, _ in ranked]

    # ── EPV cap (with floor) ──────────────────────────────────────────────────
    max_k   = max(int(n_events_expanded // EPV), FLOOR)
    capped  = ranked[:max_k]

    if len(capped) <= FLOOR:
        return [f for f, _ in capped]

    # ── 25th-percentile filter within cap ────────────────────────────────────
    vals    = np.array([v for _, v in capped])
    p25     = float(np.percentile(vals, 25))
    filtered = [(f, v) for f, v in capped if v >= p25]

    # ── Floor protection ─────────────────────────────────────────────────────
    if len(filtered) < FLOOR:
        filtered = capped[:FLOOR]

    return [f for f, _ in filtered]


def _check_calibration(y_true_list, y_pred_list,
                       clf_name, mod, exp_name, fold_idx):
    """
    Estimate Platt calibration slope from inner-loop OOF predictions —
    DIAGNOSTIC ONLY. Since the pipeline now always applies calibration
    (see `_apply_global_calibration` below), this function is retained
    for its slope/Brier diagnostics in the terminal log; its return
    values are recorded in `fold_dict["calibration"]` for transparency
    but are no longer used as a gate.

    Fits LogisticRegression(C=1e6) on predicted_prob → true_label to obtain
    the Platt slope:
      slope ≈ 1.0 : well calibrated
      slope < 0.80: probabilities compressed  (RF/ET typical)
      slope > 1.20: probabilities overconfident

    Returns
    -------
    slope       : float  — Platt calibration slope
    needs_platt : bool   — True iff slope ∉ [0.80, 1.20]  (recorded but unused)
    diag        : str    — formatted diagnostic line for terminal output
    """
    y_true = np.array(y_true_list, dtype=float)
    y_pred = np.array(y_pred_list, dtype=float)

    tag = f"[CAL] {exp_name}/{mod} fold={fold_idx+1:04d} clf={clf_name:<15}"

    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return 1.0, False, f"{tag} → insufficient data (n={len(y_true)})"

    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(y_pred.reshape(-1, 1), y_true)
        slope = float(cal.coef_[0][0])

        brier_raw = float(brier_score_loss(y_true, y_pred))
        cal_prob  = cal.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        brier_cal = float(brier_score_loss(y_true, cal_prob))
        delta_b   = brier_raw - brier_cal

        needs_platt = not (0.80 <= slope <= 1.20)
        status      = ("COMPRESSED"    if slope < 0.80
                       else "OVERCONF" if slope > 1.20
                       else "OK")
        action      = " ★ Platt APPLIED" if needs_platt else ""
        diag = (f"{tag} slope={slope:.3f} [{status}]  "
                f"Brier {brier_raw:.4f}→{brier_cal:.4f} "
                f"(Δ={delta_b:+.4f}){action}")
        return slope, needs_platt, diag

    except Exception as e:
        return 1.0, False, f"{tag} → failed: {e}"


def _fit_global_platt(y_true, y_pred_raw):
    """
    Fit a 2-parameter Platt sigmoid on (raw predicted probability → label)
    using all available inner-OOF data for the winner classifier. Returns
    a fitted LogisticRegression on a single feature (raw score).

    Why global OOF calibration instead of nested CalibratedClassifierCV:
    - The calibrator sees ALL cc-training-fold OOF predictions (~80-120
      patients) instead of being refit inside cv=3 splits of ~30 patients.
    - Sigmoid has 2 parameters (slope + intercept) — 30 patients is
      noticeably underpowered for this; 100+ is stable.
    - Applied uniformly to every modality: all OOF columns entering fusion
      are on the same (calibrated) probability scale, removing the
      heteroscedasticity that arises when some modalities are Platt-wrapped
      and others are not.

    Assumption: the calibration curve of the outer-refit model is close to
    that of the inner-fold models. For a 2-parameter sigmoid this is
    usually a safe assumption — slope/intercept calibration captures
    systematic, estimator-family miscalibration that is fairly stable
    across training set sizes.

    Returns None if there is insufficient data or only one class present.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_raw, dtype=float)
    # Guard rails
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return None
    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(y_pred.reshape(-1, 1), y_true)
        return cal
    except Exception:
        return None


def _apply_global_platt(calibrator, y_pred_raw):
    """Apply a fitted Platt calibrator to an array of raw probabilities."""
    if calibrator is None:
        return np.asarray(y_pred_raw, dtype=float)
    y = np.asarray(y_pred_raw, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(y)[:, 1]


def make_oof_signature(clf_name, best_params, signature_feats,
                       cc_train_raw_df, y_cc_train, cc_train_pids,
                       mod_full_df, feat_cols_raw, test_pids_set,
                       inner_splits, ac,
                       fold_cache=None, inner_jobs=1):
    """
    Generate RAW (uncalibrated) OOF probability scores for the complete-case
    training patients using the winner classifier on the winner signature
    features.

    CALIBRATION IS APPLIED SEPARATELY.
    The caller (_fit_signature_model) fits a global Platt sigmoid on
    (raw_OOF, y_cc_train) after this function returns, then applies it to
    both OOF and outer-test predictions. This centralises calibration,
    gives the sigmoid maximum training data (~100 patients vs ~30 in the
    former nested cv=3 approach), and ensures all modality OOF columns
    entering fusion are on the same calibrated scale.

    PERFORMANCE NOTE
    ----------------
    Stage A Pass 1 already preprocessed each inner fold and cached the result
    in `fold_cache`. We accept that cache here and reuse the preprocessed
    arrays instead of re-running NZV → correlation filter → impute → scale,
    which is the dominant per-fold cost for RNA/DNA modalities. Only the
    model fit itself needs to be repeated (different classifier / params /
    feature subset). When `fold_cache=None` the function falls back to the
    original from-raw behaviour for backwards compatibility.

    For each inner fold:
      Inner training : all modality patients minus (outer_test ∪ inner_val_cc),
                       preprocessed from raw (or reused from fold_cache).
      Inner validation: cc training patients at inner split relative indices.
      Features used   : intersection of signature_feats with inner fold's
                        post-preprocessing column set (some features may be
                        removed by inner-fold NZV/correlation filters).

    Returns
    -------
    np.array of shape (len(y_cc_train),) — RAW OOF probabilities.
    Unfitted positions (failed inner folds) default to 0.5 (neutral).
    """
    oof    = np.full(len(y_cc_train), np.nan)
    failed = 0
    cfg    = CLASSIFIERS[clf_name]
    sig_set = set(signature_feats)
    use_cache = fold_cache is not None and len(fold_cache) == len(inner_splits)

    for fold_pos, (i_tr_rel, i_va_rel) in enumerate(inner_splits):
        if use_cache and fold_cache[fold_pos] is not None:
            # Reuse preprocessed arrays from Stage A Pass 1
            X_itr_p, y_itr, X_iva_p, _y_iva_cached, fcols_inner_list = \
                fold_cache[fold_pos]
        else:
            # Fallback: preprocess from raw
            X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
            val_pids  = set(int(p) for p in cc_train_pids[i_va_rel])
            excluded  = test_pids_set | val_pids
            itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
            X_itr_raw = mod_full_df.loc[itr_mask, feat_cols_raw]
            y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

            if len(X_itr_raw) == 0 or len(np.unique(y_itr)) < 2:
                failed += 1; continue

            try:
                X_itr_p, X_iva_p, fcols_inner = preprocess_fold_3(
                    X_itr_raw, X_iva_raw, ac, y_train=y_itr)
                fcols_inner_list = list(fcols_inner)
            except Exception:
                failed += 1; continue

        # Intersect signature with inner-fold surviving features
        sig_idx = [i for i, f in enumerate(fcols_inner_list) if f in sig_set]
        if len(sig_idx) == 0:
            failed += 1; continue

        try:
            X_itr_sig = X_itr_p[:, sig_idx]
            X_iva_sig = X_iva_p[:, sig_idx]

            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(
                clf_name, best_params, inner_jobs))
            # No CalibratedClassifierCV wrap here — produces RAW OOF.
            # Calibration is applied globally by the caller.

            m.fit(X_itr_sig, y_itr)
            oof[i_va_rel] = m.predict_proba(X_iva_sig)[:, 1]

        except Exception:
            failed += 1

    if failed == len(inner_splits):
        print(f"  [WARN] make_oof_signature: all inner folds failed "
              f"for {clf_name} — using neutral 0.5")
    return np.where(np.isnan(oof), 0.5, oof)


def _neutral_fold_result(y_te, y_cc_train, fcols_list, reason="all classifiers failed"):
    """Return neutral (0.5) predictions with complete fold_dict structure."""
    print(f"  [WARN] _neutral_fold_result: {reason}")
    # Neutral cross-arm predictor: returns NaN for any input → filtered out
    # downstream. Keeps the dict shape consistent.
    def _neutral_predict(X_raw_df):
        return np.full(len(X_raw_df), np.nan)
    return {
        "metrics":          compute_fold_metrics(y_te, np.full(len(y_te), 0.5)),
        "y_test":           y_te,
        "y_pred":           np.full(len(y_te),       0.5),
        "winner_clf":       "none",
        "winner_signature": [],
        "signature_size":   0,
        "n_events_inner":   0,
        "inner_cv_aurocs_A": {},   # match _fit_signature_model key name
        "inner_cv_auroc_B":  0.0,
        "stage_b_status":    "fallback_stage_a",
        "inner_cv_params":   {},
        "inner_importance": {},
        "calibration":      {},
        "platt_applied":    False,
        "features":         fcols_list,
        "oof_shap":         None,
        "_oof":             np.full(len(y_cc_train), 0.5),
        "_cross_arm_predict": _neutral_predict,
    }


def _params_with_inner_jobs(clf_name, params, inner_jobs):
    """
    Inject n_jobs=inner_jobs only for tree-ensemble classifiers that actually
    support it (RandomForest, ExtraTrees). Returned dict is safe to pass to
    estimator.set_params(**...).

    HistGradientBoosting does not expose n_jobs (single-threaded by design);
    SVM and LogisticRegression(saga) also do not benefit from this flag.
    """
    if clf_name in ("RandomForest", "ExtraTrees") and inner_jobs > 1:
        p = dict(params)
        p["n_jobs"] = inner_jobs
        return p
    return params


def _fit_signature_model(
    X_tr_p, y_tr_mod,
    X_te_p, y_te,
    fcols,
    df_mod_train,
    inner_splits,
    cc_train_pids,
    y_cc_train,
    cc_train_raw_df,
    mod_full_df,
    feat_cols_raw,
    test_pids_set,
    ac,
    active_clfs,
    n_events_expanded,
    mod, exp_name, fold_idx,
    inner_jobs=1,
    outer_prep=None,
    outer_feat_cols_raw=None,
):
    """
    Multi-classifier signature discovery pipeline (primary analysis).

    STAGE A — Classifier comparison with fixed parameters
    -------------------------------------------------------
    For each inner fold:
      1. Build expanded inner training set and cc inner validation set.
      2. Preprocess (Tier 2 + Tier 3 + imputer + scaler) on expanded inner
         training; apply to cc inner validation.
      3. For each C_i ∈ {ElasticNet, RF, ET, HGB, SVM_Lin}:
           Fit with STAGE_A_PARAMS → AUROC on cc inner val.
           Compute feature importance (selection freq / mean |SHAP|).
           Accumulate calibration OOF predictions.

    After K inner folds per C_i:
      - mean_auroc_A_i    (AUROC on cc inner val, Stage A params)
      - importance_A_i    (normalised across K folds)
      - signature_A_i     (EPV-capped, min 5)
      - calibration_i     (Platt slope + needs_platt flag)

    STAGE B — Winner tuning
    -----------------------
    Winner = argmax_i mean_auroc_A_i  (among classifiers with valid signature).
    GridSearchCV on X_tr_p using cc_inner_splits_expanded → best_params_winner.
    Outer refit on winner_signature features; Platt-calibrate if slope ∉ [0.80,1.20].
    OOF via make_oof_signature (expanded inner train, same model+signature).
    SHAP on signature features.

    NO LEAKAGE:
    - Outer test set (X_te_p, y_te) is only used for final prediction.
    - All model selection, feature ranking, and tuning use inner splits.
    - Calibration assessed from inner-loop OOF predictions only.
    - Preprocessing in every inner fold is fitted on inner training only.

    Parameters
    ----------
    X_tr_p          : np.ndarray — outer-preprocessed expanded training features
    y_tr_mod        : np.array  — pCR labels for expanded training patients
    X_te_p          : np.ndarray — outer-preprocessed test features
    y_te            : np.array  — pCR labels for test patients
    fcols           : sequence  — feature names after outer preprocessing
    df_mod_train    : pd.DataFrame — expanded training patients (patient_id, pCR, feats)
    inner_splits    : list of (i_tr_rel, i_va_rel) — cc-relative indices
    cc_train_pids   : np.array  — patient IDs for cc training patients (ordered)
    y_cc_train      : np.array  — pCR labels for cc training patients (same order)
    cc_train_raw_df : pd.DataFrame — raw modality features for cc training patients
    mod_full_df     : pd.DataFrame — all modality patients (patient_id, pCR, feats)
    feat_cols_raw   : list — raw feature column names (without patient_id/pCR)
    test_pids_set   : set  — outer test patient IDs (always excluded from training)
    ac              : bool — apply Tier 3 correlation filter
    active_clfs     : list — classifiers to evaluate (SVM_RBF excluded externally)
    n_events_expanded : int — pCR=1 count in expanded outer training (for EPV)
    mod, exp_name, fold_idx : str/int — for terminal diagnostic output

    Returns
    -------
    dict with complete fold result including signature metadata and _oof key.
    """
    fcols_list = list(fcols)
    # SVM_RBF excluded from signature: no SHAP capability
    sig_clfs = [c for c in active_clfs if c != "SVM_RBF"]

    if not sig_clfs:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "no eligible classifiers")

    # ── STAGE A — PASS 1: Cross-classifier percentile-rank importance ─────────
    # For each inner fold: fit each classifier, compute importance,
    # convert to percentile ranks within that fold/classifier, accumulate.
    # This makes importance scale-invariant across classifier types:
    # selection frequencies (linear) and mean|SHAP| (tree) are both
    # mapped to [0,1] rank space before averaging.
    #
    # Also cache preprocessed inner fold data for Pass 2 (pruned evaluation).
    # Calibration OOF predictions accumulated here for Platt check.

    # Per-classifier accumulators (percentile-rank importance)
    rank_acc   = {c: defaultdict(float) for c in sig_clfs}
    cal_acc    = {c: {"y_true": [], "y_pred": []} for c in sig_clfs}
    n_success  = {c: 0 for c in sig_clfs}

    # Cache for Pass 2 — list of (X_itr_p, y_itr, X_iva_p, y_iva, fcols_inner)
    fold_cache = []

    for i_tr_rel, i_va_rel in inner_splits:

        # ── Inner validation: cc training patients ────────────────────────────
        X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
        y_iva     = y_cc_train[i_va_rel]
        val_pids  = set(int(p) for p in cc_train_pids[i_va_rel])

        # ── Inner training: expanded modality patients ────────────────────────
        excluded  = test_pids_set | val_pids
        itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
        X_itr_raw = mod_full_df.loc[itr_mask, feat_cols_raw]
        y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

        if (len(X_itr_raw) < 5 or len(np.unique(y_itr)) < 2
                or len(np.unique(y_iva)) < 2):
            fold_cache.append(None)
            continue

        # ── Preprocess once per inner fold ───────────────────────────────────
        try:
            X_itr_p_i, X_iva_p_i, fcols_i = preprocess_fold_3(
                X_itr_raw, X_iva_raw, ac, y_train=y_itr)
        except Exception:
            fold_cache.append(None)
            continue

        fcols_i_list = list(fcols_i)
        n_feats_i    = len(fcols_i_list)
        fold_cache.append((X_itr_p_i, y_itr, X_iva_p_i, y_iva, fcols_i_list))

        # ── Fit each classifier, compute percentile-rank importance ───────────
        for clf_name in sig_clfs:
            if clf_name not in STAGE_A_PARAMS:
                continue
            cfg = CLASSIFIERS[clf_name]
            try:
                m = cfg["build"]()
                m.set_params(**_params_with_inner_jobs(
                    clf_name, STAGE_A_PARAMS[clf_name], inner_jobs))
                m.fit(X_itr_p_i, y_itr)

                y_val_pred = m.predict_proba(X_iva_p_i)[:, 1]
                cal_acc[clf_name]["y_true"].extend(y_iva.tolist())
                cal_acc[clf_name]["y_pred"].extend(y_val_pred.tolist())
                n_success[clf_name] += 1

                # Raw importance for this fold — computed on INNER TRAINING
                # features (not inner val) to avoid signature-selection
                # optimism in Stage A Pass 2. SHAP and coef-based importance
                # remain well-defined on training data.
                imp_raw = _compute_inner_importance(
                    clf_name, m, X_itr_p_i, fcols_i_list)

                # Convert to percentile ranks (scale-invariant across classifiers)
                if imp_raw and n_feats_i > 1:
                    vals_arr = np.array(list(imp_raw.values()), dtype=float)
                    # Percentile rank: 0 = least important, 1 = most important
                    ranks    = (np.argsort(np.argsort(vals_arr)) + 1) / n_feats_i
                    for feat, rank in zip(imp_raw.keys(), ranks):
                        rank_acc[clf_name][feat] += float(rank)
                elif imp_raw:
                    for feat in imp_raw:
                        rank_acc[clf_name][feat] += 1.0

            except Exception:
                pass

    # Average percentile ranks across successful inner folds
    mean_rank = {}
    for clf_name in sig_clfs:
        ns = n_success[clf_name]
        if ns > 0:
            mean_rank[clf_name] = {
                f: rank_acc[clf_name][f] / ns
                for f in rank_acc[clf_name]
            }
        else:
            mean_rank[clf_name] = {}

    # ── Derive signature per classifier using new pruning rules ───────────────
    signatures = {
        clf: _derive_signature(mean_rank[clf], mod, n_events_expanded)
        for clf in sig_clfs
    }

    # ── Calibration check per classifier ──────────────────────────────────────
    calibration = {}
    for clf_name in sig_clfs:
        slope, needs_platt, diag = _check_calibration(
            cal_acc[clf_name]["y_true"],
            cal_acc[clf_name]["y_pred"],
            clf_name, mod, exp_name, fold_idx)
        calibration[clf_name] = {"slope": slope, "needs_platt": needs_platt}
        print(diag, flush=True)

    # ── STAGE A — PASS 2: Evaluate PRUNED signatures on inner val folds ───────
    # Re-use cached preprocessed fold data. For each inner fold, fit the
    # pruned signature model and score on the CC inner val set.
    # Mean pruned val AUROC → winner selection criterion.
    # This validates the signature itself on held-out inner data rather
    # than selecting by the all-feature model performance.

    pruned_auroc_acc = {c: [] for c in sig_clfs}

    for fold_data in fold_cache:
        if fold_data is None:
            continue
        X_itr_p_i, y_itr, X_iva_p_i, y_iva, fcols_i_list = fold_data

        for clf_name in sig_clfs:
            sig = signatures[clf_name]
            if not sig:
                continue
            cfg = CLASSIFIERS[clf_name]
            try:
                # Select signature columns that survived inner preprocessing
                sig_set_i = set(sig) & set(fcols_i_list)
                if not sig_set_i:
                    continue
                sig_idx_i = [fcols_i_list.index(f) for f in fcols_i_list
                             if f in sig_set_i]
                X_itr_sig = X_itr_p_i[:, sig_idx_i]
                X_iva_sig = X_iva_p_i[:, sig_idx_i]

                m = cfg["build"]()
                m.set_params(**_params_with_inner_jobs(
                    clf_name, STAGE_A_PARAMS[clf_name], inner_jobs))
                m.fit(X_itr_sig, y_itr)

                if len(np.unique(y_iva)) < 2:
                    continue
                y_pred_sig = m.predict_proba(X_iva_sig)[:, 1]
                pruned_auroc_acc[clf_name].append(
                    float(roc_auc_score(y_iva, y_pred_sig)))
            except Exception:
                pass

    # Mean pruned val AUROC per classifier (winner criterion)
    mean_pruned_auroc = {
        clf: (float(np.mean(pruned_auroc_acc[clf]))
              if pruned_auroc_acc[clf] else 0.0)
        for clf in sig_clfs
    }

    # Winner = highest mean pruned inner val AUROC with a non-empty signature
    valid_clfs = [c for c in sig_clfs
                  if mean_pruned_auroc[c] > 0 and len(signatures[c]) > 0]

    if not valid_clfs:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "no classifier produced a valid pruned signature")

    winner_clf  = max(valid_clfs, key=lambda c: mean_pruned_auroc[c])
    winner_sig  = signatures[winner_clf]

    # Store both all-feature and pruned AUROCs for reporting
    mean_aurocs_A = mean_pruned_auroc   # now reflects pruned performance

    # ── STAGE B: GridSearchCV for winner on expanded training ─────────────────
    # Map inner splits to expanded training positions so that GridSearchCV
    # correctly uses expanded training for fitting and cc patients for scoring.
    cc_inner_splits_exp = _map_cc_splits_to_expanded(
        inner_splits, cc_train_pids, df_mod_train)

    best_params = STAGE_A_PARAMS.get(winner_clf, {})   # fallback if GS fails
    stage_b_cv_auroc = mean_aurocs_A[winner_clf]        # fallback if GS fails
    # stage_b_status distinguishes "tuned" (stage_b_cv_auroc is from Stage B
    # GridSearchCV with best_params) from "fallback_stage_a" (GS failed, we
    # fell back to Stage A Pass 2 pruned AUROC with STAGE_A_PARAMS). The two
    # are not directly comparable across folds; always inspect this flag
    # before using inner_cv_auroc_B in aggregate analyses.
    stage_b_status = "fallback_stage_a"

    if cc_inner_splits_exp:
        cfg = CLASSIFIERS[winner_clf]
        # refit=False: gs.best_estimator_ is NEVER used — we rebuild the
        # winner manually on the signature subset below. Skipping refit saves
        # one full fit per fold on the expanded training set.
        # n_jobs=inner_jobs: parameter combinations run in parallel when the
        # CPU budget allows (threadpool_limits=1 guarantees BLAS does not
        # oversubscribe inside these sub-fits).
        gs  = GridSearchCV(cfg["build"](), cfg["grid"],
                           cv=cc_inner_splits_exp,
                           scoring="roc_auc", refit=False,
                           n_jobs=inner_jobs)
        try:
            gs.fit(X_tr_p, y_tr_mod)
            best_params      = gs.best_params_
            stage_b_cv_auroc = float(gs.best_score_)
            stage_b_status   = "tuned"
        except Exception as e:
            print(f"  [WARN] Stage B GridSearchCV failed for {winner_clf}: {e}")

    # ── Outer refit on winner signature features (RAW, no calibration wrap) ──
    sig_set   = set(winner_sig)
    sig_idx   = [i for i, f in enumerate(fcols_list) if f in sig_set]

    if len(sig_idx) == 0:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "winner signature empty after outer preprocessing")

    X_tr_sig   = X_tr_p[:, sig_idx]
    X_te_sig   = X_te_p[:, sig_idx]
    sig_feats  = [fcols_list[i] for i in sig_idx]

    # Build the outer model WITHOUT a CalibratedClassifierCV wrap.
    # Calibration is applied globally below using a Platt sigmoid fit on
    # raw inner-OOF predictions — see _fit_global_platt for rationale.
    outer_m = CLASSIFIERS[winner_clf]["build"]()
    outer_m.set_params(**_params_with_inner_jobs(
        winner_clf, best_params, inner_jobs))
    outer_m.fit(X_tr_sig, y_tr_mod)
    y_pred_raw = outer_m.predict_proba(X_te_sig)[:, 1]

    # ── Raw OOF scores (uncalibrated) ─────────────────────────────────────────
    # Pass fold_cache so preprocessing (NZV + correlation + impute + scale)
    # is reused from Stage A Pass 1 rather than recomputed per inner fold.
    # For RNA/DNA this is typically the largest single speedup in the pipeline.
    oof_raw = make_oof_signature(
        clf_name       = winner_clf,
        best_params    = best_params,
        signature_feats= winner_sig,
        cc_train_raw_df= cc_train_raw_df,
        y_cc_train     = y_cc_train,
        cc_train_pids  = cc_train_pids,
        mod_full_df    = mod_full_df,
        feat_cols_raw  = feat_cols_raw,
        test_pids_set  = test_pids_set,
        inner_splits   = inner_splits,
        ac             = ac,
        fold_cache     = fold_cache,
        inner_jobs     = inner_jobs,
    )

    # ── Global Platt calibration (ALWAYS applied, per design) ─────────────────
    # Fit a 2-parameter sigmoid on (raw_OOF, y_cc_train) and apply it to both
    # OOF (for fusion training) and outer-test predictions. This uniformly
    # calibrates all modality OOF columns so the fusion layer sees
    # homogeneous probability inputs, regardless of whether the individual
    # modality's winner classifier naturally produces compressed (RF/ET) or
    # overconfident outputs.
    platt_cal = _fit_global_platt(y_cc_train, oof_raw)
    if platt_cal is not None:
        y_pred = _apply_global_platt(platt_cal, y_pred_raw)
        oof    = _apply_global_platt(platt_cal, oof_raw)
        platt_applied = True
    else:
        # Insufficient data to fit a calibrator (very small cc training fold
        # or single-class OOF). Fall back to raw scores.
        y_pred = y_pred_raw
        oof    = oof_raw
        platt_applied = False

    # ── SHAP on signature features ────────────────────────────────────────────
    # SHAP is computed on the uncalibrated model (outer_m); the Platt
    # sigmoid is monotonic, so SHAP feature rankings and signs are
    # unchanged by calibration — only the probability scale shifts.
    feat_shap = compute_shap(winner_clf, outer_m, X_tr_sig, X_te_sig, sig_feats)

    # ── Build a cross-arm predictor closure ───────────────────────────────────
    # Captures the fitted, calibrated unimodal model for this fold so it can
    # later be applied to raw features of patients NOT in the fold's original
    # training/test set (e.g. opposite-arm complete-case patients for the
    # counterfactual analysis). The closure is NOT pickled into the PKL —
    # it's consumed downstream in _process_single_fold_inner to compute
    # cross-arm predictions, which are then stored as plain floats.
    #
    # Importantly this uses the EXACT same preprocessing pipeline (NZV +
    # correlation filter + imputer + scaler) and the EXACT same Platt
    # calibrator as the in-arm predictions, so in-arm vs cross-arm
    # probabilities are on identical scales.
    sig_set_closure = set(sig_feats)

    def _cross_arm_predict(X_raw_df):
        """Return calibrated P(pCR) for rows of X_raw_df using this fold's
        unimodal model. X_raw_df must have the modality's raw feature columns
        (missing ones are filled with NaN and imputed via outer_prep)."""
        if outer_prep is None or outer_feat_cols_raw is None:
            return np.full(len(X_raw_df), np.nan)
        # Align to the raw feature columns this fold was trained on
        X_aligned = X_raw_df.reindex(columns=outer_feat_cols_raw,
                                     fill_value=np.nan)
        # Apply the fitted imputer + scaler (NZV/corr pruning already baked
        # into outer_prep["columns"])
        X_p = apply_imputer_scaler(X_aligned, outer_prep)
        # Select the winner signature columns
        prep_cols = outer_prep["columns"]
        sig_idx_local = [i for i, f in enumerate(prep_cols)
                         if f in sig_set_closure]
        if not sig_idx_local:
            return np.full(len(X_raw_df), np.nan)
        X_sig = X_p[:, sig_idx_local]
        # Raw probability, then Platt-calibrate if available
        p_raw = outer_m.predict_proba(X_sig)[:, 1]
        if platt_cal is not None:
            return _apply_global_platt(platt_cal, p_raw)
        return p_raw

    # ── Assemble fold result ──────────────────────────────────────────────────
    return {
        "metrics":            compute_fold_metrics(y_te, y_pred),
        "y_test":             y_te,
        "y_pred":             y_pred,
        # Signature
        "winner_clf":         winner_clf,
        "winner_signature":   sig_feats,
        "signature_size":     len(sig_feats),
        # EPV was computed from expanded-training event count (not CC).
        # Signature applies to CC test patients — therefore signature size
        # can exceed what EPV/5 would suggest from CC events alone.
        # This is intentional: expanded training provides the events.
        "n_events_inner":     n_events_expanded,
        # Per-classifier results
        "inner_cv_aurocs_A":  mean_aurocs_A,
        "inner_cv_auroc_B":   stage_b_cv_auroc,
        "stage_b_status":     stage_b_status,     # "tuned" | "fallback_stage_a"
        "inner_cv_params":    best_params,
        "inner_importance":   {c: dict(mean_rank[c]) for c in sig_clfs},
        "signatures_all":     signatures,
        "calibration":        calibration,        # diagnostic only
        "platt_applied":      platt_applied,      # True unless calibrator fit failed
        # SHAP
        "features":           sig_feats,
        "oof_shap":           feat_shap,
        # OOF (popped in run_experiment before writing to PKL)
        "_oof":               oof,
        # Cross-arm predictor closure — consumed by _process_single_fold_inner
        # before PKL serialisation (also underscore-prefixed = transient).
        "_cross_arm_predict": _cross_arm_predict,
    }



# ==============================================================================
# SECTION 4b — LEGACY MODEL LOGIC (SHAP, INNER CV, OOF, FUSION)
# ==============================================================================
# Used by supplementary modes (best_per_fold, ensemble_weighted).


def preprocess_fold_3(X_tr_df, X_te_df, apply_corr=True, y_train=None):
    """Thin wrapper: returns (X_train, X_test, feature_names) from 6-value preprocess_fold.

    y_train is forwarded to the correlation filter so it can use y-aware
    (univariate AUROC) keeper selection. Without y_train, the filter falls
    back to variance-based keeper selection.
    """
    X_tr_p, X_te_p, _, _, _, fcols = preprocess_fold(
        X_tr_df, X_te_df, apply_corr_filter=apply_corr, y_train=y_train)
    return X_tr_p, X_te_p, list(fcols)


def preprocess_fold_3_with_prep(X_tr_df, X_te_df, apply_corr=True, y_train=None):
    """
    Same as preprocess_fold_3 but also returns the fitted preprocessor dict
    (imputer + scaler + final columns after NZV/correlation filtering).

    Used when the outer-fold model needs to predict on NEW raw data later —
    e.g. cross-arm counterfactual prediction, where a DHP-trained model must
    transform T-DM1 patients' raw features through the same preprocessing
    that its training set saw.

    Returns
    -------
    X_train_proc : np.ndarray
    X_test_proc  : np.ndarray
    feature_names : list[str]
    preprocessor  : dict with keys 'imputer', 'scaler', 'columns'
    """
    X_tr_p, X_te_p, prep, _, _, fcols = preprocess_fold(
        X_tr_df, X_te_df, apply_corr_filter=apply_corr, y_train=y_train)
    return X_tr_p, X_te_p, list(fcols), prep

def compute_shap(clf_name, model, X_train, X_test, feature_names):
    """
    Feature-level SHAP for a fitted unimodal model on outer test patients.

    Handles CalibratedClassifierCV wrappers (applied when Platt scaling is
    needed): SHAP is computed on the first base estimator inside the wrapper,
    using the full outer training set as the background distribution.

    Linear models (ElasticNet_LR): shap.LinearExplainer (exact).
    Tree models (RF, ET, HGB):     shap.TreeExplainer (exact, prob output).
    SVM_Linear:                    coefficient-based approximation.
    SVM_RBF:                       None (KernelSHAP too slow for production).

    Returns dict or None on failure.
    """
    stype = CLASSIFIERS.get(clf_name, {}).get("shap_type", "none")

    # Unwrap Platt calibration: use first base estimator for SHAP.
    # The base estimator captures feature importance faithfully;
    # the calibration layer only transforms the output probability.
    shap_model = model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        shap_model = model.calibrated_classifiers_[0].estimator

    try:
        if stype == "linear":
            exp = shap.LinearExplainer(shap_model, X_train,
                                       feature_names=feature_names)
            sv  = exp.shap_values(X_test)
        elif stype == "tree":
            exp = shap.TreeExplainer(shap_model, data=X_train,
                                     feature_names=feature_names,
                                     model_output="probability")
            sv  = exp.shap_values(X_test)
            if isinstance(sv, list): sv = sv[1]
            elif sv.ndim == 3:       sv = sv[:, :, 1]
        elif stype == "linear_svm":
            coef = shap_model.coef_[0]
            sv   = (X_test - X_train.mean(axis=0)) * coef[np.newaxis, :]
        else:
            return None
        return {"feature_names": list(feature_names),
                "shap_values":   np.array(sv),
                "X_test_scaled": np.array(X_test)}
    except Exception:
        return None


def compute_fusion_shap(model, X_train, X_test, mod_order):
    """Modality-level SHAP for fusion LR model (5 inputs = 5 modalities)."""
    try:
        exp = shap.LinearExplainer(model, X_train, feature_names=mod_order)
        sv  = exp.shap_values(X_test)
        return {"feature_names": list(mod_order),
                "shap_values":   np.array(sv),
                "X_test_scaled": np.array(X_test)}
    except Exception:
        return None


def inner_cv_all(X_train, y_train, inner_splits, active_clfs, inner_jobs=1):
    """GridSearchCV for every active classifier. Returns {clf: result_dict}."""
    out = {}
    for name in active_clfs:
        cfg  = CLASSIFIERS[name]
        base = cfg["build"]()
        # n_jobs=inner_jobs: parameter-combination parallelism. Safe under
        # threadpool_limits(1) because sub-fits do not oversubscribe BLAS.
        gs   = GridSearchCV(base, cfg["grid"], cv=inner_splits,
                            scoring="roc_auc", refit=True, n_jobs=inner_jobs)
        try:
            gs.fit(X_train, y_train)
            out[name] = {"model":       gs.best_estimator_,
                         "params":      gs.best_params_,
                         "inner_auroc": float(gs.best_score_),
                         "cv_scores":   {str(p): float(s)
                                         for p, s in zip(
                                             gs.cv_results_["params"],
                                             gs.cv_results_["mean_test_score"])}}
        except Exception as e:
            print(f"  [WARN] {name} failed for this fold: {type(e).__name__}: {e}")
            out[name] = {"model": None, "params": {},
                         "inner_auroc": 0.0, "cv_scores": {}}
    return out


def make_oof(clf_name, params, X_raw_df, y_train, inner_splits, apply_corr,
             inner_jobs=1):
    """
    OOF scores for one (modality, classifier) using fixed hyperparams.
    Falls back to 0.5 (neutral probability) for any failed inner fold.
    """
    oof         = np.full(len(y_train), np.nan)   # nan = not yet filled
    failed      = 0
    cfg         = CLASSIFIERS[clf_name]
    for i_tr, i_va in inner_splits:
        try:
            X_itr_p, X_iva_p, _ = preprocess_fold_3(
                X_raw_df.iloc[i_tr], X_raw_df.iloc[i_va], apply_corr,
                y_train=y_train[i_tr])
            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
            m.fit(X_itr_p, y_train[i_tr])
            oof[i_va] = m.predict_proba(X_iva_p)[:, 1]
        except Exception:
            failed += 1
    if failed == len(inner_splits):
        print(f"  [WARN] make_oof: all {failed} inner folds failed "
              f"for {clf_name} — using neutral 0.5 OOF scores")
    return np.where(np.isnan(oof), 0.5, oof)   # fill unfitted indices with 0.5


def fit_fusion(oof_dict, y_train, inner_splits, mod_order, inner_jobs=1):
    """
    Fit a single Fused_ElasticNet meta-learner on the 5-column OOF matrix.

    ElasticNet (L1+L2, l1_ratio=0.5) is used rather than Ridge because:
    - L1 can zero out non-contributing modalities, producing an interpretable
      sparse weighting — a publishable finding in itself.
    - L2 handles the inherent collinearity between modality OOF predictions
      without the instability of pure L1.
    - With 5 inputs and ~46 pCR events (Global), EPV≈9 — well within the
      range where ElasticNet is stable.
    C is tuned by inner CV over FUSION_C_GRID.
    """
    X   = np.column_stack([oof_dict[m] for m in mod_order])
    base = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=L1_RATIO, max_iter=2000,
        random_state=None)
    gs = GridSearchCV(base, {"C": FUSION_C_GRID}, cv=inner_splits,
                      scoring="roc_auc", refit=True, n_jobs=inner_jobs)
    gs.fit(X, y_train)
    m     = gs.best_estimator_
    coefs = m.coef_[0]
    return {
        "Fused_ElasticNet": {
            "model":               m,
            "tuned_C":             float(gs.best_params_["C"]),
            "modality_weights":    {mod: float(c)
                                    for mod, c in zip(mod_order, coefs)},
            "selected_modalities": [mod for mod, c
                                    in zip(mod_order, coefs) if abs(c) > 1e-6],
        }
    }


def make_oof_expanded(clf_name, params,
                      cc_train_raw_df,  # pd.DataFrame: raw feat cols for cc training patients
                      y_cc_train,       # np.array: pCR for cc training patients (same order)
                      cc_train_pids,    # np.array: patient IDs for cc training patients
                      mod_full_df,      # pd.DataFrame: all modality patients (patient_id, pCR, feats)
                      feat_cols,        # list: feature column names
                      test_pids_set,    # set: patient IDs in the outer test set (always excluded)
                      inner_splits,     # list of (i_tr_rel, i_va_rel) into cc_train
                      apply_corr,       # bool: apply Tier 3 correlation filter
                      inner_jobs=1):
    """
    Generate OOF probability scores for complete-case training patients,
    using the EXPANDED modality training set in each inner fold.

    For each inner fold:
      - Inner VALIDATION: the subset of complete-case training patients at
        inner split indices (i_va_rel). pCR always available (complete-case).
      - Inner TRAINING: ALL modality-m patients EXCEPT test patients and the
        inner validation complete-case patients.

    This ensures:
      1. OOF scores are always generated for every complete-case training
         patient (the fusion model's training targets are complete-case, so
         it needs scores for all of them).
      2. Inner training uses the maximum available data per modality,
         matching the outer-fold training strategy.
      3. No leakage: test patients are always excluded from all inner fits.

    Falls back to neutral probability 0.5 for any inner fold that fails
    (degenerate folds: single-class inner validation, fit error, etc.).
    """
    oof    = np.full(len(y_cc_train), np.nan)
    failed = 0
    cfg    = CLASSIFIERS[clf_name]

    for i_tr_rel, i_va_rel in inner_splits:
        # ── Inner validation: complete-case patients ──────────────────────────
        X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
        y_iva     = y_cc_train[i_va_rel]
        val_pids  = set(cc_train_pids[i_va_rel])

        # ── Inner training: all modality patients except test + inner val ─────
        excluded  = test_pids_set | val_pids
        itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
        X_itr_raw = mod_full_df.loc[itr_mask, feat_cols]
        y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

        # Skip if degenerate (only one class in inner training)
        if len(X_itr_raw) == 0 or len(np.unique(y_itr)) < 2:
            failed += 1
            continue

        try:
            X_itr_p, X_iva_p, _ = preprocess_fold_3(
                X_itr_raw, X_iva_raw, apply_corr, y_train=y_itr)
            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
            m.fit(X_itr_p, y_itr)
            oof[i_va_rel] = m.predict_proba(X_iva_p)[:, 1]
        except Exception:
            failed += 1

    if failed == len(inner_splits):
        print(f"  [WARN] make_oof_expanded: all {failed} inner folds failed "
              f"for {clf_name} — using neutral 0.5 OOF scores")
    return np.where(np.isnan(oof), 0.5, oof)

# ==============================================================================
# SECTION 5 — EXPERIMENT RUNNER
# ==============================================================================

ALL_MODS = ["Clin", "RNA", "DNA", "Prot", "WSI"]


# =============================================================================
# SECTION 4c — PARALLEL OUTER FOLD WORKER
# =============================================================================

def _process_single_fold(fi, tr_idx, te_idx, inner_splits,
                          df_cc_exp, y_cc, features, clin_key,
                          mod_datasets, mode, active_clfs,
                          use_expanded, ac_map, exp_name,
                          inner_jobs=1, nzv_freq=None,
                          cross_arm_df=None, cross_arm_label=None):
    """
    Process one complete outer fold: all 5 modalities + fusion.

    Designed to run as an independent joblib worker — all inputs are
    read-only and passed by value (cloudpickle serialisation via loky).
    Returns a dict keyed by modality/fusion name, each holding a tuple
    (fold_result_dict, oof_score_array) so the caller can assemble the
    full results structure.

    THREAD DISCIPLINE
    -----------------
    BLAS/OpenMP thread caps are enforced THREE ways:
      1. Parent-process env vars set at module import time (pre-numpy).
      2. parallel_backend(..., inner_max_num_threads=1) on the outer Parallel.
      3. threadpool_limits(1) runtime context below — catches anything the
         first two miss (e.g. threadpools lazily initialised inside sklearn
         C extensions). This is the only method that is guaranteed effective
         once numpy is already imported.

    When inner_jobs > 1 (nested-parallelism regime, more CPUs than folds),
    tree ensembles and GridSearchCV inside this worker are allowed to use
    `inner_jobs` threads — BUT threadpool_limits=1 means those threads
    don't spawn additional BLAS threads inside.

    nzv_freq: per-experiment NZV dominant-frequency threshold override
    (0.95 for global, 0.98 for arm). Mutates this worker's copy of the
    module global before any preprocessing runs. Because loky workers
    have their own process memory space, this does not bleed between
    experiments.
    """
    if nzv_freq is not None:
        import sys
        sys.modules[__name__].NZV_FREQ_THRESHOLD = nzv_freq

    # Runtime guard: limit BLAS threads to 1 inside this worker regardless of
    # what the parent did. We still allow sklearn tree ensembles and
    # GridSearchCV to use `inner_jobs` parallel workers, because their
    # parallelism is over trees / parameter combinations, NOT over BLAS.
    with threadpool_limits(limits=1):
        return _process_single_fold_inner(
            fi, tr_idx, te_idx, inner_splits,
            df_cc_exp, y_cc, features, clin_key,
            mod_datasets, mode, active_clfs,
            use_expanded, ac_map, exp_name, inner_jobs,
            cross_arm_df, cross_arm_label)


def _process_single_fold_inner(fi, tr_idx, te_idx, inner_splits,
                                df_cc_exp, y_cc, features, clin_key,
                                mod_datasets, mode, active_clfs,
                                use_expanded, ac_map, exp_name,
                                inner_jobs,
                                cross_arm_df=None, cross_arm_label=None):
    import warnings
    warnings.filterwarnings("ignore")

    y_te          = y_cc[te_idx]
    test_pids     = set(df_cc_exp.iloc[te_idx]["patient_id"].values)
    cc_train_pids = df_cc_exp.iloc[tr_idx]["patient_id"].values
    y_cc_train    = y_cc[tr_idx]

    oof_scores = {}
    test_preds = {}
    fold_results = {}   # {mod: fold_res_dict}

    for mod in ALL_MODS:
        ac      = ac_map[mod]
        mod_key = clin_key if mod == "Clin" else mod
        cols    = [c for c in features.get(mod_key, [])
                   if c in df_cc_exp.columns]
        if not cols:
            raise RuntimeError(
                f"[{exp_name}/{mode}] No columns for modality '{mod}'.")

        X_te_df = df_cc_exp[cols].iloc[te_idx]

        if use_expanded:
            df_mod       = mod_datasets[mod_key]
            feat_cols_m  = [c for c in cols if c in df_mod.columns
                             and c not in ("patient_id", "pCR")]
            df_mod_train = df_mod[~df_mod["patient_id"].isin(test_pids)]
            X_tr_df      = df_mod_train[feat_cols_m]
            y_tr_mod     = df_mod_train["pCR"].values
            X_tr_p, X_te_p, fcols, outer_prep = preprocess_fold_3_with_prep(
                X_tr_df, X_te_df, ac, y_train=y_tr_mod)
            cc_train_raw_df = (df_cc_exp[feat_cols_m].iloc[tr_idx]
                               .reset_index(drop=True))
            n_events_exp = int(y_tr_mod.sum())

            if mode == "elasticnet":
                fold_res = _fit_signature_model(
                    X_tr_p=X_tr_p, y_tr_mod=y_tr_mod,
                    X_te_p=X_te_p, y_te=y_te, fcols=fcols,
                    df_mod_train=df_mod_train, inner_splits=inner_splits,
                    cc_train_pids=cc_train_pids, y_cc_train=y_cc_train,
                    cc_train_raw_df=cc_train_raw_df, mod_full_df=df_mod,
                    feat_cols_raw=feat_cols_m, test_pids_set=test_pids,
                    ac=ac, active_clfs=active_clfs,
                    n_events_expanded=n_events_exp,
                    mod=mod, exp_name=exp_name, fold_idx=fi,
                    inner_jobs=inner_jobs,
                    outer_prep=outer_prep,
                    outer_feat_cols_raw=feat_cols_m,
                )
            else:
                raise NotImplementedError(
                    f"Expanded training for mode={mode!r} not implemented.")

        else:
            X_tr_df = df_cc_exp[cols].iloc[tr_idx]
            y_tr    = y_cc[tr_idx]
            X_tr_p, X_te_p, fcols, outer_prep = preprocess_fold_3_with_prep(
                X_tr_df, X_te_df, ac, y_train=y_tr)

            if mode == "elasticnet":
                feat_cols_cc    = [c for c in cols
                                   if c not in ("patient_id", "pCR")]
                df_cc_mod       = df_cc_exp[["patient_id", "pCR"]
                                             + feat_cols_cc].copy()
                df_mod_train_cc = (df_cc_mod
                                   [~df_cc_mod["patient_id"].isin(test_pids)]
                                   .reset_index(drop=True))
                cc_train_raw_df_cc = (df_cc_exp[feat_cols_cc]
                                      .iloc[tr_idx].reset_index(drop=True))
                n_events_cc = int(y_tr.sum())
                fold_res = _fit_signature_model(
                    X_tr_p=X_tr_p, y_tr_mod=y_tr,
                    X_te_p=X_te_p, y_te=y_te, fcols=fcols,
                    df_mod_train=df_mod_train_cc, inner_splits=inner_splits,
                    cc_train_pids=cc_train_pids, y_cc_train=y_cc_train,
                    cc_train_raw_df=cc_train_raw_df_cc,
                    mod_full_df=df_cc_mod, feat_cols_raw=feat_cols_cc,
                    test_pids_set=test_pids, ac=ac, active_clfs=active_clfs,
                    n_events_expanded=n_events_cc,
                    mod=mod, exp_name=exp_name, fold_idx=fi,
                    inner_jobs=inner_jobs,
                    outer_prep=outer_prep,
                    outer_feat_cols_raw=feat_cols_cc,
                )
            elif mode == "best_per_fold":
                fold_res = _fit_best_per_fold(
                    X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    X_tr_df.reset_index(drop=True), ac, active_clfs,
                    inner_jobs=inner_jobs)
            elif mode == "ensemble_weighted":
                fold_res = _fit_ensemble(
                    X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    X_tr_df.reset_index(drop=True), ac, active_clfs,
                    inner_jobs=inner_jobs)

        fold_res["fold_idx"] = fi
        oof_scores[mod]  = fold_res.pop("_oof")
        test_preds[mod]  = fold_res["y_pred"]
        fold_results[mod] = fold_res

    # ── Cross-arm unimodal predictions (before fusion, uses modality closures)
    # For arm experiments only (dhp → predicts on T-DM1 patients, and
    # vice-versa). Each fold's per-modality winner + calibrator is applied to
    # every patient in the opposite arm to produce a per-patient, per-fold
    # cross-arm probability. These are stacked into a 5-column matrix and
    # pushed through this fold's fusion model to produce a fused cross-arm
    # P(pCR) per patient, on the SAME scale as in-arm P(pCR) predictions.
    cross_arm_unimodal = {}   # {mod: np.array of shape (n_cross,) }
    cross_arm_pids     = None
    do_cross_arm       = (exp_name in ("dhp", "tdm1")
                          and cross_arm_df is not None
                          and len(cross_arm_df) > 0)

    if do_cross_arm:
        cross_arm_pids = np.asarray(cross_arm_df["patient_id"].values,
                                    dtype=np.int64)
        n_cross        = len(cross_arm_pids)
        for mod in ALL_MODS:
            mod_key = clin_key if mod == "Clin" else mod
            cols    = [c for c in features.get(mod_key, [])
                       if c in cross_arm_df.columns
                       and c not in ("patient_id", "pCR")]
            predictor = fold_results[mod].get("_cross_arm_predict")
            if predictor is None or not cols:
                cross_arm_unimodal[mod] = np.full(n_cross, np.nan)
                continue
            try:
                X_cross = cross_arm_df[cols]
                p_cross = predictor(X_cross)
                cross_arm_unimodal[mod] = np.asarray(p_cross, dtype=float)
            except Exception as e:
                print(f"  [WARN] Cross-arm predict failed fold={fi} mod={mod}: "
                      f"{type(e).__name__}: {e}")
                cross_arm_unimodal[mod] = np.full(n_cross, np.nan)

    # Pop the transient closures before any PKL serialisation can touch them
    for mod in ALL_MODS:
        fold_results[mod].pop("_cross_arm_predict", None)

    # ── Fusion ────────────────────────────────────────────────────────────
    fus_fit  = fit_fusion(oof_scores, y_cc_train, inner_splits, ALL_MODS,
                          inner_jobs=inner_jobs)
    X_fus_tr = np.column_stack([oof_scores[m] for m in ALL_MODS])
    X_fus_te = np.column_stack([test_preds[m] for m in ALL_MODS])
    for fkey, fd in fus_fit.items():
        y_pf     = fd["model"].predict_proba(X_fus_te)[:, 1]
        fus_shap = compute_fusion_shap(fd["model"], X_fus_tr, X_fus_te, ALL_MODS)
        fusion_fold = {
            "fold_idx":            fi,
            "metrics":             compute_fold_metrics(y_te, y_pf),
            "y_test":              y_te,
            "y_pred":              y_pf,
            "tuned_C":             fd["tuned_C"],
            "modality_weights":    fd["modality_weights"],
            "selected_modalities": fd["selected_modalities"],
            "oof_shap":            fus_shap,
        }

        # ── Cross-arm fused prediction ────────────────────────────────────
        # Push per-modality cross-arm columns through the fitted fusion
        # model. Patients missing ANY modality column (all-NaN from a failed
        # predictor) are skipped. Stored as {patient_id: P_alt}.
        if do_cross_arm and cross_arm_unimodal:
            X_fus_cross = np.column_stack(
                [cross_arm_unimodal.get(m, np.full(n_cross, np.nan))
                 for m in ALL_MODS])
            valid_rows = ~np.any(np.isnan(X_fus_cross), axis=1)
            cross_preds = {}
            if valid_rows.any():
                try:
                    p_alt = fd["model"].predict_proba(
                        X_fus_cross[valid_rows])[:, 1]
                    valid_pids = cross_arm_pids[valid_rows]
                    cross_preds = {int(pid): float(p)
                                   for pid, p in zip(valid_pids, p_alt)}
                except Exception as e:
                    print(f"  [WARN] Cross-arm fusion predict failed "
                          f"fold={fi}: {type(e).__name__}: {e}")
            fusion_fold["cross_arm_preds"] = cross_preds
            fusion_fold["cross_arm_label"] = cross_arm_label

        fold_results[fkey] = fusion_fold

    return fold_results


def run_experiment(df_cc_exp, features, clin_key, splits, exp_name,
                   output_dir, mode, active_clfs,
                   mod_datasets=None,
                   cross_arm_df=None, cross_arm_label=None):
    """
    Run one experiment (global/dhp/tdm1) in the specified mode.

    PARALLELISATION STRATEGY:
    Outer folds are embarrassingly parallel — each fold is fully independent
    (same read-only data, independent random draws). joblib.Parallel dispatches
    N_JOBS workers, each running _process_single_fold for one outer fold.
    N_JOBS is set via --n_jobs (default: all available CPUs).

    All n_jobs=1 inside each worker (GridSearchCV, tree classifiers) to prevent
    CPU oversubscription. OMP/BLAS thread counts are set to 1 per worker.
    The loky backend (joblib default) uses process-based workers, avoiding
    Python GIL and BLAS thread-pool issues.

    EXPANDED TRAINING STRATEGY (mod_datasets provided, --training_data expanded):
    Each unimodal model trains on ALL patients who have data for that modality,
    minus the current outer test patients.

    COMPLETE-CASE-ONLY STRATEGY (mod_datasets=None, --training_data cc_only):
    All modalities train exclusively on the complete-case patients.

    Saves {exp_name}_{mode}_results.pkl and returns the results dict.
    """
    from joblib import Parallel, delayed

    y_cc       = df_cc_exp["pCR"].values
    n_folds    = len(splits)
    ac_map     = {m: (m in CORR_FILTER_MODS) for m in ALL_MODS}
    use_expanded = (mod_datasets is not None) and (mode == "elasticnet")

    print(f"\n[{exp_name.upper()} | {mode}] {n_folds} outer folds"
          + (" | expanded training" if use_expanded else "")
          + f" | n_jobs={N_JOBS}")

    if use_expanded and splits:
        _, te_idx_0, _ = splits[0]
        test_pids_0 = set(df_cc_exp.iloc[te_idx_0]["patient_id"].values)
        print(f"  Per-modality training sizes (approx., fold 1):")
        for mod in ALL_MODS:
            mod_key = clin_key if mod == "Clin" else mod
            df_mod  = mod_datasets[mod_key]
            n_train = (~df_mod["patient_id"].isin(test_pids_0)).sum()
            n_cc    = len(df_cc_exp) - len(te_idx_0)
            print(f"    {mod:<4}: {n_train:3d} patients  (was {n_cc} cc-only, +{n_train-n_cc})")

    # ── Parallel outer fold execution ──────────────────────────────────────
    # Each worker returns {mod: fold_res_dict} for one complete fold.
    #
    # CPU budget allocation:
    #   When n_jobs >= n_folds the outer loop CANNOT saturate all CPUs on its
    #   own, so we split the budget: n_outer_workers processes, each allowed
    #   inner_jobs threads for tree ensembles / GridSearchCV. This is the
    #   "CPU-rich" regime (e.g. 16 CPUs, 5 folds).
    #   When n_jobs < n_folds we use outer-only parallelism with inner=1.
    #
    # parallel_backend(..., inner_max_num_threads=1) is the correct way to
    # prevent BLAS oversubscription. The old os.environ[...] assignment
    # inside each worker ran too late (BLAS pools are initialised at
    # numpy import time).
    #
    # batch_size=1 prevents a worker from hoarding a batch of slow folds
    # (e.g. HGB-winner folds) while other workers sit idle — i.e. it
    # short-circuits the long tail of the fold distribution.
    from joblib import parallel_backend
    n_outer_workers, inner_jobs = _resolve_parallel_budget(n_folds, N_JOBS)
    # Per-experiment NZV threshold: arm cohorts (n≈50-60) need a looser
    # threshold (0.98) so low-prevalence binary mutation features are not
    # culled just for appearing in <5% of an already-small training fold.
    nzv_freq = NZV_FREQ_GLOBAL if exp_name == "global" else NZV_FREQ_ARM
    print(f"  CPU budget      : {n_outer_workers} outer worker(s) × "
          f"{inner_jobs} inner job(s) = {n_outer_workers * inner_jobs} threads")
    print(f"  NZV freq thresh : {nzv_freq} ({'global' if exp_name == 'global' else 'arm'})")
    if cross_arm_df is not None and exp_name in ("dhp", "tdm1"):
        print(f"  Cross-arm       : {cross_arm_label or 'opposite arm'} "
              f"(n={len(cross_arm_df)}) — per-fold calibrated predictions "
              f"will be saved to Fused_ElasticNet['cross_arm_preds']")

    with parallel_backend("loky", n_jobs=n_outer_workers,
                          inner_max_num_threads=1):
        fold_result_list = Parallel(
            n_jobs=n_outer_workers, verbose=5,
            batch_size=1, pre_dispatch="2*n_jobs",
        )(
            delayed(_process_single_fold)(
                fi, tr_idx, te_idx, inner_splits,
                df_cc_exp, y_cc, features, clin_key,
                mod_datasets, mode, active_clfs,
                use_expanded, ac_map, exp_name,
                inner_jobs, nzv_freq,
                cross_arm_df, cross_arm_label,
            )
            for fi, (tr_idx, te_idx, inner_splits) in enumerate(splits)
        )

    # ── Assemble results dict (preserves fold_idx order) ─────────────────
    results = {m: [] for m in ALL_MODS + ["Fused_ElasticNet"]}
    for fold_results in fold_result_list:
        for key, res in fold_results.items():
            results[key].append(res)

    # ── Pooled metrics: concat y_test / y_pred across folds, pick one ─────
    # Youden threshold on pooled predictions. This gives an honest
    # operating-point (Sens/Spec) pair; the per-fold Sens/Spec are
    # upper-envelope optimistic because each fold picks its own
    # Youden-best threshold on ~30 test patients.
    results["_pooled_metrics"] = {
        key: compute_pooled_metrics(results[key])
        for key in ALL_MODS + ["Fused_ElasticNet"]
        if results.get(key)
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = output_dir / f"{exp_name}_{mode}_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  [SAVE] {out_path.name}")

    _print_summary(mode, results)
    return results


# ── Per-mode fitting helpers ─────────────────────────────────────────────────

def _fit_elasticnet(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    # Expanded OOF parameters (primary pipeline):
                    cc_train_raw_df=None, y_cc_train=None, cc_train_pids=None,
                    mod_full_df=None, feat_cols=None, test_pids_set=None, ac=None,
                    # Legacy fallback (not used in primary pipeline):
                    X_raw_df=None,
                    inner_jobs=1):
    """
    Elastic-net LR with C tuned by inner CV and expanded-training OOF generation.

    The model is fitted on X_tr_p / y_tr (all modality patients minus test).
    OOF scores are generated for the complete-case training patients using
    make_oof_expanded, which augments inner training with all modality patients.
    This makes OOF scores consistent with the outer-fold training strategy.
    """
    base = LogisticRegression(penalty="elasticnet", solver="saga",
                              l1_ratio=L1_RATIO, max_iter=2000,
                              random_state=None)
    gs = GridSearchCV(base, {"C": ELASTICNET_C_GRID}, cv=inner_splits,
                      scoring="roc_auc", refit=True, n_jobs=inner_jobs)
    gs.fit(X_tr_p, y_tr)
    model  = gs.best_estimator_
    best_C = float(gs.best_params_["C"])

    coefs         = model.coef_[0]
    selected_mask = np.abs(coefs) > 1e-6
    y_pred        = model.predict_proba(X_te_p)[:, 1]

    # OOF generation — expanded (primary) or legacy fallback
    if mod_full_df is not None:
        oof = make_oof_expanded(
            "ElasticNet_LR", {"C": best_C},
            cc_train_raw_df, y_cc_train, cc_train_pids,
            mod_full_df, feat_cols, test_pids_set,
            inner_splits, ac, inner_jobs=inner_jobs)
    else:
        # Legacy fallback: cc-only OOF (used if expanded data not provided)
        oof = make_oof("ElasticNet_LR", {"C": best_C}, X_raw_df,
                       y_tr, inner_splits, ac, inner_jobs=inner_jobs)

    feat_shap = compute_shap("ElasticNet_LR", model, X_tr_p, X_te_p, fcols)

    return {
        "metrics":           compute_fold_metrics(y_te, y_pred),
        "y_test":            y_te,
        "y_pred":            y_pred,
        "tuned_C":           best_C,
        "cv_C_scores":       {float(C): float(s) for C, s in
                              zip(ELASTICNET_C_GRID,
                                  gs.cv_results_["mean_test_score"])},
        "features":          list(fcols),
        "coefs":             coefs.tolist(),
        "selected":          selected_mask.tolist(),
        "selected_features": [f for f, s in zip(fcols, selected_mask) if s],
        "oof_shap":          feat_shap,
        "_oof":              oof,
    }


def _fit_best_per_fold(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                       X_raw_df, ac, active_clfs, inner_jobs=1):
    """Inner CV selects best classifier; OOF and SHAP from winner."""
    clf_res  = inner_cv_all(X_tr_p, y_tr, inner_splits, active_clfs,
                            inner_jobs=inner_jobs)
    # Filter out classifiers where fitting failed (model is None)
    valid    = {c: r for c, r in clf_res.items() if r["model"] is not None}
    if not valid:
        # All classifiers failed — return neutral predictions
        neutral = np.full(len(y_te), 0.5)
        return {"metrics": compute_fold_metrics(y_te, neutral),
                "y_test": y_te, "y_pred": neutral,
                "selected_clf": "none", "inner_aurocs": {},
                "best_params": {}, "features": list(fcols),
                "oof_shap": None, "_oof": np.full(len(y_tr), 0.5)}
    best_clf = max(valid, key=lambda c: valid[c]["inner_auroc"])
    best_est = valid[best_clf]["model"]
    best_par = valid[best_clf]["params"]

    y_pred    = best_est.predict_proba(X_te_p)[:, 1]
    oof       = make_oof(best_clf, best_par, X_raw_df, y_tr, inner_splits, ac,
                         inner_jobs=inner_jobs)
    feat_shap = compute_shap(best_clf, best_est, X_tr_p, X_te_p, fcols)

    return {
        "metrics":      compute_fold_metrics(y_te, y_pred),
        "y_test":       y_te,
        "y_pred":       y_pred,
        "selected_clf": best_clf,
        "inner_aurocs": {c: clf_res[c]["inner_auroc"] for c in clf_res},
        "best_params":  best_par,
        "features":     list(fcols),
        "oof_shap":     feat_shap,
        "_oof":         oof,
    }


def _fit_ensemble(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                  X_raw_df, ac, active_clfs, inner_jobs=1):
    """AUROC-proportional ensemble of all classifiers."""
    clf_res = inner_cv_all(X_tr_p, y_tr, inner_splits, active_clfs,
                           inner_jobs=inner_jobs)
    valid   = {c: r for c, r in clf_res.items() if r["model"] is not None}
    if not valid:
        # All classifiers failed — return neutral predictions
        neutral = np.full(len(y_te), 0.5)
        return {"metrics": compute_fold_metrics(y_te, neutral),
                "y_test": y_te, "y_pred": neutral,
                "clf_weights": {}, "inner_aurocs": {},
                "features": list(fcols), "oof_shap": None,
                "_oof": np.full(len(y_tr), 0.5)}
    raw_au  = {c: max(r["inner_auroc"], 0.0) for c, r in valid.items()}
    total   = sum(raw_au.values())
    w       = ({c: raw_au[c] / total for c in raw_au} if total > 0
               else {c: 1.0 / len(valid) for c in valid})

    oof_ens  = np.zeros(len(y_tr))
    test_ens = np.zeros(len(y_te))
    shap_acc = None; X_acc = None

    for clf_name, wt in w.items():
        est = valid[clf_name]["model"]
        par = valid[clf_name]["params"]
        test_ens += wt * est.predict_proba(X_te_p)[:, 1]
        oof_ens  += wt * make_oof(clf_name, par, X_raw_df,
                                   y_tr, inner_splits, ac,
                                   inner_jobs=inner_jobs)
        sh = compute_shap(clf_name, est, X_tr_p, X_te_p, fcols)
        if sh is not None:
            sv = sh["shap_values"]
            shap_acc = wt * sv if shap_acc is None else shap_acc + wt * sv
            X_acc    = sh["X_test_scaled"]

    feat_shap = ({"feature_names": list(fcols),
                  "shap_values":   shap_acc,
                  "X_test_scaled": X_acc}
                 if shap_acc is not None else None)

    return {
        "metrics":      compute_fold_metrics(y_te, test_ens),
        "y_test":       y_te,
        "y_pred":       test_ens,
        "clf_weights":  w,
        "inner_aurocs": {c: clf_res[c]["inner_auroc"] for c in clf_res},
        "features":     list(fcols),
        "oof_shap":     feat_shap,
        "_oof":         oof_ens,
    }


def _print_summary(mode, results):
    print(f"\n  {'Model':<22}  {'AUROC':>7}  {'Sens':>7}  {'Spec':>7}  Notes")
    for mod in ALL_MODS + ["Fused_ElasticNet"]:
        folds = results.get(mod, [])
        if not folds: continue
        au = np.mean([f["metrics"]["AUROC"] for f in folds])
        sn_vals = [f["metrics"].get("Sensitivity", np.nan) for f in folds]
        sp_vals = [f["metrics"].get("Specificity", np.nan) for f in folds]
        sn = np.nanmean(sn_vals) if any(not np.isnan(v) for v in sn_vals) else float("nan")
        sp = np.nanmean(sp_vals) if any(not np.isnan(v) for v in sp_vals) else float("nan")
        note = ""
        if mod in ALL_MODS:
            if "winner_clf" in folds[0]:
                # Signature discovery mode (primary analysis)
                top = Counter(f["winner_clf"] for f in folds).most_common(1)[0]
                sig_sizes = [f.get("signature_size", 0) for f in folds]
                n_platt = sum(1 for f in folds if f.get("platt_applied", False))
                note = (f"winner={top[0]} ({top[1]/len(folds)*100:.0f}%)  "
                        f"~{int(np.mean(sig_sizes))} feats  "
                        f"Platt={n_platt/len(folds)*100:.0f}%")
            elif "selected_clf" in folds[0]:
                top = Counter(f["selected_clf"] for f in folds).most_common(1)[0]
                note = f"best={top[0]} ({top[1]/len(folds)*100:.0f}%)"
            elif "tuned_C" in folds[0]:
                cs   = [f["tuned_C"] for f in folds if f.get("tuned_C")]
                top  = Counter(cs).most_common(1)[0] if cs else (None, 0)
                nsel = int(np.mean([len(f.get("selected_features", [])) for f in folds]))
                note = f"C={top[0]} ({top[1]/len(folds)*100:.0f}%)  ~{nsel} feats"
            elif "clf_weights" in folds[0]:
                wts = {c: np.mean([f["clf_weights"].get(c, 0) for f in folds])
                       for c in folds[0]["clf_weights"]}
                top_clf = max(wts, key=wts.get) if wts else "?"
                note = f"top={top_clf} (w={wts.get(top_clf,0):.2f})"
        elif "selected_modalities" in folds[0]:
            sel = Counter(tuple(sorted(f.get("selected_modalities",[])))
                          for f in folds).most_common(1)[0][0]
            note = "sel=" + ",".join(sel) if sel else "all"
        sn_s = f"{sn:>7.3f}" if not np.isnan(sn) else "    ---"
        sp_s = f"{sp:>7.3f}" if not np.isnan(sp) else "    ---"
        print(f"  {mod:<22}  {au:>7.3f}  {sn_s}  {sp_s}  {note}")

    # ── Pooled operating-point table ─────────────────────────────────────────
    # Sens/Spec here are computed at a SINGLE Youden threshold on the
    # concatenated-across-folds (y_test, y_pred). They are the honest
    # deployment numbers; the per-fold table above is the upper envelope.
    pooled = results.get("_pooled_metrics", {})
    if pooled:
        print(f"\n  {'Model':<22}  {'AUROC':>7}  {'Sens':>7}  {'Spec':>7}  "
              f"{'Thresh':>7}  (pooled across folds)")
        for mod in ALL_MODS + ["Fused_ElasticNet"]:
            p = pooled.get(mod)
            if not p or np.isnan(p.get("AUROC", np.nan)):
                continue
            print(f"  {mod:<22}  {p['AUROC']:>7.3f}  "
                  f"{p['Sensitivity']:>7.3f}  {p['Specificity']:>7.3f}  "
                  f"{p['Threshold']:>7.3f}  N={p['N_pooled']}")

# ==============================================================================
# SECTION 6 — SPLITS MANAGEMENT
# ==============================================================================

def load_or_generate_splits(splits_dir, exp_name, y,
                             n_outer, n_repeats, n_inner, seed):
    """Load primary-pipeline splits PKL or generate fresh. Returns 3-tuple list."""
    pkl = Path(splits_dir) / f"{exp_name}_cv_splits.pkl" if splits_dir else None
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    if pkl and pkl.exists():
        with open(pkl, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict) and "outer" in raw:
            splits = []
            for fi, (tr, te) in enumerate(raw["outer"]):
                inn = raw["inner"].get(fi, raw["inner"].get(str(fi), []))
                if not inn: inn = list(skf.split(np.zeros(len(tr)), y[tr]))
                splits.append((tr, te, inn))
        elif isinstance(raw, list) and len(raw[0]) == 3:
            splits = raw
        else:
            splits = [(tr, te, list(skf.split(np.zeros(len(tr)), y[tr])))
                      for tr, te in raw]
        print(f"  [SPLITS] Loaded {len(splits)} folds from {pkl.name}")
    else:
        rskf   = RepeatedStratifiedKFold(n_splits=n_outer, n_repeats=n_repeats,
                                         random_state=seed)
        splits = [(tr, te, list(skf.split(np.zeros(len(tr)), y[tr])))
                  for tr, te in rskf.split(np.zeros(len(y)), y)]
        # Save splits for reproducibility and sharing across modes
        if splits_dir:
            out = Path(splits_dir) / f"{exp_name}_cv_splits.pkl"
            with open(out, "wb") as f:
                pickle.dump({"outer": [(tr, te) for tr, te, _ in splits],
                             "inner": {fi: inn for fi, (_, _, inn) in enumerate(splits)}},
                            f)
            print(f"  [SPLITS] Generated {len(splits)} folds → saved to {out.name}")
        else:
            print(f"  [SPLITS] Generated {len(splits)} folds (not saved — no splits_dir)")
    return splits


# ==============================================================================
# SECTION 6b — CONSENSUS MODEL FINALIZATION (R2 protocol)
# ==============================================================================
#
# The discovery CV loop (Section 5) produces a distribution of per-fold
# models — each fold has its own winner classifier, hyperparameters, and
# signature. For the Nature Cancer paper, the scientific deliverable is a
# SINGLE consensus signature per modality and a SINGLE fusion model. The
# functions here produce those deliverables in two stages:
#
#   (1) finalize_consensus()    Aggregate per-fold objects into a fixed
#                                consensus: per-modality signature (top-K by
#                                mean SHAP importance), per-modality winner
#                                classifier (modal) with modal hyperparameters.
#
#   (2) evaluate_consensus()    Honest OOF re-evaluation of the FROZEN
#                                consensus under the SAME outer-CV splits.
#                                Signature is frozen. The classifier and the
#                                fusion elastic-net are re-fit WITHIN EACH
#                                FOLD using only that fold's training data —
#                                never using test-fold outcomes. This is the
#                                R2 protocol: consensus choices carry some
#                                selection-optimism from discovery (they were
#                                chosen with knowledge of all 110 CC outcomes)
#                                but the weights and fusion coefficients are
#                                re-estimated honestly per fold, so the OOF
#                                AUROC reported is not optimistic for those
#                                estimation steps.
#
# The PRIMARY HEADLINE AUROC for the paper is the pooled-OOF AUROC of the
# fused consensus model produced by evaluate_consensus().
# ==============================================================================

def _aggregate_signature(folds, size_strategy="median",
                          df_cc=None, feat_cols=None):
    """
    Return (consensus_sig, K, mean_importance_dict).

    Consensus signature = top-K features by cluster-pooled global importance,
    with one representative selected per correlated cluster.

    PROBLEM BEING SOLVED
    ---------------------
    The per-fold Tier 3 correlation filter always keeps exactly one member
    from each correlated cluster {A, B} per fold — but different folds can
    choose different representatives because the AUROC-based keeper flips
    with the training set.  This has two consequences:

    (1) Both A and B can accumulate selection frequency and appear together
        in the raw top-K consensus list.

    (2) Even after deduplication picks A as the keeper, A's recorded
        frequency underestimates the true stability of the signal, because
        the 80 folds where B was kept (and B's SHAP was recorded instead of
        A's) contribute nothing to A's imp_sum.  The biological signal
        {A or B} was present in every fold, but A's frequency only reflects
        the subset of folds where A happened to win the per-fold AUROC
        competition.

    SOLUTION: CLUSTER-LEVEL IMPORTANCE POOLING
    -------------------------------------------
    Before ranking, build correlation clusters from the FULL complete-case
    dataset (df_cc) using the same |r| >= CORR_THRESHOLD threshold.  For
    each cluster, pool the imp_sums of ALL members:

        cluster_imp_sum = sum(imp_sum[m] for m in cluster)
        cluster_imp_cnt = sum(imp_cnt[m] for m in cluster)

    Assign the pooled signal to the representative (the member with the
    highest personal imp_sum — i.e. the one the per-fold filter chose more
    often and/or with higher SHAP):

        pooled_global_imp[rep] = cluster_imp_sum / n_winner_folds

    This correctly credits the representative with the full frequency of
    the biological signal, not just the subset of folds where it personally
    survived the per-fold filter.  Non-clustered features are unaffected.

    The returned mean_imp dict uses per-feature denominators (for reporting),
    while pooled_global_imp is used only for ranking and keeper selection.

    K = median per-fold signature size (rounded up).
    """
    winner_folds = [f for f in folds
                    if f.get("winner_clf", "") not in ("", "none")
                    and f.get("winner_signature")]
    if not winner_folds:
        return [], 0, {}

    n_winner_folds = len(winner_folds)

    # ── Step 1: accumulate per-feature importance sums and counts ─────────
    imp_sum   = defaultdict(float)
    imp_cnt   = defaultdict(int)
    feat_freq = Counter()

    for fold in winner_folds:
        w   = fold["winner_clf"]
        ii  = fold.get("inner_importance", {}).get(w, {})
        sig = set(fold.get("winner_signature", []))
        for feat in sig:
            val = float(ii.get(feat, 0.0))
            imp_sum[feat]   += abs(val)
            imp_cnt[feat]   += 1
            feat_freq[feat] += 1

    # mean_imp: per-feature mean (selected-folds denominator) — for reporting
    mean_imp = {f: imp_sum[f] / imp_cnt[f] for f in imp_sum}

    # ── Step 2: build correlation clusters and pool importance ────────────
    # cluster_of[f] → index of cluster containing f
    # clusters      → list of sets of feature names
    # pooled_global_imp[f] → imp_sum of f's entire cluster / n_winner_folds
    # representative[cluster_idx] → the member with the highest personal
    #                                imp_sum (most consistently selected
    #                                and/or most strongly scored)
    all_feats = list(imp_sum.keys())

    # Default: each feature is its own cluster (no pooling)
    pooled_global_imp = {f: imp_sum[f] / n_winner_folds for f in all_feats}
    cluster_freq      = dict(feat_freq)   # for reporting cluster-level freq
    to_remove         = set()

    if df_cc is not None and feat_cols is not None and len(all_feats) > 1:
        present  = [f for f in all_feats if f in df_cc.columns]
        cont     = [f for f in present if df_cc[f].dropna().nunique() > 2]

        if len(cont) > 1:
            threshold = CORR_THRESHOLD if CORR_THRESHOLD else 0.90
            corr_mat  = df_cc[cont].corr().abs()

            # ── Connected components (transitive closure) ─────────────────
            # A greedy "star" approach (for each feature, find all features
            # correlated with IT) misses transitive chains:
            #   A-B >= threshold, B-C >= threshold, A-C < threshold
            # → star clustering creates {A,B} and {C} separately, losing the
            #   B-C competition that exists in the per-fold filter.
            # Connected components via BFS correctly groups {A, B, C} because
            # B mediates the relationship between A and C.
            adj = defaultdict(set)
            for i, fa in enumerate(cont):
                for fb in cont[i+1:]:
                    if corr_mat.loc[fa, fb] >= threshold:
                        adj[fa].add(fb)
                        adj[fb].add(fa)

            visited  = set()
            clusters = []
            for start in sorted(cont, key=lambda f: -imp_sum.get(f, 0)):
                if start in visited:
                    continue
                component = set()
                queue = [start]
                while queue:
                    node = queue.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    component.add(node)
                    queue.extend(adj[node] - visited)
                clusters.append(component)

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue   # no pooling needed for singletons

                # Representative = highest personal imp_sum member
                rep    = max(cluster, key=lambda f: imp_sum.get(f, 0))
                others = cluster - {rep}

                # Pool: sum all members' imp_sums and counts
                pool_imp_sum = sum(imp_sum.get(m, 0.0) for m in cluster)
                pool_imp_cnt = sum(imp_cnt.get(m, 0)   for m in cluster)
                pool_freq    = sum(feat_freq.get(m, 0)  for m in cluster)

                # Assign pooled signal to representative
                pooled_global_imp[rep] = pool_imp_sum / n_winner_folds
                cluster_freq[rep]      = pool_freq   # total folds any member seen
                # mean_imp for rep: keep personal denominator for interpretability
                # but note in log that it reflects personal folds only

                # Mark others for removal from ranking pool
                to_remove |= others

                print(
                    f"  [CONSENSUS-POOL] cluster {sorted(cluster)} → "
                    f"rep={rep} "
                    f"(personal freq={feat_freq.get(rep,0)/n_winner_folds:.2f} "
                    f"pool freq={pool_freq/n_winner_folds:.2f} "
                    f"personal global_imp={imp_sum.get(rep,0)/n_winner_folds:.3f} "
                    f"pooled_global_imp={pool_imp_sum/n_winner_folds:.3f})"
                )
                for other in sorted(others):
                    print(
                        f"    dropped: {other} "
                        f"(personal freq={feat_freq.get(other,0)/n_winner_folds:.2f} "
                        f"personal global_imp={imp_sum.get(other,0)/n_winner_folds:.3f})"
                    )

    # ── Step 3: rank by pooled_global_imp, remove non-representatives ─────
    ranked_all = sorted(
        [f for f in all_feats if f not in to_remove],
        key=lambda f: (-pooled_global_imp[f], -cluster_freq.get(f, 0), f)
    )

    # ── Step 4: determine K ───────────────────────────────────────────────
    sig_sizes = [len(f["winner_signature"]) for f in winner_folds]
    if size_strategy == "median":
        K = int(np.ceil(float(np.median(sig_sizes))))
    elif size_strategy == "mean":
        K = int(np.ceil(float(np.mean(sig_sizes))))
    elif size_strategy == "mode":
        K = Counter(sig_sizes).most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown size_strategy: {size_strategy}")

    consensus_sig = ranked_all[:K]
    return consensus_sig, K, mean_imp


def _aggregate_classifier(folds):
    """
    Return (modal_clf, modal_params, support_fraction).

    Modal winner_clf across folds. Ties broken by mean inner-CV Stage B
    AUROC (or Stage A when Stage B unavailable) among the tied classifiers.
    Modal hyperparameters = the most common parameter dict among folds
    whose winner_clf == the modal classifier.
    """
    winner_folds = [f for f in folds
                    if f.get("winner_clf", "") not in ("", "none")]
    if not winner_folds:
        return "none", {}, 0.0

    clf_counts = Counter(f["winner_clf"] for f in winner_folds)
    top_count  = max(clf_counts.values())
    tied       = [c for c, n in clf_counts.items() if n == top_count]

    if len(tied) > 1:
        # Tie-break by mean inner AUROC among tied classifiers
        def _mean_auroc(c):
            vals = [f.get("inner_cv_auroc_B", 0.0) or
                    f.get("inner_cv_aurocs_A", {}).get(c, 0.0)
                    for f in winner_folds if f["winner_clf"] == c]
            return float(np.mean(vals)) if vals else 0.0
        modal_clf = max(tied, key=_mean_auroc)
    else:
        modal_clf = tied[0]

    # Modal parameter dict among folds whose winner is modal_clf
    modal_folds = [f for f in winner_folds if f["winner_clf"] == modal_clf]
    param_strs  = [str(sorted((f.get("inner_cv_params") or {}).items()))
                   for f in modal_folds]
    if param_strs:
        top_param_str = Counter(param_strs).most_common(1)[0][0]
        modal_params  = next(
            (f["inner_cv_params"] for f, s in zip(modal_folds, param_strs)
             if s == top_param_str),
            {}
        )
    else:
        modal_params = {}

    support_fraction = top_count / len(winner_folds)
    return modal_clf, modal_params, support_fraction


def finalize_consensus(results, ALL_MODS=("Clin", "RNA", "DNA", "Prot", "WSI"),
                        df_cc=None, features=None):
    """
    Aggregate per-fold discovery results into a fixed consensus.

    df_cc     : the complete-case DataFrame (all patients, all columns) —
                used by _aggregate_signature to deduplicate correlated
                features in the consensus pool.  If None, the
                post-consensus correlation filter is skipped.
    features  : dict from define_modality_features() — used to resolve
                modality-specific column lists for the correlation check.

    Returns a dict:
      {mod: {signature: [feat], K: int, winner_clf: str, params: dict,
             support_fraction: float, mean_importance: {feat: float}},
       ...}

    This is the SCIENTIFIC DELIVERABLE — one signature per modality, one
    classifier choice per modality, one set of hyperparameters per modality.
    These are reported in the paper's Results section and in Supplementary
    Table S? as the final PREDIX HER2 multimodal signature.
    """
    # Modality → column list mapping (used for correlation check)
    mod_feat_cols = {}
    if features is not None:
        mod_feat_cols = {
            "Clin": features.get("Clin_global", []),
            "RNA":  features.get("RNA",  []),
            "DNA":  features.get("DNA",  []),
            "Prot": features.get("Prot", []),
            "WSI":  features.get("WSI",  []),
        }

    consensus = {}
    for mod in ALL_MODS:
        folds = results.get(mod, [])
        # Pass df_cc and modality columns only for high-dimensional modalities
        # where correlated clusters are expected (RNA, DNA).
        # Clin/WSI/Prot are small enough that the per-fold filter already
        # handles them; the global check is safe for all but only meaningful
        # for RNA/DNA.
        df_for_corr   = df_cc if mod in CORR_FILTER_MODS else None
        cols_for_corr = mod_feat_cols.get(mod) if mod in CORR_FILTER_MODS else None

        sig, K, imp   = _aggregate_signature(
            folds, df_cc=df_for_corr, feat_cols=cols_for_corr)
        clf, prm, sup = _aggregate_classifier(folds)
        consensus[mod] = {
            "signature":         sig,
            "K":                 K,
            "winner_clf":        clf,
            "params":            prm,
            "support_fraction":  sup,
            "mean_importance":   imp,
            "n_folds":           len(folds),
        }
        print(f"  [CONSENSUS] {mod:<5} K={K:2d}  clf={clf:<16} "
              f"(support={sup*100:4.0f}%) "
              f"sig=[{', '.join(sig[:4])}{', ...' if len(sig) > 4 else ''}]")
    return consensus


def _refit_consensus_unimodal_fold(
    mod, consensus_mod, X_tr_raw_df, y_tr, X_te_raw_df, inner_splits,
    ac, inner_jobs, y_cc_train, df_cc_train=None):
    """
    Refit a single modality's consensus classifier within ONE outer fold.

    Two training regimes:

    cc-only mode: X_tr_raw_df IS the CC training fold. df_cc_train is None
        (or equal to X_tr_raw_df). Inner CV runs directly on X_tr_raw_df,
        and inner_splits index positions into X_tr_raw_df. The resulting
        oof array has len == len(y_tr) == len(CC training fold).

    expanded mode: X_tr_raw_df is the EXPANDED pool for this modality
        (all patients with this modality available, minus CC test
        patients). df_cc_train is the CC training fold. Inner CV runs by
        validating on CC training positions (inner_splits index into
        df_cc_train) and training on the expanded pool minus the
        inner-val CC patients. The resulting oof array has len ==
        len(df_cc_train) == len(CC training fold), so it aligns with
        other modalities' oof for the fusion layer.
    """
    consensus_sig = set(consensus_mod["signature"])
    clf_name      = consensus_mod["winner_clf"]
    params        = consensus_mod["params"] or {}

    # Strip non-feature columns (patient_id, pCR) before preprocessing.
    # Keep references to the originals so we can align CC indices by pid.
    def _feats_only(df_):
        return df_[[c for c in df_.columns
                    if c not in ("patient_id", "pCR")]]

    X_tr_feats   = _feats_only(X_tr_raw_df)
    X_te_feats   = _feats_only(X_te_raw_df)
    cc_df_full   = df_cc_train if df_cc_train is not None else X_tr_raw_df
    cc_df_feats  = _feats_only(cc_df_full)

    if clf_name == "none" or not consensus_sig:
        # Degenerate — return neutral
        n_cc_tr = len(cc_df_full)
        n_te    = len(X_te_raw_df)
        return (np.full(n_cc_tr, 0.5), np.full(n_cc_tr, 0.5),
                np.full(n_te, 0.5), None, [])

    # (1) Preprocess using the FULL training source for the outer refit.
    # In expanded mode this is the expanded pool; in cc-only it's CC train.
    X_tr_p, X_te_p, fcols, _prep = preprocess_fold_3_with_prep(
        X_tr_feats, X_te_feats, ac, y_train=y_tr)

    # (2) Intersect consensus signature with surviving features.
    surviving = [f for f in fcols if f in consensus_sig]
    dropped   = consensus_sig - set(surviving)
    if dropped:
        print(f"  [CONSENSUS-EVAL] {mod}: {len(dropped)} consensus features "
              f"dropped by fold preprocessing: {sorted(dropped)}")
    if not surviving:
        n_cc_tr = len(cc_df_full)
        n_te    = len(X_te_raw_df)
        return (np.full(n_cc_tr, 0.5), np.full(n_cc_tr, 0.5),
                np.full(n_te, 0.5), None, [])

    sig_idx = [fcols.index(f) for f in surviving]
    X_tr_sig = X_tr_p[:, sig_idx]
    X_te_sig = X_te_p[:, sig_idx]

    # (3) Fit outer model on the full training source
    cfg   = CLASSIFIERS[clf_name]
    model = cfg["build"]()
    try:
        model.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
    except ValueError:
        pass
    model.fit(X_tr_sig, y_tr)
    y_pred_raw_test = model.predict_proba(X_te_sig)[:, 1]

    # (4) Inner OOF — indexed to CC training positions so the fusion layer
    # can stack all 5 modality OOFs into one matrix. In expanded mode,
    # inner training pulls from the expanded pool (minus inner-val CC
    # patients); in cc-only mode, inner training is the CC inner training.
    n_cc_tr = len(cc_df_full)
    oof_raw = np.full(n_cc_tr, 0.5)

    # Detect expanded mode by: training pool is distinct from CC training
    expanded_mode = (df_cc_train is not None
                     and len(X_tr_raw_df) != len(cc_df_full))

    if expanded_mode:
        # X_tr_raw_df is the expanded pool. Align by patient_id: drop CC
        # inner-val patients from the expanded training pool for each
        # inner fold. Both frames carry patient_id (we passed them in
        # with that column).
        exp_pids = (X_tr_raw_df["patient_id"].values
                    if "patient_id" in X_tr_raw_df.columns else None)
        cc_pids  = (cc_df_full["patient_id"].values
                    if "patient_id" in cc_df_full.columns else None)

        for i_tr, i_va in inner_splits:
            if cc_pids is not None and exp_pids is not None:
                val_pid_set = set(cc_pids[i_va])
                mask_tr   = ~np.isin(exp_pids, list(val_pid_set))
                Xi_tr_raw = X_tr_feats[mask_tr]
                y_i_tr    = y_tr[mask_tr.nonzero()[0]]
                Xi_va_raw = cc_df_feats.iloc[i_va]
            else:
                # Safety fallback if pids missing: use CC training for inner CV
                Xi_tr_raw = cc_df_feats.iloc[i_tr]
                Xi_va_raw = cc_df_feats.iloc[i_va]
                y_i_tr    = y_cc_train[i_tr]

            if len(np.unique(y_i_tr)) < 2:
                continue
            try:
                Xi_tr_p, Xi_va_p, fcols_i = preprocess_fold_3(
                    Xi_tr_raw, Xi_va_raw, ac, y_train=y_i_tr)
                surv_i = [f for f in fcols_i if f in consensus_sig]
                if not surv_i:
                    continue
                sig_idx_i = [fcols_i.index(f) for f in surv_i]
                m_i = cfg["build"]()
                try:
                    m_i.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
                except ValueError:
                    pass
                m_i.fit(Xi_tr_p[:, sig_idx_i], y_i_tr)
                oof_raw[i_va] = m_i.predict_proba(Xi_va_p[:, sig_idx_i])[:, 1]
            except Exception as e:
                print(f"  [CONSENSUS-EVAL] {mod} inner fold failed: {e}")
    else:
        # cc-only mode — inner CV directly on CC training fold
        for i_tr, i_va in inner_splits:
            Xi_tr_raw = X_tr_feats.iloc[i_tr]
            Xi_va_raw = X_tr_feats.iloc[i_va]
            y_i_tr    = y_tr[i_tr]
            if len(np.unique(y_i_tr)) < 2:
                continue
            try:
                Xi_tr_p, Xi_va_p, fcols_i = preprocess_fold_3(
                    Xi_tr_raw, Xi_va_raw, ac, y_train=y_i_tr)
                surv_i = [f for f in fcols_i if f in consensus_sig]
                if not surv_i:
                    continue
                sig_idx_i = [fcols_i.index(f) for f in surv_i]
                m_i = cfg["build"]()
                try:
                    m_i.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
                except ValueError:
                    pass
                m_i.fit(Xi_tr_p[:, sig_idx_i], y_i_tr)
                oof_raw[i_va] = m_i.predict_proba(Xi_va_p[:, sig_idx_i])[:, 1]
            except Exception as e:
                print(f"  [CONSENSUS-EVAL] {mod} inner fold failed: {e}")

    # (5) Global Platt on inner OOF (CC-indexed), applied to outer-test preds
    platt_cal = _fit_global_platt(y_cc_train, oof_raw)
    if platt_cal is not None:
        y_pred_test = _apply_global_platt(platt_cal, y_pred_raw_test)
        oof_cal     = _apply_global_platt(platt_cal, oof_raw)
    else:
        y_pred_test = y_pred_raw_test
        oof_cal     = oof_raw

    return oof_raw, oof_cal, y_pred_test, platt_cal, surviving


def _evaluate_consensus_single_fold(
    fi, tr_idx, te_idx, inner_splits,
    df_cc_exp, y_cc, features, clin_key,
    consensus, ac_map, ALL_MODS, inner_jobs, nzv_freq,
    mod_datasets=None):
    """One outer fold of the consensus re-evaluation."""
    # Set the module-scope NZV threshold for THIS worker process before
    # any preprocessing runs, mirroring what _process_single_fold does in
    # the discovery path. Required because remove_near_zero_variance reads
    # NZV_FREQ_THRESHOLD from the module scope rather than as an argument.
    # We use globals() rather than sys.modules[__name__] so this works
    # correctly regardless of how the module was imported (as script,
    # importlib, or loky worker process).
    globals()["NZV_FREQ_THRESHOLD"] = nzv_freq

    y_te          = y_cc[te_idx]
    y_cc_train    = y_cc[tr_idx]

    oof_by_mod   = {}
    test_by_mod  = {}
    metrics_by_mod = {}
    surviving_by_mod = {}

    # Complete-case test patients (identified by patient_id). Used to
    # exclude them from the expanded training pool when mod_datasets is
    # provided — matches the discovery phase's expanded-training protocol.
    test_pids = set(df_cc_exp.iloc[te_idx]["patient_id"].values)

    for mod in ALL_MODS:
        cons_mod = consensus.get(mod, {})
        if not cons_mod.get("signature"):
            n_tr, n_te = len(tr_idx), len(te_idx)
            oof_by_mod[mod]  = np.full(n_tr, 0.5)
            test_by_mod[mod] = np.full(n_te, 0.5)
            metrics_by_mod[mod] = compute_fold_metrics(y_te, np.full(n_te, 0.5))
            surviving_by_mod[mod] = []
            continue

        fc_key    = clin_key if mod == "Clin" else mod
        feat_cols = [c for c in features.get(fc_key, [])
                     if c in df_cc_exp.columns
                     and c not in ("patient_id", "pCR")]
        # Keep patient_id alongside feature columns so the consensus refit
        # function can align CC indices during inner CV in expanded mode.
        feat_cols_with_pid = feat_cols + ["patient_id"]
        X_te_raw  = df_cc_exp.iloc[te_idx][feat_cols]

        # Choose training source: expanded pool (all modality patients
        # minus this fold's CC test patients) OR cc-only (this fold's CC
        # training patients). Must mirror the discovery training strategy.
        if mod_datasets is not None:
            mod_key = clin_key if mod == "Clin" else mod
            df_mod  = mod_datasets.get(mod_key)
            if df_mod is None:
                # Modality dataset missing → fall back to cc training
                X_tr_raw = df_cc_exp.iloc[tr_idx][feat_cols_with_pid]
                y_tr     = y_cc[tr_idx]
            else:
                feat_cols_m = [c for c in feat_cols if c in df_mod.columns]
                df_mod_tr   = df_mod[~df_mod["patient_id"].isin(test_pids)]
                X_tr_raw    = df_mod_tr[feat_cols_m + ["patient_id"]]
                y_tr        = df_mod_tr["pCR"].values
        else:
            X_tr_raw = df_cc_exp.iloc[tr_idx][feat_cols_with_pid]
            y_tr     = y_cc[tr_idx]

        _oof_raw, oof_cal, y_pred_test, _platt, surviving = \
            _refit_consensus_unimodal_fold(
                mod, cons_mod, X_tr_raw, y_tr, X_te_raw, inner_splits,
                ac_map[mod], inner_jobs, y_cc_train,
                df_cc_train=df_cc_exp.iloc[tr_idx][feat_cols_with_pid])

        oof_by_mod[mod]     = oof_cal
        test_by_mod[mod]    = y_pred_test
        metrics_by_mod[mod] = compute_fold_metrics(y_te, y_pred_test)
        surviving_by_mod[mod] = surviving

    # Fusion: refit elastic-net LR on stacked OOF → predict on stacked test
    X_fus_tr = np.column_stack([oof_by_mod[m] for m in ALL_MODS])
    X_fus_te = np.column_stack([test_by_mod[m] for m in ALL_MODS])
    fus_fit  = fit_fusion({m: oof_by_mod[m] for m in ALL_MODS},
                           y_cc_train, inner_splits, list(ALL_MODS),
                           inner_jobs=inner_jobs)
    fusion_dict = fus_fit.get("Fused_ElasticNet", {})
    if fusion_dict and fusion_dict.get("model") is not None:
        y_pred_fus = fusion_dict["model"].predict_proba(X_fus_te)[:, 1]
        fused_metrics = compute_fold_metrics(y_te, y_pred_fus)
        modality_weights = fusion_dict.get("modality_weights", {})
        tuned_C          = fusion_dict.get("tuned_C", None)
    else:
        y_pred_fus = np.full(len(y_te), 0.5)
        fused_metrics = compute_fold_metrics(y_te, y_pred_fus)
        modality_weights = {m: 0.0 for m in ALL_MODS}
        tuned_C = None

    return {
        "fold_idx":          fi,
        "y_test":            y_te,
        "test_idx":          te_idx,
        "unimodal_y_pred":   test_by_mod,
        "unimodal_metrics":  metrics_by_mod,
        "unimodal_surviving": surviving_by_mod,
        "fused_y_pred":      y_pred_fus,
        "fused_metrics":     fused_metrics,
        "modality_weights":  modality_weights,
        "tuned_C":           tuned_C,
    }


def evaluate_consensus(df_cc_exp, features, clin_key, splits, exp_name,
                       output_dir, consensus, active_clfs_unused=None,
                       ALL_MODS=("Clin", "RNA", "DNA", "Prot", "WSI"),
                       mod_datasets=None):
    """
    Run the frozen-consensus OOF re-evaluation on the same CV splits used
    for discovery. Saves {exp_name}_consensus_eval.pkl with per-fold results
    and pooled metrics.

    mod_datasets (optional)
        Per-modality expanded training datasets (same dict shape used by the
        discovery phase). If provided, per-modality classifier REFITS happen
        on the expanded pool (all patients with that modality available,
        minus this fold's CC test patients) to mirror the discovery training
        strategy. OOF and test predictions are still evaluated on CC
        patients only (the only cohort with all 5 modalities). If None, the
        cc-only discovery strategy is mirrored: classifier refits happen on
        the fold's CC training patients.

    Returns a dict with:
      folds:           per-fold results (list)
      pooled:          {mod: {AUROC, AUPRC, Brier, Sens, Spec, Threshold,
                              N_pooled}, "Fused": {...}}
      consensus:       the consensus dict used for evaluation (for audit)
    """
    from joblib import Parallel, delayed, parallel_backend

    y_cc   = df_cc_exp["pCR"].values
    n_folds = len(splits)
    ac_map  = {m: (m in CORR_FILTER_MODS) for m in ALL_MODS}
    use_expanded = mod_datasets is not None
    print(f"\n[{exp_name.upper()} | CONSENSUS-EVAL] "
          f"Frozen-signature OOF over {n_folds} outer folds"
          + (" | expanded training" if use_expanded else " | cc-only training"))

    n_outer_workers, inner_jobs = _resolve_parallel_budget(n_folds, N_JOBS)
    print(f"  CPU budget      : {n_outer_workers} outer × {inner_jobs} inner")

    # Per-experiment NZV threshold (global uses 0.95; dhp/tdm1 use 0.98 by default)
    nzv_freq = NZV_FREQ_GLOBAL if exp_name == "global" else NZV_FREQ_ARM
    print(f"  NZV freq thresh : {nzv_freq}")

    with parallel_backend("loky", n_jobs=n_outer_workers,
                          inner_max_num_threads=1):
        fold_results = Parallel(
            n_jobs=n_outer_workers, verbose=5,
            batch_size=1, pre_dispatch="2*n_jobs",
        )(
            delayed(_evaluate_consensus_single_fold)(
                fi, tr_idx, te_idx, inner_splits,
                df_cc_exp, y_cc, features, clin_key,
                consensus, ac_map, tuple(ALL_MODS), inner_jobs, nzv_freq,
                mod_datasets,
            )
            for fi, (tr_idx, te_idx, inner_splits) in enumerate(splits)
        )

    fold_results.sort(key=lambda f: f["fold_idx"])

    # Pooled metrics: concatenate (y_test, y_pred) across all folds and
    # compute AUROC / AUPRC / Brier / Youden Sens/Spec on the pool.
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  brier_score_loss, roc_curve)
    pooled = {}

    def _pool_metrics(y_t_all, y_p_all):
        if len(np.unique(y_t_all)) < 2:
            return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                    "Sensitivity": np.nan, "Specificity": np.nan,
                    "Threshold": np.nan, "N_pooled": len(y_t_all)}
        fpr, tpr, thr = roc_curve(y_t_all, y_p_all)
        yi = int(np.argmax(tpr - fpr))
        return {
            "AUROC":       float(roc_auc_score(y_t_all, y_p_all)),
            "AUPRC":       float(average_precision_score(y_t_all, y_p_all)),
            "Brier":       float(brier_score_loss(y_t_all, y_p_all)),
            "Sensitivity": float(tpr[yi]),
            "Specificity": float(1.0 - fpr[yi]),
            "Threshold":   float(thr[yi]),
            "N_pooled":    int(len(y_t_all)),
        }

    for mod in ALL_MODS:
        y_t_all = np.concatenate([f["y_test"] for f in fold_results])
        y_p_all = np.concatenate([f["unimodal_y_pred"][mod] for f in fold_results])
        pooled[mod] = _pool_metrics(y_t_all, y_p_all)
        # Fold-averaged as a secondary measurement
        fold_aurocs = [f["unimodal_metrics"][mod]["AUROC"]
                        for f in fold_results]
        pooled[mod]["mean_fold_AUROC"] = float(np.nanmean(fold_aurocs))
        pooled[mod]["std_fold_AUROC"]  = float(np.nanstd(fold_aurocs))

    # Fusion pooled
    y_t_all = np.concatenate([f["y_test"] for f in fold_results])
    y_p_all = np.concatenate([f["fused_y_pred"] for f in fold_results])
    pooled["Fused_ElasticNet"] = _pool_metrics(y_t_all, y_p_all)
    fold_aurocs_fus = [f["fused_metrics"]["AUROC"] for f in fold_results]
    pooled["Fused_ElasticNet"]["mean_fold_AUROC"] = float(np.nanmean(fold_aurocs_fus))
    pooled["Fused_ElasticNet"]["std_fold_AUROC"]  = float(np.nanstd(fold_aurocs_fus))

    out_pkl = Path(output_dir) / f"{exp_name}_consensus_eval.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "folds":     fold_results,
            "pooled":    pooled,
            "consensus": consensus,
            "exp_name":  exp_name,
        }, f)
    print(f"  [SAVE] {out_pkl.name}")

    # Print summary table
    print(f"\n  CONSENSUS OOF PERFORMANCE — {exp_name}")
    print(f"  {'Model':<20} {'Pooled AUROC':>14} {'Mean fold AUROC':>18} "
          f"{'Pooled AUPRC':>14} {'Pooled Sens':>13} {'Pooled Spec':>13}")
    for mod in list(ALL_MODS) + ["Fused_ElasticNet"]:
        p = pooled[mod]
        print(f"  {mod:<20} "
              f"{p['AUROC']:>14.4f} "
              f"{p['mean_fold_AUROC']:>14.4f} ± {p['std_fold_AUROC']:.3f} "
              f"{p['AUPRC']:>14.4f} "
              f"{p['Sensitivity']:>13.4f} "
              f"{p['Specificity']:>13.4f}")

    return {
        "folds":     fold_results,
        "pooled":    pooled,
        "consensus": consensus,
    }


def write_consensus_summary(consensus, eval_result, exp_name, output_dir):
    """Write a human-readable consensus_summary.txt for the paper."""
    path = Path(output_dir) / f"{exp_name}_consensus_summary.txt"
    lines = []
    lines.append("=" * 70)
    lines.append(f"PREDIX HER2 — CONSENSUS MODEL SUMMARY ({exp_name})")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Per-modality consensus signatures (R2 protocol)")
    lines.append("-" * 70)
    for mod, c in consensus.items():
        lines.append(f"\n  {mod}")
        lines.append(f"    Winner classifier:     {c['winner_clf']}")
        lines.append(f"    Classifier support:    {c['support_fraction']*100:.0f}% of folds")
        lines.append(f"    Hyperparameters:       {c['params']}")
        lines.append(f"    Signature size (K):    {c['K']}")
        lines.append(f"    Signature features (ranked by mean |importance|):")
        for rank, feat in enumerate(c["signature"], 1):
            imp = c["mean_importance"].get(feat, 0.0)
            lines.append(f"      {rank:2d}. {feat:<40} |imp| = {imp:.4f}")

    lines.append("")
    lines.append("Frozen-consensus OOF performance")
    lines.append("-" * 70)
    pooled = eval_result["pooled"]
    lines.append(f"\n  {'Model':<22} {'Pooled AUROC':>14} "
                 f"{'Mean fold AUROC':>20} {'Pooled AUPRC':>14}")
    for mod in list(consensus.keys()) + ["Fused_ElasticNet"]:
        p = pooled[mod]
        lines.append(f"  {mod:<22} {p['AUROC']:>14.4f} "
                     f"{p['mean_fold_AUROC']:>14.4f} ± {p['std_fold_AUROC']:.3f} "
                     f"{p['AUPRC']:>14.4f}")

    lines.append("")
    lines.append("Notes")
    lines.append("-" * 70)
    lines.append("  - Signatures selected by top-K mean |SHAP importance| across discovery folds")
    lines.append("    where K = median per-fold signature size from the winner-classifier folds.")
    lines.append("  - Consensus choices (signature + classifier identity + hyperparameters) are FROZEN.")
    lines.append("  - Classifier weights and fusion coefficients ARE REFIT within each CV fold,")
    lines.append("    using only that fold's training data — no test-fold leakage.")
    lines.append("  - Pooled AUROC is the PRIMARY HEADLINE for the paper.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  [SAVE] {path.name}")


# ==============================================================================
# SECTION 7 — CLI + MAIN
# ==============================================================================

def parse_args():
    """Single source of truth for all parameters."""
    p = argparse.ArgumentParser(
        description="PREDIX HER2 multimodal pCR prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--data_path",   type=Path, required=True,
        help="Input dataset (.txt, tab-separated).")
    p.add_argument("--results_dir", type=Path, default=Path("./results"),
        help="Output directory for PKL files.")
    p.add_argument("--splits_dir",  type=Path, default=None,
        help="Directory for CV split PKLs (read or write). Defaults to "
             "--results_dir if not set. IMPORTANT: when running both "
             "--training_data strategies for comparison, point both runs "
             "at the same --splits_dir so that outer test folds are "
             "identical across strategies. The first run generates and "
             "saves the splits; the second run loads them.")

    # Mode
    p.add_argument("--mode",
        choices=["elasticnet", "best_per_fold", "ensemble_weighted", "all"],
        default="elasticnet",
        help="elasticnet: primary analysis (elastic-net LR, tuned C). "
             "best_per_fold: best classifier per modality per fold. "
             "ensemble_weighted: AUROC-weighted ensemble. "
             "all: run all three modes.")

    # Training data strategy
    p.add_argument("--training_data",
        choices=["expanded", "cc_only"],
        default="expanded",
        help="expanded (default): each unimodal model trains on ALL patients "
             "with that modality available, minus the current outer test fold. "
             "Test sets remain complete-case for paired comparisons. "
             "This maximises training data and is the recommended strategy. "
             "cc_only: restrict training to the complete-case patients only "
             "(those with all five modalities). Produces a more conservative "
             "estimate; useful to isolate the effect of expanded training or "
             "when modality-specific patients are not available.")

    # Classifiers (used for best_per_fold and ensemble_weighted only)
    p.add_argument("--classifiers", nargs="+",
        default=["ElasticNet_LR", "RandomForest", "ExtraTrees",
                 "HistGradBoost", "SVM_RBF", "SVM_Linear"],
        help="Classifiers evaluated in Stage A (signature discovery). "
             "SVM_RBF is automatically excluded from signature ranking "
             "(no SHAP capability) but can be listed without error. "
             "Winner selected per modality per fold by inner CV AUROC.")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Parallelism
    p.add_argument("--n_jobs", type=int, default=-1,
        help="Number of parallel workers for the outer fold loop. "
             "-1 = use all available CPUs (default). "
             "1 = sequential (useful for debugging). "
             "Set to the number of CPUs allocated in your SLURM job "
             "(e.g. --cpus-per-task=32 → --n_jobs=32).")

    # Outer CV
    p.add_argument("--outer_folds_global", type=int, default=5)
    p.add_argument("--outer_folds_arm",    type=int, default=5)
    p.add_argument("--repeats_global",     type=int, default=20,
        help="Production: 200.")
    p.add_argument("--repeats_arm",        type=int, default=10,
        help="Production: 100.")

    # Inner CV
    p.add_argument("--inner_folds_global", type=int, default=5)
    p.add_argument("--inner_folds_arm",    type=int, default=3,
        help="3 ensures ≥25 inner training patients.")

    # Preprocessing
    p.add_argument("--corr_threshold", type=float, default=0.90)
    p.add_argument("--nzv_freq_global", type=float, default=0.95,
        help="NZV dominant-value-frequency threshold for GLOBAL experiment. "
             "A feature whose most common value occupies ≥ this fraction of "
             "training samples is removed.")
    p.add_argument("--nzv_freq_arm",    type=float, default=0.98,
        help="NZV threshold for ARM experiments (DHP, TDM1). Higher than "
             "--nzv_freq_global because arm cohorts are small (n≈50-60) and "
             "the 0.95 cutoff silently culls low-prevalence binary mutation "
             "features (e.g. features present in ~5% of patients) which are "
             "precisely the biologically meaningful DNA features the study "
             "is interested in. 0.98 keeps features present in ≥ ~2%% of "
             "arm training patients.")
    p.add_argument("--nzv_ratio",      type=float, default=20.0)

    # Stability thresholds (elasticnet mode reporting)
    p.add_argument("--stability_thresh_global", type=float, default=0.60)
    p.add_argument("--stability_thresh_arm",    type=float, default=0.50)

    # Experiments
    p.add_argument("--experiments", nargs="+",
        choices=["global", "dhp", "tdm1"],
        default=["global", "dhp", "tdm1"])

    # Consensus finalization — frozen-signature re-evaluation (R2)
    p.add_argument("--consensus", action="store_true", default=True,
        help="After discovery CV completes, aggregate per-fold signatures "
             "into a per-modality consensus, then re-evaluate that frozen "
             "consensus under the SAME CV splits (classifier + fusion re-fit "
             "within each fold, signature frozen). This is the R2 protocol "
             "from the Nature Cancer methods: signatures and classifier "
             "choices are the scientific deliverable; performance reported "
             "is the consensus OOF AUROC from this re-evaluation.")
    p.add_argument("--no-consensus", dest="consensus", action="store_false",
        help="Skip the consensus finalization phase (speeds up smoke tests).")

    return p.parse_args()


def main():
    global DATA_PATH, RESULTS_DIR, RANDOM_SEED, CLASSIFIERS
    global GLOBAL_N_OUTER_FOLDS, GLOBAL_N_REPEATS, GLOBAL_N_INNER_FOLDS
    global ARM_N_OUTER_FOLDS, ARM_N_REPEATS, ARM_N_INNER_FOLDS
    global CORR_THRESHOLD, NZV_RATIO_THRESHOLD
    global NZV_FREQ_GLOBAL, NZV_FREQ_ARM, NZV_FREQ_THRESHOLD
    global STABILITY_THRESHOLD_GLOBAL, STABILITY_THRESHOLD_ARM
    global N_JOBS

    args = parse_args()

    DATA_PATH                = args.data_path
    RESULTS_DIR              = args.results_dir
    RANDOM_SEED              = args.seed
    GLOBAL_N_OUTER_FOLDS     = args.outer_folds_global
    GLOBAL_N_REPEATS         = args.repeats_global
    GLOBAL_N_INNER_FOLDS     = args.inner_folds_global
    ARM_N_OUTER_FOLDS        = args.outer_folds_arm
    ARM_N_REPEATS            = args.repeats_arm
    ARM_N_INNER_FOLDS        = args.inner_folds_arm
    CORR_THRESHOLD           = args.corr_threshold
    NZV_FREQ_GLOBAL          = args.nzv_freq_global
    NZV_FREQ_ARM             = args.nzv_freq_arm
    NZV_FREQ_THRESHOLD       = args.nzv_freq_global  # default; overridden per-experiment
    NZV_RATIO_THRESHOLD      = args.nzv_ratio
    STABILITY_THRESHOLD_GLOBAL = args.stability_thresh_global
    STABILITY_THRESHOLD_ARM    = args.stability_thresh_arm
    N_JOBS                     = args.n_jobs

    CLASSIFIERS  = build_classifiers(RANDOM_SEED)
    active_clfs  = [c for c in args.classifiers if c in CLASSIFIERS]
    splits_dir   = args.splits_dir or RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Warn when splits_dir is not explicitly set in cc_only mode.
    # In that case splits default to RESULTS_DIR, which differs between
    # the two strategy runs — test fold identity is not guaranteed.
    splits_dir_explicit = args.splits_dir is not None
    if args.training_data == "cc_only" and not splits_dir_explicit:
        print(
            "\n[WARNING] --splits_dir not set.\n"
            "  Outer test folds will be saved to --results_dir, which is "
            "separate from your expanded-training run.\n"
            "  If you intend to compare expanded vs cc_only strategies, "
            "re-run BOTH with the same --splits_dir so test folds are "
            "guaranteed identical:\n"
            "    --splits_dir ./shared_splits\n"
        )

    # Determine which modes to run
    modes = (["elasticnet", "best_per_fold", "ensemble_weighted"]
             if args.mode == "all" else [args.mode])

    print("=" * 70)
    print("PREDIX HER2 — MULTIMODAL pCR PREDICTION PIPELINE")
    print(f"  Mode(s)        : {modes}")
    if len(modes) > 1 or modes[0] != "elasticnet":
        print(f"  Classifiers    : {active_clfs}")
    print(f"  Experiments    : {args.experiments}")
    print(f"  Training data  : {args.training_data}  "
          f"({'per-modality expanded sets' if args.training_data == 'expanded' else 'complete-case only'})")
    print(f"  Splits dir     : {splits_dir}"
          f"{'  ← shared, test folds guaranteed identical' if splits_dir_explicit else '  ← defaults to results_dir (not shared)'}")
    print(f"  Global CV      : {GLOBAL_N_OUTER_FOLDS}-fold × {GLOBAL_N_REPEATS} "
          f"= {GLOBAL_N_OUTER_FOLDS*GLOBAL_N_REPEATS} outer folds")
    print(f"  Arm CV         : {ARM_N_OUTER_FOLDS}-fold × {ARM_N_REPEATS} "
          f"= {ARM_N_OUTER_FOLDS*ARM_N_REPEATS} outer folds per arm")
    print(f"  Results dir    : {RESULTS_DIR}")
    print(f"  Parallel jobs  : {N_JOBS} ({'all CPUs' if N_JOBS == -1 else 'sequential' if N_JOBS == 1 else f'{N_JOBS} workers'})")
    print("=" * 70)

    df_enc   = load_and_encode_data(DATA_PATH)
    features = define_modality_features(df_enc)
    df_cc    = get_complete_case(df_enc, features)
    df_dhp   = df_cc[df_cc["Clin_Arm"] == 0].reset_index(drop=True)
    df_tdm1  = df_cc[df_cc["Clin_Arm"] == 1].reset_index(drop=True)
    print(f"\n[COHORT] complete-case n={len(df_cc)} "
          f"(DHP={len(df_dhp)}, T-DM1={len(df_tdm1)}), "
          f"pCR={df_cc['pCR'].mean():.3f}")

    # ── Per-modality datasets ──────────────────────────────────────────────────
    # Only computed when expanded training is requested.
    # In cc_only mode, mod_datasets is None → run_experiment uses the
    # complete-case training set only (use_expanded=False branch).
    if args.training_data == "expanded":
        mod_datasets_global = get_modality_datasets(df_enc, features)
        mod_datasets_dhp    = get_modality_datasets(
            df_enc[df_enc["Clin_Arm"] == 0].copy(), features)
        mod_datasets_tdm1   = get_modality_datasets(
            df_enc[df_enc["Clin_Arm"] == 1].copy(), features)

        print("\n[EXPANDED TRAINING] Available patients per modality:")
        print(f"  {'Modality':<6}  {'Global':>8}  {'DHP':>6}  {'T-DM1':>7}  "
              f"{'CC baseline':>12}")
        for mod_key, label in [("Clin_global","Clin"),("RNA","RNA"),
                                ("DNA","DNA"),("Prot","Prot"),("WSI","WSI")]:
            n_g = len(mod_datasets_global[mod_key])
            n_d = len(mod_datasets_dhp.get(mod_key, pd.DataFrame()))
            n_t = len(mod_datasets_tdm1.get(mod_key, pd.DataFrame()))
            print(f"  {label:<6}  {n_g:>8}  {n_d:>6}  {n_t:>7}  "
                  f"{'110 / 59 / 51':>12}")
    else:
        mod_datasets_global = None
        mod_datasets_dhp    = None
        mod_datasets_tdm1   = None
        print("\n[CC-ONLY TRAINING] Training restricted to complete-case "
              f"patients (n={len(df_cc)}) for all modalities.")

    exp_map = {
        "global": (df_cc,   "Clin_global", mod_datasets_global,
                   GLOBAL_N_OUTER_FOLDS, GLOBAL_N_REPEATS, GLOBAL_N_INNER_FOLDS),
        "dhp":    (df_dhp,  "Clin_arm",    mod_datasets_dhp,
                   ARM_N_OUTER_FOLDS,    ARM_N_REPEATS,    ARM_N_INNER_FOLDS),
        "tdm1":   (df_tdm1, "Clin_arm",    mod_datasets_tdm1,
                   ARM_N_OUTER_FOLDS,    ARM_N_REPEATS,    ARM_N_INNER_FOLDS),
    }

    # Cross-arm CC dataframes for counterfactual analysis. Only arm experiments
    # consume these — global uses cross_arm_df=None and produces no cross_arm_preds.
    # Pre-treatment features make cross-arm prediction meaningful: applying a
    # DHP-trained model to T-DM1 patients' pre-treatment features estimates
    # P(pCR | they had received DHP). Saved per-fold under Fused_ElasticNet
    # fold dict key "cross_arm_preds" = {patient_id: float}.
    cross_arm_map = {
        "global": (None, None),
        "dhp":    (df_tdm1.reset_index(drop=True), "T-DM1"),
        "tdm1":   (df_dhp.reset_index(drop=True),  "DHP"),
    }

    for exp_name in args.experiments:
        df_cc_exp, clin_key, mod_ds, n_outer, n_rep, n_inner = exp_map[exp_name]
        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(exist_ok=True)

        # CV splits defined on complete-case patients — shared across all modes
        splits = load_or_generate_splits(
            splits_dir, exp_name, df_cc_exp["pCR"].values,
            n_outer, n_rep, n_inner, RANDOM_SEED)

        cross_arm_df, cross_arm_label = cross_arm_map[exp_name]

        for mode in modes:
            # All active classifiers are passed for all modes.
            # _fit_signature_model (elasticnet + expanded) uses all of them
            # for signature discovery and selects the winner by inner AUROC.
            # _fit_best_per_fold and _fit_ensemble also use all active_clfs.
            run_experiment(
                df_cc_exp=df_cc_exp, features=features, clin_key=clin_key,
                splits=splits, exp_name=exp_name, output_dir=exp_dir,
                mode=mode,
                active_clfs=active_clfs,
                mod_datasets=mod_ds,
                cross_arm_df=cross_arm_df,
                cross_arm_label=cross_arm_label)

        # ── R2 CONSENSUS FINALIZATION ─────────────────────────────────────
        # After the discovery CV loop completes, (1) aggregate per-fold
        # signatures and classifier choices into a single consensus, and
        # (2) re-evaluate that FROZEN consensus under the SAME CV splits
        # with classifier + fusion re-fit within each fold. The pooled
        # OOF AUROC from (2) is the PRIMARY HEADLINE for the paper.
        if args.consensus and "elasticnet" in modes:
            disc_pkl = exp_dir / f"{exp_name}_elasticnet_results.pkl"
            if not disc_pkl.exists():
                print(f"\n[{exp_name.upper()} | CONSENSUS] "
                      f"Discovery PKL not found — skipping consensus phase.")
                continue
            print("\n" + "=" * 70)
            print(f"[{exp_name.upper()}] R2 CONSENSUS FINALIZATION")
            print("=" * 70)
            with open(disc_pkl, "rb") as f:
                disc_results = pickle.load(f)
            print(f"\n  [1/2] Aggregating per-fold discovery into consensus ...")
            consensus = finalize_consensus(
                disc_results, ALL_MODS=ALL_MODS,
                df_cc=df_cc_exp, features=features)

            print(f"\n  [2/2] Re-evaluating frozen consensus under same CV splits ...")
            eval_result = evaluate_consensus(
                df_cc_exp=df_cc_exp, features=features, clin_key=clin_key,
                splits=splits, exp_name=exp_name, output_dir=exp_dir,
                consensus=consensus, ALL_MODS=ALL_MODS,
                mod_datasets=mod_ds)  # None in cc_only; expanded dict otherwise

            write_consensus_summary(consensus, eval_result, exp_name, exp_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Results → {RESULTS_DIR}")
    print("  Report :")
    print(f"    python3 generate_report.py \\")
    print(f"        --results_dir {RESULTS_DIR} --out_dir ./report")
    print("=" * 70)


if __name__ == "__main__":
    main()
