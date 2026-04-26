#!/usr/bin/env bash
# =============================================================================
# run_predix_pipeline.sh
# =============================================================================
# Complete execution script for the PREDIX HER2 multimodal pCR prediction
# pipeline. Runs discovery cross-validation, consensus finalisation, and
# report generation for all three experimental scenarios (Global, DHP, T-DM1).
#
# USAGE
# -----
#   # Minimal test run (completes in ~5 min on a laptop):
#   bash run_predix_pipeline.sh --test
#
#   # Full production run (paper settings, requires HPC):
#   bash run_predix_pipeline.sh
#
#   # Custom data path:
#   bash run_predix_pipeline.sh --data /path/to/your_data.txt
#
# SLURM (HPC) USAGE
# -----------------
#   sbatch run_predix_pipeline.sh
#   (Edit the #SBATCH lines below to match your cluster allocation)
#
# REQUIREMENTS
# ------------
#   Python 3.10+  with:
#     numpy==2.2.0   pandas==2.2.3   scikit-learn==1.6.1
#     scipy==1.14.1  shap==0.46.0    openpyxl==3.1.5
#     matplotlib==3.9.4
#
#   Install in a virtual environment:
#     module load Python/3.11.5-GCCcore-13.2.0   # on Alvis HPC
#     python -m venv predix_env
#     source predix_env/bin/activate
#     pip install numpy==2.2.0 pandas==2.2.3 scikit-learn==1.6.1 \
#                 scipy==1.14.1 shap==0.46.0 openpyxl==3.1.5 \
#                 matplotlib==3.9.4
#
# OUTPUT STRUCTURE
# ----------------
#   results/
#   ├── shared_splits/          ← CV split PKLs (reusable across runs)
#   ├── global/
#   │   ├── global_elasticnet_results.pkl   ← discovery folds
#   │   └── global_consensus_eval.pkl       ← consensus re-evaluation
#   ├── dhp/
#   │   ├── dhp_elasticnet_results.pkl
#   │   └── dhp_consensus_eval.pkl
#   └── tdm1/
#       ├── tdm1_elasticnet_results.pkl
#       └── tdm1_consensus_eval.pkl
#   report/
#   ├── figures/                ← fig01 … fig14 (PDF)
#   └── tables/                 ← Excel workbook + CSV exports
#
# =============================================================================

# ── SLURM directives (only active when submitted via sbatch) ─────────────────
#SBATCH --job-name=predix_her2
#SBATCH --account=YOUR_PROJECT_ID          # ← replace with your allocation
#SBATCH --time=12:00:00                    # 12 h is sufficient for production
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32                 # matches --n_jobs below
#SBATCH --mem=64G
#SBATCH --output=logs/predix_%j.out
#SBATCH --error=logs/predix_%j.err

set -euo pipefail   # exit on error, unset variable, or pipe failure

# =============================================================================
# CONFIGURATION — edit these paths before running
# =============================================================================

# Path to the input dataset
DATA_PATH="data/clin_multiomics_curated_metrics_PREDIX_HER2_withNA.txt"

# Root output directory for PKL results
RESULTS_DIR="results"

# Directory where CV split PKLs are saved/loaded.
# IMPORTANT: all runs that should share the same outer test folds must
# point to the same SPLITS_DIR. The first run writes the splits; subsequent
# runs read them.
SPLITS_DIR="${RESULTS_DIR}/shared_splits"

# Output directory for figures and tables (generate_report.py)
REPORT_DIR="report"

# Log directory
LOG_DIR="logs"

# Number of parallel workers (set to your CPU allocation; -1 = all CPUs)
N_JOBS=32

# Random seed for reproducibility
SEED=42

# =============================================================================
# PRODUCTION SETTINGS (paper values)
# These control the number of cross-validation repetitions.
# For a quick test, pass --test flag (overridden below).
# =============================================================================
REPEATS_GLOBAL=40     # 5 folds × 40 repeats = 200 outer folds
REPEATS_ARM=20        # 5 folds × 20 repeats = 100 outer folds per arm

# =============================================================================
# PARSE COMMAND-LINE FLAGS
# =============================================================================
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test)
            TEST_MODE=true
            REPEATS_GLOBAL=2
            REPEATS_ARM=2
            N_JOBS=2
            echo "[INFO] TEST MODE: repeats reduced to 2 (fast smoke test)"
            shift ;;
        --data)
            DATA_PATH="$2"
            shift 2 ;;
        --results)
            RESULTS_DIR="$2"
            SPLITS_DIR="${RESULTS_DIR}/shared_splits"
            shift 2 ;;
        --report)
            REPORT_DIR="$2"
            shift 2 ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2 ;;
        --help|-h)
            echo "Usage: bash run_predix_pipeline.sh [--test] [--data PATH] [--results DIR] [--report DIR] [--n_jobs N]"
            exit 0 ;;
        *)
            echo "[WARN] Unknown argument: $1 (ignored)"
            shift ;;
    esac
done

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Load HPC module if running on a cluster with environment modules.
# Comment out or modify for your cluster's module system.
if command -v module &>/dev/null; then
    echo "[ENV] Loading Python module ..."
    module purge
    module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || true
fi

# Activate virtual environment if it exists
VENV_PATH="${HOME}/predix_env"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ENV] Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    echo "[WARN] Virtual environment not found at ${VENV_PATH}."
    echo "[WARN] Running with system Python. Ensure dependencies are installed."
fi

# Verify Python version and key packages
echo "[ENV] Python: $(python3 --version)"
echo "[ENV] scikit-learn: $(python3 -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "[ENV] shap:         $(python3 -c 'import shap;    print(shap.__version__)'    2>/dev/null || echo 'NOT FOUND')"

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

echo ""
echo "============================================================"
echo " PREDIX HER2 Multimodal pCR Pipeline"
echo "============================================================"
echo " Data:        ${DATA_PATH}"
echo " Results:     ${RESULTS_DIR}"
echo " Splits:      ${SPLITS_DIR}"
echo " Report:      ${REPORT_DIR}"
echo " n_jobs:      ${N_JOBS}"
echo " Seed:        ${SEED}"
echo " Repeats:     Global=${REPEATS_GLOBAL}  Arm=${REPEATS_ARM}"
echo " Total folds: Global=$(( 5 * REPEATS_GLOBAL ))  Arm=$(( 5 * REPEATS_ARM ))"
echo " Test mode:   ${TEST_MODE}"
echo " Node:        $(hostname)"
echo " Start:       $(date)"
echo "============================================================"
echo ""

# Check input data
if [[ ! -f "${DATA_PATH}" ]]; then
    echo "[ERROR] Data file not found: ${DATA_PATH}"
    echo "        Set DATA_PATH at the top of this script or pass --data /path/to/file"
    exit 1
fi

# Check pipeline scripts exist
for script in multimodal_pcr_pipeline.py generate_report.py; do
    if [[ ! -f "${script}" ]]; then
        echo "[ERROR] Script not found: ${script}"
        echo "        Run this bash script from the same directory as the Python scripts."
        exit 1
    fi
done

# Create output directories
mkdir -p "${RESULTS_DIR}" "${SPLITS_DIR}" "${REPORT_DIR}" "${LOG_DIR}"

# =============================================================================
# STEP 1 — DISCOVERY CROSS-VALIDATION (primary analysis)
# =============================================================================
# Runs the full repeated stratified nested CV for all three scenarios.
# Each scenario produces:
#   {RESULTS_DIR}/{scenario}/{scenario}_elasticnet_results.pkl
#
# Key pipeline decisions made here:
#   • Tier 1 biological deduplication (hardcoded in TIER1_REMOVE)
#   • Tier 2 NZV filtering (thresholds below)
#   • Tier 3 correlation filtering (|r| >= 0.90, RNA and DNA only)
#   • Stage A: 5 classifiers, EPV=5 cap, 25th-percentile filter
#   • Stage B: GridSearchCV hyperparameter tuning
#   • Late fusion: ElasticNet meta-learner on OOF probability matrix

echo "------------------------------------------------------------"
echo "[STEP 1] Running discovery cross-validation ..."
echo "         This is the main computational step."
echo "         Estimated time with ${N_JOBS} CPUs:"
echo "           Test mode: ~5 min"
echo "           Production: ~3-8 hours depending on hardware"
echo "------------------------------------------------------------"

python3 multimodal_pcr_pipeline.py \
    --data_path         "${DATA_PATH}" \
    --results_dir       "${RESULTS_DIR}" \
    --splits_dir        "${SPLITS_DIR}" \
    --mode              elasticnet \
    --training_data     expanded \
    --experiments       global dhp tdm1 \
    --classifiers       ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear \
    --repeats_global    "${REPEATS_GLOBAL}" \
    --repeats_arm       "${REPEATS_ARM}" \
    --outer_folds_global  5 \
    --outer_folds_arm     5 \
    --inner_folds_global  5 \
    --inner_folds_arm     3 \
    --corr_threshold    0.90 \
    --nzv_freq_global   0.95 \
    --nzv_freq_arm      0.98 \
    --nzv_ratio         20.0 \
    --stability_thresh_global 0.60 \
    --stability_thresh_arm    0.50 \
    --n_jobs            "${N_JOBS}" \
    --seed              "${SEED}" \
    --consensus \
    2>&1 | tee "${LOG_DIR}/step1_discovery.log"

echo "[STEP 1] Completed: $(date)"
echo ""

# Verify expected PKL outputs exist
echo "[CHECK] Verifying PKL outputs ..."
for exp in global dhp tdm1; do
    PKL="${RESULTS_DIR}/${exp}/${exp}_elasticnet_results.pkl"
    CONS="${RESULTS_DIR}/${exp}/${exp}_consensus_eval.pkl"
    if [[ -f "${PKL}" ]]; then
        SIZE=$(du -h "${PKL}" | cut -f1)
        echo "        ✓ ${PKL} (${SIZE})"
    else
        echo "        ✗ MISSING: ${PKL}"
        echo "[ERROR] Discovery PKL not found. Check step1_discovery.log for errors."
        exit 1
    fi
    if [[ -f "${CONS}" ]]; then
        SIZE=$(du -h "${CONS}" | cut -f1)
        echo "        ✓ ${CONS} (${SIZE})"
    else
        echo "        ⚠ ${CONS} — consensus not found (check --consensus flag was set)"
    fi
done
echo ""

# =============================================================================
# STEP 2 — REPORT GENERATION
# =============================================================================
# Reads the PKL files from Step 1 and generates all figures and Excel tables.
# This is fast (~2-5 min) and can be re-run independently without re-running Step 1.
#
# To regenerate figures without re-running the pipeline:
#   python3 generate_report.py --results_dir results/ --out_dir report/

echo "------------------------------------------------------------"
echo "[STEP 2] Generating figures and tables ..."
echo "         Output: ${REPORT_DIR}/"
echo "------------------------------------------------------------"

python3 generate_report.py \
    --results_dir "${RESULTS_DIR}" \
    --out_dir     "${REPORT_DIR}" \
    2>&1 | tee "${LOG_DIR}/step2_report.log"

echo "[STEP 2] Completed: $(date)"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================

echo "============================================================"
echo " Pipeline complete: $(date)"
echo "============================================================"
echo ""
echo " Results (PKL files):  ${RESULTS_DIR}/"
echo " Figures (PDF):        ${REPORT_DIR}/figures/"
echo " Tables (Excel/CSV):   ${REPORT_DIR}/tables/"
echo " Logs:                 ${LOG_DIR}/"
echo ""
echo " Key output files:"
for exp in global dhp tdm1; do
    echo "   PKL: ${RESULTS_DIR}/${exp}/${exp}_elasticnet_results.pkl"
done
echo ""
echo " Primary figures for paper:"
echo "   fig02_consensus_signatures.pdf  — frozen multimodal signatures"
echo "   fig03_consensus_roc.pdf         — pooled OOF ROC (primary AUROC)"
echo "   fig14_performance_CI.pdf        — AUROC with 95% CI per model"
echo ""
echo " To regenerate figures without re-running the pipeline:"
echo "   python3 generate_report.py --results_dir ${RESULTS_DIR} --out_dir ${REPORT_DIR}"
echo ""
echo "============================================================"
