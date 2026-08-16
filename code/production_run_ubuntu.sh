#!/usr/bin/env bash
# =============================================================================
# PREDIX HER2 - PRODUCTION RUN (Ubuntu, flat directory layout)
# =============================================================================
# Bash mirror of production_run.ps1 (which is running on the Windows laptop).
# Identical steps, settings and seed, so the two machines produce the same
# analysis; late-digit differences between them can only come from library
# versions (each run's run_provenance.json records its exact versions - quote
# numbers from ONE machine only, whichever you designate).
#
# Layout: run from a directory containing, FLAT (no data/ subdir):
#   multimodal_pcr_pipeline.py  generate_report.py  revision_analyses.py
#   external_validation.py      cv_estimands.py  (shared estimand module,
#   imported by the three post-processing scripts)
#   tests/test_statistics.py    requirements.txt
#   clin_multiomics_curated_metrics_PREDIX_HER2_new.txt
#   RNA_curated_metrics_ISPY2.txt  RNA_curated_metrics_NCT02326974.txt
#
# Launch (survives logout, blocks system sleep for the duration):
#   nohup systemd-inhibit --what=sleep:idle --why="PREDIX production run" \
#       bash production_run_ubuntu.sh > logs/nohup.out 2>&1 &
# Progress:
#   cat logs/production_status.txt
#   tail -n 20 logs/step1_models.log
# =============================================================================

set -u
cd "$(dirname "$0")"

DATA="clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
SEED=42
RESULTS="results"
REPORT="report"
LOGS="logs"

# All cores minus two (leave headroom for the system); at least 1.
NPROC=$(nproc)
NJOBS=$(( NPROC > 2 ? NPROC - 2 : NPROC ))

mkdir -p "${LOGS}"
STATUS="${LOGS}/production_status.txt"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "${STATUS}"
}

run_step() {
    local name="$1" logfile="$2"
    shift 2
    log_status "START ${name}"
    if "$@" > "${LOGS}/${logfile}" 2>&1; then
        log_status "DONE  ${name}"
    else
        local rc=$?
        log_status "FAILED ${name} (exit ${rc}) - see ${LOGS}/${logfile}. STOPPING."
        exit 1
    fi
}

PY=python3
command -v "${PY}" >/dev/null || { echo "python3 not found"; exit 1; }

log_status "PRODUCTION RUN LAUNCHED (n_jobs=${NJOBS} of ${NPROC} cores, seed=${SEED}, data=${DATA})"

# Step 0: statistics test suite (fast gate)
run_step "step0 tests" "step0_tests.log" \
    "${PY}" tests/test_statistics.py

# Step 1: models - the main computational step (hours)
run_step "step1 models (5x200 global, 5x100 arms)" "step1_models.log" \
    "${PY}" multimodal_pcr_pipeline.py \
        --data_path "${DATA}" \
        --results_dir "${RESULTS}" --splits_dir "${RESULTS}/shared_splits" \
        --mode elasticnet --training_data expanded \
        --experiments global dhp tdm1 \
        --classifiers ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear \
        --repeats_global 200 --repeats_arm 100 \
        --outer_folds_global 5 --outer_folds_arm 5 \
        --inner_folds_global 5 --inner_folds_arm 3 \
        --univariate_screen in_fold --feature_pool curated \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

# Step 2: figures and tables
run_step "step2 report" "step2_report.log" \
    "${PY}" generate_report.py --results_dir "${RESULTS}" --out_dir "${REPORT}"

# Step 3: revision analyses.
# NOTE: runs with the DEFAULT S-group spec (still unconfirmed).
# When the confirmed spec exists, re-run JUST this step with --s_group_spec;
# it reads only the PKLs, so the re-run takes minutes, not hours.
run_step "step3 revision analyses" "step3_revision.log" \
    "${PY}" revision_analyses.py --results_dir "${RESULTS}" --out_dir "${REPORT}" \
        --data_path "${DATA}" --n_boot 2000 --n_perm 1000

# Step 4a: transferable feature lists
run_step "step4a shared features" "step4a_shared.log" \
    "${PY}" external_validation.py --predix "${DATA}" \
        --ispy2 RNA_curated_metrics_ISPY2.txt \
        --nct RNA_curated_metrics_NCT02326974.txt \
        --out_dir "${REPORT}" --export_shared_features_only

# Step 4b: RNA-only pipeline runs (locked models), one per cohort
run_step "step4b RNA-only dhp (I-SPY2 features)" "step4b_rna_ispy2.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_ispy2 --splits_dir results_rna_ispy2/splits \
        --mode elasticnet --training_data cc_only --experiments dhp \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_I-SPY2.txt" \
        --repeats_arm 100 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

run_step "step4b RNA-only tdm1 (NCT features)" "step4b_rna_nct.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_nct --splits_dir results_rna_nct/splits \
        --mode elasticnet --training_data cc_only --experiments tdm1 \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_NCT02326974.txt" \
        --repeats_arm 100 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

# Step 4c: locked external validation
run_step "step4c locked external validation" "step4c_external.log" \
    "${PY}" external_validation.py --predix "${DATA}" \
        --ispy2 RNA_curated_metrics_ISPY2.txt \
        --nct RNA_curated_metrics_NCT02326974.txt \
        --locked_ispy2 results_rna_ispy2 --locked_nct results_rna_nct \
        --out_dir "${REPORT}" --n_boot 2000

log_status "PRODUCTION RUN COMPLETE. Results: ${RESULTS} | Report: ${REPORT}"
log_status "Quote numbers ONLY from ${REPORT}/tables/revision (patient-level bootstrap CIs)."
