#!/usr/bin/env bash
# =============================================================================
# PREDIX HER2 — production run
# =============================================================================
#
# WHAT THIS RUNS
# --------------
# The analysis reported in the manuscript, end to end, from the curated input
# matrix to every deposited table and figure:
#
#   0  preflight        checks the input file and enforces the deduplication
#                       contract; the run does not start if either fails
#   1  model fitting    the only step that fits models -- hours, not minutes
#   2  report           tables and figures from the fold-level artefacts
#   3  revision         confidence intervals, comparisons, calibration, EPV
#   4  external         locked signatures applied once to the external cohorts
#   5  tests            the statistical primitives
#   6  scale check      reads the provenance back and fails if the run did not
#                       execute the design below
#
# THE FLAGS THAT DECIDE THE ANALYSIS
# ----------------------------------
#   --training_data cc_only    Every model is trained AND evaluated on the
#                              complete multimodal cases. THIS IS NOT THE
#                              PIPELINE'S DEFAULT (the default is `expanded`),
#                              so a command line that omits it does not run this
#                              analysis.
#   --dedup_per_scenario       Within-modality redundancy is removed inside each
#                              scenario's own cohort, once, before any split.
#                              Outcome-blind: correlation, exact agreement and
#                              Cohen's kappa between FEATURES only. Off by
#                              default. See feature_deduplication.py.
#   --univariate_screen in_fold  The association step runs inside each training
#                              fold; no held-out patient influences which
#                              features enter a model.
#   --repeats_global 200 / --repeats_arm 100
#                              NOT DEFAULTS. The pipeline defaults to 20 and 10
#                              (its own help text says "Production: 200/100").
#                              A run that omits them completes, writes every
#                              artefact and looks normal; only the provenance
#                              shows it ran at a tenth of the design. Step 6
#                              exists for exactly that reason.
#   --seed 42                  Cross-validation partitions are fully determined
#                              by this.
#
# Left at pipeline defaults, recorded so the design is in one place:
#   --signature_source winner_folds   --corr_threshold 0.90
#   --feature_pool curated            5 classifier families, inner 5 / 3 folds
#
# WHAT MUST SIT NEXT TO THIS FILE
# -------------------------------
#   multimodal_pcr_pipeline.py  feature_deduplication.py  cv_estimands.py
#   revision_analyses.py  generate_report.py  external_validation.py
#   apply_locked_external_validation.py  apply_locked_signatures.py
#   preflight.py  tests/test_statistics.py
#   clin_multiomics_curated_metrics_PREDIX_HER2_new.txt
#   RNA_curated_metrics_ISPY2.txt  RNA_curated_metrics_NCT02326974.txt
#   RNA_curated_metrics_TransNEO_rawcounts.txt  DNA_curated_metrics_TransNEO.txt
#   (the TransNEO files are third-party and are not redistributed with the code)
# =============================================================================
set -uo pipefail

TRAINING_DATA="cc_only"          # the locked design; see the header
# PASSED EXPLICITLY. The pipeline's default classifier set is not the same as
# the set a run used unless the run said so, and a run that omitted this flag
# once competed one more family than the manuscript describes. Step 6 asserts
# the provenance recorded exactly this list.
CLASSIFIERS="ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_RBF SVM_Linear"
NJOBS="${NJOBS:-$(( $(nproc) - 2 ))}"
(( NJOBS < 1 )) && NJOBS=1
SEED=42
DATA="clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
RESULTS="results"
REPORT="report"
LOGS="logs"
mkdir -p "${LOGS}"

log_status () { echo "[$(date '+%F %T')] $*" | tee -a "${LOGS}/status.txt"; }

run_step () {   # run_step <label> <logfile> <command...>
  local label="$1"; shift
  local logf="$1"; shift
  log_status "START ${label}"
  local t0=$SECONDS
  "$@" > "${LOGS}/${logf}" 2>&1
  local rc=$?
  local mins=$(( (SECONDS - t0) / 60 ))
  if [ $rc -ne 0 ]; then
    log_status "FAILED ${label} (rc=${rc}, ${mins} min) — see ${LOGS}/${logf}"
    tail -n 25 "${LOGS}/${logf}"
    exit $rc
  fi
  log_status "OK ${label} (${mins} min)"
}

log_status "PRODUCTION RUN LAUNCHED  training_data=${TRAINING_DATA}  njobs=${NJOBS}  seed=${SEED}"

run_step "preflight" "step0_preflight.log" python3 preflight.py

run_step "model fitting" "step1_model.log" \
  python3 multimodal_pcr_pipeline.py \
    --data_path "${DATA}" \
    --results_dir "${RESULTS}" \
    --splits_dir shared_splits \
    --training_data "${TRAINING_DATA}" \
    --dedup_per_scenario \
    --univariate_screen in_fold \
    --feature_pool curated \
    --experiments global dhp tdm1 \
    --outer_folds_global 5 --repeats_global 200 --inner_folds_global 5 \
    --outer_folds_arm    5 --repeats_arm    100 --inner_folds_arm    3 \
    --classifiers ${CLASSIFIERS} \
    --consensus \
    --seed "${SEED}" \
    --n_jobs "${NJOBS}"

run_step "report" "step2_report.log" \
  python3 generate_report.py --results_dir "${RESULTS}" --out_dir "${REPORT}"

run_step "revision analyses" "step3_revision.log" \
  python3 revision_analyses.py --results_dir "${RESULTS}" --out_dir "${REPORT}" \
                               --data_path "${DATA}" --n_boot 2000

# The locked signature, classifier and hyper-parameters are read from results/
# and applied ONCE to each external cohort; nothing is refitted on external data.
# A model is scored in a cohort only if every feature of its signature is
# measured there, otherwise it is recorded as not scoreable with the missing
# features named. Writes into report/tables/revision/.
run_step "external validation" "step4_external.log" \
  python3 apply_locked_external_validation.py --run .

run_step "statistics tests" "step5_tests.log" python3 tests/test_statistics.py

run_step "scale check" "step6_scale.log" python3 - "${TRAINING_DATA}" "${CLASSIFIERS}" <<'PYEOF'
import json, sys
from pathlib import Path
mode, clfs = sys.argv[1], sys.argv[2].split()
WANT = {"repeats_global": 200, "repeats_arm": 100, "outer_folds_global": 5,
        "outer_folds_arm": 5, "dedup_per_scenario": True, "seed": 42,
        "univariate_screen": "in_fold", "training_data": mode,
        "classifiers": clfs}
par = json.loads(Path("results/run_provenance.json").read_text(encoding="utf-8-sig"))["parameters"]
bad = [f"{k} = {par.get(k)!r}, expected {v!r}" for k, v in WANT.items() if par.get(k) != v]
print(f"training_data={par.get('training_data')!r} repeats={par.get('repeats_global')}/"
      f"{par.get('repeats_arm')} dedup={par.get('dedup_per_scenario')} seed={par.get('seed')}")
if bad:
    print("\nTHE RUN DID NOT EXECUTE THE DESIGN THIS SCRIPT DESCRIBES:")
    for b in bad:
        print("   ", b)
    sys.exit(1)
print("the run executed the described design.")
PYEOF

log_status "PRODUCTION RUN COMPLETE"
echo
echo "================================================================"
echo "COPY BACK, PRESERVING THESE NAMES:"
echo "  results/  report/  shared_splits/  logs/"
echo
echo "The per-scenario deduplication audits are at"
echo "  results/<scenario>/<scenario>_deduplication_audit.csv"
echo "They record every removed feature, what was kept in its place,"
echo "and which statistic justified it."
echo "================================================================"
tar -czf logs.tar.gz "${LOGS}"
