# =============================================================================
# PREDIX HER2 - production run (Windows PowerShell mirror of production_run_ubuntu.sh)
# =============================================================================
# The .sh is the reference; if the two ever disagree, the .sh is right. Same
# steps, same flags, same design. Read the header of production_run_ubuntu.sh
# for what each flag decides and why the fold counts are passed explicitly.
#
# Models are trained and evaluated on the complete multimodal cases. Within-
# modality redundancy is removed inside each scenario's cohort before any split
# (outcome-blind). The univariate screen runs inside each training fold.
# =============================================================================
$ErrorActionPreference = "Continue"
$TRAINING_DATA = "cc_only"           # the locked design
# Passed explicitly and asserted by the scale check; see the .sh header.
$CLASSIFIERS = "ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_RBF SVM_Linear"
$NCORES = [int]$env:NUMBER_OF_PROCESSORS
$NJOBS  = [Math]::Max(1, $NCORES - 2)
$SEED   = 42
$DATA   = "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
$LOGS   = "logs"
New-Item -ItemType Directory -Force $LOGS | Out-Null

function Log-Status([string]$msg) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
  $line | Tee-Object -FilePath "$LOGS\status.txt" -Append
}

function Run-Step([string]$label, [string]$logf, [string]$cmd) {
  Log-Status "START $label"
  $t0 = Get-Date
  Invoke-Expression "$cmd *> `"$LOGS\$logf`""
  $rc = $LASTEXITCODE
  $mins = [int]((Get-Date) - $t0).TotalMinutes
  if ($rc -ne 0) {
    Log-Status "FAILED $label (rc=$rc, $mins min) - see $LOGS\$logf"
    Get-Content "$LOGS\$logf" -Tail 25
    exit $rc
  }
  Log-Status "OK $label ($mins min)"
}

Log-Status "PRODUCTION RUN LAUNCHED  training_data=$TRAINING_DATA  njobs=$NJOBS  seed=$SEED"

Run-Step "preflight" "step0_preflight.log" "python preflight.py"

Run-Step "model fitting" "step1_model.log" ("python multimodal_pcr_pipeline.py " +
  "--data_path $DATA --results_dir results --splits_dir shared_splits " +
  "--training_data $TRAINING_DATA --dedup_per_scenario " +
  "--univariate_screen in_fold --feature_pool curated " +
  "--experiments global dhp tdm1 " +
  "--outer_folds_global 5 --repeats_global 200 --inner_folds_global 5 " +
  "--outer_folds_arm 5 --repeats_arm 100 --inner_folds_arm 3 " +
  "--classifiers $CLASSIFIERS " +
  "--consensus --seed $SEED --n_jobs $NJOBS")

Run-Step "report" "step2_report.log" "python generate_report.py --results_dir results --out_dir report"

Run-Step "revision analyses" "step3_revision.log" ("python revision_analyses.py " +
  "--results_dir results --out_dir report --data_path $DATA --n_boot 2000")

Run-Step "external validation" "step4_external.log" "python apply_locked_external_validation.py --run ."

Run-Step "statistics tests" "step5_tests.log" "python tests/test_statistics.py"

# Scale check: a tenth-scale run completes and looks normal; only the
# provenance shows it. Read it back rather than trust it.
$scale = @"
import json, sys
from pathlib import Path
mode, clfs = sys.argv[1], sys.argv[2].split()
WANT = {'repeats_global': 200, 'repeats_arm': 100, 'outer_folds_global': 5,
        'outer_folds_arm': 5, 'dedup_per_scenario': True, 'seed': 42,
        'univariate_screen': 'in_fold', 'training_data': mode,
        'classifiers': clfs}
par = json.loads(Path('results/run_provenance.json').read_text(encoding='utf-8-sig'))['parameters']
bad = [f'{k} = {par.get(k)!r}, expected {v!r}' for k, v in WANT.items() if par.get(k) != v]
print(f"training_data={par.get('training_data')!r} repeats={par.get('repeats_global')}/{par.get('repeats_arm')} dedup={par.get('dedup_per_scenario')}")
if bad:
    print('THE RUN DID NOT EXECUTE THE DESCRIBED DESIGN:'); [print('   ', b) for b in bad]; sys.exit(1)
print('the run executed the described design.')
"@
$scale | Set-Content -Encoding ascii "$LOGS\_scale_check.py"
Run-Step "scale check" "step6_scale.log" "python $LOGS\_scale_check.py $TRAINING_DATA `"$CLASSIFIERS`""

Log-Status "PRODUCTION RUN COMPLETE"
Write-Host ""
Write-Host "COPY BACK, PRESERVING THESE NAMES:  results\  report\  shared_splits\  logs\"
Write-Host "Deduplication audits: results\<scenario>\<scenario>_deduplication_audit.csv"
