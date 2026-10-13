param([switch]$SkipBuild, [string]$OutputDirectory='build_verify\architecture_20260831')
# One discoverable entry point; keep dated runners as reproducible historical suites.
$tests=@('training_regression','audit_regression','vindicator_regression',
    'recheck_regression','recheck_vindicator_regression','trainer_exit_regression',
    'recheck2_regression','recheck2_watchdog_regression','recheck2_ui_regression',
    'metric_log_regression','architecture_ui_regression','architecture_vindicator_regression',
    'train_fix_regression','checkpoint_memory_regression',
    'training_gate_regression','sentinel_regression','pool400_checkpoint_roundtrip',
    'trajectory_analyzer_regression','probe3d_regression',
    'rollback_wide_metric_regression','step_outcome_regression',
    'step_outcome_ui_regression','step_outcome_trainer_regression') # GPT6-Astra: outcome quality and full-width rollback
& "$PSScriptRoot\run_recheck2_regressions.ps1" -SkipBuild:$SkipBuild -Tests $tests -OutputDirectory $OutputDirectory
