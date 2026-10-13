# GPT6-Astra: execute only the isolated smoke binary, never Trainer.
param([string]$Exe = "$PSScriptRoot/../build_verify/workload_profile/smoke.exe")
$ErrorActionPreference='Stop'
$Exe=(Resolve-Path -LiteralPath $Exe).Path
$directory=Join-Path (Split-Path $Exe) ('capture_'+(Get-Date -Format yyyyMMdd_HHmmss)+'_'+$PID)
New-Item -ItemType Directory -Path $directory|Out-Null
$oldCount=$env:DQN_GPU_PROFILE_BATCHES
$oldDir=$env:DQN_GPU_PROFILE_DIR
try {
    $baseline=Join-Path $directory 'baseline.bin'
    $profiled=Join-Path $directory 'profiled.bin'
    $env:DQN_GPU_PROFILE_BATCHES='0'
    & $Exe $baseline
    if($LASTEXITCODE -ne 0){throw 'Baseline GPU test failed'}
    $env:DQN_GPU_PROFILE_BATCHES='1'
    $env:DQN_GPU_PROFILE_DIR=$directory
    & $Exe $profiled
    if($LASTEXITCODE -ne 0){throw 'Profiled GPU test failed'}
    if((Get-FileHash $baseline).Hash -ne (Get-FileHash $profiled).Hash){throw 'Profiling changed trained parameters'}
    $traces=@(Get-ChildItem -LiteralPath $directory -Filter 'gpu_profile_*.csv')
    if($traces.Count -ne 1){throw 'Expected exactly one bounded capture'}
    & python "$PSScriptRoot/../GPU_Analyzer/analyze_workload_trace.py" $traces[0].FullName --output (Join-Path $directory 'summary.json')
    if($LASTEXITCODE -ne 0){throw 'Trace validation failed'}
    Write-Output "PASS parameter parity and bounded capture; inspect event validity in $directory/summary.json"
} finally {
    $env:DQN_GPU_PROFILE_BATCHES=$oldCount
    $env:DQN_GPU_PROFILE_DIR=$oldDir
}
