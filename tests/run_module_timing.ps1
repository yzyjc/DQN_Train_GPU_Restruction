# GPT6-Astra: rotated OFF/modules/deep comparison; isolated network, no Trainer.
param([string]$Exe="$PSScriptRoot/../build_verify/workload_profile/smoke.exe")
$ErrorActionPreference='Stop'
$Exe=(Resolve-Path -LiteralPath $Exe).Path
$dir=Join-Path (Split-Path $Exe) ('module_compare_'+(Get-Date -Format yyyyMMdd_HHmmss)+'_'+$PID)
New-Item -ItemType Directory -Path $dir|Out-Null
$oldMode=$env:DQN_GPU_PROFILE_MODE;$oldCount=$env:DQN_GPU_PROFILE_BATCHES;$oldDir=$env:DQN_GPU_PROFILE_DIR
try {
    $modes=@('off','modules','deep');$hash=$null
    for($round=0;$round -lt 3;$round++) {
        for($i=0;$i -lt 3;$i++) {
            $mode=$modes[($i+$round)%3]
            $env:DQN_GPU_PROFILE_MODE=$mode
            $env:DQN_GPU_PROFILE_BATCHES=if($mode -eq 'off'){'0'}else{'1'}
            $env:DQN_GPU_PROFILE_DIR=$dir
            $output=Join-Path $dir ($mode+'_'+$round+'.bin')
            & $Exe $output
            if($LASTEXITCODE -ne 0){throw "Failed $mode round $round"}
            $actual=(Get-FileHash -LiteralPath $output).Hash
            if($hash -and $hash -ne $actual){throw 'Parameter mismatch'}
            $hash=$actual
        }
    }
    if(@(Get-ChildItem $dir -Filter 'gpu_modules_*.csv').Count -ne 3){throw 'Missing background module captures'}
    if(@(Get-ChildItem $dir -Filter 'gpu_profile_*.csv').Count -ne 3){throw 'Missing deep captures'}
    $json=& python "$PSScriptRoot/../GPU_Analyzer/compare_module_timing.py" $dir
    if($LASTEXITCODE -ne 0){throw 'Invalid module comparison'}
    $report=$json | ConvertFrom-Json
    foreach($capture in $report.module_captures) {
        $c=$capture.operation_counters
        if($c.kernel_launches -ne 1025 -or $c.memset_calls -ne 193 -or $c.memset_bytes -ne 464004 -or $c.sync_calls -ne 225){throw 'Incomplete batch operation coverage'}
        if($c.h2d_calls -ne 1 -or $c.h2d_bytes -ne 3328 -or $c.d2h_calls -ne 2 -or $c.d2h_bytes -ne 132 -or $c.d2d_calls -ne 34 -or $c.d2d_bytes -ne 6792008){throw 'Incorrect copy accounting'}
    }
    Write-Output $json
    Write-Output "PASS 9/9 parameter equivalence; reports=$dir"
} finally {
    $env:DQN_GPU_PROFILE_MODE=$oldMode;$env:DQN_GPU_PROFILE_BATCHES=$oldCount;$env:DQN_GPU_PROFILE_DIR=$oldDir
}
