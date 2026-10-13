# GPT6-Astra: capture only the isolated GPU smoke executable; never attach to Trainer.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RdpCli,
    [string]$SmokeExe = (Join-Path $PSScriptRoot '..\build_verify\workload_profile\smoke.exe'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\build_verify\rdts'),
    # The isolated batch-32 smoke emits about 1,025 dispatches. Allow a full
    # batch capture so callers are not limited to a prefix profile.
    [ValidateRange(1,2048)][int]$DispatchCount = 32,
    [ValidateRange(10000,60000)][int]$HoldMs = 60000,
    [switch]$InstructionTracing,
    [Parameter(Mandatory=$true)][switch]$IConfirmNoTraining
)

$ErrorActionPreference = 'Stop'
$RdpCli = (Resolve-Path -LiteralPath $RdpCli).Path
$SmokeExe = (Resolve-Path -LiteralPath $SmokeExe).Path
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if (-not $IConfirmNoTraining) { throw 'Pass -IConfirmNoTraining only after confirming no formal training is running.' }

$runtimeName = 'amdhip64_7.dll'
$runtimeSource = Join-Path (Split-Path -Parent $SmokeExe) $runtimeName
if (-not (Test-Path -LiteralPath $runtimeSource)) {
    $hipBin = if ($env:HIP_PATH) { Join-Path $env:HIP_PATH 'bin' } else { $null }
    if ($hipBin) { $runtimeSource = Join-Path $hipBin $runtimeName }
}
if (-not (Test-Path -LiteralPath $runtimeSource)) { throw "HIP runtime not found: $runtimeName beside smoke executable or under HIP_PATH\bin." }

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$captureDir = Join-Path $OutputRoot ('capture_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + $PID)
New-Item -ItemType Directory -Path $captureDir | Out-Null
$appDir = Join-Path $captureDir 'app'
New-Item -ItemType Directory -Path $appDir | Out-Null
$captureExe = Join-Path $appDir 'smoke.exe'
Copy-Item -LiteralPath $SmokeExe -Destination $captureExe
Copy-Item -LiteralPath $runtimeSource -Destination (Join-Path $appDir $runtimeName)

$baseline = Join-Path $captureDir 'baseline.bin'
$profiled = Join-Path $captureDir 'profiled.bin'
$profile = Join-Path $captureDir 'hip_batch32_counters.rgp'
$stdout = Join-Path $captureDir 'rdp_stdout.txt'
$stderr = Join-Path $captureDir 'rdp_stderr.txt'
$oldHold = $env:DQN_RGP_HOLD_MS
$rdp = $null
try {
    Remove-Item Env:DQN_RGP_HOLD_MS -ErrorAction SilentlyContinue
    & $SmokeExe $baseline
    if ($LASTEXITCODE -ne 0) { throw "Unprofiled GPU smoke failed with exit code $LASTEXITCODE." }

    $rdpArgs = @('--verbose','--mode','profiling','--process','smoke.exe',
        '--rgp-capture-mode','dispatch','--rgp-render-op-count',[string]$DispatchCount,
        '--rgp-auto-capture',"dispatch:1:$DispatchCount",'--rgp-counter-collection',
        '--output',$profile)
    if ($InstructionTracing) { $rdpArgs += '--rgp-instruction-tracing' }
    $quotedArgs = $rdpArgs | ForEach-Object { '"' + ([string]$_).Replace('"','\"') + '"' }
    $rdp = Start-Process -FilePath $RdpCli -ArgumentList ($quotedArgs -join ' ') -WorkingDirectory $appDir `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 2
    if ($rdp.HasExited) { throw "RDP CLI exited early ($($rdp.ExitCode)); inspect $stdout and $stderr." }

    $env:DQN_RGP_HOLD_MS = [string]$HoldMs
    Push-Location $appDir
    try { & $captureExe $profiled } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw "Profiled GPU smoke failed with exit code $LASTEXITCODE." }
    if ((Get-FileHash -LiteralPath $baseline -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $profiled -Algorithm SHA256).Hash) {
        throw 'Baseline and profiled parameter snapshots differ.'
    }

    if (-not $rdp.WaitForExit(180000)) {
        Stop-Process -Id $rdp.Id -Force -ErrorAction SilentlyContinue
        throw "RDP capture timed out; inspect $stdout and $stderr."
    }
    if (-not (Test-Path -LiteralPath $profile) -or (Get-Item -LiteralPath $profile).Length -le 0) {
        throw "RDP reported no usable profile; inspect $stdout and $stderr."
    }
    $rdpLog = Get-Content -LiteralPath $stdout -Raw
    if ($rdpLog -notmatch '\[Status\] Done' -or $rdpLog -notmatch 'Successfully processed derived counters') {
        throw "RDP did not report a completed counter capture; inspect $stdout and $stderr."
    }

    $captureHash = (Get-FileHash -LiteralPath $profile -Algorithm SHA256).Hash
    $rdpExitText = if ($rdp.ExitCode -eq 0) { '0 (normal)' } else { "$($rdp.ExitCode) (abnormal exit after trace completion log)" }
    @(
        'RGP capture: isolated HIP batch=32 smoke workload',
        "Dispatch capture count requested: $DispatchCount",
        "Counters enabled: yes",
        "Profile: $profile",
        "Profile bytes: $((Get-Item -LiteralPath $profile).Length)",
        "Profile SHA256: $captureHash",
        "RDP CLI exit: $rdpExitText",
        "Instruction tracing: $($InstructionTracing.IsPresent)",
        "Parameter parity SHA256: $((Get-FileHash -LiteralPath $profiled -Algorithm SHA256).Hash)",
        'This is a synthetic smoke workload, not a Trainer capture. WGP metrics are out of scope.'
    ) | Set-Content -LiteralPath (Join-Path $captureDir 'capture_manifest.txt') -Encoding UTF8
    if ($rdp.ExitCode -eq 0) { Write-Output "PASS: RGP capture and parameter parity. Output: $captureDir" }
    else { Write-Warning "Profile and counters completed and parameter parity passed, but RDP CLI exited $($rdp.ExitCode). See manifest and logs: $captureDir" }
} finally {
    if ($rdp -and -not $rdp.HasExited) { Stop-Process -Id $rdp.Id -Force -ErrorAction SilentlyContinue }
    if ($null -eq $oldHold) { Remove-Item Env:DQN_RGP_HOLD_MS -ErrorAction SilentlyContinue }
    else { $env:DQN_RGP_HOLD_MS = $oldHold }
}
