param([Parameter(Mandatory=$true)][string]$Executable)
# Collaboration signature: GPT6-Astra. Hidden isolated TUI; automatic Trainer startup disabled.
$ErrorActionPreference='Stop'
$exe=(Resolve-Path $Executable).Path
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if(!$exe.StartsWith((Join-Path $repo 'build_verify')+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'This smoke test only accepts an isolated build_verify executable.'}
$directory=Split-Path $exe
$previous=$env:DQN_TUI_AUTOSTART
$env:DQN_TUI_AUTOSTART='0'
$process=$null
try{
    $process=Start-Process -FilePath $exe -WorkingDirectory $directory -WindowStyle Hidden -PassThru
    $endpoint=Join-Path $directory 'tui_http_endpoint.json'
    $info=$null
    for($i=0;$i -lt 100;++$i){
        if($process.HasExited){throw "Isolated TUI exited early: $($process.ExitCode)"}
        if(Test-Path $endpoint){$candidate=Get-Content -Raw $endpoint | ConvertFrom-Json;if($candidate.pid -eq $process.Id){$info=$candidate;break}}
        Start-Sleep -Milliseconds 100
    }
    if(!$info){throw 'TUI HTTP discovery did not appear.'}
    $headers=@{Authorization='Bearer '+$info.token}
    $health=Invoke-RestMethod ($info.url+'/api/health') -TimeoutSec 5
    if($health.status -ne 'ok'){throw 'Invalid health response.'}
    $status=Invoke-RestMethod ($info.url+'/api/status') -Headers $headers -TimeoutSec 5
    if($status.trainer_connected -ne $false){throw 'Expected offline Trainer; no training should start in this smoke test.'}
    try{Invoke-RestMethod ($info.url+'/api/config') -Method Patch -Headers $headers -ContentType 'application/json' -Body '{"key":"learning_rate","value":2}' -TimeoutSec 5 | Out-Null;throw 'Invalid parameter was accepted.'}
    catch{if(!$_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 400){throw}}
    "PASS isolated TUI GUI process + embedded HTTP health/offline status/invalid-command tutorial; PID=$($process.Id), port=$($info.port)"
}finally{
    $env:DQN_TUI_AUTOSTART=$previous
    if($process -and !$process.HasExited){Stop-Process -Id $process.Id}
    # Forced shutdown is confined to this smoke-test TUI. Remove its own discovery entry.
    if($process -and (Test-Path (Join-Path $directory 'tui_http_endpoint.json'))){
        $path=Join-Path $directory 'tui_http_endpoint.json'
        $owner=Get-Content -Raw $path | ConvertFrom-Json
        if($owner.pid -eq $process.Id){Remove-Item -LiteralPath $path}
    }
}
