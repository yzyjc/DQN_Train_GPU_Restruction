param([switch]$NoTrain,[switch]$Hidden)
# Collaboration signature: GPT6-Astra. Prepare an installed SDK, not a bundled driver.
$ErrorActionPreference='Stop'
$roots=@($env:HIP_PATH,'D:\AMD_ROCm')
foreach($base in @('C:\Program Files\AMD\ROCm','C:\Program Files\AMD\HIP')){
    if(Test-Path -LiteralPath $base){$roots+=@(Get-ChildItem -LiteralPath $base -Directory | Sort-Object Name -Descending | Select-Object -ExpandProperty FullName)}
}
$sdk=$roots | Where-Object {$_ -and (Test-Path -LiteralPath (Join-Path $_ 'bin\amdhip64_7.dll'))} | Select-Object -First 1
if(!$sdk){throw 'HIP SDK not found. Install HIP SDK 7.13.0 and set HIP_PATH to its root. See README.md.'}
$exe=Join-Path $PSScriptRoot 'DQN_Main_TUI_No_Terminal.exe'
if(!(Test-Path -LiteralPath $exe)){throw 'TUI executable missing; fully extract the release ZIP first.'}
if(Get-Process DQN_Train_Trainer_NoTerminal,DQN_Main_TUI_No_Terminal -ErrorAction SilentlyContinue | Where-Object {$_.Path -and (Split-Path $_.Path) -eq $PSScriptRoot}){throw 'This installation is already running.'}
foreach($name in @('amdhip64_7.dll','amd_comgr0713.dll','amdocl64.dll')){
    $source=Join-Path $sdk ('bin\'+$name)
    if(!(Test-Path -LiteralPath $source)){throw "Matching SDK dependency missing: $name. Use HIP SDK 7.13.0."}
    $target=Join-Path $PSScriptRoot $name
    if(!(Test-Path -LiteralPath $target) -or (Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){Copy-Item -LiteralPath $source -Destination $target -Force}
}
$env:PATH=(Join-Path $sdk 'bin')+';'+$env:PATH
$env:DQN_TUI_AUTOSTART=if($NoTrain){'0'}else{'1'}
if($Hidden){Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot -WindowStyle Hidden}
else{Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot}
