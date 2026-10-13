param(
    [string]$VsRoot = 'D:\Program Files\Microsoft Visual Studio\18\Community',
    [string]$RocmRoot = 'D:\AMD_ROCm'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$output = Join-Path $repo 'build_verify\training_regressions'
$runtime = Join-Path $output ('run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path (Join-Path $runtime 'logs'), (Join-Path $runtime 'checkpoints') -Force | Out-Null
$msbuild = Join-Path $VsRoot 'MSBuild\Current\Bin\MSBuild.exe'
& $msbuild (Join-Path $repo 'DQN_Core\DQN_Core.vcxproj') /t:Build /p:Configuration=Release /p:Platform=x64 "/p:OutDir=$output\" "/p:DqnRocmRoot=$RocmRoot" /m:1 /v:minimal /nologo
if ($LASTEXITCODE) { throw 'Core build failed' }
$devcmd = Join-Path $VsRoot 'Common7\Tools\VsDevCmd.bat'
$compile = '"{0}" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /MD /EHsc /utf-8 /I"{1}\DQN_Core\include" /I"{2}\include" "{1}\tests\training_regression.cpp" "{3}\DQN_Core.lib" /Fe:"{3}\training_regression.exe" /Fo:"{3}\training_regression.obj" /link /LTCG /LIBPATH:"{2}\lib" amdhip64.lib /STACK:16777216' -f $devcmd,$repo,$RocmRoot,$output
& cmd /d /s /c $compile
if ($LASTEXITCODE) { throw 'Regression executable build failed' }
$savedPath = $env:PATH
Push-Location $runtime
try {
    $env:PATH = "$RocmRoot\bin;" + $savedPath
    & (Join-Path $output 'training_regression.exe') *> regression_output.txt
    $testExit = $LASTEXITCODE
    Get-Content regression_output.txt
    if ($testExit) { throw "Regression failed ($testExit). Evidence: $runtime" }
} finally {
    $env:PATH = $savedPath
    Pop-Location
}
Write-Output "PASS. Isolated test evidence: $runtime"
