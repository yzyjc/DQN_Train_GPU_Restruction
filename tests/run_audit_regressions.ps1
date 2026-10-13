param(
    [string]$VsRoot = 'D:\Program Files\Microsoft Visual Studio\18\Community',
    [string]$RocmRoot = 'D:\AMD_ROCm'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$output = Join-Path $repo 'build_verify\fix_20260831'
$runtime = Join-Path $output ('run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
& (Join-Path $VsRoot 'MSBuild\Current\Bin\MSBuild.exe') (Join-Path $repo 'DQN_Core\DQN_Core.vcxproj') /t:Build /p:Configuration=Release /p:Platform=x64 "/p:OutDir=$output\" /m:1 /v:minimal /nologo
if ($LASTEXITCODE) { throw 'Core build failed' }
$compile = '"{0}\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I"{1}\DQN_Core\include" "{1}\tests\audit_regression.cpp" "{2}\DQN_Core.lib" /Fe:"{2}\audit_regression.exe" /Fo:"{2}\audit_regression.obj" /link /LTCG /LIBPATH:"{3}\lib" amdhip64.lib /STACK:16777216 && cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 "{1}\tests\vindicator_regression.cpp" /Fe:"{2}\vindicator_regression.exe" /Fo:"{2}\vindicator_regression.obj"' -f $VsRoot,$repo,$output,$RocmRoot
& cmd /d /s /c $compile
if ($LASTEXITCODE) { throw 'Audit regression build failed' }
$savedPath = $env:PATH
Push-Location $runtime
try {
    $env:PATH = "$RocmRoot\bin;" + $savedPath
    foreach ($test in @('audit_regression','vindicator_regression')) {
        & (Join-Path $output "$test.exe") *> "$test.txt"
        $testExit = $LASTEXITCODE
        Get-Content "$test.txt"
        if ($testExit) { throw "$test failed. Evidence: $runtime" }
    }
} finally {
    $env:PATH = $savedPath
    Pop-Location
}
Write-Output "PASS. Isolated audit regressions: $runtime"
