param([string]$VsRoot='D:\Program Files\Microsoft Visual Studio\18\Community', [string]$RocmRoot='D:\AMD_ROCm',
    [string[]]$Tests=@('training_regression','audit_regression','vindicator_regression','recheck_regression','recheck_vindicator_regression','trainer_exit_regression'))
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$output=Join-Path $repo 'build_verify\recheck_fix_20260831'
$runtime=Join-Path $output ('run_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
& (Join-Path $VsRoot 'MSBuild\Current\Bin\MSBuild.exe') (Join-Path $repo 'DQN_Core\DQN_Core.vcxproj') /t:Build /p:Configuration=Release /p:Platform=x64 "/p:OutDir=$output\" /m:1 /v:minimal /nologo
if($LASTEXITCODE){throw 'Core build failed'}
$savedPath=$env:PATH
try {
    $env:PATH="$RocmRoot\bin;"+$savedPath
    foreach($test in $Tests) {
        $link=if($test -like '*vindicator*'){''}else{'"{0}\DQN_Core.lib" /link /LTCG /LIBPATH:"{1}\lib" amdhip64.lib user32.lib gdi32.lib /STACK:16777216' -f $output,$RocmRoot}
        $compile='"{0}\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I"{1}\DQN_Core\include" /I"{4}\include" "{1}\tests\{2}.cpp" /Fe:"{3}\{2}.exe" /Fo:"{3}\{2}.obj" {5}' -f $VsRoot,$repo,$test,$output,$RocmRoot,$link
        & cmd /d /s /c $compile
        if($LASTEXITCODE){throw "$test build failed"}
        $testDir=Join-Path $runtime $test
        New-Item -ItemType Directory -Path $testDir -Force | Out-Null
        New-Item -ItemType Directory -Path (Join-Path $testDir 'logs'), (Join-Path $testDir 'checkpoints') -Force | Out-Null
        Push-Location $testDir
        try {
            & (Join-Path $output "$test.exe") *> output.txt
            $result=$LASTEXITCODE
            Get-Content output.txt
            if($result){throw "$test failed ($result): $testDir"}
        } finally {Pop-Location}
    }
} finally {$env:PATH=$savedPath}
Write-Output "PASS. Evidence: $runtime"
