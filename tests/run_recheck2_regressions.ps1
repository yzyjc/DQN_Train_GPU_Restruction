param([switch]$SkipBuild, [switch]$AllowLiveTrainerInterference, [string[]]$Tests=@('training_regression','audit_regression','vindicator_regression','recheck_regression','recheck_vindicator_regression','trainer_exit_regression','recheck2_regression','recheck2_watchdog_regression','recheck2_ui_regression','probe3d_regression'), [string]$OutputDirectory='build_verify\recheck2_fix_20260831')
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$output=if([IO.Path]::IsPathRooted($OutputDirectory)){$OutputDirectory}else{Join-Path $repo $OutputDirectory}
$vs='D:\Program Files\Microsoft Visual Studio\18\Community';$rocm='D:\AMD_ROCm'
if(!$AllowLiveTrainerInterference -and (Get-Process DQN_Train_Trainer_NoTerminal -ErrorAction SilentlyContinue)) {
    # These integration tests use the production named pipe or Probe mapping.
    # Keep safe unit/renderer tests available beside a real training session.
    $unsafe=@('audit_regression','trainer_exit_regression','recheck2_regression','train_fix_regression')
    $requested=$Tests
    $Tests=@($Tests | Where-Object { $_ -notin $unsafe })
    $skipped=@($requested | Where-Object { $_ -in $unsafe })
    if($skipped.Count){Write-Warning ('Live Trainer detected; skipped intrusive tests: '+($skipped -join ', '))}
}
New-Item -ItemType Directory -Path $output -Force | Out-Null
if(!$SkipBuild){
    foreach($project in @('DQN_Core','DQN_Train_Trainer_NoTerminal','DQN_Main_TUI_No_Terminal','Watchdog_NoTerminal','Log_Vindicator')){
        & "$vs\MSBuild\Current\Bin\MSBuild.exe" "$repo\$project\$project.vcxproj" /t:Build /p:Configuration=Release /p:Platform=x64 /p:PreferredToolArchitecture=x64 "/p:OutDir=$output\" /m:1 /v:minimal /nologo
        if($LASTEXITCODE){throw "$project build failed"}
    }
}
$runtime=Join-Path $output ('run_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $runtime | Out-Null
$savedPath=$env:PATH
try{
    $env:PATH="$rocm\bin;"+$savedPath
    foreach($test in $Tests){
        $objs=''
        if($test -like '*ui_regression'){
            $objs=(@('dqn_report','dqn_replay','gpu_stubs','playground','terminal_window','virtual_terminal',
                'transition_overlay','neural_viewer_3d','neural_viewer_camera','neural_viewer_layout') | ForEach-Object {'"'+$repo+'\DQN_Main_TUI_No_Terminal\x64\Release_NoTerminal\'+$_+'.obj"'}) -join ' '
        }elseif($test -eq 'probe3d_regression'){
            $objs=(@('neural_viewer_3d','neural_viewer_camera','neural_viewer_layout') | ForEach-Object {'"'+$repo+'\DQN_Main_TUI_No_Terminal\x64\Release_NoTerminal\'+$_+'.obj"'}) -join ' '
        }
        $libs=if($test -like '*vindicator*' -or $test -eq 'metric_log_regression'){'/link'}elseif($test -like '*watchdog*'){'/link user32.lib gdi32.lib'}elseif($test -eq 'probe3d_regression'){'"{0}\DQN_Core.lib" /link /LTCG d3d11.lib d3dcompiler.lib user32.lib gdi32.lib /STACK:1048576' -f $output}elseif($test -like '*ui_regression'){'"{0}\DQN_Core.lib" /link /LTCG /LIBPATH:"{1}\lib" amdhip64.lib d3d11.lib d3dcompiler.lib user32.lib gdi32.lib /STACK:16777216' -f $output,$rocm}else{'"{0}\DQN_Core.lib" /link /LTCG /LIBPATH:"{1}\lib" amdhip64.lib user32.lib gdi32.lib /STACK:16777216' -f $output,$rocm}
        $compile='"{0}\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I"{1}\DQN_Core\include" /I"{2}\include" "{1}\tests\{3}.cpp" /Fe:"{4}\{3}.exe" /Fo:"{4}\{3}.obj" {5} {6}' -f $vs,$repo,$rocm,$test,$output,$objs,$libs
        & cmd /d /s /c $compile
        if($LASTEXITCODE){throw "$test build failed"}
        $dir=Join-Path $runtime $test;New-Item -ItemType Directory -Path $dir | Out-Null
        New-Item -ItemType Directory -Path "$dir\logs","$dir\checkpoints" | Out-Null
        Push-Location $dir
        try{
            # ROCm writes ordinary diagnostics to stderr. Redirect in cmd so
            # Windows PowerShell does not turn successful native stderr into a
            # noisy NativeCommandError/RemoteException record.
            $native='"{0}" > output.txt 2>&1' -f (Join-Path $output "$test.exe")
            & $env:ComSpec /d /s /c $native
            $code=$LASTEXITCODE
            Get-Content output.txt
            if($code){throw "$test failed ($code): $dir"}
        }finally{Pop-Location}
    }
}finally{$env:PATH=$savedPath}
"PASS. Evidence: $runtime"
