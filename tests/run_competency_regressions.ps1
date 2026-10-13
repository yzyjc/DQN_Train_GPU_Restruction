param([switch]$SkipBuild,[switch]$IncludeGpu,[string]$OutputDirectory='build_verify\competency_suite')
# Collaboration signature: GPT6-Astra. Each test gets a new runtime directory.
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$out=if([IO.Path]::IsPathRooted($OutputDirectory)){$OutputDirectory}else{Join-Path $repo $OutputDirectory}
$vs='D:\Program Files\Microsoft Visual Studio\18\Community'
$rocm='D:\AMD_ROCm'
if(Get-Process DQN_Train_Trainer_NoTerminal -ErrorAction SilentlyContinue){throw 'Stop: live Trainer detected; this suite uses the Trainer pipe and Probe mappings.'}
New-Item -ItemType Directory -Force $out | Out-Null
if(!$SkipBuild){foreach($p in @('DQN_Core','DQN_Train_Trainer_NoTerminal','DQN_Main_TUI_No_Terminal')){
    & "$vs\MSBuild\Current\Bin\amd64\MSBuild.exe" "$repo\$p\$p.vcxproj" /t:Build /p:Configuration=Release /p:Platform=x64 /p:PreferredToolArchitecture=x64 /p:BuildProjectReferences=false "/p:OutDir=$out\" "/p:IntDir=$out\int_$p\" /m:1 /v:minimal /nologo > "$out\build_$p.log" 2>&1
    if($LASTEXITCODE){throw "$p build failed: $out\build_$p.log"}
}}
Copy-Item -LiteralPath "$rocm\bin\amdhip64_7.dll" -Destination "$out\amdhip64_7.dll"
$tests=@('outcome_revision_regression','probe_diagnostics_regression','competency_state_regression',
 'competency_checkpoint_regression','competency_view_regression','local_http_api_regression','local_http_transport_regression','trainer_http_bridge_regression',
 'checkpoint_v13_persistence_regression','independent_side_isolation_regression','reward_propagation_contract_regression','hot_config_course_regression','competency_rehearsal_lr_regression')
if($IncludeGpu){$tests+=@('step_outcome_regression','competency_gpu_regression')}
$runtime=Join-Path $out ('run_'+(Get-Date -Format yyyyMMdd_HHmmss_fff));New-Item -ItemType Directory -Force $runtime | Out-Null
foreach($test in $tests){
    $extra=''
    if($test -eq 'local_http_transport_regression'){$extra='"'+$repo+'\DQN_Main_TUI_No_Terminal\local_http_server.cpp"'}
    if($test -eq 'trainer_http_bridge_regression'){$extra='"'+$repo+'\DQN_Main_TUI_No_Terminal\trainer_http_bridge.cpp"'}
    $compile='"{0}\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I"{1}\DQN_Core\include" "{1}\tests\{2}.cpp" {3} /Fo:"{4}/" /Fe:"{4}\{2}.exe" "{4}\DQN_Core.lib" /link /LTCG /LIBPATH:"{5}\lib" amdhip64.lib user32.lib gdi32.lib ws2_32.lib bcrypt.lib /STACK:16777216' -f $vs,$repo,$test,$extra,$out,$rocm
    & cmd /d /s /c $compile > "$out\compile_$test.log" 2>&1
    if($LASTEXITCODE){throw "$test compile failed: $out\compile_$test.log"}
    $dir=Join-Path $runtime $test;New-Item -ItemType Directory -Force $dir,"$dir\checkpoints","$dir\logs" | Out-Null
    Push-Location $dir
    try{& cmd /d /s /c ('"'+$out+'\'+$test+'.exe" > output.txt 2>&1');$rc=$LASTEXITCODE;Get-Content output.txt -Tail 2;if($rc){throw "$test failed ($rc): $dir"}}finally{Pop-Location}
}
"PASS. Evidence: $runtime"
