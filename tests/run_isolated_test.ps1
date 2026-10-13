# 隔离测试运行器：显式运行库路径 + 已加载模块取证
#
# 背景：本机同时存在两个不同版本的 amdhip64_7.dll
#   C:\Windows\SYSTEM32\amdhip64_7.dll  2026-05-05  17,512,464 B  （旧，会让 GPU 调用 0xC0000005）
#   D:\AMD_ROCm\bin\amdhip64_7.dll      2026-05-15  20,687,360 B  （正确）
# Windows 先搜索 exe 所在目录、再搜索 SYSTEM32，因此只要把正确版本放到
# exe 同目录就会赢过 SYSTEM32。本脚本不写 SYSTEM32、不改正式目录，
# 只在隔离输出目录内放置 DLL，并把实际加载到的模块路径打印出来取证。
param(
    [Parameter(Mandatory = $true)][string]$TestName,
    [string]$LibDir = 'build_verify\indep_base_20260920',
    [string]$OutputDir = 'build_verify\indep_gpu_run_20260920',
    [switch]$SkipDll,
    [switch]$Build
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$rocm = 'D:\AMD_ROCm'
$vs = 'D:\Program Files\Microsoft Visual Studio\18\Community'
$lib = if ([IO.Path]::IsPathRooted($LibDir)) { $LibDir } else { Join-Path $repo $LibDir }
$exe = Join-Path $lib "$TestName.exe"

if ($Build) {
    $compile = ('"{0}\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul && ' +
        'cl /nologo /O2 /MD /EHsc /std:c++17 /utf-8 /I"{1}\DQN_Core\include" /I"{2}\include" ' +
        '"{1}\tests\{3}.cpp" /Fe:"{4}\{3}.exe" /Fo:"{4}\{3}.obj" "{4}\DQN_Core.lib" ' +
        '/link /LTCG /LIBPATH:"{2}\lib" amdhip64.lib user32.lib gdi32.lib /STACK:16777216') -f $vs, $repo, $rocm, $TestName, $lib
    & cmd /d /s /c $compile
    if ($LASTEXITCODE) { throw "$TestName build failed" }
}
if (!(Test-Path $exe)) { throw "missing test exe: $exe (build it first, or pass -Build)" }

$rundir = Join-Path $repo (Join-Path $OutputDir $TestName)
# 每次运行都从**完全干净**的目录开始：测试会自己 CreateDirectoryA 建子目录并断言
# 成功，复用上一条运行的残留会让它因 ERROR_ALREADY_EXISTS 假失败；
# 复用旧日志也会让「重建尾部/滚动窗口」类断言读到旧数据。
if (Test-Path $rundir) { Remove-Item $rundir -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $rundir -Force | Out-Null
New-Item -ItemType Directory -Path "$rundir\logs", "$rundir\checkpoints" -Force | Out-Null

# 显式运行库路径：Windows 只搜索「EXE 所在目录」+ SYSTEM32，不会搜索工作目录，
# 所以必须把测试 EXE 和正确的 amdhip64_7.dll 放进同一个目录，
# 让 exe 目录优先命中 ROCm 版本而不是 SYSTEM32 同名旧版本。
$localExe = Join-Path $rundir (Split-Path $exe -Leaf)
Copy-Item $exe $localExe -Force
$loadedFrom = '(not copied)'
if (-not $SkipDll) {
    $src = Join-Path $rocm 'bin\amdhip64_7.dll'
    if (!(Test-Path $src)) { $src = Join-Path $repo 'x64\Release\amdhip64_7.dll' }
    Copy-Item $src (Join-Path $rundir 'amdhip64_7.dll') -Force
    $loadedFrom = (Get-Item $src).FullName
}

$savedPath = $env:PATH
try {
    $env:PATH = "$rocm\bin;$savedPath"
    Push-Location $rundir
    try {
        & $env:ComSpec /d /s /c "`"$localExe`" > output.txt 2>&1"
        $code = $LASTEXITCODE
    } finally { Pop-Location }
} finally { $env:PATH = $savedPath }

# 取证：本次使用的 DLL 与 SYSTEM32 中的同名 DLL 指纹对照，并**落盘**到 run_env.txt
# （只写控制台的话 output.txt 里看不到，事后无法复核）。
$used = Get-Item (Join-Path $rundir 'amdhip64_7.dll') -ErrorAction SilentlyContinue
$sys = Get-Item 'C:\Windows\System32\amdhip64_7.dll' -ErrorAction SilentlyContinue
$envLines = @(
    "isolated_test_runner run_env",
    "test                  : $TestName",
    "run dir               : $rundir",
    "command               : `"$localExe`" (cwd=$rundir, PATH=$rocm\bin;<saved>)",
    "exit code             : $code"
)
if ($used) {
    $envLines += "runtime dll file      : $($used.FullName)"
    $envLines += "runtime dll size      : $($used.Length)"
    $envLines += "runtime dll sha256    : $((Get-FileHash $used.FullName -Algorithm SHA256).Hash)"
}
if ($sys) {
    $envLines += "SYSTEM32 amdhip64_7   : $($sys.FullName)"
    $envLines += "SYSTEM32 size         : $($sys.Length)"
    $envLines += "SYSTEM32 sha256       : $((Get-FileHash $sys.FullName -Algorithm SHA256).Hash)"
    if ($used) {
        $envLines += "SYSTEM32 used         : NO (exe-directory copy wins the search order)"
    } else {
        $envLines += "SYSTEM32 used         : YES (no DLL beside the exe; this is the negative control)"
    }
}
$envLines | Set-Content -Path (Join-Path $rundir 'run_env.txt') -Encoding utf8

Write-Host "=== $TestName ==="
if ($used) {
    Write-Host ("runtime dll in test dir : {0} bytes  sha256={1}" -f $used.Length,
        (Get-FileHash $used.FullName -Algorithm SHA256).Hash.Substring(0, 16))
} else {
    Write-Host "runtime dll in test dir : (none; -SkipDll control run)"
}
if ($sys) {
    Write-Host ("SYSTEM32 amdhip64_7.dll : {0} bytes  sha256={1}" -f $sys.Length,
        (Get-FileHash $sys.FullName -Algorithm SHA256).Hash.Substring(0, 16))
}
Write-Host ("dll source              : {0}" -f $loadedFrom)
Write-Host ("exit code               : {0}" -f $code)
Write-Host ("env evidence            : {0}" -f (Join-Path $rundir 'run_env.txt'))
if (Test-Path (Join-Path $rundir 'output.txt')) {
    Get-Content (Join-Path $rundir 'output.txt') | Select-Object -Last 25
}
if ($code -ne 0) { exit $code }
