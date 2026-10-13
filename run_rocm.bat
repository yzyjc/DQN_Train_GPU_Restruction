@echo off
REM ============================================
REM  Run DQN trainer with ROCm GPU
REM  PATH points to D:\AMD_ROCm\bin for runtime DLLs.
REM  NOTE: x64\Release already contains the matching
REM  7.13 HIP DLL set (required - System32 has stale copies).
REM ============================================

set "ROCM_PATH=D:\AMD_ROCm"
set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"

set "HIP_PLATFORM=amd"
set "HIP_PATH=D:\AMD_ROCm"

cd /d "%~dp0x64\Release"
if not exist "DQN_Train_Trainer.exe" (
    echo ERROR: DQN_Train_Trainer.exe not found in %CD%
    pause
    exit /b 1
)
echo Starting DQN trainer (ROCm GPU)...
.\DQN_Train_Trainer.exe
