@echo off
setlocal enabledelayedexpansion
REM ============================================
REM  Build DQN_Train_Trainer with ROCm hipcc
REM  Full ROCm build: project compiled with hipcc, HIP statically linked
REM ============================================

set "ROCM_PATH=D:\AMD_ROCm"
set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PROJECT=E:\source\repos\DQN_Train_GPU_Restruction"
set "OUT=%PROJECT%\x64\Release"
set "CORE_INCLUDE=%PROJECT%\DQN_Core\include"
set "CORE_SRC=%PROJECT%\DQN_Core\src"

set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"

if not exist "%OUT%" mkdir "%OUT%"

echo ========================================
echo Building with ROCm hipcc
echo ========================================

REM CPU source files (excludes accelerator abstraction and orphan ipc_protocol)
set "CPU_SRCS=board pattern checkpoint memory_system ipc_server utils dqn_agent attention neural_net"

echo.
echo [1/3] Compiling DQN_Core CPU sources...
for %%f in (%CPU_SRCS%) do (
    echo   %%f.cpp
    "%HIPCC%" -O2 -std=c++20 -fms-compatibility -D_CRT_SECURE_NO_WARNINGS -D__HIP_PLATFORM_AMD__ -I"%CORE_INCLUDE%" -I"%ROCM_PATH%\include" -c "%CORE_SRC%\%%f.cpp" -o "%OUT%\%%f.obj"
    if !errorlevel! neq 0 (
        echo   FAILED: %%f.cpp
        pause
        exit /b 1
    )
)
echo   OK

echo.
echo [2/3] Compiling GPU neural network (gpu_neural_net.hip)...
"%HIPCC%" --offload-arch=gfx1151 -O2 -std=c++20 -fms-compatibility -D_CRT_SECURE_NO_WARNINGS -D__HIP_PLATFORM_AMD__ -I"%CORE_INCLUDE%" -I"%ROCM_PATH%\include" -c "%CORE_SRC%\gpu_neural_net.hip" -o "%OUT%\gpu_neural_net.obj"
if !errorlevel! neq 0 (
    echo   FAILED: gpu_neural_net.hip
    pause
    exit /b 1
)
echo   OK

echo.
echo [3/3] Compiling trainer...
echo   DQN_Train_Trainer.cpp
"%HIPCC%" -O2 -std=c++20 -fms-compatibility -D_CRT_SECURE_NO_WARNINGS -D__HIP_PLATFORM_AMD__ -I"%CORE_INCLUDE%" -I"%ROCM_PATH%\include" -c "%PROJECT%\DQN_Train_Trainer\DQN_Train_Trainer.cpp" -o "%OUT%\DQN_Train_Trainer.obj"
if !errorlevel! neq 0 (
    echo   FAILED: DQN_Train_Trainer.cpp
    pause
    exit /b 1
)
echo   OK

echo.
echo [4/4] Linking...
"%HIPCC%" -O2 -o "%OUT%\DQN_Train_Trainer.exe" "%OUT%\DQN_Train_Trainer.obj" "%OUT%\board.obj" "%OUT%\pattern.obj" "%OUT%\checkpoint.obj" "%OUT%\memory_system.obj" "%OUT%\ipc_server.obj" "%OUT%\utils.obj" "%OUT%\dqn_agent.obj" "%OUT%\attention.obj" "%OUT%\neural_net.obj" "%OUT%\gpu_neural_net.obj" -L"%ROCM_PATH%\lib" -lamdhip64 -lkernel32 -luser32 -ladvapi32 -lwinmm -lxinput

if !errorlevel! neq 0 (
    echo FAILED
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed!
echo EXE: %OUT%\DQN_Train_Trainer.exe
echo ========================================
pause
