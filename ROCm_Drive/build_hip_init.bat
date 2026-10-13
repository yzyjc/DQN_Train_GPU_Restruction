@echo off
REM Build HIP initialization object file with hipcc
REM This object is linked into the training program

echo ========================================
echo Building HIP Init Object
echo ========================================

set "ROCM_PATH=D:\AMD_ROCm"
set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PROJECT_DIR=E:\source\repos\DQN_Train_GPU_Restruction"
set "TARGET_DIR=%PROJECT_DIR%\x64\Release"

set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"

REM Compile hip_init_test.cpp to object file
echo Compiling hip_init_test.cpp with hipcc...
"%HIPCC%" --offload-arch=gfx1151 -c -O2 -D_CRT_SECURE_NO_WARNINGS -D__HIP_PLATFORM_AMD__ ^
    -I"%ROCM_PATH%\include" ^
    "%PROJECT_DIR%\DQN_Train_Trainer\hip_init_test.cpp" ^
    -o "%TARGET_DIR%\hip_init_test.obj"

if %errorlevel% neq 0 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed!
echo Object: %TARGET_DIR%\hip_init_test.obj
echo.
echo Link this object into your training program:
echo   link ... hip_init_test.obj -lamdhip64
echo ========================================
pause
