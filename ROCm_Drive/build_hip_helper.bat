@echo off
REM Build HIP Helper DLL with hipcc
REM This DLL must be compiled with hipcc so HIP runtime works

echo ========================================
echo Building HIP Helper DLL
echo ========================================

set "ROCM_PATH=D:\AMD_ROCm"
set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PROJECT_DIR=E:\source\repos\DQN_Train_GPU_Restruction\ROCm_Drive"
set "TARGET_DIR=E:\source\repos\DQN_Train_GPU_Restruction\x64\Release"

set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"

REM Compile with hipcc (shared library)
echo Compiling with hipcc...
"%HIPCC%" --offload-arch=gfx1151 -shared -O2 ^
    -o "%TARGET_DIR%\hip_helper.dll" ^
    "%PROJECT_DIR%\src\hip_helper_dll.cpp" ^
    -I"%PROJECT_DIR%\src" ^
    -I"%ROCM_PATH%\include" ^
    -D__HIP_PLATFORM_AMD__ ^
    -lhipblas -lamdhip64 ^
    -Wl,--out-implib,"%TARGET_DIR%\hip_helper.lib"

if %errorlevel% neq 0 (
    echo ERROR: hipcc compilation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed!
echo DLL: %TARGET_DIR%\hip_helper.dll
echo LIB: %TARGET_DIR%\hip_helper.lib
echo ========================================
pause
