@echo off
REM ROCm_Drive v2 Build Script

echo ========================================
echo ROCm_Drive v2 Build Script
echo ========================================

set "ROCM_PATH=D:\AMD_ROCm"
set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "LDDLINK=%ROCM_PATH%\lib\llvm\bin\lld-link.exe"
set "PROJECT_DIR=E:\source\repos\DQN_Train_GPU_Restruction\ROCm_Drive"
set "BUILD_DIR=%PROJECT_DIR%\build"
set "TARGET_DIR=E:\source\repos\DQN_Train_GPU_Restruction\x64\Release"
set "INCLUDE_DIR=%PROJECT_DIR%\include"

set "MSVC_LIB=D:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\lib\x64"
set "UCRT_LIB=E:\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64"
set "WINSDK_UM=E:\Windows Kits\10\Lib\10.0.22621.0\um\x64"

set "PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\llvm\bin;%PATH%"

REM Clean
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%" 2>nul
mkdir "%BUILD_DIR%"

REM ===== Step 1: Compile HIP kernels =====
echo [1/3] Compiling HIP kernels (hipcc, gfx1151)...
"%HIPCC%" --offload-arch=gfx1151 -O2 -c ^
    "%PROJECT_DIR%\src\gpu_kernels.hip" ^
    -o "%BUILD_DIR%\gpu_kernels.obj" ^
    -I"%INCLUDE_DIR%" ^
    -I"%ROCM_PATH%\include" ^
    -D__HIP_PLATFORM_AMD__ ^
    -DROCMDRIVE_EXPORTS ^
    -D_WINDOWS ^
    -D_USRDLL ^
    -D_CRT_SECURE_NO_WARNINGS

if %errorlevel% neq 0 (
    echo ERROR: HIP kernel compilation failed!
    pause
    exit /b 1
)
echo   OK: gpu_kernels.obj

REM ===== Step 2: Compile host C++ =====
echo [2/3] Compiling host C++...
"%HIPCC%" --offload-arch=gfx1151 -O2 -c ^
    "%PROJECT_DIR%\src\rocm_accelerator_v2.cpp" ^
    -o "%BUILD_DIR%\rocm_accelerator_v2.obj" ^
    -I"%INCLUDE_DIR%" ^
    -I"%ROCM_PATH%\include" ^
    -D__HIP_PLATFORM_AMD__ ^
    -DROCMDRIVE_EXPORTS ^
    -D_WINDOWS ^
    -D_USRDLL ^
    -D_CRT_SECURE_NO_WARNINGS

if %errorlevel% neq 0 (
    echo ERROR: rocm_accelerator_v2 compilation failed!
    pause
    exit /b 1
)
echo   OK: rocm_accelerator_v2.obj

"%HIPCC%" --offload-arch=gfx1151 -O2 -c ^
    "%PROJECT_DIR%\dllmain_v2.cpp" ^
    -o "%BUILD_DIR%\dllmain_v2.obj" ^
    -I"%INCLUDE_DIR%" ^
    -I"%ROCM_PATH%\include" ^
    -D__HIP_PLATFORM_AMD__ ^
    -DROCMDRIVE_EXPORTS ^
    -D_WINDOWS ^
    -D_USRDLL ^
    -D_CRT_SECURE_NO_WARNINGS

if %errorlevel% neq 0 (
    echo ERROR: dllmain_v2 compilation failed!
    pause
    exit /b 1
)
echo   OK: dllmain_v2.obj

REM ===== Step 3: Link =====
echo [3/3] Linking...
"%LDDLINK%" /DLL /OUT:"%BUILD_DIR%\rocm_drive.dll" ^
    /NOLOGO ^
    /DEF:"%PROJECT_DIR%\rocm_drive.def" ^
    "%BUILD_DIR%\dllmain_v2.obj" ^
    "%BUILD_DIR%\rocm_accelerator_v2.obj" ^
    "%BUILD_DIR%\gpu_kernels.obj" ^
    "%ROCM_PATH%\lib\amdhip64.lib" ^
    "%ROCM_PATH%\lib\hipblas.lib" ^
    "%UCRT_LIB%\ucrt.lib" ^
    "%MSVC_LIB%\vcruntime.lib" ^
    "%MSVC_LIB%\msvcrt.lib" ^
    "%WINSDK_UM%\kernel32.lib" ^
    "%WINSDK_UM%\advapi32.lib" ^
    /LIBPATH:"%ROCM_PATH%\lib" ^
    /LIBPATH:"%MSVC_LIB%" ^
    /LIBPATH:"%UCRT_LIB%" ^
    /LIBPATH:"%WINSDK_UM%"

if %errorlevel% neq 0 (
    echo ERROR: Linking failed!
    pause
    exit /b 1
)
echo   OK: rocm_drive.dll

REM ===== Copy DLL =====
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
copy "%BUILD_DIR%\rocm_drive.dll" "%TARGET_DIR%\"
echo.
echo ========================================
echo Build completed!
echo DLL: %TARGET_DIR%\rocm_drive.dll
echo ========================================
pause
