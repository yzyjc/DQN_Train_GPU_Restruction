@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem One-click reset for DQN runtime training state.
rem Source, configuration files, executables and build/regression evidence are kept.

for %%I in ("%~dp0.") do set "DQN_REPO=%%~fI"
set "DQN_NO_PAUSE="
if /I "%~1"=="--no-pause" set "DQN_NO_PAUSE=1"

set "DQN_BUSY="
for %%P in (
    DQN_Train_Trainer_NoTerminal.exe
    DQN_Train_Trainer.exe
    Watchdog_NoTerminal.exe
    Watchdog.exe
    DQN_Main_TUI_No_Terminal.exe
    DQN_Main_TUI.exe
    Log_Vindicator.exe
) do (
    tasklist /FI "IMAGENAME eq %%P" /NH 2>nul | find /I "%%P" >nul
    if not errorlevel 1 set "DQN_BUSY=1"
)

if defined DQN_BUSY (
    echo [ABORTED] A Trainer, Watchdog, TUI or Vindicator process is still running.
    echo Close it normally, then run this script again.
    echo No training data was removed.
    call :pause_if_needed
    exit /b 2
)

echo [RESET] Repository: "%DQN_REPO%"

rem Current runtime and known legacy/staging runtime locations.  Each target is
rem anchored below this repository; the cleanup routine never deletes a runtime
rem directory itself or any executable/configuration file in it.
call :clean_runtime "%DQN_REPO%"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\x64\Release"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\x64\StagedPipeFix"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\DQN_Train_Trainer\x64\Release"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\DQN_Train_Trainer_backup\x64\Release"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\DQN_Train_Trainer_NoTerminal\x64\Release"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\Watchdog\x64\Release"
if errorlevel 1 goto cleanup_failed
call :clean_runtime "%DQN_REPO%\Watchdog_backup\x64\Release"
if errorlevel 1 goto cleanup_failed

rem These directories are explicitly training-data backups, not source backups.
for /D %%D in ("%DQN_REPO%\backup_training_*") do (
    if exist "%%~fD\" (
        echo [DELETE] "%%~fD"
        rmdir /S /Q "%%~fD"
        if exist "%%~fD\" goto cleanup_failed
    )
)

echo.
echo [DONE] All runtime training data was removed.
echo        Programs, source, params.txt, dqn_config.txt and train_config.txt were kept.
call :pause_if_needed
exit /b 0

:cleanup_failed
echo.
echo [FAILED] Cleanup stopped because a runtime path could not be cleaned safely.
call :pause_if_needed
exit /b 3

:clean_runtime
set "DQN_RUNTIME=%~f1"
if not defined DQN_RUNTIME exit /b 3
rem DQN_RUNTIME is never user input: every call above is either DQN_REPO itself
rem or a fixed child path constructed from DQN_REPO.  Do not add arguments to
rem this subroutine without preserving that invariant.
if not exist "%DQN_RUNTIME%\" exit /b 0

for %%D in (logs checkpoints checkpoints_temp) do (
    if exist "%DQN_RUNTIME%\%%D\" (
        echo [DELETE] "%DQN_RUNTIME%\%%D"
        rmdir /S /Q "%DQN_RUNTIME%\%%D"
        if exist "%DQN_RUNTIME%\%%D\" exit /b 4
    )
)

for %%F in (
    current_snapshot.bin
    current_model_package.bin
    population_progress.bin
    population_progress.bin.tmp
    memory_cache.dat
    memory_kb.dat
    memory_cache_b.dat
    memory_cache_w.dat
    memory_kb_b.dat
    memory_kb_w.dat
    latest_eval.txt
    latest_eval.txt.tmp
    rollback_to.txt
    recovery_override.txt
    recovery_override.txt.tmp
    intentional_shutdown.flag
    intentional_shutdown.tmp
    training_fault.flag
    training_fault.tmp
    watchdog_shutdown.request
    watchdog_shutdown.request.tmp
    watchdog_shutdown.ack
    watchdog_shutdown.ack.tmp
    gpu_training.log
    batch_crash.txt
    wd_cmd.txt
    wd_cmd.tmp
    wd_snapshot_request.txt
    wd_snapshot_request.tmp
    wd_snapshot_result.txt
    wd_snapshot_result.tmp
    wd_trainer_command.txt
    wd_trainer_command.tmp
    wd_trainer_result.txt
    wd_trainer_result.tmp
) do (
    if exist "%DQN_RUNTIME%\%%F" (
        echo [DELETE] "%DQN_RUNTIME%\%%F"
        del /F /Q "%DQN_RUNTIME%\%%F"
        if exist "%DQN_RUNTIME%\%%F" exit /b 4
    )
)

rem Skill metrics are training/evaluation history and are versioned by filename.
if exist "%DQN_RUNTIME%\metric_skill_v*.bin" (
    echo [DELETE] "%DQN_RUNTIME%\metric_skill_v*.bin"
    del /F /Q "%DQN_RUNTIME%\metric_skill_v*.bin"
    if exist "%DQN_RUNTIME%\metric_skill_v*.bin" exit /b 4
)

exit /b 0

:pause_if_needed
if not defined DQN_NO_PAUSE pause
exit /b 0
