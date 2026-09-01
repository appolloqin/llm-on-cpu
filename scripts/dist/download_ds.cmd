@echo off
REM Write tiny DeepSeek stub weights (DSS1) — NOT a real model download
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not exist "bin\make_fake_ds.exe" (
  echo ERROR: bin\make_fake_ds.exe not found — build the project first
  exit /b 1
)
if not exist "models" mkdir models
set "OUT=models\fake_ds.dskq"
if not "%~1"=="" set "OUT=%~1"
"bin\make_fake_ds.exe" "%OUT%"
if errorlevel 1 exit /b 1
echo OK. Stub weights: %OUT%
echo Next: start_ds.cmd
echo     ^(DS-STUB-v0 — for smoke only; edit configs\engine_ds_nvfp4.yaml^)
exit /b 0
