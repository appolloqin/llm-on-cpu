@echo off
REM Start DeepSeek stub server (DS-STUB-v0). Default: configs\engine_ds_nvfp4.yaml
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not exist "bin\llmoc_server_ds.exe" (
  echo ERROR: bin\llmoc_server_ds.exe not found — build first
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_ds_nvfp4.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
if not exist "models\fake_ds.dskq" (
  echo WARN: models\fake_ds.dskq missing — run download_ds.cmd first
)
echo Starting DS stub with %CFG%
echo   port default 15086 — stub only, not paper CSA/HCA
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=8"
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server_ds.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo Server exited with code %EC%
  pause
)
exit /b %EC%
