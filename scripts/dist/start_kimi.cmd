@echo off
REM Start Kimi stub server (Kimi-STUB-v0). Default: configs\engine_kimi_hybrid.yaml
REM 单卡 pure_gpu 装不下 → 自动降级 layer_stream
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not exist "bin\llmoc_server_kimi.exe" (
  echo ERROR: bin\llmoc_server_kimi.exe not found — build first
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_kimi_hybrid.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
if not exist "models\fake_kimi.kimiq" (
  echo WARN: models\fake_kimi.kimiq missing — run download_kimi.cmd first
)
echo Starting Kimi stub with %CFG%
echo   port default 15087 — stub only; prefer hybrid_gpu
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=8"
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server_kimi.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo Server exited with code %EC%
  pause
)
exit /b %EC%
