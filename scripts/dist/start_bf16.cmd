@echo off
REM Start BF16 server. Default: configs\engine.yaml
REM Modes: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream
REM Optional: start_bf16.cmd path\to\engine.yaml
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

if not exist "bin\llmoc_server.exe" (
  echo ERROR: bin\llmoc_server.exe not found
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
if not exist "models\Qwen3.5-4B.lwc" (
  echo WARN: models\Qwen3.5-4B.lwc missing — run download_bf16.cmd first
)
echo Starting BF16 server with %CFG%
echo   http://127.0.0.1:15085/
echo   hybrid/pure_gpu: uncomment tiers.gpu_vram_gb；layer_stream: set mode + layer_stream:
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=32"
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Server exited with code %EC%. See console above and logs\ if present.
  pause
)
exit /b %EC%
