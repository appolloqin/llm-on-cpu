@echo off
REM Start INT4 server. Default: configs\engine_int4.yaml
REM Modes in yaml: pure_cpu | hybrid_gpu | pure_gpu | auto | layer_stream
REM Optional: start_int4.cmd configs\engine_int4_mtp.yaml
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

if not exist "bin\llmoc_server_int4.exe" (
  echo ERROR: bin\llmoc_server_int4.exe not found
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_int4.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
if not exist "models\Qwen3.5-4B.int4.qlwc" (
  echo WARN: models\Qwen3.5-4B.int4.qlwc missing — run download_int4.cmd first
)
echo Starting INT4 server with %CFG%
echo   http://127.0.0.1:15085/  ^(edit server.port in yaml if needed^)
echo   hybrid/pure_gpu: set tiers.gpu_vram_gb；layer_stream: see layer_stream: block
REM HX/INT4: 32 逻辑线程会掉到 ~5 tok/s；未设置时由 server 自动 cap 到 8
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=8"
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server_int4.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Server exited with code %EC%. See console above and logs\ if present.
  pause
)
exit /b %EC%
