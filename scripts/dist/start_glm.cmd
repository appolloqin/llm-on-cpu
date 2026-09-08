@echo off
REM Start GLM MoE NVFP4 server (siblings: llmoc_server_glm_bf16 / _int4).
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=32"
if not exist "bin\llmoc_server_glm_nvfp4.exe" (
  echo ERROR: bin\llmoc_server_glm_nvfp4.exe not found — build the project first
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_glm_nvfp4.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
if not exist "models\GLM-5.3-Flash.nvfp4.glmq" (
  echo ERROR: models\GLM-5.3-Flash.nvfp4.glmq missing — run download_glm.cmd first
  exit /b 1
)
echo Starting GLM NVFP4 server with %CFG%
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server_glm_nvfp4.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo Server exited with code %EC%
  pause
)
exit /b %EC%
