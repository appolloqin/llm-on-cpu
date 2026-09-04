@echo off
REM Start GLM server (NVFP4 config by default).
REM Modes: pure_cpu | hybrid_gpu | pure_gpu （pure_gpu: AWQ/NVFP4 可走 GPU dequant+cuBLAS）
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not defined OMP_NUM_THREADS set "OMP_NUM_THREADS=32"
if not exist "bin\llmoc_server_glm.exe" (
  echo ERROR: bin\llmoc_server_glm.exe not found — build the project first
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_glm_nvfp4.yaml"
if not exist "%CFG%" (
  echo ERROR: config not found: %CFG%
  exit /b 1
)
REM Fail early if default NVFP4 pack missing (chat used to soft-fail with opaque error)
findstr /I /C:"nvfp4.glmq" "%CFG%" >nul 2>&1
if not errorlevel 1 (
  if not exist "models\GLM-5.3-Flash.nvfp4.glmq" (
    echo ERROR: models\GLM-5.3-Flash.nvfp4.glmq missing — run download_glm.cmd first
    exit /b 1
  )
)
findstr /I /C:"awq.glmq" "%CFG%" >nul 2>&1
if not errorlevel 1 (
  if not exist "models\GLM-5.3-Flash.awq.glmq" (
    echo ERROR: models\GLM-5.3-Flash.awq.glmq missing — run download_glm.cmd --awq first
    exit /b 1
  )
)
echo Starting GLM server with %CFG%
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%
"bin\llmoc_server_glm.exe" --config "%CFG%"
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo Server exited with code %EC%
  pause
)
exit /b %EC%
