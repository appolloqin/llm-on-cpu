@echo off
REM Auto: download / import NVFP4 (default) or AWQ for GLM-5.3-Flash
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

where node >nul 2>&1
if errorlevel 1 (
  echo ERROR: Node.js ^>= 18 required in PATH.
  exit /b 1
)

set "QUANT=nvfp4"
set "EXTRA="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--quant" (
  set "QUANT=%~2"
  shift
  shift
  goto parse
)
if /I "%~1"=="--nvfp4" (
  set "QUANT=nvfp4"
  shift
  goto parse
)
if /I "%~1"=="--awq" (
  set "QUANT=awq"
  shift
  goto parse
)
set "EXTRA=%EXTRA% %~1"
shift
goto parse

:run
echo == [GLM] prepare quant=%QUANT%  ^(default: LibertAIDAI/GLM-5.3-Flash-NVFP4^)
call node tools\glm\prepare_glm.mjs --quant %QUANT% --prune-hf %EXTRA%
if errorlevel 1 (
  echo ERROR: prepare_glm failed. See log above.
  exit /b 1
)
echo.
if /I "%QUANT%"=="awq" (
  echo Next: start_glm.cmd configs\engine_glm_int4.yaml
) else (
  echo Next: start_glm.cmd
)
exit /b 0
