@echo off
REM Auto: download / convert / prune as needed (BF16)
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

where node >nul 2>&1
if errorlevel 1 (
  echo ERROR: Node.js ^>= 18 required in PATH.
  exit /b 1
)

set "MODEL=Qwen/Qwen3.5-4B"
set "EXTRA="

:parse
if "%~1"=="" goto run
if /I "%~1"=="--model" (
  set "MODEL=%~2"
  shift
  shift
  goto parse
)
set "EXTRA=%EXTRA% %~1"
shift
goto parse

:run
for %%I in ("%MODEL%") do set "SHORT=%%~nxI"
echo == [BF16] prepare %MODEL%  ^(auto-skip done steps^)
call node tools\prepare_model.mjs --model %MODEL% --prune-hf %EXTRA%
if errorlevel 1 (
  echo ERROR: prepare_model failed. See log above.
  exit /b 1
)
echo.
echo OK. Engine weights: models\%SHORT%.lwc
echo     Tokenizer keep: models\%SHORT%-hf\
echo Next: start_bf16.cmd
exit /b 0
