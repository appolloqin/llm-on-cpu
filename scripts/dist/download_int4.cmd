@echo off
REM Auto: download / convert / INT4 / prune as needed
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
echo == [INT4] prepare %MODEL%  ^(auto-skip done steps^)
call node tools\prepare_model.mjs --model %MODEL% --prune-hf --int4 %EXTRA%
if errorlevel 1 (
  echo ERROR: prepare_model failed. See log above.
  exit /b 1
)
echo.
echo OK. Engine weights: models\%SHORT%.int4.qlwc
echo     Tokenizer keep: models\%SHORT%-hf\
echo Next: start_int4.cmd
exit /b 0
