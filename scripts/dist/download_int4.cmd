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
REM 已有引擎 INT4 权重则直接提示；prepare_model 也会用 QLW1 魔数再确认并跳过重转
if exist "models\%SHORT%.int4.qlwc" (
  echo == [INT4] found models\%SHORT%.int4.qlwc — will skip re-download / re-quantize unless --force-int4
)
echo == [INT4] prepare %MODEL%  ^(BF16 HF -^> LWC -^> QLWC; auto-skip if already INT4^)
echo     Tip: pass BF16 base id ^(e.g. Qwen/Qwen3.5-4B^), not *-AWQ / *-GPTQ repos
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
