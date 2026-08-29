@echo off
REM 下载 → 转 LWC → 删除原 HF 大权重（保留 config/tokenizer）
setlocal EnableExtensions
cd /d "%~dp0"

where node >nul 2>&1
if errorlevel 1 (
  echo ERROR: 需要 Node.js ^>= 18，并加入 PATH。
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
echo == [1/2] 下载 + 转 LWC + 删除原 safetensors  %MODEL%
call node tools\prepare_model.mjs --model %MODEL% --prune-hf %EXTRA%
if errorlevel 1 exit /b 1
echo.
echo OK. 引擎权重: models\%SHORT%.lwc
echo     旁路保留: models\%SHORT%-hf\ （仅 config/tokenizer，大权重已删）
echo 若模型不是 Qwen3.5-4B，请改 configs\engine.yaml。
echo 然后运行: start_bf16.cmd
exit /b 0
