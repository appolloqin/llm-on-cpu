@echo off
REM 下载 → 转 LWC → INT4 量化 → 删除原 HF 大权重 + 中间 BF16 LWC
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
echo == [1/3] 下载 + 转 LWC + 删除原 safetensors  %MODEL%
call node tools\prepare_model.mjs --model %MODEL% --prune-hf %EXTRA%
if errorlevel 1 exit /b 1

echo == [2/3] INT4 量化
call node tools\quantize_int4.mjs --src models\%SHORT%.lwc --out models\%SHORT%.int4.qlwc --method gptq
if errorlevel 1 exit /b 1

echo == [3/3] 删除中间 BF16 LWC
if exist "models\%SHORT%.lwc" (
  del /f /q "models\%SHORT%.lwc"
  echo   removed models\%SHORT%.lwc
)

echo.
echo OK. 引擎权重: models\%SHORT%.int4.qlwc
echo     旁路保留: models\%SHORT%-hf\ （仅 config/tokenizer，大权重已删）
echo 若模型不是 Qwen3.5-4B，请改 configs\engine_int4.yaml。
echo 然后运行: start_int4.cmd
exit /b 0
