@echo off
REM 启动 BF16（无量化）服务，端口 15085
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "bin\llmoc_server.exe" (
  echo ERROR: 找不到 bin\llmoc_server.exe
  exit /b 1
)
if not exist "models\Qwen3.5-4B.lwc" (
  echo WARN: 未找到 models\Qwen3.5-4B.lwc — 请先运行 download_bf16.cmd
)
echo Starting BF16 server on http://127.0.0.1:15085/
"bin\llmoc_server.exe" --config configs\engine.yaml
exit /b %ERRORLEVEL%
