@echo off
REM 启动 INT4 服务，端口 15085
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "bin\llmoc_server_int4.exe" (
  echo ERROR: 找不到 bin\llmoc_server_int4.exe
  exit /b 1
)
if not exist "models\Qwen3.5-4B.int4.qlwc" (
  echo WARN: 未找到 models\Qwen3.5-4B.int4.qlwc — 请先运行 download_int4.cmd
)
echo Starting INT4 server on http://127.0.0.1:15085/
"bin\llmoc_server_int4.exe" --config configs\engine_int4.yaml
exit /b %ERRORLEVEL%
