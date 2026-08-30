@echo off
REM Start BF16 (unquantized) server on port 15085
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

if not exist "bin\llmoc_server.exe" (
  echo ERROR: bin\llmoc_server.exe not found
  exit /b 1
)
if not exist "models\Qwen3.5-4B.lwc" (
  echo WARN: models\Qwen3.5-4B.lwc missing — run download_bf16.cmd first
)
echo Starting BF16 server on http://127.0.0.1:15085/
"bin\llmoc_server.exe" --config configs\engine.yaml
exit /b %ERRORLEVEL%
