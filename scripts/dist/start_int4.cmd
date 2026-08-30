@echo off
REM Start INT4 server on port 15085
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

if not exist "bin\llmoc_server_int4.exe" (
  echo ERROR: bin\llmoc_server_int4.exe not found
  exit /b 1
)
if not exist "models\Qwen3.5-4B.int4.qlwc" (
  echo WARN: models\Qwen3.5-4B.int4.qlwc missing — run download_int4.cmd first
)
echo Starting INT4 server on http://127.0.0.1:15085/
echo Logs: logs\llmoc-YYYY-MM-DD.log  (LLMOC_LOG_DIR / LLMOC_LOG / LLMOC_PROFILE)
"bin\llmoc_server_int4.exe" --config configs\engine_int4.yaml
set "EC=%ERRORLEVEL%"
if not "%EC%"=="0" (
  echo.
  echo Server exited with code %EC%. See console above and logs\ if present.
  pause
)
exit /b %EC%
