@echo off
REM Write tiny Kimi stub weights (KIM1) — NOT a real model download
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
if not exist "bin\make_fake_kimi.exe" (
  echo ERROR: bin\make_fake_kimi.exe not found — build the project first
  exit /b 1
)
if not exist "models" mkdir models
set "OUT=models\fake_kimi.kimiq"
if not "%~1"=="" set "OUT=%~1"
"bin\make_fake_kimi.exe" "%OUT%"
if errorlevel 1 exit /b 1
echo OK. Stub weights: %OUT%
echo Next: start_kimi.cmd
echo     ^(Kimi-STUB-v0；单卡 pure_gpu 会自动降级 layer_stream^)
exit /b 0
