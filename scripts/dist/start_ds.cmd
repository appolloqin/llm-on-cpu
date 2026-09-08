@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist "bin\llmoc_server_ds_nvfp4.exe" (
  echo ERROR: bin\llmoc_server_ds_nvfp4.exe not found — build first
  exit /b 1
)
set "CFG=%~1"
if "%CFG%"=="" set "CFG=configs\engine_ds_nvfp4.yaml"
"bin\llmoc_server_ds_nvfp4.exe" --config "%CFG%"
