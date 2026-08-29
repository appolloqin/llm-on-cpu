@echo off
rem llm-on-cpu :: configure (Windows dev, MSVC x64 + bundled Ninja/CMake)
setlocal enabledelayedexpansion
set "BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VARS=%BT%\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

if not exist "%VARS%" (
  echo [configure.cmd] vcvars64.bat not found under "%BT%"
  exit /b 1
)

call "%VARS%" x64 || exit /b 1
set "PATH=%NINJA%;%PATH%"

"%CMAKE%" -S . -B build\msvc-x64 -G Ninja -DCMAKE_BUILD_TYPE=Release %*
exit /b %ERRORLEVEL%
