@echo off
rem llm-on-cpu :: incremental build
setlocal
set "BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "VARS=%BT%\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=%BT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

call "%VARS%" x64 >nul || exit /b 1
set "PATH=%NINJA%;%PATH%"
"%CMAKE%" --build build\msvc-x64 %*
exit /b %ERRORLEVEL%
