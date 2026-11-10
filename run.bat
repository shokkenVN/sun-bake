@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"
title sun-bake

echo ===================================================
echo                  sun-bake launcher
echo ===================================================
echo.

:: 1. Detect Python executable
set "PYTHON_EXE="
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PYTHON_EXE=py -3"
    )
)

if "%PYTHON_EXE%"=="" goto :err_no_python

:: 2. Check Python version (>= 3.9) and 64-bit architecture
%PYTHON_EXE% -c "import sys; is_compat = (sys.version_info >= (3, 9) and sys.maxsize > 2**32); sys.exit(0 if is_compat else 1)" >nul 2>&1
if %ERRORLEVEL% neq 0 goto :err_incompat_python

echo [OK] Compatible Python detected:
%PYTHON_EXE% --version
echo.

:: 3. Check dependencies
echo Checking dependencies...
%PYTHON_EXE% -c "import numpy, cv2, PIL, psd_tools, pytoshop, six, packbits, PySide6, qdarktheme" >nul 2>&1
if %ERRORLEVEL% neq 0 goto :install_deps

echo [OK] All dependencies are already installed.
goto :check_dll

:install_deps
echo [INFO] Missing dependencies detected. Installing from requirements.txt...
echo.
if not exist "requirements.txt" goto :err_no_reqs

%PYTHON_EXE% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 goto :err_pip_failed

echo.
echo [OK] Dependencies installed successfully.

:check_dll
echo.
:: 4. Check for DLL
if not exist "Live2DCubismCore.dll" (
    echo [WARNING] Live2DCubismCore.dll was not found next to main.py.
    echo Please place Live2DCubismCore.dll in this directory before loading models.
    echo.
)

:: 5. Launch sun-bake
echo Starting sun-bake...
echo.
%PYTHON_EXE% main.py %*
if %ERRORLEVEL% neq 0 (
    echo.
    echo [INFO] sun-bake exited with code %ERRORLEVEL%.
    pause
)
exit /b 0

:err_no_python
echo [ERROR] Python is not installed or not found in PATH.
echo Please install 64-bit Python 3.9 or higher from https://www.python.org/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:err_incompat_python
echo [ERROR] Incompatible Python environment detected.
echo sun-bake requires 64-bit Python 3.9 or higher.
echo Current detected Python:
%PYTHON_EXE% --version
echo.
pause
exit /b 1

:err_no_reqs
echo [ERROR] requirements.txt not found in %~dp0
echo.
pause
exit /b 1

:err_pip_failed
echo.
echo [ERROR] Failed to install dependencies.
echo Please check your network connection or pip configuration.
echo.
pause
exit /b 1
