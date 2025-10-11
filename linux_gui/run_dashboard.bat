@echo off
REM CoreAI3D Dashboard Startup Script for Windows
REM This script launches the CoreAI3D Windows GUI Dashboard

echo CoreAI3D Dashboard Startup
echo =========================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "dashboard.py" (
    echo Error: dashboard.py not found in current directory
    echo Please run this script from the windows_gui directory
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import PyQt6, psutil, aiohttp, websockets" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo Some dependencies are missing.
    echo Installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install requirements
        echo Please install manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo Starting CoreAI3D Dashboard...
echo Press Ctrl+C to exit
echo.

REM Start the dashboard
python run_dashboard.py

echo.
echo Dashboard stopped.
pause