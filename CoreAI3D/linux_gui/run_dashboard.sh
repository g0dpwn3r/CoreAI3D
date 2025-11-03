#!/bin/bash
# CoreAI3D Dashboard Startup Script for Linux
# This script launches the CoreAI3D Linux GUI Dashboard

echo "CoreAI3D Dashboard Startup"
echo "=========================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    echo "On Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "On CentOS/RHEL: sudo yum install python3 python3-pip"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "dashboard.py" ]; then
    echo "Error: dashboard.py not found in current directory"
    echo "Please run this script from the linux_gui directory"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import PyQt6, psutil, aiohttp, websockets" &> /dev/null
if [ $? -ne 0 ]; then
    echo
    echo "Some dependencies are missing."
    echo "Installing requirements..."
    if command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
    else
        pip install -r requirements.txt
    fi
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install requirements"
        echo "Please install manually: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo
echo "Starting CoreAI3D Dashboard..."
echo "Press Ctrl+C to exit"
echo

# Start the dashboard
python3 run_dashboard.py

echo
echo "Dashboard stopped."