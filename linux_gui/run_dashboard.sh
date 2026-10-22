#!/bin/bash
# CoreAI3D Dashboard Startup Script for Linux
# This script launches the CoreAI3D GUI Dashboard (dashboard.py)

set -e

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
if ! python3 -c "import PyQt6, keyring, cryptography" &> /dev/null; then
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

# Source the local .env (COREAI3D_API_URL / COREAI3D_WS_URL) so the Python
# process inherits the backend endpoints. The GUI runs on bare metal and must
# reach the host-published ports (127.0.0.1:1906 HTTP, 127.0.0.1:1907 WS),
# not the container's internal 8080/8081.
if [ -f ".env" ]; then
    set -a
    . ./.env
    set +a
    echo "Loaded backend endpoints from .env:"
    echo "  COREAI3D_API_URL=${COREAI3D_API_URL}"
    echo "  COREAI3D_WS_URL=${COREAI3D_WS_URL}"
else
    echo "Warning: .env not found - using built-in defaults (127.0.0.1:1906 / :1907)"
fi

# Start the dashboard
python3 dashboard.py

echo
echo "Dashboard stopped."
