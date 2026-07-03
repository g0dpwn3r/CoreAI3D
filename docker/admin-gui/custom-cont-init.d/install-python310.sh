#!/bin/bash
# Install Python 3.10 with pip and venv from deadsnakes PPA

echo "[init] Installing Python 3.10 from deadsnakes PPA..."

# Install prerequisites
apt-get update
apt-get install -y software-properties-common

# Add deadsnakes PPA
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

# Install Python 3.10 with required packages
apt-get install -y python3.10 python3.10-venv python3.10-dev

# Set up alternative so python3.10 can be easily accessed
update-alternatives --install /usr/bin/python3.10 python3.10 /usr/bin/python3.10 1

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

echo "[init] Python 3.10 installation complete"
python3.10 --version
