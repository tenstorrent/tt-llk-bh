#!/bin/bash
set -x  # Print each command as it gets executed
set -e  # Exit immediately if a command exits with a non-zero status

set -euo pipefail

# Update packages and install gawk (if necessary)
sudo apt update
sudo apt install -y gawk

# **************** DOWNLOAD & INSTALL SFPI ****************************
# Download and extract SFPI release
wget https://github.com/tenstorrent/sfpi/releases/download/v6.0.0/sfpi-release.tgz -O sfpi-release.tgz
if [ ! -f "sfpi-release.tgz" ]; then
    echo "SFPI release not found!"
    exit 1
fi

tar -xzvf sfpi-release.tgz
rm -f sfpi-release.tgz

# **************** DOWNLOAD & INSTALL TT-LENS ****************************
pip install git+https://github.com/tenstorrent/tt-debuda.git@d4ce04c3d4e68cccdf0f53b0b5748680a8a573ed

# **************** DOWNLOAD & INSTALL TT_SMI ****************************
git clone https://github.com/tenstorrent/tt-smi
cd tt-smi
python3 -m venv .venv
source .venv/bin/activate

# Ensure pip is installed and upgraded
pip install --upgrade pip

# Install necessary packages
pip install .
pip install pytest pytest-cov

# Detect architecture for chip
tt-smi -ls > ../arch.dump
result=$(python3 helpers/find_arch.py ["Wormhole" "Blackhole" "Grayskull"] arch.dump)
export CHIP_ARCH="$result"
cd ..

# Install torch and related packages (with fallback to CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# **************** SETUP PYTHON VENV **********************************
# Ensure python3.10-venv is installed, fallback to python3.8-venv
if ! dpkg -l | grep -q python3.10-venv; then
    echo "python3.10-venv not found, attempting to install python3.8-venv..."
    sudo apt install -y python3.8-venv || { echo "Failed to install python3.8-venv."; exit 1; }
else
    sudo apt install -y python3.10-venv
fi

# Set up Python virtual environment if not already set
# if [ -z "${PYTHON_ENV_DIR:-}" ]; then
#     PYTHON_ENV_DIR="$(pwd)/.venv"
# fi

# echo "Creating virtual environment in: $PYTHON_ENV_DIR"
# python3 -m venv "$PYTHON_ENV_DIR"
# source "$PYTHON_ENV_DIR/bin/activate"

# Install pip and upgrade if necessary
python3 -m ensurepip
pip install --upgrade pip

# Install needed packages
pip install -U pytest pytest-cov
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Script completed successfully!"
