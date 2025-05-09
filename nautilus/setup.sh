#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n poetry python=3.11 -c conda-forge -y
micromamba run -n poetry pipx ensurepath
export PATH="/root/.local/bin:$PATH"
micromamba run -n poetry pipx install poetry

eval "$(micromamba shell hook --shell=bash)"
micromamba create -n acc python=3.11 -c conda-forge -y
export PATH="/root/.local/bin:$PATH"
micromamba run -n acc apt-get update -y 
micromamba run -n acc apt-get install ffmpeg libsm6 libxext6  -y
micromamba run -n acc apt-get install gcc -y
micromamba run -n acc poetry install

echo "Changing directory to Scenic..."
cd Scenic || { echo "ERROR: Failed to change directory to Scenic." >&2; exit 1; }

echo "Current directory: $(pwd)"
echo "Installing package in editable mode from current directory..."
micromamba run -n acc pip install -e .

cd ..
echo "Current directory: $(pwd)"

micromamba run -n acc poetry run inv setup
