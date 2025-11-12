#!/bin/bash

# Setup script for Calabi-Yau project

echo "======================================"
echo "Calabi-Yau Project Setup"
echo "======================================"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data models results

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To get started:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run quick experiment: python run_experiment.py"
echo "3. Or open Jupyter: jupyter notebook notebooks/calabi_yau_experiment.ipynb"
echo ""
