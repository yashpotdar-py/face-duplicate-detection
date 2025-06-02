#!/bin/bash

# GPU-Accelerated Face Duplicate Detection System
# Installation and Setup Script

set -e

echo "🚀 Setting up GPU-Accelerated Face Duplicate Detection System"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  No NVIDIA GPU detected. CPU mode will be used."
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv face_detection_env
source face_detection_env/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    # Install CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # Install CPU version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "📦 Installing other requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p results
mkdir -p data/Thumbnails
mkdir -p data/Videos

# Download models (optional, models will be downloaded on first use)
echo "🔄 Models will be downloaded automatically on first use"

# Make main.py executable
chmod +x main.py

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "🎯 Quick Start:"
echo "  1. Activate environment: source face_detection_env/bin/activate"
echo "  2. Show system info:     python main.py info"
echo "  3. Process images:       python main.py process-images --input-dir data/Thumbnails"
echo "  4. Process videos:       python main.py process-videos --input-dir data/Videos"
echo "  5. Process all:          python main.py process-all --data-dir data"
echo ""
echo "📚 For help:               python main.py --help"
echo ""
