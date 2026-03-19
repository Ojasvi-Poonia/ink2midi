#!/bin/bash
# Setup script for OMR project on Ubuntu with NVIDIA RTX 3070 Ti (CUDA 12.2, 8GB VRAM)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "=========================================="
echo " OMR Project Setup (RTX 3070 Ti / CUDA 12.2)"
echo "=========================================="

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

echo ""
echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Create virtual environment if not already in one
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated."
else
    echo "Using existing virtual environment: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
fi

# Upgrade pip
pip install --upgrade pip

# Step 1: Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.2 driver)
echo ""
echo "Step 1: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after install!'
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Step 2: Install PyTorch Geometric
echo ""
echo "Step 2: Installing PyTorch Geometric..."
pip install torch-geometric

# Step 3: Install project dependencies
echo ""
echo "Step 3: Installing project dependencies..."
pip install -r requirements.txt

# Step 4: Install project in editable mode
echo ""
echo "Step 4: Installing project in editable mode..."
pip install -e .

# Step 5: Create data directories
echo ""
echo "Step 5: Creating directory structure..."
mkdir -p data/raw data/processed data/splits
mkdir -p checkpoints/detection checkpoints/gnn
mkdir -p outputs/midi outputs/viz

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python3 -c "
import torch
import torch_geometric
from ultralytics import YOLO
import music21
import cv2
import numpy as np

print('All imports successful!')
print(f'  PyTorch:          {torch.__version__}')
print(f'  CUDA available:   {torch.cuda.is_available()}')
print(f'  CUDA version:     {torch.version.cuda}')
print(f'  cuDNN version:    {torch.backends.cudnn.version()}')
print(f'  GPU:              {torch.cuda.get_device_name(0)}')
print(f'  PyG:              {torch_geometric.__version__}')
print(f'  OpenCV:           {cv2.__version__}')
print(f'  music21:          {music21.VERSION_STR}')

# Quick CUDA test
x = torch.randn(100, 100, device='cuda')
y = x @ x.T
print(f'  CUDA compute OK:  {y.shape}')

# Test FP16 (mixed precision)
with torch.amp.autocast('cuda'):
    z = torch.randn(100, 100, device='cuda')
    w = z @ z.T
print(f'  FP16 AMP OK:      {w.dtype}')
"

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download datasets:    make download"
echo "  2. Prepare datasets:     make prepare"
echo "  3. Train detector:       make train-detector"
echo "  4. Train GNN:            make train-gnn"
echo "  5. Run inference:        python scripts/run_inference.py --image <path>"
echo ""
echo "For faster training (uses cuDNN benchmark):"
echo "  make train-all-fast"
echo ""
