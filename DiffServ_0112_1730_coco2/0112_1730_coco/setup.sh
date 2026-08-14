#!/bin/bash

set -e

echo "Creating environment..."

conda env create -f environment.yml

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate DIFS

echo "Installing pip packages..."

echo "Installing PyTorch..."
pip install -r requirements-torch.txt

echo "Installing remaining packages..."
pip install -r requirements.txt

echo "Verifying installation..."

python - <<EOF
import torch
import torchvision
import torchaudio
import diffusers

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Diffusers:", diffusers.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

echo "Done."