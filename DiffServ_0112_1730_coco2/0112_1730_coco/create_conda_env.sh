#!/bin/bash
# Recreate the shared DIFS_Gen conda environment on a fresh machine.
# Usage:
#   ./create_conda_env.sh          # human-readable spec (environment.yml)
#   ./create_conda_env.sh --lock   # fully pinned spec (environment_snapshot/environment.lock.yml)
set -e

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

ENV_NAME="DIFS_Gen"
SPEC="environment.yml"
LOCK="environment_snapshot/environment.lock.yml"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

if [ "$1" = "--lock" ]; then
  if [ ! -f "$LOCK" ]; then
    echo "error: $LOCK not found" >&2
    exit 1
  fi
  echo "Creating/updating '$ENV_NAME' from fully pinned lock ($LOCK)..."
  conda env create -n "$ENV_NAME" -f "$LOCK" 2>/dev/null || conda env update -n "$ENV_NAME" -f "$LOCK"
  echo "Installing pip layer from $TORCH_INDEX (torch==2.5.1+cu124 requires the cu124 index)..."
  "$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/pip" install --index-url "$TORCH_INDEX" -r requirements-lock.txt
else
  echo "Creating/updating '$ENV_NAME' from $SPEC..."
  conda env create -n "$ENV_NAME" -f "$SPEC" 2>/dev/null || conda env update -n "$ENV_NAME" -f "$SPEC"
fi

echo "Verifying '$ENV_NAME'..."
"$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin/python" - <<'EOF'
import torch, torchvision, torchaudio, diffusers, grpc
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("grpc", grpc.__version__)
print("diffusers", diffusers.__version__)
EOF

echo "Done. Activate with:  conda activate $ENV_NAME"
