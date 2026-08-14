#!/usr/bin/env bash
#
# capture_context.sh
#
# Packages the entire DIFS benchmark context into a portable tarball.
# Run on Machine A, transfer bundle to Machine B, run setup_machine.sh.
#
# Usage:
#   bash capture_context.sh [--output /path/to/output.tar.gz] [--model-dir /path/to/models]
#
# What's captured:
#   - Source code (all modified files, scripts, configs)
#   - Benchmark results (metrics CSVs, plot data, generated plots)
#   - Reports (PDFs, LaTeX sources, build artifacts)
#   - Environment snapshots (pip freeze, GPU info, system specs)
#   - Model path map (symlink targets for /tmp/model/)
#   - SD 3.5 config files (model_index.json, transformer config, etc.)
#   - Requirements files
#

set -euo pipefail

############################################################
# Parse arguments
############################################################
OUTPUT=""
MODEL_DIR_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)   OUTPUT="$2"; shift 2 ;;
        --model-dir) MODEL_DIR_OVERRIDE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--output path.tar.gz] [--model-dir /path/to/models]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_NAME="DIFS_Benchmark"

if [ -z "$OUTPUT" ]; then
    OUTPUT="/tmp/${PROJECT_NAME}_${TIMESTAMP}.tar.gz"
fi

# Detect model base directory from symlinks or default
if [ -z "$MODEL_DIR_OVERRIDE" ]; then
    # Try to find model dir from existing symlinks
    FIRST_LINK=$(readlink /tmp/model/stable-diffusion-v1-4 2>/dev/null || true)
    if [ -n "$FIRST_LINK" ]; then
        MODEL_DIR_OVERRIDE=$(dirname "$FIRST_LINK")
    else
        MODEL_DIR_OVERRIDE="/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models"
    fi
fi

echo "============================================"
echo " DIFS Benchmark - Capture Context"
echo "============================================"
echo ""
echo "Project:      $SCRIPT_DIR"
echo "Model dir:    $MODEL_DIR_OVERRIDE"
echo "Output:       $OUTPUT"
echo ""

############################################################
# Create temporary staging directory
############################################################
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

echo "[1/7] Capturing environment snapshots..."
mkdir -p "$STAGE/environment"

# System info
uname -a > "$STAGE/environment/system_info.txt" 2>/dev/null || true
free -h > "$STAGE/environment/memory.txt" 2>/dev/null || true
df -h "$SCRIPT_DIR" > "$STAGE/environment/disk.txt" 2>/dev/null || true

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader \
    > "$STAGE/environment/gpu_info.txt" 2>/dev/null || echo "No NVIDIA GPU detected" > "$STAGE/environment/gpu_info.txt"

# Python version
python3 --version > "$STAGE/environment/python_version.txt" 2>&1

# pip freeze
pip freeze > "$STAGE/environment/pip_freeze.txt" 2>/dev/null || true

# Conda env (optional)
conda env export --no-builds > "$STAGE/environment/conda_env.yml" 2>/dev/null || true

# PyTorch / CUDA info
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
" > "$STAGE/environment/pytorch_info.txt" 2>/dev/null || true

# Git state
if [ -d "$SCRIPT_DIR/.git" ]; then
    cd "$SCRIPT_DIR"
    git log --oneline -10 > "$STAGE/environment/git_log.txt" 2>/dev/null || true
    git diff --stat > "$STAGE/environment/git_diff_stat.txt" 2>/dev/null || true
    git status --short > "$STAGE/environment/git_status.txt" 2>/dev/null || true
fi

echo "  Done."

############################################################
echo "[2/7] Capturing model path map..."
############################################################
mkdir -p "$STAGE/model_map"

# Record symlinks in /tmp/model
if [ -d /tmp/model ]; then
    for link in /tmp/model/*; do
        [ -L "$link" ] || continue
        name=$(basename "$link")
        target=$(readlink -f "$link" 2>/dev/null || echo "BROKEN:$link")
        echo "$name -> $target" >> "$STAGE/model_map/symlinks.txt"
    done
fi

# Record model sizes from the source directory
echo "# Model sizes from $MODEL_DIR_OVERRIDE" > "$STAGE/model_map/model_sizes.txt"
echo "# Format: model_name size_bytes" >> "$STAGE/model_map/model_sizes.txt"
if [ -d "$MODEL_DIR_OVERRIDE" ]; then
    for d in "$MODEL_DIR_OVERRIDE"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        size=$(du -sb "$d" 2>/dev/null | cut -f1)
        echo "$name $size" >> "$STAGE/model_map/model_sizes.txt"
    done
    # Also check single-file models
    for f in "$MODEL_DIR_OVERRIDE"/*.safetensors; do
        [ -f "$f" ] || continue
        name=$(basename "$f")
        size=$(stat -c%s "$f" 2>/dev/null || echo 0)
        echo "$name $size" >> "$STAGE/model_map/model_sizes.txt"
    done
fi

echo "  Done."

############################################################
echo "[3/7] Capturing source code..."
############################################################
mkdir -p "$STAGE/code"

# Copy key source directories and files
# Use rsync for clean copy with exclusions
rsync -a --exclude='*.pyc' --exclude='__pycache__' --exclude='build/' \
    --exclude='.git' --exclude='*.safetensors' --exclude='model_checkpoints' \
    "$SCRIPT_DIR/" "$STAGE/code/" 2>/dev/null || \
    cp -r "$SCRIPT_DIR" "$STAGE/code/" 2>/dev/null || true

echo "  Done."

############################################################
echo "[4/7] Capturing benchmark results..."
############################################################
mkdir -p "$STAGE/results"

# Copy results directory
if [ -d "$SCRIPT_DIR/results" ]; then
    rsync -a "$SCRIPT_DIR/results/" "$STAGE/results/" 2>/dev/null || \
        cp -r "$SCRIPT_DIR/results" "$STAGE/results/" 2>/dev/null || true
fi

# Copy metrics directory
if [ -d "$SCRIPT_DIR/metrics" ]; then
    rsync -a "$SCRIPT_DIR/metrics/" "$STAGE/results/metrics/" 2>/dev/null || \
        cp -r "$SCRIPT_DIR/metrics" "$STAGE/results/metrics/" 2>/dev/null || true
fi

# Copy worker logs
if [ -d "$SCRIPT_DIR/WorkersLogs" ]; then
    rsync -a "$SCRIPT_DIR/WorkersLogs/" "$STAGE/results/WorkersLogs/" 2>/dev/null || \
        cp -r "$SCRIPT_DIR/WorkersLogs" "$STAGE/results/WorkersLogs/" 2>/dev/null || true
fi

echo "  Done."

############################################################
echo "[5/7] Capturing SD 3.5 config files..."
############################################################
mkdir -p "$STAGE/sd35_configs"

# Save all config JSON files from the SD 3.5 model directory
SD35_DIR="/tmp/model/stable-diffusion-3.5-large"
if [ -d "$SD35_DIR" ]; then
    # Save model_index.json
    [ -f "$SD35_DIR/model_index.json" ] && \
        cp "$SD35_DIR/model_index.json" "$STAGE/sd35_configs/"

    # Save transformer config
    [ -f "$SD35_DIR/transformer/config.json" ] && \
        mkdir -p "$STAGE/sd35_configs/transformer" && \
        cp "$SD35_DIR/transformer/config.json" "$STAGE/sd35_configs/transformer/"

    # Save scheduler config
    [ -f "$SD35_DIR/scheduler/config.json" ] && \
        mkdir -p "$STAGE/sd35_configs/scheduler" && \
        cp "$SD35_DIR/scheduler/config.json" "$STAGE/sd35_configs/scheduler/"
    [ -f "$SD35_DIR/scheduler_config.json" ] && \
        cp "$SD35_DIR/scheduler_config.json" "$STAGE/sd35_configs/"

    # Save text encoder configs
    for enc in text_encoder text_encoder_2 text_encoder_3; do
        [ -f "$SD35_DIR/$enc/config.json" ] && \
            mkdir -p "$STAGE/sd35_configs/$enc" && \
            cp "$SD35_DIR/$enc/config.json" "$STAGE/sd35_configs/$enc/"
    done

    # Save tokenizer configs
    for tok in tokenizer tokenizer_2 tokenizer_3; do
        if [ -d "$SD35_DIR/$tok" ] && [ "$(ls -A "$SD35_DIR/$tok" 2>/dev/null)" ]; then
            rsync -a "$SD35_DIR/$tok/" "$STAGE/sd35_configs/$tok/" 2>/dev/null || \
                cp -r "$SD35_DIR/$tok" "$STAGE/sd35_configs/$tok/" 2>/dev/null || true
        fi
    done
fi

echo "  Done."

############################################################
echo "[6/7] Capturing generated plots..."
############################################################
mkdir -p "$STAGE/plots"

# Copy all plot PNGs
if [ -d "$SCRIPT_DIR/results/model_benchmark/plots" ]; then
    rsync -a "$SCRIPT_DIR/results/model_benchmark/plots/" "$STAGE/plots/" 2>/dev/null || \
        cp -r "$SCRIPT_DIR/results/model_benchmark/plots" "$STAGE/plots/" 2>/dev/null || true
fi

echo "  Done."

############################################################
echo "[7/7] Building tarball..."
############################################################
# Create the tarball
tar czf "$OUTPUT" -C "$STAGE" .

SIZE=$(du -h "$OUTPUT" | cut -f1)
FILE_COUNT=$(tar tzf "$OUTPUT" | wc -l)

echo ""
echo "============================================"
echo " Context captured successfully!"
echo "============================================"
echo ""
echo "Bundle:      $OUTPUT"
echo "Size:        $SIZE"
echo "Files:       $FILE_COUNT"
echo ""
echo "Contents:"
echo "  code/         - All source files and scripts"
echo "  results/      - Benchmark metrics, CSVs, reports"
echo "  plots/        - Generated plot PNGs"
echo "  sd35_configs/ - SD 3.5 config/tokenizer files"
echo "  model_map/    - Symlink targets and model sizes"
echo "  environment/  - pip freeze, GPU info, system specs"
echo ""
echo "To transfer to another machine:"
echo "  scp $OUTPUT user@newmachine:/path/to/"
echo ""
echo "On the new machine, run:"
echo "  tar xzf ${PROJECT_NAME}_*.tar.gz"
echo "  bash code/setup_machine.sh"
