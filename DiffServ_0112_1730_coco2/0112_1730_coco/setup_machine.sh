#!/usr/bin/env bash
#
# setup_machine.sh
#
# Sets up the DIFS benchmark on a new machine from a captured context bundle.
# Handles: dependency installation, model downloads, symlinks, SD 3.5 configs.
#
# Usage:
#   # From extracted bundle:
#   cd /path/to/extracted/bundle/code
#   bash setup_machine.sh [--model-dir /path/to/models] [--skip-download] [--gpu-mem 8]
#
#   # From original project directory (no bundle):
#   bash setup_machine.sh [--model-dir /path/to/models] [--gpu-mem 8]
#
# What it does:
#   1. Installs Python dependencies (core + benchmark packages)
#   2. Detects GPU capabilities (VRAM, compute capability)
#   3. Adjusts benchmark config based on available GPU memory
#   4. Creates model directory structure
#   5. Downloads missing models via download_diffusion_models.sh
#   6. Creates /tmp/model symlinks
#   7. Sets up SD 3.5 config files and tokenizer
#   8. Verifies everything works
#

set -euo pipefail

############################################################
# Parse arguments
############################################################
MODEL_DIR=""
SKIP_DOWNLOAD=false
GPU_MEM_GB=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)     MODEL_DIR="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --gpu-mem)       GPU_MEM_GB="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--model-dir /path/to/models] [--skip-download] [--gpu-mem GB]"
            echo ""
            echo "Options:"
            echo "  --model-dir PATH    Base directory for model weights (default: auto-detect)"
            echo "  --skip-download     Skip model downloads (assume models already present)"
            echo "  --gpu-mem GB        Manually specify GPU VRAM in GB (default: auto-detect)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Determine project root
if [ -f "src/model_registry.py" ]; then
    PROJECT_DIR="$(pwd)"
elif [ -f "../../src/model_registry.py" ]; then
    PROJECT_DIR="$(cd ../.. && pwd)"
else
    PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

cd "$PROJECT_DIR"

echo "============================================"
echo " DIFS Benchmark - Machine Setup"
echo "============================================"
echo ""
echo "Project: $PROJECT_DIR"
echo ""

############################################################
# Step 1: Detect GPU
############################################################
echo "[1/8] Detecting GPU..."

if [ "$GPU_MEM_GB" -gt 0 ]; then
    echo "  Using specified GPU memory: ${GPU_MEM_GB} GB"
else
    GPU_MEM_BYTES=$(python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0).total_mem)
else:
    print(0)
" 2>/dev/null || echo 0)

    if [ "$GPU_MEM_BYTES" -gt 0 ]; then
        GPU_MEM_GB=$((GPU_MEM_BYTES / 1073741824))
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo "  Detected: $GPU_NAME (${GPU_MEM_GB} GB)"
    else
        GPU_MEM_GB=4
        echo "  WARNING: No GPU detected, assuming 4 GB (CPU-only mode)"
    fi
fi

# Determine benchmark config based on GPU memory
if [ "$GPU_MEM_GB" -le 4 ]; then
    BENCH_STEPS=10
    OFFLOAD="yes"
    echo "  Config: steps=10, CPU offloading enabled (GPU <= 4 GB)"
elif [ "$GPU_MEM_GB" -le 8 ]; then
    BENCH_STEPS=15
    OFFLOAD="yes"
    echo "  Config: steps=15, CPU offloading enabled (GPU <= 8 GB)"
elif [ "$GPU_MEM_GB" -le 16 ]; then
    BENCH_STEPS=20
    OFFLOAD="yes"
    echo "  Config: steps=20, CPU offloading enabled (GPU <= 16 GB)"
else
    BENCH_STEPS=20
    OFFLOAD="no"
    echo "  Config: steps=20, CPU offloading NOT needed (GPU > 16 GB)"
fi
echo ""

############################################################
# Step 2: Install Python dependencies
############################################################
echo "[2/8] Installing Python dependencies..."

# Install core DIFS requirements
if [ -f "requirements-benchmark.txt" ]; then
    echo "  Installing from requirements-benchmark.txt..."
    pip install -r requirements-benchmark.txt -q 2>/dev/null || \
        pip install -r requirements-benchmark.txt 2>&1 | tail -5
elif [ -f "requirements.txt" ]; then
    echo "  Installing from requirements.txt..."
    pip install -r requirements.txt -q 2>/dev/null || \
        pip install -r requirements.txt 2>&1 | tail -5
fi

# Verify critical packages
python3 -c "
import diffusers, transformers, torch, safetensors
print(f'  diffusers:    {diffusers.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  torch:        {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
print(f'  safetensors:  {safetensors.__version__}')
" 2>/dev/null || echo "  WARNING: Some packages missing"
echo ""

############################################################
# Step 3: Determine model directory
############################################################
echo "[3/8] Setting up model directory..."

if [ -z "$MODEL_DIR" ]; then
    # Try to detect from existing symlinks
    FIRST_LINK=$(readlink /tmp/model/stable-diffusion-v1-4 2>/dev/null || true)
    if [ -n "$FIRST_LINK" ]; then
        PARENT=$(dirname "$FIRST_LINK")
        # Check if parent already ends with diffusion-models
        if [ "$(basename "$PARENT")" = "diffusion-models" ]; then
            MODEL_DIR="$PARENT"
        else
            MODEL_DIR="$PARENT/diffusion-models"
        fi
        echo "  Detected from symlinks: $MODEL_DIR"
    else
        # Check model_map from bundle
        if [ -f "../model_map/symlinks.txt" ]; then
            FIRST_TARGET=$(grep "stable-diffusion-v1-4" ../model_map/symlinks.txt | awk '{print $3}')
            if [ -n "$FIRST_TARGET" ]; then
                PARENT=$(dirname "$FIRST_TARGET")
                if [ "$(basename "$PARENT")" = "diffusion-models" ]; then
                    MODEL_DIR="$PARENT"
                else
                    MODEL_DIR="$PARENT/diffusion-models"
                fi
            fi
        fi
    fi

    # Final fallback
    if [ -z "$MODEL_DIR" ]; then
        MODEL_DIR="/mnt/ssd4t/kwadwo/Downloads/sd-models/downloaded_diffusion_models/diffusion-models"
    fi
fi

mkdir -p "$MODEL_DIR"
echo "  Model directory: $MODEL_DIR"
echo ""

############################################################
# Step 4: Download models
############################################################
echo "[4/8] Checking model downloads..."

# Model registry (mirrors model_registry.py)
declare -A MODELS=(
    ["CompVis/stable-diffusion-v1-4"]="stable-diffusion-v1-4"
    ["stable-diffusion-v1-5/stable-diffusion-v1-5"]="stable-diffusion-v1-5"
    ["sd2-community/stable-diffusion-2-1"]="stable-diffusion-2-1"
    ["stabilityai/stable-diffusion-xl-base-1.0"]="stable-diffusion-xl-base-1.0"
    ["stabilityai/stable-diffusion-xl-refiner-1.0"]="stable-diffusion-xl-refiner-1.0"
    ["Lykon/dreamshaper-8"]="dreamshaper-8"
    ["camenduru/stable-diffusion-3.5-large"]="stable-diffusion-3.5-large"
)

declare -A MODEL_KEYS=(
    ["stable-diffusion-v1-4"]="CompVis/stable-diffusion-v1-4"
    ["stable-diffusion-v1-5"]="stable-diffusion-v1-5/stable-diffusion-v1-5"
    ["stable-diffusion-2-1"]="sd2-community/stable-diffusion-2-1"
    ["stable-diffusion-xl-base-1.0"]="stabilityai/stable-diffusion-xl-base-1.0"
    ["stable-diffusion-xl-refiner-1.0"]="stabilityai/stable-diffusion-xl-refiner-1.0"
    ["dreamshaper-8"]="Lykon/dreamshaper-8"
    ["stable-diffusion-3.5-large"]="camenduru/stable-diffusion-3.5-large"
)

# Check which models are present
MISSING=()
for name in stable-diffusion-v1-4 stable-diffusion-v1-5 stable-diffusion-xl-base-1.0 \
            dreamshaper-8 stable-diffusion-2-1 stable-diffusion-3.5-large; do
    if [ -d "$MODEL_DIR/$name" ]; then
        count=$(find "$MODEL_DIR/$name" -name "*.safetensors" -o -name "*.ckpt" 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "  OK: $name ($count weight files)"
        else
            echo "  EMPTY: $name (no weight files)"
            MISSING+=("$name")
        fi
    else
        echo "  MISSING: $name"
        MISSING+=("$name")
    fi
done

# Check single-file SD 1.4
if [ -f "$MODEL_DIR/sd-v1-4.safetensors" ]; then
    echo "  OK: sd-v1-4.safetensors (single file)"
else
    echo "  MISSING: sd-v1-4.safetensors"
    MISSING+=("sd-v1-4.safetensors")
fi

if [ ${#MISSING[@]} -gt 0 ] && [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "  Downloading ${#MISSING[@]} missing models..."
    echo "  This may take several hours depending on network speed."
    echo ""

    export MODEL_DIR
    bash src/scripts/download_diffusion_models.sh 2>&1 | tail -20
elif [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "  WARNING: ${#MISSING[@]} models missing but --skip-download set."
    echo "  Missing: ${MISSING[*]}"
    echo "  Run without --skip-download to fetch them."
fi
echo ""

############################################################
# Step 5: Create /tmp/model symlinks
############################################################
echo "[5/8] Creating model symlinks in /tmp/model..."

mkdir -p /tmp/model

# Directory-based models
for name in stable-diffusion-v1-4 stable-diffusion-v1-5 stable-diffusion-xl-base-1.0 \
            stable-diffusion-xl-refiner-1.0 dreamshaper-8 stable-diffusion-2-1 \
            stable-diffusion-3.5-large; do
    src="$MODEL_DIR/$name"
    dst="/tmp/model/$name"
    if [ -d "$src" ]; then
        if [ -L "$dst" ]; then
            # Update existing symlink
            rm "$dst"
            ln -s "$src" "$dst"
            echo "  Updated: $name"
        elif [ ! -e "$dst" ]; then
            ln -s "$src" "$dst"
            echo "  Created: $name"
        else
            echo "  Exists:  $name (not a symlink, skipping)"
        fi
    else
        echo "  SKIP:    $name (source not found)"
    fi
done

# Single-file models
SF_SRC="$MODEL_DIR/sd-v1-4.safetensors"
SF_DST="/tmp/model/sd-v1-4.safetensors"
if [ -f "$SF_SRC" ]; then
    if [ -L "$SF_DST" ]; then
        rm "$SF_DST"
        ln -s "$SF_SRC" "$SF_DST"
    elif [ ! -e "$SF_DST" ]; then
        ln -s "$SF_SRC" "$SF_DST"
    fi
    echo "  OK: sd-v1-4.safetensors"
fi

echo ""

############################################################
# Step 6: Set up SD 3.5 config files
############################################################
echo "[6/8] Setting up SD 3.5 config files..."

SD35_DIR="/tmp/model/stable-diffusion-3.5-large"
if [ -d "$SD35_DIR" ]; then
    # Try to restore from bundle first
    if [ -f "../sd35_configs/model_index.json" ]; then
        echo "  Restoring configs from captured bundle..."
        cp "../sd35_configs/model_index.json" "$SD35_DIR/" 2>/dev/null || true
        [ -d "../sd35_configs/transformer" ] && \
            mkdir -p "$SD35_DIR/transformer" && \
            cp "../sd35_configs/transformer/config.json" "$SD35_DIR/transformer/" 2>/dev/null || true
        [ -d "../sd35_configs/scheduler" ] && \
            mkdir -p "$SD35_DIR/scheduler" && \
            cp "../sd35_configs/scheduler/config.json" "$SD35_DIR/scheduler/" 2>/dev/null || true
        [ -f "../sd35_configs/scheduler_config.json" ] && \
            cp "../sd35_configs/scheduler_config.json" "$SD35_DIR/" 2>/dev/null || true
        for enc in text_encoder text_encoder_2 text_encoder_3; do
            [ -d "../sd35_configs/$enc" ] && \
                mkdir -p "$SD35_DIR/$enc" && \
                cp "../sd35_configs/$enc/config.json" "$SD35_DIR/$enc/" 2>/dev/null || true
        done
        for tok in tokenizer tokenizer_2 tokenizer_3; do
            [ -d "../sd35_configs/$tok" ] && \
                rsync -a "../sd35_configs/$tok/" "$SD35_DIR/$tok/" 2>/dev/null || \
                cp -r "../sd35_configs/$tok" "$SD35_DIR/$tok" 2>/dev/null || true
        done
        echo "  Restored from bundle."
    fi

    # Create any missing configs with defaults
    if [ ! -f "$SD35_DIR/model_index.json" ]; then
        cat > "$SD35_DIR/model_index.json" << 'EOF'
{
  "_class_name": "StableDiffusion3Pipeline",
  "_diffusers_version": "0.36.0",
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
  "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
  "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
  "text_encoder_3": ["transformers", "T5EncoderModel"],
  "tokenizer": ["transformers", "CLIPTokenizer"],
  "tokenizer_2": ["transformers", "CLIPTokenizer"],
  "tokenizer_3": ["transformers", "T5TokenizerFast"],
  "transformer": ["diffusers", "SD3Transformer2DModel"],
  "vae": ["diffusers", "AutoencoderKL"]
}
EOF
        echo "  Created model_index.json"
    fi

    if [ ! -f "$SD35_DIR/transformer/config.json" ]; then
        mkdir -p "$SD35_DIR/transformer"
        cat > "$SD35_DIR/transformer/config.json" << 'EOF'
{
  "_class_name": "SD3Transformer2DModel",
  "_diffusers_version": "0.36.0",
  "sample_size": 128,
  "in_channels": 16,
  "joint_attention_dim": 4096,
  "cross_attention_dim": 4096,
  "caption_projection_dim": 1536,
  "pooled_projection_dim": 2048,
  "out_channels": 16,
  "num_layers": 38,
  "num_single_layers": 38,
  "num_heads": 38,
  "attention_head_dim": 128,
  "crops_coords_top_left": [0, 0],
  "use_spatial_transformer": true,
  "rope_base": 10000.0,
  "rope_scaling": {"type": "linear", "factor": 1.0}
}
EOF
        echo "  Created transformer/config.json"
    fi

    if [ ! -f "$SD35_DIR/scheduler/config.json" ]; then
        mkdir -p "$SD35_DIR/scheduler"
        cat > "$SD35_DIR/scheduler/config.json" << 'EOF'
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.36.0",
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "use_dynamic_shifting": false,
  "base_shift": 0.95,
  "max_shift": 1.15,
  "base_image_seq_len": 256,
  "max_image_seq_len": 4096,
  "invert_sigmas": false,
  "shift_terminal": false,
  "use_karras_sigmas": false,
  "sigma_min": 0.0361,
  "sigma_max": 14.6146,
  "timestep_spacing": "linspace",
  "timestep_type": "discrete",
  "prediction_type": "epsilon",
  "t_range": [0.0, 1.0],
  "final_model_num_steps": 1000,
  "sigma_schedule": "logn_normal",
  "sigma_shift": 3.0,
  "sigma_data": 1.0,
  "timestep_transform": null
}
EOF
        cp "$SD35_DIR/scheduler/config.json" "$SD35_DIR/scheduler_config.json"
        echo "  Created scheduler configs"
    fi

    # Text encoder configs
    if [ ! -f "$SD35_DIR/text_encoder/config.json" ]; then
        mkdir -p "$SD35_DIR/text_encoder"
        echo '{"_class_name":"CLIPTextModelWithProjection","_diffusers_version":"0.36.0","projection_dim":1280,"hidden_size":1280,"intermediate_size":5120,"num_attention_heads":16,"num_hidden_layers":23,"num_positions":77,"vocab_size":49408,"bos_token_id":49406,"eos_token_id":49407}' | python3 -m json.tool > "$SD35_DIR/text_encoder/config.json"
        echo "  Created text_encoder/config.json"
    fi

    if [ ! -f "$SD35_DIR/text_encoder_2/config.json" ]; then
        mkdir -p "$SD35_DIR/text_encoder_2"
        echo '{"_class_name":"CLIPTextModelWithProjection","_diffusers_version":"0.36.0","projection_dim":768,"hidden_size":768,"intermediate_size":3072,"num_attention_heads":12,"num_hidden_layers":12,"num_positions":77,"vocab_size":49408,"bos_token_id":49406,"eos_token_id":49407}' | python3 -m json.tool > "$SD35_DIR/text_encoder_2/config.json"
        echo "  Created text_encoder_2/config.json"
    fi

    if [ ! -f "$SD35_DIR/text_encoder_3/config.json" ]; then
        mkdir -p "$SD35_DIR/text_encoder_3"
        echo '{"_name_or_path":"t5xxl","architectures":["T5EncoderModel"],"decoder_start_token_id":0,"d_ff":16384,"d_model":4096,"dropout_rate":0.1,"eos_token_id":1,"vocab_size":32128,"num_layers":24,"num_heads":64,"relative_attention_num_buckets":32,"relative_attention_max_distance":128,"tie_word_embeddings":false}' | python3 -m json.tool > "$SD35_DIR/text_encoder_3/config.json"
        echo "  Created text_encoder_3/config.json"
    fi

    # T5 tokenizer
    if [ ! -f "$SD35_DIR/tokenizer_3/tokenizer_config.json" ]; then
        echo "  Downloading T5 tokenizer..."
        python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('t5-small')
tok.save_pretrained('$SD35_DIR/tokenizer_3')
print('  T5 tokenizer saved')
" 2>/dev/null || echo "  WARNING: Could not download T5 tokenizer"
    fi
else
    echo "  SD 3.5 directory not found at $SD35_DIR"
    echo "  Download models first, then re-run this script."
fi
echo ""

############################################################
# Step 7: Update benchmark config for this machine
############################################################
echo "[7/8] Updating benchmark config..."

# Update BENCHMARK_STEPS in experiment_config.py
CONFIG_FILE="src/experiments/experiment_config.py"
if [ -f "$CONFIG_FILE" ]; then
    # Replace BENCHMARK_STEPS value
    sed -i "s/^BENCHMARK_STEPS\s*=.*/BENCHMARK_STEPS = $BENCH_STEPS/" "$CONFIG_FILE"
    echo "  Set BENCHMARK_STEPS = $BENCH_STEPS in $CONFIG_FILE"

    # Update MODEL_SWITCH_MODE based on GPU
    if [ "$OFFLOAD" = "yes" ]; then
        sed -i 's/^MODEL_SWITCH_MODE\s*=.*/MODEL_SWITCH_MODE = "hot"/' "$CONFIG_FILE"
        echo "  Set MODEL_SWITCH_MODE = 'hot' (CPU offloading enabled)"
    else
        sed -i 's/^MODEL_SWITCH_MODE\s*=.*/MODEL_SWITCH_MODE = "cold"/' "$CONFIG_FILE"
        echo "  Set MODEL_SWITCH_MODE = 'cold' (direct GPU loading)"
    fi
fi
echo ""

############################################################
# Step 8: Verify setup
############################################################
echo "[8/8] Verifying setup..."

ERRORS=0

# Check Python packages
python3 -c "import diffusers; assert diffusers.__version__ >= '0.36.0'" 2>/dev/null || \
    { echo "  ERROR: diffusers >= 0.36.0 required (SD 3.5 support)"; ERRORS=$((ERRORS+1)); }

python3 -c "import transformers; assert transformers.__version__ >= '4.50.0'" 2>/dev/null || \
    { echo "  WARNING: transformers >= 4.50.0 recommended"; }

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || \
    echo "  WARNING: CUDA not available, will use CPU (very slow)"

# Check symlinks
for name in stable-diffusion-v1-4 stable-diffusion-xl-base-1.0 dreamshaper-8; do
    if [ -L "/tmp/model/$name" ] && [ -d "/tmp/model/$name" ]; then
        echo "  OK: /tmp/model/$name"
    else
        echo "  MISSING: /tmp/model/$name"
        ERRORS=$((ERRORS+1))
    fi
done

# Check SD 3.5 configs
if [ -f "/tmp/model/stable-diffusion-3.5-large/model_index.json" ]; then
    echo "  OK: SD 3.5 model_index.json"
else
    echo "  WARNING: SD 3.5 config files missing"
fi

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "============================================"
    echo " Setup completed with $ERRORS error(s)"
    echo "============================================"
else
    echo "============================================"
    echo " Setup complete!"
    echo "============================================"
fi
echo ""
echo "Environment: ${GPU_MEM_GB}GB GPU, steps=${BENCH_STEPS}, offload=${OFFLOAD}"
echo ""
echo "To run the benchmark:"
echo "  cd $PROJECT_DIR"
echo "  python src/experiments/run_model_benchmark.py \\"
echo "    --models 1 2 3 4 5 7 \\"
echo "    --modes horizontal vertical hybrid \\"
echo "    --steps $BENCH_STEPS"
echo ""
echo "To run SD 3.5 (if loading is resolved):"
echo "  python src/experiments/run_model_benchmark.py --models 8 --modes horizontal --steps $BENCH_STEPS"
echo ""
echo "To generate plots:"
echo "  python src/plots/generate_plot_csvs.py"
echo "  python src/plots/generate_all_plots.py"
echo "  python src/plots/generate_line_plots.py"
echo "  python src/plots/generate_switching_plots.py"
echo ""
echo "To compile reports:"
echo "  cd results/model_benchmark/Reports"
echo "  pdflatex DIFS_Benchmark_Report.tex && pdflatex DIFS_Benchmark_Report.tex"
echo "  pdflatex DIFS_Benchmark_Problems_and_Solutions.tex && pdflatex DIFS_Benchmark_Problems_and_Solutions.tex"
echo ""
echo "To capture context for transfer to another machine:"
echo "  bash capture_context.sh"
