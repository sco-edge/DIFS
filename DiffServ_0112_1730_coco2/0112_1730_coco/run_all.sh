#!/usr/bin/env bash
#
# run_all.sh
#
# Master script to run the full DIFS benchmark pipeline:
#   1. Download models (if needed)
#   2. Create model symlinks
#   3. Run benchmarks for all models × scaling modes
#   4. Generate plot CSVs from raw metrics
#   5. Generate all plots (bar + line + switching)
#
# Usage:
#   ./run_all.sh                        # Full pipeline
#   ./run_all.sh --download-only        # Only download models
#   ./run_all.sh --benchmark-only       # Only run benchmarks (skip download)
#   ./run_all.sh --plots-only           # Only regenerate plots
#   ./run_all.sh --models 1 2 3 4 5 7 8 # Specific models
#   ./run_all.sh --modes horizontal vertical hybrid
#   ./run_all.sh --collect-switching     # Measure pairwise switching times (needs running worker)
#
# Environment:
#   HF_TOKEN=hf_xxx ./run_all.sh       # Set HF token for gated repos
#

set -Eeuo pipefail

############################################################
# PATHS
############################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SRC_DIR="$PROJECT_ROOT/src"
RESULTS_DIR="$PROJECT_ROOT/results/model_benchmark"
MODEL_DIR="${MODEL_DIR:-/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models/diffusion-models}"
SYMLINK_DIR="/tmp/model"

# Models: local_subdir -> model_id
declare -A MODEL_MAP=(
    ["sd-v1-4.safetensors"]=1
    ["stable-diffusion-v1-4"]=2
    ["stable-diffusion-v1-5"]=3
    ["stable-diffusion-2-1"]=4
    ["stable-diffusion-xl-base-1.0"]=5
    ["dreamshaper-8"]=7
    ["stable-diffusion-3.5-medium"]=8
)

ALL_MODEL_NAMES=("${!MODEL_MAP[@]}")

############################################################
# ARGUMENTS
############################################################

MODE=""
SELECTED_MODELS=()
SELECTED_MODES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --download-only)
            MODE="download"
            shift
            ;;
        --benchmark-only)
            MODE="benchmark"
            shift
            ;;
        --plots-only)
            MODE="plots"
            shift
            ;;
        --collect-switching)
            MODE="switching"
            shift
            ;;
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_MODELS+=("$1")
                shift
            done
            ;;
        --modes)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SELECTED_MODES+=("$1")
                shift
            done
            ;;
        --help|-h)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Default models to benchmark
if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=(1 2 3 4 5 7 8)
fi

# Default scaling modes
if [[ ${#SELECTED_MODES[@]} -eq 0 ]]; then
    SELECTED_MODES=(horizontal vertical hybrid)
fi

############################################################
# HELPER FUNCTIONS
############################################################

log_step() {
    echo ""
    echo "============================================"
    echo " $1"
    echo "============================================"
    echo ""
}

check_prerequisites() {
    log_step "Checking prerequisites"

    # Check Python
    if ! command -v python &>/dev/null; then
        echo "ERROR: python not found"
        exit 1
    fi
    echo "  Python: $(python --version 2>&1)"

    # Check pip packages
    for pkg in torch diffusers transformers grpcio pandas matplotlib seaborn numpy; do
        if python -c "import $pkg" 2>/dev/null; then
            echo "  $pkg: $(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo 'installed')"
        else
            echo "  WARNING: $pkg not installed"
        fi
    done

    # Check HF CLI
    if command -v huggingface-cli &>/dev/null; then
        echo "  huggingface-cli: available"
    elif command -v hf &>/dev/null; then
        echo "  hf: available"
    else
        echo "  WARNING: huggingface CLI not found"
    fi

    echo ""
}

############################################################
# STEP 1: DOWNLOAD MODELS
############################################################

download_models() {
    log_step "Step 1: Downloading models"

    if [[ -n "${HF_TOKEN:-}" ]]; then
        echo "  HF_TOKEN is set"
    else
        echo "  WARNING: HF_TOKEN not set. Gated repos may fail."
        echo "  Set with: export HF_TOKEN=hf_xxxxxxxx"
        echo ""
    fi

    bash "$SRC_DIR/scripts/download_diffusion_models.sh"
}

############################################################
# STEP 2: CREATE SYMLINKS
############################################################

create_symlinks() {
    log_step "Step 2: Creating model symlinks in $SYMLINK_DIR"

    mkdir -p "$SYMLINK_DIR"

    for name in "${ALL_MODEL_NAMES[@]}"; do
        target="$MODEL_DIR/$name"
        link="$SYMLINK_DIR/$name"

        if [[ -e "$target" ]]; then
            if [[ -L "$link" ]]; then
                rm "$link"
            fi
            ln -s "$target" "$link"
            echo "  Linked: $name -> $target"
        else
            echo "  WARNING: $target not found, skipping symlink for $name"
        fi
    done

    echo ""
    echo "Current symlinks:"
    ls -la "$SYMLINK_DIR/"
}

############################################################
# STEP 3: RUN BENCHMARKS
############################################################

run_benchmarks() {
    log_step "Step 3: Running benchmarks"

    MODEL_ARGS="${SELECTED_MODELS[*]}"
    MODE_ARGS="${SELECTED_MODES[*]}"

    echo "  Models: $MODEL_ARGS"
    echo "  Modes:  $MODE_ARGS"
    echo ""

    cd "$PROJECT_ROOT"

    python "$SRC_DIR/experiments/run_model_benchmark.py" \
        --models $MODEL_ARGS \
        --modes $MODE_ARGS
}

############################################################
# STEP 4: GENERATE PLOT CSVs
############################################################

generate_csvs() {
    log_step "Step 4: Generating plot data CSVs"

    python "$SRC_DIR/plots/generate_plot_csvs.py" \
        --results-dir "$RESULTS_DIR"
}

############################################################
# STEP 5: GENERATE PLOTS
############################################################

generate_plots() {
    log_step "Step 5: Generating plots"

    PLOT_DATA="$RESULTS_DIR/plot_data"
    PLOT_OUTPUT="$RESULTS_DIR/plots"

    echo "  Generating bar plots..."
    python "$SRC_DIR/plots/generate_all_plots.py" \
        --plot-data-dir "$PLOT_DATA" \
        --output-dir "$PLOT_OUTPUT"

    echo ""
    echo "  Generating line plots..."
    python "$SRC_DIR/plots/generate_line_plots.py" \
        --plot-data-dir "$PLOT_DATA" \
        --output-dir "$PLOT_OUTPUT"

    echo ""
    echo "  Generating switching time plots..."
    python "$SRC_DIR/plots/generate_switching_plots.py" \
        --plot-data-dir "$PLOT_DATA" \
        --output-dir "$PLOT_OUTPUT"
}

############################################################
# STEP 6: COLLECT PAIRWISE SWITCHING TIMES
############################################################

collect_switching() {
    log_step "Collecting pairwise switching times"

    WORKER="${WORKER:-localhost:50060}"
    MODEL_ARGS="${SELECTED_MODELS[*]}"
    OUTPUT="$RESULTS_DIR/plot_data/pairwise_switching.csv"

    echo "  Worker: $WORKER"
    echo "  Models: $MODEL_ARGS"
    echo "  Output: $OUTPUT"
    echo ""
    echo "  NOTE: The worker must be running. Start it with:"
    echo "    python src/1/pytorch_container_diffusion_v3.py --port 50060 --model-id 1"
    echo ""

    python "$SRC_DIR/experiments/collect_pairwise_switching.py" \
        --worker "$WORKER" \
        --models $MODEL_ARGS \
        --output "$OUTPUT" \
        --warmup 1 \
        --repeat 3
}

############################################################
# MAIN
############################################################

echo ""
echo "============================================"
echo " DIFS Benchmark Pipeline"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Results dir:  $RESULTS_DIR"
echo "Model dir:    $MODEL_DIR"
echo "Symlink dir:  $SYMLINK_DIR"
echo ""
echo "Selected models: ${SELECTED_MODELS[*]}"
echo "Selected modes:  ${SELECTED_MODES[*]}"
echo ""

case "$MODE" in
    download)
        check_prerequisites
        download_models
        ;;
    benchmark)
        check_prerequisites
        create_symlinks
        run_benchmarks
        ;;
    plots)
        generate_csvs
        generate_plots
        ;;
    switching)
        collect_switching
        generate_csvs
        generate_plots
        ;;
    "")
        # Full pipeline
        check_prerequisites
        download_models
        create_symlinks
        run_benchmarks
        generate_csvs
        generate_plots
        ;;
esac

log_step "Pipeline complete!"

echo "Results:   $RESULTS_DIR"
echo "Plots:     $RESULTS_DIR/plots/"
echo "CSV data:  $RESULTS_DIR/plot_data/"
echo ""
echo "To regenerate only plots (no benchmarks):"
echo "  $0 --plots-only"
echo ""
echo "To rerun specific models:"
echo "  $0 --models 4 8 --modes horizontal"
