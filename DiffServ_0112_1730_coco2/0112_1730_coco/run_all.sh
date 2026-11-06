#!/usr/bin/env bash

# run_all.sh
#
# Simple one-shot driver for DIFS_OC_v3:
#   - setup       - conda env + pinned Python deps, DiffusionDB prompts, model weights  
#   - benchmark   - baseline comparison (R6) via run_baseline_benchmark.py
#   - slo         - live SLO benchmark (real GPU) via run_slo_benchmark.py
#   - plots       - all v3 figures from plots/slo_metrics.csv and generate_slo_plots.py
#   - report      - populate the LaTeX tables and compile the PDF  
#
# How to run:
#   # Full pipeline: setup + benchmark + slo + plots + PDF report
#   ./run_all.sh --all
#
#   # Run only SLO benchmark (live)
#   ./run_all.sh --slo
#
#   # Skip existing SLO CSV from previous runs
#   ./run_all.sh --force
#
# Requirements:
#   * Run from this directory (the script cd's here automatically).
#   * real_stats.npz (33 MB, FID reference) must be present here; it is NOT tracked by git.
#   * conda environment (DIFS_Gen, Python 3.10)
#   * HF_TOKEN required for gated SD-3.5-medium snapshot if models not already present  
#   * Output images are deleted after a completed run unless --keep-output-images is set.
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENV_NAME="${DIFS_ENV_NAME:-DIFS_Gen}"
PYTHON="${DIFS_PYTHON:-python3}"
MODEL_DIR="${DIFS_MODEL_DIR:-}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUT_DIR="${DIFS_OUTPUT_DIR:-/tmp/diffserv_output}"
KEEP_OUTPUT_IMAGES=0
FORCE=0
SKIP_SLO=0

# ---------------------------------------------------------------------------
# Logging / helpers  
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --all)            SKIP_SLO=0 ;;
        --slo)            SKIP_SLO=0 ;;  
        --force)          FORCE=1 ;;
        --keep-output-images) KEEP_OUTPUT_IMAGES=1 ;;
        -h|--help)        echo "Usage: $0 [options]"; exit 0 ;;
        *)                die "Unknown option: $1" ;;
    esac
    shift
done

cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Setup conda and environment 
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    local conda_bin
    for conda_bin in \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda"; do
        if [ -x "$conda_bin" ]; then
            PATH="$(dirname "$conda_bin"):$PATH"
            export PATH
            log "Found conda at $conda_bin"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
        warn "conda not found; using $PYTHON. Dependencies must already be installed."
    else
        die "conda not found and no --python given."
    fi
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda activate "$ENV_NAME"
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
else
    die "Conda env '$ENV_NAME' not found. Run with --setup first, or pass --python."
fi

log "Using Python: $PYTHON (env: $ENV_NAME)"

# ---------------------------------------------------------------------------
# Run SLO benchmark (live)
# ---------------------------------------------------------------------------

if [ "$SKIP_SLO" -eq 0 ]; then
    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_5model/slo_benchmark_results.csv"
    
    if [ -f "$CSV_PATH" ] && [ "$FORCE" -eq 0 ]; then
        log "SLO benchmark already exists, skipping (--force to regenerate)"
    else
        log "Running live SLO benchmark..."
        # Use the direct mode for a clean benchmark run 
        PYTHONPATH=src python -m src.experiments.run_slo_benchmark --direct --rerun
        
        if [ ! -f "$CSV_PATH" ]; then
            die "SLO benchmark failed to generate CSV"
        fi
        
        log "Live SLO benchmark complete"
    fi
fi

# ---------------------------------------------------------------------------
# Generate plots (from CSV and metrics)
# ---------------------------------------------------------------------------

log "Generating SLO plots from CSV ..."
# Note: generate_slo_plots.py reads from plot_data_5model/slo_benchmark_results.csv
PYTHONPATH=src python -m src.plots.generate_slo_plots

log "All v3 pipeline complete"
