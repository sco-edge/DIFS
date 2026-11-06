#!/usr/bin/env bash

# run_all.sh
#
# Simple one-shot driver for DIFS_OC_v4:
#   - setup       - conda env + pinned Python deps, DiffusionDB prompts, model weights  
#   - benchmark   - baseline comparison (R6) via run_baseline_benchmark.py
#   - plots       - all v4 figures from baseline_comparison.csv and plot_baseline_comparison.py
#   - report      - populate the LaTeX tables and compile the PDF  
#
# How to run:
#   # Full pipeline: setup + benchmark + plots + PDF report
#   ./run_all.sh --all
#
#   # Run only baseline benchmark 
#   ./run_all.sh --benchmark
#
#   # Skip existing benchmark CSV from previous runs
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
SKIP_BENCHMARK=0

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
        --all)            SKIP_BENCHMARK=0 ;;
        --benchmark)      SKIP_BENCHMARK=0 ;;  
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
# Run baseline benchmark 
# ---------------------------------------------------------------------------

if [ "$SKIP_BENCHMARK" -eq 0 ]; then
    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_baseline/baseline_comparison.csv"
    
    if [ -f "$CSV_PATH" ] && [ "$FORCE" -eq 0 ]; then
        log "Baseline benchmark already exists, skipping (--force to regenerate)"
    else
        log "Running baseline comparison benchmark..."
        PYTHONPATH=src python run_baseline_benchmark.py --baseline all
        
        if [ ! -f "$CSV_PATH" ]; then
            die "Baseline benchmark failed to generate CSV"
        fi
        
        log "Baseline benchmark complete"
    fi
fi

# ---------------------------------------------------------------------------
# Generate plots (from CSV)
# ---------------------------------------------------------------------------

log "Generating baseline plots from CSV ..."
PYTHONPATH=src python src/plots/plot_baseline_comparison.py

# ---------------------------------------------------------------------------
# Compile PDF report (if TeX available)
# ---------------------------------------------------------------------------

REPORT_DIR="$PROJECT_ROOT/results/model_benchmark/Reports"
REPORT_TEX="$REPORT_DIR/DIFS_OC_v4_Benchmark_Report.tex"  
REPORT_PDF="$REPORT_TEX.pdf"

if [ -f "$REPORT_TEX" ]; then
    log "Compiling PDF from LaTeX..."
    cd "$REPORT_DIR"
    
    # Build PDF (attempt 2 passes for references)
    pdflatex -interaction=nonstopmode DIFS_OC_v4_Benchmark_Report.tex >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode DIFS_OC_v4_Benchmark_Report.tex >/dev/null 2>&1 || true
    
    if [ -f "$REPORT_PDF" ]; then
        log "PDF report generated successfuly: $REPORT_PDF"
    else
        warn "Could not compile PDF report"
    fi
    
    cd "$PROJECT_ROOT"
else
    warn "No LaTeX file found for report generation"
fi

log "All v4 pipeline complete"
