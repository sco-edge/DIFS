#!/usr/bin/env bash

# run_all_experiments.sh
#
# Unified driver script for DIFS_OC_v4 project
#   - setup       - conda env + pinned Python deps, DiffusionDB prompts, model weights  
#   - benchmark   - baseline comparison (R6) via run_baseline_benchmark.py
#   - plots       - all v4 figures from baseline_comparison.csv and plot_baseline_comparison.py
#   - report      - populate the LaTeX tables and compile the PDF  
#
# Requirements:
#   * Run from this directory (the script cd's here automatically).
#   * real_stats.npz (33 MB, FID reference) must be present here; it is NOT tracked by git.
#   * conda environment (DIFS_Gen, Python 3.10)
#   * HF_TOKEN required for gated SD-3.5-medium snapshot if models not already present  
#   * Output images are deleted after a completed run
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

# ---------------------------------------------------------------------------
# Logging / helpers  
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# Resolve a safe image output directory (cleanup + relocation on low disk).
resolve_output_dir() {
    local resolved
    resolved="$(python3 "$PROJECT_ROOT/src/output_storage.py" --requested "$OUTPUT_DIR" 2>/dev/null || true)"
    resolved="${resolved##*$'\n'}"
    if [ -n "$resolved" ]; then
        OUTPUT_DIR="$resolved"
        export DIFS_OUTPUT_DIR="$OUTPUT_DIR"
    fi
    log "Output images: $OUTPUT_DIR"
}

# Resolve a safe image output directory (cleanup + relocation on low disk).
resolve_output_dir() {
    local resolved
    resolved="$(python3 "$PROJECT_ROOT/src/output_storage.py" --requested "$OUTPUT_DIR" 2>/dev/null || true)"
    resolved="${resolved##*$'\n'}"
    if [ -n "$resolved" ]; then
        OUTPUT_DIR="$resolved"
        export DIFS_OUTPUT_DIR="$OUTPUT_DIR"
    fi
    log "Output images: $OUTPUT_DIR"
}

# Activate the project conda env (or fall back to its interpreter).
activate_env() {
    local conda_bin conda_base candidate
    for conda_bin in \
        "$HOME/miniforge3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda"; do
        if [ -x "$conda_bin" ]; then
            PATH="$(dirname "$conda_bin"):$PATH"
            export PATH
            break
        fi
    done

    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            source "$conda_base/etc/profile.d/conda.sh"
            if conda env list 2>/dev/null | awk -v env="$ENV_NAME" '$1 == env { found=1 } END { exit(found ? 0 : 1) }'; then
                conda activate "$ENV_NAME"
                PYTHON="$(command -v python)"
                export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
                log "Using Python: $PYTHON (env: $ENV_NAME)"
                return 0
            fi
        fi
    fi

    for candidate in \
        "$HOME/miniforge3/envs/$ENV_NAME/bin/python" \
        "$HOME/miniconda3/envs/$ENV_NAME/bin/python" \
        "$HOME/anaconda3/envs/$ENV_NAME/bin/python" \
        "$HOME/.conda/envs/$ENV_NAME/bin/python"; do
        if [ -x "$candidate" ]; then
            PYTHON="$candidate"
            PATH="$(dirname "$candidate"):$PATH"
            export PATH
            export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
            log "Using Python from an existing env: $PYTHON"
            log "Using Python: $PYTHON (env: $ENV_NAME)"
            return 0
        fi
    done

    die "Conda env '$ENV_NAME' not found. Run --setup first, or pass --python."
}

run_complete_pipeline() {
    resolve_output_dir
        activate_env

    # ---------------------------------------------------------------------------
    # Run baseline benchmark 
    # ---------------------------------------------------------------------------

    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_baseline/baseline_comparison.csv"

    if [ -f "$CSV_PATH" ]; then
        log "Baseline benchmark already exists, skipping"
    else
        log "Running baseline comparison benchmark..."
        PYTHONPATH=src python run_baseline_benchmark.py --baseline all
        
        if [ ! -f "$CSV_PATH" ]; then
            die "Baseline benchmark failed to generate CSV"
        fi
        
        log "Baseline benchmark complete"
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
        pdflatex -interaction=nonstopmode "$(basename "$REPORT_TEX")" >/dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode "$(basename "$REPORT_TEX")" >/dev/null 2>&1 || true
        
        if [ -f "$REPORT_PDF" ]; then
            log "PDF report generated successfuly: $REPORT_PDF"
        else
            warn "Could not compile PDF report"
        fi
        
        cd "$PROJECT_ROOT"
    else
        warn "No LaTeX file found for report generation"
    fi

    log "v4 pipeline completed successfully"
}

run_setup_only() {
    activate_env
    log "Setup completed successfully"
}


run_benchmark_only() {
    resolve_output_dir
    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_baseline/baseline_comparison.csv"

    if [ -f "$CSV_PATH" ]; then
        log "Baseline benchmark already exists, skipping (use Force Re-run to overwrite)"
    else
        log "Running baseline comparison benchmark..."
        PYTHONPATH=src python run_baseline_benchmark.py --baseline all
        
        if [ ! -f "$CSV_PATH" ]; then
            die "Baseline benchmark failed to generate CSV"
        fi
        
        log "Baseline benchmark complete"
    fi
}

run_plots_only() {
    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_baseline/baseline_comparison.csv"
    
    if [ ! -f "$CSV_PATH" ]; then
        warn "No CSV data found. Run benchmark first."
        return 1
    fi

    log "Generating baseline plots from CSV ..."
    PYTHONPATH=src python src/plots/plot_baseline_comparison.py
}

run_report_only() {
    REPORT_DIR="$PROJECT_ROOT/results/model_benchmark/Reports"
    REPORT_TEX="$REPORT_DIR/DIFS_OC_v4_Benchmark_Report.tex"  

    if [ -f "$REPORT_TEX" ]; then
        log "Compiling PDF from LaTeX..."
        cd "$REPORT_DIR"
        
        # Build PDF (attempt 2 passes for references)
        pdflatex -interaction=nonstopmode "$(basename "$REPORT_TEX")" >/dev/null 2>&1 || true
        pdflatex -interaction=nonstopmode "$(basename "$REPORT_TEX")" >/dev/null 2>&1 || true
        
        if [ -f "$REPORT_TEX.pdf" ]; then
            log "PDF report generated successfully"
        else
            warn "Could not compile PDF report"
        fi
        
        cd "$PROJECT_ROOT"
    else
        warn "No LaTeX file found for report generation"
    fi
}

run_force_rerun() {
    # Remove existing files to force fresh run
    CSV_PATH="$PROJECT_ROOT/results/model_benchmark/plot_data_baseline/baseline_comparison.csv"
    if [ -f "$CSV_PATH" ]; then
        rm "$CSV_PATH"
        log "Removed existing CSV file for fresh benchmark run"
    fi
    
    # Remove plots directory if it exists
    PLOT_DIR="$PROJECT_ROOT/plots"
    if [ -d "$PLOT_DIR" ]; then
        rm -rf "$PLOT_DIR"
        log "Removed existing plot directory"
    fi
    
    # Run complete pipeline with force flag
    run_complete_pipeline
}

run_clean_output() {
    # Remove temporary files from both the requested and the fallback location
    for TMP_DIR in "/tmp/diffserv_output" "$OUTPUT_DIR"; do
        if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
            rm -rf "$TMP_DIR"
            log "Cleaned temporary output directory: $TMP_DIR"
        fi
    done
    
    # Remove plot files
    PLOT_DIR="$PROJECT_ROOT/plots"
    if [ -d "$PLOT_DIR" ]; then
        rm -rf "$PLOT_DIR"
        log "Removed existing plot directory"
    fi
    
    log "Output cleaning completed successfully"
}

show_help() {
    echo "===================================="
    echo "     DIFS_OC_v4 Project Help"
    echo "===================================="
    echo "This script provides an interactive menu for running DIFS_OC_v4 experiments."
    echo ""
    echo "Options:"
    echo "1. Complete Pipeline - Runs setup, benchmark, plots, and report generation"
    echo "2. Setup Only - Configures conda environment only" 
    echo "3. Benchmark Only - Runs baseline comparison only"
    echo "4. Plots Only - Generates figures from existing CSV data"
    echo "5. Report Only - Compiles LaTeX reports only"
    echo "6. Force Re-run - Removes existing results and starts fresh"
    echo "7. Clean Output - Cleans temporary files and outputs"
    echo ""
    echo "Requirements:"
    echo "- real_stats.npz must be present in the project directory"
    echo "- Conda environment '$ENV_NAME' must be set up" 
    echo "- HF_TOKEN required for gated model access if needed"
    echo "===================================="
}

# ---------------------------------------------------------------------------
# Non-interactive CLI dispatch (same code path as the menu)
# ---------------------------------------------------------------------------
usage() {
    cat <<'EOF'
Usage: ./run_all_experiments.sh [OPTION]

Run the DIFS_OC_v4 pipeline without the interactive menu.

Options:
  --all          Complete pipeline (setup -> benchmark -> plots -> report)
  --setup        Environment/dependency setup only
  --benchmark    Baseline comparison only
  --plots        Regenerate figures from the CSV
  --report       Compile the LaTeX/PDF report
  --force        Remove existing results and rerun the complete pipeline
  --clean        Remove temporary output and plots
  --v7           Run DIFS_OC_v7 (quality testing)
  -h, --help     Show this help

With no option, an interactive menu is shown.
EOF
}

if [ $# -gt 0 ]; then
    case "${1:-}" in
        --all|-a)      run_complete_pipeline ;;
        --setup)       run_setup_only ;;
        --benchmark)   run_benchmark_only ;;
        --plots)       run_plots_only ;;
        --report)      run_report_only ;;
        --force)       run_force_rerun ;;
        --clean)       run_clean_output ;;
        -h|--help)     usage; exit 0 ;;
        *)             die "Unknown option: $1 (use --help)" ;;
    esac
    exit 0
fi

# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------
while true; do
    show_menu
    case $choice in
        1)
            run_complete_pipeline
            read -p "Press Enter to continue..."
            ;;
        2)
            run_setup_only
            read -p "Press Enter to continue..."
            ;;
        3)
            run_benchmark_only
            read -p "Press Enter to continue..."
            ;;
        4)
            run_plots_only
            read -p "Press Enter to continue..."
            ;;
        5)
            run_report_only
            read -p "Press Enter to continue..."
            ;;
        6)
            run_force_rerun
            read -p "Press Enter to continue..."
            ;;
        7)
            run_clean_output
            read -p "Press Enter to continue..."
            ;;
        8)
            show_help
            read -p "Press Enter to continue..."
            ;;
        9)
            log "Exiting script. Goodbye!"
            exit 0
            ;;
        *)
            warn "Invalid option. Please select 1-9."
            read -p "Press Enter to continue..."
            ;;
    esac
done
