#!/usr/bin/env bash

# run_all_projects.sh
#
# Unified driver script for all DIFS_OC projects (v1 through v5)
# Provides interactive menu to manage all 5 projects from one interface
#

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Logging / helpers
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Display interactive menu
# ---------------------------------------------------------------------------
show_menu() {
    clear 2>/dev/null || true
    echo "===================================="
    echo "   DIFS_OC Projects Manager"
    echo "===================================="
    echo "Available projects:"
    echo "1. DIFS_OC_v1 - Legacy project"
    echo "2. DIFS_OC_v2 - SLO benchmarking"
    echo "3. DIFS_OC_v3 - SLO benchmarking (updated)"
    echo "4. DIFS_OC_v4 - Baseline comparison"
    echo "5. DIFS_OC_v5 - Switching modes"
    echo "6. All Projects - Run all projects"
    echo "7. Help/Information"
    echo "8. Exit"
    echo "===================================="
    read -p "Please select an option (1-8): " choice
    echo ""
}

# ---------------------------------------------------------------------------
# Project-specific functions
# ---------------------------------------------------------------------------

run_project() {
    local proj="$1"
    local dir="../$proj/DiffServ_0112_1730_coco2/0112_1730_coco"
    log "Running $proj project..."
    if [ ! -d "$dir" ]; then
        die "Failed to navigate to $proj directory: $dir"
    fi
    ( cd "$dir" && if [ -f "run_all_experiments.sh" ]; then bash ./run_all_experiments.sh; else log "No run script found in $proj"; fi )
}

run_project_v1() { run_project DIFS_OC_v1; }
run_project_v2() { run_project DIFS_OC_v2; }
run_project_v3() { run_project DIFS_OC_v3; }
run_project_v4() { run_project DIFS_OC_v4; }
run_project_v5() { run_project DIFS_OC_v5; }

run_all_projects() {
    log "Running all DIFS_OC projects..."
    run_project_v1
    run_project_v2
    run_project_v3
    run_project_v4
    run_project_v5
    log "All projects completed!"
}

show_help() {
    echo "===================================="
    echo "     DIFS_OC Projects Help"
    echo "===================================="
    echo "This script provides an interactive menu for managing all DIFS_OC projects."
    echo ""
    echo "Options:"
    echo "1. DIFS_OC_v1 - Legacy project"
    echo "2. DIFS_OC_v2 - SLO benchmarking"
    echo "3. DIFS_OC_v3 - SLO benchmarking (updated)"
    echo "4. DIFS_OC_v4 - Baseline comparison"
    echo "5. DIFS_OC_v5 - Switching modes"
    echo "6. All Projects - Run all projects sequentially"
    echo ""
    echo "===================================="
}

# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------
while true; do
    show_menu
    case "${choice:-}" in
        1) run_project_v1; read -p "Press Enter to continue...";;
        2) run_project_v2; read -p "Press Enter to continue...";;
        3) run_project_v3; read -p "Press Enter to continue...";;
        4) run_project_v4; read -p "Press Enter to continue...";;
        5) run_project_v5; read -p "Press Enter to continue...";;
        6) run_all_projects; read -p "Press Enter to continue...";;
        7) show_help; read -p "Press Enter to continue...";;
        8) log "Exiting script. Goodbye!"; exit 0;;
        *) warn "Invalid option. Please select 1-8."; read -p "Press Enter to continue...";;
    esac
done
