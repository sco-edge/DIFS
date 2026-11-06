#!/usr/bin/env bash

# manage_projects.sh
#
# Unified management script for all DIFS_OC projects (v1 through v5)
# Provides interactive menu to run experiments across all 5 projects
# This allows switching between different versions of the same project
# to compare their behaviors and outputs.

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Logging / helpers  
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Project-specific functions
# ---------------------------------------------------------------------------

run_project_v1() {
    log "Running DIFS_OC_v1 project (legacy with simulated benchmark)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v1/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v1 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v1"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

run_project_v2() {
    log "Running DIFS_OC_v2 project (SLO benchmarking)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v2/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v2 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v2"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

run_project_v3() {
    log "Running DIFS_OC_v3 project (SLO benchmarking)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v3/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v3 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v3"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

run_project_v4() {
    log "Running DIFS_OC_v4 project (baseline comparison)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v4/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v4 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v4"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

run_project_v5() {
    log "Running DIFS_OC_v5 project (switching modes)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v5/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v5 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v5"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

run_project_v6() {
    log "Running DIFS_OC_v6 project (current iteration)..."
    cd /home/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v6/DiffServ_0112_1730_coco2/0112_1730_coco || die "Failed to navigate to DIFS_OC_v6 directory"
    
    if [ -f "run_all_experiments.sh" ]; then
        bash run_all_experiments.sh
    else
        log "No run script found in DIFS_OC_v6"
    fi
    
    cd - > /dev/null || die "Failed to return to previous directory"
}

# ---------------------------------------------------------------------------
# Display interactive menu
# ---------------------------------------------------------------------------
show_menu() {
    clear
    echo "===================================="
    echo "   DIFS_OC Projects Manager"
    echo "===================================="
    echo "Available projects:"
    echo "1. DIFS_OC_v1 - Legacy project (simulated)"
    echo "2. DIFS_OC_v2 - SLO benchmarking"
    echo "3. DIFS_OC_v3 - SLO benchmarking (updated)"
    echo "4. DIFS_OC_v4 - Baseline comparison"
    echo "5. DIFS_OC_v5 - Switching modes"
    echo "6. DIFS_OC_v6 - Current project"
    echo "7. All Projects - Run all projects (v1 through v6)"
    echo "8. Help/Information"
    echo "9. Exit"
    echo "===================================="
    read -p "Please select an option (1-9): " choice
    echo ""
}

# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------
while true; do
    show_menu
    case $choice in
        1)
            run_project_v1
            read -p "Press Enter to continue..."
            ;;
        2)
            run_project_v2
            read -p "Press Enter to continue..."
            ;;
        3)
            run_project_v3
            read -p "Press Enter to continue..."
            ;;
        4)
            run_project_v4
            read -p "Press Enter to continue..."
            ;;
        5)
            run_project_v5
            read -p "Press Enter to continue..."
            ;;
        6)
            run_project_v6
            read -p "Press Enter to continue..."
            ;;
        7)
            log "Running all DIFS_OC projects (v1 through v6)..."
            run_project_v1
            run_project_v2
            run_project_v3
            run_project_v4
            run_project_v5
            run_project_v6
            log "All projects completed!"
            read -p "Press Enter to continue..."
            ;;
        8)
            echo "===================================="
            echo "     DIFS_OC Projects Help"
            echo "===================================="
            echo "This script provides an interactive menu for managing all DIFS_OC projects."
            echo ""
            echo "Projects:"
            echo "- DIFS_OC_v1: Legacy project with simulated benchmark"  
            echo "- DIFS_OC_v2: SLO benchmarking implementation"
            echo "- DIFS_OC_v3: Updated SLO benchmarking"
            echo "- DIFS_OC_v4: Baseline comparison"
            echo "- DIFS_OC_v5: Switching modes investigation"
            echo "- DIFS_OC_v6: Current project iteration"
            echo ""
            echo "Requirements: All projects must be present in their respective directories."
            echo "Each project requires:"
            echo "- real_stats.npz (33 MB) in the project root"  
            echo "- Conda environment 'DIFS_Gen' with Python 3.10"
            echo "- HF_TOKEN for gated model access when needed"
            echo "===================================="
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