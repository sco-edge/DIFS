#!/bin/bash

# Author: KB
# Date: 2026. 04. 17. (금) 11:22:10 KST
# Purpose: for monitoring the system resources being consumed when 'run_multiple_clients.sh' is run

# 1. Configuration - Set your custom name or use default
SESSION_NAME=${1:-grpc_test}
LOG_DIR="WorkersLogs"

# 2. Check for dependencies
for pkg in btop nvtop tmux; do
    if ! command -v $pkg &> /dev/null; then
        echo "Installing $pkg... (Sudo password may be required)"
        sudo apt update && sudo apt install -y $pkg
    fi
done

# 3. Clean up existing session if it's already running (Fresh Start)
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Existing session '$SESSION_NAME' found. Killing it for a fresh run..."
    tmux kill-session -t "$SESSION_NAME"
    sleep 1
fi

# 4. Prepare Environment
mkdir -p "$LOG_DIR"

# 5. Create the Session (Detached)
tmux new-session -d -s "$SESSION_NAME"

# --- PANE 0: The Worker Loop (Left Side) ---
tmux send-keys -t "$SESSION_NAME.0" "
for i in {1..10}; do
   echo 'Starting instance \$i...'
   python src/client_async_v3.py -query Q3 -steps 20 -sample 3 -port 50050 > '$LOG_DIR/worker_\$i.log' 2>&1 &
   sleep 0.5
done
wait
echo 'All workers finished. Check $LOG_DIR/ for reports.'" C-m

# --- PANE 1: System Monitor (Top Right) ---
tmux split-window -h -t "$SESSION_NAME.0"
tmux send-keys -t "$SESSION_NAME.1" "btop" C-m

# --- PANE 2: GPU Monitor (Bottom Right) ---
tmux split-window -v -t "$SESSION_NAME.1"
tmux send-keys -t "$SESSION_NAME.2" "nvtop" C-m

# 6. Final Layout Adjustment & Attach
# Make the worker pane (left) slightly larger
tmux resize-pane -t "$SESSION_NAME.0" -L 10 
tmux attach-session -t "$SESSION_NAME"
