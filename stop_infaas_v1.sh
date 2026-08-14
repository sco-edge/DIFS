#!/bin/bash
# stop_infaas.sh — Stops local DIFS services with PID confirmation
# Provides feedback on which processes were running and killed

# Author: KB(Grok)
# Purpose: To stop DIFS

echo -e "\033[1;34m==> Checking running DIFS processes...\033[0m"

# Function to find and kill a process with PID feedback
kill_with_pid() {
    local pattern="$1"
    local name="$2"
    
    # Find PIDs matching the pattern
    local pids=$(pgrep -f "$pattern")
    
    if [ -n "$pids" ]; then
        echo -e "\033[1;33m   • $name running (PIDs: $pids) → stopping...\033[0m"
        kill $pids 2>/dev/null || kill -9 $pids 2>/dev/null
        echo -e "\033[1;32m     ✔ $name stopped (PIDs: $pids)\033[0m"
    else
        echo -e "\033[0;36m   • $name not running\033[0m"
    fi
}

# Stop core services
kill_with_pid "modelreg_server"     "Model Registration Server"
kill_with_pid "queryfe_server"      "Query Frontend Server"
kill_with_pid "worker_daemon"       "Worker Daemon"
kill_with_pid "gpu_daemon"          "GPU Daemon"

# Stop Redis
kill_with_pid "redis-server"        "Redis Server"

# Optional: Stop any lingering CLI or test processes
kill_with_pid "infaas_online_query" "Online Query CLI"
kill_with_pid "infaas_modelregistration" "Model Registration CLI"
kill_with_pid "redis_md_test"       "Redis Metadata Test"

echo -e "\033[1;32m\nAll local DIFS services have been stopped.\033[0m"
echo -e "\033[1;32mYou can now safely restart with: ./start_infaas.sh\033[0m"
