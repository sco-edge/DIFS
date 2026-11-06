#!/bin/bash
# watch_gpu.sh - poll the RTX 6000 until it is free enough for a clean
# benchmark run, then (optionally) run a command.
# Usage:
#   ./watch_gpu.sh [--free-mb 30000] [--max-util 50] [--interval 60] [--command "CMD..."]
# Defaults: free >= 30000 MiB AND util <= 50%, poll every 60 s,
#           log to /tmp/opencode/gpu_wait.log
set -u

FREE_MB=30000
MAX_UTIL=50
INTERVAL=60
COMMAND=""
LOG=/tmp/opencode/gpu_wait.log

while [[ $# -gt 0 ]]; do
    case "$1" in
        --free-mb) FREE_MB="$2"; shift 2;;
        --max-util) MAX_UTIL="$2"; shift 2;;
        --interval) INTERVAL="$2"; shift 2;;
        --command) shift; COMMAND="$*"; break;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

mkdir -p "$(dirname "$LOG")"
echo "[$(date +%F\ %T)] watch_gpu.sh started (need >= ${FREE_MB} MiB free AND util <= ${MAX_UTIL}%, poll ${INTERVAL}s)" | tee -a "$LOG"

while true; do
    read -r UTIL MEM_USED MEM_TOTAL <<< "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | sed 's/,/ /g')"
    FREE=$(( MEM_TOTAL - MEM_USED ))
    APPS=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null | tr '\n' '; ')
    echo "[$(date +%F\ %T)] util=${UTIL}% mem_used=${MEM_USED}/${MEM_TOTAL} MiB free=${FREE} MiB | ${APPS}" | tee -a "$LOG"
    if [ "$FREE" -ge "$FREE_MB" ] && [ "$UTIL" -le "$MAX_UTIL" ]; then
        echo "[$(date +%F\ %T)] GPU FREE (${FREE} MiB >= ${FREE_MB} MiB, util ${UTIL}% <= ${MAX_UTIL}%)." | tee -a "$LOG"
        break
    fi
    sleep "$INTERVAL"
done

if [ -n "$COMMAND" ]; then
    echo "[$(date +%F\ %T)] Running: $COMMAND" | tee -a "$LOG"
    eval "$COMMAND"
fi
