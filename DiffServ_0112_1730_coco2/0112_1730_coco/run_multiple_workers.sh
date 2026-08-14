#!/bin/bash

# Default to 10 clients if no argument is provided
NUM_CLIENTS=${1:-3}
LOG_DIR="WorkersLogs"

# Ensure log directory exists and is clean
mkdir -p "$LOG_DIR"

echo "Launching $NUM_CLIENTS clients..."

for i in $(seq 1 $NUM_CLIENTS); do
   echo "Starting instance $i of $NUM_CLIENTS..."
   python src/client_async_v3.py -prompt_count 5 -steps 20 -sampler 1 -port 50050 > "$LOG_DIR/worker_$i.log" 2>&1 &
   sleep 0.5
done

echo "All $NUM_CLIENTS clients are running in the background."
echo "Use './monitor_system.sh' to view resource usage."
wait
echo "All clients have finished."
