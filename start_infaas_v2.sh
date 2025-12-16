#!/bin/bash -e
# start_infaas.sh — Local single-node GPU INFaaS
# No AWS, no EC2, no S3

# Author: KB(GPT)
# Purpose: To start DIFS


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="build"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== INFaaS Local GPU Startup ==="

# --------------------------------------------------
# 1. Build (only if needed)
# --------------------------------------------------
if [ ! -f "$BUILD_DIR/src/master/queryfe_server" ]; then
  echo "[1/5] Building INFaaS..."
  rm -rf "$BUILD_DIR"
  mkdir "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  cd ..
else
  echo "[1/5] Build exists → skipping compile"
fi

# --------------------------------------------------
# 2. Start Redis
# --------------------------------------------------
echo "[2/5] Starting Redis..."
if ! redis-cli ping >/dev/null 2>&1; then
  redis-server --daemonize yes
  sleep 5
fi
redis-cli ping | grep -q PONG || { echo "Redis failed"; exit 1; }
echo "Redis OK"

# --------------------------------------------------
# 3. Start Model Registration Server
# --------------------------------------------------
echo "[3/5] Starting ModelReg..."
"$BUILD_DIR/src/master/modelreg_server" \
  > "$LOG_DIR/modelreg.log" 2>&1 &
MODELREG_PID=$!
sleep 2

# --------------------------------------------------
# 4. Start Query Frontend Server
# --------------------------------------------------
echo "[4/5] Starting QueryFE..."
"$BUILD_DIR/src/master/queryfe_server" \
  > "$LOG_DIR/queryfe.log" 2>&1 &
QUERYFE_PID=$!
sleep 2

# --------------------------------------------------
# 5. Start GPU Worker
# --------------------------------------------------
echo "[5/5] Starting GPU Worker..."
export CUDA_VISIBLE_DEVICES=0
"$BUILD_DIR/src/worker/query_executor" \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
sleep 3

# --------------------------------------------------
# Health summary
# --------------------------------------------------
echo
echo "INFaaS running:"
echo "  Redis        : OK"
echo "  ModelReg     : PID $MODELREG_PID"
echo "  QueryFE      : PID $QUERYFE_PID"
echo "  Worker       : PID $WORKER_PID (GPU $CUDA_VISIBLE_DEVICES)"
echo
echo "Logs: $LOG_DIR/"
echo "Stop with: ./stop_infaas.sh"
echo

trap "./stop_infaas.sh" INT TERM
wait
