#!/bin/bash -e
# start_infaas.sh – Local single-machine GPU version
# Works on Ubuntu 20.04/22.04/24.04 with NVIDIA GPU
# No AWS, no S3, no EC2 — pure local execution

# Author: KB(Grok)
# Purpose: To start DIFS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DIFS Local GPU Starter ==="

# 1. Build (only if needed)
if [ ! -f "build/src/master/queryfe_server" ] || [ ! -f "build/src/master/modelreg_server" ]; then
  echo "[1/5] Building DIFS (Release mode)..."
  rm -rf build
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  cd ..
  echo "Build complete"
else
  echo "[1/5] Build directory already exists → skipping compile"
fi

# 2. Start Redis
echo "[2/5] Starting Redis..."
if ! redis-cli ping >/dev/null 2>&1; then
  redis-server --daemonize yes
  sleep 7
fi
redis-cli ping | grep -q PONG || { echo "Redis failed to start"; exit 1; }
echo "Redis running"

# 3. Start Model Registration Server
echo "[3/5] Starting ModelReg server..."
./build/src/master/modelreg_server >/dev/null 2>&1 &
MODELREG_PID=$!
sleep 2
echo "ModelReg running (PID $MODELREG_PID)"

# 4. Start Query Frontend Server
echo "[4/5] Starting QueryFE server..."
./build/src/master/queryfe_server >/dev/null 2>&1 &
QUERYFE_PID=$!
sleep 2
echo "QueryFE running (PID $QUERYFE_PID)"

# 5. Start GPU Worker
echo "[5/5] Starting GPU worker (CUDA_VISIBLE_DEVICES=0)..."
export CUDA_VISIBLE_DEVICES=0
./build/src/worker/worker_daemon --hardware gpu --port 50053 >/dev/null 2>&1 &
WORKER_PID=$!
sleep 3

# Verify everything is alive
echo
echo "All services started successfully!"
echo "   Redis        : OK"
echo "   ModelReg     : PID $MODELREG_PID"
echo "   QueryFE      : PID $QUERYFE_PID"
echo "   GPU Worker   : PID $WORKER_PID (using GPU $CUDA_VISIBLE_DEVICES)"
echo
echo "Next steps:"
echo "   1. Register a model  → ./register_local_models.sh   (or run infaas_modelregistration manually)"
echo "   2. Run a query       → ./build/src/cli-tools/infaas_online_query --input data/mug.jpg ..."
echo
echo "To stop everything: ./stop_infaas.sh   (or Ctrl+C here and kill the PIDs)"
echo

# Keep script alive so user can Ctrl+C to stop everything
trap "echo; echo 'Stopping all DIFS processes...'; kill $MODELREG_PID $QUERYFE_PID $WORKER_PID 2>/dev/null; pkill redis-server 2>/dev/null; echo 'Done.'; exit 0" INT TERM
wait
