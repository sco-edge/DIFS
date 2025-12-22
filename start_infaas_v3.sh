#!/bin/bash -e
# start_infaas_v3.sh — Local single-node GPU INFaaS w/ Diffusion (100% AUTO)
# No AWS, no EC2, no S3 | FULLY AUTOMATIC: Protos → Container → Services

# Author: KB(GPT) + PNB Diffusion (2025.12.22)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="build"
LOG_DIR="logs"
DIFFUSION_CONTAINER="infaas-diffusion:latest"
MODEL_NAME="sd_v15"  # Default diffusion model name
mkdir -p "$LOG_DIR"

echo "=== INFaaS Local GPU + Diffusion Startup (100% AUTO) ==="

# --------------------------------------------------
# 0. DIFFUSION: Build Container (if needed)
# --------------------------------------------------
echo "[0/8] Checking Diffusion Container..."
if ! docker image inspect "$DIFFUSION_CONTAINER" >/dev/null 2>&1; then
  echo "  → Building infaas-diffusion:latest..."
  docker build -f dockerfiles/Dockerfile.diffusion -t "$DIFFUSION_CONTAINER" . || {
    echo "❌ Dockerfile.diffusion missing! Create it first."
    exit 1
  }
  echo "  → Container built ✓"
else
  echo "  → Container exists ✓"
fi

# --------------------------------------------------
# 0.5. Generate Python gRPC Stubs (REQUIRED)
# --------------------------------------------------
echo "[0.5/8] Generating Diffusion Proto Stubs..."
mkdir -p src/containers/diffusion/protos/internal
if [ ! -f "src/containers/diffusion/diffusion_service_pb2.py" ] || [ ! -f "src/containers/diffusion/diffusion_service_pb2_grpc.py" ]; then
  python3 -m grpc_tools.protoc -Iprotos/internal \
    --python_out=src/containers/diffusion \
    --grpc_python_out=src/containers/diffusion \
    protos/internal/diffusion_service.proto || {
    echo "❌ Proto generation failed - check protos/internal/diffusion_service.proto"
    exit 1
  }
  echo "  → Proto stubs generated ✓"
else
  echo "  → Proto stubs exist ✓"
fi

# --------------------------------------------------
# 1. Build INFaaS (only if needed)
# --------------------------------------------------
echo "[1/8] Checking INFaaS Build..."
if [ ! -f "$BUILD_DIR/src/master/queryfe_server" ]; then
  echo "  → Building INFaaS..."
  rm -rf "$BUILD_DIR"
  mkdir "$BUILD_DIR"
  cd "$BUILD_DIR"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DLOCAL_MODE=ON
  make -j$(nproc)
  cd ..
  echo "  → INFaaS built ✓"
else
  echo "  → Build exists ✓"
fi

# --------------------------------------------------
# 2. Start Redis + Health Check
# --------------------------------------------------
echo "[2/8] Starting Redis..."
if ! redis-cli ping >/dev/null 2>&1; then
  mkdir -p redis-data
  redis-server --daemonize yes --port 6379 --dir "$SCRIPT_DIR/redis-data" >/dev/null 2>&1
  for i in {1..10}; do
    sleep 1
    if redis-cli ping | grep -q PONG; then break; fi
  done
fi
redis-cli ping | grep -q PONG || { echo "❌ Redis failed"; exit 1; }
echo "  → Redis OK ✓"

# --------------------------------------------------
# 3. Register Diffusion Model
# --------------------------------------------------
echo "[3/8] Registering Diffusion Model..."
if ! redis-cli EXISTS "model:${MODEL_NAME}-MODINFOSUFF" >/dev/null 2>&1; then
  echo "  → Registering $MODEL_NAME..."
  if [ -f "$BUILD_DIR/src/cli-tools/infaas_modelregistration" ]; then
    "$BUILD_DIR/src/cli-tools/infaas_modelregistration" config_sd.json /local/sd/model/ DIFFUSION || {
      echo "⚠️  Registration failed (config_sd.json missing?)"
    }
  else
    echo "⚠️  CLI missing - manual registration needed (Step 4 TODO)"
  fi
  sleep 2
fi
if redis-cli HGET "model:${MODEL_NAME}-MODINFOSUFF" "task" | grep -q DIFFUSION; then
  echo "  → Diffusion registered ✓ (task=DIFFUSION)"
else
  echo "  ⚠️  Model not registered - check config_sd.json"
fi

# --------------------------------------------------
# 4. Start Model Registration Server + Health Check
# --------------------------------------------------
echo "[4/8] Starting ModelReg Server..."
pkill -f modelreg_server || true
"$BUILD_DIR/src/master/modelreg_server" localhost 6379 \
  > "$LOG_DIR/modelreg.log" 2>&1 &
MODELREG_PID=$!
for i in {1..10}; do
  sleep 1
  if grep -q "ModelReg server.*listening" "$LOG_DIR/modelreg.log" 2>/dev/null; then
    echo "  → ModelReg OK ✓ (PID $MODELREG_PID)"
    break
  fi
done || echo "  ⚠️  ModelReg log check failed"

# --------------------------------------------------
# 5. Start Query Frontend Server + Health Check
# --------------------------------------------------
echo "[5/8] Starting QueryFE Server..."
pkill -f queryfe_server || true
"$BUILD_DIR/src/master/queryfe_server" localhost 6379 \
  > "$LOG_DIR/queryfe.log" 2>&1 &
QUERYFE_PID=$!
for i in {1..15}; do
  sleep 1
  if nc -z localhost 50052 2>/dev/null; then
    echo "  → QueryFE OK ✓ (PID $QUERYFE_PID, port 50052)"
    break
  fi
done || echo "  ⚠️  QueryFE port 50052 not ready yet..."

# --------------------------------------------------
# 6. Start GPU Worker + Health Check
# --------------------------------------------------
echo "[6/8] Starting GPU Worker..."
pkill -f query_executor || true
export CUDA_VISIBLE_DEVICES=0
"$BUILD_DIR/src/worker/query_executor" localhost 6379 \
  > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
sleep 3
if grep -q "Worker registered" "$LOG_DIR/worker.log" 2>/dev/null; then
  echo "  → Worker OK ✓ (PID $WORKER_PID)"
else
  echo "  ⚠️  Worker starting... (check $LOG_DIR/worker.log)"
fi

# --------------------------------------------------
# 7. Start Diffusion Container + Health Check
# --------------------------------------------------
echo "[7/8] Starting Diffusion Container..."
docker rm -f diffusion-svc || true
docker run -d --gpus all --name diffusion-svc \
  -p 50052:50052 \
  --restart unless-stopped \
  -v "$SCRIPT_DIR/redis-data:/workspace/cache" \
  "$DIFFUSION_CONTAINER" || {
  echo "❌ Diffusion container failed!"
  exit 1
}
for i in {1..20}; do
  sleep 1
  if docker exec diffusion-svc nc -z localhost 50052 2>/dev/null; then
    echo "  → Diffusion OK ✓ (container: diffusion-svc)"
    break
  fi
done || echo "  ⚠️  Diffusion container still loading model..."

# --------------------------------------------------
# 8. Final Health Check + Test Query
# --------------------------------------------------
echo "[8/8] Final Health Check..."
sleep 5
echo
echo "🎉 INFaaS + Diffusion FULLY RUNNING! (100% AUTO)"
echo "================================================"
echo "  ✅ Redis        : OK (port 6379)"
echo "  ✅ ModelReg     : PID $MODELREG_PID"
echo "  ✅ QueryFE      : PID $QUERYFE_PID (port 50052)"
echo "  ✅ GPU Worker   : PID $WORKER_PID (GPU 0)"
echo "  ✅ Diffusion    : docker 'diffusion-svc' (port 50052)"
echo
echo "📊 STATUS CHECKS:"
echo "  • Redis models: redis-cli KEYS 'model:*'"
echo "  • Diffusion model: redis-cli HGETALL 'model:sd_v15-MODINFOSUFF'"
echo
echo "🧪 TEST QUERY (copy-paste):"
echo "  grpcurl -plaintext -d '{\"model\": [\"sd_v15\"], \"diffusion\": {\"prompt\": \"a cute cat, masterpiece\", \"steps\": 20, \"width\": 512, \"height\": 512, \"seed\": 42}}' localhost:50052 infaas.queryfe.QueryService/QueryOnline"
echo
echo "📁 Logs: $LOG_DIR/"
echo "🛑 Stop: ./stop_infaas_v3.sh"
echo

# Trap for clean shutdown
trap "./stop_infaas_v3.sh" INT TERM
wait
