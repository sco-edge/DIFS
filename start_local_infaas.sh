#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO_ROOT/build"

export INFAAS_SINGLE_NODE=1

echo "[infaas] Building INFaaS..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
make -j$(nproc)
cd "$REPO_ROOT"

echo "[infaas] Starting Redis..."
redis-server "$REPO_ROOT/src/metadata-store/redis-serv.conf" &
sleep 0.5

echo "[infaas] Starting modelreg_server..."
"$BUILD_DIR/bin/modelreg_server" --port=50051 &

echo "[infaas] Starting queryfe_server..."
"$BUILD_DIR/bin/queryfe_server" --port=50052 &

echo "[infaas] Starting local worker..."
"$REPO_ROOT/src/worker/start_worker.sh" gpu 127.0.0.1 50051 &
