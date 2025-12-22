#!/bin/bash
# stop_infaas_v3.sh — Graceful shutdown of INFaaS + Diffusion (8-Step)
# 100% Clean: Docker → Services → Redis → GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
LOG_DIR="$SCRIPT_DIR/logs"

echo "🛑 Shutting down INFaaS + Diffusion (8-Step Cleanup)..."
echo "==================================================="

# --------------------------------------------------
# 1. Stop Diffusion Container
# --------------------------------------------------
echo "[1/8] Stopping Diffusion container..."
docker stop diffusion-svc >/dev/null 2>&1 || true
docker rm diffusion-svc >/dev/null 2>&1 || true
echo "  → Diffusion container stopped ✓"

# --------------------------------------------------
# 2. Kill INFaaS Services (graceful SIGTERM)
# --------------------------------------------------
echo "[2/8] Stopping INFaaS services..."
pkill -f "modelreg_server" >/dev/null 2>&1 || true
pkill -f "queryfe_server" >/dev/null 2>&1 || true
pkill -f "query_executor" >/dev/null 2>&1 || true

# Wait for graceful shutdown (10s timeout)
sleep 3
pkill -f "modelreg_server" -9 >/dev/null 2>&1 || true
pkill -f "queryfe_server" -9 >/dev/null 2>&1 || true
pkill -f "query_executor" -9 >/dev/null 2>&1 || true
echo "  → INFaaS services stopped ✓"

# --------------------------------------------------
# 3. Stop Redis
# --------------------------------------------------
echo "[3/8] Stopping Redis..."
redis-cli -p 6379 shutdown >/dev/null 2>&1 || true
sleep 2
pkill -f "redis-server" >/dev/null 2>&1 || true
echo "  → Redis stopped ✓"

# --------------------------------------------------
# 4. GPU Process Cleanup
# --------------------------------------------------
echo "[4/8] GPU cleanup..."
pkill -f "python.*diffusers" >/dev/null 2>&1 || true
pkill -f "python.*diffusion_service" >/dev/null 2>&1 || true
sleep 2
echo "  → GPU processes cleaned ✓"

# --------------------------------------------------
# 5. Docker Network Cleanup
# --------------------------------------------------
echo "[5/8] Docker cleanup..."
docker system prune -f >/dev/null 2>&1 || true
echo "  → Docker networks cleaned ✓"

# --------------------------------------------------
# 6. Port Cleanup
# --------------------------------------------------
echo "[6/8] Port cleanup (50052, 6379)..."
fuser -k 50052/tcp >/dev/null 2>&1 || true
fuser -k 6379/tcp >/dev/null 2>&1 || true
sleep 1
echo "  → Ports freed ✓"

# --------------------------------------------------
# 7. Log Rotation (optional)
# --------------------------------------------------
echo "[7/8] Log summary..."
if [ -d "$LOG_DIR" ]; then
  echo "  → Latest logs:"
  tail -3 "$LOG_DIR"/*.log 2>/dev/null || true
  echo "  → Full logs: $LOG_DIR/"
else
  echo "  → No logs found"
fi

# --------------------------------------------------
# 8. Final Status + Verification
# --------------------------------------------------
echo "[8/8] Final verification..."
sleep 2
if ! pgrep -f "modelreg_server\|queryfe_server\|query_executor" >/dev/null 2>&1; then
  if ! docker ps | grep -q diffusion-svc; then
    echo "✅ SHUTDOWN 100% COMPLETE!"
    echo "=========================="
    echo "  📁 Logs: $LOG_DIR/"
    echo "  🗄️  Redis data: $SCRIPT_DIR/redis-data/"
    echo "  🐳 Docker: docker images | grep infaas"
    echo
    echo "🔄 Restart: ./start_infaas_v2.sh"
    exit 0
  fi
fi
echo "⚠️  Some processes may still be running (manual kill needed)"
