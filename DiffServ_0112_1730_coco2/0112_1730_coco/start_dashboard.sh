#!/usr/bin/env bash
# Start the updated DIFS_OC_v7 tracker GUI dashboard and populate from latest results
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src"
LIVE_RESULTS_DIR="$PROJECT_ROOT/results/model_benchmark_live"
TRACK_PORT="${TRACK_PORT:-8080}"
PYTHON_BIN="${DIFS_PYTHON_BIN:-$(command -v python3)}"

echo "[DASH] Project root: $PROJECT_ROOT"
echo "[DASH] Populating dashboard from completed results..."
if [ -f "$PROJECT_ROOT/src/tracking/populate_from_csv.py" ]; then
  DIFS_LIVE_RESULTS_DIR="$LIVE_RESULTS_DIR" "$PYTHON_BIN" "$PROJECT_ROOT/src/tracking/populate_from_csv.py" --project-root "$PROJECT_ROOT" 2>&1 | tail -4
else
  echo "[DASH] WARN: populate_from_csv.py not found"
fi

echo "[DASH] Freeing port $TRACK_PORT..."
pkill -f "tracking.tracker_gui_server.*$TRACK_PORT" 2>/dev/null || true
pkill -f "tracking.dash_app.*$TRACK_PORT" 2>/dev/null || true
fuser -k ${TRACK_PORT}/tcp 2>/dev/null || true
sleep 2

echo "[DASH] Starting tracker GUI on http://127.0.0.1:$TRACK_PORT"
mkdir -p "$PROJECT_ROOT/results/model_benchmark/Reports"
PYTHONPATH="$PROJECT_ROOT/src" nohup "$PYTHON_BIN" -m tracking.tracker_gui_server \
  --project-root "$PROJECT_ROOT" --port "$TRACK_PORT" \
  > "$PROJECT_ROOT/results/model_benchmark/Reports/tracker_gui.log" 2>&1 &
echo "[DASH] Started. Waiting for server..."
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fsS --max-time 2 "http://127.0.0.1:$TRACK_PORT/api/metrics" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -s "http://127.0.0.1:$TRACK_PORT/api/metrics" | "$PYTHON_BIN" -c "import json,sys; d=json.load(sys.stdin); print('[DASH] history:', len(d.get('history',[])), 'current keys:', list(d.get('current',{}).keys()))" 2>&1 | tail -3
echo "[DASH] Open http://127.0.0.1:$TRACK_PORT"
