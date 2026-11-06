"""Standalone tracker GUI that follows the shared metrics JSON file.

The benchmark process updates ``metrics/realtime/performance_metrics.json``
while this process serves the dashboard.  Keeping the GUI in its own process
means the dashboard stays available for the whole runner session, including
switching, plotting, and report stages.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from tracking.realtime_tracker import RealTimeTracker


def _apply(tracker: RealTimeTracker, payload: str) -> None:
    data = json.loads(payload)
    with tracker._lock:
        tracker.baseline_metrics = data.get("baseline", {})
        tracker.current_metrics = data.get("current", {})
        tracker.history = data.get("history", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="DIFS_OC_v7 tracker GUI")
    parser.add_argument("--project-root", default=".", help="v7 project root")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    tracker = RealTimeTracker(root)
    last_payload = None
    if tracker.metrics_file.exists():
        last_payload = tracker.metrics_file.read_text()
        try:
            _apply(tracker, last_payload)
        except Exception:
            last_payload = None

    tracker.start_gui_server(port=args.port, host=args.host)
    print(
        f"[TRACKER] GUI dashboard: http://{args.host}:{args.port} "
        f"(following {tracker.metrics_file})",
        flush=True,
    )

    running = True

    def _stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while running:
        try:
            if tracker.metrics_file.exists():
                payload = tracker.metrics_file.read_text()
                if payload != last_payload:
                    _apply(tracker, payload)
                    last_payload = payload
                    print("[TRACKER] reloaded metrics", flush=True)
        except Exception as exc:
            print(f"[TRACKER] reload error: {exc}", flush=True)
        time.sleep(1)

    tracker.stop_gui_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
