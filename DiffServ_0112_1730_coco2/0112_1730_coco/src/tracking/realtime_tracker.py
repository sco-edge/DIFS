"""
Real-time performance tracker for DIFS_OC_v7.

Two interfaces share the same JSON state file:
  * CLI dashboard: ``display_cli_dashboard()``
  * HTTP GUI:      ``start_gui_server(port)`` then open http://localhost:<port>

The GUI uses only the Python standard library (no Flask/Plotly/Internet
dependency), so it works on an offline GPU host.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_TARGETS = {
    "slo_violation_rate": 0.05,
    "cold_start_latency": 0.50,
    "hot_start_latency": 0.50,
    "error_to_normal_ratio": 0.30,
}

# Metrics where a lower value is better.
LOWER_IS_BETTER = {
    "slo_violation_rate",
    "cold_start_latency",
    "hot_start_latency",
    "error_to_normal_ratio",
}


class RealTimeTracker:
    def __init__(self, project_root, history_limit: int = 600):
        self.project_root = Path(project_root)
        self.metrics_file = self.project_root / "metrics" / "realtime" / "performance_metrics.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.baseline_metrics: Dict[str, Any] = {}
        self.current_metrics: Dict[str, Any] = {}
        self.targets: Dict[str, float] = dict(DEFAULT_TARGETS)
        self.history_limit = max(10, int(history_limit))
        self.history = []

        self._lock = threading.RLock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def update_metrics(self, metrics: Dict[str, Any]):
        """Merge new metrics, timestamp them, and persist atomically."""
        with self._lock:
            self.current_metrics.update(metrics)
            self.current_metrics["timestamp"] = datetime.now().isoformat()
            self.current_metrics["elapsed_time"] = round(time.time() - self.start_time, 3)
            self.history.append(
                {
                    "timestamp": self.current_metrics["timestamp"],
                    "elapsed_time": self.current_metrics["elapsed_time"],
                    "metrics": {
                        k: v
                        for k, v in self.current_metrics.items()
                        if isinstance(v, (int, float))
                    },
                }
            )
            if len(self.history) > self.history_limit:
                self.history = self.history[-self.history_limit :]
            self._save_metrics()

    def compare_with_baseline(self) -> Dict[str, float]:
        """Return percent improvement (positive = better than baseline)."""
        with self._lock:
            if not self.baseline_metrics:
                return {}
            comparison: Dict[str, float] = {}
            for key, current in self.current_metrics.items():
                if key not in self.baseline_metrics:
                    continue
                baseline = self.baseline_metrics[key]
                if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
                    continue
                if baseline == 0:
                    continue
                if key in LOWER_IS_BETTER:
                    improvement = ((baseline - current) / baseline) * 100.0
                else:
                    improvement = ((current - baseline) / baseline) * 100.0
                comparison[key] = round(improvement, 3)
            return comparison

    def objective_status(self) -> Dict[str, Dict[str, Any]]:
        """Compare improvements against the v7 targets."""
        improvements = self.compare_with_baseline()
        result = {}
        for key, target in self.targets.items():
            pct = improvements.get(key)
            result[key] = {
                "target_pct": round(target * 100.0, 2),
                "actual_pct": pct,
                "met": bool(pct is not None and pct >= target * 100.0),
            }
        return result

    def reset(self) -> None:
        """Clear all tracked metrics, history, and baseline. Persists a minimal state."""
        with self._lock:
            self.baseline_metrics = {}
            self.current_metrics = {}
            self.history = []
            self._save_metrics()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "baseline": dict(self.baseline_metrics),
                "current": dict(self.current_metrics),
                "comparison": self.compare_with_baseline(),
                "objectives": self.objective_status(),
                "history": list(self.history),
                "metrics_file": str(self.metrics_file),
            }

    def _save_metrics(self):
        data = {
            "baseline": self.baseline_metrics,
            "current": self.current_metrics,
            "comparison": self.compare_with_baseline(),
            "objectives": self.objective_status(),
            "history": self.history,
        }
        tmp = self.metrics_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.metrics_file)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @staticmethod
    def _bar(fraction: float, width: int = 24) -> str:
        fraction = max(0.0, min(1.0, float(fraction)))
        filled = int(round(fraction * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def display_cli_dashboard(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            current = {
                k: v
                for k, v in self.current_metrics.items()
                if k not in {"timestamp", "elapsed_time"}
            }
            comparison = self.compare_with_baseline()
            objectives = self.objective_status()

        print()
        print("=" * 68)
        print(" DIFS_OC_v7 REAL-TIME PERFORMANCE TRACKER")
        print("=" * 68)
        print(f" Elapsed: {elapsed:.1f}s   State: {self.metrics_file}")
        print("-" * 68)
        if not current:
            print(" No metrics recorded yet.")
        for key, value in current.items():
            if isinstance(value, float):
                print(f" {key:<28} {value:.4f}")
            else:
                print(f" {key:<28} {value}")
        print("-" * 68)
        print(" Objectives (positive improvement is better):")
        for key, info in objectives.items():
            actual = info["actual_pct"]
            if actual is None:
                print(f"  {key:<28} no baseline  target >= {info['target_pct']:.1f}%")
                continue
            progress = actual / info["target_pct"] if info["target_pct"] else 0.0
            status = "OK  " if info["met"] else "MISS"
            print(
                f"  {status} {key:<26} {actual:6.2f}% "
                f"(target {info['target_pct']:5.1f}%) {self._bar(progress)}"
            )
        print("=" * 68)

    # ------------------------------------------------------------------
    # HTTP GUI
    # ------------------------------------------------------------------
    def start_gui_server(self, port: int = 8080, host: str = "127.0.0.1"):
        """Start the GUI server in a daemon thread. Returns the HTTPServer."""
        if self._httpd is not None:
            return self._httpd

        tracker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # keep stdout clean
                return

            def _send(self, code: int, body: bytes, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in {"/", "/index.html"}:
                    self._send(200, GUI_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path.startswith("/api/metrics"):
                    payload = json.dumps(tracker.snapshot()).encode("utf-8")
                    self._send(200, payload, "application/json")
                elif self.path.startswith("/api/health"):
                    self._send(200, b'{"ok": true}', "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        httpd = ThreadingHTTPServer((host, int(port)), Handler)
        thread = threading.Thread(
            target=httpd.serve_forever, name="difsv7-tracker-gui", daemon=True
        )
        thread.start()
        self._httpd = httpd
        self._http_thread = thread
        print(f"[TRACKER] GUI dashboard: http://{host}:{port}")
        return httpd

    def stop_gui_server(self):
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._http_thread = None

    def serve_forever(self, port: int = 8080, host: str = "127.0.0.1"):
        """Blocking variant used by the CLI/GUI menu."""
        self.start_gui_server(port=port, host=host)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[TRACKER] Stopping GUI dashboard...")
        finally:
            self.stop_gui_server()


GUI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DIFS_OC_v7 Performance Tracker</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0;
         background:#0f1419; color:#d7dde5; }
  header { padding:16px 22px; background:#151b24; border-bottom:1px solid #2c3a4f; }
  h1 { margin:0; font-size:18px; letter-spacing:.5px; }
  .sub { color:#8b9bb0; font-size:12px; margin-top:4px; }
  main { padding:18px 22px; display:grid; gap:18px; max-width:1400px; margin:0 auto; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }
  .card { background:#18202c; border:1px solid #2c3a4f; border-radius:8px; padding:12px 14px; }
  .card h2 { margin:0 0 6px; font-size:11px; color:#8b9bb0; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; }
  .value { font-size:24px; font-weight:600; }
  .target { font-size:11px; color:#8b9bb0; margin-top:6px; }
  .ok { color:#4ade80; } .miss { color:#f87171; }
  .bar { height:6px; background:#0b0f15; border-radius:3px; margin-top:8px; overflow:hidden; }
  .bar > div { height:100%; background:linear-gradient(90deg,#38bdf8,#4ade80); width:0; transition:width .3s; }
  .section-title { font-size:13px; color:#8b9bb0; margin:4px 0 8px; text-transform:uppercase; letter-spacing:0.5px; }
  .grid-2x2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width: 1000px) { .grid-2x2 { grid-template-columns:1fr; } }
  .plot-card { background:#18202c; border:1px solid #2c3a4f; border-radius:8px; padding:12px; }
  .plot-card h3 { margin:0 0 8px; font-size:12px; color:#d7dde5; font-weight:600; }
  .plot-card .units { color:#8b9bb0; font-weight:400; }
  .plot-card canvas { width:100%; height:200px; background:#101722; border:1px solid #223044; border-radius:6px; }
  .legend { display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; font-size:11px; color:#8b9bb0; }
  .legend-item { display:flex; align-items:center; gap:6px; }
  .legend-dot { width:10px; height:2px; border-radius:1px; }
  table { border-collapse:collapse; width:100%; font-size:12px; background:#18202c; border:1px solid #2c3a4f; border-radius:8px; overflow:hidden; }
  th,td { text-align:left; padding:8px 12px; border-bottom:1px solid #223044; }
  th { color:#8b9bb0; font-weight:600; background:#151b24; text-transform:uppercase; font-size:11px; letter-spacing:0.5px; }
  tr:last-child td { border-bottom:none; }
  .status-ok { color:#4ade80; } .status-miss { color:#f87171; }
  .muted { color:#8b9bb0; font-size:11px; }
</style>
</head>
<body>
<header>
  <h1>DIFS_OC_v7 &mdash; Performance Tracker</h1>
  <div class="sub" id="sub">connecting...</div>
</header>
<main>
  <section>
    <div class="section-title">Objective Cards</div>
    <div class="cards" id="cards"></div>
  </section>
  
  <section>
    <div class="section-title">Performance Metrics (2x2 Grid)</div>
    <div class="grid-2x2">
      <div class="plot-card">
        <h3>SLO Violation Rate <span class="units">(% of requests)</span></h3>
        <canvas id="chart_slo_violation_rate" width="400" height="200"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#38bdf8"></div><span>Current Run</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.8); height:1px; border-style:dashed"></div><span>Baseline</span></div>
        </div>
      </div>
      <div class="plot-card">
        <h3>Cold Start Latency <span class="units">(seconds)</span></h3>
        <canvas id="chart_cold_start_latency" width="400" height="200"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#fbbf24"></div><span>Current Run</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.8); height:1px; border-style:dashed"></div><span>Baseline</span></div>
        </div>
      </div>
      <div class="plot-card">
        <h3>Hot Start Latency <span class="units">(seconds)</span></h3>
        <canvas id="chart_hot_start_latency" width="400" height="200"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div><span>Current Run</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.8); height:1px; border-style:dashed"></div><span>Baseline</span></div>
        </div>
      </div>
      <div class="plot-card">
        <h3>Error / Normal Latency Ratio <span class="units">(ratio)</span></h3>
        <canvas id="chart_error_to_normal_ratio" width="400" height="200"></canvas>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#4ade80"></div><span>Current Run</span></div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.8); height:1px; border-style:dashed"></div><span>Baseline</span></div>
        </div>
      </div>
    </div>
  </section>
  
  <section>
    <div class="section-title">Current Results</div>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Baseline</th>
          <th>Current</th>
          <th>Improvement</th>
          <th>Target</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </section>
  
  <div class="muted" id="foot"></div>
</main>
<script>
const METRICS = ["slo_violation_rate","cold_start_latency","hot_start_latency","error_to_normal_ratio"];
const LABEL = {
  slo_violation_rate: "SLO Violation Rate",
  cold_start_latency: "Cold Start Latency",
  hot_start_latency: "Hot Start Latency",
  error_to_normal_ratio: "Error / Normal Latency"
};
const UNITS = {
  slo_violation_rate: "%",
  cold_start_latency: "s",
  hot_start_latency: "s",
  error_to_normal_ratio: ""
};
const COLORS = ["#38bdf8","#fbbf24","#a78bfa","#4ade80"];
let history = [];

function fmt(v, key){
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "number") {
    if (key === "slo_violation_rate") return (v * 100).toFixed(2) + "%";
    if (Math.abs(v) < 0.001) return v.toExponential(2);
    if (Math.abs(v) < 10) return v.toFixed(4);
    return v.toFixed(2);
  }
  return String(v);
}

function fmtRaw(v, key){
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "number") {
    if (key === "slo_violation_rate") return (v * 100).toFixed(2);
    if (Math.abs(v) < 0.001) return v.toExponential(2);
    if (Math.abs(v) < 10) return v.toFixed(4);
    return v.toFixed(2);
  }
  return String(v);
}

function drawChart(canvasId, metricKey, colorIdx){
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext("2d");
  const W = c.width, H = c.height, pad = 32;
  ctx.clearRect(0,0,W,H);
  
  // Grid
  ctx.strokeStyle = "#2c3a4f";
  ctx.lineWidth = 0.8;
  for (let i=0;i<=5;i++){
    const y = pad + (H-2*pad)*i/5;
    ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(W-pad,y); ctx.stroke();
  }
  for (let i=0;i<=5;i++){
    const x = pad + (W-2*pad)*i/5;
    ctx.beginPath(); ctx.moveTo(x,pad); ctx.lineTo(x,H-pad); ctx.stroke();
  }
  
  if (history.length < 2){
    ctx.fillStyle = "#8b9bb0";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("waiting for samples...", pad+8, H/2);
    ctx.fillText("Time / Samples", W/2 - 38, H - 4);
    if (metricKey === "slo_violation_rate") ctx.fillText("Violation %", 6, pad - 2);
    else if (metricKey === "error_to_normal_ratio") ctx.fillText("Ratio", 6, pad - 2);
    else ctx.fillText("Time (s)", 6, pad - 2);
    return;
  }
  
  const vals = history.map(h => h.metrics && h.metrics[metricKey]).filter(v => typeof v === "number");
  if (!vals.length){
    ctx.fillStyle = "#8b9bb0";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText("no data", pad+8, H/2);
    return;
  }
  
  let min = Math.min(...vals), max = Math.max(...vals);
  if (metricKey === "slo_violation_rate") { min = min * 100; max = max * 100; }
  // Add small padding
  const range = max - min || 1e-9;
  min = min - range * 0.05;
  max = max + range * 0.05;
  const span = (max - min) || 1e-9;
  
  // Axis labels
  ctx.fillStyle = "#8b9bb0";
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillText("Time / Samples", W/2 - 36, H - 3);
  if (metricKey === "slo_violation_rate") {
    ctx.save();
    ctx.translate(12, H/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("SLO Violation (%)", 0, 0);
    ctx.restore();
  } else if (metricKey === "error_to_normal_ratio") {
    ctx.save();
    ctx.translate(12, H/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("Ratio", 0, 0);
    ctx.restore();
  } else {
    ctx.save();
    ctx.translate(12, H/2);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("Latency (s)", 0, 0);
    ctx.restore();
  }
  
  // Y-axis ticks
  ctx.fillStyle = "#6b7a8f";
  ctx.font = "8px ui-monospace, monospace";
  for (let i=0;i<=5;i++){
    const tval = min + (span)*i/5;
    const ty = H - pad - ((H-2*pad)*i/5);
    let tstr;
    if (Math.abs(tval) < 0.001) tstr = tval.toExponential(1);
    else if (Math.abs(tval) >= 100) tstr = tval.toFixed(0);
    else if (Math.abs(tval) >= 10) tstr = tval.toFixed(1);
    else tstr = tval.toFixed(2);
    ctx.fillText(tstr, 4, ty + 3);
  }
  
  // Plot line
  ctx.strokeStyle = COLORS[colorIdx % COLORS.length];
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  history.forEach((h, i) => {
    let v = h.metrics && h.metrics[metricKey];
    if (typeof v !== "number") return;
    if (metricKey === "slo_violation_rate") v = v * 100;
    const x = pad + (W-2*pad) * (i/Math.max(1,history.length-1));
    const y = H - pad - ((v - min)/span) * (H-2*pad);
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  
  // Baseline reference line
  if (typeof window._base !== "undefined" && typeof window._base[metricKey] === "number") {
    let baseVal = window._base[metricKey];
    if (metricKey === "slo_violation_rate") baseVal = baseVal * 100;
    if (baseVal >= min && baseVal <= max) {
      const baseY = H - pad - ((baseVal - min)/span) * (H-2*pad);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(251,191,36,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(pad, baseY);
      ctx.lineTo(W-pad, baseY);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(251,191,36,0.9)";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText("baseline", W-pad-40, baseY-3);
    }
  }
  
  // Peak annotation
  if (vals.length > 0) {
    let peakIdx = 0, peakVal = metricKey === "slo_violation_rate" ? vals[0]*100 : vals[0];
    for (let i=1; i<vals.length; i++) {
      let v = metricKey === "slo_violation_rate" ? vals[i]*100 : vals[i];
      if (v > peakVal) { peakVal = v; peakIdx = i; }
    }
    if (peakIdx < history.length) {
      const x = pad + (W-2*pad) * (peakIdx/Math.max(1,history.length-1));
      const y = H - pad - ((peakVal - min)/span) * (H-2*pad);
      ctx.fillStyle = COLORS[colorIdx % COLORS.length];
      ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = "#d7dde5";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(peakVal.toFixed(metricKey === "slo_violation_rate" ? 1 : 2), x+5, y-3);
    }
  }
}

async function tick(){
  try {
    const res = await fetch("/api/metrics", {cache:"no-store"});
    const data = await res.json();
    const cur = data.current || {}, base = data.baseline || {}, comp = data.comparison || {};
    const obj = data.objectives || {};
    window._base = base;
    history = (data.history || []).slice(-150);
    
    document.getElementById("sub").textContent =
      "elapsed " + (cur.elapsed_time !== undefined ? cur.elapsed_time.toFixed(1) : "n/a") + "s  |  " + (cur.timestamp || "");
    
    const cards = document.getElementById("cards");
    cards.innerHTML = "";
    METRICS.forEach(key => {
      const info = obj[key] || {};
      const met = info.met === true;
      const pct = comp[key];
      const div = document.createElement("div");
      div.className = "card";
      let displayVal = cur[key];
      if (key === "slo_violation_rate" && typeof displayVal === "number") displayVal = (displayVal * 100).toFixed(2) + "%";
      else if (typeof displayVal === "number") {
        if (Math.abs(displayVal) < 10) displayVal = displayVal.toFixed(4);
        else displayVal = displayVal.toFixed(2);
      } else displayVal = "n/a";
      div.innerHTML =
        "<h2>" + LABEL[key] + "</h2>" +
        "<div class='value'>" + displayVal + (key !== "slo_violation_rate" && typeof cur[key] === "number" && key.includes("latency") ? "s" : (key === "slo_violation_rate" && typeof cur[key] === "number" ? "" : (UNITS[key] ? UNITS[key] : ""))) + "</div>" +
        "<div class='target " + (met?"ok":"miss") + "'>" +
          (pct === undefined || pct === null ? "no baseline" : (pct>=0?"+":"") + pct.toFixed(2) + "% vs baseline") +
          " | target &ge; " + (info.target_pct !== undefined ? info.target_pct.toFixed(1) : "n/a") + "%</div>" +
        "<div class='bar'><div style='width:" +
          Math.max(0, Math.min(100, pct===undefined||pct===null||!info.target_pct?0:(Math.abs(pct)/Math.abs(info.target_pct))*100)) + "%'></div></div>";
      cards.appendChild(div);
    });
    
    const rows = document.getElementById("rows");
    rows.innerHTML = "";
    METRICS.forEach(key => {
      const info = obj[key] || {};
      const met = info.met === true;
      const tr = document.createElement("tr");
      const baseDisp = fmt(base[key], key);
      const curDisp = fmt(cur[key], key);
      let impDisp = "n/a";
      if (comp[key] !== undefined && comp[key] !== null) {
        impDisp = (comp[key]>=0?"+":"") + comp[key].toFixed(2) + "%";
      }
      tr.innerHTML = 
        "<td>" + LABEL[key] + "</td>" +
        "<td>" + baseDisp + "</td>" +
        "<td>" + curDisp + "</td>" +
        "<td class='" + (comp[key] !== undefined && comp[key] !== null && comp[key]>=0 ? "status-ok" : "status-miss") + "'>" + impDisp + "</td>" +
        "<td>&ge; " + (info.target_pct !== undefined ? info.target_pct.toFixed(1) : "n/a") + "%</td>" +
        "<td class='" + (met?"status-ok":"status-miss") + "'>" + (met?"MET":"NOT MET") + "</td>";
      rows.appendChild(tr);
    });
    
    document.getElementById("foot").textContent = "Source: " + (data.metrics_file || "metrics/realtime/performance_metrics.json") + " | Auto-refresh: 1s";
    
    METRICS.forEach((key, idx) => {
      drawChart("chart_" + key, key, idx);
    });
  } catch (e) {
    document.getElementById("sub").textContent = "disconnected: " + e;
  }
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


tracker = None
