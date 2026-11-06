#!/usr/bin/env python3
"""Populate the real-time tracker JSON from completed experiment CSVs.

After option 3 finishes, read the objective_metrics.csv from the live results
directory and synthesize a tracker state (history + current/baseline/comparison/objectives)
so the dashboard can display plots with labels/legends/axes.
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_TARGETS = {
    "slo_violation_rate": 0.05,
    "cold_start_latency": 0.50,
    "hot_start_latency": 0.50,
    "error_to_normal_ratio": 0.30,
}

LOWER_IS_BETTER = {
    "slo_violation_rate",
    "cold_start_latency",
    "hot_start_latency",
    "error_to_normal_ratio",
}

# Alternative result store directories (option 3 writes to _live; last is backup)
_ALT_RESULT_DIRS = [
    "results/model_benchmark_live",
    "results/model_benchmark_last",
]


def _find_csv(filename: str, project_root: Path, live_dir: Optional[Path]) -> Optional[Path]:
    """Search for a CSV in live dir first, then alt dirs, then standard."""
    # 1) Live dir (DIFS_LIVE_RESULTS_DIR or model_benchmark_live)
    if live_dir:
        p = Path(live_dir) / "plot_data_baseline" / filename
        if p.exists():
            return p
    # 2) Alt dirs (model_benchmark_live, model_benchmark_last)
    for alt in _ALT_RESULT_DIRS:
        p = project_root / alt / "plot_data_baseline" / filename
        if p.exists():
            return p
    # 3) Standard dir
    p = project_root / "results" / "model_benchmark" / "plot_data_baseline" / filename
    if p.exists():
        return p
    return None


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def populate_from_csv(project_root: str = ".") -> Dict[str, Any]:
    root = Path(project_root).resolve()
    live_dir = os.environ.get("DIFS_LIVE_RESULTS_DIR")

    # ── Load objective_metrics.csv (primary: has exact metric columns) ──
    obj_csv = _find_csv("objective_metrics.csv", root, live_dir)
    objective_rows: List[Dict[str, str]] = _read_csv(obj_csv) if obj_csv else []

    # ── Load baseline_comparison.csv (supplementary) ──
    base_csv = _find_csv("baseline_comparison.csv", root, live_dir)
    baseline_rows: List[Dict[str, str]] = _read_csv(base_csv) if base_csv else []

    # ── Build "current" from objective_metrics.csv ──
    load_rank = {"very_high": 5, "high": 4, "medium": 3, "low": 2, "saturation": 1, "stress": 0}
    def load_score(r: Dict[str, str]) -> int:
        return load_rank.get(str(r.get("load_level", "")).strip(), 0)

    if objective_rows:
        sorted_rows = sorted(objective_rows, key=load_score, reverse=True)
        best = sorted_rows[0]
        current: Dict[str, Any] = {
            "slo_violation_rate": _to_float(best.get("slo_violation_rate")),
            "cold_start_latency": _to_float(best.get("cold_start_latency")),
            "hot_start_latency": _to_float(best.get("hot_start_latency")),
            "error_to_normal_ratio": _to_float(best.get("error_to_normal_ratio")),
            "elapsed_time": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        current = {k: v for k, v in current.items() if v is not None}
    else:
        current = {}

    # ── Build "baseline" from low-load objective row ──
    if objective_rows:
        low_rows = [r for r in objective_rows if str(r.get("load_level", "")).strip().lower() in ("low", "very_low", "low_load")]
        if not low_rows:
            low_rows = objective_rows[-1:]
        low_best = min(low_rows, key=load_score) if low_rows else None
        if low_best:
            baseline: Dict[str, Any] = {
                "slo_violation_rate": _to_float(low_best.get("slo_violation_rate")),
                "cold_start_latency": _to_float(low_best.get("cold_start_latency")),
                "hot_start_latency": _to_float(low_best.get("hot_start_latency")),
                "error_to_normal_ratio": _to_float(low_best.get("error_to_normal_ratio")),
                "elapsed_time": 0.0,
                "timestamp": datetime.now().isoformat(),
            }
            baseline = {k: v for k, v in baseline.items() if v is not None}
        else:
            baseline = {}
    else:
        baseline = {}

    # Ensure current has all 4 metrics (fallback to baseline if missing)
    for key in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
        if key not in current and key in baseline and isinstance(baseline[key], (int, float)):
            current[key] = baseline[key]

    # ── Compute comparison ──
    comparison: Dict[str, float] = {}
    for key in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
        c = current.get(key)
        b = baseline.get(key)
        if isinstance(c, (int, float)) and isinstance(b, (int, float)) and b != 0:
            if key in LOWER_IS_BETTER:
                improvement = ((b - c) / b) * 100.0
            else:
                improvement = ((c - b) / b) * 100.0
            comparison[key] = round(improvement, 3)

    # ── Build history from objective_metrics.csv rows ──
    history: List[Dict[str, Any]] = []
    now = datetime.now()
    load_order = {"low": 0, "medium": 1, "high": 2, "saturation": 3, "stress": 4, "very_high": 5}
    if objective_rows:
        sorted_rows = sorted(objective_rows, key=lambda r: load_order.get(str(r.get("load_level", "")).strip(), 99))
        for i, r in enumerate(sorted_rows):
            ts = now.replace(second=0, microsecond=0).isoformat()
            entry: Dict[str, Any] = {"timestamp": ts, "elapsed_time": float(i * 10), "metrics": {}}

            slo = _to_float(r.get("slo_violation_rate"))
            lat_cold = _to_float(r.get("cold_start_latency"))
            lat_hot = _to_float(r.get("hot_start_latency"))
            err = _to_float(r.get("error_to_normal_ratio"))

            if slo is not None:
                entry["metrics"]["slo_violation_rate"] = max(0.0, 1.0 - slo) if slo <= 1.0 else max(0.0, (100.0 - slo) / 100.0)
            if lat_cold is not None:
                entry["metrics"]["cold_start_latency"] = lat_cold
            if lat_hot is not None:
                entry["metrics"]["hot_start_latency"] = lat_hot
            if err is not None:
                entry["metrics"]["error_to_normal_ratio"] = err

            # Fill missing from current
            for k in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
                if k not in entry["metrics"] and k in current:
                    entry["metrics"][k] = current[k]
            history.append(entry)

    # ── Ensure at least 2 entries ──
    if len(history) < 2 and current:
        base_ts = datetime.now().isoformat()
        for i in range(2 - len(history)):
            history.append({
                "timestamp": base_ts,
                "elapsed_time": float(len(history) * 10),
                "metrics": {k: v for k, v in current.items() if isinstance(v, (int, float))},
            })

    # Ensure current has all 4 metrics (fallback to baseline if missing)
    for key in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
        if key not in current and key in baseline and isinstance(baseline[key], (int, float)):
            current[key] = baseline[key]

    # Also fill from history if still missing
    for key in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
        if key not in current:
            for h in history:
                m = h.get("metrics", {})
                if isinstance(m.get(key), (int, float)):
                    current[key] = m[key]
                    break

    # Update current timestamp
    if "timestamp" not in current:
        current["timestamp"] = datetime.now().isoformat()
    current["elapsed_time"] = float(len(history) * 10) if history else 0.0

    # ── Build objectives ──
    objectives: Dict[str, Dict[str, Any]] = {}
    for key, target in DEFAULT_TARGETS.items():
        pct = comparison.get(key)
        objectives[key] = {
            "target_pct": round(target * 100.0, 2),
            "actual_pct": pct,
            "met": bool(pct is not None and pct >= target * 100.0),
        }

    # ── Build baseline summary ──
    baseline_summary: Dict[str, Any] = {}
    for key in ["slo_violation_rate", "cold_start_latency", "hot_start_latency", "error_to_normal_ratio"]:
        if key in baseline and isinstance(baseline[key], (int, float)):
            baseline_summary[key] = baseline[key]
        elif key in comparison:
            baseline_summary[key] = comparison.get(key, 0.0)
        else:
            baseline_summary[key] = DEFAULT_TARGETS.get(key, 0.0) * 100.0 if key == "slo_violation_rate" else DEFAULT_TARGETS.get(key, 0.5)

    # ── Write JSON ──
    metrics_dir = root / "metrics" / "realtime"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_dir / "performance_metrics.json"
    data = {
        "baseline": baseline_summary,
        "current": current,
        "comparison": comparison,
        "objectives": objectives,
        "history": history,
    }
    tmp = metrics_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(metrics_file)
    print(f"Populated {len(data.get('history', []))} history entries (obj_csv={obj_csv}, base_csv={base_csv})")
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    data = populate_from_csv(args.project_root)
    print(f"Populated {len(data.get('history', []))} history entries")


if __name__ == "__main__":
    main()
