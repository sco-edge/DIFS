#!/usr/bin/env python3
"""
compile_report_v7.py

Builds the DIFS_OC_v7 performance report from v7's own benchmark outputs.

Inputs (all optional; missing data is reported instead of fabricated):
    results/model_benchmark/plot_data_baseline/baseline_comparison.csv
    results/model_benchmark/plot_data_baseline/objective_metrics.csv
    metrics/realtime/performance_metrics.json
    results/model_benchmark/plot_data_baseline/switch_*/pairwise_switching.csv

Output:
    results/model_benchmark/Reports/DIFS_OC_v7_Performance_Report.pdf

Usage:
    PYTHONPATH=src python compile_report_v7.py
    PYTHONPATH=src python compile_report_v7.py --no-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Resolve the input data dir the same way the plot scripts do, so the runner
# can point the report at the live/last results via DIFS_PLOT_DATA_DIR.
if os.environ.get("DIFS_PLOT_DATA_DIR"):
    DATA_DIR = Path(os.environ["DIFS_PLOT_DATA_DIR"])
    if DATA_DIR.name != "plot_data_baseline":
        DATA_DIR = DATA_DIR / "plot_data_baseline"
elif os.environ.get("DIFS_LAST_RESULTS_DIR"):
    DATA_DIR = Path(os.environ["DIFS_LAST_RESULTS_DIR"]) / "plot_data_baseline"
elif os.environ.get("DIFS_LIVE_RESULTS_DIR"):
    DATA_DIR = Path(os.environ["DIFS_LIVE_RESULTS_DIR"]) / "plot_data_baseline"
else:
    DATA_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"
BASELINE_CSV = DATA_DIR / "baseline_comparison.csv"
OBJECTIVE_CSV = DATA_DIR / "objective_metrics.csv"
TRACKER_JSON = PROJECT_ROOT / "metrics" / "realtime" / "performance_metrics.json"
REPORT_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "Reports"
TEX_FILE = REPORT_DIR / "DIFS_OC_v7_Performance_Report.tex"

SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_LABELS = {
    "clipper_large": "Clipper Large-Only",
    "clipper_small": "Clipper Small-Only",
    "infaas": "INFaaS",
    "difs_v3": "\\textbf{DIFS\\_v3 (v7 stack)}",
}
DIFS_SYSTEM = "difs_v3"
LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]
STEP_ORDER = [10, 20, 30, 40]
SWITCH_MODES = ["hot", "force_upload", "cold_every"]

OBJECTIVES = {
    "slo_violation_rate": {
        "label": "SLO violation rate",
        "target": 5.0,
        "kind": "fraction",
    },
    "cold_start_latency": {
        "label": "Cold-start latency",
        "target": 50.0,
        "kind": "seconds",
    },
    "hot_start_latency": {
        "label": "Hot-start latency",
        "target": 50.0,
        "kind": "milliseconds",
    },
    "error_to_normal_ratio": {
        "label": "Error / normal latency",
        "target": 30.0,
        "kind": "ratio",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _tex_escape(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def load_baseline_rows():
    if not BASELINE_CSV.exists():
        return []
    rows = []
    with open(BASELINE_CSV) as f:
        for row in csv.DictReader(f):
            row["steps"] = _as_int(row.get("steps"))
            row["requests_submitted"] = _as_int(row.get("requests_submitted"))
            row["requests_completed"] = _as_int(row.get("requests_completed"))
            row["requests_dropped"] = _as_int(row.get("requests_dropped"))
            row["latency_slo_attainment_pct"] = _as_float(
                row.get("latency_slo_attainment_pct"), 0.0
            )
            row["accuracy_slo_attainment_pct"] = _as_float(
                row.get("accuracy_slo_attainment_pct"), 0.0
            )
            row["throughput_req_per_s"] = _as_float(
                row.get("throughput_req_per_s"), 0.0
            )
            rows.append(row)
    return rows


def load_objective_rows():
    if not OBJECTIVE_CSV.exists():
        return []
    rows = []
    with open(OBJECTIVE_CSV) as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in (
                "prompt_count", "optimization_enabled", "requests_submitted"
            ):
                parsed[key] = _as_int(row.get(key))
            for key in (
                "slo_violation_rate", "cold_start_latency",
                "hot_start_latency", "error_to_normal_ratio",
            ):
                parsed[key] = _as_float(row.get(key))
            rows.append(parsed)
    return rows


def load_tracker_state():
    if not TRACKER_JSON.exists():
        return None
    try:
        return json.loads(TRACKER_JSON.read_text())
    except Exception:
        return None


def find_reference_csv():
    # Only use DIFS_OC_v7's own results; do not pull from DIFS_OC_v6
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def system_aggregates(rows):
    aggregates = {}
    for system in SYSTEM_ORDER:
        subset = [r for r in rows if r.get("baseline") == system]
        if not subset:
            continue
        submitted = sum(r["requests_submitted"] for r in subset)
        completed = sum(r["requests_completed"] for r in subset)
        weights = submitted or len(subset)
        if submitted:
            latency = sum(
                r["latency_slo_attainment_pct"] * r["requests_submitted"]
                for r in subset
            ) / submitted
            accuracy = sum(
                r["accuracy_slo_attainment_pct"] * r["requests_submitted"]
                for r in subset
            ) / submitted
        else:
            latency = sum(r["latency_slo_attainment_pct"] for r in subset) / weights
            accuracy = sum(r["accuracy_slo_attainment_pct"] for r in subset) / weights
        throughput = sum(r["throughput_req_per_s"] for r in subset) / len(subset)
        completion = 100.0 * completed / submitted if submitted else 0.0
        aggregates[system] = {
            "latency_slo_pct": latency,
            "accuracy_slo_pct": accuracy,
            "throughput": throughput,
            "completion_pct": completion,
            "cells": len({r["load_level"] for r in subset}),
        }
    return aggregates


def objective_aggregates(rows):
    aggregates = {}
    for system in SYSTEM_ORDER:
        subset = [r for r in rows if r.get("baseline") == system]
        if not subset:
            continue
        result = {}
        weighted = [
            (r["requests_submitted"], r["slo_violation_rate"])
            for r in subset
            if r.get("slo_violation_rate") is not None
            and r.get("requests_submitted", 0) > 0
        ]
        if weighted:
            total = sum(n for n, _ in weighted)
            result["slo_violation_rate"] = sum(n * v for n, v in weighted) / total
        else:
            values = [
                r["slo_violation_rate"] for r in subset
                if r.get("slo_violation_rate") is not None
            ]
            if values:
                result["slo_violation_rate"] = statistics.median(values)
        for key in ("cold_start_latency", "hot_start_latency",
                    "error_to_normal_ratio"):
            values = [r[key] for r in subset if r.get(key) is not None]
            if values:
                result[key] = statistics.median(values)
        aggregates[system] = result
    return aggregates


def _fmt_objective(key, value):
    if value is None:
        return "n/a"
    kind = OBJECTIVES[key]["kind"]
    if kind == "fraction":
        return f"{value * 100.0:.2f}\\%"
    if kind == "seconds":
        return f"{value:.3f}\\,s"
    if kind == "milliseconds":
        return f"{value * 1000.0:.3f}\\,ms"
    return f"{value:.3f}"


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_run_config(rows, objective_rows, tracker):
    systems = [s for s in SYSTEM_ORDER if any(r["baseline"] == s for r in rows)]
    loads = sorted(
        {r["load_level"] for r in rows}, key=lambda x: (
            LOAD_ORDER.index(x) if x in LOAD_ORDER else len(LOAD_ORDER)
        )
    )
    optimization = sorted(
        {str(r.get("optimization_enabled", "")) for r in objective_rows}
    )
    prompt_counts = sorted(
        {str(r.get("prompt_count", "")) for r in objective_rows
         if r.get("prompt_count")}
    )
    lines = [
        f"Report generated & {datetime.now().isoformat(timespec='seconds')} \\\\",
        f"Baseline CSV & {BASELINE_CSV.name if BASELINE_CSV.exists() else 'missing'} "
        f"({len(rows)} rows) \\\\",
        f"Objective CSV & {OBJECTIVE_CSV.name if OBJECTIVE_CSV.exists() else 'missing'} "
        f"({len(objective_rows)} cells) \\\\",
        f"Tracker snapshot & "
        f"{'present' if tracker else 'missing'} \\\\",
        f"Systems & {', '.join(systems) if systems else 'none'} \\\\",
        f"Load levels & {', '.join(loads) if loads else 'none'} \\\\",
        f"Prompt count & {', '.join(prompt_counts) if prompt_counts else 'n/a'} \\\\",
        f"Optimizer enabled & {', '.join(optimization) if optimization else 'n/a'} \\\\",
        f"V6 reference CSV & "
        f"{find_reference_csv() or 'not found'} \\\\",
    ]
    return "\n".join(lines)


def build_objective_rows(objective_rows, tracker):
    aggregates = objective_aggregates(objective_rows)
    current = (tracker or {}).get("current", {})
    reference = (tracker or {}).get("baseline", {})

    difs_values = aggregates.get(DIFS_SYSTEM, {})
    competitors = [
        s for s in SYSTEM_ORDER if s != DIFS_SYSTEM and s in aggregates
    ]

    lines = []
    for key, spec in OBJECTIVES.items():
        difs_value = difs_values.get(key)
        if difs_value is None and key in current:
            difs_value = _as_float(current.get(key))

        base_value = None
        base_label = "n/a"
        for system in competitors:
            value = aggregates.get(system, {}).get(key)
            if value is None:
                continue
            if base_value is None or value < base_value:
                base_value = value
                base_label = SYSTEM_LABELS[system]
        if base_value is None:
            value = _as_float(reference.get(key))
            if value is not None:
                base_value = value
                base_label = "v6 reference"

        if base_value is None or difs_value is None:
            lines.append(
                f"{spec['label']} & --- & --- & --- & --- & "
                f"$\\ge$ {spec['target']:.0f}\\% & MISS \\\\"
            )
            continue

        if base_value > 0:
            improvement = (base_value - difs_value) / base_value * 100.0
            improvement_text = f"{improvement:+.2f}\\%"
            met = improvement >= spec["target"]
            status = (
                "\\textcolor{green!50!black}{MET}" if met
                else "\\textcolor{red!70!black}{MISS}"
            )
        else:
            improvement_text = "n/a"
            status = "n/a"

        lines.append(
            f"{spec['label']} & {base_label} & "
            f"{_fmt_objective(key, base_value)} & "
            f"{_fmt_objective(key, difs_value)} & "
            f"{improvement_text} & "
            f"$\\ge$ {spec['target']:.0f}\\% & {status} \\\\"
        )
    return "\n".join(lines)


def build_system_rows(rows):
    aggregates = system_aggregates(rows)
    lines = []
    for system in SYSTEM_ORDER:
        agg = aggregates.get(system)
        if not agg:
            continue
        lines.append(
            f"{SYSTEM_LABELS[system]} & {agg['latency_slo_pct']:.1f} & "
            f"{agg['accuracy_slo_pct']:.1f} & {agg['throughput']:.4f} & "
            f"{agg['completion_pct']:.1f} & {agg['cells']} \\\\"
        )
    return "\n".join(lines) if lines else (
        "\\multicolumn{6}{c}{No v7 baseline rows found. Run the benchmark first.} \\\\"
    )


def build_per_cell_rows(rows):
    def sort_key(row):
        load = row["load_level"]
        return (
            SYSTEM_ORDER.index(row["baseline"])
            if row["baseline"] in SYSTEM_ORDER else len(SYSTEM_ORDER),
            LOAD_ORDER.index(load) if load in LOAD_ORDER else len(LOAD_ORDER),
            row["steps"],
        )

    lines = []
    for row in sorted(rows, key=sort_key):
        submitted = row["requests_submitted"]
        completed = row["requests_completed"]
        completion = 100.0 * completed / submitted if submitted else 0.0
        system = SYSTEM_LABELS.get(row["baseline"], row["baseline"])
        lines.append(
            f"{system} & {row['load_level']} & {row['steps']} & "
            f"{row['latency_slo_attainment_pct']:.1f} & "
            f"{row['accuracy_slo_attainment_pct']:.1f} & "
            f"{row['throughput_req_per_s']:.4f} & "
            f"{completed}/{submitted} ({completion:.0f}\\%) & "
            f"{row['requests_dropped']} \\\\"
        )
    return "\n".join(lines) if lines else (
        "\\multicolumn{8}{c}{No per-step rows found.} \\\\"
    )


def build_switching_rows():
    lines = []
    for mode in SWITCH_MODES:
        path = DATA_DIR / f"switch_{mode}" / "pairwise_switching.csv"
        if not path.exists():
            lines.append(
                f"{mode} & --- & --- & --- & --- \\\\"
            )
            continue
        values = []
        try:
            with open(path) as f:
                for row in csv.DictReader(f):
                    for key, value in row.items():
                        if key is None or key.startswith("from_model"):
                            continue
                        numeric = _as_float(value)
                        if numeric is not None and numeric > 0:
                            values.append(numeric)
        except Exception:
            values = []
        if values:
            lines.append(
                f"{mode} & {len(values)} & "
                f"{statistics.median(values):.2f} & "
                f"{min(values):.2f} & {max(values):.2f} \\\\"
            )
        else:
            lines.append(f"{mode} & 0 & --- & --- & --- \\\\")
    return "\n".join(lines)


def fill_table(tex, label, new_rows):
    pattern = (
        rf"(\\label\{{{label}\}}.*?\\midrule\s*\n)"
        r"(.*?)(\\bottomrule)"
    )

    def replacer(match):
        return match.group(1) + "\n" + new_rows + "\n" + match.group(3)

    tex, count = re.subn(pattern, replacer, tex, count=1, flags=re.DOTALL)
    if count:
        print(f"[COMPILE] Populated table {label}")
    else:
        print(f"[COMPILE] WARNING: table {label} not found in template")
    return tex


# ---------------------------------------------------------------------------
# Report + PDF
# ---------------------------------------------------------------------------

def populate_tex(rows, objective_rows, tracker):
    if not TEX_FILE.exists():
        print(f"[COMPILE] ERROR: report template missing: {TEX_FILE}")
        return False
    tex = TEX_FILE.read_text()
    tex = fill_table(tex, "tab:run_config",
                     build_run_config(rows, objective_rows, tracker))
    tex = fill_table(tex, "tab:objectives",
                     build_objective_rows(objective_rows, tracker))
    tex = fill_table(tex, "tab:systems", build_system_rows(rows))
    tex = fill_table(tex, "tab:per_cell", build_per_cell_rows(rows))
    tex = fill_table(tex, "tab:switching", build_switching_rows())
    TEX_FILE.write_text(tex)
    print(f"[COMPILE] Wrote populated {TEX_FILE}")
    return True


def generate_plots():
    if not BASELINE_CSV.exists():
        print("[COMPILE] Skipping plots: v7 baseline CSV missing.")
        return
    scripts = [
        PROJECT_ROOT / "src" / "plots" / "plot_baseline_comparison.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switching_overhead.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switch_modes.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switch_slo.py",
    ]
    for script in scripts:
        if not script.exists():
            print(f"[COMPILE] Skipping missing plot script {script.name}")
            continue
        print(f"[COMPILE] Running {script.name}...")
        try:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
                check=False,
            )
        except Exception as exc:
            print(f"[COMPILE] WARNING: {script.name} failed: {exc}")


def compile_pdf():
    if shutil.which("pdflatex") is None:
        print("[COMPILE] pdflatex not found; LaTeX source was written but no PDF.")
        return
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    for attempt in range(1, 3):
        print(f"[COMPILE] pdflatex pass {attempt}/2...")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", TEX_FILE.name],
            cwd=str(REPORT_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[COMPILE] WARNING: pdflatex pass {attempt} returned "
                  f"{result.returncode}")
    log_file = TEX_FILE.with_suffix(".log")
    if log_file.exists():
        errors = [
            line for line in log_file.read_text().splitlines()
            if line.startswith("!") or "Error" in line
        ]
        if errors:
            print("[COMPILE] LaTeX errors:")
            for line in errors[-10:]:
                print(f"  {line}")
    pdf = TEX_FILE.with_suffix(".pdf")
    if pdf.exists():
        print(f"[COMPILE] PDF generated: {pdf} ({pdf.stat().st_size / 1e6:.2f} MB)")
    else:
        print("[COMPILE] ERROR: PDF not generated. Check the LaTeX log.")


def main():
    parser = argparse.ArgumentParser(description="Compile the DIFS_OC_v7 report")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not regenerate figures before compiling")
    args = parser.parse_args()

    rows = load_baseline_rows()
    objective_rows = load_objective_rows()
    tracker = load_tracker_state()
    print(f"[COMPILE] baseline rows={len(rows)} "
          f"objective cells={len(objective_rows)} "
          f"tracker={'yes' if tracker else 'no'}")

    if not populate_tex(rows, objective_rows, tracker):
        sys.exit(1)
    if not args.no_plots:
        generate_plots()
    compile_pdf()


if __name__ == "__main__":
    main()
