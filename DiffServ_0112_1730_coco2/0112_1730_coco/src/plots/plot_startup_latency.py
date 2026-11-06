#!/usr/bin/env python3
"""
plot_startup_latency.py

Plot the startup-latency reduction for:
    * INFaaS cold start (baseline, 0% target)
    * INFaaS warm start (25% target)
    * DIFS warm start with grouping (50% target)

Input:
    <plot_data_baseline>/startup_latency.csv

Output:
    <plot_data_baseline>/plots/startup_latency.png

How to run
    ./run_all_experiments.sh            # menu -> 11 -> 1 or 4
    PYTHONPATH=src python src/plots/plot_startup_latency.py
"""
import csv
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[PLOT] matplotlib not available; skipping startup-latency plot.")
    sys.exit(0)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

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

CSV_PATH = DATA_DIR / "startup_latency.csv"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_ORDER = ["baseline_cold", "infaas_warm", "difs_warm"]
LABELS = {
    "baseline_cold": "INFaaS cold\n(baseline)",
    "infaas_warm": "INFaaS warm",
    "difs_warm": "DIFS warm\n(grouped)",
}
COLORS = {
    "baseline_cold": "#7f7f7f",
    "infaas_warm": "#1f77b4",
    "difs_warm": "#2ca02c",
}
TARGETS = {"baseline_cold": 0.0, "infaas_warm": 25.0, "difs_warm": 50.0}


def read_rows():
    if not CSV_PATH.exists():
        print(f"[PLOT] Missing startup CSV: {CSV_PATH}")
        print("[PLOT] Run the startup-latency experiment first (menu 11 -> 1).")
        return []
    rows = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            system = (row.get("system") or "").strip()
            if not system:
                continue
            try:
                row["reduction_pct"] = float(row.get("reduction_pct", 0) or 0)
                row["total_warm_s"] = float(row.get("total_warm_s", 0) or 0)
                row["total_cold_s"] = float(row.get("total_cold_s", 0) or 0)
                row["warm_start_events"] = int(float(row.get("warm_start_events", 0) or 0))
                row["cold_start_events"] = int(float(row.get("cold_start_events", 0) or 0))
            except (TypeError, ValueError):
                continue
            rows[system] = row
    return rows


def main():
    rows = read_rows()
    if not rows:
        return 0

    systems = [s for s in SYSTEM_ORDER if s in rows]
    if not systems:
        print("[PLOT] No known systems in startup_latency.csv")
        return 0

    fig, (ax, ax_time) = plt.subplots(
        1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.4, 1.0]}
    )

    xpos = list(range(len(systems)))
    reductions = [rows[s]["reduction_pct"] for s in systems]
    bars = ax.bar(xpos, reductions,
                  color=[COLORS[s] for s in systems], width=0.6)
    for x, value in zip(xpos, reductions):
        ax.annotate(f"{value:.1f}%", xy=(x, value), xytext=(0, 4),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold")

    for target, style in ((0.0, ":"), (25.0, "--"), (50.0, "--")):
        ax.axhline(target, color="#d62728" if target else "#555555",
                   linestyle=style, linewidth=1.4,
                   label=f"{target:.0f}% target")

    ax.set_xticks(xpos)
    ax.set_xticklabels([LABELS[s] for s in systems], fontsize=10)
    ax.set_ylabel("Startup-latency reduction (%)", fontsize=12)
    ax.set_title("Startup Latency Reduction vs Targets", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(-5, 70)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")

    times = [rows[s]["total_warm_s"] for s in systems]
    time_bars = ax_time.bar(xpos, times,
                            color=[COLORS[s] for s in systems], width=0.6)
    for x, system in zip(xpos, systems):
        value = rows[system]["total_warm_s"]
        events = rows[system]["warm_start_events"]
        ax_time.annotate(f"{value:.2f}s\n{events} starts", xy=(x, value),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=8)
    ax_time.set_xticks(xpos)
    ax_time.set_xticklabels([LABELS[s].replace("\n", " ") for s in systems],
                            fontsize=8, rotation=15)
    ax_time.set_ylabel("Total startup time (s)", fontsize=11)
    ax_time.set_title("Measured Startup Totals", fontsize=11,
                      fontweight="bold")
    ax_time.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    for bar in bars + tuple(time_bars):
        bar.set_edgecolor("black")
        bar.set_linewidth(0.6)

    fig.tight_layout()
    out = PLOT_DIR / "startup_latency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")

    for system in systems:
        row = rows[system]
        print(f"[PLOT] {system}: reduction={row['reduction_pct']:.2f}% "
              f"(target {TARGETS[system]:.0f}%), "
              f"starts={row['warm_start_events']}, "
              f"total={row['total_warm_s']:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
