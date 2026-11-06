"""
plot_baseline_comparison.py

Generates comparison plots from baseline_comparison.csv:

  1. ACCURACY-LATENCY:  Y = Latency SLO attainment, X = Accuracy SLO attainment
  2. ACCURACY-THROUGHPUT: Y = Throughput (req/s), X = Accuracy SLO attainment
  3. LATENCY BAR CHART: Grouped bars by system, one group per load level
  4. ACCURACY BAR CHART: Same grouping for accuracy SLO
  5. THROUGHPUT BAR CHART: Same grouping for throughput
  6. COMPLETION RATE BAR CHART: completed / submitted

How to run
    Recommended (regenerates every v6 figure):
        ./run_all_experiments.sh --plots

    Directly, from the project root:
        PYTHONPATH=src python src/plots/plot_baseline_comparison.py

    Input:
        results/model_benchmark/plot_data_baseline/baseline_comparison.csv
    Output:
        results/model_benchmark/plot_data_baseline/plots/*.png
"""

import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[PLOT] matplotlib not available; skipping plots.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Allow overriding data paths for live/last runs
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
DATA_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "baseline_comparison.csv"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_STYLES = {
    "clipper_large": {"color": "#d62728", "marker": "s", "label": "Clipper Large-Only",
                      "linestyle": "--", "hatch": "//"},
    "clipper_small": {"color": "#ff7f0e", "marker": "^", "label": "Clipper Small-Only",
                      "linestyle": "--", "hatch": "\\\\"},
    "infaas":        {"color": "#1f77b4", "marker": "D", "label": "INFaaS",
                      "linestyle": "-.", "hatch": "xx"},
    "difs_v3":       {"color": "#2ca02c", "marker": "o", "label": "DIFS_OC_v3 (Ours)",
                      "linestyle": "-", "hatch": "//"},
}

LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]
LOAD_MARKERS = {
    "low":        {"size": 80,  "label_suffix": " (low)"},
    "medium":     {"size": 120, "label_suffix": " (med)"},
    "high":       {"size": 180, "label_suffix": " (high)"},
    "saturation": {"size": 180, "label_suffix": " (sat)"},
    "stress":     {"size": 180, "label_suffix": " (stress)"},
}

# Distinct hatches per system for B&W readability
HATCHES = ["//", "\\\\", "xx", "//"]


def load_data():
    if not CSV_PATH.exists():
        print(f"[PLOT] CSV not found: {CSV_PATH}")
        sys.exit(1)
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            row["latency_slo_attainment_pct"] = float(row["latency_slo_attainment_pct"])
            row["accuracy_slo_attainment_pct"] = float(row["accuracy_slo_attainment_pct"])
            row["throughput_req_per_s"] = float(row["throughput_req_per_s"])
            row["requests_submitted"] = int(row.get("requests_submitted", 0))
            row["requests_completed"] = int(row.get("requests_completed", 0))
            row["requests_dropped"] = int(row.get("requests_dropped", 0))
            rows.append(row)
    return rows


# ---- Scatter plots ----

# NOTE on accuracy SLO: with the step-count-based accuracy definition, every
# completed request is served at its own required step count, so accuracy SLO
# attainment is 100.0% for ALL rows (every system/load/step).  Plotting it on
# an axis therefore collapses the figure to a 1-D degenerate scatter.  The two
# scatters below therefore use the metrics that actually differentiate the
# systems (throughput, latency SLO, step count) instead of the constant
# accuracy axis.  Accuracy is still reported in the bar chart (Fig 6.6) where
# its constant 100% value is stated explicitly.

def _legend_for_systems(ax):
    for bname, style in SYSTEM_STYLES.items():
        ax.plot([], [], color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markersize=7,
                label=style["label"], linewidth=1.5)


def plot_throughput_latency(rows, out_path):
    """Throughput (req/s) vs Latency SLO Attainment (%).

    Replaces the former accuracy-latency scatter. Because accuracy SLO is
    constant (100%) we drop it and show the real trade-off: achieving high
    latency SLO costs throughput (Clipper Large-Only) and vice versa.
    One point per per-step row (4 per system/load cell); size = load level.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for bname, style in SYSTEM_STYLES.items():
        subset = [r for r in rows if r["baseline"] == bname]
        if not subset:
            continue
        for r in sorted(subset, key=lambda x: x["load_level"]):
            load = r["load_level"]
            ms = LOAD_MARKERS.get(load, LOAD_MARKERS["medium"])
            ax.scatter(
                r["throughput_req_per_s"],
                r["latency_slo_attainment_pct"],
                c=style["color"], marker=style["marker"],
                s=ms["size"], edgecolors="black", linewidths=0.5, zorder=3,
            )
    _legend_for_systems(ax)
    for load, ms in LOAD_MARKERS.items():
        ax.scatter([], [], c="gray", marker="o", s=ms["size"],
                   edgecolors="black", linewidths=0.5,
                   label=f"Load: {load}", zorder=3)
    ax.set_xlabel("Throughput (req/s)", fontsize=12)
    ax.set_ylabel("Latency SLO Attainment (%)", fontsize=12)
    ax.set_title("Throughput\u2013Latency SLO Trade-off by System and Load", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
              bbox_to_anchor=(0.0, -0.02), ncol=2)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_latency_by_steps(rows, out_path):
    """Latency SLO Attainment (%) vs step count (10/20/30/40).

    Replaces the former accuracy-throughput scatter. It isolates WHERE the
    latency degradation occurs in the step axis: Clipper Large-Only collapses
    to 0% at 40 steps, INFaaS degrades at 30/40, while Clipper Small-Only and
    DIFS_v3 stay at 100% across all steps. Lines connect steps per system.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    STEPS = [10, 20, 30, 40]
    for bname, style in SYSTEM_STYLES.items():
        subset = [r for r in rows if r["baseline"] == bname]
        if not subset:
            continue
        # group per load, average latency across the steps for a smooth line
        for load in LOAD_ORDER:
            cell = [r for r in subset if r["load_level"] == load]
            if not cell:
                continue
            x = [r["steps"] for r in cell]
            y = [r["latency_slo_attainment_pct"] for r in cell]
            order = sorted(range(len(x)), key=lambda i: x[i])
            x = [x[i] for i in order]
            y = [y[i] for i in order]
            ms = LOAD_MARKERS.get(load, LOAD_MARKERS["medium"])
            ax.plot(x, y, color=style["color"], linewidth=0.8,
                    alpha=0.45, zorder=1)
            ax.scatter(x, y, c=style["color"], marker=style["marker"],
                       s=ms["size"], edgecolors="black", linewidths=0.4,
                       zorder=3, alpha=0.85)
    _legend_for_systems(ax)
    for load, ms in LOAD_MARKERS.items():
        ax.scatter([], [], c="gray", marker="o", s=ms["size"],
                   edgecolors="black", linewidths=0.5,
                   label=f"Load: {load}", zorder=3)
    ax.set_xticks(STEPS)
    ax.set_xlabel("Step Count", fontsize=12)
    ax.set_ylabel("Latency SLO Attainment (%)", fontsize=12)
    ax.set_title("Latency SLO Attainment by Step Count and System", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
              bbox_to_anchor=(0.0, -0.02), ncol=2)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


# ---- Bar charts ----

def _cell_aggregate(rows, bname, load, metric_key, agg="mean"):
    """Aggregate a metric across all step-count rows of one (baseline, load) cell.

    The CSV holds one row per step count (10, 20, 30, 40), so a cell has 4 rows.
    Singling out row[0] (steps=10) misrepresents a cell whenever SLO attainment
    varies with step count.  We aggregate the cell:
      - 'mean'  : average over the step counts
      - 'min'   : worst case (min attainment / max throughput)
      - 'sum'   : total requests across step counts (for completion rate)
    """
    cell = [r for r in rows if r["baseline"] == bname and r["load_level"] == load]
    if not cell:
        return 0.0
    if agg == "sum":
        return sum(r[metric_key] for r in cell)
    vals = [r[metric_key] for r in cell]
    return sum(vals) / len(vals) if agg == "mean" else min(vals)


def _grouped_bar(rows, metric_key, ylabel, title, out_path, fmt="{:.1f}",
                 show_hatches=True, ylim_top=None, label_offset=0.8,
                 agg="mean"):
    """Generic grouped bar chart: x = load_level, groups = systems."""
    n_sys = len(SYSTEM_ORDER)
    n_load = len(LOAD_ORDER)
    bar_width = 0.18
    x = np.arange(n_load)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, bname in enumerate(SYSTEM_ORDER):
        vals = []
        for load in LOAD_ORDER:
            vals.append(_cell_aggregate(rows, bname, load, metric_key, agg=agg))
        style = SYSTEM_STYLES[bname]
        hatch = HATCHES[i] if show_hatches else None
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                      label=style["label"], color=style["color"],
                      edgecolor="black", linewidth=0.5,
                      hatch=hatch, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + label_offset,
                        fmt.format(v), ha="center", va="bottom", fontsize=8,
                        fontweight="bold")

    ax.set_xticks(x + bar_width * (n_sys - 1) / 2)
    ax.set_xticklabels([l.capitalize() for l in LOAD_ORDER], fontsize=12)
    ax.set_xlabel("Load Level", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(bottom=0)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9,
              edgecolor="black", fancybox=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_latency_bars(rows, out_path):
    _grouped_bar(rows, "latency_slo_attainment_pct",
                 "Latency SLO Attainment (%)",
                 "Latency SLO Attainment by System and Load Level",
                 out_path, ylim_top=105)


def plot_accuracy_bars(rows, out_path):
    _grouped_bar(rows, "accuracy_slo_attainment_pct",
                 "Accuracy SLO Attainment (%)",
                 "Accuracy SLO Attainment by System and Load Level",
                 out_path, ylim_top=105)


def plot_throughput_bars(rows, out_path):
    _grouped_bar(rows, "throughput_req_per_s",
                 "Throughput (req/s)",
                 "Throughput by System and Load Level",
                 out_path, fmt="{:.4f}", label_offset=0.002)


def plot_completion_bars(rows, out_path):
    for bname in SYSTEM_ORDER:
        for load in LOAD_ORDER:
            sub = _cell_aggregate(rows, bname, load, "requests_submitted", agg="sum")
            comp = _cell_aggregate(rows, bname, load, "requests_completed", agg="sum")
            for r in rows:
                if r["baseline"] == bname and r["load_level"] == load:
                    r["completion_pct"] = 100.0 * comp / sub if sub else 0.0
    _grouped_bar(rows, "completion_pct",
                 "Completion Rate (%)",
                 "Request Completion Rate by System and Load Level",
                 out_path, ylim_top=115, label_offset=1.0)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    if not rows:
        print("[PLOT] No data rows found.")
        return
    plot_throughput_latency(rows, PLOT_DIR / "throughput_latency.png")
    plot_latency_by_steps(rows, PLOT_DIR / "latency_by_steps.png")
    plot_latency_bars(rows, PLOT_DIR / "latency_slo_bars.png")
    plot_accuracy_bars(rows, PLOT_DIR / "accuracy_slo_bars.png")
    plot_throughput_bars(rows, PLOT_DIR / "throughput_bars.png")
    plot_completion_bars(rows, PLOT_DIR / "completion_rate_bars.png")
    print("[PLOT] Done.")


if __name__ == "__main__":
    main()
