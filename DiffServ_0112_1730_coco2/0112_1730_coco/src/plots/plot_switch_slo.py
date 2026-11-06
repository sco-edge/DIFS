#!/usr/bin/env python3
"""
plot_switch_slo.py

DIFS_OC_v5: generates SLO-context plots for the model-switching experiment.

The v5 switching experiment measures how much time a model switch costs under the
three worker switch modes (hot / force_upload / cold_every).  The framework's
latency SLO is a fixed budget (60,000 ms) for every model and step count; the
accuracy SLO is defined by step count and is 100% for all systems (switching mode
does not change step counts, so it does not affect the accuracy SLO).

These plots place the measured switch times against the fixed latency SLO budget so
a scheduler can decide how many / which switches fit within budget:

  1. switching_slo_budget_fraction.png : switch time as % of the 60 s latency SLO
       (one bar per (mode, target model), log scale)
  2. switching_slo_affordable_switches.png : how many switches fit inside the 60 s
       budget (median, log scale)
  3. switching_slo_vs_budget.png : median + max switch time vs. the 60 s budget line
       on a single log axis

Uses v5's own measured data (switch_<mode>/pairwise_switching.csv) and the fixed
framework SLO constants only.  No benchmark output is imported from other projects.

Usage:
    DIFS_Gen/bin/python src/plots/plot_switch_slo.py
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
    print("[PLOT] matplotlib not available; skipping SLO plots.")
    sys.exit(0)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"
PLOT_DIR = DATA_DIR / "plots"

LATENCY_SLO_MS = 60000.0  # fixed 60 s latency SLO for all models / step counts

MODES = ["hot", "force_upload", "cold_every"]
MODE_LABELS = {
    "hot": "hot\n(cache flip)",
    "force_upload": "force\\_upload\n(op.2)",
    "cold_every": "cold\\_every\n(op.3)",
}
MODE_COLORS = {
    "hot": "#1f77b4",
    "force_upload": "#ff7f0e",
    "cold_every": "#d62728",
}
MODE_UNITS = {"hot": "ms", "force_upload": "ms", "cold_every": "s"}
MODE_DIVISOR = {"hot": 1.0, "force_upload": 1.0, "cold_every": 1000.0}
MODE_HATCHES = {"hot": "", "force_upload": "///", "cold_every": "xxx"}
GRID_KW = dict(axis="y", which="both", linestyle="--", linewidth=0.5,
               alpha=0.35, color="#777777")


def _annotate_bar_levels(ax, bars, values, fmt, margin=0.7):
    """
    Identify each bar's level through the axis + grid (not a big bold label on
    top of every bar).  A thin small-font label is written in the left margin at
    the bar's height, marked by an axis tick, and joined to the bar by a dashed
    leader line.  fmt: callable(value) -> label string.
    """
    anchor = bars[0].get_x() - margin
    seen = {}
    for bar, v in zip(bars, values):
        xc = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax.plot([anchor, xc], [y, y], color="#444444", lw=0.6, ls=":", zorder=1)
        ax.plot([anchor - 0.04, anchor], [y, y], color="black", lw=1.0, zorder=4)
        dy = 0.0
        if y in seen:
            seen[y] += 1
            dy = seen[y] * 0.015 * y if y != 0 else 0.02
        else:
            seen[y] = 0
        ax.text(anchor - 0.14, y + dy, fmt(v), ha="right", va="center",
                fontsize=7, color="black", zorder=4)

MODEL_ID_ORDER = ["SD-v1-5", "SD-2-1", "SDXL-Base", "SD-3-5-Medium", "SD-3-5-Large"]
MODEL_LABELS = {
    "SD-v1-5": "v1.5 (3)",
    "SD-2-1": "2.1 (4)",
    "SDXL-Base": "XL (5)",
    "SD-3-5-Medium": "3.5-M (8)",
    "SD-3-5-Large": "3.5-L (9)",
}


def load_matrix(mode):
    path = DATA_DIR / f"switch_{mode}" / "pairwise_switching.csv"
    matrix = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        tos = [h.strip() for h in header[1:]]
        for row in reader:
            fm = row[0].strip()
            matrix[fm] = {}
            for i, val in enumerate(row[1:]):
                matrix[fm][tos[i]] = float(val) / MODE_DIVISOR[mode]
    return matrix


def median_by_target(mode):
    matrix = load_matrix(mode)
    med = {}
    for tm in MODEL_ID_ORDER:
        vals = [matrix[f][tm] for f in matrix if f != tm]
        med[tm] = float(np.median(vals)) if vals else 0.0
    return med


def plot_budget_fraction():
    """Switch time as a percentage of the 60 s latency SLO."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(MODEL_ID_ORDER))
    width = 0.26
    for k, mode in enumerate(MODES):
        med = median_by_target(mode)
        pct = [med[m] * 100.0 / LATENCY_SLO_MS for m in MODEL_ID_ORDER]
        bars = ax.bar(x + (k - 1) * width, pct, width,
                      color=MODE_COLORS[mode], hatch=MODE_HATCHES[mode],
                      edgecolor="black", linewidth=0.5, zorder=3,
                      label=MODE_LABELS[mode].replace("\\", ""))
        for bar, v in zip(bars, pct):
            lab = f"{v:.2e}" if v > 0 and (v < 0.01 or v >= 10) else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.2,
                    lab, ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ID_ORDER], fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 100)
    ax.set_ylabel("Switch Time (% of 60 s Latency SLO)", fontsize=12, fontweight="bold")
    ax.set_title("Model-Switch Cost as Fraction of the 60 s Latency SLO Budget",
                 fontsize=13, fontweight="bold")
    ax.grid(**GRID_KW)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="black", loc="lower right")
    fig.tight_layout()
    out = PLOT_DIR / "switching_slo_budget_fraction.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out.name}")


def plot_affordable_switches():
    """How many switches fit inside the 60 s latency SLO window (median)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MODES))
    counts = []
    for mode in MODES:
        med = median_by_target(mode)
        # worst (largest) median switch — the least affordable case
        c = LATENCY_SLO_MS / max(med.values())
        counts.append(c)
    bars = ax.bar(x, counts, 0.5, color=[MODE_COLORS[m] for m in MODES],
                  hatch=[MODE_HATCHES[m] for m in MODES],
                  edgecolor="black", linewidth=0.5, zorder=3)
    _annotate_bar_levels(ax, bars, counts, lambda v: f"{v:,.0f}")
    ax.set_xlim(-1.3, len(MODES) - 0.5)
    ax.set_yscale("log")
    ax.set_ylim(1, counts[0] * 3)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m].replace("\\", "") for m in MODES], fontsize=11)
    ax.set_ylabel("Affordable Switches within 60 s Latency SLO",
                  fontsize=12, fontweight="bold")
    ax.set_title("Model Switches Affordable within the 60 s Latency SLO",
                 fontsize=13, fontweight="bold")
    ax.grid(**GRID_KW)
    fig.tight_layout()
    out = PLOT_DIR / "switching_slo_affordable_switches.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out.name}")


def plot_vs_budget():
    """Median & max switch time per mode vs. the 60 s latency SLO line on one log axis."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(MODES))
    for k, mode in enumerate(MODES):
        med = median_by_target(mode)
        med_v = float(np.median(list(med.values())))
        max_v = float(max(med.values()))
        unit = MODE_UNITS[mode]
        ax.plot([x[k] - 0.15, x[k] + 0.15], [med_v, med_v], color=MODE_COLORS[mode],
                lw=4, zorder=3, label=None)
        ax.scatter([x[k], x[k]], [med_v, max_v], marker="_", s=160, lw=2,
                   color=MODE_COLORS[mode], zorder=4)
        ax.text(x[k] - 0.26, med_v, f"med {med_v:.2f} {unit}", ha="right", va="center",
                fontsize=8, fontweight="bold", color=MODE_COLORS[mode])
        ax.text(x[k] + 0.26, max_v, f"max {max_v:.2f} {unit}", ha="left", va="center",
                fontsize=8, fontweight="bold", color=MODE_COLORS[mode])

    ax.axhline(LATENCY_SLO_MS / 1000.0, color="black", lw=2, ls="--", zorder=1)
    ax.text(len(MODES) - 0.5, LATENCY_SLO_MS / 900.0, "60 s latency SLO budget",
            fontsize=10, fontweight="bold", ha="right", color="black")

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1e5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m].replace("\\", "") for m in MODES], fontsize=11)
    ax.set_ylabel("Switch Time (s, log)", fontsize=12, fontweight="bold")
    ax.set_title("Median and Max Model-Switch Time vs. the 60 s Latency SLO",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = PLOT_DIR / "switching_slo_vs_budget.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out.name}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_budget_fraction()
    plot_affordable_switches()
    plot_vs_budget()
    print("\n[PLOT] v5 switching SLO plots generated.")


if __name__ == "__main__":
    main()
