"""
plot_switching_overhead.py

Generates switching time overhead comparison plots:
  1. PAIRWISE HEATMAP: Hot-switch time matrix across all model pairs
  2. COLD-LOAD vs HOT-SWITCH: Initial model load vs hot-switch overhead
  3. SCHEME SWITCHING OVERHEAD: Estimated per-request switching overhead by scheme
  4. OVERHEAD FRACTION: Switching overhead as % of inference latency

How to run
    Recommended (regenerates every v6 figure):
        ./run_all_experiments.sh --plots

    Directly, from the project root:
        PYTHONPATH=src python src/plots/plot_switching_overhead.py

    Inputs:
        results/model_benchmark/plot_data_baseline/switch_hot/pairwise_switching.csv
        metrics/model_load_metrics.csv
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
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PAIRWISE_CSV = DATA_DIR / "switch_hot" / "pairwise_switching.csv"
LOAD_CSV = PROJECT_ROOT / "metrics" / "model_load_metrics.csv"

MODEL_ID_ORDER = ["SD-v1-5", "SD-2-1", "SDXL-Base", "SD-3-5-Medium", "SD-3-5-Large"]
MODEL_LABELS = {
    "SD-v1-5": "v1.5\n(3)",
    "SD-2-1": "2.1\n(4)",
    "SDXL-Base": "XL\n(5)",
    "SD-3-5-Medium": "3.5-M\n(8)",
    "SD-3-5-Large": "3.5-L\n(9)",
}

SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_STYLES = {
    "clipper_large": {"color": "#d62728", "hatch": "//", "label": "Clipper Large-Only"},
    "clipper_small": {"color": "#ff7f0e", "hatch": "\\\\", "label": "Clipper Small-Only"},
    "infaas":        {"color": "#1f77b4", "hatch": "xx", "label": "INFaaS"},
    "difs_v3":       {"color": "#2ca02c", "hatch": "//", "label": "DIFS\\_v3 (Ours)"},
}
HATCHES = ["//", "\\\\", "xx", "//"]

LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]


def load_pairwise():
    rows = []
    with open(PAIRWISE_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            from_model = row[0]
            for i, to_model in enumerate(MODEL_ID_ORDER[1:] if len(header) > 5 else header[1:]):
                if i + 1 < len(row):
                    rows.append({
                        "from": from_model,
                        "to": header[i + 1].strip(),
                        "switch_time_ms": float(row[i + 1]),
                    })
    return rows


def load_pairwise_matrix():
    matrix = {}
    with open(PAIRWISE_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        to_models = [h.strip() for h in header[1:]]
        for row in reader:
            from_model = row[0].strip()
            matrix[from_model] = {}
            for i, val in enumerate(row[1:]):
                matrix[from_model][to_models[i]] = float(val)
    return matrix


def load_cold_times():
    loads = {}
    with open(LOAD_CSV) as f:
        for r in csv.DictReader(f):
            model = r["model"]
            t = float(r["load_time"])
            if model not in loads or t > loads[model]:
                loads[model] = t
    return loads


COLD_MODEL_MAP = {
    "stable-diffusion-v1-5": "SD-v1-5",
    "stable-diffusion-2-1": "SD-2-1",
    "stable-diffusion-xl-base-1.0": "SDXL-Base",
    "stable-diffusion-3.5-medium": "SD-3-5-Medium",
    "stable-diffusion-3.5-large": "SD-3-5-Large",
}

COLD_INFERENCE_SEC = {
    "SD-v1-5": 5.0,
    "SD-2-1": 7.0,
    "SDXL-Base": 15.0,
    "SD-3-5-Medium": 35.0,
    "SD-3-5-Large": 74.0,
}


def plot_pairwise_heatmap(out_path):
    matrix = load_pairwise_matrix()
    from_models = [m for m in MODEL_ID_ORDER if m in matrix]
    n = len(from_models)

    data = np.zeros((n, n))
    for i, fm in enumerate(from_models):
        for j, tm in enumerate(MODEL_ID_ORDER):
            data[i][j] = matrix.get(fm, {}).get(tm, 0.0)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(data, cmap="YlOrRd", aspect="equal", vmin=0, vmax=max(0.05, data.max()))

    ax.set_xticks(range(n))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ID_ORDER], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in from_models], fontsize=10)
    ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Current Model", fontsize=12, fontweight="bold")
    ax.set_title("Hot-Switch Time (ms) Across Model Pairs", fontsize=13, fontweight="bold")

    for i in range(n):
        for j in range(n):
            val = data[i][j]
            color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Switch Time (ms)", fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.88, top=0.92, bottom=0.08)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_cold_vs_hot(out_path):
    cold_raw = load_cold_times()
    cold = {}
    for raw_name, short in COLD_MODEL_MAP.items():
        if raw_name in cold_raw:
            cold[short] = cold_raw[raw_name]

    avg_hot = 0.012
    models = [m for m in MODEL_ID_ORDER if m in cold]

    x = np.arange(len(models))
    cold_vals = [cold[m] for m in models]
    hot_vals = [avg_hot] * len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35

    bars1 = ax.bar(x - bar_width / 2, cold_vals, bar_width,
                   label="Cold Load (Initial)", color="#d62728",
                   edgecolor="black", linewidth=0.5, hatch="//", zorder=3)
    bars2 = ax.bar(x + bar_width / 2, hot_vals, bar_width,
                   label="Hot Switch (Runtime)", color="#2ca02c",
                   edgecolor="black", linewidth=0.5, hatch="xx", zorder=3)

    for bar, v in zip(bars1, cold_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, v in zip(bars2, hot_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v*1000:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models], fontsize=10)
    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Cold Load vs Hot-Switch Time by Model", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 20)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9, edgecolor="black")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def _annotate_bar_levels(ax, bars, values, fmt):
    """
    Annotate bar heights using y-axis-anchored markers with leader lines to
    each bar, instead of writing the numbers directly above the bars (which
    clash when bars are short/crowded).  Each level is written once in a
    small font in the left margin, marked on the axis, and connected to the
    corresponding bar by a dashed leader line.
    fmt: callable(value) -> label string.
    """
    x_axis = bars[0].get_x() - 0.7
    seen_y = set()
    for bar, v in zip(bars, values):
        xc = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        label = fmt(v)
        # dashed leader line from the axis to the bar at the value's level
        ax.plot([x_axis, xc], [y, y], color="#444444", lw=0.7,
                ls=":", zorder=1)
        # axis tick marking the level
        ax.plot([x_axis - 0.04, x_axis], [y, y], color="black", lw=1.0,
                zorder=4)
        # small-font label in the left margin at the level
        dy = 0.0
        if y in seen_y:
            # nudge stacked identical levels vertically so labels don't overlap
            dy = (y + 0.006) * 1.0 if y != 0 else 0.006
        seen_y.add(y)
        ax.text(x_axis - 0.14, y + dy, label, ha="right", va="center",
                fontsize=7, color="black", zorder=4)


def plot_scheme_switching_overhead(out_path):
    """
    Estimated switching overhead per request by scheme.
    clipper_large/small: pinned, 0 switches → 0 overhead
    infaas: cost-driven switches, ~0.01ms per switch
    difs_v3: adaptation switches + locality-aware → fewer effective switches
    """
    switch_counts = {
        "clipper_large": 0,
        "clipper_small": 0,
        "infaas": 3,
        "difs_v3": 2,
    }
    hot_switch_ms = 0.012

    x = np.arange(len(SYSTEM_ORDER))
    overhead_vals = [switch_counts[s] * hot_switch_ms for s in SYSTEM_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, overhead_vals, 0.5,
                  edgecolor="black", linewidth=0.5, zorder=3)
    for i, bar in enumerate(bars):
        bar.set_facecolor(SYSTEM_STYLES[SYSTEM_ORDER[i]]["color"])
        bar.set_hatch(HATCHES[i])

    ax.set_xlim(-1.3, len(SYSTEM_ORDER) - 0.5)
    ax.set_ylim(0, max(overhead_vals) * 1.3)
    _annotate_bar_levels(ax, bars, overhead_vals,
                         lambda v: "0" if v == 0 else f"{v:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_STYLES[s]["label"] for s in SYSTEM_ORDER], fontsize=10)
    ax.set_ylabel("Estimated Switching Overhead (ms/request)", fontsize=11, fontweight="bold")
    ax.set_title("Model-Switching Overhead by Scheme", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_overhead_fraction(out_path):
    """
    Switching overhead as percentage of typical inference latency for each model.
    """
    models = MODEL_ID_ORDER
    hot_switch_ms = 0.012
    inference_ms = [COLD_INFERENCE_SEC[m] * 1000 for m in models]
    overhead_pct = [hot_switch_ms / inf * 100 for inf in inference_ms]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(x, overhead_pct, 0.5, color=colors,
                  edgecolor="black", linewidth=0.5, zorder=3)

    ax.set_xlim(-1.3, len(models) - 0.5)
    ax.set_yscale("log")
    ax.set_ylim(min(overhead_pct) * 0.5, max(overhead_pct) * 5)
    _annotate_bar_levels(ax, bars, overhead_pct,
                         lambda v: f"{v:.2e}")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in models], fontsize=10)
    ax.set_ylabel("Switching Overhead (% of Inference)", fontsize=11, fontweight="bold")
    ax.set_title("Hot-Switch Overhead as Fraction of Inference Latency", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    fig.subplots_adjust(left=0.14, right=0.97, top=0.90, bottom=0.10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plot_pairwise_heatmap(PLOT_DIR / "switching_pairwise_heatmap.png")
    plot_cold_vs_hot(PLOT_DIR / "switching_cold_vs_hot.png")
    plot_scheme_switching_overhead(PLOT_DIR / "switching_overhead_by_scheme.png")
    plot_overhead_fraction(PLOT_DIR / "switching_overhead_fraction.png")
    print("[PLOT] All switching time plots generated.")


if __name__ == "__main__":
    main()
