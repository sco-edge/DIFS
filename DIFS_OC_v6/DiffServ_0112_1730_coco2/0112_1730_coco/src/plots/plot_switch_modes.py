#!/usr/bin/env python3
"""
plot_switch_modes.py

DIFS_OC_v5: generates switching-time plots from the pairwise switching
measurements run under the three opt-in worker switch modes:

  hot          : cache pointer flip (baseline, ~0.01 ms)
  force_upload : cache hit + genuine DRAM->GPU weight re-upload + sync
  cold_every   : always reload target model from disk (never reuse cache)

Reads results/model_benchmark/plot_data_baseline/switch_<mode>/pairwise_switching.csv
for each mode and writes plots into .../plots/.

Plots generated:
  1. Per-mode pairwise heatmaps
      - switching_pairwise_hot.png           (ms)
      - switching_pairwise_force_upload.png  (ms)
      - switching_pairwise_cold_every.png    (s)
  2. Cross-mode grouped bars (median switch time by target model, log scale)
      - switching_cross_mode_by_target.png
  3. Overall median / range comparison across modes
      - switching_mode_overview.png
  4. "Sweep" grouped heatmap: mode x target-model median switch time
      - switching_mode_target_heatmap.png   (log colour, ms)

How to run
    Recommended (regenerates every v6 figure):
        ./run_all_experiments.sh --plots

    Directly, from the project root:
        PYTHONPATH=src python src/plots/plot_switch_modes.py

    Inputs:
        results/model_benchmark/plot_data_baseline/switch_<mode>/pairwise_switching.csv
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"
PLOT_DIR = DATA_DIR / "plots"

MODES = ["hot", "force_upload", "cold_every"]
MODE_LABELS = {
    "hot": "hot\n(cache flip)",
    "force_upload": "force\\_upload\n(op.2)",
    "cold_every": "cold\\_every\n(op.3)",
}
MODE_UNITS = {"hot": "ms", "force_upload": "ms", "cold_every": "s"}
MODE_DIVISOR = {"hot": 1.0, "force_upload": 1.0, "cold_every": 1000.0}
MODE_COLORS = {
    "hot": "#1f77b4",
    "force_upload": "#ff7f0e",
    "cold_every": "#d62728",
}

MODEL_ID_ORDER = ["SD-v1-5", "SD-2-1", "SDXL-Base", "SD-3-5-Medium", "SD-3-5-Large"]
MODEL_LABELS = {
    "SD-v1-5": "v1.5\n(3)",
    "SD-2-1": "2.1\n(4)",
    "SDXL-Base": "XL\n(5)",
    "SD-3-5-Medium": "3.5-M\n(8)",
    "SD-3-5-Large": "3.5-L\n(9)",
}

# Hatch per mode for B&W readability (matches v4 hatch vocabulary).
MODE_HATCHES = {"hot": "", "force_upload": "///", "cold_every": "xxx"}
GRID_KW = dict(axis="y", which="both", linestyle="--", linewidth=0.5,
               alpha=0.35, color="#777777")


def _annotate_bar_levels(ax, bars, values, fmt, margin=0.7):
    """
    Identify each bar's level using the axis + grid instead of a big bold label
    on top of every bar (which becomes unreadable when bars are short/crowded).
    A thin small-font label is written in the left margin at the bar's height,
    marked by an axis tick, and joined to the bar by a dashed leader line.
    fmt: callable(value) -> label string.
    """
    anchor = bars[0].get_x() - margin
    seen = {}
    for bar, v in zip(bars, values):
        xc = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        # dashed leader line from the axis to the bar at the value's level
        ax.plot([anchor, xc], [y, y], color="#444444", lw=0.6, ls=":", zorder=1)
        # axis tick marking the level
        ax.plot([anchor - 0.04, anchor], [y, y], color="black", lw=1.0, zorder=4)
        # thin small-font label in the left margin at the level (nudge repeats)
        dy = 0.0
        if y in seen:
            seen[y] += 1
            dy = seen[y] * 0.015 * y if y != 0 else 0.02
        else:
            seen[y] = 0
        ax.text(anchor - 0.14, y + dy, fmt(v), ha="right", va="center",
                fontsize=7, color="black", zorder=4)


def load_matrix(mode):
    path = DATA_DIR / f"switch_{mode}" / "pairwise_switching.csv"
    matrix = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        to_models = [h.strip() for h in header[1:]]
        for row in reader:
            fm = row[0].strip()
            matrix[fm] = {}
            for i, val in enumerate(row[1:]):
                matrix[fm][to_models[i]] = float(val) / MODE_DIVISOR[mode]
    return matrix


def _imshow_matrix(ax, matrix, display_models, fmt, cmap, vmin, vmax,
                   title, cb_label, text_color_thresh=None, axis_labels=True):
    n = len(display_models)
    data = np.zeros((n, n))
    for i, fm in enumerate(display_models):
        for j, tm in enumerate(display_models):
            data[i][j] = matrix.get(fm, {}).get(tm, 0.0)

    im = ax.imshow(data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in display_models], fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in display_models], fontsize=10)
    if axis_labels:
        ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("Current Model", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if text_color_thresh is not None:
        for i in range(n):
            for j in range(n):
                val = data[i][j]
                color = "white" if val > text_color_thresh else "black"
                ax.text(j, i, fmt(val), ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cb_label, fontsize=11)
    return data


def plot_pairwise_heatmaps():
    for mode in MODES:
        matrix = load_matrix(mode)
        if not matrix:
            continue
        units = MODE_UNITS[mode]
        div = MODE_DIVISOR[mode]
        all_vals = np.array([matrix[f][t] for f in matrix for t in matrix[f] if f != t])
        vmax = 0.95 * all_vals.max() if all_vals.size else 1.0
        vmin = 0.0

        if mode == "hot":
            cmap, thresh, fmt = "YlOrRd", vmax * 0.6, lambda v: f"{v:.2f}"
        elif mode == "force_upload":
            cmap, thresh, fmt = "YlOrRd", vmax * 0.7, lambda v: f"{v:.0f}"
        else:  # cold_every
            cmap, thresh, fmt = "YlOrBr", vmax * 0.7, lambda v: f"{v:.1f}"

        fig, ax = plt.subplots(figsize=(8.5, 7))
        title = {
            "hot": "Hot-Switch Time (ms) Across Model Pairs",
            "force_upload": f"force\\_upload Switch Time ({units}) Across Model Pairs",
            "cold_every": f"cold\\_every Switch Time ({units}) Across Model Pairs",
        }[mode]
        _imshow_matrix(ax, matrix, MODEL_ID_ORDER, fmt, cmap, vmin, vmax,
                       title, f"Switch Time ({units})", text_color_thresh=thresh)
        fig.subplots_adjust(left=0.15, right=0.9, top=0.92, bottom=0.09)
        fig.savefig(PLOT_DIR / f"switching_pairwise_{mode}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved switching_pairwise_{mode}.png")


def median_by_target(mode):
    matrix = load_matrix(mode)
    med = {}
    for tm in MODEL_ID_ORDER:
        vals = [matrix[f][tm] for f in matrix if f != tm]
        med[tm] = float(np.median(vals)) if vals else 0.0
    return med


def plot_cross_mode_by_target():
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(MODEL_ID_ORDER))
    width = 0.26
    for k, mode in enumerate(MODES):
        med = median_by_target(mode)
        vals = [med[m] for m in MODEL_ID_ORDER]
        bars = ax.bar(x + (k - 1) * width, vals, width,
                      color=MODE_COLORS[mode], hatch=MODE_HATCHES[mode],
                      edgecolor="black", linewidth=0.5, zorder=3,
                      label=MODE_LABELS[mode].replace("\\", ""))
        for bar, v in zip(bars, vals):
            lab = f"{v:.1f}" if mode == "cold_every" else (f"{v:.0f}" if v >= 1 else f"{v:.2f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.03,
                    lab, ha="center", va="bottom", fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in MODEL_ID_ORDER],
                       fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 20000)
    ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Switch Time", fontsize=12, fontweight="bold")
    ax.set_title("Median Switch Time by Target Model Across Switch Modes (log)",
                 fontsize=13, fontweight="bold")
    ax.grid(**GRID_KW)
    ax.legend(fontsize=10, framealpha=0.9, edgecolor="black", loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "switching_cross_mode_by_target.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Saved switching_cross_mode_by_target.png")


def plot_mode_overview():
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODES))
    medians, mins, maxs = [], [], []
    for mode in MODES:
        med = median_by_target(mode)
        vals = np.array([med[m] for m in MODEL_ID_ORDER])
        medians.append(np.median(vals))
        mins.append(vals.min() if vals.size else 0)
        maxs.append(vals.max() if vals.size else 0)

    med_bars, max_bars = [], []
    for i, mode in enumerate(MODES):
        c = MODE_COLORS[mode]
        max_bars.extend(ax.bar(x[i], maxs[i] if maxs[i] > 0 else 1e-3, 0.5,
                               color=c, alpha=0.35, hatch=MODE_HATCHES[mode],
                               edgecolor="black", linewidth=0.5, zorder=2))
        med_bars.extend(ax.bar(x[i], max(medians[i], 1e-3), 0.5, color=c,
                               alpha=0.95, edgecolor="black", linewidth=0.5,
                               zorder=3))
        # thin label for the max only (identified via the axis level)
        ax.text(x[i], maxs[i] * 1.18, f"max {maxs[i]:.1f}{MODE_UNITS[mode]}",
                ha="center", fontsize=7, color="black")

    _annotate_bar_levels(ax, med_bars, [max(m, 1e-3) for m in medians],
                         lambda v: f"med {v:.2f}")
    ax.set_xlim(-1.3, len(MODES) - 0.5)

    ax.set_yscale("log")
    ax.set_ylim(0.005, 50000)
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[m].replace("\\", "") for m in MODES], fontsize=11)
    ax.set_ylabel("Switch Time (log)", fontsize=12, fontweight="bold")
    ax.set_title("Aggregate Switch Time Overview by Mode", fontsize=13, fontweight="bold")
    ax.grid(**GRID_KW)
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", alpha=0.95),
         plt.Rectangle((0, 0), 1, 1, fc="#1f77b4", alpha=0.35)],
        ["median", "max range"], fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "switching_mode_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Saved switching_mode_overview.png")


def plot_mode_target_heatmap():
    n_modes = len(MODES)
    data = np.full((n_modes, len(MODEL_ID_ORDER)), np.nan)
    for k, mode in enumerate(MODES):
        med = median_by_target(mode)
        for j, m in enumerate(MODEL_ID_ORDER):
            data[k][j] = med[m]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(data, cmap="viridis", aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=np.nanmin(data),
                                                  vmax=np.nanmax(data)))
    ax.set_xticks(range(len(MODEL_ID_ORDER)))
    ax.set_xticklabels([MODEL_LABELS.get(m, m).replace("\n", " ") for m in MODEL_ID_ORDER],
                       fontsize=10)
    ax.set_yticks(range(n_modes))
    ax.set_yticklabels([MODE_LABELS[m].replace("\\", "") for m in MODES], fontsize=10)
    ax.set_xlabel("Target Model", fontsize=12, fontweight="bold")
    ax.set_title("Median Switch Time (ms, log colour) by Mode and Target Model",
                 fontsize=13, fontweight="bold")
    for k in range(n_modes):
        for j in range(len(MODEL_ID_ORDER)):
            v = data[k][j]
            ms = v if MODES[k] != "cold_every" else v * 1000
            ax.text(j, k, f"{ms:.0f}" if ms >= 1 else f"{ms:.2f}",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if (np.log10(v) - np.log10(np.nanmin(data))) > 0.5 * (
                        np.log10(np.nanmax(data)) - np.log10(np.nanmin(data))) else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Median Switch Time (ms, log)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "switching_mode_target_heatmap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Saved switching_mode_target_heatmap.png")


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pairwise_heatmaps()
    plot_cross_mode_by_target()
    plot_mode_overview()
    plot_mode_target_heatmap()
    print("[PLOT] All v5 switch-mode plots generated.")


if __name__ == "__main__":
    main()
