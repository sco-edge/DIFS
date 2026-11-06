#!/usr/bin/env python3
"""
generate_switching_plots.py

Reads switching time CSV from results/model_benchmark/plot_data/ and generates
model switching time plots (bar + line).

Usage:
    python generate_switching_plots.py [--plot-data-dir DIR] [--output-dir DIR]
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_ORDER = ["SD-v1.4-dir", "SD-v1.4-safetensors", "SD-v1.5", "SD-2.1", "SDXL-Base", "SD-3.5-Medium", "SD-3.5-Large", "DreamShaper-8"]
HATCHES = ["//", "\\\\", "xx"]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_DATA = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plot_data")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plots")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Horizontal": "#1f77b4", "Vertical": "#ff7f0e", "Hybrid": "#2ca02c"}
GRID_KW = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, color="#cccccc")


def _load_csv(plot_data, name):
    path = os.path.join(plot_data, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found, skipping.", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def _mode_values(mode_data, models, col):
    """Reindex per-mode values to the full model list (NaN for missing combos)."""
    return mode_data.set_index("model").reindex(models)[col].to_numpy()


# ── PLOT 1: Grouped Bar Chart ────────────────────────────────────────────────
def plot_switching_bar(plot_data, output_dir):
    df = _load_csv(plot_data, "switching_time_bar.csv")
    if df is None or len(df) == 0:
        print("  Skipping switching_time_bar (no data)")
        return

    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode]
        color_key = mode.capitalize()
        bars = ax.bar(
            x + offsets[i], _mode_values(mode_data, models, "switch_time_sec"), w,
            label=mode.capitalize(), color=COLORS.get(color_key, "#333"),
            edgecolor="white", hatch=HATCHES[i % len(HATCHES)],
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Switch Time (seconds)", fontsize=11)
    ax.set_title("Model Switching Time (Load Time) by Model and Scaling Mode",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, df["switch_time_sec"].max() * 1.15)
    ax.grid(**GRID_KW)
    fig.tight_layout()

    path = os.path.join(output_dir, "switching_time_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: switching_time_bar.png  ({os.path.getsize(path):,} bytes)")


# ── PLOT 2: Line Plot ────────────────────────────────────────────────────────
def plot_switching_line(plot_data, output_dir):
    df = _load_csv(plot_data, "switching_time_line.csv")
    if df is None or len(df) == 0:
        print("  Skipping switching_time_line (no data)")
        return

    models = df["model"].values
    x = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        color_key = mode.capitalize()
        ax.plot(
            x, vals, marker=markers[i % len(markers)], markersize=8, linewidth=2,
            label=mode.capitalize(), color=COLORS.get(color_key, "#333"),
        )
        for j, v in enumerate(vals):
            if not np.isnan(v):
                offset_y = 8 if mode.capitalize() != "Vertical" else -14
                ax.annotate(f"{v:.2f}s", (j, v), textcoords="offset points",
                            xytext=(0, offset_y), ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Switch Time (seconds)", fontsize=11)
    ax.set_title("Model Switching Time (Load Time) by Scaling Mode",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.18)
    ax.grid(**GRID_KW)
    fig.tight_layout()

    path = os.path.join(output_dir, "switching_time_line.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: switching_time_line.png  ({os.path.getsize(path):,} bytes)")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate model switching time plots.")
    parser.add_argument("--plot-data-dir", default=None, help="Path to plot_data/ directory")
    parser.add_argument("--output-dir", default=None, help="Path to output plots directory")
    args = parser.parse_args()

    plot_data = args.plot_data_dir or DEFAULT_PLOT_DATA
    output_dir = args.output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    print("Generating switching time plots ...")
    plot_switching_bar(plot_data, output_dir)
    plot_switching_line(plot_data, output_dir)
    # ── PLOT 3: Pairwise Switching Time Heatmap ──────────────────────────────
    plot_pairwise_heatmap(plot_data, output_dir)

    print(f"\nSwitching plots saved to:\n  {output_dir}")


# ── PLOT 3: Pairwise Switching Time Heatmap ──────────────────────────────────
def plot_pairwise_heatmap(plot_data, output_dir):
    """
    Reads pairwise_switching.csv (N x N matrix) or derives it from load times.
    Plots a heatmap with model names on both axes.
    """
    pw_path = os.path.join(plot_data, "pairwise_switching.csv")
    derived = False

    if os.path.exists(pw_path):
        # Read N x N matrix CSV (first column = from_model, rest = to_model)
        df_raw = pd.read_csv(pw_path, index_col=0)
        model_names = list(df_raw.columns)
        matrix = df_raw.values.astype(float)
        print(f"  Loaded pairwise switching matrix: {pw_path}")
    else:
        # Derive from switching_time_bar.csv: off-diag = target load time, diag = 0
        bar_path = os.path.join(plot_data, "switching_time_bar.csv")
        if not os.path.exists(bar_path):
            print("  Skipping pairwise heatmap (no data)")
            return
        df_bar = pd.read_csv(bar_path)
        # Average across scaling modes
        avg_load = df_bar.groupby("model")["switch_time_sec"].mean().reset_index()
        avg_load.columns = ["model", "load_time_sec"]
        avg_load["model"] = pd.Categorical(avg_load["model"], categories=MODEL_ORDER, ordered=True)
        avg_load = avg_load.sort_values("model").reset_index(drop=True)
        model_names = avg_load["model"].tolist()
        n = len(model_names)
        # Off-diagonal: target model's load time; Diagonal: 0
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = avg_load.iloc[j]["load_time_sec"]
        derived = True
        print("  Derived pairwise switching matrix from average load times")

    n = len(model_names)
    short = [m.split("/")[0] if "/" in m else m for m in model_names]
    # Truncate long names
    short = [s[:18] + ".." if len(s) > 20 else s for s in short]

    # ── Heatmap ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), max(8, n * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(
                j, i, f"{val:.1f}s",
                ha="center", va="center", fontsize=7,
                color=text_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Switch Time (seconds)", fontsize=10)

    source_label = "(derived from load times)" if derived else "(measured via gRPC)"
    ax.set_title(
        f"Pairwise Model Switching Time {source_label}\n"
        "From Model (row) → To Model (column)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("To Model", fontsize=11)
    ax.set_ylabel("From Model", fontsize=11)
    fig.tight_layout()

    path = os.path.join(output_dir, "pairwise_switching_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: pairwise_switching_heatmap.png  ({os.path.getsize(path):,} bytes)")

    # ── Matrix table (CSV) ───────────────────────────────────────────────────
    out_csv = os.path.join(plot_data, "plot_pairwise_switching.csv")
    df_out = pd.DataFrame(matrix, index=short, columns=short)
    df_out.to_csv(out_csv)
    print(f"  Saved: plot_pairwise_switching.csv")


if __name__ == "__main__":
    main()
