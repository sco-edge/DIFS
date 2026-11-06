#!/usr/bin/env python3
"""
generate_line_plots.py

Reads per-plot CSVs from results/model_benchmark/plot_data/ and generates
line plot visualizations.

Usage:
    python generate_line_plots.py [--plot-data-dir DIR] [--output-dir DIR]
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_DATA = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plot_data")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plots")

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150

MARKERS = ["o", "s", "^"]
COLORS = ["#4C72B0", "#55A868", "#C44E52"]
GRID_KW = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, color="#cccccc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name, output_dir):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}  ({os.path.getsize(path) / 1024:.1f} KB)")


def _annotate_points(ax, x, vals, fmt="{:.2f}", fontsize=8, offset=8):
    for xi, v in zip(x, vals):
        if v is not None and not np.isnan(v):
            ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, offset),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=fontsize)


def _load_csv(plot_data, name):
    path = os.path.join(plot_data, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found, skipping.", file=sys.stderr)
        return None
    return pd.read_csv(path)


def _load_csv_required(plot_data, name):
    df = _load_csv(plot_data, name)
    if df is None:
        raise FileNotFoundError(f"Required data file missing: {name}")
    return df


# ===================================================================
# Plot 1 – Inference Latency Comparison (line)
# ===================================================================
def plot_latency_comparison(plot_data, output_dir):
    df = _load_csv_required(plot_data, "line_1_latency_comparison.csv")
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.2, markersize=8, label=mode.capitalize())
        _annotate_points(ax, X, vals)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Inference Latency (seconds)", fontsize=12)
    ax.set_title("Inference Latency by Model and Scaling Mode",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.12)
    ax.grid(**GRID_KW)
    _save(fig, "line_latency_comparison.png", output_dir)


# ===================================================================
# Plot 2 – Model Load Time (line)
# ===================================================================
def plot_model_load_time(plot_data, output_dir):
    df = _load_csv(plot_data, "line_2_model_load_time.csv")
    if df is None or len(df) == 0:
        print("  Skipping line_model_load_time (no data)")
        return
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.2, markersize=8, label=mode.capitalize())
        _annotate_points(ax, X, vals, fmt="{:.2f}")
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Model Load Time (seconds)", fontsize=12)
    ax.set_title("Model Load Time by Scaling Mode",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.18)
    ax.grid(**GRID_KW)
    _save(fig, "line_model_load_time.png", output_dir)


# ===================================================================
# Plot 3 – GPU Memory Usage (line)
# ===================================================================
def plot_gpu_memory(plot_data, output_dir):
    df = _load_csv(plot_data, "line_3_gpu_memory.csv")
    if df is None or len(df) == 0:
        print("  Skipping line_gpu_memory (no data)")
        return
    X = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(X, df["gpu_used_mb"].values, marker="o", color="#4C72B0",
            linewidth=2.2, markersize=8, label="Used")
    ax.plot(X, df["gpu_peak_mb"].values, marker="s", color="#DD8452",
            linewidth=2.2, markersize=8, label="Peak")
    _annotate_points(ax, X, df["gpu_used_mb"].values, fmt="{:.1f}", offset=10)
    _annotate_points(ax, X, df["gpu_peak_mb"].values, fmt="{:.1f}", offset=10)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["model"].values], fontsize=10)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage per Model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, df["gpu_peak_mb"].max() * 1.20)
    ax.grid(**GRID_KW)
    _save(fig, "line_gpu_memory.png", output_dir)


# ===================================================================
# Plot 4 – Throughput (line)
# ===================================================================
def plot_throughput(plot_data, output_dir):
    df = _load_csv_required(plot_data, "line_4_throughput.csv")
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.2, markersize=8, label=mode.capitalize())
        _annotate_points(ax, X, vals, fmt="{:.4f}", fontsize=8)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Throughput (images / sec)", fontsize=12)
    ax.set_title("Throughput (images/sec) by Model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.25)
    ax.grid(**GRID_KW)
    _save(fig, "line_throughput.png", output_dir)


# ===================================================================
# Plot 5 – Combined 2x2 overview (line)
# ===================================================================
def plot_all_metrics(plot_data, output_dir):
    lat_df = _load_csv(plot_data, "line_1_latency_comparison.csv")
    load_df = _load_csv(plot_data, "line_2_model_load_time.csv")
    gpu_df = _load_csv(plot_data, "line_3_gpu_memory.csv")
    tp_df = _load_csv(plot_data, "line_4_throughput.csv")

    if lat_df is None or tp_df is None:
        print("  Skipping line_all_metrics (missing data)")
        return

    models = lat_df["model"].values
    X = np.arange(len(models))
    modes = [c for c in lat_df.columns if c != "model"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Benchmark Overview -- Line Plots",
                 fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Latency
    ax = axes[0, 0]
    for i, mode in enumerate(modes):
        vals = lat_df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Inference Latency", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 2: Load Time
    ax = axes[0, 1]
    if load_df is not None and len(load_df) > 0:
        load_modes = [c for c in load_df.columns if c != "model"]
        for i, mode in enumerate(load_modes):
            vals = load_df[mode].values.astype(float)
            ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
    ax.set_ylabel("Load Time (s)")
    ax.set_title("Model Load Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 3: GPU Memory
    ax = axes[1, 0]
    if gpu_df is not None and len(gpu_df) > 0:
        X_gpu = np.arange(len(gpu_df))
        ax.plot(X_gpu, gpu_df["gpu_used_mb"].values, marker="o", color="#4C72B0",
                linewidth=2, markersize=7, label="Used")
        ax.plot(X_gpu, gpu_df["gpu_peak_mb"].values, marker="s", color="#DD8452",
                linewidth=2, markersize=7, label="Peak")
        ax.set_xticks(X_gpu)
        ax.set_xticklabels([m.replace(" ", "\n") for m in gpu_df["model"].values], fontsize=8)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 4: Throughput
    ax = axes[1, 1]
    for i, mode in enumerate(modes):
        vals = tp_df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=7, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=8)
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Throughput", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "line_all_metrics.png", output_dir)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate benchmark line plots.")
    parser.add_argument("--plot-data-dir", default=None, help="Path to plot_data/ directory")
    parser.add_argument("--output-dir", default=None, help="Path to output plots directory")
    args = parser.parse_args()

    plot_data = args.plot_data_dir or DEFAULT_PLOT_DATA
    output_dir = args.output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    print("Generating line plots ...")
    plot_latency_comparison(plot_data, output_dir)
    plot_model_load_time(plot_data, output_dir)
    plot_gpu_memory(plot_data, output_dir)
    plot_throughput(plot_data, output_dir)
    plot_all_metrics(plot_data, output_dir)
    print(f"\nAll 5 line plots saved to:\n  {output_dir}")


if __name__ == "__main__":
    main()
