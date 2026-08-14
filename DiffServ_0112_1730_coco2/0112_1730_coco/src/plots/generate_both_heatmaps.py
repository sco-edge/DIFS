#!/usr/bin/env python3
"""
generate_both_heatmaps.py

Generates TWO pairwise switching time heatmap PNGs:
  1. pairwise_switching_heatmap_derived.png  — always, from switching_time_bar.csv
  2. pairwise_switching_heatmap_measured.png — only if pairwise_switching.csv exists

This is a companion to generate_switching_plots.py; does not modify it.

Usage:
    python generate_both_heatmaps.py [--plot-data-dir DIR] [--output-dir DIR]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
MODEL_ORDER = ["SD-v1.4-dir", "SD-v1.4-safetensors", "SD-v1.5", "SD-2.1",
               "SDXL-Base", "SD-3.5-Medium", "SD-3.5-Large", "DreamShaper-8"]

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_DATA = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plot_data")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plots")


def _draw_heatmap(matrix, model_names, output_dir, filename, source_label):
    n = len(model_names)
    short = [m.split("/")[0] if "/" in m else m for m in model_names]
    short = [s[:18] + ".." if len(s) > 20 else s for s in short]

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), max(8, n * 1.0)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}s", ha="center", va="center", fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Switch Time (seconds)", fontsize=10)

    ax.set_title(
        f"Pairwise Model Switching Time {source_label}\n"
        "From Model (row) → To Model (column)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("To Model", fontsize=11)
    ax.set_ylabel("From Model", fontsize=11)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}.png  ({os.path.getsize(path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Generate both pairwise switching time heatmaps.")
    parser.add_argument("--plot-data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    plot_data = args.plot_data_dir or DEFAULT_PLOT_DATA
    output_dir = args.output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    print("Generating both pairwise switching time heatmaps ...")

    # ── 1. Derived (always) ──────────────────────────────────────────────────
    bar_path = os.path.join(plot_data, "switching_time_bar.csv")
    if os.path.exists(bar_path):
        df_bar = pd.read_csv(bar_path)
        avg_load = df_bar.groupby("model")["switch_time_sec"].mean().reset_index()
        avg_load.columns = ["model", "load_time_sec"]
        avg_load["model"] = pd.Categorical(avg_load["model"], categories=MODEL_ORDER, ordered=True)
        avg_load = avg_load.sort_values("model").reset_index(drop=True)
        model_names = avg_load["model"].tolist()
        n = len(model_names)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = avg_load.iloc[j]["load_time_sec"]
        _draw_heatmap(matrix, model_names, output_dir,
                      "pairwise_switching_heatmap_derived",
                      "(derived from load times)")
    else:
        print("  WARNING: switching_time_bar.csv not found, skipping derived heatmap")

    # ── 2. Measured (only if pairwise_switching.csv exists) ──────────────────
    pw_path = os.path.join(plot_data, "pairwise_switching.csv")
    if os.path.exists(pw_path):
        df_raw = pd.read_csv(pw_path, index_col=0)
        model_names = list(df_raw.columns)
        matrix = df_raw.values.astype(float)
        print(f"  Loaded pairwise switching matrix: {pw_path}")
        _draw_heatmap(matrix, model_names, output_dir,
                      "pairwise_switching_heatmap_measured",
                      "(measured via gRPC)")
    else:
        print("  WARNING: pairwise_switching.csv not found, skipping measured heatmap")
        print("  Run collect_pairwise_switching.py against a live worker to generate it.")

    print(f"\nHeatmaps saved to:\n  {output_dir}")


if __name__ == "__main__":
    main()
