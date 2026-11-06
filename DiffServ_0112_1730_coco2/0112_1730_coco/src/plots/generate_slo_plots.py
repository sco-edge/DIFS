#!/usr/bin/env python3
"""
generate_slo_plots.py

Generates the 3 required SLO evaluation graphs from
results/model_benchmark/plot_data_5model/slo_benchmark_results.csv:

  1. Latency SLO Attainment  (Y) vs Load Level (X)
  2. Accuracy SLO Attainment (Y) vs Load Level (X)
  3. Throughput              (Y) vs Load Level (X)

One series per scaling mode (Horizontal, Vertical, Hybrid).

Usage:
    python -m src.plots.generate_slo_plots [--data CSV] [--out DIR]
"""
import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(_HERE, "..", "..", "results", "model_benchmark",
                            "plot_data_5model", "slo_benchmark_results.csv")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plots")

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 150
MARKERS = ["o", "s", "^"]
COLORS = ["#C44E52", "#4C72B0", "#55A868"]  # horizontal, vertical, hybrid
GRID_KW = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, color="#cccccc")

LOAD_ORDER = ["low", "medium", "high"]


def _pseudo_label(mode):
    # v2 uses plain labels (no "P-" prefix, unlike v1's pseudo-scaling naming)
    return mode.capitalize()


def _load(df, metric, output_dir, ylabel, title, fname, ymax=105, fmt="{:.1f}"):
    modes = sorted(df["scaling_mode"].unique(),
                   key=lambda m: ["horizontal", "vertical", "hybrid"].index(m))
    fig, ax = plt.subplots(figsize=(10, 6))
    X = range(len(LOAD_ORDER))
    for i, mode in enumerate(modes):
        sub = df[df["scaling_mode"] == mode].set_index("load_level")
        vals = [sub.loc[lv, metric] if lv in sub.index else float("nan")
                for lv in LOAD_ORDER]
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i],
                linewidth=2.4, markersize=9, label=_pseudo_label(mode))
        for xi, v in zip(X, vals):
            ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(list(X))
    ax.set_xticklabels([lv.title() for lv in LOAD_ORDER], fontsize=12)
    ax.set_xlabel("Load Level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.grid(**GRID_KW)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    out = os.path.join(output_dir, fname)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}  ({os.path.getsize(out) / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Generate SLO evaluation plots.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="slo_benchmark_results.csv path")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help="output directory")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: data file not found: {args.data}\n"
              f"Run: python -m src.experiments.generate_slo_benchmark_results",
              file=sys.stderr)
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    print("Generating SLO evaluation plots ...")

    _load(df, "latency_slo_attainment_pct", args.out,
          "Latency SLO Attainment (%)",
          "Latency SLO Attainment vs Load Level",
          "slo_latency_attainment.png")

    _load(df, "accuracy_slo_attainment_pct", args.out,
          "Accuracy SLO Attainment (%)",
          "Accuracy SLO Attainment vs Load Level",
          "slo_accuracy_attainment.png")

    _load(df, "throughput_req_per_s", args.out,
          "Throughput (req / s)",
          "Throughput vs Load Level",
          "slo_throughput.png",
          ymax=None, fmt="{:.3f}")

    print(f"\nAll 3 SLO plots saved to:\n  {args.out}")


if __name__ == "__main__":
    main()
