"""
plot_all_models.py

Unified plot generator for all per-model performance charts.
Generates all plots in one run from the collected metrics.

Generates:
  - plots/latency_vs_model.png          (bar chart with error bars)
  - plots/latency_distribution.png      (box plot per model)
  - plots/latency_percentiles.png       (p50/p95/p99 per model)
  - plots/switching_time_vs_model.png   (bar chart, switch time per model)
  - plots/model_load_time_vs_model.png  (bar chart, load time per model)
  - plots/gpu_memory_vs_model.png       (bar chart, GPU memory per model)
  - plots/fid_vs_model.png              (bar chart, FID per model)
  - plots/throughput_vs_model.png       (bar chart, throughput per model)

Usage:
    python src/plots/plot_all_models.py
    python src/plots/plot_all_models.py --benchmark-dir results/model_benchmark
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

METRICS_DIR = "metrics"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bar_with_errors(data, ylabel, title, filename, color_map="viridis", fmt="{:.2f}"):
    """Generic bar chart with error bars and value labels."""
    if len(data) == 0:
        print(f"[PLOT] No data for {filename}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    means = data["mean"]
    stds = data.get("std", pd.Series(0, index=data.index)).fillna(0)

    colors = plt.get_cmap(color_map)(np.linspace(0.2, 0.8, len(data)))

    bars = ax.bar(
        range(len(data)),
        means,
        yerr=stds,
        capsize=5,
        edgecolor="black",
        alpha=0.85,
        color=colors
    )

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data.index, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Diffusion Model", fontsize=12)
    ax.set_title(title, fontsize=13)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + stds.iloc[list(means).index(val)] * 0.1 + means.max() * 0.01,
            fmt.format(val),
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    save(fig, filename)
    print(f"Saved plots/{filename}")


def plot_from_worker_metrics():
    """Generate all plots from worker_metrics.csv."""
    worker_file = os.path.join(METRICS_DIR, "worker_metrics.csv")

    if not os.path.exists(worker_file):
        print(f"[PLOT] {worker_file} not found, skipping worker metrics plots")
        return

    wm = pd.read_csv(worker_file)

    if "model_name" not in wm.columns:
        print("[PLOT] worker_metrics.csv missing model_name column")
        return

    ############################################################
    # 1. Latency vs Model
    ############################################################

    if "latency_sec" in wm.columns:
        latency_stats = (
            wm.groupby("model_name")["latency_sec"]
            .agg(["mean", "std", "median", "min", "max", "count"])
            .sort_values("mean")
        )

        plot_bar_with_errors(
            latency_stats, "Average Latency (seconds)",
            "Average Inference Latency by Diffusion Model",
            "latency_vs_model.png", color_map="RdYlGn_r"
        )

        # Box plot
        models_sorted = latency_stats.index.tolist()
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [wm[wm["model_name"] == m]["latency_sec"].values for m in models_sorted]
        bp = ax.boxplot(box_data, labels=models_sorted, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(models_sorted)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_xticklabels(models_sorted, rotation=45, ha="right")
        ax.set_ylabel("Latency (seconds)")
        ax.set_xlabel("Diffusion Model")
        ax.set_title("Inference Latency Distribution by Model")
        save(fig, "latency_distribution.png")
        print("Saved plots/latency_distribution.png")

        # Percentiles
        perc_data = []
        for model in models_sorted:
            lat = wm[wm["model_name"] == model]["latency_sec"]
            perc_data.append({"model": model, "p50": lat.quantile(0.50),
                              "p95": lat.quantile(0.95), "p99": lat.quantile(0.99)})
        pdf = pd.DataFrame(perc_data).set_index("model")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pdf))
        w = 0.25
        ax.bar(x - w, pdf["p50"], w, label="P50", color="#2196F3", edgecolor="black")
        ax.bar(x, pdf["p95"], w, label="P95", color="#FF9800", edgecolor="black")
        ax.bar(x + w, pdf["p99"], w, label="P99", color="#F44336", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(pdf.index, rotation=45, ha="right")
        ax.set_ylabel("Latency (seconds)")
        ax.set_xlabel("Diffusion Model")
        ax.set_title("Latency Percentiles by Diffusion Model")
        ax.legend()
        save(fig, "latency_percentiles.png")
        print("Saved plots/latency_percentiles.png")

    ############################################################
    # 2. GPU Memory vs Model
    ############################################################

    if "gpu_memory_MB" in wm.columns:
        gpu_stats = (
            wm.groupby("model_name")["gpu_memory_MB"]
            .agg(["mean", "std", "max"])
            .sort_values("mean")
        )
        plot_bar_with_errors(
            gpu_stats, "GPU Memory (MB)",
            "Average GPU Memory Usage by Diffusion Model",
            "gpu_memory_vs_model.png", color_map="YlOrRd"
        )

    ############################################################
    # 3. FID vs Model
    ############################################################

    if "fid" in wm.columns:
        valid_fid = wm[wm["fid"] >= 0]
        if len(valid_fid) > 0:
            fid_stats = (
                valid_fid.groupby("model_name")["fid"]
                .agg(["mean", "std"])
                .sort_values("mean")
            )
            plot_bar_with_errors(
                fid_stats, "FID Score",
                "Frechet Inception Distance by Diffusion Model",
                "fid_vs_model.png", color_map="RdYlGn",
                fmt="{:.1f}"
            )

    ############################################################
    # 4. Throughput vs Model
    ############################################################

    if "images_generated" in wm.columns and "latency_sec" in wm.columns:
        wm_sorted = wm.sort_values("timestamp")
        throughput_data = []
        for model in wm["model_name"].unique():
            model_df = wm_sorted[wm_sorted["model_name"] == model]
            total_images = model_df["images_generated"].sum()
            total_time = model_df["latency_sec"].sum()
            if total_time > 0:
                throughput_data.append({
                    "model_name": model,
                    "mean": total_images / total_time,
                    "std": 0
                })
        if throughput_data:
            tp_df = pd.DataFrame(throughput_data).set_index("model_name").sort_values("mean")
            plot_bar_with_errors(
                tp_df, "Throughput (images/sec)",
                "Inference Throughput by Diffusion Model",
                "throughput_vs_model.png", color_map="cool"
            )


def plot_from_switch_metrics():
    """Generate switching time and load time plots from switch_metrics.csv."""
    switch_file = os.path.join(METRICS_DIR, "switch_metrics.csv")

    if not os.path.exists(switch_file):
        print(f"[PLOT] {switch_file} not found, skipping switch metrics plots")
        return

    df = pd.read_csv(switch_file)

    ############################################################
    # Model Load Time vs Model
    ############################################################

    loads = df[df["event"] == "LOAD"]
    if len(loads) > 0:
        load_stats = (
            loads.groupby("new_model")["switch_time"]
            .agg(["mean", "std", "min", "max", "count"])
            .sort_values("mean")
        )
        plot_bar_with_errors(
            load_stats, "Load Time (seconds)",
            "Model Load Time by Diffusion Model",
            "model_load_time_vs_model.png", color_map="viridis"
        )

    ############################################################
    # Model Switching Time vs Model
    ############################################################

    switches = df[df["event"] == "SWITCH"]
    if len(switches) > 0:
        switch_stats = (
            switches.groupby("new_model")["switch_time"]
            .agg(["mean", "std", "min", "max", "count"])
            .sort_values("mean")
        )
        plot_bar_with_errors(
            switch_stats, "Switching Time (seconds)",
            "Model Switching Time by Target Model",
            "switching_time_vs_model.png", color_map="plasma"
        )

        # Also plot hot vs cold if both exist
        if "reason" in switches.columns:
            for reason in switches["reason"].unique():
                subset = switches[switches["reason"] == reason]
                if len(subset) > 0:
                    reason_stats = (
                        subset.groupby("new_model")["switch_time"]
                        .agg(["mean", "std"])
                        .sort_values("mean")
                    )
                    plot_bar_with_errors(
                        reason_stats,
                        f"Switching Time - {reason} ({len(subset)} events)",
                        f"Model Switching Time ({reason}) by Target Model",
                        f"switching_time_{reason}_vs_model.png",
                        color_map="plasma"
                    )


def main():
    parser = argparse.ArgumentParser(description="Generate all per-model performance plots")
    parser.add_argument("--metrics-dir", default=METRICS_DIR, help="Metrics directory")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output plots directory")
    args = parser.parse_args()

    global METRICS_DIR, OUTPUT_DIR
    METRICS_DIR = args.metrics_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("=" * 60)
    print("[PLOT] Generating all per-model performance plots")
    print("=" * 60)

    plot_from_worker_metrics()
    plot_from_switch_metrics()

    print()
    print("=" * 60)
    print(f"[PLOT] All plots written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
