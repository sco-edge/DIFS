"""
plot_switching_time_vs_model.py

Plots model switching time vs diffusion model.

Reads from metrics/switch_metrics.csv (LOAD and SWITCH events)
and metrics/worker_metrics.csv (for per-model inference data).

Generates:
  - plots/switching_time_vs_model.png     (bar chart, mean switch time per model)
  - plots/switching_hot_vs_cold.png       (grouped bar, hot vs cold per model)
  - plots/model_load_time_vs_model.png    (bar chart, model load time per model)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

METRICS_DIR = "metrics"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)


############################################################
# 1. Model Load Time vs Model
############################################################

switch_file = os.path.join(METRICS_DIR, "switch_metrics.csv")

if os.path.exists(switch_file):

    df = pd.read_csv(switch_file)

    # LOAD events = initial model loading
    loads = df[df["event"] == "LOAD"].copy()

    if len(loads) > 0:
        # Group by model (the "new_model" column holds the model name for LOAD events)
        load_times = (
            loads.groupby("new_model")["switch_time"]
            .agg(["mean", "std", "min", "max", "count"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            range(len(load_times)),
            load_times["mean"],
            yerr=load_times["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            alpha=0.8,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(load_times)))
        )

        ax.set_xticks(range(len(load_times)))
        ax.set_xticklabels(load_times.index, rotation=45, ha="right")

        ax.set_ylabel("Load Time (seconds)")
        ax.set_xlabel("Diffusion Model")
        ax.set_title("Model Load Time by Diffusion Model")

        # Add value labels on bars
        for bar, val in zip(bars, load_times["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}s",
                ha="center", va="bottom", fontsize=9
            )

        save(fig, "model_load_time_vs_model.png")
        print("Saved plots/model_load_time_vs_model.png")
        print(load_times)
    else:
        print("[PLOT] No LOAD events in switch_metrics.csv")

    # SWITCH events = hot/cold model switching
    switches = df[df["event"] == "SWITCH"].copy()

    if len(switches) > 0:
        switch_times = (
            switches.groupby("new_model")["switch_time"]
            .agg(["mean", "std", "min", "max", "count"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            range(len(switch_times)),
            switch_times["mean"],
            yerr=switch_times["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            alpha=0.8,
            color=plt.cm.plasma(np.linspace(0.2, 0.8, len(switch_times)))
        )

        ax.set_xticks(range(len(switch_times)))
        ax.set_xticklabels(switch_times.index, rotation=45, ha="right")

        ax.set_ylabel("Switching Time (seconds)")
        ax.set_xlabel("Target Model")
        ax.set_title("Model Switching Time by Target Model")

        for bar, val in zip(bars, switch_times["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}s",
                ha="center", va="bottom", fontsize=9
            )

        save(fig, "switching_time_vs_model.png")
        print("Saved plots/switching_time_vs_model.png")
        print(switch_times)
    else:
        print("[PLOT] No SWITCH events in switch_metrics.csv")

else:
    print(f"[PLOT] {switch_file} not found")


############################################################
# 2. Inference Latency vs Model (from worker_metrics.csv)
############################################################

worker_file = os.path.join(METRICS_DIR, "worker_metrics.csv")

if os.path.exists(worker_file):

    wm = pd.read_csv(worker_file)

    if "model_name" in wm.columns and "latency_sec" in wm.columns:

        latency_by_model = (
            wm.groupby("model_name")["latency_sec"]
            .agg(["mean", "std", "min", "max", "count"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            range(len(latency_by_model)),
            latency_by_model["mean"],
            yerr=latency_by_model["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            alpha=0.8,
            color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(latency_by_model)))
        )

        ax.set_xticks(range(len(latency_by_model)))
        ax.set_xticklabels(latency_by_model.index, rotation=45, ha="right")

        ax.set_ylabel("Average Latency (seconds)")
        ax.set.xlabel("Diffusion Model")
        ax.set_title("Average Inference Latency by Diffusion Model")

        for bar, val in zip(bars, latency_by_model["mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f}s",
                ha="center", va="bottom", fontsize=9
            )

        save(fig, "latency_vs_model.png")
        print("Saved plots/latency_vs_model.png")
        print(latency_by_model)

else:
    print(f"[PLOT] {worker_file} not found")
