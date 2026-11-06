# Date: 2026.06.21
# Purpose: For plotting performance metrics
# Adapted: from 'plot_metrics.py'

import os
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DIR = "metrics"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


############################################################
# helper
############################################################

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300)
    plt.close(fig)


############################################################
# Worker Metrics
############################################################

worker = pd.read_csv(os.path.join(METRICS_DIR,
                                  "worker_metrics.csv"))

############################################################
# Latency
############################################################

fig = plt.figure(figsize=(6,4))
plt.plot(worker["latency_sec"])
plt.xlabel("Inference")
plt.ylabel("Latency (sec)")
plt.title("Inference Latency")
save(fig, "latency.png")


############################################################
# Throughput
############################################################

worker["cum_images"] = worker["images_generated"].cumsum()

worker["elapsed"] = worker["timestamp"] - worker["timestamp"].iloc[0]

worker["throughput"] = worker["cum_images"] / worker["elapsed"].replace(0,1)

fig = plt.figure(figsize=(6,4))
plt.plot(worker["throughput"])
plt.xlabel("Inference")
plt.ylabel("Images/sec")
plt.title("Worker Throughput")
save(fig,"throughput.png")


############################################################
# GPU Memory
############################################################

fig = plt.figure(figsize=(6,4))

plt.plot(worker["gpu_memory_MB"],label="Allocated")

plt.plot(worker["gpu_reserved_MB"],label="Reserved")

plt.plot(worker["gpu_peak_MB"],label="Peak")

plt.legend()

plt.xlabel("Inference")

plt.ylabel("GPU Memory (MB)")

plt.title("GPU Memory")

save(fig,"gpu_memory.png")


############################################################
# FID
############################################################

valid = worker[worker["fid"] >= 0]

fig = plt.figure(figsize=(6,4))

plt.plot(valid["fid"])

plt.xlabel("Inference")

plt.ylabel("FID")

plt.title("FID")

save(fig,"fid.png")


############################################################
# Batch Size Distribution
############################################################

fig = plt.figure(figsize=(6,4))

worker["batch_size"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Batch Size")

plt.ylabel("Count")

plt.title("Batch Size Distribution")

save(fig,"batch_sizes.png")


############################################################
# Images Generated
############################################################

fig = plt.figure(figsize=(6,4))

plt.plot(worker["images_generated"].cumsum())

plt.xlabel("Inference")

plt.ylabel("Images")

plt.title("Total Images Generated")

save(fig,"images_generated.png")


############################################################
# Model Load Times
############################################################

file=os.path.join(METRICS_DIR,"model_load_metrics.csv")

if os.path.exists(file):

    load=pd.read_csv(file)

    fig=plt.figure(figsize=(6,4))

    plt.bar(load["model"],load["load_time"])

    plt.xticks(rotation=45)

    plt.ylabel("Seconds")

    plt.title("Model Load Time")

    save(fig,"model_load_time.png")


############################################################
# Model Switch Times
############################################################

file=os.path.join(METRICS_DIR,"model_switch_metrics.csv")

if os.path.exists(file):

    sw=pd.read_csv(file)

    fig=plt.figure(figsize=(6,4))

    plt.plot(sw["switch_time"])

    plt.xlabel("Switch")

    plt.ylabel("Seconds")

    plt.title("Model Switching Time")

    save(fig,"model_switch_time.png")


############################################################
# GPU Memory History
############################################################

file=os.path.join(METRICS_DIR,"gpu_memory_metrics.csv")

if os.path.exists(file):

    gpu=pd.read_csv(file)

    fig=plt.figure(figsize=(6,4))

    plt.plot(gpu["allocated_mb"],label="Allocated")

    plt.plot(gpu["reserved_mb"],label="Reserved")

    plt.legend()

    plt.title("GPU Memory History")

    save(fig,"gpu_memory_history.png")


print()

print("Plots written to:",OUTPUT_DIR)


############################################################
# PER-MODEL PLOTS (if worker_metrics has model_name column)
############################################################

import pandas as _pd
import numpy as _np

worker_file = os.path.join(METRICS_DIR, "worker_metrics.csv")

if os.path.exists(worker_file):

    _wm = _pd.read_csv(worker_file)

    if "model_name" in _wm.columns and "latency_sec" in _wm.columns:

        ############################################################
        # Latency vs Model (bar chart with error bars)
        ############################################################

        _lat = (
            _wm.groupby("model_name")["latency_sec"]
            .agg(["mean", "std"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        _colors = plt.cm.RdYlGn_r(_np.linspace(0.2, 0.8, len(_lat)))

        _bars = ax.bar(
            range(len(_lat)),
            _lat["mean"],
            yerr=_lat["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            color=_colors
        )

        ax.set_xticks(range(len(_lat)))
        ax.set_xticklabels(_lat.index, rotation=45, ha="right")
        ax.set_ylabel("Average Latency (seconds)")
        ax.set_xlabel("Diffusion Model")
        ax.set_title("Average Inference Latency by Model")

        for _bar, _val in zip(_bars, _lat["mean"]):
            ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + 0.05,
                f"{_val:.2f}s",
                ha="center", va="bottom", fontsize=9
            )

        save(fig, "latency_vs_model.png")
        print("Saved latency_vs_model.png")

        ############################################################
        # GPU Memory vs Model
        ############################################################

        if "gpu_memory_MB" in _wm.columns:
            _gpu = (
                _wm.groupby("model_name")["gpu_memory_MB"]
                .agg(["mean", "std"])
                .sort_values("mean")
            )

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(
                range(len(_gpu)),
                _gpu["mean"],
                yerr=_gpu["std"].fillna(0),
                capsize=5,
                edgecolor="black",
                color=plt.cm.YlOrRd(_np.linspace(0.2, 0.8, len(_gpu)))
            )

            ax.set_xticks(range(len(_gpu)))
            ax.set_xticklabels(_gpu.index, rotation=45, ha="right")
            ax.set_ylabel("GPU Memory (MB)")
            ax.set_xlabel("Diffusion Model")
            ax.set_title("GPU Memory Usage by Model")

            save(fig, "gpu_memory_vs_model.png")
            print("Saved gpu_memory_vs_model.png")


############################################################
# SWITCH METRICS vs MODEL
############################################################

switch_file = os.path.join(METRICS_DIR, "switch_metrics.csv")

if os.path.exists(switch_file):

    _sw = _pd.read_csv(switch_file)

    # Model load times
    _loads = _sw[_sw["event"] == "LOAD"]

    if len(_loads) > 0:
        _load_stats = (
            _loads.groupby("new_model")["switch_time"]
            .agg(["mean", "std"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            range(len(_load_stats)),
            _load_stats["mean"],
            yerr=_load_stats["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            color=plt.cm.viridis(_np.linspace(0.2, 0.8, len(_load_stats)))
        )

        ax.set_xticks(range(len(_load_stats)))
        ax.set_xticklabels(_load_stats.index, rotation=45, ha="right")
        ax.set_ylabel("Load Time (seconds)")
        ax.set_xlabel("Diffusion Model")
        ax.set_title("Model Load Time by Model")

        save(fig, "model_load_time_vs_model.png")
        print("Saved model_load_time_vs_model.png")

    # Model switching times
    _switches = _sw[_sw["event"] == "SWITCH"]

    if len(_switches) > 0:
        _switch_stats = (
            _switches.groupby("new_model")["switch_time"]
            .agg(["mean", "std"])
            .sort_values("mean")
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.bar(
            range(len(_switch_stats)),
            _switch_stats["mean"],
            yerr=_switch_stats["std"].fillna(0),
            capsize=5,
            edgecolor="black",
            color=plt.cm.plasma(_np.linspace(0.2, 0.8, len(_switch_stats)))
        )

        ax.set_xticks(range(len(_switch_stats)))
        ax.set_xticklabels(_switch_stats.index, rotation=45, ha="right")
        ax.set_ylabel("Switching Time (seconds)")
        ax.set_xlabel("Target Model")
        ax.set_title("Model Switching Time by Model")

        save(fig, "switching_time_vs_model.png")
        print("Saved switching_time_vs_model.png")


############################################################
# LATENCY PERCENTILES vs MODEL
############################################################

if os.path.exists(worker_file) and "model_name" in _wm.columns and "latency_sec" in _wm.columns:

    _models = _wm["model_name"].unique()
    _perc_data = []

    for _m in _models:
        _lat = _wm[_wm["model_name"] == _m]["latency_sec"]
        _perc_data.append({
            "model": _m,
            "p50": _lat.quantile(0.50),
            "p95": _lat.quantile(0.95),
            "p99": _lat.quantile(0.99),
        })

    _pdf = _pd.DataFrame(_perc_data).set_index("model")

    fig, ax = plt.subplots(figsize=(8, 5))
    _x = _np.arange(len(_pdf))
    _w = 0.25

    ax.bar(_x - _w, _pdf["p50"], _w, label="P50", color="#2196F3", edgecolor="black")
    ax.bar(_x, _pdf["p95"], _w, label="P95", color="#FF9800", edgecolor="black")
    ax.bar(_x + _w, _pdf["p99"], _w, label="P99", color="#F44336", edgecolor="black")

    ax.set_xticks(_x)
    ax.set_xticklabels(_pdf.index, rotation=45, ha="right")
    ax.set_ylabel("Latency (seconds)")
    ax.set_xlabel("Diffusion Model")
    ax.set_title("Latency Percentiles by Model")
    ax.legend()

    save(fig, "latency_percentiles.png")
    print("Saved latency_percentiles.png")