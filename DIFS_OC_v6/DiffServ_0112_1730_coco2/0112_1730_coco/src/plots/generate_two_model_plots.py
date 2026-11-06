#!/usr/bin/env python3
"""
generate_two_model_plots.py

Filters existing CSVs for SD-2.1 and SD-3.5-Medium only, generates all plots,
and saves them to separate directories so they don't overwrite the all-model plots.

Usage:
    python generate_two_model_plots.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_HERE, "..", "..")
ALL_MODEL_PLOT_DATA = os.path.join(PROJECT_ROOT, "results", "model_benchmark", "plot_data")
TWO_MODEL_PLOT_DATA = os.path.join(PROJECT_ROOT, "results", "model_benchmark", "two_models_plot_data")
TWO_MODEL_PLOTS = os.path.join(PROJECT_ROOT, "results", "model_benchmark", "two_models_plots")

TWO_MODELS = ["SD-2.1", "SD-3.5-Medium"]

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
DPI = 150

GRID_KW = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, color="#cccccc")
MARKERS = ["o", "s", "^"]
COLORS = ["#4C72B0", "#55A868", "#C44E52"]
SWITCH_COLORS = {"Horizontal": "#1f77b4", "Vertical": "#ff7f0e", "Hybrid": "#2ca02c"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(fig, name, output_dir):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _value_labels(ax, bars, fmt="{:.1f}", fontsize=10):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize,
        )


def _annotate_points(ax, x, vals, fmt="{:.2f}", fontsize=9, offset=8):
    for xi, v in zip(x, vals):
        if v is not None and not np.isnan(v):
            ax.annotate(fmt.format(v), xy=(xi, v), xytext=(0, offset),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=fontsize)


def _filter_csv_long(df, model_col="model"):
    """Filter long-format CSV to only TWO_MODELS."""
    return df[df[model_col].isin(TWO_MODELS)].copy()


def _filter_csv_wide(df, model_col="model"):
    """Filter wide-format CSV to only TWO_MODELS."""
    return df[df[model_col].isin(TWO_MODELS)].copy()


# ---------------------------------------------------------------------------
# Step 1: Create filtered CSVs
# ---------------------------------------------------------------------------
def create_filtered_csvs():
    os.makedirs(TWO_MODEL_PLOT_DATA, exist_ok=True)
    print("Creating filtered CSVs for SD-2.1 and SD-3.5-Medium ...")

    # Bar plot CSVs (long format)
    for name in ["plot_1_latency_comparison.csv", "plot_2_model_load_time.csv",
                  "plot_4_throughput.csv", "switching_time_bar.csv"]:
        src = os.path.join(ALL_MODEL_PLOT_DATA, name)
        if os.path.exists(src):
            df = pd.read_csv(src)
            df_filtered = _filter_csv_long(df)
            dst = os.path.join(TWO_MODEL_PLOT_DATA, name)
            df_filtered.to_csv(dst, index=False)
            print(f"  Filtered: {name} ({len(df)} -> {len(df_filtered)} rows)")

    # GPU memory CSV (special format: model, gpu_used_mb, gpu_peak_mb)
    for name in ["plot_3_gpu_memory.csv", "line_3_gpu_memory.csv"]:
        src = os.path.join(ALL_MODEL_PLOT_DATA, name)
        if os.path.exists(src):
            df = pd.read_csv(src)
            df_filtered = _filter_csv_wide(df)
            dst = os.path.join(TWO_MODEL_PLOT_DATA, name)
            df_filtered.to_csv(dst, index=False)
            print(f"  Filtered: {name} ({len(df)} -> {len(df_filtered)} rows)")

    # Wide format CSVs (line plots, heatmap, radar)
    for name in ["line_1_latency_comparison.csv", "line_2_model_load_time.csv",
                  "line_4_throughput.csv", "plot_6_scaling_mode_heatmap.csv",
                  "switching_time_line.csv"]:
        src = os.path.join(ALL_MODEL_PLOT_DATA, name)
        if os.path.exists(src):
            df = pd.read_csv(src)
            df_filtered = _filter_csv_wide(df)
            dst = os.path.join(TWO_MODEL_PLOT_DATA, name)
            df_filtered.to_csv(dst, index=False)
            print(f"  Filtered: {name} ({len(df)} -> {len(df_filtered)} rows)")

    # Radar CSV
    src = os.path.join(ALL_MODEL_PLOT_DATA, "plot_5_radar_model_comparison.csv")
    if os.path.exists(src):
        df = pd.read_csv(src)
        df_filtered = _filter_csv_wide(df)
        dst = os.path.join(TWO_MODEL_PLOT_DATA, "plot_5_radar_model_comparison.csv")
        df_filtered.to_csv(dst, index=False)
        print(f"  Filtered: plot_5_radar_model_comparison.csv ({len(df)} -> {len(df_filtered)} rows)")

    print(f"\nAll filtered CSVs saved to:\n  {TWO_MODEL_PLOT_DATA}\n")


# ---------------------------------------------------------------------------
# Step 2: Generate bar plots
# ---------------------------------------------------------------------------
def plot_latency_comparison():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_1_latency_comparison.csv"))
    models = df["model"].unique()
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode].sort_values("model")
        bars = ax.bar(x + offsets[i], mode_data["latency_sec"].values, w, label=mode.capitalize())
        _value_labels(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Inference Latency (seconds)", fontsize=12)
    ax.set_title("Inference Latency: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["latency_sec"].max() * 1.15)
    ax.grid(**GRID_KW)
    _save(fig, "plot_1_latency_comparison.png", TWO_MODEL_PLOTS)


def plot_model_load_time():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_2_model_load_time.csv"))
    models = df["model"].unique()
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode].sort_values("model")
        bars = ax.bar(x + offsets[i], mode_data["load_time_sec"].values, w, label=mode.capitalize())
        _value_labels(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Model Load Time (seconds)", fontsize=12)
    ax.set_title("Model Load Time: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["load_time_sec"].max() * 1.18)
    ax.grid(**GRID_KW)
    _save(fig, "plot_2_model_load_time.png", TWO_MODEL_PLOTS)


def plot_gpu_memory():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_3_gpu_memory.csv"))
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_u = ax.bar(x - w / 2, df["gpu_used_mb"].values, w, label="Used", color="#4C72B0")
    bars_p = ax.bar(x + w / 2, df["gpu_peak_mb"].values, w, label="Peak", color="#DD8452")
    _value_labels(ax, bars_u, fmt="{:.1f}")
    _value_labels(ax, bars_p, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["model"].values], fontsize=11)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["gpu_peak_mb"].max() * 1.2)
    ax.grid(**GRID_KW)
    _save(fig, "plot_3_gpu_memory.png", TWO_MODEL_PLOTS)


def plot_throughput():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_4_throughput.csv"))
    horiz = df[df["scaling_mode"] == "horizontal"].sort_values("model")
    if len(horiz) == 0:
        horiz = df.groupby("model")["throughput_images_per_sec"].mean().reset_index()
    x = np.arange(len(horiz))
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v < 0.02 else "#4C72B0" for v in horiz["throughput_images_per_sec"].values]
    bars = ax.bar(x, horiz["throughput_images_per_sec"].values, color=colors, edgecolor="white", linewidth=0.8)
    _value_labels(ax, bars, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in horiz["model"].values], fontsize=11)
    ax.set_ylabel("Throughput (images / sec)", fontsize=12)
    ax.set_title("Inference Throughput: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.set_ylim(0, horiz["throughput_images_per_sec"].max() * 1.35)
    ax.grid(**GRID_KW)
    _save(fig, "plot_4_throughput.png", TWO_MODEL_PLOTS)


def plot_throughput_per_minute():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_4_throughput.csv"))
    modes = df["scaling_mode"].unique()
    models = [m for m in TWO_MODELS if m in set(df["model"])]
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode].sort_values("model")
        img_per_min = mode_data["throughput_images_per_sec"].values * 60
        bars = ax.bar(x + offsets[i], img_per_min, w, label=mode.capitalize())
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Throughput (images / minute)", fontsize=12)
    ax.set_title("Inference Throughput: SD 2.1 vs SD 3.5 Medium (images/min)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    all_vals = []
    for mode in modes:
        mode_data = df[df["scaling_mode"] == mode]
        all_vals.extend((mode_data["throughput_images_per_sec"].values * 60).tolist())
    ax.set_ylim(0, max(all_vals) * 1.3)
    ax.grid(**GRID_KW)
    _save(fig, "plot_4b_throughput_per_minute.png", TWO_MODEL_PLOTS)


def plot_radar():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_5_radar_model_comparison.csv"))
    metrics = ["Latency\n(inv)", "Throughput", "Load Time\n(inv)", "GPU\nEfficiency"]
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    inv_lat = [1.0 / v if v > 0 else 0 for v in df["avg_latency_sec"]]
    throughput = df["throughput_images_per_sec"].tolist()
    inv_load = [1.0 / v if v > 0 else 0 for v in df["avg_load_time_sec"]]
    gpu_eff = [1.0 / v if v > 0 else 0 for v in df["gpu_peak_mb"]]

    raw = list(zip(inv_lat, throughput, inv_load, gpu_eff))
    norms = []
    for col in zip(*raw):
        mn, mx = min(col), max(col)
        norms.append([(v - mn) / (mx - mn) if mx != mn else 0.5 for v in col])

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    palette = ["#4C72B0", "#C44E52"]
    for idx in range(len(df)):
        vals = [norms[j][idx] for j in range(n)]
        vals += vals[:1]
        label = df["model"].iloc[idx].replace(" ", "\n")
        ax.plot(angles, vals, linewidth=2.5, label=label, color=palette[idx % len(palette)])
        ax.fill(angles, vals, alpha=0.12, color=palette[idx % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Model Comparison Radar: SD 2.1 vs SD 3.5 Medium\n(normalised, higher = better)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    _save(fig, "plot_5_model_comparison_radar.png", TWO_MODEL_PLOTS)


def plot_heatmap():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "plot_6_scaling_mode_heatmap.csv"))
    modes = [c for c in df.columns if c != "model"]
    data = df[modes].values
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=[m.replace(" ", "\n") for m in df["model"].values],
        yticklabels=[s.capitalize() for s in modes],
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"fontsize": 12},
    )
    ax.set_title("Inference Latency Heatmap: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Scaling Mode", fontsize=12)
    fig.tight_layout()
    _save(fig, "plot_6_scaling_mode_impact.png", TWO_MODEL_PLOTS)


def plot_architecture():
    """Copy architecture diagram (same for both)."""
    src = os.path.join(ALL_MODEL_PLOT_DATA, "..", "plots", "plot_7_system_architecture.png")
    if os.path.exists(src):
        import shutil
        dst = os.path.join(TWO_MODEL_PLOTS, "plot_7_system_architecture.png")
        shutil.copy2(src, dst)
        print("  Copied: plot_7_system_architecture.png")


# ---------------------------------------------------------------------------
# Step 3: Generate line plots
# ---------------------------------------------------------------------------
def line_latency_comparison():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_1_latency_comparison.csv"))
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.5, markersize=10, label=mode.capitalize())
        _annotate_points(ax, X, vals, fontsize=10)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Inference Latency (seconds)", fontsize=12)
    ax.set_title("Inference Latency: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.12)
    ax.grid(**GRID_KW)
    _save(fig, "line_latency_comparison.png", TWO_MODEL_PLOTS)


def line_model_load_time():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_2_model_load_time.csv"))
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.5, markersize=10, label=mode.capitalize())
        _annotate_points(ax, X, vals, fmt="{:.2f}", fontsize=10)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Model Load Time (seconds)", fontsize=12)
    ax.set_title("Model Load Time: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.18)
    ax.grid(**GRID_KW)
    _save(fig, "line_model_load_time.png", TWO_MODEL_PLOTS)


def line_gpu_memory():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_3_gpu_memory.csv"))
    X = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X, df["gpu_used_mb"].values, marker="o", color="#4C72B0",
            linewidth=2.5, markersize=10, label="Used")
    ax.plot(X, df["gpu_peak_mb"].values, marker="s", color="#DD8452",
            linewidth=2.5, markersize=10, label="Peak")
    _annotate_points(ax, X, df["gpu_used_mb"].values, fmt="{:.1f}", offset=10)
    _annotate_points(ax, X, df["gpu_peak_mb"].values, fmt="{:.1f}", offset=10)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["model"].values], fontsize=11)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.set_ylim(0, df["gpu_peak_mb"].max() * 1.20)
    ax.grid(**GRID_KW)
    _save(fig, "line_gpu_memory.png", TWO_MODEL_PLOTS)


def line_throughput():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_4_throughput.csv"))
    models = df["model"].values
    X = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                linewidth=2.5, markersize=10, label=mode.capitalize())
        _annotate_points(ax, X, vals, fmt="{:.4f}", fontsize=9)
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Throughput (images / sec)", fontsize=12)
    ax.set_title("Throughput: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.25)
    ax.grid(**GRID_KW)
    _save(fig, "line_throughput.png", TWO_MODEL_PLOTS)


def line_all_metrics():
    lat_df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_1_latency_comparison.csv"))
    load_df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_2_model_load_time.csv"))
    gpu_df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_3_gpu_memory.csv"))
    tp_df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "line_4_throughput.csv"))

    models = lat_df["model"].values
    X = np.arange(len(models))
    modes = [c for c in lat_df.columns if c != "model"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Benchmark Overview: SD 2.1 vs SD 3.5 Medium -- Line Plots",
                 fontsize=16, fontweight="bold", y=0.98)

    # Subplot 1: Latency
    ax = axes[0, 0]
    for i, mode in enumerate(modes):
        vals = lat_df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=8, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Latency (s)")
    ax.set_title("Inference Latency", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 2: Load Time
    ax = axes[0, 1]
    load_modes = [c for c in load_df.columns if c != "model"]
    for i, mode in enumerate(load_modes):
        vals = load_df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=8, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Load Time (s)")
    ax.set_title("Model Load Time", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 3: GPU Memory
    ax = axes[1, 0]
    X_gpu = np.arange(len(gpu_df))
    ax.plot(X_gpu, gpu_df["gpu_used_mb"].values, marker="o", color="#4C72B0",
            linewidth=2, markersize=8, label="Used")
    ax.plot(X_gpu, gpu_df["gpu_peak_mb"].values, marker="s", color="#DD8452",
            linewidth=2, markersize=8, label="Peak")
    ax.set_xticks(X_gpu)
    ax.set_xticklabels([m.replace(" ", "\n") for m in gpu_df["model"].values], fontsize=9)
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    # Subplot 4: Throughput
    ax = axes[1, 1]
    for i, mode in enumerate(modes):
        vals = tp_df[mode].values.astype(float)
        ax.plot(X, vals, marker=MARKERS[i], color=COLORS[i], linewidth=2, markersize=8, label=mode.capitalize())
    ax.set_xticks(X)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax.set_ylabel("Throughput (img/s)")
    ax.set_title("Throughput", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(**GRID_KW)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "line_all_metrics.png", TWO_MODEL_PLOTS)


# ---------------------------------------------------------------------------
# Step 4: Generate switching plots
# ---------------------------------------------------------------------------
def switching_bar():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "switching_time_bar.csv"))
    models = df["model"].unique()
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode].sort_values("model")
        color_key = mode.capitalize()
        bars = ax.bar(
            x + offsets[i], mode_data["switch_time_sec"].values, w,
            label=mode.capitalize(), color=SWITCH_COLORS.get(color_key, "#333"),
            edgecolor="white",
        )
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}s",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Switch Time (seconds)", fontsize=12)
    ax.set_title("Model Switching Time: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["switch_time_sec"].max() * 1.15)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _save(fig, "switching_time_bar.png", TWO_MODEL_PLOTS)


def switching_line():
    df = pd.read_csv(os.path.join(TWO_MODEL_PLOT_DATA, "switching_time_line.csv"))
    models = df["model"].values
    x = np.arange(len(models))
    modes = [c for c in df.columns if c != "model"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, mode in enumerate(modes):
        vals = df[mode].values.astype(float)
        color_key = mode.capitalize()
        ax.plot(
            x, vals, marker=MARKERS[i % len(MARKERS)], markersize=10, linewidth=2.5,
            label=mode.capitalize(), color=SWITCH_COLORS.get(color_key, "#333"),
        )
        for j, v in enumerate(vals):
            if not np.isnan(v):
                offset_y = 8 if mode.capitalize() != "Vertical" else -14
                ax.annotate(f"{v:.2f}s", (j, v), textcoords="offset points",
                            xytext=(0, offset_y), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=11)
    ax.set_ylabel("Switch Time (seconds)", fontsize=12)
    ax.set_title("Model Switching Time: SD 2.1 vs SD 3.5 Medium", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    all_vals = df[modes].values.flatten()
    ax.set_ylim(0, np.nanmax(all_vals) * 1.18)
    ax.grid(**GRID_KW)
    fig.tight_layout()
    _save(fig, "switching_time_line.png", TWO_MODEL_PLOTS)


def pairwise_heatmap():
    """Derive pairwise switching matrix from switching_time_bar.csv for 2 models."""
    bar_path = os.path.join(TWO_MODEL_PLOT_DATA, "switching_time_bar.csv")
    if not os.path.exists(bar_path):
        print("  Skipping pairwise heatmap (no data)")
        return
    df_bar = pd.read_csv(bar_path)
    avg_load = df_bar.groupby("model")["switch_time_sec"].mean().reset_index()
    avg_load.columns = ["model", "load_time_sec"]
    avg_load = avg_load.sort_values("load_time_sec").reset_index(drop=True)
    model_names = avg_load["model"].tolist()
    n = len(model_names)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = avg_load.iloc[j]["load_time_sec"]

    short = [m.split("/")[0] if "/" in m else m for m in model_names]
    short = [s[:18] + ".." if len(s) > 20 else s for s in short]

    fig, ax = plt.subplots(figsize=(max(8, n * 2), max(6, n * 1.5)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            text_color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}s", ha="center", va="center", fontsize=11, color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Switch Time (seconds)", fontsize=11)

    ax.set_title(
        "Pairwise Model Switching Time: SD 2.1 vs SD 3.5 Medium\n"
        "From Model (row) -> To Model (column)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("To Model", fontsize=12)
    ax.set_ylabel("From Model", fontsize=12)
    fig.tight_layout()
    _save(fig, "pairwise_switching_heatmap.png", TWO_MODEL_PLOTS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(TWO_MODEL_PLOTS, exist_ok=True)

    # Step 1: Create filtered CSVs
    create_filtered_csvs()

    # Step 2: Bar plots
    print("Generating bar plots ...")
    plot_latency_comparison()
    plot_model_load_time()
    plot_gpu_memory()
    plot_throughput()
    plot_throughput_per_minute()
    plot_radar()
    plot_heatmap()
    plot_architecture()

    # Step 3: Line plots
    print("\nGenerating line plots ...")
    line_latency_comparison()
    line_model_load_time()
    line_gpu_memory()
    line_throughput()
    line_all_metrics()

    # Step 4: Switching plots
    print("\nGenerating switching plots ...")
    switching_bar()
    switching_line()
    pairwise_heatmap()

    print(f"\nAll plots saved to:\n  {TWO_MODEL_PLOTS}")
    print(f"\nFiltered CSVs saved to:\n  {TWO_MODEL_PLOT_DATA}")


if __name__ == "__main__":
    main()
