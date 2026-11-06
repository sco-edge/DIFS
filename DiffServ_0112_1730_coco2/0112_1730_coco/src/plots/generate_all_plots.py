#!/usr/bin/env python3
"""
generate_all_plots.py

Reads per-plot CSVs from results/model_benchmark/plot_data/ and generates
bar chart visualizations + architecture diagram.

Usage:
    python generate_all_plots.py [--plot-data-dir DIR] [--output-dir DIR]
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns

MODEL_ORDER = ["SD-v1.4-dir", "SD-v1.4-safetensors", "SD-v1.5", "SD-2.1", "SDXL-Base", "SD-3.5-Medium", "SD-3.5-Large", "DreamShaper-8"]
HATCHES = ["//", "\\\\", "xx"]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PLOT_DATA = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plot_data")
DEFAULT_OUTPUT = os.path.join(_HERE, "..", "..", "results", "model_benchmark", "plots")

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")
DPI = 150

# Grid style for all plots
GRID_KW = dict(axis="y", linestyle="--", linewidth=0.6, alpha=0.7, color="#cccccc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _value_labels(ax, bars, fmt="{:.1f}", fontsize=9):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize,
        )


def _save(fig, name, output_dir):
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _load_csv(plot_data, name):
    path = os.path.join(plot_data, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found, skipping.", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    return df


def _load_csv_required(plot_data, name):
    df = _load_csv(plot_data, name)
    if df is None:
        raise FileNotFoundError(f"Required data file missing: {name}")
    return df


def _mode_values(mode_data, models, col):
    """Reindex per-mode values to the full model list (NaN for missing combos)."""
    return mode_data.set_index("model").reindex(models)[col].to_numpy()


# ===================================================================
# Plot 1 – Inference Latency Comparison
# ===================================================================
def plot_latency_comparison(plot_data, output_dir):
    df = _load_csv_required(plot_data, "plot_1_latency_comparison.csv")
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode]
        bars = ax.bar(x + offsets[i], _mode_values(mode_data, models, "latency_sec"), w, label=mode.capitalize(), hatch=HATCHES[i % len(HATCHES)])
        _value_labels(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Inference Latency (seconds)", fontsize=12)
    ax.set_title("Average Inference Latency per Model by Scaling Mode", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["latency_sec"].max() * 1.15)
    ax.grid(**GRID_KW)
    _save(fig, "plot_1_latency_comparison.png", output_dir)


# ===================================================================
# Plot 2 – Model Load Time
# ===================================================================
def plot_model_load_time(plot_data, output_dir):
    df = _load_csv(plot_data, "plot_2_model_load_time.csv")
    if df is None or len(df) == 0:
        print("  Skipping plot_2 (no load time data)")
        return
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    modes = df["scaling_mode"].unique()
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode]
        if len(mode_data) == 0:
            continue
        bars = ax.bar(x + offsets[i], _mode_values(mode_data, models, "load_time_sec"), w, label=mode.capitalize(), hatch=HATCHES[i % len(HATCHES)])
        _value_labels(ax, bars, fmt="{:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Model Load Time (seconds)", fontsize=12)
    ax.set_title("Model Load Time per Model by Scaling Mode", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["load_time_sec"].max() * 1.18)
    ax.grid(**GRID_KW)
    _save(fig, "plot_2_model_load_time.png", output_dir)


# ===================================================================
# Plot 3 – GPU Memory
# ===================================================================
def plot_gpu_memory(plot_data, output_dir):
    df = _load_csv(plot_data, "plot_3_gpu_memory.csv")
    if df is None or len(df) == 0:
        print("  Skipping plot_3 (no GPU memory data)")
        return
    df = df.set_index("model").reindex([m for m in MODEL_ORDER if m in set(df["model"])]).reset_index()
    x = np.arange(len(df))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    bars_u = ax.bar(x - w / 2, df["gpu_used_mb"].values, w, label="Used", color="#4C72B0", hatch=HATCHES[0])
    bars_p = ax.bar(x + w / 2, df["gpu_peak_mb"].values, w, label="Peak", color="#DD8452", hatch=HATCHES[1])
    _value_labels(ax, bars_u, fmt="{:.1f}")
    _value_labels(ax, bars_p, fmt="{:.1f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in df["model"].values], fontsize=10)
    ax.set_ylabel("GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage per Model (Used vs Peak)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, df["gpu_peak_mb"].max() * 1.2)
    ax.grid(**GRID_KW)
    _save(fig, "plot_3_gpu_memory.png", output_dir)


# ===================================================================
# Plot 4 – Throughput
# ===================================================================
def plot_throughput(plot_data, output_dir):
    df = _load_csv_required(plot_data, "plot_4_throughput.csv")
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    # Use horizontal mode as representative
    horiz = df[df["scaling_mode"] == "horizontal"]
    if len(horiz) == 0:
        horiz = df.groupby("model")["throughput_images_per_sec"].mean().reset_index()
    horiz = horiz.set_index("model").reindex(models).reset_index()
    x = np.arange(len(horiz))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v < 0.02 else "#4C72B0" for v in horiz["throughput_images_per_sec"].values]
    bars = ax.bar(x, horiz["throughput_images_per_sec"].values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars, horiz["throughput_images_per_sec"].values):
        bar.set_hatch("///" if v < 0.02 else "")
    _value_labels(ax, bars, fmt="{:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in horiz["model"].values], fontsize=10)
    ax.set_ylabel("Throughput (images / sec)", fontsize=12)
    ax.set_title("Inference Throughput per Model", fontsize=14, fontweight="bold")
    ax.set_ylim(0, horiz["throughput_images_per_sec"].max() * 1.35)
    ax.grid(**GRID_KW)
    _save(fig, "plot_4_throughput.png", output_dir)


# ===================================================================
# Plot 4b – Throughput (images per minute)
# ===================================================================
def plot_throughput_per_minute(plot_data, output_dir):
    df = _load_csv_required(plot_data, "plot_4_throughput.csv")
    modes = df["scaling_mode"].unique()
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(models))
    w = 0.25
    offsets = np.linspace(-w, w, len(modes))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, mode in enumerate(modes):
        mode_data = df[df["scaling_mode"] == mode].sort_values("model")
        img_per_min = _mode_values(mode_data, models, "throughput_images_per_sec") * 60
        bars = ax.bar(x + offsets[i], img_per_min, w, label=mode.capitalize(), hatch=HATCHES[i % len(HATCHES)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=10)
    ax.set_ylabel("Throughput (images / minute)", fontsize=12)
    ax.set_title("Inference Throughput per Model (images/min)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    all_vals = []
    for mode in modes:
        mode_data = df[df["scaling_mode"] == mode]
        all_vals.extend((mode_data["throughput_images_per_sec"].values * 60).tolist())
    ax.set_ylim(0, max(all_vals) * 1.3)
    ax.grid(**GRID_KW)
    _save(fig, "plot_4b_throughput_per_minute.png", output_dir)


# ===================================================================
# Plot 5 – Radar / Spider Chart
# ===================================================================
def plot_radar(plot_data, output_dir):
    df = _load_csv_required(plot_data, "plot_5_radar_model_comparison.csv")
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
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#DD8452", "#937860"]
    for idx in range(len(df)):
        vals = [norms[j][idx] for j in range(n)]
        vals += vals[:1]
        label = df["model"].iloc[idx].replace(" ", "\n")
        ax.plot(angles, vals, linewidth=2, label=label, color=palette[idx % len(palette)])
        ax.fill(angles, vals, alpha=0.08, color=palette[idx % len(palette)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Model Comparison Radar\n(normalised, higher = better)",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    _save(fig, "plot_5_model_comparison_radar.png", output_dir)


# ===================================================================
# Plot 6 – Heatmap (scaling mode × model)
# ===================================================================
def plot_heatmap(plot_data, output_dir):
    df = _load_csv_required(plot_data, "plot_6_scaling_mode_heatmap.csv")
    modes = [c for c in df.columns if c != "model"]
    data = df[modes].values
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        data, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=[m.replace(" ", "\n") for m in df["model"].values],
        yticklabels=[s.capitalize() for s in modes],
        linewidths=0.5, linecolor="white", ax=ax,
        annot_kws={"fontsize": 11},
    )
    ax.set_title("Inference Latency Heatmap: Model x Scaling Mode", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Scaling Mode", fontsize=12)
    fig.tight_layout()
    _save(fig, "plot_6_scaling_mode_impact.png", output_dir)


# ===================================================================
# Plot 7 – System Architecture Diagram (procedural, no data needed)
# ===================================================================
def plot_architecture(output_dir):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "DIFS OC v2 \u2014 INFaaS Cost-Based Scaling Architecture",
        fontsize=16, fontweight="bold", pad=15,
    )

    def box(x, y, w, h, text, color, fontsize=10, ec="black"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=ec, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, **kw):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8, **kw))

    def cost_arrow(x1, y1, x2, y2, **kw):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#CC3333", lw=2.2, **kw))

    CX = 2.25  # center x of main pipeline boxes
    # === MAIN PIPELINE (left column) ===
    box(0.5, 8.2, 3.5, 1.2, "Client\n(gRPC / REST)", "#BBDDF7", fontsize=11)
    box(0.5, 6.0, 3.5, 1.2, "Scheduler\nserver.py + scheduler.py", "#A8D5A2", fontsize=11)
    box(0.5, 3.8, 3.5, 1.2, "Worker Pool\nworker_pool.py (model_worker_counts)", "#E6C6A8", fontsize=11)
    box(0.5, 1.6, 3.5, 1.0, "Worker Nodes\npytorch_container_diffusion_v3.py", "#C7B8EA", fontsize=10)
    box(0.5, 0.2, 3.5, 0.9, "GPU (RTX 6000 Ada, 48 GB)\nCPU Offload", "#F7A8A8", fontsize=10)

    # Pipeline arrows (bottom edge to top edge of next)
    arrow(CX, 8.2, CX, 7.2)
    arrow(CX, 6.0, CX, 5.0)
    arrow(CX, 3.8, CX, 2.6)   # Worker Nodes top = 1.6 + 1.0 = 2.6
    arrow(CX, 1.6, CX, 1.1)

    # === INFaaS AUTOSCALER & COST MODEL (center-right) ===
    box(5.5, 7.2, 3.5, 1.2, "AutoScaler (INFaaS)\nautoscaler.py", "#FF7043", fontsize=11, ec="#CC3333")
    box(5.5, 5.2, 3.5, 1.2, "Cost Model\nStrategy A (sum) vs B (min)\nSTRATEGY_THRESHOLD = 1.5", "#FFCC80", fontsize=9, ec="#CC3333")
    box(5.5, 3.2, 3.5, 1.2, "Per-Model QPS Tracker\ncollect_per_model_qps()\nmodel popularity", "#FFE0B2", fontsize=9)

    ASX = 5.5  # left edge of AutoScaler column
    # Cost arrows (red) to scheduler and worker pool
    # Arrow starts from inside AutoScaler box (y=7.2 to 8.4)
    cost_arrow(ASX, 7.8, 4.0, 6.6, connectionstyle="arc3,rad=-0.3")
    cost_arrow(ASX, 7.8, 4.0, 4.4, connectionstyle="arc3,rad=-0.3")
    ax.text(4.2, 7.0, "scale_infaas()", fontsize=8, color="#CC3333", fontweight="bold", ha="center")
    ax.text(4.2, 5.6, "hot_swap()", fontsize=8, color="#CC3333", fontweight="bold", ha="center")

    # Arrows between autoscaler components (vertical: Cost Model below AutoScaler, QPS below that)
    arrow(7.25, 6.4, 7.25, 7.2)   # Cost Model top → AutoScaler bottom
    arrow(7.25, 4.4, 7.25, 5.2)   # QPS top → Cost Model bottom

    # Workers report to QPS tracker (dotted line, enters QPS from bottom)
    ax.annotate("", xy=(7.25, 3.2), xytext=(4.0, 2.1),
                 arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2, linestyle="dotted"))
    ax.text(5.6, 2.7, "per-model QPS", fontsize=8, color="gray", ha="center", fontweight="bold")

    # === SUPPORTING SERVICES (far right) ===
    box(10.5, 8.2, 4, 1.0, "Metrics & Observability\nPrometheus / Grafana", "#B0E0E6", fontsize=10)
    box(10.5, 6.2, 4, 1.0, "Model Registry\nsafetensors / checkpoints", "#D4A8D4", fontsize=10)
    box(10.5, 4.2, 4, 1.0, "Shared Storage\nNFS / MinIO", "#F5DEB3", fontsize=10)
    box(10.5, 2.2, 4, 1.0, "Model Variant Cache\nin-memory hot-swap buffer", "#C8E6C9", fontsize=10)

    # Arrows to/from right column
    arrow(10.5, 8.7, 9, 8.7)
    arrow(10.5, 6.7, 9, 6.7)
    arrow(10.5, 4.7, 9, 4.7)
    arrow(10.5, 2.7, 9, 2.7)

    # === LEGEND / MODE DESCRIPTION ===
    ax.add_patch(mpatches.FancyBboxPatch(
        (5.5, 0.1), 9, 1.1, boxstyle="round,pad=0.1",
        facecolor="#FFF8E1", edgecolor="#FFB300", linewidth=1.2,
    ))
    ax.text(10, 0.6,
        "Horizontal:  add/remove worker containers (cost = Strategy B \u2014 fastest variant)\n"
        "Vertical:    hot-swap model variants on existing workers (cost = Strategy A \u2014 sum of all variants)\n"
        "Hybrid:      cost-based decision \u2014 if vertical cost < 1.5\u00d7 horizontal cost, use vertical; else horizontal\n"
        "Delayed scale-down: workers must be idle for 20 consecutive cycles before removal",
        ha="center", va="center", fontsize=7.5, color="#333")

    _save(fig, "plot_7_system_architecture.png", output_dir)


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate benchmark bar chart plots.")
    parser.add_argument("--plot-data-dir", default=None, help="Path to plot_data/ directory")
    parser.add_argument("--output-dir", default=None, help="Path to output plots directory")
    args = parser.parse_args()

    plot_data = args.plot_data_dir or DEFAULT_PLOT_DATA
    output_dir = args.output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)

    print("Generating bar plots ...")
    plot_latency_comparison(plot_data, output_dir)
    plot_model_load_time(plot_data, output_dir)
    plot_gpu_memory(plot_data, output_dir)
    plot_throughput(plot_data, output_dir)
    plot_throughput_per_minute(plot_data, output_dir)
    plot_radar(plot_data, output_dir)
    plot_heatmap(plot_data, output_dir)
    plot_architecture(output_dir)
    print(f"\nAll 8 plots saved to:\n  {output_dir}")


if __name__ == "__main__":
    main()
