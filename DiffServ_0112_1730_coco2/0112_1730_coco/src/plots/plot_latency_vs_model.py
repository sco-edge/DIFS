"""
plot_latency_vs_model.py

Plots inference latency vs diffusion model with error bars and statistics.

Reads from metrics/worker_metrics.csv.

Generates:
  - plots/latency_vs_model.png          (bar chart with error bars)
  - plots/latency_distribution.png      (box plot per model)
  - plots/latency_percentiles.png       (p50/p95/p99 per model)
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


worker_file = os.path.join(METRICS_DIR, "worker_metrics.csv")

if not os.path.exists(worker_file):
    print(f"[PLOT] {worker_file} not found")
    exit(0)

wm = pd.read_csv(worker_file)

if "model_name" not in wm.columns or "latency_sec" not in wm.columns:
    print("[PLOT] worker_metrics.csv missing model_name or latency_sec columns")
    exit(0)


############################################################
# 1. Bar Chart with Error Bars (mean +/- std)
############################################################

latency_stats = (
    wm.groupby("model_name")["latency_sec"]
    .agg(["mean", "std", "median", "min", "max", "count"])
    .sort_values("mean")
)

fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(latency_stats)))

bars = ax.bar(
    range(len(latency_stats)),
    latency_stats["mean"],
    yerr=latency_stats["std"].fillna(0),
    capsize=5,
    edgecolor="black",
    alpha=0.85,
    color=colors
)

ax.set_xticks(range(len(latency_stats)))
ax.set_xticklabels(latency_stats.index, rotation=45, ha="right", fontsize=10)

ax.set_ylabel("Average Latency (seconds)", fontsize=12)
ax.set_xlabel("Diffusion Model", fontsize=12)
ax.set_title("Average Inference Latency by Diffusion Model\n(with std dev error bars)", fontsize=13)

# Value labels
for bar, val, std in zip(bars, latency_stats["mean"], latency_stats["std"].fillna(0)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + std + 0.05,
        f"{val:.2f}s",
        ha="center", va="bottom", fontsize=9, fontweight="bold"
    )

# Add sample count annotation
for i, (_, row) in enumerate(latency_stats.iterrows()):
    ax.text(
        i, 0.02,
        f"n={int(row['count'])}",
        ha="center", va="bottom", fontsize=8, color="white", fontweight="bold"
    )

save(fig, "latency_vs_model.png")
print("Saved plots/latency_vs_model.png")


############################################################
# 2. Box Plot per Model (latency distribution)
############################################################

models_sorted = latency_stats.index.tolist()

fig, ax = plt.subplots(figsize=(10, 6))

box_data = [wm[wm["model_name"] == m]["latency_sec"].values for m in models_sorted]

bp = ax.boxplot(
    box_data,
    labels=models_sorted,
    patch_artist=True,
    showmeans=True,
    meanprops=dict(marker="D", markerfacecolor="red", markersize=6)
)

colors = plt.cm.Set3(np.linspace(0, 1, len(models_sorted)))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)

ax.set_xticklabels(models_sorted, rotation=45, ha="right")
ax.set_ylabel("Latency (seconds)")
ax.set_xlabel("Diffusion Model")
ax.set_title("Inference Latency Distribution by Model")

save(fig, "latency_distribution.png")
print("Saved plots/latency_distribution.png")


############################################################
# 3. Percentile Comparison (p50, p95, p99)
############################################################

percentile_data = []
for model in models_sorted:
    model_latencies = wm[wm["model_name"] == model]["latency_sec"]
    percentile_data.append({
        "model": model,
        "p50": model_latencies.quantile(0.50),
        "p95": model_latencies.quantile(0.95),
        "p99": model_latencies.quantile(0.99),
    })

pdf = pd.DataFrame(percentile_data).set_index("model")

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(pdf))
width = 0.25

bars_p50 = ax.bar(x - width, pdf["p50"], width, label="P50", color="#2196F3", edgecolor="black")
bars_p95 = ax.bar(x, pdf["p95"], width, label="P95", color="#FF9800", edgecolor="black")
bars_p99 = ax.bar(x + width, pdf["p99"], width, label="P99", color="#F44336", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(pdf.index, rotation=45, ha="right")
ax.set_ylabel("Latency (seconds)")
ax.set_xlabel("Diffusion Model")
ax.set_title("Latency Percentiles by Diffusion Model")
ax.legend()

save(fig, "latency_percentiles.png")
print("Saved plots/latency_percentiles.png")


############################################################
# Print Summary
############################################################

print()
print("=" * 60)
print("Latency Summary by Model")
print("=" * 60)
print(latency_stats.to_string())
print()
print("Percentiles:")
print(pdf.to_string())
