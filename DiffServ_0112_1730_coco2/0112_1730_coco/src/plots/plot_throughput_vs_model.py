import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("metrics/worker_metrics.csv")

df["throughput"] = (
    df["images_generated"] /
    df["latency_sec"]
)

thr = (
    df
    .groupby("model_name")["throughput"]
    .mean()
)

plt.figure(figsize=(8,5))

thr.plot(
    kind="bar",
    edgecolor="black"
)

plt.ylabel("Images / second")
plt.xlabel("Diffusion model")
plt.title("Throughput by diffusion model")

plt.tight_layout()

Path("plots").mkdir(exist_ok=True)

plt.savefig(
    "plots/throughput_vs_model.png",
    dpi=300
)

print(thr)