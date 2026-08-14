import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("metrics/worker_metrics.csv")

gpu = (
    df
    .groupby("model_name")["gpu_memory_MB"]
    .mean()
)

plt.figure(figsize=(8,5))

gpu.plot(
    kind="bar",
    edgecolor="black"
)

plt.ylabel("GPU memory (MB)")
plt.xlabel("Diffusion model")
plt.title("Average GPU memory by model")

plt.tight_layout()

Path("plots").mkdir(exist_ok=True)

plt.savefig(
    "plots/gpu_memory_vs_model.png",
    dpi=300
)