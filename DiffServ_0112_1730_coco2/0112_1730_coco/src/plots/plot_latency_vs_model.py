import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

metrics = pd.read_csv("metrics/worker_metrics.csv")

lat = (
    metrics
    .groupby("model_name")["latency_sec"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8,5))

lat.plot(
    kind="bar",
    edgecolor="black"
)

plt.ylabel("Average latency (seconds)")
plt.xlabel("Diffusion model")
plt.title("Average inference latency by diffusion model")

plt.tight_layout()

Path("plots").mkdir(exist_ok=True)

plt.savefig(
    "plots/latency_vs_model.png",
    dpi=300
)

print(lat)
print("Saved plots/latency_vs_model.png")