import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics/worker_metrics.csv")

fid = (
    df[df["fid"] >= 0]
    .groupby("model_name")["fid"]
    .mean()
)

plt.figure(figsize=(8,5))

fid.plot(
    kind="bar",
    edgecolor="black"
)

plt.ylabel("FID")
plt.title("FID by diffusion model")

plt.tight_layout()

plt.savefig(
    "plots/fid_vs_model.png",
    dpi=300
)