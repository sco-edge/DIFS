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