# Date: 2026.05.15
# Purpose: For plotting performance metrics
# Adapted: from 'plot_metrics.py'

import pandas as pd
import matplotlib.pyplot as plt

############################################################
# LOAD METRICS
############################################################

server_df = pd.read_csv("metrics/server_metrics.csv")

client_df = pd.read_csv("metrics/client_metrics.csv")

############################################################
# LATENCY PLOT
############################################################

plt.figure(figsize=(10,5))

plt.plot(client_df["latency"])

plt.xlabel("Response Index")
plt.ylabel("Latency (s)")
plt.title("Client End-to-End Latency")

plt.grid(True)

plt.savefig("metrics/client_latency.png")

############################################################
# QUEUE LENGTH
############################################################

plt.figure(figsize=(10,5))

plt.plot(server_df["queue_length"])

plt.xlabel("Sample")
plt.ylabel("Queue Length")
plt.title("Scheduler Queue Length")

plt.grid(True)

plt.savefig("metrics/queue_length.png")

############################################################
# ARRIVAL VS SERVICE RATE
############################################################

plt.figure(figsize=(10,5))

plt.plot(server_df["arrival_rate"], label="Arrival Rate λ")

plt.plot(server_df["service_rate"], label="Service Rate μ")

plt.xlabel("Sample")
plt.ylabel("Requests/sec")
plt.title("Arrival vs Service Rate")

plt.legend()

plt.grid(True)

plt.savefig("metrics/rates.png")

############################################################
# WORKER COUNT
############################################################

plt.figure(figsize=(10,5))

plt.plot(server_df["workers"])

plt.xlabel("Sample")
plt.ylabel("Workers")
plt.title("Autoscaling Worker Count")

plt.grid(True)

plt.savefig("metrics/workers.png")

############################################################
# THROUGHPUT
############################################################

throughput = []

timestamps = client_df["timestamp"].values

for i in range(1, len(timestamps)):

    dt = timestamps[i] - timestamps[i-1]

    if dt > 0:
        throughput.append(1.0 / dt)
    else:
        throughput.append(0)

plt.figure(figsize=(10,5))

plt.plot(throughput)

plt.xlabel("Sample")
plt.ylabel("Images/sec")
plt.title("Inference Throughput")

plt.grid(True)

plt.savefig("metrics/throughput.png")

############################################################
# SUMMARY
############################################################

print("\n===== DIFS PERFORMANCE SUMMARY =====")

print(f"Average latency: {client_df['latency'].mean():.3f}s")

print(f"Max latency: {client_df['latency'].max():.3f}s")

print(f"Average queue length: {server_df['queue_length'].mean():.2f}")

print(f"Max queue length: {server_df['queue_length'].max():.2f}")

print(f"Average workers: {server_df['workers'].mean():.2f}")

print(f"Peak workers: {server_df['workers'].max()}")

print(f"Average arrival rate λ: {server_df['arrival_rate'].mean():.2f}")

print(f"Average service rate μ: {server_df['service_rate'].mean():.2f}")

print("\nPlots saved in metrics/")