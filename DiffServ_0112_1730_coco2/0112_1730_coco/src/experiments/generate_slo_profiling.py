#!/usr/bin/env python3

"""
generate_slo_profiling.py

Generates slo_exec_profiling.csv for the SLO pipeline from the existing
benchmark latency data (measured at BENCHMARK_STEPS=20, one request per model
with sequential CPU offload on RTX 6000 Ada).

The accuracy SLO levels map to canonical step counts:
    low=20, medium=30, high=40
Because the benchmark was measured at 20 steps, execution times for 30 and 40
steps are extrapolated with the standard diffusion assumption that inference
time scales approximately linearly with the number of denoising steps:
    exec(steps) = exec_20 * (steps / 20)

This file is a lightweight substitute for a fresh GPU profiling run
(src/experiments/profile_steps_slo.py). If real profiling data becomes
available, overwrite the produced CSV with the measured values.

Output: results/model_benchmark/plot_data_5model/slo_exec_profiling.csv
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_registry import ALL_MODELS

# The 5 models used in the 5-model benchmark and the CSV names in the plot data
MODEL_NAME_MAP = {
    "SD-v1.5": 3,        # ALL_MODELS[3] short_name "SD-v1-5"
    "SD-2.1": 4,         # ALL_MODELS[4] short_name "SD-2-1"
    "SDXL-Base": 5,      # ALL_MODELS[5] short_name "SDXL-Base"
    "SD-3.5-Medium": 8,  # ALL_MODELS[8] short_name "SD-3-5-Medium"
    "SD-3.5-Large": 9,   # ALL_MODELS[9] short_name "SD-3-5-Large"
}

BASE_STEPS = 20  # BENCHMARK_STEPS used for the measured latency data

# Canonical step counts per accuracy level
ACCURACY_STEPS = {"low": 20, "medium": 30, "high": 40}

LATENCY_CSV = Path("results/model_benchmark/plot_data_5model/plot_1_latency_comparison.csv")
OUTPUT_CSV = Path("results/model_benchmark/plot_data_5model/slo_exec_profiling.csv")


def load_latency_csv(path):
    """Return {model_name: avg_latency_sec} averaged across scaling modes."""
    rows = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["model"].strip()
            try:
                lat = float(row["latency_sec"])
            except (KeyError, ValueError):
                continue
            rows.setdefault(name, []).append(lat)
    return {name: sum(v) / len(v) for name, v in rows.items()}


def main():
    if not LATENCY_CSV.exists():
        print(f"[PROFILING] ERROR: benchmark latency CSV not found: {LATENCY_CSV}")
        sys.exit(1)

    measured = load_latency_csv(LATENCY_CSV)
    print(f"[PROFILING] Loaded {len(measured)} model averages from {LATENCY_CSV}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "model_name", "steps", "exec_time_sec", "source"]
        )
        writer.writeheader()

        for csv_name, model_id in MODEL_NAME_MAP.items():
            if csv_name not in measured:
                print(f"[PROFILING] WARNING: no latency data for {csv_name}, skipping")
                continue

            exec_20 = measured[csv_name]
            model_name = ALL_MODELS[model_id]["short_name"]

            for steps in (20, 30, 40):
                exec_time = exec_20 * (steps / BASE_STEPS)
                source = "measured" if steps == BASE_STEPS else "linear_extrapolation"
                writer.writerow({
                    "model": model_id,
                    "model_name": model_name,
                    "steps": steps,
                    "exec_time_sec": round(exec_time, 3),
                    "source": source,
                })
                print(
                    f"[PROFILING] model={model_id} ({model_name}) "
                    f"steps={steps} exec={exec_time:.3f}s [{source}]"
                )

    print(f"[PROFILING] Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
