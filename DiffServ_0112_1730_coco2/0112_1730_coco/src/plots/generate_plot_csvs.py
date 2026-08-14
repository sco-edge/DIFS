#!/usr/bin/env python3
"""
generate_plot_csvs.py

Reads raw benchmark metrics from results/model_benchmark/ and produces
per-plot CSV data files in results/model_benchmark/plot_data/.

Data sources:
  - benchmark_summary.csv              -> latency per (mode, model)
  - <mode>/model_<name>/metrics/model_load_metrics.csv  -> load_time
  - <mode>/model_<name>/metrics/gpu_memory_metrics.csv  -> allocated_mb, reserved_mb
  - <mode>/model_<name>/metrics/worker_metrics.csv      -> gpu_memory_MB, gpu_peak_MB

Usage:
    python generate_plot_csvs.py [--results-dir DIR]
"""
import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
import glob


def find_results_dir():
    """Default: results/model_benchmark relative to this script."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "results", "model_benchmark",
    )


def load_benchmark_summary(results_dir):
    path = os.path.join(results_dir, "benchmark_summary.csv")
    if not os.path.exists(path):
        print(f"ERROR: benchmark_summary.csv not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df["model_name"] = df["model_name"].apply(normalize_model_name)
    return df


def normalize_model_name(name):
    """Convert SD-v1-5 -> SD-v1.5, SD-2-1 -> SD-2.1, SD-3-5-* -> SD-3.5-*"""
    return re.sub(r'(\d+)-(\d+)', r'\1.\2', name)


MODEL_ORDER = ["SD-v1.4-dir", "SD-v1.4-safetensors", "SD-v1.5", "SD-2.1",
               "SDXL-Base", "SD-3.5-Medium", "SD-3.5-Large", "DreamShaper-8"]

def load_per_run_metrics(results_dir):
    """
    Scan all <mode>/model_<name>/metrics/*.csv and return a DataFrame
    with columns: mode, model_name, load_time, gpu_memory_MB, gpu_peak_MB, latency_sec
    """
    rows = []
    pattern = os.path.join(results_dir, "*", "model_*", "metrics")
    for metrics_dir in sorted(glob.glob(pattern)):
        parts = metrics_dir.split(os.sep)
        # extract mode and model_name from path
        # .../results/model_benchmark/<mode>/model_<name>/metrics
        mode = parts[-3]
        model_name = normalize_model_name(parts[-2].replace("model_", ""))

        # model_load_metrics.csv
        load_path = os.path.join(metrics_dir, "model_load_metrics.csv")
        load_time = None
        if os.path.exists(load_path):
            try:
                df_load = pd.read_csv(load_path)
                if "load_time" in df_load.columns and len(df_load) > 0:
                    load_time = df_load["load_time"].mean()
            except Exception:
                pass

        # gpu_memory_metrics.csv
        gpu_mem_path = os.path.join(metrics_dir, "gpu_memory_metrics.csv")
        gpu_memory_MB = None
        gpu_peak_MB = None
        if os.path.exists(gpu_mem_path):
            try:
                df_gpu = pd.read_csv(gpu_mem_path)
                if "allocated_mb" in df_gpu.columns and len(df_gpu) > 0:
                    gpu_memory_MB = df_gpu["allocated_mb"].mean()
                if "reserved_mb" in df_gpu.columns and len(df_gpu) > 0:
                    gpu_peak_MB = df_gpu["reserved_mb"].max()
            except Exception:
                pass

        # worker_metrics.csv (fallback for gpu memory)
        worker_path = os.path.join(metrics_dir, "worker_metrics.csv")
        if gpu_memory_MB is None or gpu_peak_MB is None:
            if os.path.exists(worker_path):
                try:
                    df_w = pd.read_csv(worker_path)
                    if gpu_memory_MB is None and "gpu_memory_MB" in df_w.columns and len(df_w) > 0:
                        gpu_memory_MB = df_w["gpu_memory_MB"].mean()
                    if gpu_peak_MB is None and "gpu_peak_MB" in df_w.columns and len(df_w) > 0:
                        gpu_peak_MB = df_w["gpu_peak_MB"].max()
                except Exception:
                    pass

        # average latency from worker_metrics
        latency_sec = None
        if os.path.exists(worker_path):
            try:
                df_w = pd.read_csv(worker_path)
                if "latency_sec" in df_w.columns and len(df_w) > 0:
                    latency_sec = df_w["latency_sec"].mean()
            except Exception:
                pass

        rows.append({
            "mode": mode,
            "model_name": model_name,
            "load_time": load_time,
            "gpu_memory_MB": gpu_memory_MB,
            "gpu_peak_MB": gpu_peak_MB,
            "latency_sec": latency_sec,
        })

    return pd.DataFrame(rows)


def build_plot_data(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary = load_benchmark_summary(results_dir)
    per_run = load_per_run_metrics(results_dir)

    MODES = sorted(summary["scaling_mode"].unique())
    present = set(summary["model_name"].unique())
    MODELS_ORDER = [m for m in MODEL_ORDER if m in present]

    # ── Plot 1: Inference Latency (from benchmark_summary) ──────────────────
    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "model": row["model_name"],
            "scaling_mode": row["scaling_mode"],
            "latency_sec": round(row["avg_latency_sec"], 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "plot_1_latency_comparison.csv"), index=False)
    print("  plot_1_latency_comparison.csv")

    # ── Plot 2: Model Load Time ─────────────────────────────────────────────
    if per_run["load_time"].notna().any():
        rows = []
        for _, row in per_run[per_run["load_time"].notna()].iterrows():
            rows.append({
                "model": row["model_name"],
                "scaling_mode": row["mode"],
                "load_time_sec": round(row["load_time"], 2),
            })
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["model", "scaling_mode", "load_time_sec"])
    df.to_csv(os.path.join(output_dir, "plot_2_model_load_time.csv"), index=False)
    print("  plot_2_model_load_time.csv")

    # ── Plot 3: GPU Memory ──────────────────────────────────────────────────
    gpu_data = per_run[per_run["gpu_memory_MB"].notna()].drop_duplicates(
        subset=["model_name"], keep="first"
    )
    gpu_data = gpu_data.set_index("model_name").reindex(MODELS_ORDER).reset_index()
    gpu_data = gpu_data[gpu_data["model_name"].notna()]
    if len(gpu_data) > 0:
        df = pd.DataFrame({
            "model": gpu_data["model_name"].values,
            "gpu_used_mb": gpu_data["gpu_memory_MB"].round(2).values,
            "gpu_peak_mb": gpu_data["gpu_peak_MB"].round(2).values,
        })
    else:
        df = pd.DataFrame(columns=["model", "gpu_used_mb", "gpu_peak_mb"])
    df.to_csv(os.path.join(output_dir, "plot_3_gpu_memory.csv"), index=False)
    print("  plot_3_gpu_memory.csv")

    # ── Plot 4: Throughput (derived: workload / latency) ────────────────────
    workload = summary["workload"].iloc[0] if "workload" in summary.columns else 5
    rows = []
    for _, row in summary.iterrows():
        tp = workload / row["avg_latency_sec"]
        rows.append({
            "model": row["model_name"],
            "scaling_mode": row["scaling_mode"],
            "throughput_images_per_sec": round(tp, 6),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "plot_4_throughput.csv"), index=False)
    print("  plot_4_throughput.csv")

    # ── Plot 5: Radar (averaged across modes) ───────────────────────────────
    rows = []
    for model in MODELS_ORDER:
        model_rows = summary[summary["model_name"] == model]
        avg_lat = model_rows["avg_latency_sec"].mean()
        tp = workload / avg_lat

        load_rows = per_run[per_run["model_name"] == model]
        avg_load = load_rows["load_time"].mean() if load_rows["load_time"].notna().any() else 0

        gpu_rows = per_run[(per_run["model_name"] == model) & per_run["gpu_peak_MB"].notna()]
        peak = gpu_rows["gpu_peak_MB"].iloc[0] if len(gpu_rows) > 0 else 0

        rows.append({
            "model": model,
            "avg_latency_sec": round(avg_lat, 2),
            "avg_load_time_sec": round(avg_load, 2),
            "gpu_peak_mb": round(peak, 2),
            "throughput_images_per_sec": round(tp, 6),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "plot_5_radar_model_comparison.csv"), index=False)
    print("  plot_5_radar_model_comparison.csv")

    # ── Plot 6: Heatmap ─────────────────────────────────────────────────────
    rows = []
    for model in MODELS_ORDER:
        row_data = {"model": model}
        for mode in MODES:
            val = summary[
                (summary["model_name"] == model) & (summary["scaling_mode"] == mode)
            ]["avg_latency_sec"]
            row_data[mode] = round(val.iloc[0], 2) if len(val) > 0 else None
        rows.append(row_data)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "plot_6_scaling_mode_heatmap.csv"), index=False)
    print("  plot_6_scaling_mode_heatmap.csv")

    # ── Line Plot CSVs (wide format) ────────────────────────────────────────
    for prefix, col_src, col_dst, src_df in [
        ("line_1", "avg_latency_sec", "latency_sec", summary),
        ("line_4", None, None, summary),  # throughput derived below
    ]:
        pass  # handled below

    # Line 1: Latency wide
    rows = []
    for model in MODELS_ORDER:
        row_data = {"model": model}
        for mode in MODES:
            val = summary[
                (summary["model_name"] == model) & (summary["scaling_mode"] == mode)
            ]["avg_latency_sec"]
            row_data[mode] = round(val.iloc[0], 2) if len(val) > 0 else None
        rows.append(row_data)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "line_1_latency_comparison.csv"), index=False)
    print("  line_1_latency_comparison.csv")

    # Line 2: Load Time wide
    rows = []
    for model in MODELS_ORDER:
        row_data = {"model": model}
        for mode in MODES:
            val = per_run[
                (per_run["model_name"] == model) & (per_run["mode"] == mode)
            ]["load_time"]
            row_data[mode] = round(val.iloc[0], 2) if len(val) > 0 and val.notna().any() else None
        rows.append(row_data)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "line_2_model_load_time.csv"), index=False)
    print("  line_2_model_load_time.csv")

    # Line 3: GPU Memory wide
    gpu_models = per_run[per_run["gpu_memory_MB"].notna()].drop_duplicates(
        subset=["model_name"], keep="first"
    ).set_index("model_name").reindex(MODELS_ORDER).reset_index()
    if len(gpu_models) > 0 and gpu_models["gpu_memory_MB"].notna().any():
        df = pd.DataFrame({
            "model": gpu_models["model_name"].values,
            "gpu_used_mb": gpu_models["gpu_memory_MB"].round(2).values,
            "gpu_peak_mb": gpu_models["gpu_peak_MB"].round(2).values,
        })
    else:
        df = pd.DataFrame(columns=["model", "gpu_used_mb", "gpu_peak_mb"])
    df.to_csv(os.path.join(output_dir, "line_3_gpu_memory.csv"), index=False)
    print("  line_3_gpu_memory.csv")

    # Line 4: Throughput wide
    rows = []
    for model in MODELS_ORDER:
        row_data = {"model": model}
        for mode in MODES:
            val = summary[
                (summary["model_name"] == model) & (summary["scaling_mode"] == mode)
            ]["avg_latency_sec"]
            if len(val) > 0:
                row_data[mode] = round(workload / val.iloc[0], 6)
            else:
                row_data[mode] = None
        rows.append(row_data)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "line_4_throughput.csv"), index=False)
    print("  line_4_throughput.csv")

    # ── Switching Time CSVs ─────────────────────────────────────────────────
    rows = []
    for _, row in per_run[per_run["load_time"].notna()].iterrows():
        rows.append({
            "model": row["model_name"],
            "scaling_mode": row["mode"],
            "switch_time_sec": round(row["load_time"], 2),
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "switching_time_bar.csv"), index=False)
    print("  switching_time_bar.csv")

    rows = []
    for model in MODELS_ORDER:
        row_data = {"model": model}
        for mode in MODES:
            val = per_run[
                (per_run["model_name"] == model) & (per_run["mode"] == mode)
            ]["load_time"]
            row_data[mode] = round(val.iloc[0], 2) if len(val) > 0 and val.notna().any() else None
        rows.append(row_data)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "switching_time_line.csv"), index=False)
    print("  switching_time_line.csv")

    # ── Pairwise Switching Time Matrix ───────────────────────────────────────
    # If raw pairwise CSV exists (from collect_pairwise_switching.py), use it
    pw_raw = os.path.join(results_dir, "plot_data", "pairwise_switching.csv")
    pw_derived = os.path.join(output_dir, "plot_pairwise_switching.csv")

    if not os.path.exists(pw_raw):
        # Derive N x N matrix from average load times
        load_avg = per_run[per_run["load_time"].notna()].groupby("model_name")["load_time"].mean()
        if len(load_avg) > 0:
            load_avg = load_avg.sort_index()
            model_list = list(load_avg.index)
            n = len(model_list)
            matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i][j] = round(load_avg.iloc[j], 2)
            df_pw = pd.DataFrame(matrix, index=model_list, columns=model_list)
            df_pw.to_csv(pw_derived)
            print(f"  plot_pairwise_switching.csv (derived from load times)")
        else:
            print("  plot_pairwise_switching.csv SKIPPED (no load time data)")
    else:
        print(f"  plot_pairwise_switching.csv (from {pw_raw})")

    # ── Manifest ────────────────────────────────────────────────────────────
    manifest = """\
# Plot Data Files -- DIFS Benchmark
# Auto-generated by generate_plot_csvs.py from raw benchmark metrics.
# Run: python src/plots/generate_plot_csvs.py --results-dir results/model_benchmark

plot_file,data_csv,script,original_source_csvs
plot_1_latency_comparison.png,plot_1_latency_comparison.csv,generate_all_plots.py,benchmark_summary.csv
plot_2_model_load_time.png,plot_2_model_load_time.csv,generate_all_plots.py,*/metrics/model_load_metrics.csv
plot_3_gpu_memory.png,plot_3_gpu_memory.csv,generate_all_plots.py,*/metrics/gpu_memory_metrics.csv + */metrics/worker_metrics.csv
plot_4_throughput.png,plot_4_throughput.csv,generate_all_plots.py,derived from benchmark_summary.csv (workload / avg_latency_sec)
plot_4b_throughput_per_minute.png,plot_4_throughput.csv,generate_all_plots.py,derived from benchmark_summary.csv (throughput * 60)
plot_5_model_comparison_radar.png,plot_5_radar_model_comparison.csv,generate_all_plots.py,benchmark_summary.csv + */metrics/*
plot_6_scaling_mode_impact.png,plot_6_scaling_mode_heatmap.csv,generate_all_plots.py,benchmark_summary.csv
plot_7_system_architecture.png,(no data -- drawn procedurally),generate_all_plots.py,N/A
line_latency_comparison.png,line_1_latency_comparison.csv,generate_line_plots.py,benchmark_summary.csv
line_model_load_time.png,line_2_model_load_time.csv,generate_line_plots.py,*/metrics/model_load_metrics.csv
line_gpu_memory.png,line_3_gpu_memory.csv,generate_line_plots.py,*/metrics/gpu_memory_metrics.csv + */metrics/worker_metrics.csv
line_throughput.png,line_4_throughput.csv,generate_line_plots.py,derived from benchmark_summary.csv
line_all_metrics.png,(uses line_1 through line_4 CSVs),generate_line_plots.py,benchmark_summary.csv + */metrics/*
switching_time_bar.png,switching_time_bar.csv,generate_switching_plots.py,*/metrics/model_load_metrics.csv
switching_time_line.png,switching_time_line.csv,generate_switching_plots.py,*/metrics/model_load_metrics.csv
pairwise_switching_heatmap.png,plot_pairwise_switching.csv,generate_switching_plots.py,collect_pairwise_switching.py or derived from load times
"""
    manifest_path = os.path.join(output_dir, "PLOT_MANIFEST.csv")
    with open(manifest_path, "w") as f:
        f.write(manifest)
    print(f"\n  Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-plot CSV data files from raw benchmark metrics."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to results/model_benchmark/ (default: auto-detect relative to script)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir or find_results_dir()
    output_dir = os.path.join(results_dir, "plot_data")

    if not os.path.exists(os.path.join(results_dir, "benchmark_summary.csv")):
        print(f"ERROR: No benchmark_summary.csv found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading metrics from: {results_dir}")
    print(f"Writing plot CSVs to: {output_dir}\n")

    build_plot_data(results_dir, output_dir)

    print(f"\nAll CSV files saved to: {output_dir}")


if __name__ == "__main__":
    main()
