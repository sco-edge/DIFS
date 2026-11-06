"""
run_model_benchmark.py

Runs a full model benchmark across all 7 SD models and all 3 scaling modes.
For each (scaling_mode, model_id) pair:
  1. Starts server + worker(s) with that model
  2. Sends benchmark workload via client
  3. Collects per-model metrics (latency, switching time, GPU memory, FID)
  4. Stops server + worker(s)
  5. Records results

Outputs: results/model_benchmark/<scaling_mode>/model_<name>/

Usage:
    python -m src.experiments.run_model_benchmark
    python -m src.experiments.run_model_benchmark --models 1 2 3
    python -m src.experiments.run_model_benchmark --modes vertical horizontal
    python -m src.experiments.run_model_benchmark --docker
"""

import time
import json
import csv
import shutil
import argparse
import sys
from pathlib import Path

# Ensure src/ is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import PROJECT_ROOT, RESULTS_DIR, METRICS_DIR, CLIENT_RECEIVED_DIR

# Add the experiments package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_registry import get_model_filename, get_model_short_name, ALL_MODELS
from experiments.experiment_config import (
    MODELS, SCALING_MODES, BENCHMARK_WORKLOAD, BENCHMARK_STEPS,
    BENCHMARK_SAMPLER, SEED, POST_EXPERIMENT_WAIT,
    SCHEDULER_HOST, SCHEDULER_PORT, SERVER_PORT, WORKER_PORT,
    SERVER_COMMAND, CLIENT_SCRIPT, WORKER_SCRIPT,
    RUN_DIRECT
)
from experiments.process_manager import ProcessManager


def clear_metrics():
    """Clear previous metrics CSV files for a clean benchmark run."""
    metrics_dir = Path("metrics")
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.csv"):
            f.unlink()
        print(f"[BENCHMARK] Cleared metrics directory")


def clear_client_images():
    """Clear previously received client images."""
    if CLIENT_RECEIVED_DIR.exists():
        for f in CLIENT_RECEIVED_DIR.glob("*.png"):
            f.unlink()
        print(f"[BENCHMARK] Cleared client images")


def save_benchmark_config(run_dir, scaling_mode, model_id, workload, steps, sampler, seed):
    """Save experiment configuration for this benchmark run."""
    config = {
        "scaling_mode": scaling_mode,
        "model_id": model_id,
        "model_name": get_model_short_name(model_id),
        "model_filename": get_model_filename(model_id),
        "workload": workload,
        "steps": steps,
        "sampler": sampler,
        "seed": seed,
    }
    with open(run_dir / "experiment.json", "w") as f:
        json.dump(config, f, indent=4)


def copy_benchmark_metrics(run_dir):
    """Copy metrics CSVs to the benchmark run directory."""
    metrics_out = run_dir / "metrics"
    metrics_out.mkdir(exist_ok=True)
    for f in Path("metrics").glob("*.csv"):
        shutil.copy(f, metrics_out / f.name)


def copy_benchmark_images(run_dir):
    """Copy generated images to the benchmark run directory."""
    images_out = run_dir / "images"
    images_out.mkdir(exist_ok=True)
    for f in CLIENT_RECEIVED_DIR.glob("*.png"):
        shutil.copy(f, images_out / f.name)


def run_single_benchmark(pm, scaling_mode, model_id, workload, steps, sampler, seed):
    """
    Run a single benchmark: one model, one scaling mode.
    Returns the run_dir Path.
    """
    model_name = get_model_short_name(model_id)

    print()
    print("=" * 70)
    print(f"[BENCHMARK] Scaling={scaling_mode} | Model={model_name} (id={model_id})")
    print(f"[BENCHMARK] Workload={workload} | Steps={steps} | Sampler={sampler}")
    print("=" * 70)

    # Clear for clean measurement
    clear_metrics()
    clear_client_images()

    # Create output directory
    run_dir = RESULTS_DIR / "model_benchmark" / scaling_mode / f"model_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set scaling mode in config
    import config as runtime_config
    runtime_config.SCALING_MODE = scaling_mode

    # Reset port counter so worker always uses port 50060
    pm.worker_port_counter = WORKER_PORT

    # Start worker
    pm.start_worker(model_id=model_id)

    # Start server
    pm.start_server()

    # Run client
    cmd = [
        sys.executable,
        CLIENT_SCRIPT,
        "-prompt_count", str(workload),
        "-host", SCHEDULER_HOST,
        "-port", str(SERVER_PORT),
        "-steps", str(steps),
        "-sampler", str(sampler),
        "-seed", str(seed)
    ]

    print(f"[BENCHMARK] Running client...")
    import subprocess
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"[BENCHMARK WARNING] Client exited with code {result.returncode}")

    # Wait for metrics to flush
    time.sleep(POST_EXPERIMENT_WAIT)

    # Save results
    save_benchmark_config(run_dir, scaling_mode, model_id, workload, steps, sampler, seed)
    copy_benchmark_metrics(run_dir)
    copy_benchmark_images(run_dir)

    # Shutdown
    pm.stop_server()
    pm.stop_worker()

    print(f"[BENCHMARK] Completed: {scaling_mode}/{model_name}")

    return run_dir


def generate_summary_csv(benchmark_results_dir):
    """
    Read all per-model experiment.json files and generate a combined summary CSV.
    """
    summary_rows = []
    summary_file = benchmark_results_dir / "benchmark_summary.csv"

    for scaling_dir in sorted(benchmark_results_dir.iterdir()):
        if not scaling_dir.is_dir():
            continue

        scaling_mode = scaling_dir.name

        for model_dir in sorted(scaling_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            config_file = model_dir / "experiment.json"
            if not config_file.exists():
                continue

            with open(config_file) as f:
                cfg = json.load(f)

            # Check if worker_metrics.csv exists for latency data
            worker_metrics = model_dir / "metrics" / "worker_metrics.csv"
            avg_latency = None
            if worker_metrics.exists():
                import pandas as pd
                df = pd.read_csv(worker_metrics)
                if "latency_sec" in df.columns and len(df) > 0:
                    avg_latency = df["latency_sec"].mean()

            # Check for switch_metrics.csv
            switch_metrics = model_dir / "metrics" / "switch_metrics.csv"
            avg_switch = None
            if switch_metrics.exists():
                import pandas as pd
                df = pd.read_csv(switch_metrics)
                if "switch_time" in df.columns and len(df) > 0:
                    avg_switch = df["switch_time"].mean()

            summary_rows.append({
                "scaling_mode": scaling_mode,
                "model_id": cfg.get("model_id", ""),
                "model_name": cfg.get("model_name", ""),
                "workload": cfg.get("workload", ""),
                "steps": cfg.get("steps", ""),
                "sampler": cfg.get("sampler", ""),
                "avg_latency_sec": avg_latency,
                "avg_switch_time_ms": avg_switch,
            })

    if summary_rows:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n[BENCHMARK] Summary written to: {summary_file}")
    else:
        print(f"\n[BENCHMARK] No results to summarize")


def main():
    parser = argparse.ArgumentParser(description="DIFS Model Benchmark Runner")
    parser.add_argument(
        "--models", type=int, nargs="+", default=None,
        help="Model IDs to benchmark (default: all 7)"
    )
    parser.add_argument(
        "--modes", type=str, nargs="+", default=None,
        help="Scaling modes to test (default: all 3)"
    )
    parser.add_argument(
        "--workload", type=int, default=BENCHMARK_WORKLOAD,
        help=f"Number of prompts per query (default: {BENCHMARK_WORKLOAD})"
    )
    parser.add_argument(
        "--steps", type=int, default=BENCHMARK_STEPS,
        help=f"Inference steps (default: {BENCHMARK_STEPS})"
    )
    parser.add_argument(
        "--sampler", type=int, default=BENCHMARK_SAMPLER,
        help=f"Sampler ID (default: {BENCHMARK_SAMPLER})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--docker", action="store_true",
        help="Use Docker mode instead of direct Python"
    )
    parser.add_argument(
        "--skip-models", type=int, nargs="+", default=[],
        help="Model IDs to skip"
    )

    args = parser.parse_args()

    models_to_test = args.models or MODELS
    modes_to_test = args.modes or SCALING_MODES

    # Filter out skipped models
    models_to_test = [m for m in models_to_test if m not in args.skip_models]

    print()
    print("=" * 70)
    print("[BENCHMARK] DIFS Model Benchmark")
    print("=" * 70)
    print(f"  Models:    {models_to_test}")
    print(f"  Modes:     {modes_to_test}")
    print(f"  Workload:  {args.workload}")
    print(f"  Steps:     {args.steps}")
    print(f"  Sampler:   {args.sampler}")
    print(f"  Seed:      {args.seed}")
    print(f"  Launch:    {'Docker' if args.docker else 'Direct Python'}")
    print("=" * 70)

    pm = ProcessManager(run_direct=not args.docker)

    benchmark_results_dir = RESULTS_DIR / "model_benchmark"

    total_runs = len(modes_to_test) * len(models_to_test)
    current_run = 0

    for scaling_mode in modes_to_test:
        for model_id in models_to_test:
            current_run += 1
            model_name = get_model_short_name(model_id)

            print(f"\n{'#' * 70}")
            print(f"# Run {current_run}/{total_runs}: {scaling_mode} / {model_name}")
            print(f"{'#' * 70}")

            try:
                run_single_benchmark(
                    pm,
                    scaling_mode,
                    model_id,
                    args.workload,
                    args.steps,
                    args.sampler,
                    args.seed
                )
            except Exception as e:
                print(f"[BENCHMARK ERROR] {scaling_mode}/{model_name}: {e}")
                # Try to clean up
                try:
                    pm.stop_server()
                    pm.stop_worker()
                except Exception:
                    pass

    # Generate summary
    generate_summary_csv(benchmark_results_dir)

    print()
    print("=" * 70)
    print("[BENCHMARK] All runs completed")
    print(f"[BENCHMARK] Results in: {benchmark_results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
