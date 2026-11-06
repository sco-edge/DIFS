"""
run_slo_benchmark.py

Live SLO benchmark on the real GPU stack. For each (scaling_mode,
load_level) pair:

  1. Start the server + a worker pool sized by SLO_WORKERS_BY_LEVEL
     (capped by MAX_WORKERS for a single node). Each worker boots with
     a pinned model (sticky pinning), and the scheduler spreads load
     across the pool via worker_pool.get_next_worker().
  2. Emit a Poisson arrival stream of requests at the load level's
     arrival rate over a per-level window (window = 300 / rate) drawn
     from a pool of 300 COCO captions, drawing an accuracy level
     (low/medium/high) per request and passing it to the client via
     -accuracy (the scheduler derives the latency SLO = 2.5 x exec
     and records per-request SLO outcomes).
  3. Read metrics/slo_metrics.csv + worker_metrics.csv and compute:
       - latency SLO attainment (%)
       - accuracy SLO attainment (%)  (output passed the worker quality
         gate: valid, non-degenerate image delivered at the requested step
         level, counted from metrics/accuracy_metrics.csv written by each
         worker; requests without a quality_ok=True row count as misses)
       - throughput (req/s)
  4. Write the row into
     results/model_benchmark/plot_data_5model/slo_benchmark_results.csv

The high load level requests 20 workers (> MAX_WORKERS = 5), so a
single-node live run cannot reach the target pool; run it across
multiple nodes/GPUs (--workers) or accept the single-node ceiling.

Values for levels not covered by a live run are preserved from the
simulated dataset (generate_slo_benchmark_results.py).

Usage:
    python -m src.experiments.run_slo_benchmark
    python -m src.experiments.run_slo_benchmark --modes vertical hybrid
    python -m src.experiments.run_slo_benchmark --levels low medium
    python -m src.experiments.run_slo_benchmark --rerun
    python -m src.experiments.run_slo_benchmark --rerun --direct

Experimental rerun mode:
    --rerun   Re-runs only the (scaling_mode, load_level) pairs that do
              NOT already have a live measurement in
              slo_benchmark_results.csv (simulated rows are treated as
              missing). By default reruns keep Docker in the loop
              (worker containers, image difs-diffusion-worker); pass
              --direct to force direct Python-subprocess workers.
"""

import time
import csv
import sys
import random
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from project_paths import PROJECT_ROOT, RESULTS_DIR
from model_registry import get_model_short_name
from experiments.experiment_config import (
    SCALING_MODES, SLO_LOAD_LEVELS, SLO_WINDOW_SEC_BY_LEVEL,
    SLO_PROMPT_COUNT, SLO_WORKERS_BY_LEVEL, SEED,
    SCHEDULER_HOST, SCHEDULER_PORT, WORKER_PORT, SERVER_PORT,
    SERVER_COMMAND, CLIENT_SCRIPT, WORKER_SCRIPT, RUN_DIRECT,
    SLO_CLIENT_TIMEOUT_MULT,
)
from experiments.process_manager import ProcessManager
from config import MAX_WORKERS

PLOT_DIR = RESULTS_DIR / "model_benchmark" / "plot_data_5model"
OUT_CSV = PLOT_DIR / "slo_benchmark_results.csv"
ACCURACY_LEVELS = ["low", "medium", "high"]
SLO_METRICS_CSV = Path("metrics") / "slo_metrics.csv"
# Models profiled in slo_exec_profiling.csv; workers boot pinned to these.
PROFILED_MODEL_IDS = [3, 4, 5, 8, 9]


def clear_metrics():
    metrics_dir = Path("metrics")
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.csv"):
            f.unlink()


def compute_metrics(window_sec):
    """
    Read scheduler SLO metrics written by metrics.record_slo() plus the
    worker-side quality gate in metrics/accuracy_metrics.csv.
    Returns (latency_total, latency_met, accuracy_ok).
    """
    if not SLO_METRICS_CSV.exists():
        return 0, 0, 0
    lat_met = 0
    lat_total = 0
    with open(SLO_METRICS_CSV) as f:
        for row in csv.DictReader(f):
            lat_total += 1
            if row.get("met", "").strip().lower() == "true":
                lat_met += 1
    acc_ok = 0
    acc_path = Path("metrics") / "accuracy_metrics.csv"
    if acc_path.exists():
        with open(acc_path) as f:
            for row in csv.DictReader(f):
                if row.get("quality_ok", "").strip().lower() == "true":
                    acc_ok += 1
    return lat_total, lat_met, acc_ok


def run_level_benchmark(pm, scaling_mode, load_level, rate, window_sec, seed,
                        max_workers=MAX_WORKERS):
    """One (mode, level) live run. Returns the results row dict."""
    print()
    print("=" * 70)
    print(f"[SLO-BENCH] Mode={scaling_mode} | Load={load_level} "
          f"(rate={rate} req/s, window={window_sec}s)")
    print("=" * 70)

    clear_metrics()
    import config as runtime_config
    runtime_config.SCALING_MODE = scaling_mode

    # Multi-worker pool: request SLO_WORKERS_BY_LEVEL workers, capped by
    # the single-node MAX_WORKERS ceiling. Each worker boots with a pinned
    # model (sticky routing); the scheduler spreads load via
    # worker_pool.get_next_worker().
    requested = SLO_WORKERS_BY_LEVEL[load_level]
    n_workers = min(requested, max_workers)
    if n_workers < requested:
        print(f"[SLO-BENCH] WARNING: {load_level} requests {requested} workers "
              f"but only {n_workers} available (MAX_WORKERS={max_workers}); "
              f"scale out across nodes to reach the target pool.")

    ports = []
    model_map = {}
    pm.worker_port_counter = WORKER_PORT
    for i in range(n_workers):
        model_id = PROFILED_MODEL_IDS[i % len(PROFILED_MODEL_IDS)]
        port = pm.worker_port_counter
        pm.worker_port_counter += 1
        try:
            pm.start_worker_at_port(model_id=model_id, port=port)
            ports.append(port)
            model_map[port] = model_id
        except Exception as e:
            print(f"[SLO-BENCH WARNING] Could not start worker on port "
                  f"{port}: {e}")
            break
    if not ports:
        raise RuntimeError("No workers could be started for the live SLO run")
    print(f"[SLO-BENCH] Worker pool ready on ports {ports} "
          f"(pinned models: "
          f"{[PROFILED_MODEL_IDS[i % len(PROFILED_MODEL_IDS)] for i in range(len(ports))]})")

    pm.start_server_multi(extra_worker_ports=ports[1:],
                          worker_model_map=model_map)

    rng = random.Random(seed)
    print(f"[SLO-BENCH] Emitting up to {SLO_PROMPT_COUNT} requests "
          f"(Poisson, rate={rate:.4f} req/s, window={window_sec}s) ...")

    t_end = time.time() + window_sec
    next_arrival = time.time()
    emitted = 0
    while emitted < SLO_PROMPT_COUNT and time.time() < t_end:
        accuracy = rng.choice(ACCURACY_LEVELS)
        cmd = [
            sys.executable, CLIENT_SCRIPT,
            "-prompt_count", "1",
            "-host", SCHEDULER_HOST,
            "-port", str(SERVER_PORT),
            "-accuracy", accuracy,
            "-seed", str(seed),
        ]
        subprocess.run(cmd, cwd=PROJECT_ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        emitted += 1
        next_arrival += rng.expovariate(rate)
        time.sleep(max(0.0, next_arrival - time.time()))

    time.sleep(5)  # allow SLO metrics to flush

    lat_total, lat_met, acc_ok = compute_metrics(window_sec)
    completed = lat_total
    pm.stop_server()
    for p in ports:
        pm.stop_worker_at_port(p)

    lat_att = 100.0 * lat_met / lat_total if lat_total else 0.0
    acc_att = 100.0 * acc_ok / emitted if emitted else 0.0
    throughput = completed / window_sec
    print(f"[SLO-BENCH] completed={completed}/{emitted} "
          f"latencySLO={lat_att:.1f}% accuracySLO={acc_att:.1f}% "
          f"throughput={throughput:.4f} req/s")

    return {
        "load_level": load_level,
        "scaling_mode": scaling_mode,
        "requests_submitted": emitted,
        "requests_completed": completed,
        "latency_slo_attainment_pct": round(lat_att, 2),
        "accuracy_slo_attainment_pct": round(acc_att, 2),
        "throughput_req_per_s": round(throughput, 4),
        "source": "live",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DIFS Live SLO Benchmark")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="Scaling modes (default: all 3)")
    parser.add_argument("--levels", type=str, nargs="+", default=None,
                        help=f"Load levels (default: {list(SLO_LOAD_LEVELS)})")
    parser.add_argument("--window", type=int, default=None,
                        help="Override arrival window (s) for all levels "
                             "(default: per-level 300/rate)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Max workers for the pool per level "
                             f"(default: MAX_WORKERS={MAX_WORKERS}; target pool "
                             f"sizes are SLO_WORKERS_BY_LEVEL)")
    parser.add_argument("--docker", action="store_true",
                        help="Use Docker instead of direct Python")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run only (mode, level) pairs without a live "
                             "result in the results CSV; defaults to Docker "
                             "launch so reruns do not bypass the container "
                             "stack (override with --direct)")
    parser.add_argument("--direct", action="store_true",
                        help="Force direct Python-subprocess workers "
                             "(default; used to override --rerun's Docker "
                             "default)")
    args = parser.parse_args()

    modes = args.modes or SCALING_MODES
    levels = args.levels or list(SLO_LOAD_LEVELS)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Read any existing results so legacy rows can be backfilled and, in
    # rerun mode, pairs that already have a live measurement can be skipped.
    existing = {}
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                row.setdefault("source", "simulated")
                existing[(row["scaling_mode"], row["load_level"])] = row

    live_rows = []

    # Launch mode precedence: --direct > --docker > rerun defaults to
    # Docker (so reruns do not bypass the container stack) > direct.
    if args.direct:
        run_direct = True
    elif args.docker or args.rerun:
        run_direct = False
    else:
        run_direct = True
    pm = ProcessManager(run_direct=run_direct)

    for mode in modes:
        for level in levels:
            if args.rerun and (mode, level) in existing and \
                    existing[(mode, level)].get("source") == "live":
                print(f"[SLO-BENCH] Skip {mode}/{level}: live result already "
                      f"present")
                continue
            try:
                window = args.window or SLO_WINDOW_SEC_BY_LEVEL[level]
                row = run_level_benchmark(
                    pm, mode, level, SLO_LOAD_LEVELS[level], window, args.seed,
                    max_workers=args.workers)
                live_rows.append(row)
            except Exception as e:
                print(f"[SLO-BENCH ERROR] {mode}/{level}: {e}")
                try:
                    pm.stop_server()
                    pm.stop_worker()
                except Exception:
                    pass

    # Merge with any existing (simulated/live) rows, keeping live runs on top.
    merged = dict(existing)
    for row in live_rows:
        merged[(row["scaling_mode"], row["load_level"])] = row

    if merged:
        keys = list(merged.values())[0].keys()
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in merged.values():
                writer.writerow(row)
        print(f"\n[SLO-BENCH] Wrote {len(merged)} rows to {OUT_CSV}")
    else:
        print(f"\n[SLO-BENCH] Nothing to run; results CSV unchanged at {OUT_CSV}")


if __name__ == "__main__":
    main()
