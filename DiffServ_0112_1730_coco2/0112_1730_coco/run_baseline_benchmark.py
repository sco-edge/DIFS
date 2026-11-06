"""
run_baseline_benchmark.py

Baseline comparison benchmark for DIFS_OC_v4.
Runs 4 configurations against the same Poisson workload. Baselines 1-3 are
behavioral emulations implemented *inside* the DIFS codebase using the
hardcoded architecture profiles in src/baseline_profiles.py — they are NOT
the original open-source Clipper/INFaaS systems.  A run selects one profile
with DIFS_BASELINE_PROFILE so results are deterministic.  The comparison is
an ablation study isolating DIFS's scheduling, admission control, locality
awareness, and fair-share adaptation:

  1. clipper_large  — Clipper-style Large-Only emulation: FIFO, model 9 only,
                      steps 10/20/30/40, no model switching, no step adaptation
  2. clipper_small  — Clipper-style Small-Only emulation: FIFO, models 3+4,
                      model switching, steps 10/20/30/40, no step adaptation
  3. infaas         — INFaaS-style emulation: FIFO, cost-based model switching
                      (autoscaler on), steps 10/20/30/40, no step adaptation
  4. difs_v3        — full DIFS_OC_v3: EDF, locality-aware, fair-share step
                      adaptation 10-40, all 5 models

3-layer OOM prevention:
  Layer 1: GPU memory guard — checks free VRAM before each cell,
           waits for external processes if insufficient.
  Layer 2: OOM retry with backoff — retries failed cells up to 2 times.
  Layer 3: Robust cleanup — try/finally ensures workers/server are always
           killed, even on crash.

Usage:
    python run_baseline_benchmark.py --baseline clipper_large
    python run_baseline_benchmark.py --baseline all
    python run_baseline_benchmark.py --baseline all --levels low medium
    python run_baseline_benchmark.py --baseline clipper_small --workers 5
"""

import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — this file lives in v4 root; src/ is a sibling directory.
# ---------------------------------------------------------------------------
V4_ROOT = Path(__file__).resolve().parent
SRC_DIR = V4_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import PROJECT_ROOT, RESULTS_DIR
from model_registry import get_model_short_name
from experiments.experiment_config import (
    SLO_LOAD_LEVELS, SLO_WINDOW_SEC_BY_LEVEL,
    SLO_PROMPT_COUNT, SLO_WORKERS_BY_LEVEL, SEED,
    SCHEDULER_HOST, SERVER_PORT, CLIENT_SCRIPT,
)
from experiments.process_manager import ProcessManager
from slo import ACCURACY_STEP_MAP
from config import MAX_WORKERS
from baseline_profiles import BASELINES, RUNNABLE_BASELINES, PROFILE_ENV_VAR

# ---------------------------------------------------------------------------
# OOM prevention constants
# ---------------------------------------------------------------------------
# Minimum free GPU memory (MiB) required to safely start workers.
# Each SD3.5-Large worker uses ~3200 MiB; 5 workers need ~16000 MiB.
# We add 2000 MiB headroom for the server process + CUDA overhead.
MIN_FREE_GPU_MIB = 15000
GPU_WAIT_TIMEOUT_SEC = 600   # wait up to 10 min for GPU
GPU_POLL_INTERVAL_SEC = 30   # check every 30 s
MAX_CELL_RETRIES = 2         # retry failed cells up to 2 times
RETRY_BACKOFF_SEC = [60, 120]  # wait between retries

# ---------------------------------------------------------------------------
# Baseline configurations
# ---------------------------------------------------------------------------
# The profiles are hardcoded in src/baseline_profiles.py (single source of
# truth, imported as BASELINES above).  They are selected at run time with the
# single DIFS_BASELINE_PROFILE variable; the legacy per-toggle variables are
# ignored so manual runs are deterministic.

# Output paths
PLOT_DIR = RESULTS_DIR / "model_benchmark" / "plot_data_baseline"
OUT_CSV = PLOT_DIR / "baseline_comparison.csv"

# Accuracy levels include the extra "very_high" level for baselines
ACCURACY_LEVELS_BASELINE = ["low", "medium", "high", "very_high"]
ACCURACY_LEVELS_DIFS = ["low", "medium", "high", "very_high"]


# ---------------------------------------------------------------------------
# Layer 1: GPU memory guard
# ---------------------------------------------------------------------------

def get_gpu_memory_info():
    """Return (free_mib, total_mib) from nvidia-smi, or None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().splitlines()[0]
        parts = [x.strip() for x in line.split(",")]
        free_mib = int(parts[0])
        total_mib = int(parts[1])
        return free_mib, total_mib
    except Exception:
        return None


def get_external_gpu_processes():
    """Return list of (pid, mem_mib, name) for non-DIFS GPU processes."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        procs = []
        for line in result.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",", 2)]
            if len(parts) >= 3:
                pid = int(parts[0])
                mem = int(parts[1])
                name = parts[2]
                procs.append((pid, mem, name))
        return procs
    except Exception:
        return []


def wait_for_gpu(required_mib=None):
    """
    Wait until enough GPU memory is available.
    Returns True if memory is available, False if timed out.
    """
    if required_mib is None:
        required_mib = MIN_FREE_GPU_MIB

    info = get_gpu_memory_info()
    if info is None:
        print("[GPU GUARD] Cannot query GPU; proceeding without guard.")
        return True

    free, total = info
    if free >= required_mib:
        print(f"[GPU GUARD] OK: {free} MiB free / {total} MiB total "
              f"(need {required_mib} MiB)")
        return True

    print(f"[GPU GUARD] INSUFFICIENT: {free} MiB free / {total} MiB total "
          f"(need {required_mib} MiB)")

    ext_procs = get_external_gpu_processes()
    if ext_procs:
        print(f"[GPU GUARD] External GPU processes consuming memory:")
        for pid, mem, name in ext_procs:
            print(f"  PID {pid}: {mem} MiB — {name}")

    start = time.time()
    while time.time() - start < GPU_WAIT_TIMEOUT_SEC:
        elapsed = int(time.time() - start)
        print(f"[GPU GUARD] Waiting... ({elapsed}s / {GPU_WAIT_TIMEOUT_SEC}s)")

        time.sleep(GPU_POLL_INTERVAL_SEC)

        info = get_gpu_memory_info()
        if info is None:
            print("[GPU GUARD] Cannot query GPU; proceeding anyway.")
            return True

        free, total = info
        if free >= required_mib:
            print(f"[GPU GUARD] NOW OK: {free} MiB free (waited {elapsed}s)")
            return True

    print(f"[GPU GUARD] TIMEOUT after {GPU_WAIT_TIMEOUT_SEC}s. "
          f"Still only {free} MiB free. Will attempt cell anyway.")
    return False


# ---------------------------------------------------------------------------
# Layer 3: Robust cleanup
# ---------------------------------------------------------------------------

def force_cleanup(pm, ports):
    """Nuclear cleanup: kill all known processes, then sweep for orphans."""
    print("[CLEANUP] Starting robust cleanup...")

    # Try graceful stop via ProcessManager
    try:
        pm.stop_server()
    except Exception:
        pass

    for p in ports:
        try:
            pm.stop_worker_at_port(p)
        except Exception:
            pass

    # Also clear any single-worker tracking
    try:
        pm.stop_worker()
    except Exception:
        pass

    # Nuclear option: kill any remaining container_diffusion processes
    # that were spawned for this run's ports
    try:
        result = subprocess.run(
            ["pgrep", "-f", "container_diffusio[n]"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
            time.sleep(3)
            # Force kill survivors
            result2 = subprocess.run(
                ["pgrep", "-f", "container_diffusio[n]"],
                capture_output=True, text=True, timeout=5,
            )
            if result2.stdout.strip():
                for pid in result2.stdout.strip().split("\n"):
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
    except Exception:
        pass

    # Kill scheduler if still lingering on SERVER_PORT
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split("\n"):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass

    # Wait for ports to be released
    time.sleep(3)
    print("[CLEANUP] Done.")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def clear_metrics():
    metrics_dir = PROJECT_ROOT / "metrics"
    if metrics_dir.exists():
        for f in metrics_dir.glob("*.csv"):
            f.unlink()


def compute_metrics(step_map=None, emitted_by_accuracy=None):
    """Read slo_metrics.csv, return per-step breakdowns with accuracy SLO.

    Accuracy SLO is defined as: steps_used >= step_map[accuracy_level].
    This correctly penalises systems that downgrade step count under load.

    Dropped/timed-out requests are counted as SLO violations.  When
    *emitted_by_accuracy* is provided, we compute how many requests were
    emitted per step count (via step_map) but never appeared in
    slo_metrics.csv, and treat them as both latency and accuracy misses.

    Args:
        step_map: dict mapping accuracy_level -> required steps.
                  Defaults to ACCURACY_STEP_MAP from slo.py.
        emitted_by_accuracy: dict mapping accuracy_level -> count of
                  requests emitted by the benchmark loop.

    Returns:
        step_data: dict mapping steps -> {lat_total, lat_met, acc_met,
                   emitted, dropped}
        overall: (lat_total, lat_met, acc_met, dropped)
    """
    if step_map is None:
        step_map = ACCURACY_STEP_MAP

    slo_path = PROJECT_ROOT / "metrics" / "slo_metrics.csv"

    step_data = {}
    overall_lat_total = 0
    overall_lat_met = 0
    overall_acc_met = 0

    print(f"[METRICS] Reading SLO from {slo_path} (exists={slo_path.exists()})")
    print(f"[METRICS] step_map={step_map}")

    if slo_path.exists():
        with open(slo_path) as f:
            for row in csv.DictReader(f):
                steps = int(row["steps"])
                accuracy_level = row.get("accuracy_level", "").strip().lower()
                if steps not in step_data:
                    step_data[steps] = {"lat_total": 0, "lat_met": 0, "acc_met": 0,
                                         "emitted": 0, "dropped": 0}
                step_data[steps]["lat_total"] += 1
                overall_lat_total += 1
                if row.get("met", "").strip().lower() == "true":
                    step_data[steps]["lat_met"] += 1
                    overall_lat_met += 1
                # Accuracy SLO: did the system serve at >= the required steps?
                required_steps = step_map.get(accuracy_level, steps)
                if steps >= required_steps:
                    step_data[steps]["acc_met"] += 1
                    overall_acc_met += 1

    # --- Drop accounting: emitted - completed = dropped (timeout/violation) ---
    overall_emitted = 0
    overall_dropped = 0

    if emitted_by_accuracy:
        # Map accuracy levels to step counts to get emitted per step
        emitted_by_step = {}
        for level, count in emitted_by_accuracy.items():
            steps_req = step_map.get(level, 20)
            emitted_by_step[steps_req] = emitted_by_step.get(steps_req, 0) + count

        for steps_req, emit_count in emitted_by_step.items():
            completed_at_step = step_data.get(steps_req, {}).get("lat_total", 0)
            dropped_at_step = max(0, emit_count - completed_at_step)
            if steps_req not in step_data:
                step_data[steps_req] = {"lat_total": 0, "lat_met": 0, "acc_met": 0,
                                         "emitted": 0, "dropped": 0}
            step_data[steps_req]["emitted"] = emit_count
            step_data[steps_req]["dropped"] = dropped_at_step
            overall_emitted += emit_count
            overall_dropped += dropped_at_step
    else:
        # No emitted data: carry forward existing counts
        for sd in step_data.values():
            sd["emitted"] = sd["lat_total"]
            sd["dropped"] = 0
            overall_emitted += sd["lat_total"]

    total = overall_lat_total + overall_dropped
    print(f"[METRICS] step_data keys: {sorted(step_data.keys())}, "
          f"completed={overall_lat_total}, lat_met={overall_lat_met}, "
          f"acc_met={overall_acc_met}, dropped={overall_dropped}, "
          f"total_emitted={overall_emitted}")
    return step_data, (overall_lat_total, overall_lat_met, overall_acc_met), overall_dropped


# ---------------------------------------------------------------------------
# Core cell runner
# ---------------------------------------------------------------------------

def run_level(pm, baseline_name, cfg, load_level, rate, window_sec, seed,
              max_workers=MAX_WORKERS):
    """One (baseline, load_level) run. Returns the results row dict."""
    ports = []

    print()
    print("=" * 70)
    print(f"[BASELINE] {baseline_name} | Load={load_level} "
          f"(rate={rate:.4f} req/s, window={window_sec}s)")
    print("=" * 70)

    clear_metrics()

    # Select the hardcoded architecture profile for this baseline.  Worker and
    # server subprocesses inherit this single variable; the actual toggle
    # values are fixed in src/baseline_profiles.py.
    os.environ[PROFILE_ENV_VAR] = baseline_name

    # Import config AFTER setting the profile so it picks up the hardcoded values
    import importlib
    import config as cfg_mod
    importlib.reload(cfg_mod)

    model_ids = cfg["model_ids"]
    n_workers = min(SLO_WORKERS_BY_LEVEL[load_level], max_workers)

    # Estimate required GPU memory: ~3200 MiB per worker + 2000 MiB headroom
    estimated_gpu_needed = n_workers * 3200 + 2000
    wait_for_gpu(estimated_gpu_needed)

    model_map = {}  # port -> model_id, passed to server for correct SLO
    pm.worker_port_counter = 50060

    try:
        for i in range(n_workers):
            model_id = model_ids[i % len(model_ids)]
            port = pm.worker_port_counter
            pm.worker_port_counter += 1
            try:
                pm.start_worker_at_port(model_id=model_id, port=port)
                ports.append(port)
                model_map[port] = model_id
            except Exception as e:
                print(f"[BASELINE WARNING] Could not start worker on port {port}: {e}")
                break

        if not ports:
            raise RuntimeError("No workers could be started")

        pinned = [model_ids[i % len(model_ids)] for i in range(len(ports))]
        print(f"[BASELINE] Worker pool: ports {ports}, pinned models {pinned}")
        pm.start_server_multi(extra_worker_ports=ports[1:],
                              worker_model_map=model_map)

        # All baselines including DIFS_v3 receive the same accuracy levels
        # for fair workload comparison
        acc_levels = ACCURACY_LEVELS_BASELINE

        rng = random.Random(seed)
        print(f"[BASELINE] Emitting up to {SLO_PROMPT_COUNT} requests "
              f"(rate={rate:.4f} req/s, window={window_sec}s, levels={acc_levels})")

        t_end = time.time() + window_sec
        next_arrival = time.time()
        emitted = 0
        emitted_by_accuracy = {level: 0 for level in acc_levels}
        while emitted < SLO_PROMPT_COUNT and time.time() < t_end:
            accuracy = rng.choice(acc_levels)
            emitted_by_accuracy[accuracy] += 1
            cmd = [
                sys.executable, str(CLIENT_SCRIPT),
                "-prompt_count", "1",
                "-host", SCHEDULER_HOST,
                "-port", str(SERVER_PORT),
                "-accuracy", accuracy,
                "-seed", str(seed),
            ]
            try:
                subprocess.run(cmd, cwd=str(V4_ROOT),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=300)
            except subprocess.TimeoutExpired:
                print(f"[BASELINE WARNING] Client request timed out after 300s, skipping")
            emitted += 1
            next_arrival += rng.expovariate(rate)
            time.sleep(max(0.0, next_arrival - time.time()))

        time.sleep(5)

        step_map = cfg.get("step_map", ACCURACY_STEP_MAP)
        step_data, overall, dropped = compute_metrics(
            step_map=step_map, emitted_by_accuracy=emitted_by_accuracy)
        lat_total, lat_met, acc_met = overall
        completed = lat_total
        total = lat_total + dropped
        lat_att = 100.0 * lat_met / total if total else 0.0
        acc_att = 100.0 * acc_met / total if total else 0.0
        throughput = completed / window_sec
        print(f"[BASELINE] completed={completed}/{emitted} dropped={dropped} "
              f"latencySLO={lat_att:.1f}% accuracySLO={acc_att:.1f}% "
              f"throughput={throughput:.4f} req/s")

        if not step_data:
            print(f"[BASELINE WARNING] compute_metrics returned empty step_data! "
                  f"slo_path={PROJECT_ROOT / 'metrics' / 'slo_metrics.csv'} "
                  f"exists={(PROJECT_ROOT / 'metrics' / 'slo_metrics.csv').exists()}")

        # Build per-step rows
        rows = []
        for steps in sorted(step_data.keys()):
            sd = step_data[steps]
            step_total = sd["lat_total"] + sd.get("dropped", 0)
            s_lat_att = 100.0 * sd["lat_met"] / step_total if step_total else 0.0
            s_acc_att = 100.0 * sd["acc_met"] / step_total if step_total else 0.0
            s_thru = sd["lat_total"] / window_sec
            rows.append({
                "baseline": baseline_name,
                "load_level": load_level,
                "steps": steps,
                "requests_submitted": sd.get("emitted", sd["lat_total"]),
                "requests_completed": sd["lat_total"],
                "requests_dropped": sd.get("dropped", 0),
                "latency_slo_attainment_pct": round(s_lat_att, 2),
                "accuracy_slo_attainment_pct": round(s_acc_att, 2),
                "throughput_req_per_s": round(s_thru, 4),
            })

        return rows

    finally:
        # Layer 3: ALWAYS clean up, even on crash/OOM/KeyboardInterrupt
        force_cleanup(pm, ports)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DIFS Baseline Comparison Benchmark")
    parser.add_argument("--baseline", type=str, nargs="+", default=["all"],
                        help="Baseline(s): clipper_large, clipper_small, infaas, difs_v3, all")
    parser.add_argument("--levels", type=str, nargs="+", default=None,
                        help="Load levels (default: low, medium, high)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    levels = args.levels or list(SLO_LOAD_LEVELS)

    if "all" in args.baseline:
        baselines = list(RUNNABLE_BASELINES)
    else:
        baselines = [b for b in args.baseline if b in RUNNABLE_BASELINES]
        if not baselines:
            print(f"[ERROR] No valid baselines. Choose from: {list(RUNNABLE_BASELINES)}")
            sys.exit(1)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing results (now per-step: list of row dicts per cell)
    existing = {}
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            for row in csv.DictReader(f):
                key = (row["baseline"], row["load_level"])
                if key not in existing:
                    existing[key] = []
                existing[key].append(row)

    # Pre-flight: check GPU once at the start
    print("\n[MAIN] Pre-flight GPU check...")
    info = get_gpu_memory_info()
    if info:
        free, total = info
        print(f"[MAIN] GPU: {free} MiB free / {total} MiB total")
        if free < MIN_FREE_GPU_MIB:
            print(f"[MAIN] GPU memory below threshold ({MIN_FREE_GPU_MIB} MiB). "
                  f"Waiting for external processes to release memory...")
            if not wait_for_gpu(MIN_FREE_GPU_MIB):
                print("[MAIN] WARNING: Could not get sufficient GPU memory. "
                      "Cells will be attempted with retry logic.")

    all_rows = []
    failed_cells = []

    def _save_csv(merged_dict):
        """Write current results to CSV. Called after every completed cell."""
        if not merged_dict:
            print("[SAVE] skipped: merged_dict is empty")
            return
        # Flatten all per-step rows into a single list
        all_csv_rows = []
        for key, rows in merged_dict.items():
            if isinstance(rows, list):
                all_csv_rows.extend(rows)
            else:
                all_csv_rows.append(rows)
        if not all_csv_rows:
            print(f"[SAVE] skipped: no CSV rows from {len(merged_dict)} cells")
            for key, rows in merged_dict.items():
                print(f"  {key}: type={type(rows).__name__}, "
                      f"len={len(rows) if isinstance(rows, (list, dict)) else 'N/A'}")
            return
        keys = all_csv_rows[0].keys()
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_csv_rows:
                writer.writerow(row)
        print(f"[SAVE] Wrote {len(all_csv_rows)} rows from {len(merged_dict)} cells to {OUT_CSV}")

    pm = ProcessManager(run_direct=True)

    # Global signal handler: Ctrl+C triggers emergency cleanup
    _pm_ref = pm  # capture for signal handler

    def _sigint_handler(sig, frame):
        print("\n[MAIN] SIGINT received — running emergency cleanup...")
        _pm_ref.emergency_stop_all()
        _save_csv(dict(existing))
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    for bname in baselines:
        cfg = BASELINES[bname]
        print(f"\n{'#' * 70}")
        print(f"# BASELINE: {bname}")
        print(f"# {cfg['description']}")
        print(f"{'#' * 70}")

        for level in levels:
            # Skip if already completed in a previous run
            if (bname, level) in existing:
                print(f"[SKIP] {bname}/{level} already completed in previous run.")
                all_rows.extend(existing[(bname, level)])
                continue

            success = False
            for attempt in range(1, MAX_CELL_RETRIES + 1):
                try:
                    window = SLO_WINDOW_SEC_BY_LEVEL[level]
                    rows = run_level(pm, bname, cfg, level, SLO_LOAD_LEVELS[level],
                                    window, args.seed, max_workers=args.workers)
                    all_rows.extend(rows)
                    existing[(bname, level)] = rows
                    _save_csv(existing)
                    print(f"[SAVE] Wrote {len(existing)} cells to {OUT_CSV}")
                    success = True
                    break
                except Exception as e:
                    error_str = str(e)
                    is_oom = ("out of memory" in error_str.lower()
                              or "cuda" in error_str.lower())
                    print(f"[BASELINE ERROR] {bname}/{level} "
                          f"(attempt {attempt}/{MAX_CELL_RETRIES}): {e}")
                    import traceback; traceback.print_exc()

                    if attempt < MAX_CELL_RETRIES:
                        wait_time = RETRY_BACKOFF_SEC[min(attempt - 1,
                                                          len(RETRY_BACKOFF_SEC) - 1)]
                        if is_oom:
                            print(f"[RETRY] OOM detected. Waiting {wait_time}s "
                                  f"for GPU memory to free up...")
                            wait_for_gpu(MIN_FREE_GPU_MIB)
                        else:
                            print(f"[RETRY] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"[BASELINE FAILED] {bname}/{level}: "
                              f"gave up after {MAX_CELL_RETRIES} attempts.")
                        failed_cells.append((bname, level, error_str))

    # Summary
    total_rows = sum(len(v) if isinstance(v, list) else 1 for v in existing.values())
    print(f"\n{'=' * 70}")
    print(f"[SUMMARY] Completed: {len(existing)} cells ({total_rows} per-step rows) in {OUT_CSV}")
    if failed_cells:
        print(f"[SUMMARY] FAILED ({len(failed_cells)}):")
        for bname, level, err in failed_cells:
            print(f"  - {bname}/{level}: {err[:80]}")
    else:
        print("[SUMMARY] All cells completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
