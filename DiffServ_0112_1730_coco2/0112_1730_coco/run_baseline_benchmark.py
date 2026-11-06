"""
run_baseline_benchmark.py

Baseline comparison benchmark for DIFS_OC_v6.
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

How to run
    Recommended (handles env, prompts, models, cleanup, and resuming):
        ./run_all_experiments.sh --experiments --skip-switching

    Directly, from this project directory:
        PYTHONPATH=src python run_baseline_benchmark.py --baseline all
        PYTHONPATH=src python run_baseline_benchmark.py --baseline clipper_large
        PYTHONPATH=src python run_baseline_benchmark.py --baseline all --levels low medium
        PYTHONPATH=src python run_baseline_benchmark.py --baseline clipper_small --workers 5

    Flags:
        --baseline {clipper_large,clipper_small,infaas,difs_v3,all}  (default: all)
        --levels "low medium high saturation stress"                 (default: all five)
        --rates "6 7 8 9 10"   custom arrival rates (req/min, sweep mode)
        --repeats N            seeds per rate in sweep mode            (default: 1)
        --workers N                                                 (default: 5)
        --seed N                                                    (default: 42)

    Sweep mode writes <plot_data_baseline>/slo_violation_runs.csv instead of
    baseline_comparison.csv/objective_metrics.csv, so a 5% sweep never changes
    the report tables.

    To force a clean rerun, remove
    results/model_benchmark/plot_data_baseline/baseline_comparison.csv, or use
    the driver's --force flag:
        ./run_all_experiments.sh --experiments --force

    Prerequisites:
        * real_stats.npz in this directory (not git-tracked).
        * annotations/diffusiondb_prompts.json (build with --setup).
        * DIFS_MODEL_DIR pointing at the five model snapshots.
        * A CUDA GPU with >= 15 GiB free VRAM.
    The run is resumable: completed rows in
    results/model_benchmark/plot_data_baseline/baseline_comparison.csv are
    reused unless --force is passed.
"""

import csv
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — this file lives in v4 root; src/ is a sibling directory.
# ---------------------------------------------------------------------------
V4_ROOT = Path(__file__).resolve().parent
SRC_DIR = V4_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from project_paths import PROJECT_ROOT, RESULTS_DIR
from model_registry import get_model_short_name
from baseline_profiles import BASELINES, RUNNABLE_BASELINES, PROFILE_ENV_VAR
from experiments.experiment_config import (
    SLO_LOAD_LEVELS, SLO_WINDOW_SEC_BY_LEVEL,
    SLO_PROMPT_COUNT, SLO_WORKERS_BY_LEVEL, SEED,
    SCHEDULER_HOST, SERVER_PORT, CLIENT_SCRIPT,
)
from experiments.process_manager import ProcessManager
from slo import ACCURACY_STEP_MAP
from config import MAX_WORKERS

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
LIVE_RESULTS_DIR_ENV = os.environ.get("DIFS_LIVE_RESULTS_DIR")
if LIVE_RESULTS_DIR_ENV:
    _live_base = Path(LIVE_RESULTS_DIR_ENV)
    if _live_base.name == "plot_data_baseline":
        PLOT_DIR = _live_base
    else:
        PLOT_DIR = _live_base / "plot_data_baseline"
else:
    PLOT_DIR = RESULTS_DIR / "model_benchmark" / "plot_data_baseline"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = PLOT_DIR / "baseline_comparison.csv"
OBJECTIVE_CSV = PLOT_DIR / "objective_metrics.csv"
OBJECTIVE_FIELDS = [
    "baseline", "load_level", "prompt_count", "optimization_enabled",
    "requests_submitted", "slo_violation_rate", "cold_start_latency",
    "hot_start_latency", "error_to_normal_ratio", "timestamp",
]
# SLO 5% load-sweep output (separate from the report's objective table so that
# repeated sweep runs never disturb baseline_comparison/objective_metrics).
SWEEP_CSV = PLOT_DIR / "slo_violation_runs.csv"
SWEEP_FIELDS = [
    "baseline", "load_level", "rate_req_per_min", "seed", "prompt_count",
    "requests_submitted", "requests_completed", "requests_dropped",
    "latency_slo_attainment_pct", "accuracy_slo_attainment_pct",
    "throughput_req_per_s", "slo_violation_rate", "timestamp",
]


def rate_label(rate_req_per_min):
    """Stable label for a custom arrival rate (e.g. 6.5 -> rate_6.5)."""
    return f"rate_{float(rate_req_per_min):g}"


def load_sweep_runs():
    """Return existing sweep rows keyed by (baseline, load_level, seed)."""
    rows = {}
    if not SWEEP_CSV.exists():
        return rows
    try:
        with open(SWEEP_CSV) as f:
            for row in csv.DictReader(f):
                key = (row.get("baseline"), row.get("load_level"),
                       str(row.get("seed", "")))
                rows[key] = row
    except Exception:
        return {}
    return rows


def save_sweep_run(baseline, load_level, rate_req_per_min, seed, rows,
                   metrics, prompt_count, window_sec):
    """Persist one sweep cell (one baseline/rate/seed) for the SLO plot."""
    submitted = completed = dropped = 0
    lat_weighted = acc_weighted = 0.0
    for row in rows:
        n = int(row.get("requests_submitted", 0) or 0)
        submitted += n
        completed += int(row.get("requests_completed", 0) or 0)
        dropped += int(row.get("requests_dropped", 0) or 0)
        lat_weighted += n * float(row.get("latency_slo_attainment_pct", 0.0) or 0.0)
        acc_weighted += n * float(row.get("accuracy_slo_attainment_pct", 0.0) or 0.0)

    violation = metrics.get("slo_violation_rate")
    if violation is None and submitted:
        violation = float(dropped) / submitted

    entry = {
        "baseline": baseline,
        "load_level": load_level,
        "rate_req_per_min": rate_req_per_min,
        "seed": seed,
        "prompt_count": prompt_count,
        "requests_submitted": submitted,
        "requests_completed": completed,
        "requests_dropped": dropped,
        "latency_slo_attainment_pct": round(lat_weighted / submitted, 2)
        if submitted else 0.0,
        "accuracy_slo_attainment_pct": round(acc_weighted / submitted, 2)
        if submitted else 0.0,
        "throughput_req_per_s": round(completed / window_sec, 4)
        if window_sec else 0.0,
        "slo_violation_rate": round(float(violation), 6)
        if violation is not None else "",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    existing = load_sweep_runs()
    existing[(baseline, load_level, str(seed))] = entry
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDS)
        writer.writeheader()
        for key in sorted(existing, key=lambda k: (str(k[0]), str(k[1]), int(k[2]) if str(k[2]).lstrip("-").isdigit() else 0)):
            writer.writerow({field: existing[key].get(field, "")
                             for field in SWEEP_FIELDS})
    print(f"[SAVE] Wrote SLO sweep row {baseline}/{load_level}/seed={seed} "
          f"to {SWEEP_CSV}")

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


def _snapshot_per_request_metrics(baseline, load_level, seed, mode, out_dir):
    """Copy per-request metric CSVs for this cell with provenance tags.
    Returns the consolidated rows written (may be empty)."""
    metrics_dir = PROJECT_ROOT / "metrics"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_tag = f"{baseline}__{load_level}__seed{seed}__{mode}"
    # snapshot raw files
    for name in ("client_metrics.csv", "slo_metrics.csv", "accuracy_metrics.csv", "worker_metrics.csv"):
        src = metrics_dir / name
        if src.exists():
            dst = out_dir / f"{cell_tag}__{name}"
            try:
                import shutil
                shutil.copy2(src, dst)
            except Exception:
                pass
    # build consolidated per-request table (best-effort join)
    client = metrics_dir / "client_metrics.csv"
    slo = metrics_dir / "slo_metrics.csv"
    if not (client.exists() and slo.exists()):
        return []
    slo_rows = list(csv.DictReader(slo.open()))
    # align by order when request ids don't join cleanly
    rows = []
    for i, cr in enumerate(csv.DictReader(client.open()), 1):
        sr = slo_rows[i - 1] if i - 1 < len(slo_rows) else {}
        lat_ms = float(cr.get("latency", 0) or 0) * 1000.0
        slo_ms = sr.get("latency_slo_ms", "")
        met = str(sr.get("met", "")).strip().lower()
        lat_status = "Met" if met == "true" else ("Violated" if met == "false" else "N/A")
        rows.append({
            "run_mode": mode,
            "baseline": baseline,
            "load_level_or_rate": load_level,
            "seed": seed,
            "request_index": i,
            "image_id": cr.get("image_id", ""),
            "end_to_end_latency_ms": f"{lat_ms:.3f}",
            "latency_slo_ms": slo_ms,
            "latency_slo_status": lat_status,
            "accuracy_level": sr.get("accuracy_level", ""),
            "step_size": sr.get("steps", ""),
            "model_id": sr.get("model_id", ""),
            "provenance": f"metrics snapshot after cell {cell_tag}",
        })
    return rows


def append_per_request_export(rows, out_csv):
    if not rows:
        return
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


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
# v7 objective metrics (used by the real-time tracker)
# ---------------------------------------------------------------------------

def _median_csv_column(path, column, predicate=None, scale=1.0):
    """Return the median of a numeric CSV column, or None."""
    if not path.exists():
        return None
    values = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                if predicate is not None and not predicate(row):
                    continue
                try:
                    values.append(float(row[column]) * scale)
                except (KeyError, TypeError, ValueError):
                    continue
    except Exception:
        return None
    return statistics.median(values) if values else None


def compute_objective_metrics(rows, metrics_dir, results_dir=None):
    """Derive v7's four objective metrics from one cell's outputs.

    All values are in the same units expected by ``RealTimeTracker``:
      * slo_violation_rate        fraction in [0, 1]
      * cold_start_latency        seconds (median model load)
      * hot_start_latency         seconds (median cached switch)
      * error_to_normal_ratio     median(SLO-miss latency) / median(SLO-hit latency)
    Missing measurements are omitted (not fabricated).
    """
    metrics = {}

    submitted = 0
    misses = 0.0
    for row in rows:
        n = int(row.get("requests_submitted", 0) or 0)
        if n <= 0:
            continue
        attainment = float(row.get("latency_slo_attainment_pct", 0.0) or 0.0)
        submitted += n
        misses += n * (1.0 - attainment / 100.0)
    if submitted:
        metrics["slo_violation_rate"] = round(misses / submitted, 6)

    cold = _median_csv_column(metrics_dir / "model_load_metrics.csv", "load_time")
    if cold is not None:
        metrics["cold_start_latency"] = round(cold, 6)

    hot = _median_csv_column(
        metrics_dir / "switch_metrics.csv", "switch_time", scale=1.0 / 1000.0
    )
    if hot is None:
        hot = _median_csv_column(
            PLOT_DIR / "switch_hot" / "pairwise_switching.csv",
            "SD-v1-5",
            scale=1.0 / 1000.0,
        )
    if hot is not None:
        metrics["hot_start_latency"] = round(hot, 9)

    slo_path = metrics_dir / "slo_metrics.csv"
    hit_latencies = []
    miss_latencies = []
    try:
        if slo_path.exists():
            try:
                with open(slo_path) as f:
                    for row in csv.DictReader(f):
                        try:
                            latency = float(row["latency_sec"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        if str(row.get("met", "")).strip().lower() == "true":
                            hit_latencies.append(latency)
                        else:
                            miss_latencies.append(latency)
            except Exception:
                pass
        if hit_latencies and miss_latencies:
            normal = statistics.median(hit_latencies)
            if normal > 0:
                metrics["error_to_normal_ratio"] = round(
                    statistics.median(miss_latencies) / normal, 6
                )
    except Exception:
        pass

    return metrics


def save_objective_metrics(baseline, load_level, rows, metrics, prompt_count):
    """Persist one cell's objective metrics for the v7 report."""
    submitted = 0
    for row in rows:
        submitted += int(row.get("requests_submitted", 0) or 0)
    entry = {
        "baseline": baseline,
        "load_level": load_level,
        "prompt_count": prompt_count,
        "optimization_enabled": int(
            os.environ.get("DIFS_PERF_OPTIMIZATION", "0") == "1"
        ),
        "requests_submitted": submitted,
        "slo_violation_rate": metrics.get("slo_violation_rate", ""),
        "cold_start_latency": metrics.get("cold_start_latency", ""),
        "hot_start_latency": metrics.get("hot_start_latency", ""),
        "error_to_normal_ratio": metrics.get("error_to_normal_ratio", ""),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    existing = {}
    if OBJECTIVE_CSV.exists():
        try:
            with open(OBJECTIVE_CSV) as f:
                for row in csv.DictReader(f):
                    key = (row.get("baseline"), row.get("load_level"))
                    existing[key] = row
        except Exception:
            existing = {}
    existing[(baseline, load_level)] = entry

    OBJECTIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OBJECTIVE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OBJECTIVE_FIELDS)
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow(
                {field: existing[key].get(field, "") for field in OBJECTIVE_FIELDS}
            )
    print(f"[SAVE] Wrote objective metrics for {baseline}/{load_level} to {OBJECTIVE_CSV}")


def find_v6_reference_csv():
    """Locate a DIFS_OC_v6 benchmark CSV, preferring the canonical tree."""
    relative = (
        Path("DiffServ_0112_1730_coco2") / "0112_1730_coco" / "results"
        / "model_benchmark" / "plot_data_baseline" / "baseline_comparison.csv"
    )
    candidates = [
        V4_ROOT.parents[3] / "DIFS_OC_v6" / relative,
        V4_ROOT.parents[2] / "DIFS_OC_v6" / relative,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_reference_metrics(csv_path, baseline_name):
    """Load v6 reference metrics for one baseline (best effort)."""
    reference = {}
    if csv_path is None or not Path(csv_path).exists():
        return reference

    rows = []
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return reference

    def _weighted_violation(selected):
        submitted = 0
        misses = 0.0
        for row in selected:
            n = int(row.get("requests_submitted", 0) or 0)
            attainment = float(row.get("latency_slo_attainment_pct", 0.0) or 0.0)
            submitted += n
            misses += n * (1.0 - attainment / 100.0)
        return (misses / submitted) if submitted else None

    matched = [r for r in rows if r.get("baseline") == baseline_name]
    if not matched:
        # v6 reference CSVs may contain only a subset of systems; fall back to
        # all available v6 rows rather than reporting no reference at all.
        matched = rows
    violation = _weighted_violation(matched)
    if violation is not None:
        reference["slo_violation_rate"] = round(violation, 6)

    # .../0112_1730_coco/results/model_benchmark/plot_data_baseline/<csv>
    v6_root = Path(csv_path).resolve().parents[3]
    v6_metrics = v6_root / "metrics"
    cold = _median_csv_column(v6_metrics / "model_load_metrics.csv", "load_time")
    if cold is not None:
        reference["cold_start_latency"] = round(cold, 6)
    hot = _median_csv_column(
        v6_metrics / "switch_metrics.csv", "switch_time", scale=1.0 / 1000.0
    )
    if hot is None:
        hot = _median_csv_column(
            v6_root / "results" / "model_benchmark" / "plot_data_baseline"
            / "switch_hot" / "pairwise_switching.csv",
            "SD-v1-5",
            scale=1.0 / 1000.0,
        )
    if hot is not None:
        reference["hot_start_latency"] = round(hot, 9)

    hit_latencies = []
    miss_latencies = []
    slo_path = v6_metrics / "slo_metrics.csv"
    if slo_path.exists():
        try:
            with open(slo_path) as f:
                for row in csv.DictReader(f):
                    try:
                        latency = float(row["latency_sec"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if str(row.get("met", "")).strip().lower() == "true":
                        hit_latencies.append(latency)
                    else:
                        miss_latencies.append(latency)
        except Exception:
            pass
    if hit_latencies and miss_latencies:
        normal = statistics.median(hit_latencies)
        if normal > 0:
            reference["error_to_normal_ratio"] = round(
                statistics.median(miss_latencies) / normal, 6
            )

    return reference


# ---------------------------------------------------------------------------
# Core cell runner
# ---------------------------------------------------------------------------

def run_level(pm, baseline_name, cfg, load_level, rate, window_sec, seed,
              max_workers=MAX_WORKERS, prompt_count=SLO_PROMPT_COUNT):
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
    n_workers = min(SLO_WORKERS_BY_LEVEL.get(load_level, 5), max_workers)

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
        print(f"[BASELINE] Emitting up to {prompt_count} requests "
              f"(rate={rate:.4f} req/s, window={window_sec}s, levels={acc_levels})")

        t_end = time.time() + window_sec
        next_arrival = time.time()
        emitted = 0
        emitted_by_accuracy = {level: 0 for level in acc_levels}
        while emitted < prompt_count and time.time() < t_end:
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
    parser.add_argument("--rates", type=float, nargs="+", default=None,
                        help="Custom arrival rates in req/min (enables SLO sweep mode)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Seeds per rate in sweep mode (default: 1)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--prompt-count", type=int, default=None,
                        help="Requests per load level (default: 300; use ~12 for v7 quick runs)")
    parser.add_argument("--track", action="store_true",
                        help="Start the real-time tracker (CLI + HTTP GUI)")
    parser.add_argument("--track-gui-port", type=int, default=8080,
                        help="Port for the tracker GUI (default: 8080)")
    parser.add_argument("--track-baseline-csv", type=str, default=None,
                        help="Reference baseline_comparison.csv (default: sibling DIFS_OC_v6)")
    args = parser.parse_args()

    prompt_count = args.prompt_count or SLO_PROMPT_COUNT
    sweep_mode = bool(args.rates)
    if sweep_mode:
        rates = sorted(float(r) for r in args.rates)
        if any(r <= 0 for r in rates):
            print("[ERROR] --rates values must be positive (req/min).")
            sys.exit(1)
        levels = [rate_label(r) for r in rates]
        rate_rps = {rate_label(r): r / 60.0 for r in rates}
        rate_rpm = {rate_label(r): r for r in rates}
        repeats = max(1, int(args.repeats))
        print(f"[SWEEP] Rates (req/min): {rates}; seeds per rate: {repeats}; "
              f"prompts per cell: {prompt_count}")
    else:
        levels = args.levels or list(SLO_LOAD_LEVELS)
        rate_rps = SLO_LOAD_LEVELS
        rate_rpm = {lvl: rate * 60.0 for lvl, rate in SLO_LOAD_LEVELS.items()}
        repeats = 1
        if args.repeats and args.repeats > 1:
            print("[WARN] --repeats only applies to sweep mode (--rates); "
                  "running each named level once.")

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

    # Optional real-time tracking (CLI now, HTTP GUI while the run proceeds).
    tracker = None
    track_enabled = args.track or os.environ.get("DIFS_TRACK", "0") == "1"
    reference_csv = (
        Path(args.track_baseline_csv) if args.track_baseline_csv
        else find_v6_reference_csv()
    )
    if track_enabled:
        try:
            from tracking.realtime_tracker import RealTimeTracker
            tracker = RealTimeTracker(PROJECT_ROOT)
            tracker.baseline_metrics = load_reference_metrics(reference_csv, baselines[0])
            if os.environ.get("DIFS_TRACK_GUI", "1") != "0":
                tracker.start_gui_server(args.track_gui_port)
                print(f"[TRACK] GUI: http://127.0.0.1:{args.track_gui_port}")
            else:
                print("[TRACK] GUI managed by runner; writing shared metrics file.")
            if reference_csv:
                print(f"[TRACK] Reference baseline: {reference_csv}")
            else:
                print("[TRACK] No reference CSV found; improvements will be 'n/a'")
        except Exception as exc:
            print(f"[TRACK] WARNING: tracker disabled: {exc}")
            tracker = None

    all_rows = []
    failed_cells = []
    completed_cells = 0
    sweep_existing = load_sweep_runs() if sweep_mode else {}

    import os as _os
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
        if tracker is not None:
            tracker.stop_gui_server()
        if not sweep_mode:
            _save_csv(dict(existing))
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    for bname in baselines:
        cfg = BASELINES[bname]
        print(f"\n{'#' * 70}")
        print(f"# BASELINE: {bname}")
        print(f"# {cfg['description']}")
        print(f"{'#' * 70}")

        if tracker is not None:
            tracker.baseline_metrics = load_reference_metrics(reference_csv, bname)

        force_env = os.environ.get("DIFS_FORCE_RERUN", "0") == "1"
        force_rerun = force_env or os.environ.get("FORCE") == "1"

        for level in levels:
            if level not in rate_rps:
                print(f"[ERROR] Unknown load level/rate: {level}")
                failed_cells.append((bname, level, "unknown level/rate"))
                continue

            window = max(1, round(prompt_count / rate_rps[level]))
            seeds = ([args.seed + i for i in range(repeats)]
                     if sweep_mode else [args.seed])

            for seed in seeds:
                if sweep_mode:
                    key = (bname, level, str(seed))
                    prev = sweep_existing.get(key)
                    if (prev and str(prev.get("prompt_count", "")) == str(prompt_count)
                            and not force_rerun):
                        print(f"[SKIP] {bname}/{level}/seed={seed} already "
                              f"completed in previous sweep.")
                        completed_cells += 1
                        continue
                elif (bname, level) in existing and not force_rerun:
                    print(f"[SKIP] {bname}/{level} already completed in previous run.")
                    all_rows.extend(existing[(bname, level)])
                    continue

                success = False
                cell_tag = (f"{bname}/{level}/seed={seed}" if sweep_mode
                            else f"{bname}/{level}")
                for attempt in range(1, MAX_CELL_RETRIES + 1):
                    try:
                        rows = run_level(pm, bname, cfg, level,
                                         rate_rps[level], window, seed,
                                         max_workers=args.workers,
                                         prompt_count=prompt_count)
                        cell_metrics = compute_objective_metrics(
                            rows, PROJECT_ROOT / "metrics", RESULTS_DIR
                        )
                        # Per-request export (Table 2 source) with provenance.
                        try:
                            mode = "fast_sweep" if (sweep_mode and str(PLOT_DIR).find("model_benchmark_fast") >= 0) else ("sweep" if sweep_mode else "full")
                            per_dir = PLOT_DIR / "per_request_exports"
                            per_rows = _snapshot_per_request_metrics(bname, level, seed, mode, per_dir)
                            append_per_request_export(per_rows, PLOT_DIR / "per_request_all.csv")
                        except Exception as pr_exc:
                            print(f"[SAVE] WARNING: per-request export not saved: {pr_exc}")
                        if sweep_mode:
                            save_sweep_run(
                                bname, level, rate_rpm[level], seed, rows,
                                cell_metrics, prompt_count, window
                            )
                            completed_cells += 1
                        else:
                            all_rows.extend(rows)
                            existing[(bname, level)] = rows
                            _save_csv(existing)
                            print(f"[SAVE] Wrote {len(existing)} cells to {OUT_CSV}")
                            try:
                                save_objective_metrics(
                                    bname, level, rows, cell_metrics, prompt_count
                                )
                            except Exception as obj_exc:
                                print(f"[SAVE] WARNING: objective metrics not saved: {obj_exc}")
                            if tracker is not None:
                                try:
                                    tracker.update_metrics(cell_metrics)
                                    tracker.display_cli_dashboard()
                                except Exception as track_exc:
                                    print(f"[TRACK] WARNING: could not update tracker: {track_exc}")
                        success = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        is_oom = ("out of memory" in error_str.lower()
                                  or "cuda" in error_str.lower())
                        print(f"[BASELINE ERROR] {cell_tag} "
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
                            print(f"[BASELINE FAILED] {cell_tag}: "
                                  f"gave up after {MAX_CELL_RETRIES} attempts.")
                            failed_cells.append((bname, level, seed, error_str))

                if not success:
                    continue

    if tracker is not None:
        tracker.display_cli_dashboard()
        tracker.stop_gui_server()

    # Summary
    print(f"\n{'=' * 70}")
    if sweep_mode:
        print(f"[SUMMARY] Completed: {completed_cells} sweep cells in {SWEEP_CSV}")
    else:
        total_rows = sum(len(v) if isinstance(v, list) else 1
                         for v in existing.values())
        print(f"[SUMMARY] Completed: {len(existing)} cells "
              f"({total_rows} per-step rows) in {OUT_CSV}")
    if failed_cells:
        print(f"[SUMMARY] FAILED ({len(failed_cells)}):")
        for failed in failed_cells:
            bname, level, *rest = failed
            err = rest[-1] if rest else "unknown error"
            seed_info = f"/seed={rest[0]}" if len(rest) > 1 else ""
            print(f"  - {bname}/{level}{seed_info}: {str(err)[:80]}")
    else:
        print("[SUMMARY] All cells completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
