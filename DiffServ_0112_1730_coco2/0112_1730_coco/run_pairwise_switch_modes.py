#!/usr/bin/env python3
"""
run_pairwise_switch_modes.py

Orchestrates the pairwise model-switching measurement for DIFS_OC_v5.

For each requested MODEL_SWITCH_MODE it:
  1. launches a single DIFS worker (direct mode) with that switch mode,
  2. runs collect_pairwise_switching.py against it (full 5x5 x repeat),
  3. copies the resulting CSV matrix + raw data into a per-mode folder under
     results/model_benchmark/plot_data_baseline/,
  4. tears the worker down.

Supported modes (see src/config.py / src/1/pytorch_container_diffusion_v3.py):
  hot          : cache pointer flip (baseline, v4 behaviour)
  cold_every   : always reload from disk, never reuse cache  (OPTION 3)
  force_upload : cache hit + genuine DRAM->GPU re-upload+sync (OPTION 2)

Usage:
  DIFS_Gen/bin/python run_pairwise_switch_modes.py --modes hot cold_every force_upload \
      --repeat 3 --models 3 4 5 8 9
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
WORKER_SCRIPT = SRC_DIR / "1" / "pytorch_container_diffusion_v3.py"
COLLECT_SCRIPT = SRC_DIR / "experiments" / "collect_pairwise_switching.py"
MODEL_CACHE = Path("/mnt/ssd4t/kwadwo/Downloads/sd-models/downloaded_diffusion_models/diffusion-models")
REAL_STATS = PROJECT_ROOT / "real_stats.npz"
OUT_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"

# port -> model_id that each worker boots with; only ONE worker is used here.
WORKER_PORT = 50060


def wait_for_port(port, timeout=300):
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def start_worker(model_id, mode):
    cmd = [
        sys.executable, "-u",
        str(WORKER_SCRIPT),
        "-model", str(model_id),
        "-sampler", "1",
        "-thread", "10",
        "-port", str(WORKER_PORT),
        "-npz", str(REAL_STATS),
    ]
    env = os.environ.copy()
    env["DIFS_MODEL_DIR"] = str(MODEL_CACHE)
    env["DIFS_OUTPUT_DIR"] = str(PROJECT_ROOT / "out_tmp")
    env["DIFS_MODEL_SWITCH_MODE"] = mode
    # IMPORTANT: redirect worker stdout/stderr to a file. Using an unread
    # subprocess.PIPE here caused a hang: the worker's very verbose load_model
    # output fills the 64KB pipe buffer and the worker blocks forever on write.
    worker_log = OUT_DIR / f"switch_{mode}" / "worker.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    wlog = open(worker_log, "a")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                            stdout=wlog, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL)
    print(f"[RUN] worker pid={proc.pid} model={model_id} mode={mode} port={WORKER_PORT}")
    if not wait_for_port(WORKER_PORT, timeout=600):
        print("[RUN] ERROR worker did not become ready in time; aborting")
        proc.terminate()
        sys.exit(1)
    return proc


def stop_worker(proc):
    if proc is None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait()


def run_mode(mode, models, repeat, delay):
    mode_dir = OUT_DIR / f"switch_{mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    matrix_csv = mode_dir / "pairwise_switching.csv"

    worker = None
    try:
        worker = start_worker(models[0], mode)

        cmd = [
            sys.executable, "-u", str(COLLECT_SCRIPT),
            "--worker", f"localhost:{WORKER_PORT}",
            "--models", *(str(m) for m in models),
            "--warmup", "1",
            "--repeat", str(repeat),
            "--delay", str(delay),
            "--output", str(matrix_csv),
        ]
        print(f"[RUN] === collecting mode={mode} ===\n")
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        print(f"[RUN] collect exit code = {r.returncode}")

        if r.returncode == 0 and matrix_csv.exists():
            # collect_pairwise_switching.py writes *_raw.csv next to the matrix.
            raw_src = matrix_csv.with_name("pairwise_switching_raw.csv")
            if raw_src.exists():
                print(f"[RUN] Wrote raw     -> {raw_src}")
            print(f"[RUN] Wrote matrix -> {matrix_csv}")
        else:
            print(f"[RUN] WARNING collect failed or no CSV for mode={mode}")
    finally:
        stop_worker(worker)


def main():
    ap = argparse.ArgumentParser(description="Run pairwise switching measurement across modes")
    ap.add_argument("--modes", nargs="+",
                    default=["hot", "cold_every", "force_upload"],
                    help="switch modes to measure")
    ap.add_argument("--models", nargs="+", type=int, default=[3, 4, 5, 8, 9])
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--delay", type=float, default=2.0, help="delay (s) between switches")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        run_mode(mode, args.models, args.repeat, args.delay)

    print("\n[RUN] all modes complete.")


if __name__ == "__main__":
    main()
