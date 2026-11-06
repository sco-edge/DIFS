#!/usr/bin/env python3
"""
collect_pairwise_switching.py

Connects to a running DIFS worker and measures model-to-model switching times
by calling the LoadModel gRPC RPC for every (from_model, to_model) pair.

The worker must be running with all models pre-loaded in its cache
(requires enough GPU VRAM or CPU offload).

How to run
    Recommended (starts the worker, measures all three switch modes, stops it):
        ./run_all_experiments.sh --experiments --skip-baseline

    Manual recipe, from the project root:
        # 1. Start one worker on port 50060 with the first model.
        DIFS_MODEL_DIR=/path/to/models \\
        DIFS_OUTPUT_DIR=/path/to/output \\
        DIFS_MODEL_SWITCH_MODE=hot \\
        PYTHONPATH=src python src/1/pytorch_container_diffusion_v3.py \\
            -model 3 -sampler 1 -thread 10 -port 50060 \\
            -npz real_stats.npz

        # 2. In another shell, measure all ordered pairs.
        PYTHONPATH=src python src/experiments/collect_pairwise_switching.py \\
            --worker localhost:50060 \\
            --models 3 4 5 8 9 \\
            --warmup 1 --repeat 3 --delay 2.0 \\
            --output results/model_benchmark/plot_data_baseline/switch_hot/pairwise_switching.csv

        # 3. Stop the worker, then repeat for DIFS_MODEL_SWITCH_MODE=force_upload
        #    and cold_every, writing to switch_<mode>/pairwise_switching.csv.

    Other examples:
        PYTHONPATH=src python src/experiments/collect_pairwise_switching.py \\
            --worker localhost:50060 --models 1 2 3 4 5 7 8 \\
            --output results/model_benchmark/plot_data/pairwise_switching.csv

        PYTHONPATH=src python src/experiments/collect_pairwise_switching.py \\
            --worker localhost:50060 --models 1 2 3 7 --warmup 2 --repeat 5

The script:
    1. Starts a worker with the first model
    2. Pre-loads all other models into cache (hot-switch them once)
    3. For each ordered pair (A, B), calls LoadModel to switch A -> B
    4. Records the switch_time_ms from the RPC response
    5. Writes a N x N CSV matrix
"""
import argparse
import csv
import os
import sys
import time

import grpc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import query_pb2
import query_pb2_grpc

from model_registry import ALL_MODELS


def connect_worker(address):
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=30)
    return query_pb2_grpc.QueryStub(channel)


def load_model_rpc(stub, model_filename):
    """Call LoadModel RPC and return (switch_time_ms, success)."""
    start = time.perf_counter()
    try:
        response = stub.LoadModel(
            query_pb2.LoadModelRequest(
                model_id=0,
                model_name=model_filename,
            )
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, response.switch_time_ms, response.success
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ERROR: {e}")
        return elapsed_ms, -1, False


def get_worker_status(stub):
    try:
        return stub.WorkerStatus(query_pb2.WorkerStatusRequest())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Measure pairwise model switching times via gRPC LoadModel RPC."
    )
    parser.add_argument(
        "--worker", default="localhost:50060",
        help="Worker gRPC address (default: localhost:50060)"
    )
    parser.add_argument(
        "--models", nargs="+", type=int, default=[1, 2, 3, 4, 5, 7, 8],
        help="Model IDs to include (default: 1 2 3 4 5 7 8)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Warm-up iterations before measurement (default: 1)"
    )
    parser.add_argument(
        "--repeat", type=int, default=3,
        help="Number of measurements per pair (default: 3)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: results/model_benchmark/plot_data/pairwise_switching.csv)"
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Delay between switches in seconds (default: 2.0)"
    )
    args = parser.parse_args()

    # Resolve model filenames
    model_ids = args.models
    model_names = []
    for mid in model_ids:
        if mid not in ALL_MODELS:
            print(f"WARNING: model_id {mid} not in registry, skipping")
            continue
        model_names.append(ALL_MODELS[mid]["filename"])

    n = len(model_names)
    print(f"Models ({n}): {model_names}")
    print(f"Worker: {args.worker}")
    print(f"Warmup: {args.warmup}, Repeat: {args.repeat}")
    print()

    # Connect
    print(f"Connecting to {args.worker} ...")
    stub = connect_worker(args.worker)
    status = get_worker_status(stub)
    if status:
        print(f"  Worker model: {status.model_name}")
    print()

    # Phase 1: Warm-up — load all models into cache
    print("=" * 60)
    print("Phase 1: Pre-loading all models into worker cache")
    print("=" * 60)
    for i in range(args.warmup):
        print(f"  Warm-up round {i + 1}/{args.warmup}")
        for name in model_names:
            elapsed, switch_ms, ok = load_model_rpc(stub, name)
            print(f"    {name}: {switch_ms:.2f} ms (ok={ok})")
            time.sleep(0.5)
    print()

    # Phase 2: Pairwise measurement
    print("=" * 60)
    print("Phase 2: Measuring pairwise switching times")
    print("=" * 60)

    # results[from_idx][to_idx] = list of switch_time_ms values
    results = [[[] for _ in range(n)] for _ in range(n)]

    for rep in range(args.repeat):
        print(f"\n  Repetition {rep + 1}/{args.repeat}")
        for from_idx, from_name in enumerate(model_names):
            # First switch TO from_model to ensure it's current
            load_model_rpc(stub, from_name)
            time.sleep(args.delay)

            for to_idx, to_name in enumerate(model_names):
                if from_idx == to_idx:
                    results[from_idx][to_idx].append(0.0)
                    continue

                elapsed, switch_ms, ok = load_model_rpc(stub, to_name)
                results[from_idx][to_idx].append(switch_ms)
                print(f"    {from_name} -> {to_name}: {switch_ms:.2f} ms")
                time.sleep(args.delay)

    # Phase 3: Compute averages and write CSV
    print()
    print("=" * 60)
    print("Phase 3: Writing results")
    print("=" * 60)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            SRC_DIR, "..", "results", "model_benchmark",
            "plot_data", "pairwise_switching.csv"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write raw measurements
    raw_path = output_path.replace(".csv", "_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["from_model", "to_model", "repetition", "switch_time_ms"])
        for from_idx in range(n):
            for to_idx in range(n):
                for rep_idx, val in enumerate(results[from_idx][to_idx]):
                    writer.writerow([
                        model_names[from_idx],
                        model_names[to_idx],
                        rep_idx + 1,
                        round(val, 2),
                    ])
    print(f"  Raw data: {raw_path}")

    # Write averaged matrix
    short_names = []
    for mid in model_ids:
        if mid in ALL_MODELS:
            short_names.append(ALL_MODELS[mid]["short_name"])
        else:
            short_names.append(f"model_{mid}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["from_model \\ to_model"] + short_names
        writer.writerow(header)
        for from_idx in range(n):
            row = [short_names[from_idx]]
            for to_idx in range(n):
                vals = results[from_idx][to_idx]
                avg = sum(vals) / len(vals) if vals else 0
                row.append(round(avg, 2))
            writer.writerow(row)
    print(f"  Matrix CSV: {output_path}")

    # Print summary
    print()
    print("Pairwise switching time matrix (ms):")
    print(f"{'':>20s}", end="")
    for sn in short_names:
        print(f"{sn:>14s}", end="")
    print()
    for from_idx in range(n):
        print(f"{short_names[from_idx]:>20s}", end="")
        for to_idx in range(n):
            vals = results[from_idx][to_idx]
            avg = sum(vals) / len(vals) if vals else 0
            print(f"{avg:>14.1f}", end="")
        print()

    print()
    print("Done.")


if __name__ == "__main__":
    main()
