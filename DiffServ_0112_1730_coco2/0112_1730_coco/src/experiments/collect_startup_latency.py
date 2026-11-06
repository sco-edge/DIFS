#!/usr/bin/env python3
"""
collect_startup_latency.py

Derive the startup-latency comparison from measured pairwise switching
matrices:

    * baseline     : INFaaS with a cold start for every request (0% target)
    * infaas_warm  : INFaaS warm start, FIFO, no grouping (25% target)
    * difs_warm    : DIFS warm start with model-grouping reordering (50% target)

Inputs (written by collect_pairwise_switching.py):
    switch_cold_every/pairwise_switching.csv
    switch_force_upload/pairwise_switching.csv

The cold matrix should include diagonal measurements (cold reload of the same
model).  The collector is called with --measure-diagonal for cold_every.

Output:
    <plot_data_baseline>/startup_latency.csv

How to run
    ./run_all_experiments.sh            # menu -> 11 -> 1 (or 3)
    PYTHONPATH=src python src/experiments/collect_startup_latency.py
"""
import argparse
import csv
import os
import random
import statistics
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_paths import PROJECT_ROOT  # noqa: E402
from model_registry import ALL_MODELS  # noqa: E402

DEFAULT_MODEL_IDS = [3, 4, 5, 8, 9]
TARGET_PCT = {"baseline_cold": 0.0, "infaas_warm": 25.0, "difs_warm": 50.0}


def resolve_data_dir():
    if os.environ.get("DIFS_PLOT_DATA_DIR"):
        base = Path(os.environ["DIFS_PLOT_DATA_DIR"])
        if base.name != "plot_data_baseline":
            base = base / "plot_data_baseline"
        return base
    if os.environ.get("DIFS_LAST_RESULTS_DIR"):
        return Path(os.environ["DIFS_LAST_RESULTS_DIR"]) / "plot_data_baseline"
    if os.environ.get("DIFS_LIVE_RESULTS_DIR"):
        return Path(os.environ["DIFS_LIVE_RESULTS_DIR"]) / "plot_data_baseline"
    return PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"


def read_matrix(path):
    """Read a from_model x to_model matrix into {(from, to): ms}."""
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        to_models = [name.strip() for name in header[1:]]
        data = {}
        for row in reader:
            if not row:
                continue
            from_model = row[0].strip()
            for idx, raw in enumerate(row[1:], start=1):
                if idx >= len(header):
                    break
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                data[(from_model, to_models[idx - 1])] = value
    return to_models, data


def model_cost(matrix, model, prefer_diagonal=False):
    """Representative start cost for *model* in milliseconds."""
    if prefer_diagonal:
        diagonal = matrix.get((model, model), 0.0)
        if diagonal > 0:
            return diagonal
    values = [value for (_, to_model), value in matrix.items()
              if to_model == model and value > 0]
    if not values:
        return None
    return statistics.median(values)


def fill_costs(matrix, models, prefer_diagonal=False):
    costs = {}
    missing = []
    for model in models:
        cost = model_cost(matrix, model, prefer_diagonal=prefer_diagonal)
        if cost is None:
            missing.append(model)
        else:
            costs[model] = cost / 1000.0
    if missing:
        all_values = [value / 1000.0 for (_, to_model), value in matrix.items()
                      if to_model in models and value > 0]
        fallback = statistics.median(all_values) if all_values else None
        if fallback is None:
            raise SystemExit(f"[STARTUP] No usable costs for: {', '.join(missing)}")
        for model in missing:
            costs[model] = fallback
        print(f"[STARTUP] WARNING: fallback cost used for {', '.join(missing)}")
    return costs


def build_trace(models, requests, seed):
    rng = random.Random(seed)
    return [rng.choice(models) for _ in range(requests)]


def grouped_startups(trace, window):
    """Return (warm_start_events, per_group_models) for DIFS grouping."""
    groups = []
    for start in range(0, len(trace), max(1, window)):
        chunk = trace[start:start + max(1, window)]
        seen = []
        for model in chunk:
            if model not in seen:
                seen.append(model)
        groups.extend(seen)
    return len(groups), groups


def summarize(system, requests, cold_total, warm_total, warm_events,
              group_window, seed):
    reduction = 0.0
    if cold_total > 0:
        reduction = 100.0 * (1.0 - warm_total / cold_total)
    return {
        "system": system,
        "requests": requests,
        "cold_start_events": requests,
        "warm_start_events": warm_events,
        "total_cold_s": round(cold_total, 6),
        "total_warm_s": round(warm_total, 6),
        "reduction_pct": round(reduction, 4),
        "target_pct": TARGET_PCT[system],
        "group_window": group_window,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Derive startup-latency reductions from switching matrices."
    )
    parser.add_argument("--data-dir", default=None,
                        help="plot_data_baseline directory")
    parser.add_argument("--cold-csv", default=None,
                        help="cold_every pairwise_switching.csv")
    parser.add_argument("--warm-csv", default=None,
                        help="force_upload pairwise_switching.csv")
    parser.add_argument("--models", nargs="+", type=int,
                        default=DEFAULT_MODEL_IDS,
                        help="Model IDs included in the trace")
    parser.add_argument("--requests", type=int, default=120,
                        help="Requests in the startup trace (default: 120)")
    parser.add_argument("--window", type=int, default=5,
                        help="DIFS grouping window in requests (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else resolve_data_dir()
    cold_csv = Path(args.cold_csv) if args.cold_csv else (
        data_dir / "switch_cold_every" / "pairwise_switching.csv"
    )
    warm_csv = Path(args.warm_csv) if args.warm_csv else (
        data_dir / "switch_force_upload" / "pairwise_switching.csv"
    )
    output = Path(args.output) if args.output else (data_dir / "startup_latency.csv")

    for path in (cold_csv, warm_csv):
        if not path.exists():
            print(f"[STARTUP] Missing switching matrix: {path}")
            print("[STARTUP] Run option 3 (experiments) or the pairwise "
                  "switching stage first.")
            return 1

    models = []
    for model_id in args.models:
        entry = ALL_MODELS.get(model_id)
        if entry:
            models.append(entry["short_name"])
        else:
            print(f"[STARTUP] WARNING: unknown model id {model_id}; skipping")
    if not models:
        print("[STARTUP] No valid models.")
        return 1

    cold_models, cold_matrix = read_matrix(cold_csv)
    warm_models, warm_matrix = read_matrix(warm_csv)
    shared = [model for model in models
              if model in cold_models and model in warm_models]
    if not shared:
        print("[STARTUP] No shared model names between matrices and --models.")
        print(f"  cold: {cold_models}")
        print(f"  warm: {warm_models}")
        return 1
    if len(shared) != len(models):
        print(f"[STARTUP] WARNING: using shared models only: {shared}")
    models = shared

    cold_costs = fill_costs(cold_matrix, models, prefer_diagonal=True)
    warm_costs = fill_costs(warm_matrix, models, prefer_diagonal=False)

    trace = build_trace(models, max(1, args.requests), args.seed)

    cold_total = sum(cold_costs[model] for model in trace)
    infaas_warm_total = sum(warm_costs[model] for model in trace)

    window = max(1, args.window)
    difs_events, difs_groups = grouped_startups(trace, window)
    difs_warm_total = sum(warm_costs[model] for model in difs_groups)

    rows = [
        summarize("baseline_cold", len(trace), cold_total, cold_total,
                  len(trace), window, args.seed),
        summarize("infaas_warm", len(trace), cold_total, infaas_warm_total,
                  len(trace), window, args.seed),
        summarize("difs_warm", len(trace), cold_total, difs_warm_total,
                  difs_events, window, args.seed),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[STARTUP] Models: {', '.join(models)}")
    print(f"[STARTUP] Trace: {len(trace)} requests, grouping window={window}")
    print(f"[STARTUP] Cold total: {cold_total:.3f} s")
    for row in rows:
        print(f"[STARTUP] {row['system']:14s} "
              f"warm_starts={row['warm_start_events']:4d} "
              f"warm_total={row['total_warm_s']:8.3f} s "
              f"reduction={row['reduction_pct']:6.2f}% "
              f"(target {row['target_pct']:.0f}%)")
    print(f"[STARTUP] Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
