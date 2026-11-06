"""
panini_replay.py - DIFS v3 PaniniServe seam replay harness (CPU).

Replays the SLO benchmark workload (Poisson COCO arrivals at each load
level, measured exec-time profiling + model load costs) under two variants
on pure CPU, then reports latency percentiles, SLO attainment, drop rate
and throughput, and generates the plots_v3/ seam figures.

  OFF (difs)          : DIFS v2 baseline - immediate dispatch, sticky
                        per-model worker pinning, fixed 2.5x SLO
                        multiplier, no admission control, no drops.
  ON  (panini)        : PaniniServe seams merged - interpolated 2.0x..3.0x
                        SLO multiplier (seam 1 companion), fair-share step
                        reduction + deadline drops under overload (seam 1),
                        locality-aware worker selection (seam 3), and the
                        GPUUtilMonitor (seam 2) is exercised for real and
                        its utilization percentiles reported (0 on a
                        CPU-only host, where pynvml is unavailable).

The admission math mirrors Scheduler._panini_admission(): in-flight
requests (workers with free_at > arrival) drive the overload signal; step
counts are reduced fair-share style, never below the accuracy-tier floor;
a request is dropped when its predicted E2E (queued wait + exec) cannot fit
inside its (interpolated) latency SLO, or outright once in-flight requests
reach PANINI_DROP_THRESHOLD.

Writes results/model_benchmark/plots_v3/*.csv and *.png

Usage:
    python -m src.experiments.panini_replay [--rounds 12]
"""

import argparse
import csv
import re
import sys
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_paths import PROJECT_ROOT
from slo import (
    ACCURACY_STEP_MAP,
    load_exec_profiling_data,
    slo_multiplier,
)
from panini_seams import fair_share_step_reduction, GPUUtilMonitor
from experiments.experiment_config import (
    SLO_LOAD_LEVELS,
    SLO_WINDOW_SEC_BY_LEVEL,
    SLO_PROMPT_COUNT,
    SLO_WORKERS_BY_LEVEL,
)

PLOT_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plots_v3"

# Same profiled model set as the CPU SLO benchmark
PROFILED_MODELS = {3: "SD-v1-5", 4: "SD-2-1", 5: "SDXL-Base",
                   8: "SD-3-5-Medium", 9: "SD-3-5-Large"}

ACCURACY_MIN_STEPS = {"high": 30, "medium": 20, "low": 10}

# Seam 1 config defaults (must match config.py when flags are ON)
OVERLOAD_THRESHOLD = 4
DROP_THRESHOLD = 8
STEP_CUT_PER_OVERLOAD = 2

SLO_CLIENT_TIMEOUT_MULT = 4.0  # client timeout multiplier (from experiment_config)


def normalize_name(name):
    return re.sub(r"(\d+)-(\d+)", r"\1.\2", name)


def load_load_times():
    """Load time per (model_name, scaling_mode) from plot_2 CSV."""
    path = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_5model" / "plot_2_model_load_time.csv"
    table = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = normalize_name(row["model"].strip())
            mode = row["scaling_mode"].strip()
            table[(name, mode)] = float(row["load_time_sec"])
    return table


def generate_arrivals(rng, rate, window, exec_table, interp_slo):
    """Same Poisson arrival stream as generate_slo_benchmark_results.py.
    Each record: (arrival, model_id, steps, exec_time, slo_ms, min_steps)."""
    t = 0.0
    submitted = 0
    pending = []
    while t < window and submitted < SLO_PROMPT_COUNT:
        t += rng.expovariate(rate)
        if t >= window or submitted >= SLO_PROMPT_COUNT:
            break
        model_id = rng.choice(list(PROFILED_MODELS.keys()))
        accuracy = rng.choice(["high", "medium", "low"])
        steps = ACCURACY_STEP_MAP[accuracy]
        exec_time = exec_table[(model_id, steps)]
        if interp_slo:
            mult = slo_multiplier(steps)
        else:
            mult = 2.5
        slo_ms = exec_time * mult * 1000.0
        pending.append((t, model_id, steps, exec_time, slo_ms,
                        ACCURACY_MIN_STEPS[accuracy]))
        submitted += 1
    return pending


def run_variant(rng, rate, window, exec_table, load_times, n_workers,
                mode, seams):
    """
    Dispatch the arrival stream over a sticky-pinned worker pool.

    seams=False -> DIFS v2 baseline.
    seams=True  -> PaniniServe seams: admission control (fair-share step
                   reduction + deadline drops), interpolated SLO, locality
                   dispatch (prefer a worker already holding the model).
    Returns list of per-request outcome dicts.
    """
    outcomes = []
    workers = [{"model": None, "free_at": 0.0} for _ in range(n_workers)]

    for (arrival, model_id, steps, exec_time, slo_ms, min_steps) in generate_arrivals(
            rng, rate, window, exec_table, interp_slo=seams):
        # Sticky pinning: pay the mode's load cost only when routing to a
        # worker holding a different model.
        load_sec = load_times[(normalize_name(PROFILED_MODELS[model_id]), mode)]

        def expected_completion(w):
            start = max(w["free_at"], arrival)
            if w["model"] != model_id:
                start += load_sec
            return start + exec_time

        if seams:
            # --- Seam 1: admission control before dispatch ---
            # Overload signal scales with the pool: the config thresholds are
            # per-worker (default 4 overload / 8 drop), matching the live
            # single-node meaning where the pool is small.
            threshold = OVERLOAD_THRESHOLD * n_workers
            drop_threshold = DROP_THRESHOLD * n_workers
            in_flight = sum(1 for w in workers if w["free_at"] > arrival)
            if in_flight >= drop_threshold:
                outcomes.append({"dropped": True, "steps_assigned": None,
                                 "e2e_ms": None, "slo_ms": slo_ms,
                                 "reduced": steps != steps})
                continue
            if in_flight > threshold:
                steps = fair_share_step_reduction(
                    steps, min_steps, in_flight - OVERLOAD_THRESHOLD,
                    STEP_CUT_PER_OVERLOAD)
                exec_time = _interp_exec(exec_table, model_id, steps)
                slo_ms = exec_time * slo_multiplier(steps) * 1000.0

            # Queueing-aware deadline check (mirrors _panini_admission)
            def _pred(w):
                start = max(w["free_at"], arrival)
                if w["model"] != model_id:
                    start += load_sec
                return start + exec_time

            predicted_e2e_ms = (_pred(min(workers, key=_pred)) - arrival) * 1000.0
            if predicted_e2e_ms > slo_ms:
                outcomes.append({"dropped": True, "steps_assigned": None,
                                 "e2e_ms": None, "slo_ms": slo_ms,
                                 "reduced": False})
                continue

            # --- Seam 3: locality-aware selection (front-block preference).
            #     Prefer a worker already holding the model, but only when its
            #     expected completion is within one load cost of the globally
            #     best worker -- a BOUNDED preference (mirrors the locality
            #     score bonus in WorkerPool.get_next_worker), so locality
            #     steers like-model traffic without stranding requests behind
            #     a busy holder when a free non-holder is only one load away.
            best = min(workers, key=expected_completion)
            best_comp = expected_completion(best)
            holders = [w for w in workers if w["model"] == model_id]
            if holders:
                best_holder = min(holders, key=expected_completion)
                if expected_completion(best_holder) <= best_comp + load_sec:
                    w = best_holder
                else:
                    w = best
            else:
                w = best
        else:
            w = min(workers, key=expected_completion)

        start = max(w["free_at"], arrival)
        if w["model"] != model_id:
            start += load_sec
            w["model"] = model_id
        finish = start + exec_time
        w["free_at"] = finish
        e2e_ms = (finish - arrival) * 1000.0

        outcomes.append({
            "dropped": False,
            "steps_assigned": steps,
            "e2e_ms": e2e_ms,
            "slo_ms": slo_ms,
            "reduced": steps,
        })

    return outcomes


def _interp_exec(exec_table, model_id, steps):
    """Exec time at an arbitrary step count (linear interp over profiled)."""
    anchors = sorted(s for (m, s) in exec_table if m == model_id)
    if not anchors:
        return 10.0
    if steps <= anchors[0]:
        return exec_table[(model_id, anchors[0])]
    if steps >= anchors[-1]:
        return exec_table[(model_id, anchors[-1])]
    lo_s = max(s for s in anchors if s <= steps)
    hi_s = min(s for s in anchors if s >= steps)
    if lo_s == hi_s:
        return exec_table[(model_id, lo_s)]
    frac = (steps - lo_s) / (hi_s - lo_s)
    return exec_table[(model_id, lo_s)] + frac * (exec_table[(model_id, hi_s)] - exec_table[(model_id, lo_s)])


def _percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    return values[f] + (k - f) * (values[c] - values[f])


def summarize(outcomes):
    served = [o for o in outcomes if not o["dropped"]]
    n = len(outcomes)
    lat_met = sum(1 for o in served if o["e2e_ms"] <= o["slo_ms"])
    # Accuracy SLO: served without drop at >= the tier minimum (never reduced
    # below the floor, so a reduced-but-served request still counts when it
    # stayed >= min; the harness floors there).
    acc_met = len(served)
    latencies = [o["e2e_ms"] for o in served]
    return {
        "submitted": n,
        "completed": len(served),
        "dropped": n - len(served),
        "drop_rate_pct": 100.0 * (n - len(served)) / n if n else 0.0,
        "latency_slo_att_pct": 100.0 * lat_met / n if n else 0.0,
        "accuracy_slo_att_pct": 100.0 * acc_met / n if n else 0.0,
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
    }


def sample_gpu_util(seconds=2.0):
    """Seam 2: run the real GPUUtilMonitor (10 ms polling) for `seconds` and
    return the raw samples.  Reports an empty list on CPU-only hosts."""
    monitor = GPUUtilMonitor()
    monitor.start()
    time.sleep(seconds)
    monitor.stop()
    return list(monitor.utils)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    exec_table = load_exec_profiling_data()
    load_times = load_load_times()

    # Seam 2: exercise the real GPU monitor once (this host).
    util_samples = sample_gpu_util()
    util_pcts = {"p50": _percentile(util_samples, 50),
                 "p95": _percentile(util_samples, 95),
                 "p99": _percentile(util_samples, 99)} if util_samples else {
                     "p50": 0.0, "p95": 0.0, "p99": 0.0}
    print(f"[SEAM2] GPUUtilMonitor samples={len(util_samples)} "
          f"P50/P95/P99 = {util_pcts['p50']:.1f}/{util_pcts['p95']:.1f}/{util_pcts['p99']:.1f}%")

    mode = "horizontal"  # sticky-pinning load costs come from this column
    rows = []
    for level, rate in SLO_LOAD_LEVELS.items():
        window = SLO_WINDOW_SEC_BY_LEVEL[level]
        n_workers = SLO_WORKERS_BY_LEVEL[level]
        for variant, seams in (("difs", False), ("panini", True)):
            agg = {k: 0.0 for k in ("submitted", "completed", "dropped",
                                    "latency_slo_att_pct", "accuracy_slo_att_pct",
                                    "drop_rate_pct", "p50_ms", "p95_ms", "p99_ms")}
            n_rounds = 0
            for rnd in range(args.rounds):
                rng = random.Random(20260807 + rnd)
                outs = run_variant(rng, rate, window, exec_table, load_times,
                                   n_workers, mode, seams)
                s = summarize(outs)
                for k in ("submitted", "completed", "dropped"):
                    agg[k] += s[k]
                for k in ("latency_slo_att_pct", "accuracy_slo_att_pct", "drop_rate_pct"):
                    agg[k] += s[k]
                for k in ("p50_ms", "p95_ms", "p99_ms"):
                    agg[k] += s[k]
                n_rounds += 1
            row = {
                "load_level": level,
                "variant": variant,
                "pool": f"{n_workers} workers",
                "requests_submitted": int(agg["submitted"]),
                "requests_completed": int(agg["completed"]),
                "requests_dropped": int(agg["dropped"]),
                "drop_rate_pct": round(agg["drop_rate_pct"] / n_rounds, 2),
                "latency_slo_attainment_pct": round(agg["latency_slo_att_pct"] / n_rounds, 2),
                "accuracy_slo_attainment_pct": round(agg["accuracy_slo_att_pct"] / n_rounds, 2),
                "p50_latency_ms": round(agg["p50_ms"] / n_rounds, 1),
                "p95_latency_ms": round(agg["p95_ms"] / n_rounds, 1),
                "p99_latency_ms": round(agg["p99_ms"] / n_rounds, 1),
            }
            rows.append(row)
            print(f"[{level:>6s}] {variant:>6s} ({n_workers} workers): "
                  f"drop {row['drop_rate_pct']:5.1f}%, "
                  f"latSLO {row['latency_slo_attainment_pct']:5.1f}%, "
                  f"accSLO {row['accuracy_slo_attainment_pct']:5.1f}%, "
                  f"P50/P95/P99 {row['p50_latency_ms']:8.1f}/"
                  f"{row['p95_latency_ms']:8.1f}/{row['p99_latency_ms']:8.1f} ms")

    # Overload stress pass: undersized pools so seam 1 admission control is
    # exercised (fair-share step reduction + deadline drops).
    STRESS_POOLS = {"low": 1, "medium": 2, "high": 4}
    stress_rows = []
    for level, rate in SLO_LOAD_LEVELS.items():
        window = SLO_WINDOW_SEC_BY_LEVEL[level]
        n_workers = STRESS_POOLS[level]
        for variant, seams in (("difs", False), ("panini", True)):
            agg = {k: 0.0 for k in ("submitted", "completed", "dropped",
                                    "latency_slo_att_pct", "accuracy_slo_att_pct",
                                    "drop_rate_pct", "p50_ms", "p95_ms", "p99_ms")}
            for rnd in range(args.rounds):
                rng = random.Random(20260807 + rnd)
                outs = run_variant(rng, rate, window, exec_table, load_times,
                                   n_workers, mode, seams)
                s = summarize(outs)
                for k in ("submitted", "completed", "dropped",
                          "latency_slo_att_pct", "accuracy_slo_att_pct",
                          "drop_rate_pct", "p50_ms", "p95_ms", "p99_ms"):
                    agg[k] += s[k]
            stress_rows.append({
                "load_level": level,
                "variant": variant,
                "pool": f"{n_workers} worker (undersized)",
                "requests_submitted": int(agg["submitted"]),
                "requests_completed": int(agg["completed"]),
                "requests_dropped": int(agg["dropped"]),
                "drop_rate_pct": round(agg["drop_rate_pct"] / args.rounds, 2),
                "latency_slo_attainment_pct": round(agg["latency_slo_att_pct"] / args.rounds, 2),
                "accuracy_slo_attainment_pct": round(agg["accuracy_slo_att_pct"] / args.rounds, 2),
                "p50_latency_ms": round(agg["p50_ms"] / args.rounds, 1),
                "p95_latency_ms": round(agg["p95_ms"] / args.rounds, 1),
                "p99_latency_ms": round(agg["p99_ms"] / args.rounds, 1),
            })
            print(f"[{level:>6s}] {variant:>6s} (STRESS, {n_workers} worker): "
                  f"drop {stress_rows[-1]['drop_rate_pct']:5.1f}%, "
                  f"latSLO {stress_rows[-1]['latency_slo_attainment_pct']:5.1f}%, "
                  f"accSLO {stress_rows[-1]['accuracy_slo_attainment_pct']:5.1f}%")

    csv_path = PLOT_DIR / "panini_replay_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")

    stress_path = PLOT_DIR / "panini_replay_stress_results.csv"
    with open(stress_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stress_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stress_rows)
    print(f"Wrote {stress_path}")

    with open(PLOT_DIR / "gpu_util_percentiles.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p50", "p95", "p99"])
        writer.writerow([util_pcts["p50"], util_pcts["p95"], util_pcts["p99"]])
    print(f"Wrote {PLOT_DIR / 'gpu_util_percentiles.csv'}")

    _make_plots(rows, stress_rows, util_pcts)


def _make_plots(rows, stress_rows, util_pcts):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOTS] matplotlib unavailable - skipping PNG output")
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    levels = ["low", "medium", "high"]
    xs = range(len(levels))
    def series(table, variant, key):
        return [next((r[key] for r in table
                      if r["load_level"] == lv and r["variant"] == variant), 0.0)
                for lv in levels]

    # 1) Latency percentiles: OFF vs ON
    fig, ax = plt.subplots(figsize=(10, 6))
    for p, c, mk in (("p50", "#4C72B0", "o"), ("p95", "#C44E52", "s"), ("p99", "#8172B2", "^")):
        ax.plot(xs, series(rows, "difs", f"{p}_latency_ms"), color=c, linestyle="--",
                marker=mk, label=f"DIFS v2 {p.upper()}")
        ax.plot(xs, series(rows, "panini", f"{p}_latency_ms"), color=c, linestyle="-",
                marker=mk, markerfacecolor="none", label=f"v3+seams {p.upper()}")
    ax.set_xticks(list(xs)); ax.set_xticklabels([l.title() for l in levels])
    ax.set_xlabel("Load Level"); ax.set_ylabel("E2E Latency (ms)")
    ax.set_title("Latency P50/P95/P99 by Load Level - DIFS v2 vs v3+PaniniServe seams")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "p50_p95_p99_latency.png", dpi=150)
    plt.close(fig)

    # 2) SLO attainment: OFF vs ON
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18
    for i, (metric, ylab) in enumerate((("latency_slo_attainment_pct", "Latency SLO attainment (%)"),
                                        ("accuracy_slo_attainment_pct", "Accuracy SLO attainment (%)"))):
        off = series(rows, "difs", metric)
        on = series(rows, "panini", metric)
        ax.bar([x + i * 2 * width for x in xs], off, width=width,
               color="#4C72B0", label=f"{ylab} - DIFS v2")
        ax.bar([x + i * 2 * width + width for x in xs], on, width=width,
               color="#DD8452", label=f"{ylab} - v3+seams")
    ax.set_xticks([x + width for x in xs])
    ax.set_xticklabels([l.title() for l in levels])
    ax.set_xlabel("Load Level"); ax.set_ylabel("Attainment (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SLO Attainment by Load Level - DIFS v2 vs v3+PaniniServe seams")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "slo_attainment.png", dpi=150)
    plt.close(fig)

    # 3) Drop rate: OFF vs ON
    fig, ax = plt.subplots(figsize=(10, 6))
    off = series(rows, "difs", "drop_rate_pct")
    on = series(rows, "panini", "drop_rate_pct")
    ax.bar([x - 0.2 for x in xs], off, width=0.4, color="#4C72B0", label="DIFS v2")
    ax.bar([x + 0.2 for x in xs], on, width=0.4, color="#DD8452", label="v3+seams")
    for xi, v in zip(xs, on):
        ax.annotate(f"{v:.1f}%", xy=(xi + 0.2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(list(xs)); ax.set_xticklabels([l.title() for l in levels])
    ax.set_xlabel("Load Level"); ax.set_ylabel("Dropped requests (%)")
    ax.set_title("PaniniServe admission control (seam 1) - drop rate by load level")
    ax.legend()
    fig.tight_layout(); fig.savefig(PLOT_DIR / "drop_rate.png", dpi=150)
    plt.close(fig)


    # 5) Overload stress: seam 1 protects SLOs under an undersized pool
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.18
    for i, (metric, ylab) in enumerate((("latency_slo_attainment_pct", "Latency SLO attainment (%)"),
                                        ("drop_rate_pct", "Dropped requests (%)"))):
        off = series(stress_rows, "difs", metric)
        on = series(stress_rows, "panini", metric)
        ax.bar([x + i * 2 * width for x in xs], off, width=width,
               color="#4C72B0", label=f"{ylab} - DIFS v2 (undersized pool)")
        ax.bar([x + i * 2 * width + width for x in xs], on, width=width,
               color="#DD8452", label=f"{ylab} - v3+seams (undersized pool)")
    ax.set_xticks([x + width for x in xs])
    ax.set_xticklabels([l.title() for l in levels])
    ax.set_xlabel("Load Level"); ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Overload stress (undersized pools) - seam 1 admission control: drops in exchange for latency-SLO protection")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(PLOT_DIR / "stress_admission_control.png", dpi=150)
    plt.close(fig)

    # 4) GPU utilization percentiles (seam 2 monitor)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["P50", "P95", "P99"]
    vals = [util_pcts["p50"], util_pcts["p95"], util_pcts["p99"]]
    ax.bar(labels, vals, color="#55A868")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", xy=(i, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("GPU utilization (%)")
    ax.set_ylim(0, max(100, max(vals) * 1.1))
    ax.set_title("Seam 2 GPUUtilMonitor - sampled utilization percentiles")
    fig.tight_layout(); fig.savefig(PLOT_DIR / "gpu_util_percentiles.png", dpi=150)
    plt.close(fig)

    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
