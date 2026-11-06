#!/usr/bin/env python3
"""
plot_slo_violation_vs_load.py

Plot SLO violation rate against the exact offered load (requests/minute) and
mark the evaluated integer load at which INFaaS reaches the 5% target.

Input (written by run_baseline_benchmark.py --rates ... --repeats ...):
    <plot_data_baseline>/slo_violation_runs.csv

Outputs:
    <plot_data_baseline>/plots/slo_violation_vs_load.png
    <plot_data_baseline>/slo_violation_summary.csv
    <plot_data_baseline>/slo_violation_exact_load.csv

How to run
    ./run_all_experiments.sh            # menu -> 11 -> 2 or 4
    PYTHONPATH=src python src/plots/plot_slo_violation_vs_load.py
"""
import csv
import math
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[PLOT] matplotlib not available; skipping SLO violation plot.")
    sys.exit(0)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

if os.environ.get("DIFS_PLOT_DATA_DIR"):
    DATA_DIR = Path(os.environ["DIFS_PLOT_DATA_DIR"])
    if DATA_DIR.name != "plot_data_baseline":
        DATA_DIR = DATA_DIR / "plot_data_baseline"
elif os.environ.get("DIFS_LAST_RESULTS_DIR"):
    DATA_DIR = Path(os.environ["DIFS_LAST_RESULTS_DIR"]) / "plot_data_baseline"
elif os.environ.get("DIFS_LIVE_RESULTS_DIR"):
    DATA_DIR = Path(os.environ["DIFS_LIVE_RESULTS_DIR"]) / "plot_data_baseline"
else:
    DATA_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"

CSV_PATH = DATA_DIR / "slo_violation_runs.csv"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = DATA_DIR / "slo_violation_summary.csv"
EXACT_CSV = DATA_DIR / "slo_violation_exact_load.csv"

SYSTEM_ORDER = ["infaas", "difs_v3"]
SYSTEM_STYLES = {
    "infaas": {"color": "#1f77b4", "marker": "D", "label": "INFaaS"},
    "difs_v3": {"color": "#2ca02c", "marker": "o", "label": "DIFS (ours)"},
}
TARGET_PCT = 5.0
TOLERANCE_PP = 0.5


def _read_rows():
    if not CSV_PATH.exists():
        print(f"[PLOT] Missing sweep CSV: {CSV_PATH}")
        print("[PLOT] Run the SLO 5% sweep first (menu 11 -> 2).")
        return []
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            try:
                rate = float(row.get("rate_req_per_min", ""))
                violation = float(row.get("slo_violation_rate", "")) * 100.0
            except (TypeError, ValueError):
                continue
            baseline = (row.get("baseline") or "").strip()
            if not baseline:
                continue
            rows.append({
                "baseline": baseline,
                "rate": rate,
                "violation_pct": violation,
                "seed": row.get("seed", ""),
                "submitted": int(float(row.get("requests_submitted", 0) or 0)),
            })
    return rows


def _aggregate(rows):
    grouped = {}
    for row in rows:
        key = (row["baseline"], row["rate"])
        grouped.setdefault(key, []).append(row["violation_pct"])

    summary = {}
    for (baseline, rate), values in grouped.items():
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
            ci = 1.96 * std / math.sqrt(n)
        else:
            std = 0.0
            ci = 0.0
        summary[(baseline, rate)] = {
            "baseline": baseline,
            "rate": rate,
            "n": n,
            "mean": mean,
            "std": std,
            "ci": ci,
        }
    return summary


def _write_summary(summary):
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "rate_req_per_min", "n", "mean_pct",
                         "std_pct", "ci95_pct"])
        for key in sorted(summary, key=lambda k: (k[0], k[1])):
            s = summary[key]
            writer.writerow([s["baseline"], s["rate"], s["n"],
                             round(s["mean"], 4), round(s["std"], 4),
                             round(s["ci"], 4)])
    print(f"[PLOT] Wrote {SUMMARY_CSV}")


def _select_exact_load(summary):
    inf = sorted((s for (baseline, _), s in summary.items()
                  if baseline == "infaas"), key=lambda s: s["rate"])
    if not inf:
        return None, "no INFaaS points", None

    within = [s for s in inf if abs(s["mean"] - TARGET_PCT) <= TOLERANCE_PP]
    if within:
        chosen = min(within, key=lambda s: abs(s["mean"] - TARGET_PCT))
        reason = f"within +/-{TOLERANCE_PP:g} pp of {TARGET_PCT:g}%"
    else:
        chosen = min(inf, key=lambda s: abs(s["mean"] - TARGET_PCT))
        reason = "closest measured rate (no point within tolerance)"

    crossing = None
    for left, right in zip(inf, inf[1:]):
        if (left["mean"] - TARGET_PCT) * (right["mean"] - TARGET_PCT) <= 0 \
                and left["mean"] != right["mean"]:
            crossing = left["rate"] + (
                (TARGET_PCT - left["mean"]) * (right["rate"] - left["rate"])
                / (right["mean"] - left["mean"])
            )
            break
    return chosen, reason, crossing


def _nearest(summary, baseline, rate):
    candidates = [s for (b, _), s in summary.items() if b == baseline]
    if not candidates:
        return None
    return min(candidates, key=lambda s: abs(s["rate"] - rate))


def _plot(summary, exact, exact_reason, crossing):
    if not summary:
        return

    fig, (ax, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [2.3, 1.0]}
    )

    for baseline in SYSTEM_ORDER:
        points = sorted((s for (b, _), s in summary.items() if b == baseline),
                        key=lambda s: s["rate"])
        if not points:
            continue
        style = SYSTEM_STYLES[baseline]
        xs = [p["rate"] for p in points]
        ys = [p["mean"] for p in points]
        errs = [p["ci"] for p in points]
        ax.errorbar(xs, ys, yerr=errs, color=style["color"],
                    marker=style["marker"], linewidth=2.2, markersize=8,
                    capsize=4, label=style["label"])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)

    ax.axhline(TARGET_PCT, color="#d62728", linestyle="--", linewidth=1.6,
               label=f"{TARGET_PCT:g}% target")

    if exact is not None:
        ax.axvline(exact["rate"], color="#555555", linestyle=":", linewidth=1.6)
        difs = _nearest(summary, "difs_v3", exact["rate"])
        text = (f"Exact load = {exact['rate']:g} req/min\n"
                f"INFaaS = {exact['mean']:.2f}%")
        if difs is not None:
            text += f"\nDIFS = {difs['mean']:.2f}%"
        ax.annotate(text, xy=(exact["rate"], exact["mean"]),
                    xytext=(14, -44), textcoords="offset points",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.35",
                                          fc="white", ec="#888888", alpha=0.95))
    if crossing is not None:
        ax.axvline(crossing, color="#999999", linestyle="-.", linewidth=1.2,
                   alpha=0.8)
        ax.annotate(f"interpolated crossing {crossing:.2f} req/min",
                    xy=(crossing, TARGET_PCT), xytext=(6, -28),
                    textcoords="offset points", fontsize=8, color="#555555")

    ax.set_xlabel("Input load (requests/minute)", fontsize=12)
    ax.set_ylabel("SLO violation rate (%)", fontsize=12)
    ax.set_title("SLO Violation vs Exact Input Load", fontsize=13,
                 fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(fontsize=9, loc="best")

    if exact is not None:
        bars = []
        labels = []
        colors = []
        for baseline in SYSTEM_ORDER:
            point = _nearest(summary, baseline, exact["rate"])
            if point is None:
                continue
            bars.append(point["mean"])
            labels.append(SYSTEM_STYLES[baseline]["label"])
            colors.append(SYSTEM_STYLES[baseline]["color"])
        xpos = list(range(len(bars)))
        ax_bar.bar(xpos, bars, color=colors, width=0.6)
        ax_bar.axhline(TARGET_PCT, color="#d62728", linestyle="--",
                       linewidth=1.4)
        for x, value in zip(xpos, bars):
            ax_bar.annotate(f"{value:.2f}%", xy=(x, value), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=9)
        ax_bar.set_xticks(xpos)
        ax_bar.set_xticklabels(labels, fontsize=9)
        ax_bar.set_ylim(0, max(TARGET_PCT * 1.5, max(bars) * 1.25))
        ax_bar.set_ylabel("SLO violation rate (%)", fontsize=11)
        ax_bar.set_title(f"At {exact['rate']:g} req/min", fontsize=11,
                         fontweight="bold")
        ax_bar.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    out = PLOT_DIR / "slo_violation_vs_load.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out}")

    if exact is not None:
        with open(EXACT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["baseline", "rate_req_per_min", "n",
                             "mean_pct", "ci95_pct", "selection_rule",
                             "interpolated_crossing_req_per_min"])
            for baseline in SYSTEM_ORDER:
                point = _nearest(summary, baseline, exact["rate"])
                if point is None:
                    continue
                writer.writerow([baseline, point["rate"], point["n"],
                                 round(point["mean"], 4), round(point["ci"], 4),
                                 exact_reason,
                                 round(crossing, 4) if crossing is not None else ""])
        print(f"[PLOT] Wrote {EXACT_CSV}")
        print(f"[PLOT] Exact INFaaS 5% load: {exact['rate']:g} req/min "
              f"({exact['mean']:.2f}%, {exact_reason})")
        difs = _nearest(summary, "difs_v3", exact["rate"])
        if difs is not None:
            print(f"[PLOT] DIFS at {exact['rate']:g} req/min: "
                  f"{difs['mean']:.2f}% (target <= {TARGET_PCT:g}%)")
        if crossing is not None:
            print(f"[PLOT] Interpolated continuous crossing: "
                  f"{crossing:.2f} req/min")


def main():
    rows = _read_rows()
    if not rows:
        return 0
    summary = _aggregate(rows)
    _write_summary(summary)
    exact, reason, crossing = _select_exact_load(summary)
    _plot(summary, exact, reason, crossing)
    if exact is not None and crossing is None:
        print("[PLOT] WARNING: the 5% target is not bracketed by the evaluated "
              "rates.")
        print("[PLOT] Re-run the sweep with 1-2 adjacent integer rates "
              "(DIFS_SLO_SWEEP_RATES).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
