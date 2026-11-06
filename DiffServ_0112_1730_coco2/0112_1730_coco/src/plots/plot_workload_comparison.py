"""
plot_workload_comparison.py

DIFS_OC_v6 workload-comparison analysis + plots.

Reads v6's own baseline_comparison.csv (DiffusionDB workload) and v4's
baseline_comparison.csv (COCO workload, read-only reference) and produces
side-by-side comparison figures plus a summary CSV used by the report:

  1. comparison_summary.csv   per-system aggregates (lat/acc/throughput/completion)
                                  for BOTH workloads, one row per workload
  2. latency_slo_coco_vs_diffusiondb.png    grouped bars per system x load
  3. throughput_coco_vs_diffusiondb.png     grouped bars per system x load
  4. latency_by_steps_overlay.png           latency SLO vs steps, both workloads
  5. throughput_latency_overlay.png         scatter both workloads

Provenance: this script never copies v4's numbers into v6's results — it reads
both CSVs read-only and writes outputs into v6's own plot_data_baseline.

How to run
    Recommended (with the other v6 figures):
        ./run_all_experiments.sh --plots

    Directly, from the project root:
        PYTHONPATH=src python src/plots/plot_workload_comparison.py

    Point at a different v4 reference:
        V4_CSV=/path/to/DIFS_OC_v4/.../baseline_comparison.csv \\
            PYTHONPATH=src python src/plots/plot_workload_comparison.py

    If V4_CSV is unset and the default v4 path does not exist, the overlay
    figures are skipped and only the v6 summary is written.
"""
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[WC-PLOT] matplotlib not available; skipping plots.")
    sys.exit(0)

import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# Allow overriding data paths for live/last runs
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
DATA_DIR.mkdir(parents=True, exist_ok=True)
V6_CSV = DATA_DIR / "baseline_comparison.csv"
# v4 reference CSV (read-only); overridable via env for portability
V4_CSV_DEFAULT = PROJECT_ROOT.parent.parent.parent / "DIFS_OC_v4" \
    / "DiffServ_0112_1730_coco2" / "0112_1730_coco" / "results" \
    / "model_benchmark" / "plot_data_baseline" / "baseline_comparison.csv"
PLOT_DIR = DATA_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY = DATA_DIR / "comparison_summary.csv"

SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_LABELS = {
    "clipper_large": "Clipper Large-Only",
    "clipper_small": "Clipper Small-Only",
    "infaas": "INFaaS",
    "difs_v3": "DIFS_OC_v3 (Ours)",
}
LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]
STEP_ORDER = [10, 20, 30, 40]

# styling: v6 = filled, v4 = hatched gray
V6_STYLE = {"color": "#2ca02c", "marker": "o"}
V4_STYLE = {"color": "#7f7f7f", "marker": "s"}


def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["steps"] = int(r["steps"])
            r["latency_slo_attainment_pct"] = float(r["latency_slo_attainment_pct"])
            r["accuracy_slo_attainment_pct"] = float(r["accuracy_slo_attainment_pct"])
            r["throughput_req_per_s"] = float(r["throughput_req_per_s"])
            r["requests_submitted"] = int(r.get("requests_submitted", 0))
            r["requests_completed"] = int(r.get("requests_completed", 0))
            r["requests_dropped"] = int(r.get("requests_dropped", 0))
            rows.append(r)
    return rows


def cell_aggregate(rows, bname, load, metric_key, agg="mean"):
    cell = [r for r in rows
            if r["baseline"] == bname and r["load_level"] == load]
    if not cell:
        return None
    vals = [r[metric_key] for r in cell]
    if agg == "mean":
        return sum(vals) / len(vals)
    return vals


def group_bars(rows_v4, rows_v6, metric_key, out_path, ylabel, title,
               v4_fmt="{:.1f}", v6_fmt="{:.1f}"):
    n_systems = len(SYSTEM_ORDER)
    n_loads = len(LOAD_ORDER)
    x = np.arange(n_loads)
    width = 0.9 / (2 * n_systems)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, bname in enumerate(SYSTEM_ORDER):
        v4_vals = [cell_aggregate(rows_v4, bname, load, metric_key) for load in LOAD_ORDER]
        v6_vals = [cell_aggregate(rows_v6, bname, load, metric_key) for load in LOAD_ORDER]
        offset = (i - n_systems / 2) * width * 2
        v4_pos = x + offset - width
        v6_pos = x + offset + width
        ax.bar(v6_pos, [v if v is not None else 0 for v in v6_vals],
               width=width, color=V6_STYLE["color"], edgecolor="black",
               linewidth=0.5, hatch="//", label=f"{SYSTEM_LABELS[bname]} (DiffusionDB)")
        ax.bar(v4_pos, [v if v is not None else 0 for v in v4_vals],
               width=width, color=V4_STYLE["color"], edgecolor="black",
               linewidth=0.5, hatch="..", label=f"{SYSTEM_LABELS[bname]} (COCO)")

    ax.set_xticks(x)
    ax.set_xticklabels(LOAD_ORDER, fontsize=11)
    ax.set_xlabel("Load Level", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[WC-PLOT] Saved {out_path}")


def plot_latency_by_steps_overlay(rows_v4, rows_v6, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for bname in SYSTEM_ORDER:
        for (src_rows, style, tag) in [(rows_v4, V4_STYLE, "COCO"),
                                       (rows_v6, V6_STYLE, "DiffusionDB")]:
            subset = [r for r in src_rows if r["baseline"] == bname]
            if not subset:
                continue
            xs, ys = [], []
            for steps in STEP_ORDER:
                cell = [r for r in subset if r["steps"] == steps]
                if cell:
                    xs.append(steps)
                    ys.append(sum(r["latency_slo_attainment_pct"] for r in cell) / len(cell))
            ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                    linewidth=1.2, markersize=5,
                    label=f"{SYSTEM_LABELS[bname]} ({tag})")
    ax.set_xticks(STEP_ORDER)
    ax.set_xlabel("Step Count", fontsize=12)
    ax.set_ylabel("Latency SLO Attainment (%, mean over loads)", fontsize=12)
    ax.set_title("Latency SLO Attainment vs Step Count: COCO vs DiffusionDB", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[WC-PLOT] Saved {out_path}")


def plot_throughput_latency_overlay(rows_v4, rows_v6, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for bname in SYSTEM_ORDER:
        for (src_rows, style, tag) in [(rows_v4, V4_STYLE, "COCO"),
                                       (rows_v6, V6_STYLE, "DiffusionDB")]:
            subset = [r for r in src_rows if r["baseline"] == bname]
            if not subset:
                continue
            for r in subset:
                ax.scatter(r["throughput_req_per_s"],
                           r["latency_slo_attainment_pct"],
                           c=style["color"], marker=style["marker"],
                           s=45, edgecolors="black", linewidths=0.4,
                           alpha=0.8, zorder=3,
                           label=f"{SYSTEM_LABELS[bname]} ({tag})")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              fontsize=8, ncol=3, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, -0.06))
    ax.set_xlabel("Throughput (req/s)", fontsize=12)
    ax.set_ylabel("Latency SLO Attainment (%)", fontsize=12)
    ax.set_title("Throughput\u2013Latency SLO: COCO vs DiffusionDB", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[WC-PLOT] Saved {out_path}")


def write_summary(rows_v4, rows_v6):
    """Per-system aggregates for both workloads -> comparison_summary.csv."""
    with open(OUT_SUMMARY, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "workload", "system", "n_rows",
            "latency_slo_mean_pct", "accuracy_slo_mean_pct",
            "throughput_mean_req_per_s", "completion_mean_pct",
        ])
        writer.writeheader()
        for (tag, rows) in [("coco", rows_v4), ("diffusiondb", rows_v6)]:
            for bname in SYSTEM_ORDER:
                subset = [r for r in rows if r["baseline"] == bname]
                if not subset:
                    continue
                lat = sum(r["latency_slo_attainment_pct"] for r in subset) / len(subset)
                acc = sum(r["accuracy_slo_attainment_pct"] for r in subset) / len(subset)
                thr = sum(r["throughput_req_per_s"] for r in subset) / len(subset)
                comp = [100.0 * r["requests_completed"] / r["requests_submitted"]
                        for r in subset if r["requests_submitted"]]
                comp_mean = sum(comp) / len(comp) if comp else 0.0
                writer.writerow({
                    "workload": tag, "system": bname, "n_rows": len(subset),
                    "latency_slo_mean_pct": round(lat, 2),
                    "accuracy_slo_mean_pct": round(acc, 2),
                    "throughput_mean_req_per_s": round(thr, 4),
                    "completion_mean_pct": round(comp_mean, 2),
                })
        print(f"[WC-PLOT] Wrote {OUT_SUMMARY}")


def main():
    import os
    if not V6_CSV.exists():
        print(f"[WC-PLOT] v6 CSV not found yet: {V6_CSV}")
        sys.exit(1)
    v4_csv = Path(os.environ.get("V4_CSV", str(V4_CSV_DEFAULT)))
    if not v4_csv.exists():
        print(f"[WC-PLOT] v4 CSV not found: {v4_csv}. "
              f"Set V4_CSV env var. Will only write summary from v6.")
        rows_v4 = []
    else:
        rows_v4 = load_csv(v4_csv)
    rows_v6 = load_csv(V6_CSV)
    print(f"[WC-PLOT] v6(v6)={len(rows_v6)} rows, v4(COCO)={len(rows_v4)} rows")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    write_summary(rows_v4, rows_v6)

    if rows_v4:
        group_bars(rows_v4, rows_v6, "latency_slo_attainment_pct",
                   PLOT_DIR / "latency_slo_coco_vs_diffusiondb.png",
                   "Latency SLO Attainment (%)",
                   "Latency SLO Attainment: COCO vs DiffusionDB")
        group_bars(rows_v4, rows_v6, "throughput_req_per_s",
                   PLOT_DIR / "throughput_coco_vs_diffusiondb.png",
                   "Throughput (req/s)",
                   "Throughput: COCO vs DiffusionDB", v4_fmt="{:.4f}", v6_fmt="{:.4f}")
        plot_latency_by_steps_overlay(rows_v4, rows_v6,
                                      PLOT_DIR / "latency_by_steps_overlay.png")
        plot_throughput_latency_overlay(rows_v4, rows_v6,
                                        PLOT_DIR / "throughput_latency_overlay.png")


if __name__ == "__main__":
    main()