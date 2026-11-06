#!/usr/bin/env python3
"""
compile_report_v6.py

Populates DIFS_OC_v6_WorkloadComparison_Report.tex with real numbers from
DIFS_OC_v6's own baseline_comparison.csv (DiffusionDB workload), reads v4's
baseline_comparison.csv (COCO workload) as a read-only reference for the
comparison section, generates plots (v6 mirrors + comparison overlays), and
compiles the PDF.

How to run
    Recommended (also regenerates every figure first):
        cd Templates/DIFS_OC_v6/DiffServ_0112_1730_coco2/0112_1730_coco
        ./run_all_experiments.sh --plots --report

    Directly, from this project directory:
        PYTHONPATH=src python compile_report_v6.py

    With a specific interpreter / v4 reference:
        PYTHONPATH=src /path/to/env/bin/python compile_report_v6.py
        V4_CSV=/path/to/DIFS_OC_v4/.../baseline_comparison.csv \\
            PYTHONPATH=src python compile_report_v6.py

    Inputs:
        results/model_benchmark/plot_data_baseline/baseline_comparison.csv
        results/model_benchmark/plot_data_baseline/switch_*/pairwise_switching.csv
        metrics/model_load_metrics.csv
        v4 baseline_comparison.csv (auto-detected or V4_CSV)

    Output:
        results/model_benchmark/Reports/DIFS_OC_v6_WorkloadComparison_Report.pdf

    Notes:
        * Run --plots first (or use --plots --report) so every figure exists.
        * v4's CSV is read-only; v6 results are never copied from v4.
        * pdflatex must be on PATH (/usr/bin, /bin, /usr/local/bin); the PDF
          step is skipped if TeX or a referenced figure is missing.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "plot_data_baseline"
V6_CSV = DATA_DIR / "baseline_comparison.csv"
REPORT_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "Reports"
TEX_FILE = REPORT_DIR / "DIFS_OC_v6_WorkloadComparison_Report.tex"
# v4 reference CSV (read-only); overridable via env
V4_CSV = Path(os.environ.get(
    "V4_CSV",
    str(PROJECT_ROOT.parent.parent.parent / "DIFS_OC_v4"
        / "DiffServ_0112_1730_coco2" / "0112_1730_coco" / "results"
        / "model_benchmark" / "plot_data_baseline" / "baseline_comparison.csv")))

SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_LABELS = {
    "clipper_large": "Clipper Large-Only",
    "clipper_small": "Clipper Small-Only",
    "infaas": "INFaaS",
    "difs_v3": "\\textbf{DIFS\\_v3}",
}
LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]
STEP_ORDER = [10, 20, 30, 40]


def load_csv(path):
    if not path.exists():
        print(f"[COMPILE] CSV not found: {path}")
        return []
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


def _lookup(rows, sys_name, load, steps):
    for r in rows:
        if (r["baseline"] == sys_name and r["load_level"] == load
                and r["steps"] == steps):
            return r
    return None


def build_4step_rows(rows, metric_key, fmt="{:.1f}"):
    lines = []
    for sys_name in SYSTEM_ORDER:
        for load in LOAD_ORDER:
            cols = []
            for steps in STEP_ORDER:
                r = _lookup(rows, sys_name, load, steps)
                if r:
                    cols.append(fmt.format(r[metric_key]))
                else:
                    cols.append("---")
            lines.append(f"{SYSTEM_LABELS[sys_name]}  & {load:8s} "
                         f"& {' & '.join(cols)} \\\\")
    return "\n".join(lines)


def build_completion_4step_rows(rows):
    lines = []
    for sys_name in SYSTEM_ORDER:
        for load in LOAD_ORDER:
            cols = []
            for steps in STEP_ORDER:
                r = _lookup(rows, sys_name, load, steps)
                if r:
                    comp = r["requests_completed"]
                    sub = r["requests_submitted"]
                    pct = 100.0 * comp / sub if sub else 0.0
                    cols.append(f"{comp}/{sub}\\ ({pct:.0f}\\%)")
                else:
                    cols.append("---")
            lines.append(f"{SYSTEM_LABELS[sys_name]}  & {load:8s} "
                         f"& {' & '.join(cols)} \\\\")
    return "\n".join(lines)


def _sys_means(rows, sys_name):
    subset = [r for r in rows if r["baseline"] == sys_name]
    if not subset:
        return None
    lat = sum(r["latency_slo_attainment_pct"] for r in subset) / len(subset)
    acc = sum(r["accuracy_slo_attainment_pct"] for r in subset) / len(subset)
    thr = sum(r["throughput_req_per_s"] for r in subset) / len(subset)
    return lat, acc, thr


def build_summary_cmp(rows_v4, rows_v6):
    """tab:summary_cmp executive table rows."""
    lines = []
    for sys_name in SYSTEM_ORDER:
        v4 = _sys_means(rows_v4, sys_name)
        v6 = _sys_means(rows_v6, sys_name)
        if not v6:
            continue
        v4_lat = f"{v4[0]:.1f}" if v4 else "---"
        v4_thr = f"{v4[2]:.4f}" if v4 else "---"
        lines.append(
            f"{SYSTEM_LABELS[sys_name]} & {v4_lat} & {v6[0]:.1f} "
            f"& {v4_thr} & {v6[2]:.4f} & {v6[1]:.1f} \\\\")
    return "\n".join(lines)


def build_agg_cmp(rows_v4, rows_v6):
    """tab:agg_cmp rows: one row per (workload, system)."""
    lines = []
    for (tag, rows) in [("COCO", rows_v4), ("DiffusionDB", rows_v6)]:
        for sys_name in SYSTEM_ORDER:
            m = _sys_means(rows, sys_name)
            if not m:
                continue
            lines.append(f"{tag} & {SYSTEM_LABELS[sys_name]} & {m[0]:.1f} "
                         f"& {m[2]:.4f} \\\\")
    return "\n".join(lines)


def fill_table(tex, label, new_rows):
    pattern = (
        rf"(\\label\{{{label}\}}.*?\\midrule\s*\n)"
        r"(.*?)(\\bottomrule)"
    )
    def replacer(m, nr=new_rows):
        return m.group(1) + "\n" + nr + "\n" + m.group(3)
    tex, n = re.subn(pattern, replacer, tex, count=1, flags=re.DOTALL)
    if n:
        print(f"[COMPILE] Populated table {label}")
    else:
        print(f"[COMPILE] WARNING: table {label} not found in tex")
    return tex


def populate_tex(rows_v6, rows_v4):
    tex = TEX_FILE.read_text()

    if rows_v6:
        tex = fill_table(tex, "tab:summary_cmp",
                         build_summary_cmp(rows_v4, rows_v6))
        tex = fill_table(tex, "tab:latency_v6",
                         build_4step_rows(rows_v6, "latency_slo_attainment_pct"))
        tex = fill_table(tex, "tab:accuracy_v6",
                         build_4step_rows(rows_v6, "accuracy_slo_attainment_pct"))
        tex = fill_table(tex, "tab:throughput_v6",
                         build_4step_rows(rows_v6, "throughput_req_per_s", "{:.4f}"))
        tex = fill_table(tex, "tab:completion_v6",
                         build_completion_4step_rows(rows_v6))
        tex = fill_table(tex, "tab:agg_cmp", build_agg_cmp(rows_v4, rows_v6))
    else:
        print("[COMPILE] No v6 rows; leaving result tables as placeholders.")

    # verdict/interpretation paragraphs described in chapter 5 & 6 were written
    # manually during analysis; report compile fills figures only.
    TEX_FILE.write_text(tex)
    print(f"[COMPILE] Wrote populated {TEX_FILE}")


def generate_plots():
    scripts = [
        PROJECT_ROOT / "src" / "plots" / "plot_baseline_comparison.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switching_overhead.py",
        PROJECT_ROOT / "src" / "plots" / "plot_workload_comparison.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switch_modes.py",
        PROJECT_ROOT / "src" / "plots" / "plot_switch_slo.py",
    ]
    for s in scripts:
        if not s.exists():
            print(f"[COMPILE] Skipping missing plot script {s.name}")
            continue
        env = os.environ.copy()
        print(f"[COMPILE] Running {s.name}...")
        subprocess.run(
            [sys.executable, str(s)], cwd=str(PROJECT_ROOT), env=env)


def compile_pdf():
    print("[COMPILE] Compiling PDF (two passes)...")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(Path.home())}
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(TEX_FILE.name)],
            cwd=str(REPORT_DIR), env=env,
            capture_output=True, text=True)
        if result.returncode != 0 and i == 1:
            log_file = REPORT_DIR / TEX_FILE.name.replace(".tex", ".log")
            if log_file.exists():
                lines = log_file.read_text().splitlines()
                for line in lines[-25:]:
                    print(f"  {line}")
    pdf = REPORT_DIR / TEX_FILE.name.replace(".tex", ".pdf")
    if pdf.exists():
        print(f"[COMPILE] PDF generated: {pdf} ({pdf.stat().st_size/1e6:.2f} MB)")
    else:
        print("[COMPILE] ERROR: PDF not generated. Check log.")


def main():
    rows_v6 = load_csv(V6_CSV)
    rows_v4 = load_csv(V4_CSV)
    print(f"[COMPILE] v6 rows={len(rows_v6)} (DiffusionDB), "
          f"v4 rows={len(rows_v4)} (COCO)")
    if not rows_v6:
        print("[COMPILE] ERROR: v6 CSV missing. Run benchmark first.")
        sys.exit(1)

    populate_tex(rows_v6, rows_v4)
    generate_plots()
    compile_pdf()


if __name__ == "__main__":
    main()