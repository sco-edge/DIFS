#!/usr/bin/env python3
"""
compile_report.py

Reads baseline_comparison.csv (per-step rows), populates
DIFS_OC_v4_Benchmark_Report.tex tables with genuine per-step values,
generates plots, and compiles to PDF.

Usage:
    cd Templates/DIFS_OC_v4/DiffServ_0112_1730_coco2/0112_1730_coco
    /home/kwadwo/miniforge3/envs/DIFS_Gen/bin/python compile_report.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = (PROJECT_ROOT / "results" / "model_benchmark"
            / "plot_data_baseline" / "baseline_comparison.csv")
REPORT_DIR = PROJECT_ROOT / "results" / "model_benchmark" / "Reports"
TEX_FILE = REPORT_DIR / "DIFS_OC_v4_Benchmark_Report.tex"
TEX_BACKUP = REPORT_DIR / "DIFS_OC_v4_Benchmark_Report.tex.bak"

SYSTEM_ORDER = ["clipper_large", "clipper_small", "infaas", "difs_v3"]
SYSTEM_LABELS = {
    "clipper_large": "Clipper Large-Only",
    "clipper_small": "Clipper Small-Only",
    "infaas":        "INFaaS",
    "difs_v3":       "\\textbf{DIFS\\_v3}",
}
LOAD_ORDER = ["low", "medium", "high", "saturation", "stress"]
STEP_ORDER = [10, 20, 30, 40]


def load_csv():
    if not CSV_PATH.exists():
        print(f"[COMPILE] CSV not found: {CSV_PATH}")
        sys.exit(1)
    rows = []
    with open(CSV_PATH) as f:
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
    """Find the row for (sys_name, load, steps), return None if not found."""
    for r in rows:
        if r["baseline"] == sys_name and r["load_level"] == load and r["steps"] == steps:
            return r
    return None


def build_4step_rows(rows, metric_key, fmt="{:.1f}"):
    """Build rows for a per-step table: System | Load | Steps10 | Steps20 | Steps30 | Steps40."""
    lines = []
    for sys_name in SYSTEM_ORDER:
        for load in LOAD_ORDER:
            label = SYSTEM_LABELS[sys_name]
            cols = []
            for steps in STEP_ORDER:
                r = _lookup(rows, sys_name, load, steps)
                if r:
                    cols.append(fmt.format(r[metric_key]))
                else:
                    cols.append("---")
            lines.append(f"{label}  & {load:8s} & {' & '.join(cols)} \\\\")
    return "\n".join(lines)


def build_completion_4step_rows(rows):
    """Build rows: System | Load | C/S@10 | C/S@20 | C/S@30 | C/S@40."""
    lines = []
    for sys_name in SYSTEM_ORDER:
        for load in LOAD_ORDER:
            label = SYSTEM_LABELS[sys_name]
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
            lines.append(f"{label}  & {load:8s} & {' & '.join(cols)} \\\\")
    return "\n".join(lines)


def build_summary_rows(rows):
    """Build the executive summary table rows (averaged across loads and steps)."""
    OBSERVATIONS = {
        "clipper_large": "Single large model, no adaptation",
        "clipper_small": "Fast but lower quality",
        "infaas": "Cost-driven model selection",
        "difs_v3": "EDF + fair-share adaptation",
    }
    lines = []
    for sys_name in SYSTEM_ORDER:
        sys_rows = [r for r in rows if r["baseline"] == sys_name]
        if not sys_rows:
            continue
        label = SYSTEM_LABELS[sys_name]
        avg_lat = sum(r["latency_slo_attainment_pct"] for r in sys_rows) / len(sys_rows)
        avg_acc = sum(r["accuracy_slo_attainment_pct"] for r in sys_rows) / len(sys_rows)
        avg_thru = sum(r["throughput_req_per_s"] for r in sys_rows) / len(sys_rows)
        obs = OBSERVATIONS.get(sys_name, "")
        lines.append(
            f"{label} & {avg_lat:.1f}\\% & {avg_acc:.1f}\\% "
            f"& {avg_thru:.4f} & {obs} \\\\"
        )
    return "\n".join(lines)


def populate_tex(rows):
    if not TEX_BACKUP.exists():
        import shutil
        shutil.copy2(TEX_FILE, TEX_BACKUP)
        print(f"[COMPILE] Backed up {TEX_FILE} -> {TEX_BACKUP}")

    tex = TEX_FILE.read_text()

    replacements = {
        "tab:latency": build_4step_rows(rows, "latency_slo_attainment_pct", "{:.1f}"),
        "tab:accuracy": build_4step_rows(rows, "accuracy_slo_attainment_pct", "{:.1f}"),
        "tab:throughput": build_4step_rows(rows, "throughput_req_per_s", "{:.4f}"),
        "tab:completion": build_completion_4step_rows(rows),
        "tab:summary": build_summary_rows(rows),
    }

    for label, new_rows in replacements.items():
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

    TEX_FILE.write_text(tex)
    print(f"[COMPILE] Wrote populated {TEX_FILE}")


def generate_plots():
    print("[COMPILE] Generating plots...")
    plot_script = PROJECT_ROOT / "src" / "plots" / "plot_baseline_comparison.py"
    subprocess.run(
        [sys.executable, str(plot_script)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def compile_pdf():
    print("[COMPILE] Compiling PDF (two passes)...")
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    for i in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", str(TEX_FILE.name)],
            cwd=str(REPORT_DIR),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and i == 1:
            print(f"[COMPILE] WARNING: pdflatex pass {i+1} returned {result.returncode}")
            log_file = REPORT_DIR / "DIFS_OC_v4_Benchmark_Report.log"
            if log_file.exists():
                lines = log_file.read_text().splitlines()
                for line in lines[-20:]:
                    print(f"  {line}")

    pdf = REPORT_DIR / "DIFS_OC_v4_Benchmark_Report.pdf"
    if pdf.exists():
        print(f"[COMPILE] PDF generated: {pdf}")
    else:
        print("[COMPILE] ERROR: PDF not generated. Check log.")


def main():
    rows = load_csv()
    print(f"[COMPILE] Loaded {len(rows)} per-step rows from CSV")

    populate_tex(rows)
    generate_plots()
    compile_pdf()


if __name__ == "__main__":
    main()
