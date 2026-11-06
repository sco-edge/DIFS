import csv
from pathlib import Path

BASE = Path('/mnt/ssd4t/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v7/DiffServ_0112_1730_coco2/0112_1730_coco')
OUT = Path('/mnt/ssd4t/kwadwo/Desktop/WORK/PROGRAMMING_WORLD/PROJECTS_RESEARCH/Templates/DIFS_OC_v7/MaterialForEvaluator/csv_exports')

def table1_from_baseline_comparison(src, dst, label):
    if not src.exists(): return
    rows = list(csv.DictReader(src.open()))
    if not rows: return
    idx = {(r['baseline'],r['load_level'],r['steps']): r for r in rows}
    pairs = sorted({(r['load_level'],r['steps']) for r in rows}, key=lambda x:(str(x[0]), int(x[1]) if str(x[1]).isdigit() else 0))
    with dst.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['run','quality_slo_load_level','step_size','INFaaS_latency_slo_attainment_pct','PaniniServe_DIFS_latency_slo_attainment_pct','INFaaS_accuracy_slo_attainment_pct','PaniniServe_DIFS_accuracy_slo_attainment_pct','source'])
        for load,step in pairs:
            ir=idx.get(('infaas',load,step),{}); dr=idx.get(('difs_v3',load,step),{})
            if ir or dr:
                w.writerow([label,load,step,ir.get('latency_slo_attainment_pct',''),dr.get('latency_slo_attainment_pct',''),ir.get('accuracy_slo_attainment_pct',''),dr.get('accuracy_slo_attainment_pct',''),str(src)])

def table1_from_sweep(src, dst, label):
    if not src.exists(): return
    rows=list(csv.DictReader(src.open()))
    if not rows: return
    with dst.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['run','baseline','arrival_rate_req_per_min','load_level','latency_slo_attainment_pct','accuracy_slo_attainment_pct','slo_violation_rate','requests_completed','requests_dropped','source'])
        for r in rows:
            w.writerow([label,r['baseline'],r.get('rate_req_per_min',''),r.get('load_level',''),r.get('latency_slo_attainment_pct',''),r.get('accuracy_slo_attainment_pct',''),r.get('slo_violation_rate',''),r.get('requests_completed',''),r.get('requests_dropped',''),str(src)])

def table2_from_per_request(src, dst, label):
    if not src or not src.exists(): return False
    rows=list(csv.DictReader(src.open()))
    if not rows: return False
    with dst.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['run','request_index','baseline','load_level_or_rate','end_to_end_latency_ms','latency_slo_ms','latency_slo_status','accuracy_level','step_size','source'])
        for r in rows:
            w.writerow([label,r.get('request_index',''),r.get('baseline',''),r.get('load_level_or_rate',r.get('load_level','')),r.get('end_to_end_latency_ms',''),r.get('latency_slo_ms',''),r.get('latency_slo_status',''),r.get('accuracy_level',''),r.get('step_size',''),str(src)])
    return True

# Table 1 (B14): step/latency SLO relationship
table1_from_baseline_comparison(BASE/'results/model_benchmark_live/plot_data_baseline/baseline_comparison.csv', OUT/'Table1_full_live_baseline_comparison.csv','full_live(option10-3/experiments)')
table1_from_baseline_comparison(BASE/'results/model_benchmark_last/plot_data_baseline/baseline_comparison.csv', OUT/'Table1_full_last_baseline_comparison.csv','full_last')
table1_from_sweep(BASE/'results/model_benchmark_fast/plot_data_baseline/slo_violation_runs.csv', OUT/'Table1_fast_option10-5_slo_sweep.csv','fast_option10-5')
table1_from_sweep(BASE/'results/model_benchmark_live/plot_data_baseline/slo_violation_runs.csv', OUT/'Table1_full_live_slo_sweep.csv','full_live_slo_sweep(option10-3)')

# Table 2 (B15): per-request E2E latency + latency SLO
made={}
for name,src,label in [
 ('Table2_fast_option10-5_per_request.csv', BASE/'results/model_benchmark_fast/plot_data_baseline/per_request_all.csv','fast_option10-5'),
 ('Table2_full_live_per_request.csv', BASE/'results/model_benchmark_live/plot_data_baseline/per_request_all.csv','full_live'),
 ('Table2_full_last_per_request.csv', BASE/'results/model_benchmark_last/plot_data_baseline/per_request_all.csv','full_last'),
]:
    made[name]=table2_from_per_request(src, OUT/name, label)

# pending markers if missing
for k,v in made.items():
    if not v and not (OUT/k).exists():
        with (OUT/k.replace('.csv','_PENDING.csv')).open('w',newline='') as f:
            w=csv.writer(f); w.writerow(['status','note']); w.writerow(['pending','No per_request_all.csv yet for this run; run option 10 via run_all_experiments.sh with instrumented benchmark.'])

print('exported', sorted(p.name for p in OUT.glob('Table*')))
