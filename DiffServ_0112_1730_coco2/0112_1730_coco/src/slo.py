#!/usr/bin/env python3

"""
slo.py - Service Level Objective support for DIFS

This module handles:
1. Accuracy SLO level mapping to step count
2. Auto-configured latency SLO = 2.5x execution time
3. SLO evaluation logic
"""

import csv
import os
from pathlib import Path
from typing import Dict, Tuple

# Root-relative path resolution so slo.py works from any CWD
try:
    from project_paths import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default configuration for SLOs
SLO_LATENCY_MULTIPLIER = 2.5  # 2 to 3x execution time
ACCURACY_STEP_RANGES = {
    "high": (30, 40),   # 30-40 denoising steps
    "medium": (20, 30), # 20-30 denoising steps  
    "low": (10, 20)     # 10-20 denoising steps
}

# Map accuracy levels to canonical step counts
ACCURACY_STEP_MAP = {
    "high": 40,
    "medium": 30, 
    "low": 20
}

def load_exec_profiling_data() -> Dict[Tuple[int, int], float]:
    """
    Load execution time profiling data from CSV.
    Returns dict of (model_id, steps) -> exec_time_sec
    """
    # Root-relative path to the profiling CSV
    profiling_file = PROJECT_ROOT / "results/model_benchmark/plot_data_5model/slo_exec_profiling.csv"
    
    data = {}
    
    if not profiling_file.exists():
        print(f"[SLO] Warning: Profiling file {profiling_file} not found. Using estimated values.")
        # Return estimated values based on existing benchmark (all models at steps=10)
        return estimate_from_benchmark()
    
    try:
        with open(profiling_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_id = int(row['model'])
                steps = int(row['steps'])
                exec_time = float(row['exec_time_sec'])
                data[(model_id, steps)] = exec_time
    except Exception as e:
        print(f"[SLO] Warning: Could not load profiling data: {e}")
        return estimate_from_benchmark()
        
    return data

def estimate_from_benchmark() -> Dict[Tuple[int, int], float]:
    """
    Estimate execution times from the measured 20-step benchmark latency.
    model_id: avg_exec_time_sec measured at BENCHMARK_STEPS=20 (plot_1_latency_comparison.csv).
    """
    # model_id: avg_exec_time_sec at 20 steps
    estimate_map = {
        3: 14.983,   # SD-v1-5
        4: 14.260,   # SD-2-1
        5: 31.053,   # SDXL-Base
        8: 27.620,   # SD-3-5-Medium
        9: 73.987,   # SD-3-5-Large
    }

    estimated = {}

    # Linear scaling: exec(steps) = exec_20 * (steps / 20)
    for model_id, exec_20s in estimate_map.items():
        for steps in (20, 30, 40):
            estimated[(model_id, steps)] = round(exec_20s * (steps / 20), 3)

    print("[SLO] Using estimated execution times. Run generate_slo_profiling.py for the CSV.")
    return estimated

def resolve_step_count(accuracy_level: str) -> int:
    """
    Resolve accuracy level to concrete step count.
    """
    if accuracy_level.lower() not in ACCURACY_STEP_MAP:
        raise ValueError(f"Unknown accuracy level: {accuracy_level}")
    return ACCURACY_STEP_MAP[accuracy_level.lower()]

def resolve_latency_slo_ms(model_id: int, steps: int) -> float:
    """
    Compute auto-configured latency SLO.
    Returns SLO in milliseconds.
    """
    # In real case we'd load these from profiling data
    profile_data = load_exec_profiling_data() 
    
    if (model_id, steps) in profile_data:
        exec_time = profile_data[(model_id, steps)]
    else:
        # Fall back to estimation - but this should not happen with full profiling
        print(f"[SLO] Warning: No profile data for model {model_id} at {steps} steps. Using estimate.")
        # For now just return estimated value  
        exec_time = 10.0  # placeholder
    
    slo_seconds = SLO_LATENCY_MULTIPLIER * exec_time
    
    # Return in milliseconds
    return slo_seconds * 1000

def evaluate_slo(latency_sec: float, slo_ms: float) -> bool:
    """
    Evaluate if latency meets SLO.
    Returns True if latency <= SLO (SLO is in ms)
    """
    # Convert to same units  
    latency_ms = latency_sec * 1000
    return latency_ms <= slo_ms

def get_accuracy_level_from_steps(steps: int) -> str:
    """
    Map step count back to accuracy level string.
    This is the reverse of resolve_step_count.
    """
    # Canonical step counts map to their level unambiguously
    reverse_map = {v: k for k, v in ACCURACY_STEP_MAP.items()}
    if steps in reverse_map:
        return reverse_map[steps]

    # Fall back to half-open range membership for non-canonical step counts
    low_min = ACCURACY_STEP_RANGES["low"][0]
    med_min = ACCURACY_STEP_RANGES["medium"][0]
    high_min = ACCURACY_STEP_RANGES["high"][0]
    if steps < med_min:
        return "low"
    if steps < high_min:
        return "medium"
    return "high"

if __name__ == "__main__":
    # Quick test
    print("[SLO] Testing resolution functions:")
    
    # Test accuracy step maps
    print(f"Accuracy levels: {list(ACCURACY_STEP_MAP.keys())}")
    print(f"Step count for low: {resolve_step_count('low')}")
    print(f"Step count for medium: {resolve_step_count('medium')}")
    print(f"Step count for high: {resolve_step_count('high')}")
    
    # Test SLO computation
    slo = resolve_latency_slo_ms(1, 20)
    print(f"Latency SLO for model 1 at 20 steps: {slo:.1f} ms")
    
    print("[SLO] Module ready.")