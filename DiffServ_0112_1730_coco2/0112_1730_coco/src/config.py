# Author: KB 
# Date: 2026-03-06 
# Purpose: For keeping configuration settings 
# File: config.py 

SCALE_UP_THRESHOLD = 5
SCALE_DOWN_THRESHOLD = 1

UTILIZATION_SCALE_UP = 0.8
UTILIZATION_SCALE_DOWN = 0.3

MIN_WORKERS = 1
MAX_WORKERS = 5

AUTOSCALE_INTERVAL = 5
CHECK_INTERVAL = 5

# Options: "vertical", "horizontal", "hybrid"
SCALING_MODE = "horizontal"

############################################################
# Service Level Objective Configuration
############################################################

SLO_LATENCY_MULTIPLIER = 2.5

ACCURACY_STEP_RANGES = {
    "high": (30, 40),   # 30-40 denoising steps
    "medium": (20, 30), # 20-30 denoising steps  
    "low": (10, 20)     # 10-20 denoising steps
}

DEFAULT_ACCURACY_LEVEL = "medium"

# Rolling latency-SLO violation-rate budget (1%). If the observed
# violation rate exceeds this, the autoscaler forces a scale-up.
SLO_VIOLATION_BUDGET = 0.01

# Legacy SLA_TARGETS kept for compatibility but may be removed in future
SLA_TARGETS = {
    "interactive": 500,
    "balanced": 2000,
    "batch": 8000
}

##############################################################
# PANINISERVE MERGE SEAMS (DIFS v3)
# All seams default to OFF: with these flags left False, DIFS v3
# scheduling behavior is identical to DIFS v2.  Ported from
# Templates/DIFS-Trade-off-main/PaniniServe_scheduler.py.
##############################################################

# Seam 1: overload fair-share step reduction + deadline drops (admission control).
# When True, under overload (active_requests > PANINI_OVERLOAD_THRESHOLD) the
# scheduler shrinks denoising steps fair-share style (never below the accuracy
# tier floor) and drops requests whose reduced step count still cannot meet
# their latency SLO.  Also enables the streaming-path active-request counter
# (decrement on completion) that admission control relies on.
PANINI_ADMISSION_CONTROL = False

# Seam 1 companion: interpolate the latency-SLO multiplier across the 10-40
# step range (2.0x at 10 steps .. 3.0x at 40 steps) instead of the fixed 2.5x.
PANINI_SLO_INTERPOLATED = False

# Seam 1: active-request count that marks the system as overloaded.
PANINI_OVERLOAD_THRESHOLD = 4

# Seam 1: hard drop threshold -- requests are rejected outright once in-flight
# requests reach this count (default: 2x the overload threshold).
PANINI_DROP_THRESHOLD = 8

# Seam 1: step reduction per overload unit during fair-share shrinking.
PANINI_STEP_CUT_PER_OVERLOAD = 2

# Seam 2: per-request GPU utilization monitor (10 ms polling).  When True,
# dispatch_request starts a GPUUtilMonitor for the request's execution window
# and logs the average utilization via metrics.log_gpu_metrics().  Degrades
# gracefully (logs 0) when no GPU / pynvml is available.
PANINI_GPU_MONITOR = False

# Seam 3: locality-aware worker selection.  When True, dispatch requests to a
# worker already running the request's model (front-block preference) instead
# of blindly picking the least-loaded worker.
PANINI_LOCALITY = False

MIN_WORKER_AGE = 60

##############################################################
# MODEL SWITCHING; Options: hot or cold; changing this
# switches the whole experiment from cold to hot or vice-versa
##############################################################

#MODEL_SWITCH_MODE = "cold"
MODEL_SWITCH_MODE = "hot" 

##############################################################
# INFaaS-STYLE SCALING PARAMETERS
##############################################################

# Maximum replicas per model within a single worker process (horizontal intra-process)
GPU_MAX_REPLICAS = 1
CPU_MAX_REPLICAS = 2

# Maximum containers per model (cross-process horizontal)
MAX_CONTAINERS_PER_MODEL = 3

# Consecutive idle cycles before a scale-down is triggered
SCALE_DOWN_DELAY = 20

# Cost-comparison threshold (Strategy A vs B in unified hybrid)
# If sum_cost > STRATEGY_THRESHOLD * fastest_cost, use fastest variant
STRATEGY_THRESHOLD = 1.5

# GPU memory budget (MB) for replica decisions
GPU_MEMORY_BUDGET_MB = 40000

# Per-model estimated load times (seconds) — used for cost model
# Updated dynamically from actual measurements
MODEL_LOAD_COSTS = {
    "sd-v1-4.safetensors":          15.0,
    "stable-diffusion-v1-4":        15.0,
    "stable-diffusion-v1-5":        15.0,
    "stable-diffusion-2-1":         18.0,
    "stable-diffusion-xl-base-1.0": 25.0,
    "dreamshaper-8":                15.0,
    "stable-diffusion-3.5-medium":  16.0,
}

##############################################################
# BASELINE ARCHITECTURE PROFILES (hardcoded)
# Values live in baseline_profiles.py and are selected with a single
# variable, DIFS_BASELINE_PROFILE.  The old per-toggle environment
# variables are intentionally ignored so manual runs are deterministic.
##############################################################

from baseline_profiles import active_profile as _active_profile

_BASELINE_PROFILE = _active_profile()

# Dispatch policy: "edf" (default) or "fifo"
DISPATCH_POLICY = _BASELINE_PROFILE["dispatch_policy"]

# Autoscaler: "1" = enabled (default), "0" = disabled
AUTOSCALER_ENABLED = _BASELINE_PROFILE["autoscaler"] == "1"

# PaniniServe Seam 1: fair-share step reduction (admission control)
PANINI_ADMISSION_CONTROL = _BASELINE_PROFILE["panini_admission"] == "1"

# PaniniServe Seam 3: locality-aware worker selection
PANINI_LOCALITY = _BASELINE_PROFILE["panini_locality"] == "1"
