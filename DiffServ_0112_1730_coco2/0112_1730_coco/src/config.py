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
