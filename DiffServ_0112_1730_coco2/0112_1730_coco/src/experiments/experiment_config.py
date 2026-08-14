"""
experiment_config.py

Central configuration for all DIFS experiments.
"""

from project_paths import (
    PROJECT_ROOT,
    RESULTS_DIR,
    METRICS_DIR,
    CLIENT_DIR,
)

############################################################
# Network
############################################################

SCHEDULER_HOST = "localhost"

SCHEDULER_PORT = 50050

WORKER_PORT = 50060

############################################################
# Experiment Parameters
############################################################

DEFAULT_WORKLOAD = 5

WORKLOADS = [
    5,
    10,
    20,
    50,
    100,
    200
]

DEFAULT_SAMPLER = 1

SAMPLERS = [
    1,
    2,
    3,
    4
]

DEFAULT_STEPS = 20

STEPS = [
    20,
    30,
    50
]

############################################################
# SLO Benchmark Parameters
############################################################

# Request arrival rates (req/s) per load level used by the
# SLO benchmark. Poisson traffic; each level draws from a pool of
# 300 COCO captions (captions_val2014.json):
#   low    2 req/min  = 0.0333 req/s
#   medium 4 req/min  = 0.0667 req/s
#   high   6 req/min  = 0.1000 req/s
SLO_LOAD_LEVELS = {
    "low": 2 / 60.0,
    "medium": 4 / 60.0,
    "high": 6 / 60.0,
}

# Total prompts (COCO captions) emitted per load level.
SLO_PROMPT_COUNT = 300

# Arrival window per load level: time to emit all SLO_PROMPT_COUNT
# prompts at the level's rate.
#   low:    300 / (2/60)   = 9000 s (150 min)
#   medium: 300 / (4/60)   = 4500 s (75 min)
#   high:   300 / (6/60)   = 3000 s (50 min)
SLO_WINDOW_SEC_BY_LEVEL = {
    level: round(SLO_PROMPT_COUNT / rate)
    for level, rate in SLO_LOAD_LEVELS.items()
}

SLO_CLIENT_TIMEOUT_MULT = 4.0  # drop request if E2E > 4x exec time

# Inference worker pool size per load level used by the SLO benchmark.
# Realistic assumption: infrastructure is FIXED and only the offered
# load varies, so every level is served by the same pool of 5 workers
# (= the single-node MAX_WORKERS cap, one GPU per worker). The levels
# then span the node's operating range at the measured mixed-model
# service rate (mean exec ~42.9 s -> aggregate capacity ~7 req/min):
#   low:    rate 2 req/min -> pool rho ~0.29 (comfortable)
#   medium: rate 4 req/min -> pool rho ~0.57 (approaching the knee)
#   high:   rate 6 req/min -> pool rho ~0.86 (near saturation; the SLO
#           cliff that triggers the autoscaler's scale-out signal)
SLO_WORKERS_BY_LEVEL = {
    "low": 5,
    "medium": 5,
    "high": 5,
}

############################################################
# Model Benchmark Parameters
############################################################

# 8 txt2img models available for benchmarking
# Excluded: id=6 (SDXL Refiner, img2img-only pipeline)
MODELS = [1, 2, 3, 4, 5, 7, 8, 9]

# Scaling modes to benchmark
SCALING_MODES = [
    "horizontal",
    "vertical",
    "hybrid"
]

DEFAULT_SCALING_MODE = "vertical"

# Worker launch mode: "docker" or "direct"
# "direct" runs worker as a Python subprocess (faster, no Docker rebuild needed)
# "docker" uses the difs-diffusion-worker image
RUN_DIRECT = True

BENCHMARK_WORKLOAD = 5

BENCHMARK_STEPS = 20

BENCHMARK_SAMPLER = 1

############################################################
# Timing
############################################################

SERVER_STARTUP_WAIT = 8

WORKER_STARTUP_WAIT = 10

POST_EXPERIMENT_WAIT = 3

############################################################
# Client
############################################################

CFG_SCALE = 7.5

SEED = 42

CLIENT_SCRIPT = str(PROJECT_ROOT / "src" / "client_async_v3.py")

############################################################
# Worker
############################################################

MODEL_ID = 1

THREADS = 4

WORKER_SCRIPT = str(PROJECT_ROOT / "src" / "1" / "pytorch_container_diffusion_v3.py")

############################################################
# Bootstrap
############################################################

BOOTSTRAP_SCRIPT = "src/1/bootstrap.py"

SERVER_SCRIPT = "src/server.py"

############################################################
# Server
############################################################

SERVER_PORT = 50050

SERVER_COMMAND = [
    "python",
    "src/server.py"
]