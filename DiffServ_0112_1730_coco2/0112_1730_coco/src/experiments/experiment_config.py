"""
experiment_config.py

Central configuration for all DIFS experiments.

Everything that defines an experiment should live here.
"""

"""
experiment_config.py

Central configuration for all DIFS experiments.
"""

from ..project_paths import (
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

DEFAULT_SCALING_MODE = "vertical"

SCALING_MODES = [
    "horizontal",
    "vertical",
    "hybrid"
]

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