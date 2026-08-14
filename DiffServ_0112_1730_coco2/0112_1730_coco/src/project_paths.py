from pathlib import Path
import os

############################################################
# Root directories
############################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_ROOT / "src"

WORKER_DIR = SRC_DIR / "1"

############################################################
# Configuration files
############################################################

MODELS_JSON = SRC_DIR / "models.json"

SAMPLERS_JSON = PROJECT_ROOT / "samplers.json"

ADDRESS_PORT_JSON = PROJECT_ROOT / "address_port.json"

############################################################
# Dataset
############################################################

PROMPTS_FILE = PROJECT_ROOT / "annotations" / "captions_val2014.json"

############################################################
# Statistics
############################################################

REAL_STATS = PROJECT_ROOT / "real_stats.npz"

############################################################
# Metrics
############################################################

METRICS_DIR = PROJECT_ROOT / "metrics"

############################################################
# Experiment directories
############################################################

RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = PROJECT_ROOT / "figures"

LOG_DIR = PROJECT_ROOT / "logs"

############################################################
# Diffusion model cache
############################################################

PRIMARY_MODEL_DIR = Path(
    "/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models/diffusion-models"
)

LEGACY_MODEL_DIR = Path("/tmp/model")

MODEL_CACHE = Path(
    os.environ.get(
        "DIFS_MODEL_DIR",
        str(PRIMARY_MODEL_DIR)
    )
)

############################################################
# Output directory
############################################################

DIFFUSION_OUTPUT = Path(os.environ.get("DIFS_OUTPUT_DIR", "/tmp/diffserv_output"))

OUTPUT_DIR = DIFFUSION_OUTPUT

############################################################
# Client output
############################################################

CLIENT_RECEIVED_DIR = PROJECT_ROOT / "client_received"

# Backward compatibility
CLIENT_DIR = CLIENT_RECEIVED_DIR

############################################################
# Additional directories
############################################################

EXPERIMENTS_DIR = SRC_DIR / "experiments"

WORKERS_LOG_DIR = PROJECT_ROOT / "WorkersLogs"

############################################################
# Create directories
############################################################

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

CLIENT_RECEIVED_DIR.mkdir(parents=True, exist_ok=True)

WORKERS_LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)