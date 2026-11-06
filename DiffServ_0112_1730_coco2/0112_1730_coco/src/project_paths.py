from pathlib import Path
import os

from output_storage import resolve_output_dir

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

FALLBACK_MODEL_DIR = Path(
    "/mnt/ssd4t/kwadwo/Downloads/sd-models/downloaded_diffusion_models/diffusion-models"
)

LEGACY_MODEL_DIR = Path("/tmp/model")

MODEL_DIR_CANDIDATES = (PRIMARY_MODEL_DIR, FALLBACK_MODEL_DIR)


def _resolve_model_dir():
    env_dir = os.environ.get("DIFS_MODEL_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)
    for candidate in MODEL_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return PRIMARY_MODEL_DIR


MODEL_CACHE = _resolve_model_dir()

############################################################
# Output directory
############################################################

# If /tmp/diffserv_output is on a nearly full filesystem, first delete its
# disposable images/npz files and then relocate to a larger filesystem.  The
# chosen path is exported as DIFS_OUTPUT_DIR for every subprocess.
DIFFUSION_OUTPUT = resolve_output_dir()

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