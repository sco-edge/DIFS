# Author: KB 
# Date: 2026-03-06 
# Purpose: To register different possible model requirements. Together with "workload_classifier.py", it helps determine which model to load based on workload.
# Updated: 2026-07 - Added all 7 SD models and model_id→filename resolver

import json
from project_paths import MODELS_JSON

# -----------------------------------------
# WORKLOAD-TYPE → MODEL CONFIG MAPPING
# -----------------------------------------
# Used by worker_pool.add_worker() to pick model/sampler/threads
# based on workload classification (fast, balanced, quality).

MODEL_CONFIGS = {

    "fast": {
        "model_id": 1,
        "sampler": 1,
        "threads": 4,
    },

    "balanced": {
        "model_id": 3,
        "sampler": 2,
        "threads": 10,
    },

    "quality": {
        "model_id": 5,
        "sampler": 4,
        "threads": 16,
    },
}

# -----------------------------------------
# ALL AVAILABLE SD MODELS (from models.json)
# -----------------------------------------
# Each entry maps model_id → metadata for benchmarking.
# sampler and threads are default suggestions; can be overridden.

ALL_MODELS = {
    1: {"id": 1, "filename": "sd-v1-4.safetensors",          "short_name": "SD-v1-4-safetensors",  "sampler": 1, "threads": 4},
    2: {"id": 2, "filename": "stable-diffusion-v1-4",         "short_name": "SD-v1-4-dir",          "sampler": 1, "threads": 4},
    3: {"id": 3, "filename": "stable-diffusion-v1-5",         "short_name": "SD-v1-5",              "sampler": 2, "threads": 10},
    4: {"id": 4, "filename": "stable-diffusion-2-1",          "short_name": "SD-2-1",               "sampler": 2, "threads": 10},
    5: {"id": 5, "filename": "stable-diffusion-xl-base-1.0",  "short_name": "SDXL-Base",            "sampler": 2, "threads": 16},
    6: {"id": 6, "filename": "stable-diffusion-xl-refiner-1.0","short_name": "SDXL-Refiner",        "sampler": 2, "threads": 16},
    7: {"id": 7, "filename": "dreamshaper-8",                 "short_name": "DreamShaper-8",        "sampler": 2, "threads": 10},
    8: {"id": 8, "filename": "stable-diffusion-3.5-medium",   "short_name": "SD-3-5-Medium",        "sampler": 2, "threads": 16},
    9: {"id": 9, "filename": "stable-diffusion-3.5-large",    "short_name": "SD-3-5-Large",         "sampler": 2, "threads": 16},
}


def get_model_filename(model_id):
    """Resolve a numeric model_id to the model directory/filename string."""
    entry = ALL_MODELS.get(model_id)
    if entry is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    return entry["filename"]


def get_model_short_name(model_id):
    """Return a human-readable short name for a model_id."""
    entry = ALL_MODELS.get(model_id)
    if entry is None:
        return f"model_{model_id}"
    return entry["short_name"]


def get_model_config(model_id):
    """Return the full config dict for a model_id."""
    entry = ALL_MODELS.get(model_id)
    if entry is None:
        raise ValueError(f"Unknown model_id: {model_id}")
    return entry


def load_models_json():
    """Load the models.json file and return as a list of dicts."""
    with open(str(MODELS_JSON), "r") as f:
        return json.load(f)