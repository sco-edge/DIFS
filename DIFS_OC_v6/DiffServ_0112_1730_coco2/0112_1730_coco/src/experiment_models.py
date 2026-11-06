# Author: KB 
# Date: 2026-07-10 
# Purpose: Configuration for available models for experiment 
# File: experiment_models.py 

"""
Experiment model registry.

This file contains ONLY the list of diffusion models used for
the DIFS model-switching experiments.

All filesystem paths come from project_paths.py.
"""

from pathlib import Path
from project_paths import MODEL_CACHE


MODELS = [

    {
        "model_id": 1,
        "model_name": "sd-v1-4.safetensors",
        "path": MODEL_CACHE / "sd-v1-4.safetensors",
        "display_name": "SD-v1.4 (Safetensor)"
    },

    {
        "model_id": 2,
        "model_name": "stable-diffusion-v1-4",
        "path": MODEL_CACHE / "stable-diffusion-v1-4",
        "display_name": "Stable Diffusion v1.4"
    },

    {
        "model_id": 3,
        "model_name": "stable-diffusion-v1-5",
        "path": MODEL_CACHE / "stable-diffusion-v1-5",
        "display_name": "Stable Diffusion v1.5"
    },

    {
        "model_id": 4,
        "model_name": "stable-diffusion-2-1",
        "path": MODEL_CACHE / "stable-diffusion-2-1",
        "display_name": "Stable Diffusion 2.1"
    },

    {
        "model_id": 5,
        "model_name": "stable-diffusion-xl-base-1.0",
        "path": MODEL_CACHE / "stable-diffusion-xl-base-1.0",
        "display_name": "SDXL Base"
    },

    {
        "model_id": 6,
        "model_name": "stable-diffusion-xl-refiner-1.0",
        "path": MODEL_CACHE / "stable-diffusion-xl-refiner-1.0",
        "display_name": "SDXL Refiner"
    },

    {
        "model_id": 7,
        "model_name": "dreamshaper-8",
        "path": MODEL_CACHE / "dreamshaper-8",
        "display_name": "DreamShaper 8"
    }

]


# Helper functions

MODEL_MAP = { model["model_id"]: model for model in MODELS}


def get_model(model_id):
    """
    Return the complete model dictionary for a model_id.
    """
    return MODEL_MAP[model_id]


def get_model_path(model_id):
    """
    Return pathlib.Path of the model.
    """
    return MODEL_MAP[model_id]["path"]


def get_model_name(model_id):
    """
    Return model filename or directory name.
    """
    return MODEL_MAP[model_id]["model_name"]


def get_display_name(model_id):
    """
    Return human-readable model name.
    """
    return MODEL_MAP[model_id]["display_name"]