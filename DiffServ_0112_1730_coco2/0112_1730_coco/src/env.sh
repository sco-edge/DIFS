#!/bin/bash

######################################################
# HuggingFace Cache (use SSD, not root)
######################################################

export HF_HOME=/mnt/ssd4t/huggingface_cache_moved

######################################################
# Diffusion Models
######################################################

# Prefer the primary SSD (my_ssd); fall back to the relocated copy on /mnt/ssd4t
if [ -d "/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models/diffusion-models" ]; then
    export DIFS_MODEL_DIR=/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models/diffusion-models
else
    export DIFS_MODEL_DIR=/mnt/ssd4t/kwadwo/Downloads/sd-models/downloaded_diffusion_models/diffusion-models
fi

######################################################
# Output Cache
######################################################

export DIFS_OUTPUT_DIR=/tmp/diffusion_output

######################################################
# GPU
######################################################

export CUDA_VISIBLE_DEVICES=0