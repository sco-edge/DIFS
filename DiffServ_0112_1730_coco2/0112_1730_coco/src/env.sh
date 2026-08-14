#!/bin/bash

######################################################
# HuggingFace Cache (use SSD, not root)
######################################################

export HF_HOME=/mnt/ssd4t/huggingface_cache_moved

######################################################
# Diffusion Models
######################################################

export DIFS_MODEL_DIR=/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models/diffusion-models

######################################################
# Output Cache
######################################################

export DIFS_OUTPUT_DIR=/tmp/diffusion_output

######################################################
# GPU
######################################################

export CUDA_VISIBLE_DEVICES=0