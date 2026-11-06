#!/usr/bin/bash
# Author: KB
# Purpose: To copy diffusion model as well as stats file to required location


MODEL_SOURCE="${HOME}/Downloads/sd-models"
MODEL_STATS_DEST="/tmp/model/"
MODEL_NAME="sd-v1-4.safetensors"

STATS_SOURCE="./"
STATS_NAME="real_stats.npz"


MODEL="${MODEL_SOURCE}/${MODEL_NAME}"
STATS="${STATS_SOURCE}/${STATS_NAME}"
if [[ -d "${MODEL_STATS_DEST}" ]]; then
    sudo cp -v "$MODEL" "${MODEL_STATS_DEST}"
    sudo cp -v "$STATS" "${MODEL_STATS_DEST}"
fi


