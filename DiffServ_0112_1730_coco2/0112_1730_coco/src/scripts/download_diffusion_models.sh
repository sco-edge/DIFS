#!/usr/bin/env bash
#
# download_diffusion_models.sh
#
# Stable Diffusion model downloader; To download a list of diffusion
# models from either git or hugging face or both.
#
# Since the models are quite large, move the script to a location with
# enough storage space before executing (e.g. An external harddisk or
# your Downloads folder)
#
# Supports:
#   - Git + Git LFS
#   - Hugging Face CLI
#   - Both (HF CLI preferred)
#
# Environment variables:
#   HF_TOKEN        - HuggingFace access token (required for gated repos)
#   MODEL_DIR       - Override default download directory
#   SKIP_EXISTING   - Skip models that already exist (default: true)
#
# Usage:
#   HF_TOKEN=hf_xxx ./download_diffusion_models.sh
#   MODEL_DIR=/path/to/models ./download_diffusion_models.sh
#

set -Eeuo pipefail

trap 'echo "ERROR: line $LINENO: $BASH_COMMAND"' ERR

############################################################
# CONFIGURATION
############################################################

MODEL_DIR="${MODEL_DIR:-/mnt/my_ssd/Downloads/sd-models/downloaded_diffusion_models}"

USE_GIT=false
USE_HF_CLI=true

SKIP_EXISTING="${SKIP_EXISTING:-true}"

RETRIES=3

############################################################
# MODELS TO DOWNLOAD
#
# Format: "huggingface_repo_id  local_subdir_name"
# local_subdir_name is optional (defaults to repo basename)
############################################################

# Models that require HF_TOKEN (gated repos):
#   stabilityai/stable-diffusion-2-1       -> use sd2-community/stable-diffusion-2-1 (ungated mirror)
#   stabilityai/stable-diffusion-3.5-large  -> use camenduru/stable-diffusion-3.5-large (ungated mirror)
#   camenduru/stable-diffusion-3.5-large    -> ungated mirror with single safetensors

declare -A MODELS=(
    ["CompVis/stable-diffusion-v1-4"]="stable-diffusion-v1-4"
    ["stable-diffusion-v1-5/stable-diffusion-v1-5"]="stable-diffusion-v1-5"
    ["sd2-community/stable-diffusion-2-1"]="stable-diffusion-2-1"
    ["stabilityai/stable-diffusion-xl-base-1.0"]="stable-diffusion-xl-base-1.0"
    ["stabilityai/stable-diffusion-xl-refiner-1.0"]="stable-diffusion-xl-refiner-1.0"
    ["Lykon/dreamshaper-8"]="dreamshaper-8"
    ["camenduru/stable-diffusion-3.5-large"]="stable-diffusion-3.5-large"
)

# Ordered keys for consistent iteration
MODEL_KEYS=(
    "CompVis/stable-diffusion-v1-4"
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
    "sd2-community/stable-diffusion-2-1"
    "stabilityai/stable-diffusion-xl-base-1.0"
    "stabilityai/stable-diffusion-xl-refiner-1.0"
    "Lykon/dreamshaper-8"
    "camenduru/stable-diffusion-3.5-large"
)

############################################################

mkdir -p "$MODEL_DIR"

START_TIME=$(date +%s)

WATCH_PID=""

SUCCESS=()
FAILED=()

############################################################
# HF Token handling
############################################################

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN detected. Setting up authentication..."
    export HF_TOKEN
else
    echo "WARNING: HF_TOKEN not set. Gated repos may fail."
    echo "  Set it with: export HF_TOKEN=hf_xxxxxxxx"
    echo ""
fi

############################################################
# Detect Hugging Face CLI
############################################################

HF_CMD=""

if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli"
fi

############################################################
# Tool checks
############################################################

if ! $USE_GIT && ! $USE_HF_CLI; then
    echo "ERROR: Enable Git or Hugging Face CLI."
    exit 1
fi


if $USE_GIT; then

    command -v git >/dev/null ||
        { echo "ERROR: git missing"; exit 1; }

    command -v git-lfs >/dev/null ||
        { echo "ERROR: git-lfs missing"; exit 1; }

    git lfs install

fi


if $USE_HF_CLI && [[ -z "$HF_CMD" ]]; then

    echo
    echo "ERROR: Hugging Face CLI missing"
    echo 'Install: pip install -U "huggingface_hub[cli]"'
    exit 1

fi

############################################################
# Watch total download size
############################################################

watch_download()
{
    while true
    do
        SIZE=$(du -sh "$MODEL_DIR" 2>/dev/null | awk '{print $1}')

        printf "[%s] Total downloaded: %s\n" \
            "$(date '+%H:%M:%S')" \
            "${SIZE:-0}"

        sleep 3
    done
}


stop_watch()
{
    if [[ -n "${WATCH_PID:-}" ]]; then
        kill "$WATCH_PID" >/dev/null 2>&1 || true
        wait "$WATCH_PID" 2>/dev/null || true
        WATCH_PID=""
    fi
}


trap stop_watch EXIT

############################################################
# Download functions
############################################################

download_git()
{
    local repo="$1"
    local target="$2"

    if [[ -d "$target/.git" ]]; then

        echo "Updating Git repository..."

        git -C "$target" pull

    else

        git clone \
            --progress \
            "https://huggingface.co/$repo" \
            "$target"

    fi
}


download_hf()
{
    local repo="$1"
    local target="$2"

    local token_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
        token_args=(--token "$HF_TOKEN")
    fi

    if [[ "$HF_CMD" == "hf" ]]; then

        hf download \
            "$repo" \
            --local-dir "$target" \
            "${token_args[@]}"

    else

        huggingface-cli download \
            "$repo" \
            --local-dir "$target" \
            "${token_args[@]}"

    fi
}


download_model()
{
    local repo="$1"
    local target="$2"

    local attempt=1

    while (( attempt <= RETRIES ))
    do

        echo "Attempt $attempt/$RETRIES"

        if $USE_HF_CLI; then

            download_hf "$repo" "$target" && return 0

        elif $USE_GIT; then

            download_git "$repo" "$target" && return 0

        fi

        echo "Retrying in 5 seconds..."
        sleep 5
        ((attempt++))

    done

    return 1
}

############################################################
# Main
############################################################

echo
echo "================================================"
echo " Stable Diffusion Downloader"
echo "================================================"
echo
echo "Destination: $MODEL_DIR"
echo "Git:         $USE_GIT"
echo "HF CLI:      $USE_HF_CLI"
echo "HF Token:    ${HF_TOKEN:+SET}${HF_TOKEN:-NOT SET}"
echo


TOTAL=${#MODEL_KEYS[@]}
COUNT=0


for REPO in "${MODEL_KEYS[@]}"
do

    ((++COUNT))

    NAME="${MODELS[$REPO]}"
    TARGET="$MODEL_DIR/$NAME"

    echo
    echo "================================================"
    echo "Model $COUNT/$TOTAL"
    echo "  Repo:  $REPO"
    echo "  Dir:   $TARGET"
    echo "================================================"


    if [[ -d "$TARGET" && "$SKIP_EXISTING" == true ]]; then

        # Check if directory has actual model files (not just .git)
        FILE_COUNT=$(find "$TARGET" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.ckpt" -o -name "model_index.json" \) 2>/dev/null | wc -l)

        if (( FILE_COUNT > 0 )); then
            echo "Already exists with $FILE_COUNT model file(s). Skipping."
            SUCCESS+=("$NAME")
            continue
        else
            echo "Directory exists but no model files found. Re-downloading."
        fi

    fi


    mkdir -p "$TARGET"


    watch_download &
    WATCH_PID=$!


    if download_model "$REPO" "$TARGET"; then

        SIZE=$(du -sh "$TARGET" | awk '{print $1}')

        echo
        echo "Completed: $NAME ($SIZE)"

        SUCCESS+=("$NAME")

    else

        echo
        echo "FAILED: $NAME"

        FAILED+=("$NAME")

    fi


    stop_watch

done


END_TIME=$(date +%s)
ELAPSED=$((END_TIME-START_TIME))


echo
echo "================================================"
echo " Download Summary"
echo "================================================"

echo
echo "Successful:"
printf "  - %s\n" "${SUCCESS[@]:-none}"

echo
echo "Failed:"
printf "  - %s\n" "${FAILED[@]:-none}"

echo
echo "Total downloaded:"
du -sh "$MODEL_DIR"

echo
echo "Elapsed time: ${ELAPSED}s ($(($ELAPSED / 60))m $(($ELAPSED % 60))s)"

echo
echo "Done."
