#!/usr/bin/env bash
#
# download_diffusion_models.sh
#
# Stable Diffusion model downloader; To download a lift of diffusion
# models from either git or hugging face or both.
#
# Since the models are quite large. Move the script to a location with
# enough storage space before executing (e.g. An external harddisk or
# your Downloads folder)
#
# Supports:
#   - Git + Git LFS
#   - Hugging Face CLI
#   - Both (HF CLI preferred)
#

set -Eeuo pipefail

trap 'echo "ERROR: line $LINENO: $BASH_COMMAND"' ERR

############################################################
# CONFIGURATION
############################################################

MODEL_DIR="$(pwd)/diffusion-models"

USE_GIT=false
USE_HF_CLI=true

SKIP_EXISTING=true

RETRIES=3

############################################################
# UNGATED MODELS
############################################################

MODELS=(

    "CompVis/stable-diffusion-v1-4"

    "stable-diffusion-v1-5/stable-diffusion-v1-5"

    "stabilityai/stable-diffusion-2-1"

    "stabilityai/stable-diffusion-xl-base-1.0"

    "stabilityai/stable-diffusion-xl-refiner-1.0"

    "Lykon/dreamshaper-8"

)

############################################################

mkdir -p "$MODEL_DIR"

START_TIME=$(date +%s)

WATCH_PID=""

SUCCESS=()
FAILED=()

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

    if [[ "$HF_CMD" == "hf" ]]; then

        hf download \
            "$repo" \
            --local-dir "$target"

    else

        huggingface-cli download \
            "$repo" \
            --local-dir "$target"

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

            download_hf "$repo" "$target"

        elif $USE_GIT; then

            download_git "$repo" "$target"

        fi

        return 0

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
echo


TOTAL=${#MODELS[@]}
COUNT=0


for REPO in "${MODELS[@]}"
do

    ((++COUNT))

    NAME=$(basename "$REPO")
    TARGET="$MODEL_DIR/$NAME"

    echo
    echo "================================================"
    echo "Model $COUNT/$TOTAL"
    echo "$REPO"
    echo "================================================"


    if [[ -d "$TARGET" && "$SKIP_EXISTING" == true ]]; then

        echo "Already exists. Skipping."
        SUCCESS+=("$NAME")
        continue

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
echo "Elapsed time: ${ELAPSED}s"

echo
echo "Done."
