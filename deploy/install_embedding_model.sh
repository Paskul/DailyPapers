#!/usr/bin/env bash
# Install the pinned sentence-embedding model used for screening (plan part 6) into a paperwatch home.
# Run as YOURSELF, not with sudo: the download happens as you, and sudo is used only to place the files when the home
# belongs to root (the service-account layout).
#
# Usage: deploy/install_embedding_model.sh [paperwatch home] [model name]
#        (defaults: /srv/paperwatch and bge-base-en-v1.5)
#
#   <home>/models/<model name>/model.onnx, tokenizer.json     root : paperwatch-state 0640 (dir 0750)
#
# The file list, download URLs (a pinned revision on huggingface.co), and sha256 hashes come from
# src/paperwatch/screening/embedding_model_registry.py, the same values paperwatch checks before every use. A file
# whose hash does not match is never installed. This is an operator step: at run time paperwatch never downloads a
# model, and the collector's egress rules do not allow huggingface.co. Safe to rerun (matching files are kept).
set -euo pipefail

[[ $EUID -ne 0 ]] || { echo "run as your own account, not root (the script uses sudo where needed)" >&2; exit 1; }
REPOSITORY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PAPERWATCH_HOME="$(realpath "${1:-/srv/paperwatch}")"
MODEL_NAME="${2:-bge-base-en-v1.5}"
SHARED_GROUP="${PAPERWATCH_SHARED_GROUP:-paperwatch-state}"
MODEL_DIR="$PAPERWATCH_HOME/models/$MODEL_NAME"
command -v curl >/dev/null || { echo "curl is not installed: sudo apt install curl" >&2; exit 1; }

MANIFEST="$(uv run --project "$REPOSITORY_DIR" --frozen --quiet python -m paperwatch.screening.embedding_model_registry \
    "$MODEL_NAME")"
DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf "$DOWNLOAD_DIR"' EXIT

# ---- 1. download and verify every file (as you) -------------------------------------------------------------------
while read -r FILE_NAME DOWNLOAD_URL EXPECTED_SHA256; do
    if [[ -r "$MODEL_DIR/$FILE_NAME" ]] && \
            [[ "$(sha256sum "$MODEL_DIR/$FILE_NAME" | cut -d' ' -f1)" == "$EXPECTED_SHA256" ]]; then
        echo "$FILE_NAME: already installed and verified"
        continue
    fi
    echo "$FILE_NAME: downloading $DOWNLOAD_URL"
    curl --fail --location --progress-bar --proto '=https' --tlsv1.2 --max-time 1800 \
        --output "$DOWNLOAD_DIR/$FILE_NAME" \
        "$DOWNLOAD_URL"
    ACTUAL_SHA256="$(sha256sum "$DOWNLOAD_DIR/$FILE_NAME" | cut -d' ' -f1)"
    if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
        echo "$FILE_NAME: sha256 $ACTUAL_SHA256 does not match the pinned $EXPECTED_SHA256; nothing installed" >&2
        exit 1
    fi
    echo "$FILE_NAME: sha256 verified"
done <<< "$MANIFEST"

# ---- 2. place the verified files ------------------------------------------------------------------------------------
shopt -s nullglob
DOWNLOADED_FILES=("$DOWNLOAD_DIR"/*)
if (( ${#DOWNLOADED_FILES[@]} > 0 )); then
    if [[ -w "$PAPERWATCH_HOME" ]] && [[ ! -e "$MODEL_DIR" || -w "$MODEL_DIR" ]]; then
        mkdir -p "$MODEL_DIR"                       # a home you own (development): no sudo needed
        install -m 0644 "${DOWNLOADED_FILES[@]}" "$MODEL_DIR/"
    else
        sudo install -d -o root -g "$SHARED_GROUP" -m 0750 "$PAPERWATCH_HOME/models" "$MODEL_DIR"
        sudo install -o root -g "$SHARED_GROUP" -m 0640 "${DOWNLOADED_FILES[@]}" "$MODEL_DIR/"
    fi
fi

echo
echo "Installed $MODEL_NAME in $MODEL_DIR. config.toml needs (these are the defaults):"
echo "  [screening]"
echo "  embedding_backend = \"onnx\""
echo "  embedding_model = \"$MODEL_NAME\""
echo "  embedding_model_dir = \"models/$MODEL_NAME\""
echo "Then check it: paperwatch --home $PAPERWATCH_HOME doctor   (the 'screening' line)"
