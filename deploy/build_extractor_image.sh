#!/usr/bin/env bash
# Build the pinned text-extractor image and load it into the collector account's rootless image store.
#
# Usage: deploy/build_extractor_image.sh [collector account]      (run as your normal user, NOT with sudo)
#        An empty account ("") keeps the image in your own store only (the `local` setup mode). ./setup.sh runs this.
#
# Needs the network once, for the Debian base image and packages. The collector itself never downloads images: it
# runs `podman run --pull=never` with the image id printed at the end, which you put in config.toml:
#     [extractor]
#     image = "<printed 64-hex image id>"
# The image id is the sha256 of the image configuration, so it pins the exact parser build, and it survives the
# `podman save | podman load` transfer into the collector's storage.
set -euo pipefail

COLLECTOR_ACCOUNT="${1-paperwatch-collector}"
BUILD_TAG=localhost/paperwatch-extractor:build
BASE_IMAGE_NAME=docker.io/library/debian:bookworm-slim
[[ $EUID -ne 0 ]] || { echo "run as your normal user; the script calls sudo only for the load step" >&2; exit 1; }
command -v podman >/dev/null || { echo "podman is not installed: sudo apt install podman" >&2; exit 1; }

cd "$(dirname "$0")/extractor"
podman pull "$BASE_IMAGE_NAME"
BASE_IMAGE_DIGEST="$(podman image inspect --format '{{.Digest}}' "$BASE_IMAGE_NAME")"
podman build --pull=never --build-arg "BASE_IMAGE=docker.io/library/debian@${BASE_IMAGE_DIGEST}" \
    --tag "$BUILD_TAG" .
IMAGE_ID="$(podman image inspect --format '{{.Id}}' "$BUILD_TAG")"
echo "built $BUILD_TAG = $IMAGE_ID (base debian@${BASE_IMAGE_DIGEST})"
podman run --rm --pull=never --network=none --read-only --entrypoint cat "$IMAGE_ID" \
    /opt/paperwatch-extractor/PACKAGE_VERSIONS

if [[ -z "$COLLECTOR_ACCOUNT" ]]; then
    echo "image kept in your own store (local mode)"
elif id -u "$COLLECTOR_ACCOUNT" >/dev/null 2>&1; then
    ARCHIVE_DIR="$(mktemp -d)"
    trap 'rm -rf "$ARCHIVE_DIR"' EXIT
    chmod 0755 "$ARCHIVE_DIR"
    podman save --format oci-archive --output "$ARCHIVE_DIR/extractor.tar" "$BUILD_TAG"
    chmod 0644 "$ARCHIVE_DIR/extractor.tar"
    COLLECTOR_UID="$(id -u "$COLLECTOR_ACCOUNT")"
    (cd / && sudo -u "$COLLECTOR_ACCOUNT" -H env "XDG_RUNTIME_DIR=/run/user/$COLLECTOR_UID" \
        podman load --input "$ARCHIVE_DIR/extractor.tar")
    (cd / && sudo -u "$COLLECTOR_ACCOUNT" -H env "XDG_RUNTIME_DIR=/run/user/$COLLECTOR_UID" \
        podman image exists "$IMAGE_ID") && echo "loaded into $COLLECTOR_ACCOUNT's image store"
else
    echo "account $COLLECTOR_ACCOUNT does not exist yet; image left in your own store only"
fi

echo
echo "Set in config.toml:"
echo "  [extractor]"
echo "  image = \"$IMAGE_ID\""
