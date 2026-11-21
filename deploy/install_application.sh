#!/usr/bin/env bash
# Install paperwatch read-only under /opt/paperwatch (plan part 3). Run with sudo from anywhere in the repository.
#
# Usage: sudo env "PATH=$PATH" deploy/install_application.sh
#
# The installed package directory is root-owned and not writable by any service account, so neither the code nor
# the read-only provider rules (paperwatch/network/provider_rules.toml) can be changed by the services.
# Dependencies are installed once, here, with the exact versions and hashes from uv.lock. Nothing is downloaded at
# run time. Rerun this script after pulling new code.
#
# Order (2026-11-21: a download whose sha256 did not match, most likely a large wheel cut off in transit, used to
# fail AFTER the old venv was cleared, which left /opt/paperwatch without a working paperwatch):
#   1. download every pinned wheel into a temporary directory, hash-checked, retrying the whole download a few times;
#   2. build the paperwatch wheel there too;
#   3. only then rebuild the venv in /opt, offline (--no-index), from those verified local files.
# A failed download therefore leaves the installed application exactly as it was.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo (keep your PATH so uv is found: sudo env \"PATH=\$PATH\" $0)" >&2; exit 1; }
INSTALL_DIR="${PAPERWATCH_INSTALL_DIR:-/opt/paperwatch}"
REPOSITORY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_BIN="${UV:-$(command -v uv || true)}"
[[ -n "$UV_BIN" ]] || { echo "uv not found; pass UV=/path/to/uv" >&2; exit 1; }
DOWNLOAD_ATTEMPTS=3

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT
WHEEL_DIR="$BUILD_DIR/wheels"

# Pinned, hash-checked dependency list from the lock file (no dev dependencies, not the project itself).
"$UV_BIN" export --project "$REPOSITORY_DIR" --frozen --no-dev --no-emit-project --format requirements-txt \
    --output-file "$BUILD_DIR/requirements.txt" >/dev/null

# A throwaway venv with the same interpreter does the downloading and the build, so /opt is untouched until both
# succeed. PIP_CONFIG_FILE=/dev/null: no pip configuration of the root account can redirect the downloads.
python3 -m venv "$BUILD_DIR/download-venv"
DOWNLOAD_PIP=(env PIP_CONFIG_FILE=/dev/null "$BUILD_DIR/download-venv/bin/pip")

echo "downloading the pinned dependencies (hash-checked)"
for attempt in $(seq 1 "$DOWNLOAD_ATTEMPTS"); do
    rm -rf "$WHEEL_DIR"
    if "${DOWNLOAD_PIP[@]}" download --quiet --no-cache-dir --require-hashes --only-binary=:all: \
            --retries 5 --timeout 60 --dest "$WHEEL_DIR" -r "$BUILD_DIR/requirements.txt"; then
        break
    fi
    if [[ $attempt -eq $DOWNLOAD_ATTEMPTS ]]; then
        echo "the dependency download failed $DOWNLOAD_ATTEMPTS times; $INSTALL_DIR was NOT changed." >&2
        echo "A repeated hash mismatch for the same file means it is not the file uv.lock pinned: check the" >&2
        echo "network (proxy, captive portal) before trusting it." >&2
        exit 1
    fi
    echo "download attempt $attempt failed (see above); retrying in 15 s" >&2
    sleep 15
done
"${DOWNLOAD_PIP[@]}" wheel --quiet --no-cache-dir --no-deps --wheel-dir "$WHEEL_DIR" "$REPOSITORY_DIR"

echo "installing into $INSTALL_DIR from the verified local files (offline)"
python3 -m venv --clear "$INSTALL_DIR/venv"
INSTALL_PIP=(env PIP_CONFIG_FILE=/dev/null "$INSTALL_DIR/venv/bin/pip")
"${INSTALL_PIP[@]}" install --quiet --no-cache-dir --no-index --find-links "$WHEEL_DIR" --require-hashes \
    -r "$BUILD_DIR/requirements.txt"
"${INSTALL_PIP[@]}" install --quiet --no-cache-dir --no-index --no-deps "$WHEEL_DIR"/paperwatch-*.whl

chown -R root:root "$INSTALL_DIR"
chmod -R go-w "$INSTALL_DIR"
ln -sf "$INSTALL_DIR/venv/bin/paperwatch" /usr/local/bin/paperwatch

RULES_FILE="$("$INSTALL_DIR/venv/bin/python" -c \
    'from paperwatch.network.provider_rules import packaged_provider_rules_path as p; print(p())')"
echo "installed paperwatch to $INSTALL_DIR (read-only); provider rules: $RULES_FILE"
echo "command: /usr/local/bin/paperwatch"
