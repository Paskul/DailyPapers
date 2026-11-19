#!/usr/bin/env bash
# Install paperwatch read-only under /opt/paperwatch (plan part 3). Run with sudo from anywhere in the repository.
#
# Usage: sudo env "PATH=$PATH" deploy/install_application.sh
#
# The installed package directory is root-owned and not writable by any service account, so neither the code nor
# the read-only provider rules (paperwatch/network/provider_rules.toml) can be changed by the services.
# Dependencies are installed once, here, with the exact versions and hashes from uv.lock. Nothing is downloaded at
# run time. Rerun this script after pulling new code.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo (keep your PATH so uv is found: sudo env \"PATH=\$PATH\" $0)" >&2; exit 1; }
INSTALL_DIR="${PAPERWATCH_INSTALL_DIR:-/opt/paperwatch}"
REPOSITORY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV_BIN="${UV:-$(command -v uv || true)}"
[[ -n "$UV_BIN" ]] || { echo "uv not found; pass UV=/path/to/uv" >&2; exit 1; }

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

# Pinned, hash-checked dependency list from the lock file (no dev dependencies, not the project itself).
"$UV_BIN" export --project "$REPOSITORY_DIR" --frozen --no-dev --no-emit-project --format requirements-txt \
    --output-file "$BUILD_DIR/requirements.txt" >/dev/null

python3 -m venv --clear "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --quiet --no-cache-dir --require-hashes -r "$BUILD_DIR/requirements.txt"
"$INSTALL_DIR/venv/bin/pip" install --quiet --no-cache-dir --no-deps "$REPOSITORY_DIR"

chown -R root:root "$INSTALL_DIR"
chmod -R go-w "$INSTALL_DIR"
ln -sf "$INSTALL_DIR/venv/bin/paperwatch" /usr/local/bin/paperwatch

RULES_FILE="$("$INSTALL_DIR/venv/bin/python" -c \
    'from paperwatch.network.provider_rules import packaged_provider_rules_path as p; print(p())')"
echo "installed paperwatch to $INSTALL_DIR (read-only); provider rules: $RULES_FILE"
echo "command: /usr/local/bin/paperwatch"
