#!/usr/bin/env bash
# Install the operator's settings from settings/ into the paperwatch home. Run as YOURSELF (the operator, a member of
# the paperwatch-state group), not with sudo: the script asks for sudo only where a service account must own a file.
#
# Usage: deploy/apply_settings.sh [paperwatch home]        (default: /srv/paperwatch)
#
#   settings/config.toml                 -> <home>/config.toml               you : paperwatch-state 0640
#   settings/topics.toml                 -> <home>/topics.toml               you : paperwatch-state 0640
#   settings/collector_credentials.toml  -> <home>/credentials/collector.toml  paperwatch-collector only, 0600 (sudo)
#   settings/seeds/*                     -> <home>/seeds/                    you : paperwatch-state 0640 (dir via sudo)
#
# Every file is validated with this repository's paperwatch code first; nothing is installed if validation fails.
# Settings files that do not exist are skipped, and unchanged files are left alone. Safe to rerun.
set -euo pipefail

[[ $EUID -ne 0 ]] || { echo "run as your own account, not root (the script uses sudo where needed)" >&2; exit 1; }
REPOSITORY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SETTINGS_DIR="$REPOSITORY_DIR/settings"
PAPERWATCH_HOME="$(realpath "${1:-/srv/paperwatch}")"
COLLECTOR_ACCOUNT="${PAPERWATCH_COLLECTOR_ACCOUNT:-paperwatch-collector}"
SHARED_GROUP="${PAPERWATCH_SHARED_GROUP:-paperwatch-state}"
OPERATOR_ACCOUNT="$(id -un)"

[[ -f "$PAPERWATCH_HOME/config.toml" ]] || { echo "$PAPERWATCH_HOME has no config.toml; run 'paperwatch --home $PAPERWATCH_HOME init' first" >&2; exit 1; }

# ---- 1. validate (in a scratch home, with the repository's code) ------------------------------------------------
SCRATCH_HOME="$(mktemp -d)"
trap 'rm -rf "$SCRATCH_HOME"' EXIT
for settings_name in config.toml topics.toml; do
    if [[ -f "$SETTINGS_DIR/$settings_name" ]]; then
        cp "$SETTINGS_DIR/$settings_name" "$SCRATCH_HOME/$settings_name"
    else
        cp "$PAPERWATCH_HOME/$settings_name" "$SCRATCH_HOME/$settings_name"   # validate together with the live file
    fi
done
uv run --project "$REPOSITORY_DIR" --frozen --quiet python - "$SCRATCH_HOME" "$SETTINGS_DIR" <<'PYTHON'
import sys
from pathlib import Path

from paperwatch.config import load_app_config
from paperwatch.errors import PaperwatchError, describe_failure_for_operator
from paperwatch.network.collector_credentials import load_collector_credentials

scratch_home, settings_dir = Path(sys.argv[1]), Path(sys.argv[2])
try:
    app_config = load_app_config(scratch_home)
    credentials_path = settings_dir / "collector_credentials.toml"
    credentials = load_collector_credentials(credentials_path) if credentials_path.exists() else None
except PaperwatchError as failure:
    print(describe_failure_for_operator(failure), file=sys.stderr)
    sys.exit(1)

print(f"validated: {len(app_config.topics)} topic(s): " + ", ".join(topic.topic_id for topic in app_config.topics))
problems = []
for topic in app_config.topics:
    for seed_file in topic.csl_json_files:
        if not Path(seed_file).is_absolute() and not (settings_dir / seed_file).is_file():
            problems.append(f"topic '{topic.topic_id}': CSL JSON file {seed_file} is not in settings/ "
                            f"(expected {settings_dir / seed_file})")
uses_zotero = any(topic.zotero_collections for topic in app_config.topics)
if credentials is not None and uses_zotero and not (credentials.has("zotero", "api_key")
                                                    and credentials.has("zotero", "library_id")):
    problems.append("topics use zotero_collections, but settings/collector_credentials.toml lacks [zotero] "
                    "api_key or library_id")
for problem in problems:
    print(f"problem: {problem}", file=sys.stderr)
sys.exit(1 if problems else 0)
PYTHON

# ---- 2. install ---------------------------------------------------------------------------------------------------
install_operator_file() {  # <source> <destination>: you own it, the services read it through the shared group
    local source_file="$1" destination_file="$2"
    if [[ -f "$destination_file" ]] && cmp -s "$source_file" "$destination_file"; then
        echo "unchanged  $destination_file"
    elif [[ -w "$destination_file" ]]; then
        cat "$source_file" > "$destination_file"   # keeps the existing owner, group and mode
        echo "updated    $destination_file"
    else
        sudo install -o "$OPERATOR_ACCOUNT" -g "$SHARED_GROUP" -m 0640 "$source_file" "$destination_file"
        echo "installed  $destination_file"
    fi
}

for settings_name in config.toml topics.toml; do
    [[ -f "$SETTINGS_DIR/$settings_name" ]] && install_operator_file "$SETTINGS_DIR/$settings_name" \
        "$PAPERWATCH_HOME/$settings_name"
done

if [[ -d "$SETTINGS_DIR/seeds" ]] && [[ -n "$(ls -A "$SETTINGS_DIR/seeds")" ]]; then
    [[ -d "$PAPERWATCH_HOME/seeds" ]] || sudo install -d -o "$OPERATOR_ACCOUNT" -g "$SHARED_GROUP" -m 0750 \
        "$PAPERWATCH_HOME/seeds"
    while IFS= read -r -d '' seed_file; do
        install_operator_file "$seed_file" "$PAPERWATCH_HOME/seeds/$(basename "$seed_file")"
    done < <(find "$SETTINGS_DIR/seeds" -maxdepth 1 -type f -print0)
fi

CREDENTIALS_SOURCE="$SETTINGS_DIR/collector_credentials.toml"
CREDENTIALS_DESTINATION="$PAPERWATCH_HOME/credentials/collector.toml"
if [[ -f "$CREDENTIALS_SOURCE" ]]; then
    if sudo cmp -s "$CREDENTIALS_SOURCE" "$CREDENTIALS_DESTINATION" 2>/dev/null; then
        echo "unchanged  $CREDENTIALS_DESTINATION"
    else
        sudo install -o "$COLLECTOR_ACCOUNT" -g "$COLLECTOR_ACCOUNT" -m 0600 "$CREDENTIALS_SOURCE" \
            "$CREDENTIALS_DESTINATION"
        echo "installed  $CREDENTIALS_DESTINATION (owner $COLLECTOR_ACCOUNT, mode 0600)"
    fi
else
    echo "skipped    collector credentials (no settings/collector_credentials.toml; the live file is kept)"
fi
