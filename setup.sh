#!/usr/bin/env bash
# Install or update paperwatch from ONE settings file: settings/paperwatch.toml.
#
#   ./setup.sh                  install / update everything the settings file describes (safe to re-run)
#   ./setup.sh --check          validate the settings and show the plan; change nothing
#   ./setup.sh --settings-only  only re-apply settings/paperwatch.toml (topics, config, keys, seeds), the schedule,
#                               and the monitor: the quick path after editing topics
#   ./setup.sh --settings FILE  use another settings file
#
# Run it as YOURSELF (not with sudo). In mode "hardened" it asks for sudo where a step needs root; in mode "local"
# it never does. The first run without settings/paperwatch.toml creates it (from the example, or from a former
# settings/config.toml + settings/topics.toml pair) and stops so you can edit it.
#
# Steps, hardened mode (each one skips work that is already done):
#    1. validate the settings with the application's own loader (nothing changes if they are invalid)
#    2. install the application read-only in /opt/paperwatch            (deploy/install_application.sh)
#    3. create the home, the service accounts, and permissions           (deploy/setup_service_accounts.sh)
#    4. install the pinned screening model                               (deploy/install_embedding_model.sh)
#    5. build the sandboxed PDF extractor if its image is missing        (deploy/build_extractor_image.sh)
#    6. install config.toml, topics.toml, API keys, and seed files into the home
#    7. create or migrate the database (as the worker account)
#    8. load the firewall rules                                          (deploy/apply_egress_rules.sh)
#    9. verify: access limits from both accounts, then `doctor`
#   10. daily schedule on or off, as [setup] daily_schedule says
#   11. monitor service on or off, as [setup] monitor_service says
# Local mode: the same without 2, 3, 8, and the access check; everything runs from this repository as you.
set -euo pipefail

REPOSITORY_DIR="$(cd "$(dirname "$0")" && pwd)"
SETTINGS_FILE="$REPOSITORY_DIR/settings/paperwatch.toml"
CHECK_ONLY=0
SETTINGS_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --settings-only) SETTINGS_ONLY=1; shift ;;
        --settings) SETTINGS_FILE="$(realpath "$2")"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
    esac
done

[[ $EUID -ne 0 ]] || { echo "run ./setup.sh as yourself, not with sudo (it asks for sudo where needed)" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is not installed: https://docs.astral.sh/uv/" >&2; exit 1; }
SETTINGS_DIR="$(dirname "$SETTINGS_FILE")"
OPERATOR_ACCOUNT="$(id -un)"
INSTALL_DIR=/opt/paperwatch
SYSTEM_UNIT_DIR=/etc/systemd/system
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SAVED_EXTRACTOR_IMAGE_FILE="$SETTINGS_DIR/.extractor-image-id"
DAILY_TIMERS=(paperwatch-collect.timer paperwatch-report.timer)
DAILY_UNITS=(paperwatch-collect.service paperwatch-collect.timer paperwatch-worker.service paperwatch-report.service
             paperwatch-report.timer)
MONITOR_UNIT=paperwatch-monitor.service

step() { echo; echo "== $*"; }
note() { echo "   $*"; }
warn() { echo "   WARNING: $*" >&2; }
repo_python() { uv run --project "$REPOSITORY_DIR" --frozen --quiet python "$@"; }

# ---- 0. the settings file -------------------------------------------------------------------------------------------
if [[ ! -f "$SETTINGS_FILE" ]]; then
    (cd "$REPOSITORY_DIR" && uv sync --frozen --quiet)
    if [[ -f "$SETTINGS_DIR/config.toml" && -f "$SETTINGS_DIR/topics.toml" ]]; then
        repo_python -m paperwatch.deployment.settings_file migrate "$SETTINGS_DIR/config.toml" \
            "$SETTINGS_DIR/topics.toml" "$SETTINGS_FILE"
        echo "Your former settings/config.toml and settings/topics.toml are now one file: $SETTINGS_FILE"
        echo "(the two old files are no longer read; delete them when you are happy with the new one)."
    else
        (umask 077; repo_python -m paperwatch.deployment.settings_file write-example "$SETTINGS_FILE")
        echo "Created $SETTINGS_FILE from the example."
    fi
    echo "Review its [setup] table (home, mode) and your [[topics]], then run ./setup.sh again."
    exit 0
fi
(cd "$REPOSITORY_DIR" && uv sync --frozen --quiet)

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT
chmod 0755 "$STAGE_DIR"   # the service accounts read the staged config during `init` / unit rendering
stage_settings() {  # [--extractor-image ID]
    repo_python -m paperwatch.deployment.settings_file stage "$SETTINGS_FILE" "$STAGE_DIR" "$@"
    # shellcheck source=/dev/null
    source "$STAGE_DIR/setup.env"
}

step "1. validate $SETTINGS_FILE"
stage_settings
PAPERWATCH_HOME="$(realpath -m "$PAPERWATCH_HOME")"
if [[ "$SETUP_MODE" == hardened ]]; then
    PAPERWATCH_BIN="$INSTALL_DIR/venv/bin/paperwatch"
    PAPERWATCH_PYTHON="$INSTALL_DIR/venv/bin/python"
else
    PAPERWATCH_BIN="$REPOSITORY_DIR/.venv/bin/paperwatch"
    PAPERWATCH_PYTHON="$REPOSITORY_DIR/.venv/bin/python"
fi
note "daily schedule: $([[ $DAILY_SCHEDULE == 1 ]] && echo on || echo off); monitor service: \
$([[ $MONITOR_SERVICE == 1 ]] && echo "on ($MONITOR_LISTEN_ADDRESS:$MONITOR_PORT)" || echo off)"

if [[ "$SETUP_MODE" == hardened ]] && compgen -G "$SYSTEM_UNIT_DIR/paperwatch-pilot-*" >/dev/null; then
    message="the one-day pilot is installed (its units are in $SYSTEM_UNIT_DIR); changing the deployment under it \
would disturb it. Wait until it removes itself, or stop it: sudo deploy/pilot/stop_one_day_pilot.sh"
    if [[ $CHECK_ONLY == 1 ]]; then warn "$message"; else echo "refusing: $message" >&2; exit 1; fi
fi
if [[ $CHECK_ONLY == 1 ]]; then
    echo
    echo "Settings are valid. ./setup.sh would set up $PAPERWATCH_HOME in $SETUP_MODE mode (nothing was changed)."
    exit 0
fi

# ---- helpers ------------------------------------------------------------------------------------------------------
SETTINGS_CHANGED=0
install_settings_file() {  # <source> <destination>: hardened: you + group read; local: you only
    local source_file="$1" destination_file="$2"
    if [[ -f "$destination_file" ]] && cmp -s "$source_file" "$destination_file"; then
        note "unchanged  $destination_file"; return
    fi
    SETTINGS_CHANGED=1
    if [[ "$SETUP_MODE" == local ]]; then
        install -D -m 0600 "$source_file" "$destination_file"
    elif [[ -w "$destination_file" ]]; then
        cat "$source_file" > "$destination_file"   # keeps owner, group, and mode
    else
        sudo install -o "$OPERATOR_ACCOUNT" -g "$SHARED_GROUP" -m 0640 "$source_file" "$destination_file"
    fi
    note "installed  $destination_file"
}

run_paperwatch() {  # hardened: as the worker account with a shared umask; local: as you
    if [[ "$SETUP_MODE" == hardened ]]; then
        (cd / && sudo -u "$WORKER_ACCOUNT" sh -c 'umask 0007; exec "$@"' paperwatch "$PAPERWATCH_BIN" \
            --home "$PAPERWATCH_HOME" "$@")
    else
        "$PAPERWATCH_BIN" --home "$PAPERWATCH_HOME" "$@"
    fi
}

extractor_image_present() {  # <image id>
    [[ -n "$1" ]] || return 1
    if [[ "$SETUP_MODE" == hardened ]]; then
        local collector_uid; collector_uid="$(id -u "$COLLECTOR_ACCOUNT")"
        (cd / && sudo -u "$COLLECTOR_ACCOUNT" -H env "XDG_RUNTIME_DIR=/run/user/$collector_uid" \
            podman image exists "$1")
    else
        podman image exists "$1"
    fi
}

install_settings() {
    install_settings_file "$STAGE_DIR/config.toml" "$PAPERWATCH_HOME/config.toml"
    install_settings_file "$STAGE_DIR/topics.toml" "$PAPERWATCH_HOME/topics.toml"
    if [[ -n "$SEEDS_DIR" ]]; then
        if [[ "$SETUP_MODE" == hardened && ! -d "$PAPERWATCH_HOME/seeds" ]]; then
            sudo install -d -o "$OPERATOR_ACCOUNT" -g "$SHARED_GROUP" -m 0750 "$PAPERWATCH_HOME/seeds"
        fi
        while IFS= read -r -d '' seed_file; do
            install_settings_file "$seed_file" "$PAPERWATCH_HOME/seeds/$(basename "$seed_file")"
        done < <(find "$SEEDS_DIR" -maxdepth 1 -type f -print0)
    fi
    local credentials_destination="$PAPERWATCH_HOME/credentials/collector.toml"
    if [[ -z "$CREDENTIALS_FILE" ]]; then
        note "skipped    API keys (no credentials file; any installed one is kept)"
    elif [[ "$SETUP_MODE" == local ]]; then
        if cmp -s "$CREDENTIALS_FILE" "$credentials_destination" 2>/dev/null; then
            note "unchanged  $credentials_destination"
        else
            install -D -m 0600 "$CREDENTIALS_FILE" "$credentials_destination"; chmod 0700 "$PAPERWATCH_HOME/credentials"
            note "installed  $credentials_destination (mode 0600)"
        fi
    elif sudo cmp -s "$CREDENTIALS_FILE" "$credentials_destination" 2>/dev/null; then
        note "unchanged  $credentials_destination"
    else
        sudo install -o "$COLLECTOR_ACCOUNT" -g "$COLLECTOR_ACCOUNT" -m 0600 "$CREDENTIALS_FILE" \
            "$credentials_destination"
        note "installed  $credentials_destination (only $COLLECTOR_ACCOUNT can read it)"
    fi
}

build_extractor_if_missing() {
    if [[ "$BUILD_EXTRACTOR_IMAGE" != 1 ]]; then
        [[ -n "$CONFIGURED_EXTRACTOR_IMAGE" ]] || warn "no extractor image: papers are read from their abstracts"
        return
    fi
    if ! command -v podman >/dev/null; then
        warn "Podman is not installed (sudo apt install podman), so there is no PDF extractor: papers are read from \
their abstracts. Re-run ./setup.sh after installing it."
        return
    fi
    local image_id="$CONFIGURED_EXTRACTOR_IMAGE"
    if [[ -z "$image_id" && -f "$SAVED_EXTRACTOR_IMAGE_FILE" ]]; then image_id="$(cat "$SAVED_EXTRACTOR_IMAGE_FILE")"; fi
    if extractor_image_present "$image_id"; then
        note "extractor image ${image_id:0:12}… is present"
    else
        note "building the extractor image (needs the network once, for the Debian base image)"
        local build_account=""; [[ "$SETUP_MODE" == hardened ]] && build_account="$COLLECTOR_ACCOUNT"
        local build_output
        build_output="$("$REPOSITORY_DIR/deploy/build_extractor_image.sh" "$build_account" | tee /dev/stderr)"
        image_id="$(grep -oE 'image = "[0-9a-f]{64}"' <<<"$build_output" | tail -n 1 | grep -oE '[0-9a-f]{64}')"
        [[ -n "$image_id" ]] || { echo "could not read the built image id" >&2; exit 1; }
        echo "$image_id" > "$SAVED_EXTRACTOR_IMAGE_FILE"
    fi
    if [[ "$image_id" != "$CONFIGURED_EXTRACTOR_IMAGE" ]]; then
        note "using extractor image ${image_id:0:12}… (written into the installed config.toml; to pin it in your \
settings file, set [extractor] image = \"$image_id\")"
        stage_settings --extractor-image "$image_id" >/dev/null
    fi
}

systemctl_for_scope() {  # system units with sudo (hardened), user units without (local)
    if [[ "$SETUP_MODE" == hardened ]]; then sudo systemctl "$@"; else systemctl --user "$@"; fi
}

apply_daily_schedule() {
    local unit_dir="$SYSTEM_UNIT_DIR" scope=system
    [[ "$SETUP_MODE" == local ]] && unit_dir="$USER_UNIT_DIR" && scope=user
    if [[ "$DAILY_SCHEDULE" == 1 ]]; then
        local render_dir; render_dir="$(mktemp -d)"
        local uid_argument=()
        [[ "$scope" == system ]] && uid_argument=(--collector-uid "$(id -u "$COLLECTOR_ACCOUNT")")
        "$PAPERWATCH_PYTHON" -m paperwatch.deployment.daily_schedule daily --home "$PAPERWATCH_HOME" \
            --unit-dir "$render_dir" --scope "$scope" --paperwatch-executable "$PAPERWATCH_BIN" "${uid_argument[@]}" \
            >/dev/null
        if [[ "$scope" == system ]]; then
            systemd-analyze verify "$render_dir"/*.service "$render_dir"/*.timer
            sudo install -m 0644 "$render_dir"/* "$unit_dir/"
        else
            mkdir -p "$unit_dir"; install -m 0644 "$render_dir"/* "$unit_dir/"
        fi
        rm -rf "$render_dir"
        systemctl_for_scope daemon-reload
        systemctl_for_scope enable --now "${DAILY_TIMERS[@]}" >/dev/null
        systemctl_for_scope enable paperwatch-worker.service >/dev/null
        if [[ "$SETTINGS_CHANGED" == 1 ]]; then
            systemctl_for_scope restart paperwatch-worker.service   # pick up the new settings (a job resumes)
        else
            systemctl_for_scope start paperwatch-worker.service
        fi
        note "daily schedule on: collect at the [schedule] collect_times, report at the [report] publish_time"
        systemctl_for_scope list-timers --no-pager "${DAILY_TIMERS[@]}" | sed 's/^/   /'
        if [[ "$scope" == user ]] && [[ "$(loginctl show-user "$OPERATOR_ACCOUNT" -p Linger --value 2>/dev/null)" != yes ]]; then
            warn "your user services stop when you log out; to keep them running: loginctl enable-linger"
        fi
    else
        local installed=()
        for unit_name in "${DAILY_UNITS[@]}"; do [[ -f "$unit_dir/$unit_name" ]] && installed+=("$unit_name"); done
        if (( ${#installed[@]} )); then
            systemctl_for_scope disable --now "${installed[@]}" >/dev/null 2>&1 || true
            if [[ "$scope" == system ]]; then sudo rm -f "${installed[@]/#/$unit_dir/}"; else rm -f "${installed[@]/#/$unit_dir/}"; fi
            systemctl_for_scope daemon-reload
            note "daily schedule removed (${#installed[@]} unit(s)); nothing runs on its own now"
        else
            note "daily schedule off: nothing runs on its own (run collect / work / report yourself)"
        fi
    fi
}

apply_monitor_service() {
    local unit_path="$USER_UNIT_DIR/$MONITOR_UNIT"
    if [[ "$MONITOR_SERVICE" == 1 ]]; then
        local render_dir; render_dir="$(mktemp -d)"
        "$PAPERWATCH_PYTHON" -m paperwatch.deployment.daily_schedule monitor --home "$PAPERWATCH_HOME" \
            --unit-dir "$render_dir" --scope user --paperwatch-executable "$PAPERWATCH_BIN" \
            --port "$MONITOR_PORT" --listen-address "$MONITOR_LISTEN_ADDRESS" >/dev/null
        mkdir -p "$USER_UNIT_DIR"
        install -m 0644 "$render_dir/$MONITOR_UNIT" "$unit_path"
        rm -rf "$render_dir"
        systemctl --user daemon-reload
        systemctl --user enable "$MONITOR_UNIT" >/dev/null
        systemctl --user restart "$MONITOR_UNIT"
        note "monitor running: http://$MONITOR_LISTEN_ADDRESS:$MONITOR_PORT/"
    elif [[ -f "$unit_path" ]]; then
        systemctl --user disable --now "$MONITOR_UNIT" >/dev/null 2>&1 || true
        rm -f "$unit_path"; systemctl --user daemon-reload
        note "monitor service removed (run it by hand: $PAPERWATCH_BIN --home $PAPERWATCH_HOME monitor)"
    else
        note "monitor service off (run it by hand: $PAPERWATCH_BIN --home $PAPERWATCH_HOME monitor)"
    fi
}

# ---- the steps ----------------------------------------------------------------------------------------------------
if [[ $SETTINGS_ONLY == 1 ]]; then
    [[ -f "$PAPERWATCH_HOME/config.toml" ]] || { echo "$PAPERWATCH_HOME is not set up yet; run ./setup.sh first" >&2; exit 1; }
    step "settings -> $PAPERWATCH_HOME"; install_settings
    step "schedule"; apply_daily_schedule
    step "monitor"; apply_monitor_service
    echo; echo "Settings applied. Check them with: $PAPERWATCH_BIN --home $PAPERWATCH_HOME doctor"
    exit 0
fi

if [[ "$SETUP_MODE" == hardened ]]; then
    step "2. application -> $INSTALL_DIR (read-only)"
    sudo env "PATH=$PATH" "$REPOSITORY_DIR/deploy/install_application.sh"

    step "3. home, service accounts, permissions"
    if [[ ! -f "$PAPERWATCH_HOME/config.toml" ]]; then
        sudo install -d -m 0750 "$PAPERWATCH_HOME"
        sudo install -o "$OPERATOR_ACCOUNT" -m 0640 "$STAGE_DIR/config.toml" "$PAPERWATCH_HOME/config.toml"
        sudo install -o "$OPERATOR_ACCOUNT" -m 0640 "$STAGE_DIR/topics.toml" "$PAPERWATCH_HOME/topics.toml"
    fi
    sudo "$REPOSITORY_DIR/deploy/setup_service_accounts.sh" "$PAPERWATCH_HOME" "$OPERATOR_ACCOUNT" >/dev/null
    note "accounts $COLLECTOR_ACCOUNT and $WORKER_ACCOUNT, group $SHARED_GROUP (you are a member)"
else
    step "2. home $PAPERWATCH_HOME (local mode: everything runs as you)"
    mkdir -p "$PAPERWATCH_HOME"
fi

step "4. screening model"
if [[ "$INSTALL_EMBEDDING_MODEL" == 1 ]]; then
    "$REPOSITORY_DIR/deploy/install_embedding_model.sh" "$PAPERWATCH_HOME" "$EMBEDDING_MODEL_NAME" | sed 's/^/   /'
else
    note "skipped ([setup] install_embedding_model = false, or a non-ONNX embedding backend)"
fi

step "5. PDF extractor"
build_extractor_if_missing

step "6. settings -> $PAPERWATCH_HOME"
install_settings

step "7. database"
run_paperwatch init | sed -e '/^$/d' -e '/^Next: /d' -e '/home ready at/d' -e 's/^/   /'

if [[ "$SETUP_MODE" == hardened ]]; then
    step "8. firewall (nftables egress rules)"
    sudo "$REPOSITORY_DIR/deploy/apply_egress_rules.sh" --model-port "$MODEL_PORT" | sed 's/^/   /'
    step "9. verify"
    sudo "$REPOSITORY_DIR/deploy/check_access_as_service_accounts.sh" "$PAPERWATCH_HOME" \
        || { echo "access-check FAILED: see above. Nothing was scheduled." >&2; exit 1; }
else
    step "9. verify"
fi
run_paperwatch doctor || warn "doctor reported a FAIL above (e.g. the model server is not running); fix it and \
re-run, or check later with: $PAPERWATCH_BIN --home $PAPERWATCH_HOME doctor"

step "10. schedule"; apply_daily_schedule
step "11. monitor"; apply_monitor_service

cat <<EOF

paperwatch is set up in $PAPERWATCH_HOME ($SETUP_MODE mode, topics: ${TOPIC_IDS:-none}).
  status       $PAPERWATCH_BIN --home $PAPERWATCH_HOME status
  monitor      $PAPERWATCH_BIN --home $PAPERWATCH_HOME monitor      (http://127.0.0.1:$MONITOR_PORT)
  run by hand  $PAPERWATCH_BIN --home $PAPERWATCH_HOME collect | work | report
  reports      $PAPERWATCH_HOME/output/daily/
After editing settings/paperwatch.toml:  ./setup.sh --settings-only   (or ./setup.sh for everything)
EOF
