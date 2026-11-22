#!/usr/bin/env bash
# Start the ONE-DAY pilot: collections today and tonight, the worker until the report cutoff, one
# report tomorrow morning, then the pilot removes itself. Nothing of it can run a second time (see
# src/paperwatch/deployment/one_day_pilot.py for the layers that guarantee this). Run with sudo, keeping PATH so uv
# is found:
#
#   sudo env "PATH=$PATH" deploy/pilot/install_one_day_pilot.sh [paperwatch home]     (default: /srv/paperwatch)
#
# Steps (it stops at the first failure, before any timer is enabled):
#   1. reinstall the application read-only in /opt/paperwatch (the code of this repository);
#   2. migrate the database (`init`, as the worker account);
#   3. verify the access limits from both service accounts (`access-check`), and the model server (`doctor`);
#   4. render the pilot's units into /etc/systemd/system and check them with systemd-analyze;
#   5. enable the three one-shot timers, run the first collection now (the worker starts with it), and show the
#      schedule.
# Stop the pilot early with: sudo deploy/pilot/stop_one_day_pilot.sh
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo: sudo env \"PATH=\$PATH\" $0" >&2; exit 1; }
REPOSITORY_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PAPERWATCH_HOME="$(realpath "${1:-/srv/paperwatch}")"
INSTALL_DIR=/opt/paperwatch
PAPERWATCH_BIN="$INSTALL_DIR/venv/bin/paperwatch"
UNIT_DIR=/etc/systemd/system
WORKER_ACCOUNT=paperwatch-worker
PILOT_TIMERS=(paperwatch-pilot-collect.timer paperwatch-pilot-report.timer paperwatch-pilot-finish.timer)

run_as_worker() {  # cd / : the service account cannot read the caller's directory; umask keeps WAL files shared
    (cd / && sudo -u "$WORKER_ACCOUNT" sh -c 'umask 0007; exec "$@"' paperwatch "$PAPERWATCH_BIN" \
        --home "$PAPERWATCH_HOME" "$@")
}

existing_pilot_units=("$UNIT_DIR"/paperwatch-pilot-*)  # without a match, the one literal pattern (no file)
if [[ -e "${existing_pilot_units[0]}" ]]; then
    echo "a pilot is already installed (${#existing_pilot_units[@]} unit files); stop it first:" >&2
    echo "  sudo $REPOSITORY_DIR/deploy/pilot/stop_one_day_pilot.sh" >&2
    exit 1
fi
# An earlier pilot may have left units in systemd's memory after deleting their files (a timer still "active", a
# killed worker "failed"); an "active" timer would ignore the new schedule, so clear them first.
for leftover_unit in $(systemctl list-units --all --no-legend --plain 'paperwatch-pilot-*' | awk '{print $1}'); do
    systemctl stop "$leftover_unit" 2>/dev/null || true
    systemctl reset-failed "$leftover_unit" 2>/dev/null || true
done
systemctl daemon-reload

echo "== 1. install the application read-only in $INSTALL_DIR"
"$REPOSITORY_DIR/deploy/install_application.sh"

echo "== 2. migrate the database"
run_as_worker init

echo "== 3. access limits as each service account, then the model server"
"$REPOSITORY_DIR/deploy/check_access_as_service_accounts.sh" "$PAPERWATCH_HOME" || {
    echo "access-check FAILED: the pilot was not started" >&2; exit 1; }
run_as_worker doctor || { echo "doctor FAILED: the pilot was not started" >&2; exit 1; }

echo "== 4. render the pilot units"
"$INSTALL_DIR/venv/bin/python" -m paperwatch.deployment.one_day_pilot --home "$PAPERWATCH_HOME" \
    --unit-dir "$UNIT_DIR" --paperwatch-executable "$PAPERWATCH_BIN"
systemd-analyze verify "$UNIT_DIR"/paperwatch-pilot-*.service "$UNIT_DIR"/paperwatch-pilot-*.timer

echo "== 5. enable the one-shot timers and run the first collection now"
systemctl daemon-reload
systemctl enable --now "${PILOT_TIMERS[@]}"
echo "(the first collection can take several minutes; the worker starts alongside it)"
systemctl start paperwatch-pilot-collect.service || echo "WARNING: the first collection failed; see below" >&2
journalctl --unit paperwatch-pilot-collect.service --since "-30 min" --no-pager --output cat | tail -n 40
sleep 5
echo
echo "worker: $(systemctl is-active paperwatch-pilot-worker.service)"
systemctl list-timers --all --no-pager 'paperwatch-pilot-*'
cat <<EOF

The pilot is running. Useful commands (as yourself):
  paperwatch --home $PAPERWATCH_HOME status          what was collected, queued, reviewed
  journalctl -u 'paperwatch-pilot-*' -f               live service logs (redacted)
  ls $PAPERWATCH_HOME/output/daily/                   the report appears here tomorrow after 07:00
Stop early: sudo $REPOSITORY_DIR/deploy/pilot/stop_one_day_pilot.sh
EOF
