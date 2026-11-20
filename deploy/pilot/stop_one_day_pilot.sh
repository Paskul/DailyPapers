#!/usr/bin/env bash
# Stop the one-day pilot NOW instead of waiting for its finish time: runs the pilot's own finish step, which stops
# the worker, disables the timers, and deletes every paperwatch-pilot-* unit. Nothing in the paperwatch home
# (database, notes, reports) is touched. Run with sudo:  sudo deploy/pilot/stop_one_day_pilot.sh
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
if [[ ! -f /etc/systemd/system/paperwatch-pilot-finish.service ]]; then
    echo "no pilot is installed (or it already finished)"
    exit 0
fi
systemctl start paperwatch-pilot-finish.service
echo "pilot stopped and removed; remaining paperwatch-pilot units: $(systemctl list-units --all --no-legend \
    'paperwatch-pilot-*' | wc -l)"
