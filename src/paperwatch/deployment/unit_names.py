"""Names of the systemd units of the permanent daily schedule and the monitor (the pilot's are in one_day_pilot.py).

Shared by the unit renderer (daily_schedule.py), the setup script (which removes exactly these when the schedule is
switched off), and the monitor (which reads their timers and states).
"""

from __future__ import annotations

DAILY_UNIT_PREFIX = "paperwatch-"
DAILY_COLLECT_SERVICE = f"{DAILY_UNIT_PREFIX}collect.service"
DAILY_COLLECT_TIMER = f"{DAILY_UNIT_PREFIX}collect.timer"
DAILY_WORKER_SERVICE = f"{DAILY_UNIT_PREFIX}worker.service"
DAILY_REPORT_SERVICE = f"{DAILY_UNIT_PREFIX}report.service"
DAILY_REPORT_TIMER = f"{DAILY_UNIT_PREFIX}report.timer"

DAILY_SERVICE_NAMES = (DAILY_COLLECT_SERVICE, DAILY_WORKER_SERVICE, DAILY_REPORT_SERVICE)
DAILY_TIMER_NAMES = (DAILY_COLLECT_TIMER, DAILY_REPORT_TIMER)
ALL_DAILY_UNIT_NAMES = DAILY_SERVICE_NAMES + DAILY_TIMER_NAMES

# The monitor runs as the operator (a user unit), never as a service account: it only reads.
MONITOR_SERVICE = f"{DAILY_UNIT_PREFIX}monitor.service"
