"""One JSON-ready picture of paperwatch for the monitor page: now, the plan until the next report, and history.

The timeline spans [now - history_hours, horizon], where the horizon is the end of the next report run (and, during
the pilot, its finish), so the page shows yesterday evening, the night, and what is still to come on one axis.
Every timestamp is UTC in the stored format (2026-11-20T21:49:03Z); the page shows them in the report timezone.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from paperwatch import __version__
from paperwatch.clock import format_utc_timestamp
from paperwatch.config import AppConfig
from paperwatch.monitor.activity_rows import ActivityTree, load_activity_tree
from paperwatch.monitor.current_activity import describe_now
from paperwatch.monitor.daily_plan import describe_plan, next_report_cutoff_after
from paperwatch.monitor.database_access import existing_table_names, schema_version
from paperwatch.monitor.recent_history import describe_history, latest_collect_run
from paperwatch.monitor.schedule_state import ScheduleState

DEFAULT_HISTORY_HOURS = 24.0
MINIMUM_HORIZON_AHEAD = timedelta(hours=6)
HORIZON_AFTER_CUTOFF = timedelta(hours=2)  # the report run, and the pilot's finish 100 minutes after the cutoff
ACTIVITY_LOG_TABLE = "activity_log"


def build_monitor_snapshot(connection: sqlite3.Connection, app_config: AppConfig, *, now: datetime,
                           history_hours: float, schedule: ScheduleState) -> dict[str, Any]:
    window_start = now - timedelta(hours=history_hours)
    next_report_cutoff = next_report_cutoff_after(now, app_config)
    horizon_end = max(next_report_cutoff + HORIZON_AFTER_CUTOFF, now + MINIMUM_HORIZON_AHEAD)
    has_activity_log = ACTIVITY_LOG_TABLE in existing_table_names(connection)
    activity: ActivityTree | None = load_activity_tree(connection, window_start) if has_activity_log else None

    running_job_ids = [row["job_id"] for row in connection.execute(
        "SELECT job_id FROM jobs WHERE status = 'running' AND lease_expires_at > ? ORDER BY updated_at",
        (format_utc_timestamp(now),))]
    history = describe_history(connection, app_config, now=now, window_start=window_start, activity=activity,
                               running_job_ids=running_job_ids)
    return {
        "meta": {
            "generated_at": format_utc_timestamp(now),
            "version": __version__,
            "home": str(app_config.home_dir),
            "timezone": app_config.report.timezone,
            "schema_version": schema_version(connection),
            "activity_log": has_activity_log,
            "window_start": format_utc_timestamp(window_start),
            "horizon_end": format_utc_timestamp(horizon_end),
            "topics": [topic.topic_id for topic in app_config.topics],  # ids only: questions stay private
        },
        "now": describe_now(connection, app_config, now=now, activity=activity, schedule=schedule,
                            next_report_cutoff=next_report_cutoff,
                            last_collect_run=latest_collect_run(history["lanes"]["collect"])),
        "plan": describe_plan(connection, app_config, now=now, window_start=window_start, horizon_end=horizon_end,
                              schedule=schedule, next_report_cutoff=next_report_cutoff),
        "history": history,
    }
