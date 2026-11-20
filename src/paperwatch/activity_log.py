"""What each paperwatch process is doing right now, and what it did recently (read by `paperwatch monitor`).

Every long-running step writes one row to the `activity_log` table when it starts and fills in `finished_at` and
`outcome` when it ends, so a row with no `finished_at` is a step in progress. Steps nest: a collect run contains its
stages; a worker run contains its jobs, a job its reading stages, and a stage its model calls. The nesting is
tracked with a context variable, so a caller only says what it is doing, never who its parent is:

    with recorded_activity(connection, "collect", {"trigger": "timer"}, component="collect"):   # a run
        with recorded_activity(connection, "screening"):                                       # a stage of it
            ...

Rules (the log is an aid for the operator, never a dependency of the pipeline):
  * Writing the log can never fail the step it describes: every database error is caught and logged.
  * Rows hold ids and counts, never paper text; the monitor joins ids to titles when it displays them.
  * A crashed process leaves rows without `finished_at`. The monitor recognises them by their process id (dead,
    or reused since the boot recorded with the row) and shows them as interrupted.
  * Rows older than RETENTION_DAYS are pruned when a collect run starts.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from paperwatch.clock import format_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.identifiers import new_internal_id
from paperwatch.service_log_redaction import redact_service_log_text

LOGGER = logging.getLogger(__name__)

ACTIVITY_ID_PREFIX = "a"
RETENTION_DAYS = 30
MAXIMUM_NOTE_CHARACTERS = 300
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

OUTCOME_OK = "ok"
OUTCOME_FAILED = "failed"
OUTCOME_STOPPED = "stopped"  # KeyboardInterrupt / SIGTERM while the step ran

# Components (the process that writes the rows). A step inherits its parent's component.
COMPONENT_COLLECT = "collect"
COMPONENT_WORKER = "work"
COMPONENT_REPORT = "report"
COMPONENT_OTHER = "other"  # e.g. the healthcheck call of `doctor`


@dataclass(frozen=True)
class _OpenActivity:
    activity_id: str
    component: str


_current_activity: ContextVar[_OpenActivity | None] = ContextVar("paperwatch_current_activity", default=None)


def read_boot_id() -> str | None:
    """The kernel's id of this boot: with the process id it identifies a process across reboots."""
    try:
        return BOOT_ID_PATH.read_text().strip() or None
    except OSError:
        return None


def start_activity(connection: sqlite3.Connection, step: str, detail: dict[str, Any] | None = None, *,
                   component: str | None = None) -> str | None:
    """Insert the row of a step that starts now, as a child of the current step. Returns its id (None if the row
    could not be written; the step goes on regardless)."""
    parent = _current_activity.get()
    resolved_component = component or (parent.component if parent else COMPONENT_OTHER)
    activity_id = new_internal_id(ACTIVITY_ID_PREFIX)
    try:
        connection.execute(
            """INSERT INTO activity_log (activity_id, parent_activity_id, component, step, detail_json, process_id,
                   boot_id, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (activity_id, parent.activity_id if parent else None, resolved_component, step,
             json.dumps(detail or {}, ensure_ascii=False, sort_keys=True), os.getpid(), read_boot_id(),
             utc_now_timestamp()))
    except sqlite3.Error as write_error:
        LOGGER.warning("activity log: could not record the start of %s: %s", step, write_error)
        return None
    return activity_id


def finish_activity(connection: sqlite3.Connection, activity_id: str | None, outcome: str, *,
                    note: str | None = None, detail_update: dict[str, Any] | None = None) -> None:
    """Mark a step finished. `detail_update` merges counts known only at the end (e.g. papers promoted)."""
    if activity_id is None:
        return
    try:
        if detail_update:
            row = connection.execute("SELECT detail_json FROM activity_log WHERE activity_id = ?",
                                     (activity_id,)).fetchone()
            merged_detail = {**(json.loads(row[0]) if row else {}), **detail_update}
            connection.execute("UPDATE activity_log SET detail_json = ? WHERE activity_id = ?",
                               (json.dumps(merged_detail, ensure_ascii=False, sort_keys=True), activity_id))
        connection.execute(
            "UPDATE activity_log SET finished_at = ?, outcome = ?, note = ? WHERE activity_id = ?",
            (utc_now_timestamp(), outcome, _shorten_note(note), activity_id))
    except sqlite3.Error as write_error:
        LOGGER.warning("activity log: could not record the end of %s: %s", activity_id, write_error)


@contextmanager
def recorded_activity(connection: sqlite3.Connection, step: str, detail: dict[str, Any] | None = None, *,
                      component: str | None = None) -> Iterator[ActivityHandle]:
    """Record a step around a block. The block's exception (if any) becomes the outcome and is re-raised."""
    parent = _current_activity.get()
    activity_id = start_activity(connection, step, detail, component=component)
    handle = ActivityHandle(activity_id)
    # Children attach to this step; if its row could not be written, they attach to the parent instead.
    token = None if activity_id is None else _current_activity.set(_OpenActivity(
        activity_id, component or (parent.component if parent else COMPONENT_OTHER)))
    try:
        yield handle
    except (KeyboardInterrupt, SystemExit):
        finish_activity(connection, activity_id, OUTCOME_STOPPED, detail_update=handle.detail_update)
        raise
    except BaseException as step_error:
        finish_activity(connection, activity_id, OUTCOME_FAILED, note=f"{type(step_error).__name__}: {step_error}",
                        detail_update=handle.detail_update)
        raise
    else:
        finish_activity(connection, activity_id, handle.outcome, note=handle.note,
                        detail_update=handle.detail_update)
    finally:
        if token is not None:
            _current_activity.reset(token)


@dataclass
class ActivityHandle:
    """Lets a block report how its step ended without raising (e.g. a job that failed and will retry)."""
    activity_id: str | None
    outcome: str = OUTCOME_OK
    note: str | None = None
    detail_update: dict[str, Any] | None = None

    def end_as(self, outcome: str, note: str | None = None) -> None:
        self.outcome, self.note = outcome, note

    def add_detail(self, **detail_values: Any) -> None:
        self.detail_update = {**(self.detail_update or {}), **detail_values}


def prune_old_activity(connection: sqlite3.Connection) -> int:
    """Delete finished rows older than RETENTION_DAYS (unfinished ones too, once that old: a crash long ago)."""
    cutoff_text = format_utc_timestamp(utc_now() - timedelta(days=RETENTION_DAYS))
    try:
        return connection.execute("DELETE FROM activity_log WHERE started_at < ?", (cutoff_text,)).rowcount
    except sqlite3.Error as write_error:
        LOGGER.warning("activity log: could not prune old rows: %s", write_error)
        return 0


def _shorten_note(note: str | None) -> str | None:
    """A note may quote an error that contains paper text or a URL: redact it like a service log line, and cap it."""
    if not note:
        return None
    redacted_note = redact_service_log_text(note)
    if len(redacted_note) <= MAXIMUM_NOTE_CHARACTERS:
        return redacted_note
    return redacted_note[:MAXIMUM_NOTE_CHARACTERS - 1] + "…"
