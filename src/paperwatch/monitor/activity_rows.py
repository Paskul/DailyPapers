"""The activity log (schema v10) as a tree of steps, with live and interrupted steps told apart.

A step without `finished_at` is either running now or was cut off by a crash or a kill. It is running only if its
process still exists in this boot: the row stores the process id and the kernel's boot id. Interrupted steps get an
estimated end (the last moment anything below them happened) so the timeline can still draw them.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from paperwatch.activity_log import read_boot_id
from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp

PROC_DIR = Path("/proc")


@dataclass
class ActivityStep:
    activity_id: str
    parent_activity_id: str | None
    component: str
    step: str
    detail: dict[str, Any]
    process_id: int
    boot_id: str | None
    started_at: datetime
    finished_at: datetime | None
    outcome: str | None
    note: str | None
    process_alive: bool = False
    children: list[ActivityStep] = field(default_factory=list)

    @property
    def is_running(self) -> bool:
        return self.finished_at is None and self.process_alive

    @property
    def is_interrupted(self) -> bool:
        return self.finished_at is None and not self.process_alive

    def latest_moment(self) -> datetime:
        """The last start or finish anywhere in this subtree (the estimated end of an interrupted step)."""
        moments = [self.started_at] + ([self.finished_at] if self.finished_at else [])
        moments.extend(child.latest_moment() for child in self.children)
        return max(moments)

    def running_chain(self) -> list[ActivityStep]:
        """This step and its newest running descendants, outermost first: run -> job -> stage -> model call."""
        chain = [self]
        current = self
        while True:
            running_children = [child for child in current.children if child.is_running]
            if not running_children:
                return chain
            current = max(running_children, key=lambda child: child.started_at)
            chain.append(current)

    def as_json(self, *, now: datetime) -> dict[str, Any]:
        end_moment = self.finished_at or (now if self.is_running else self.latest_moment())
        return {"step": self.step, "component": self.component, "detail": self.detail,
                "start": format_utc_timestamp(self.started_at), "end": format_utc_timestamp(end_moment),
                "open": self.finished_at is None, "running": self.is_running, "interrupted": self.is_interrupted,
                "outcome": self.outcome or ("running" if self.is_running else "interrupted"),
                "note": self.note}


@dataclass
class ActivityTree:
    steps_by_id: dict[str, ActivityStep]
    roots: list[ActivityStep]  # whole runs (or steps whose run is older than the window), oldest first

    def running_roots(self, component: str) -> list[ActivityStep]:
        return [root for root in self.roots if root.component == component and root.is_running]

    def roots_of(self, component: str, step: str | None = None) -> list[ActivityStep]:
        return [root for root in self.roots if root.component == component and (step is None or root.step == step)]

    def all_steps(self, step: str) -> list[ActivityStep]:
        return [activity for activity in self.steps_by_id.values() if activity.step == step]


def load_activity_tree(connection: sqlite3.Connection, window_start: datetime, *, proc_dir: Path = PROC_DIR,
                       current_boot_id: str | None = None) -> ActivityTree:
    """Every step that overlaps the window, plus every unfinished one, linked into trees."""
    window_text = format_utc_timestamp(window_start)
    rows = connection.execute(
        """SELECT * FROM activity_log WHERE started_at >= ? OR finished_at IS NULL OR finished_at >= ?
           ORDER BY started_at, rowid""", (window_text, window_text)).fetchall()
    current_boot_id = current_boot_id if current_boot_id is not None else read_boot_id()
    liveness_cache: dict[tuple[int, str | None], bool] = {}
    steps_by_id: dict[str, ActivityStep] = {}
    for row in rows:
        process_key = (row["process_id"], row["boot_id"])
        if process_key not in liveness_cache:
            liveness_cache[process_key] = _is_process_alive(row["process_id"], row["boot_id"], current_boot_id,
                                                            proc_dir)
        steps_by_id[row["activity_id"]] = ActivityStep(
            activity_id=row["activity_id"], parent_activity_id=row["parent_activity_id"],
            component=row["component"], step=row["step"], detail=_parse_detail(row["detail_json"]),
            process_id=row["process_id"], boot_id=row["boot_id"],
            started_at=parse_utc_timestamp(row["started_at"]),
            finished_at=parse_utc_timestamp(row["finished_at"]) if row["finished_at"] else None,
            outcome=row["outcome"], note=row["note"], process_alive=liveness_cache[process_key])
    roots: list[ActivityStep] = []
    for activity in steps_by_id.values():
        parent = steps_by_id.get(activity.parent_activity_id) if activity.parent_activity_id else None
        (parent.children if parent else roots).append(activity)
    return ActivityTree(steps_by_id=steps_by_id, roots=roots)


def _is_process_alive(process_id: int, recorded_boot_id: str | None, current_boot_id: str | None,
                      proc_dir: Path) -> bool:
    if recorded_boot_id and current_boot_id and recorded_boot_id != current_boot_id:
        return False  # written before a reboot: whatever has this process id now is another process
    return (proc_dir / str(process_id)).exists()


def _parse_detail(detail_json: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(detail_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
