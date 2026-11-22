"""Read-only access to the paperwatch database for the monitor.

The monitor never writes: the connection is opened with SQLite's `mode=ro` and `PRAGMA query_only`, so even a bug
cannot change the pipeline's state. It also never insists on the current schema version (unlike every other
command), because it must keep working while a deployed older version is running: features whose tables are missing
(the activity log before schema v10) are simply left out, and the snapshot says so.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from paperwatch.errors import DatabaseError

READ_TIMEOUT_SECONDS = 5.0


def open_read_only_database(database_path: Path) -> sqlite3.Connection:
    if not database_path.is_file():
        raise DatabaseError(
            f"database {database_path} does not exist", operator_hint="run setup (or `paperwatch init`) first"
        )
    try:
        connection = sqlite3.connect(
            f"{database_path.resolve().as_uri()}?mode=ro", uri=True, timeout=READ_TIMEOUT_SECONDS, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(READ_TIMEOUT_SECONDS * 1000)}")
        return connection
    except sqlite3.Error as sqlite_error:
        raise DatabaseError(
            f"cannot open database {database_path} read-only: {sqlite_error}",
            operator_hint="the monitor's account must be in the shared state group "
            "(paperwatch-state) to read the hardened deployment",
        ) from sqlite_error


def existing_table_names(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'"))


def schema_version(connection: sqlite3.Connection) -> int:
    return connection.execute("PRAGMA user_version").fetchone()[0]


def load_paper_titles(connection: sqlite3.Connection, paper_ids: Iterable[str | None]) -> dict[str, str]:
    """{paper_id: title}. Titles are untrusted text from providers: the page inserts them as text, never as HTML."""
    wanted_ids = sorted({paper_id for paper_id in paper_ids if paper_id})
    titles: dict[str, str] = {}
    for chunk_start in range(0, len(wanted_ids), 500):
        chunk = wanted_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(f"SELECT paper_id, title FROM papers WHERE paper_id IN ({placeholders})", chunk):  # noqa: S608  # only ? placeholders are interpolated
            titles[row["paper_id"]] = row["title"]
    return titles


def load_job_papers(connection: sqlite3.Connection, job_ids: Iterable[str | None]) -> dict[str, sqlite3.Row]:
    """{job_id: row with kind, paper_id, priority} for the given jobs."""
    wanted_ids = sorted({job_id for job_id in job_ids if job_id})
    rows: dict[str, sqlite3.Row] = {}
    for chunk_start in range(0, len(wanted_ids), 500):
        chunk = wanted_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        for row in connection.execute(
            f"SELECT job_id, kind, paper_id, priority FROM jobs WHERE job_id IN ({placeholders})",  # noqa: S608  # only ? placeholders are interpolated
            chunk,
        ):
            rows[row["job_id"]] = row
    return rows
