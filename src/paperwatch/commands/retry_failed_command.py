"""`paperwatch retry-failed [--component C] [--job JOB_ID] [--dry-run]`: give failed jobs a fresh set of attempts.

A failed job ends for good, with its reason kept. When the cause is fixed on the host side (for example the
"fix, then drop" evidence checks of 2026-11-21, roles/answer_salvage.py), this operator command puts such jobs back in
the queue. The job is the SAME logical job: it keeps its input, its priority, and its saved reading stages, so a long
paper continues with its first missing section instead of starting over. This differs from `reread`, which starts a
new reading generation and reads the paper from the beginning.

By default only jobs whose last failure came from the model's answers ("model-output") are requeued. `--component`
picks other components (repeatable), and `--job` picks individual jobs whatever their component. `--dry-run` lists
the jobs without changing them.
"""

from __future__ import annotations

import argparse
import sqlite3

from paperwatch.app_context import AppContext
from paperwatch.catalog import set_paper_status
from paperwatch.clock import utc_now_timestamp
from paperwatch.commands.command_arguments import DEFAULT_RETRY_COMPONENT
from paperwatch.db import write_transaction
from paperwatch.errors import ConfigurationError
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND

REQUEUE_NOTE_PREFIX = "requeued by the operator after: "
MAXIMUM_STORED_ERROR_CHARACTERS = 2000


def run_retry_failed_command(app: AppContext, arguments: argparse.Namespace) -> int:
    failed_job_rows = select_failed_jobs(
        app.connection, components=tuple(arguments.components or ()), job_ids=tuple(arguments.job_ids or ())
    )
    if not failed_job_rows:
        print("no failed job matches; nothing to requeue")
        return 0
    for job_row in failed_job_rows:
        print(
            f"{'would requeue' if arguments.dry_run else 'requeue'} {job_row['job_id']} ({job_row['kind']}"
            f"{', paper ' + job_row['paper_id'] if job_row['paper_id'] else ''}; failed in "
            f"[{job_row['last_error_component']}] after {job_row['attempts']} attempt(s))"
        )
    if arguments.dry_run:
        print(f"{len(failed_job_rows)} job(s) would be requeued (dry run: nothing changed)")
        return 0
    with write_transaction(app.connection):
        requeued_count = sum(requeue_failed_job(app.connection, job_row) for job_row in failed_job_rows)
    print(
        f"{requeued_count} job(s) back in the queue with a fresh set of attempts; saved reading stages are reused. "
        "The worker picks them up in priority order."
    )
    return 0


def select_failed_jobs(
    connection: sqlite3.Connection, *, components: tuple[str, ...], job_ids: tuple[str, ...]
) -> list[sqlite3.Row]:
    if job_ids:
        placeholders = ", ".join("?" for _job_id in job_ids)
        job_rows = connection.execute(
            f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) ORDER BY created_at",  # noqa: S608  # only ? placeholders are interpolated
            job_ids,
        ).fetchall()
        not_failed = sorted(set(job_ids) - {row["job_id"] for row in job_rows if row["status"] == "failed"})
        if not_failed:
            raise ConfigurationError(
                f"not a failed job: {', '.join(not_failed)}", operator_hint="`paperwatch status` lists failed jobs"
            )
        return job_rows
    selected_components = components or (DEFAULT_RETRY_COMPONENT,)
    placeholders = ", ".join("?" for _component in selected_components)
    return connection.execute(
        f"""SELECT * FROM jobs WHERE status = 'failed' AND last_error_component IN ({placeholders})
            ORDER BY created_at, job_id""",  # noqa: S608  # only ? placeholders are interpolated
        selected_components,
    ).fetchall()


def requeue_failed_job(connection: sqlite3.Connection, job_row: sqlite3.Row) -> bool:
    """Call inside a write transaction. Only a job that is still `failed` changes."""
    previous_error = f"[{job_row['last_error_component']}] {job_row['last_error'] or ''}"
    requeue_cursor = connection.execute(
        """UPDATE jobs SET status = 'queued', attempts = 0, due_at = ?, lease_token = NULL, lease_expires_at = NULL,
               last_error = ?, updated_at = ? WHERE job_id = ? AND status = 'failed'""",
        (
            utc_now_timestamp(),
            (REQUEUE_NOTE_PREFIX + previous_error)[:MAXIMUM_STORED_ERROR_CHARACTERS],
            utc_now_timestamp(),
            job_row["job_id"],
        ),
    )
    if requeue_cursor.rowcount != 1:
        return False
    if job_row["paper_id"] and job_row["kind"] == REVIEW_PAPER_JOB_KIND:
        set_paper_status(connection, job_row["paper_id"], "queued")
    return True
