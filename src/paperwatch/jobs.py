"""The job queue: unique logical jobs, short-transaction claims with leases, and lease-checked completion.

Job life cycle: queued -> running -> done, with delayed retries (back to queued) and a terminal `failed` state.
  * Claim: one short BEGIN IMMEDIATE transaction picks the next due job and stamps a fresh lease token.
  * Expired leases (a crashed worker) are reclaimable by the same claim query.
  * Finish: the result writes and `status = done` commit in ONE transaction, and only if the lease token still
    matches, so a stale worker can never overwrite a newer result.
  * Leases are short; a worker renews its lease while a long call runs (paperwatch.lease_keeper), so a crashed
    worker's job is reclaimable within one lease period instead of after the slowest possible call.
  * A service outage releases the job without charging an attempt; a paper-specific failure charges one.
Claim order (plan part 6, paperwatch.dispatch_order): the effective priority class first (a new release whose fresh
window has ended counts as class 2), then the two within-class order values (higher first), then queue age. Each
claim is one bounded job, so a higher-priority arrival runs right after the current job.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import JobDispatchOrder, JobPriorityClass
from paperwatch.errors import PaperwatchError
from paperwatch.identifiers import JOB_ID_PREFIX, new_internal_id

REVIEW_PAPER_JOB_KIND = "review_paper"  # classes 0-3: a reading of one paper
COMPARE_REVIEWS_JOB_KIND = "compare_reviews"  # class 4: queued from part 8; the worker gains its handler then

__all__ = ["JobPriorityClass", "JobDispatchOrder"]  # re-exported: callers import the classes from here

# The class a job is claimed under right now: a new release past its fresh window drops to older literature.
EFFECTIVE_PRIORITY_CLASS_SQL = f"""CASE WHEN priority = {int(JobPriorityClass.NEW_RELEASE)}
                   AND new_release_fresh_until IS NOT NULL AND new_release_fresh_until <= :now
              THEN {int(JobPriorityClass.HISTORICAL_SEARCH)} ELSE priority END"""


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    job_key: str
    kind: str
    paper_id: str | None
    input_version: str
    input_payload: dict[str, Any]
    attempts_including_this_one: int
    max_attempts: int
    lease_token: str


class LeaseLostError(Exception):
    """Raised inside a finish transaction when another worker reclaimed the job; the transaction rolls back."""


def build_review_job_key(paper_id: str, review_input_key: str) -> str:
    return f"{REVIEW_PAPER_JOB_KIND}:{paper_id}:{review_input_key}"


def enqueue_job_if_absent(connection: sqlite3.Connection, *, job_key: str, kind: str, paper_id: str | None,
                          input_version: str, input_payload: dict[str, Any], dispatch_order: JobDispatchOrder,
                          max_attempts: int, due_at: str | None = None) -> bool:
    """Insert the job unless a job with the same logical key exists (whatever its status). Returns True if added.

    `due_at` in the future holds a new job back (e.g. until `collect` has acquired the paper's source text); by
    default the job is due at once."""
    now_timestamp = utc_now_timestamp()
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO jobs (job_id, job_key, kind, paper_id, input_version, input_json, priority,
               within_class_order_primary, within_class_order_secondary, new_release_fresh_until,
               status, attempts, max_attempts, due_at, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?)""",
        (new_internal_id(JOB_ID_PREFIX), job_key, kind, paper_id, input_version, json.dumps(input_payload),
         int(dispatch_order.priority_class), dispatch_order.within_class_order_primary,
         dispatch_order.within_class_order_secondary, dispatch_order.new_release_fresh_until,
         max_attempts, due_at or now_timestamp, now_timestamp, now_timestamp))
    return insert_cursor.rowcount == 1


def upgrade_queued_job_dispatch_order_if_higher(connection: sqlite3.Connection, job_key: str,
                                                dispatch_order: JobDispatchOrder) -> bool:
    """A paper that qualifies for several classes runs in the highest one: move a still-QUEUED job up to
    `dispatch_order` if that is claimed earlier than its current order. Never moves a job down. Returns True if
    changed. Call inside a write transaction."""
    job_row = connection.execute(
        """SELECT priority, within_class_order_primary, within_class_order_secondary FROM jobs
           WHERE job_key = ? AND status = 'queued'""", (job_key,)).fetchone()
    if job_row is None:
        return False
    current_order = JobDispatchOrder(JobPriorityClass(job_row["priority"]), job_row["within_class_order_primary"],
                                     job_row["within_class_order_secondary"])
    if not dispatch_order.is_higher_than(current_order):
        return False
    connection.execute(
        """UPDATE jobs SET priority = ?, within_class_order_primary = ?, within_class_order_secondary = ?,
               new_release_fresh_until = ?, updated_at = ? WHERE job_key = ? AND status = 'queued'""",
        (int(dispatch_order.priority_class), dispatch_order.within_class_order_primary,
         dispatch_order.within_class_order_secondary, dispatch_order.new_release_fresh_until, utc_now_timestamp(),
         job_key))
    return True


def claim_next_due_job(connection: sqlite3.Connection, *, lease_duration_seconds: float) -> ClaimedJob | None:
    now_timestamp = utc_now_timestamp()
    lease_token = uuid.uuid4().hex
    lease_expires_at = format_utc_timestamp(utc_now() + timedelta(seconds=lease_duration_seconds))
    with write_transaction(connection):
        job_row = connection.execute(
            f"""SELECT * FROM jobs
               WHERE (status = 'queued' AND due_at <= :now)
                  OR (status = 'running' AND lease_expires_at <= :now)
               ORDER BY {EFFECTIVE_PRIORITY_CLASS_SQL}, within_class_order_primary DESC,
                        within_class_order_secondary DESC, created_at, job_id
               LIMIT 1""", {"now": now_timestamp}).fetchone()
        if job_row is None:
            return None
        connection.execute(
            """UPDATE jobs SET status = 'running', attempts = attempts + 1, lease_token = ?, lease_expires_at = ?,
                   updated_at = ? WHERE job_id = ?""",
            (lease_token, lease_expires_at, now_timestamp, job_row["job_id"]))
    return ClaimedJob(job_id=job_row["job_id"], job_key=job_row["job_key"], kind=job_row["kind"],
                      paper_id=job_row["paper_id"], input_version=job_row["input_version"],
                      input_payload=json.loads(job_row["input_json"]),
                      attempts_including_this_one=job_row["attempts"] + 1, max_attempts=job_row["max_attempts"],
                      lease_token=lease_token)


def finish_job_with_writes(connection: sqlite3.Connection, claimed_job: ClaimedJob,
                           write_results: Callable[[sqlite3.Connection], None]) -> bool:
    """Commit `write_results` and mark the job done atomically, only if our lease still holds. False if lost."""
    try:
        with write_transaction(connection):
            update_cursor = connection.execute(
                """UPDATE jobs SET status = 'done', lease_token = NULL, lease_expires_at = NULL,
                       last_error = NULL, last_error_component = NULL, updated_at = ?
                   WHERE job_id = ? AND lease_token = ? AND status = 'running'""",
                (utc_now_timestamp(), claimed_job.job_id, claimed_job.lease_token))
            if update_cursor.rowcount != 1:
                raise LeaseLostError(claimed_job.job_id)
            write_results(connection)
    except LeaseLostError:
        return False
    return True


def renew_job_lease(connection: sqlite3.Connection, job_id: str, lease_token: str, lease_duration_seconds: float,
                    ) -> bool:
    """Push the lease expiry forward if we still hold the lease. False means another worker reclaimed the job."""
    now = utc_now()
    renew_cursor = connection.execute(
        """UPDATE jobs SET lease_expires_at = ?, updated_at = ?
           WHERE job_id = ? AND lease_token = ? AND status = 'running'""",
        (format_utc_timestamp(now + timedelta(seconds=lease_duration_seconds)), format_utc_timestamp(now),
         job_id, lease_token))
    return renew_cursor.rowcount == 1


def release_job_without_charging_attempt(connection: sqlite3.Connection, claimed_job: ClaimedJob,
                                         failure: PaperwatchError) -> bool:
    """Put the job back exactly as it was (minus the lease) because the failure was not the job's fault.

    Returns False if our lease was already gone (another worker owns the job now); nothing is changed then.
    """
    release_cursor = connection.execute(
        """UPDATE jobs SET status = 'queued', attempts = MAX(attempts - 1, 0), lease_token = NULL,
               lease_expires_at = NULL, last_error_component = ?, last_error = ?, updated_at = ?
           WHERE job_id = ? AND lease_token = ? AND status = 'running'""",
        (failure.failed_component, str(failure), utc_now_timestamp(), claimed_job.job_id, claimed_job.lease_token))
    return release_cursor.rowcount == 1


ATTEMPT_SETTLEMENT_RETRY_SCHEDULED = "retry_scheduled"
ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY = "failed_permanently"
ATTEMPT_SETTLEMENT_LEASE_LOST = "lease_lost"


def record_job_attempt_failure(connection: sqlite3.Connection, claimed_job: ClaimedJob, failure: PaperwatchError,
                               retry_delay_seconds: int) -> str:
    """Charge the failed attempt. Returns one of the ATTEMPT_SETTLEMENT_* values.

    'lease_lost' means another worker reclaimed the job meanwhile; nothing is changed, so a stale worker's failure
    never overwrites a newer worker's state.
    """
    out_of_attempts = (claimed_job.attempts_including_this_one >= claimed_job.max_attempts
                       or failure.ends_job_immediately)  # retrying the same input cannot help
    new_status = "failed" if out_of_attempts else "queued"
    retry_due_at = format_utc_timestamp(
        utc_now() + timedelta(seconds=retry_delay_seconds * claimed_job.attempts_including_this_one))
    settle_cursor = connection.execute(
        """UPDATE jobs SET status = ?, due_at = ?, lease_token = NULL, lease_expires_at = NULL,
               last_error_component = ?, last_error = ?, updated_at = ?
           WHERE job_id = ? AND lease_token = ? AND status = 'running'""",
        (new_status, retry_due_at, failure.failed_component, str(failure), utc_now_timestamp(),
         claimed_job.job_id, claimed_job.lease_token))
    if settle_cursor.rowcount != 1:
        return ATTEMPT_SETTLEMENT_LEASE_LOST
    return ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY if out_of_attempts else ATTEMPT_SETTLEMENT_RETRY_SCHEDULED


def find_next_job_due_time(connection: sqlite3.Connection) -> datetime | None:
    """Earliest moment a queued job becomes due or a running lease expires; None if the queue is empty."""
    next_due_row = connection.execute(
        """SELECT MIN(moment) AS next_moment FROM (
               SELECT MIN(due_at) AS moment FROM jobs WHERE status = 'queued'
               UNION ALL
               SELECT MIN(lease_expires_at) FROM jobs WHERE status = 'running')""").fetchone()
    return parse_utc_timestamp(next_due_row["next_moment"]) if next_due_row["next_moment"] else None


@dataclass(frozen=True)
class QueuedClassSummary:
    """Queued jobs of one effective priority class, so the operator sees when new arrivals delay older work."""

    priority_class: JobPriorityClass
    queued_count: int
    oldest_queued_at: datetime


def summarize_queued_jobs_by_class(connection: sqlite3.Connection, moment: datetime) -> list[QueuedClassSummary]:
    return [QueuedClassSummary(JobPriorityClass(summary_row["effective_class"]), summary_row["queued_count"],
                               parse_utc_timestamp(summary_row["oldest_queued_at"]))
            for summary_row in connection.execute(
                f"""SELECT {EFFECTIVE_PRIORITY_CLASS_SQL} AS effective_class, COUNT(*) AS queued_count,
                           MIN(created_at) AS oldest_queued_at
                    FROM jobs WHERE status = 'queued' GROUP BY effective_class ORDER BY effective_class""",
                {"now": format_utc_timestamp(moment)})]
