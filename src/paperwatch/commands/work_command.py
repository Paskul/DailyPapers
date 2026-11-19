"""`paperwatch work [--once]`: claim due jobs and run them, one bounded job at a time.

Each job kind has a handler that performs the slow external work (a model call) OUTSIDE any write transaction and
returns the database writes to commit; those writes and `job done` then commit together under the lease check.
While the handler runs, a JobLeaseKeeper thread renews the job's short lease, so a crashed worker's job is
reclaimable quickly while a live worker keeps its job however long the model takes.

While no job is due, or while the model server is paused after an outage, the worker sleeps and issues zero
inference calls. A paused server is re-probed with the cheap /health endpoint on an exponential back-off.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from paperwatch.app_context import AppContext
from paperwatch.catalog import load_edition_row, set_paper_status
from paperwatch.clock import utc_now
from paperwatch.db import write_transaction
from paperwatch.errors import (InferenceLockTimeout, ModelServerUnavailable, PaperwatchError, SourceInputError,
                               UnexpectedWorkerError, describe_failure_for_operator)
from paperwatch.jobs import (ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY, ATTEMPT_SETTLEMENT_LEASE_LOST,
                             REVIEW_PAPER_JOB_KIND, ClaimedJob, claim_next_due_job, find_next_job_due_time,
                             finish_job_with_writes, record_job_attempt_failure, release_job_without_charging_attempt)
from paperwatch.lease_keeper import JobLeaseKeeper
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, ensure_model_service_available, read_service_health_state
from paperwatch.readings import ReadingJobInput, refresh_paper_status_from_work
from paperwatch.review_inputs import build_abstract_review_packet, build_reviewer_role_input
from paperwatch.review_store import insert_review_with_topic_assessments
from paperwatch.roles.base import run_role_once
from paperwatch.roles.reviewer import REVIEWER_ROLE

LOGGER = logging.getLogger(__name__)
RETRY_AFTER_LOCK_TIMEOUT_SECONDS = 5.0
MINIMUM_IDLE_SLEEP_SECONDS = 1.0

ResultWriter = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class JobHandlerResult:
    """What a handler hands back: the writes to commit with `job done`, and a short note for the operator."""

    write_results: ResultWriter
    completion_note: str


JobHandler = Callable[[AppContext, ClaimedJob], JobHandlerResult]


class WorkCycleOutcome(StrEnum):
    NO_JOB_DUE = "no_job_due"
    JOB_COMPLETED = "job_completed"
    JOB_RETRY_SCHEDULED = "job_retry_scheduled"
    JOB_FAILED_PERMANENTLY = "job_failed_permanently"
    JOB_RELEASED_UNCHARGED = "job_released_uncharged"  # e.g. inference lock busy; not the job's fault
    MODEL_SERVICE_PAUSED = "model_service_paused"
    LEASE_LOST = "lease_lost"  # another worker reclaimed the job; our result was discarded


@dataclass(frozen=True)
class WorkCycleReport:
    outcome: WorkCycleOutcome
    message: str
    job_id: str | None = None
    paper_id: str | None = None
    failure: PaperwatchError | None = None


# --------------------------------------------------------------------------------------------------------------------
# Job handlers
# --------------------------------------------------------------------------------------------------------------------


def handle_review_paper_job(app: AppContext, claimed_job: ClaimedJob) -> JobHandlerResult:
    reading_job_input = ReadingJobInput.from_payload(claimed_job.input_payload)
    edition_row = load_edition_checked_against_catalog(app, claimed_job, reading_job_input.edition_id)
    review_packet = build_abstract_review_packet(edition_row, reading_job_input.reading_generation)

    if review_packet.review_input_key != claimed_job.input_version:
        # The edition's text changed after this job was queued (a title-only edition received its abstract). The
        # collector queued a job for the new text, so this one finishes without a model call.
        def mark_superseded(connection: sqlite3.Connection) -> None:
            refresh_paper_status_from_work(connection, claimed_job.paper_id)

        return JobHandlerResult(mark_superseded, "input superseded by newer text; no model call made")

    role_result = run_role_once(role_definition=REVIEWER_ROLE,
                                role_input=build_reviewer_role_input(review_packet, app.config.topics),
                                app_config=app.config, model_client=app.model_client, connection=app.connection,
                                job_id=claimed_job.job_id)

    def write_review_and_mark_paper_complete(connection: sqlite3.Connection) -> None:
        insert_review_with_topic_assessments(connection, paper_id=claimed_job.paper_id, job_id=claimed_job.job_id,
                                             review_packet=review_packet, role_result=role_result,
                                             topics=app.config.topics)
        set_paper_status(connection, claimed_job.paper_id, "complete")

    return JobHandlerResult(write_review_and_mark_paper_complete,
                            f"review saved for paper {claimed_job.paper_id} ({review_packet.coverage})")


def load_edition_checked_against_catalog(app: AppContext, claimed_job: ClaimedJob, edition_id: str) -> sqlite3.Row:
    """Re-check a queued id against the source records at dispatch (plan part 3): the edition must still exist and
    belong to the job's paper. A mismatch is never "repaired" by reading something else."""
    try:
        edition_row = load_edition_row(app.connection, edition_id)
    except LookupError:
        raise SourceInputError(f"job {claimed_job.job_id} names edition {edition_id}, which is not in the catalog") \
            from None
    if edition_row["paper_id"] != claimed_job.paper_id:
        raise SourceInputError(f"job {claimed_job.job_id} names edition {edition_id} of paper "
                               f"{edition_row['paper_id']}, not of its own paper {claimed_job.paper_id}")
    return edition_row


JOB_HANDLERS_BY_KIND: dict[str, JobHandler] = {
    REVIEW_PAPER_JOB_KIND: handle_review_paper_job,
}


# --------------------------------------------------------------------------------------------------------------------
# One cycle: guard -> claim -> handle -> commit
# --------------------------------------------------------------------------------------------------------------------


def run_single_work_cycle(app: AppContext, *, probe_paused_model_now: bool) -> WorkCycleReport:
    try:
        ensure_model_service_available(app.connection, app.model_client, app.config.model,
                                       probe_now_even_if_backing_off=probe_paused_model_now)
    except ModelServerUnavailable as outage:
        return WorkCycleReport(WorkCycleOutcome.MODEL_SERVICE_PAUSED, str(outage), failure=outage)

    claimed_job = claim_next_due_job(app.connection, lease_duration_seconds=app.config.worker.lease_seconds)
    if claimed_job is None:
        return WorkCycleReport(WorkCycleOutcome.NO_JOB_DUE, "no job is due")
    if claimed_job.paper_id:
        set_paper_status(app.connection, claimed_job.paper_id, "reviewing")
    LOGGER.info("claimed %s (%s, attempt %d/%d)", claimed_job.job_id, claimed_job.kind,
                claimed_job.attempts_including_this_one, claimed_job.max_attempts)

    try:
        job_handler = JOB_HANDLERS_BY_KIND.get(claimed_job.kind)
        if job_handler is None:
            raise UnexpectedWorkerError(f"no handler registered for job kind '{claimed_job.kind}'")
        with JobLeaseKeeper(app.config.database_path, claimed_job,
                            lease_duration_seconds=app.config.worker.lease_seconds,
                            renewal_interval_seconds=app.config.worker.lease_renewal_interval_seconds):
            handler_result = job_handler(app, claimed_job)
    except PaperwatchError as failure:
        return _settle_failed_job(app, claimed_job, failure)
    except Exception as unexpected_exception:  # a bug: charge the attempt so it cannot loop forever
        LOGGER.exception("unexpected error while handling %s", claimed_job.job_id)
        return _settle_failed_job(app, claimed_job, UnexpectedWorkerError(repr(unexpected_exception)))

    if not finish_job_with_writes(app.connection, claimed_job, handler_result.write_results):
        return WorkCycleReport(WorkCycleOutcome.LEASE_LOST, f"lease on {claimed_job.job_id} was lost; result "
                               "discarded so a newer worker's result is not overwritten",
                               job_id=claimed_job.job_id, paper_id=claimed_job.paper_id)
    return WorkCycleReport(WorkCycleOutcome.JOB_COMPLETED,
                           f"completed {claimed_job.kind} {claimed_job.job_id}: {handler_result.completion_note}",
                           job_id=claimed_job.job_id, paper_id=claimed_job.paper_id)


def _settle_failed_job(app: AppContext, claimed_job: ClaimedJob, failure: PaperwatchError) -> WorkCycleReport:
    job_reference = dict(job_id=claimed_job.job_id, paper_id=claimed_job.paper_id, failure=failure)
    lease_lost_report = WorkCycleReport(
        WorkCycleOutcome.LEASE_LOST, f"{failure}; lease on {claimed_job.job_id} was already lost, so the newer "
        "worker's state is left untouched", **job_reference)
    with write_transaction(app.connection):
        if failure.is_service_outage or isinstance(failure, InferenceLockTimeout):
            if not release_job_without_charging_attempt(app.connection, claimed_job, failure):
                return lease_lost_report
            paper_status, outcome = "queued", (WorkCycleOutcome.MODEL_SERVICE_PAUSED if failure.is_service_outage
                                               else WorkCycleOutcome.JOB_RELEASED_UNCHARGED)
            message = f"{failure}; job {claimed_job.job_id} returned to the queue without using an attempt"
        else:
            settlement = record_job_attempt_failure(app.connection, claimed_job, failure,
                                                    app.config.worker.retry_delay_seconds)
            if settlement == ATTEMPT_SETTLEMENT_LEASE_LOST:
                return lease_lost_report
            if settlement == ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY:
                paper_status, outcome = "failed", WorkCycleOutcome.JOB_FAILED_PERMANENTLY
                message = f"{failure}; job {claimed_job.job_id} failed permanently after " \
                          f"{claimed_job.attempts_including_this_one} attempt(s)"
            else:
                paper_status, outcome = "queued", WorkCycleOutcome.JOB_RETRY_SCHEDULED
                message = f"{failure}; job {claimed_job.job_id} will retry " \
                          f"(attempt {claimed_job.attempts_including_this_one}/{claimed_job.max_attempts} used)"
        if claimed_job.paper_id:
            set_paper_status(app.connection, claimed_job.paper_id, paper_status)
    if claimed_job.paper_id:
        message += f"; paper {claimed_job.paper_id} is preserved"
    return WorkCycleReport(outcome, message, **job_reference)


# --------------------------------------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------------------------------------


EXIT_CODE_BY_OUTCOME = {
    WorkCycleOutcome.NO_JOB_DUE: 0,
    WorkCycleOutcome.JOB_COMPLETED: 0,
    WorkCycleOutcome.LEASE_LOST: 0,
}


def run_work_command(app: AppContext, arguments: argparse.Namespace) -> int:
    if arguments.once:
        # An operator ran this by hand, so probe a paused model server immediately instead of waiting.
        cycle_report = run_single_work_cycle(app, probe_paused_model_now=True)
        _print_cycle_report(cycle_report)
        if cycle_report.outcome in EXIT_CODE_BY_OUTCOME:
            return EXIT_CODE_BY_OUTCOME[cycle_report.outcome]
        return cycle_report.failure.process_exit_code if cycle_report.failure else 1
    return _run_worker_loop(app)


def _run_worker_loop(app: AppContext) -> int:
    stop_requested = threading.Event()

    def request_stop(signal_number: int, _frame: object) -> None:
        LOGGER.info("received signal %d; stopping after the current job", signal_number)
        stop_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    print("paperwatch worker started; Ctrl+C or SIGTERM stops it after the current job", flush=True)

    while not stop_requested.is_set():
        cycle_report = run_single_work_cycle(app, probe_paused_model_now=False)
        if cycle_report.outcome != WorkCycleOutcome.NO_JOB_DUE:
            _print_cycle_report(cycle_report)
        stop_requested.wait(_seconds_until_next_cycle(app, cycle_report))
    print("paperwatch worker stopped", flush=True)
    return 0


def _seconds_until_next_cycle(app: AppContext, cycle_report: WorkCycleReport) -> float:
    idle_poll_seconds = float(app.config.worker.idle_poll_seconds)
    if cycle_report.outcome == WorkCycleOutcome.NO_JOB_DUE:
        next_due_time = find_next_job_due_time(app.connection)
        if next_due_time is None:
            return idle_poll_seconds  # also how quickly newly collected work is noticed
        return min(idle_poll_seconds, max(MINIMUM_IDLE_SLEEP_SECONDS, (next_due_time - utc_now()).total_seconds()))
    if cycle_report.outcome == WorkCycleOutcome.MODEL_SERVICE_PAUSED:
        paused_until = read_service_health_state(app.connection, MODEL_SERVICE_NAME).paused_until
        if paused_until is None:
            return MINIMUM_IDLE_SLEEP_SECONDS
        return min(idle_poll_seconds, max(MINIMUM_IDLE_SLEEP_SECONDS, (paused_until - utc_now()).total_seconds()))
    if cycle_report.outcome == WorkCycleOutcome.JOB_RELEASED_UNCHARGED:
        return RETRY_AFTER_LOCK_TIMEOUT_SECONDS
    return 0.0  # a job just finished or failed: look for the next one immediately


def _print_cycle_report(cycle_report: WorkCycleReport) -> None:
    if cycle_report.failure is not None:
        print(describe_failure_for_operator(cycle_report.failure, message_override=cycle_report.message),
              file=sys.stderr, flush=True)
    else:
        print(f"work: {cycle_report.message}", flush=True)
