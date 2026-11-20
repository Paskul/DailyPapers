"""`paperwatch work [--once]`: claim due jobs and run them, one bounded job at a time.

Each job kind has a handler that performs the slow external work (model calls) OUTSIDE any write transaction and
returns the database writes to commit; those writes and `job done` then commit together under the lease check.
After a successful commit the handler's file output (the permanent reading note) is written; if that write fails,
`report` restores the note from the saved review.
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
from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.errors import (InferenceLockTimeout, ModelServerUnavailable, PaperwatchError, SourceInputError,
                               UnexpectedWorkerError, describe_failure_for_operator)
from paperwatch.jobs import (ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY, ATTEMPT_SETTLEMENT_LEASE_LOST,
                             REVIEW_PAPER_JOB_KIND, ClaimedJob, claim_next_due_job, find_next_job_due_time,
                             finish_job_with_writes, record_job_attempt_failure, release_job_without_charging_attempt)
from paperwatch.lease_keeper import JobLeaseKeeper
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, ensure_model_service_available, read_service_health_state
from paperwatch.reading.reading_pipeline import read_paper
from paperwatch.reading.review_packet import ExtractedTextUnavailable, build_review_packet
from paperwatch.readings import (ReadingEnqueueOutcome, ReadingJobInput, dispatch_order_of_job_row,
                                 enqueue_reading_if_needed, refresh_paper_status_from_work, select_reading_input)
from paperwatch.rendering.review_notes import load_selected_review, write_paper_note_if_missing
from paperwatch.review_store import insert_completed_reading

LOGGER = logging.getLogger(__name__)
RETRY_AFTER_LOCK_TIMEOUT_SECONDS = 5.0
MINIMUM_IDLE_SLEEP_SECONDS = 1.0

ResultWriter = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class JobHandlerResult:
    """What a handler hands back: the writes to commit with `job done`, a short note for the operator, and optional
    file output to produce once the commit succeeded (never before: a lost lease must leave no file behind)."""

    write_results: ResultWriter
    completion_note: str
    after_commit: Callable[[], None] | None = None


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
    """Read one paper (plan part 7): rebuild its packet from the saved edition, run the evidence-backed reading,
    and commit the review, its cited passages, and its topic assessments together with `job done`."""
    reading_job_input = ReadingJobInput.from_payload(claimed_job.input_payload)
    load_edition_checked_against_catalog(app, claimed_job, reading_job_input.edition_id)
    current_input = select_reading_input(app.connection, claimed_job.paper_id, reading_job_input.reading_generation)
    if current_input is None or current_input.review_input_key != claimed_job.input_version:
        # Newer text arrived after this job was queued (full text was extracted, or an abstract appeared). Queue
        # the reading of the current input in this job's place; this job ends without a model call.
        return _supersede_with_current_input(app, claimed_job, reading_job_input,
                                             "input superseded by newer text; no model call made")
    try:
        review_packet = build_review_packet(app.connection, app.config, claimed_job.paper_id, current_input)
    except ExtractedTextUnavailable as unavailable_text:
        return _fall_back_from_unusable_text(app, claimed_job, reading_job_input, current_input.edition_id,
                                             unavailable_text)

    completed_reading = read_paper(app, review_packet, job_id=claimed_job.job_id)
    committed_review_ids: list[str] = []

    def write_review_and_mark_paper_complete(connection: sqlite3.Connection) -> None:
        committed_review_ids.append(insert_completed_reading(connection, paper_id=claimed_job.paper_id,
                                                             job_id=claimed_job.job_id,
                                                             completed_reading=completed_reading))
        set_paper_status(connection, claimed_job.paper_id, "complete")

    def write_permanent_note() -> None:
        write_paper_note_if_missing(app.config.paper_notes_dir,
                                    load_selected_review(app.connection, committed_review_ids[0]))

    best_score = max((assessment.scored.reading_priority_score for assessment in completed_reading.topic_assessments),
                     default=None)
    return JobHandlerResult(
        write_review_and_mark_paper_complete,
        f"review saved for paper {claimed_job.paper_id} ({review_packet.coverage}, "
        f"{len(review_packet.reading_plan.sections_to_read)} reading section(s), best reading priority "
        f"{'not rated' if best_score is None else f'{best_score}/100'})",
        after_commit=write_permanent_note)


def _supersede_with_current_input(app: AppContext, claimed_job: ClaimedJob, reading_job_input: ReadingJobInput,
                                  completion_note: str) -> JobHandlerResult:
    job_row = app.connection.execute("SELECT * FROM jobs WHERE job_id = ?", (claimed_job.job_id,)).fetchone()

    def queue_current_input(connection: sqlite3.Connection) -> None:
        enqueue_outcome, _reading_input = enqueue_reading_if_needed(
            connection, claimed_job.paper_id, dispatch_order=dispatch_order_of_job_row(job_row),
            max_attempts=claimed_job.max_attempts, reading_generation=reading_job_input.reading_generation,
            operator_reason=reading_job_input.operator_reason)
        if enqueue_outcome != ReadingEnqueueOutcome.QUEUED:
            LOGGER.info("superseded job %s: current input of %s is %s", claimed_job.job_id, claimed_job.paper_id,
                        enqueue_outcome)
        refresh_paper_status_from_work(connection, claimed_job.paper_id)

    return JobHandlerResult(queue_current_input, completion_note)


def _fall_back_from_unusable_text(app: AppContext, claimed_job: ClaimedJob, reading_job_input: ReadingJobInput,
                                  edition_id: str, problem: ExtractedTextUnavailable) -> JobHandlerResult:
    """The cached full text is gone or altered: record that, and queue the next best input (usually the abstract)."""
    supersede_result = _supersede_with_current_input(
        app, claimed_job, reading_job_input, f"{problem}; the next best input was queued instead")

    def mark_text_failed_then_requeue(connection: sqlite3.Connection) -> None:
        connection.execute(
            """UPDATE edition_source_texts SET extraction_status = 'failed', last_error = ?, updated_at = ?
               WHERE edition_id = ?""", (str(problem)[:300], utc_now_timestamp(), edition_id))
        supersede_result.write_results(connection)

    return JobHandlerResult(mark_text_failed_then_requeue, supersede_result.completion_note)


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
    completion_message = f"completed {claimed_job.kind} {claimed_job.job_id}: {handler_result.completion_note}"
    if handler_result.after_commit is not None:
        try:
            handler_result.after_commit()
        except PaperwatchError as output_failure:  # the result is committed; `report` restores the note later
            completion_message += f" (note not written yet: {output_failure}; `report` will restore it)"
    return WorkCycleReport(WorkCycleOutcome.JOB_COMPLETED, completion_message,
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
