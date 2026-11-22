"""`paperwatch work [--once]`: claim due jobs and run them, one bounded job at a time.

Each job kind has a handler that performs the slow external work (model calls) OUTSIDE any write transaction and
returns the database writes to commit; those writes and `job done` then commit together under the lease check.
After a successful commit the handler's file output (the permanent reading or comparison note) is written; if that
write fails, `report` restores the note from the saved review or comparison.

Job kinds:
    review_paper       read one paper (plan part 7); its commit also queues its comparison (plan part 8)
    select_references  the research role chooses which cited works of a reviewed paper to read (plan part 8)
    compare_reviews    the comparison role compares two saved reviews on the same question (plan part 8)
Only a review_paper job changes its paper's status; the other kinds have no paper of their own.
While the handler runs, a JobLeaseKeeper thread renews the job's short lease, so a crashed worker's job is
reclaimable quickly while a live worker keeps its job however long the model takes.

While no job is due, or while the model server is paused after an outage, the worker sleeps and issues zero
inference calls. A paused server is re-probed with the cheap /health endpoint on an exponential back-off.

Stopping (SIGTERM / SIGINT, e.g. the report service at the cutoff): a wait for the model is aborted at once
(paperwatch.worker_stop), the current job goes back to the queue WITHOUT using an attempt, and the worker exits.
Saved stage results of a long paper are kept, so the next run continues with the first missing section.
"""

from __future__ import annotations

import argparse
import functools
import logging
import signal
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from paperwatch.activity_log import COMPONENT_WORKER, recorded_activity
from paperwatch.app_context import AppContext
from paperwatch.catalog import load_edition_row, set_paper_status
from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.comparisons.comparison_job import prepare_comparison, write_comparison
from paperwatch.comparisons.comparison_queue import ComparisonJobInput, queue_comparison_for_review
from paperwatch.db import write_transaction
from paperwatch.errors import (
    InferenceLockTimeout,
    ModelServerUnavailable,
    PaperwatchError,
    SourceInputError,
    UnexpectedWorkerError,
    describe_failure_for_operator,
)
from paperwatch.jobs import (
    ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY,
    ATTEMPT_SETTLEMENT_LEASE_LOST,
    COMPARE_REVIEWS_JOB_KIND,
    REVIEW_PAPER_JOB_KIND,
    ClaimedJob,
    claim_next_due_job,
    find_next_job_due_time,
    finish_job_with_writes,
    record_job_attempt_failure,
    release_job_without_charging_attempt,
)
from paperwatch.lease_keeper import JobLeaseKeeper
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, ensure_model_service_available, read_service_health_state
from paperwatch.reading.reading_pipeline import read_paper
from paperwatch.reading.review_packet import ExtractedTextUnavailable, build_review_packet
from paperwatch.readings import (
    ReadingEnqueueOutcome,
    ReadingJobInput,
    dispatch_order_of_job_row,
    enqueue_reading_if_needed,
    refresh_paper_status_from_work,
    select_reading_input,
)
from paperwatch.references.reference_selection import (
    SELECT_REFERENCES_JOB_KIND,
    ReferenceSelectionJobInput,
    prepare_reference_selection,
    write_reference_selection_outcome,
)
from paperwatch.rendering.comparison_notes import write_comparison_note_if_missing
from paperwatch.rendering.review_notes import load_selected_review, write_paper_note_if_missing
from paperwatch.research_profile import load_active_seed_library_paper_ids
from paperwatch.review_store import insert_completed_reading
from paperwatch.roles.base import run_role_with_one_repair
from paperwatch.roles.comparison import COMPARISON_ROLE
from paperwatch.roles.research import RESEARCH_ROLE
from paperwatch.service_log_redaction import redact_service_log_text
from paperwatch.worker_stop import (
    WorkerStopRequested,
    request_worker_stop,
    reset_worker_stop_request,
    stop_was_requested,
    wait_for_stop_request,
)

LOGGER = logging.getLogger(__name__)
RETRY_AFTER_LOCK_TIMEOUT_SECONDS = 5.0
MINIMUM_IDLE_SLEEP_SECONDS = 1.0

ResultWriter = Callable[[sqlite3.Connection], None]


@dataclass(frozen=True)
class JobHandlerResult:
    """What a handler hands back: the writes to commit with `job done`, a short note for the operator, and optional
    file output to produce once the commit succeeded (never before: a lost lease must leave no file behind)."""

    write_results: ResultWriter
    completion_note: str | Callable[[], str]  # a callable is evaluated after the commit (e.g. counts it decided)
    after_commit: Callable[[], None] | None = None

    def describe_completion(self) -> str:
        return self.completion_note() if callable(self.completion_note) else self.completion_note


JobHandler = Callable[[AppContext, ClaimedJob], JobHandlerResult]


class WorkCycleOutcome(StrEnum):
    NO_JOB_DUE = "no_job_due"
    JOB_COMPLETED = "job_completed"
    JOB_RETRY_SCHEDULED = "job_retry_scheduled"
    JOB_FAILED_PERMANENTLY = "job_failed_permanently"
    JOB_RELEASED_UNCHARGED = "job_released_uncharged"  # e.g. inference lock busy; not the job's fault
    MODEL_SERVICE_PAUSED = "model_service_paused"
    LEASE_LOST = "lease_lost"  # another worker reclaimed the job; our result was discarded
    STOPPED = "stopped"  # a stop request interrupted the job; it went back to the queue without using an attempt


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
        return _supersede_with_current_input(
            app, claimed_job, reading_job_input, "input superseded by newer text; no model call made"
        )
    try:
        review_packet = build_review_packet(app.connection, app.config, claimed_job.paper_id, current_input)
    except ExtractedTextUnavailable as unavailable_text:
        return _fall_back_from_unusable_text(
            app, claimed_job, reading_job_input, current_input.edition_id, unavailable_text
        )

    completed_reading = read_paper(app, review_packet, job_id=claimed_job.job_id)
    committed_review_ids: list[str] = []

    def write_review_and_mark_paper_complete(connection: sqlite3.Connection) -> None:
        committed_review_ids.append(
            insert_completed_reading(
                connection,
                paper_id=claimed_job.paper_id,
                job_id=claimed_job.job_id,
                completed_reading=completed_reading,
            )
        )
        set_paper_status(connection, claimed_job.paper_id, "complete")
        # Plan part 8: compare the new review with its closest comparable neighbour (within the daily limit).
        queue_comparison_for_review(connection, app.config, committed_review_ids[0], utc_now())

    def write_permanent_note() -> None:
        write_paper_note_if_missing(
            app.config.paper_notes_dir, load_selected_review(app.connection, committed_review_ids[0])
        )

    best_score = max(
        (assessment.scored.reading_priority_score for assessment in completed_reading.topic_assessments), default=None
    )
    return JobHandlerResult(
        write_review_and_mark_paper_complete,
        f"review saved for paper {claimed_job.paper_id} ({review_packet.coverage}, "
        f"{len(review_packet.reading_plan.sections_to_read)} reading section(s), best reading priority "
        f"{'not rated' if best_score is None else f'{best_score}/100'})",
        after_commit=write_permanent_note,
    )


def _supersede_with_current_input(
    app: AppContext, claimed_job: ClaimedJob, reading_job_input: ReadingJobInput, completion_note: str
) -> JobHandlerResult:
    job_row = app.connection.execute("SELECT * FROM jobs WHERE job_id = ?", (claimed_job.job_id,)).fetchone()

    def queue_current_input(connection: sqlite3.Connection) -> None:
        enqueue_outcome, _reading_input = enqueue_reading_if_needed(
            connection,
            claimed_job.paper_id,
            dispatch_order=dispatch_order_of_job_row(job_row),
            max_attempts=claimed_job.max_attempts,
            reading_generation=reading_job_input.reading_generation,
            operator_reason=reading_job_input.operator_reason,
        )
        if enqueue_outcome != ReadingEnqueueOutcome.QUEUED:
            LOGGER.info(
                "superseded job %s: current input of %s is %s",
                claimed_job.job_id,
                claimed_job.paper_id,
                enqueue_outcome,
            )
        refresh_paper_status_from_work(connection, claimed_job.paper_id)

    return JobHandlerResult(queue_current_input, completion_note)


def _fall_back_from_unusable_text(
    app: AppContext,
    claimed_job: ClaimedJob,
    reading_job_input: ReadingJobInput,
    edition_id: str,
    problem: ExtractedTextUnavailable,
) -> JobHandlerResult:
    """The cached full text is gone or altered: record that, and queue the next best input (usually the abstract)."""
    supersede_result = _supersede_with_current_input(
        app, claimed_job, reading_job_input, f"{problem}; the next best input was queued instead"
    )

    def mark_text_failed_then_requeue(connection: sqlite3.Connection) -> None:
        connection.execute(
            """UPDATE edition_source_texts SET extraction_status = 'failed', last_error = ?, updated_at = ?
               WHERE edition_id = ?""",
            (str(problem)[:300], utc_now_timestamp(), edition_id),
        )
        supersede_result.write_results(connection)

    return JobHandlerResult(mark_text_failed_then_requeue, supersede_result.completion_note)


def load_edition_checked_against_catalog(app: AppContext, claimed_job: ClaimedJob, edition_id: str) -> sqlite3.Row:
    """Re-check a queued id against the source records at dispatch (plan part 3): the edition must still exist and
    belong to the job's paper. A mismatch is never "repaired" by reading something else."""
    try:
        edition_row = load_edition_row(app.connection, edition_id)
    except LookupError:
        raise SourceInputError(
            f"job {claimed_job.job_id} names edition {edition_id}, which is not in the catalog"
        ) from None
    if edition_row["paper_id"] != claimed_job.paper_id:
        raise SourceInputError(
            f"job {claimed_job.job_id} names edition {edition_id} of paper "
            f"{edition_row['paper_id']}, not of its own paper {claimed_job.paper_id}"
        )
    return edition_row


def handle_select_references_job(app: AppContext, claimed_job: ClaimedJob) -> JobHandlerResult:
    """Plan part 8: the research role chooses up to three cited works the parent's argument depends on (and maybe one
    gap search); the host admits the choices as class-3 readings within the daily limit."""
    job_input = ReferenceSelectionJobInput.from_payload(claimed_job.input_payload)
    active_seed_paper_ids = load_active_seed_library_paper_ids(app.connection, app.config)
    try:
        prepared = prepare_reference_selection(
            app.connection, app.config, job_input, active_seed_paper_ids=active_seed_paper_ids, moment=utc_now()
        )
    except LookupError as missing_review:
        raise SourceInputError(f"job {claimed_job.job_id}: {missing_review}") from None
    role_result = None
    if prepared.candidates:
        role_result = run_role_with_one_repair(
            role_definition=RESEARCH_ROLE,
            role_input=prepared.role_input,
            app_config=app.config,
            model_client=app.model_client,
            connection=app.connection,
            job_id=claimed_job.job_id,
        )
    write_summaries = []

    def write_selection(connection: sqlite3.Connection) -> None:
        write_summaries.append(
            write_reference_selection_outcome(
                connection,
                app.config,
                prepared,
                role_result,
                job_id=claimed_job.job_id,
                active_seed_paper_ids=active_seed_paper_ids,
                moment=utc_now(),
            )
        )

    def describe_after_commit() -> str:  # the outcome is known only once the commit admitted the choices
        return (
            f"references of review {job_input.parent_review_id}: "
            + (write_summaries[0].describe() if write_summaries else "not written")
            + ("" if role_result else " (no unread candidate; no model call made)")
        )

    return JobHandlerResult(write_selection, describe_after_commit)


def handle_compare_reviews_job(app: AppContext, claimed_job: ClaimedJob) -> JobHandlerResult:
    """Plan part 8: compare two saved reviews on the same question; store the comparison once, then write its note."""
    prepared = prepare_comparison(
        app.connection, app.config, ComparisonJobInput.from_payload(claimed_job.input_payload)
    )
    if prepared.role_input is None:
        return JobHandlerResult(
            lambda connection: None,
            f"comparison {prepared.job_input.comparison_key} not made: "
            f"{prepared.reason_without_model_call}; no model call made",
        )
    role_result = run_role_with_one_repair(
        role_definition=COMPARISON_ROLE,
        role_input=prepared.role_input,
        app_config=app.config,
        model_client=app.model_client,
        connection=app.connection,
        job_id=claimed_job.job_id,
    )
    written: list[tuple[str, bool]] = []

    def store_comparison(connection: sqlite3.Connection) -> None:
        written.append(write_comparison(connection, prepared, role_result))

    def write_comparison_note() -> None:
        write_comparison_note_if_missing(app.connection, app.config.comparison_notes_dir, written[0][0])

    return JobHandlerResult(
        store_comparison,
        f"comparison of reviews {prepared.job_input.review_id_a} and "
        f"{prepared.job_input.review_id_b} "
        f"({role_result.validated_output['conditions_comparability']})",
        after_commit=write_comparison_note,
    )


JOB_HANDLERS_BY_KIND: dict[str, JobHandler] = {
    REVIEW_PAPER_JOB_KIND: handle_review_paper_job,
    SELECT_REFERENCES_JOB_KIND: handle_select_references_job,
    COMPARE_REVIEWS_JOB_KIND: handle_compare_reviews_job,
}


# --------------------------------------------------------------------------------------------------------------------
# One cycle: guard -> claim -> handle -> commit
# --------------------------------------------------------------------------------------------------------------------


def run_single_work_cycle(app: AppContext, *, probe_paused_model_now: bool) -> WorkCycleReport:
    try:
        ensure_model_service_available(
            app.connection, app.model_client, app.config.model, probe_now_even_if_backing_off=probe_paused_model_now
        )
    except ModelServerUnavailable as outage:
        return WorkCycleReport(WorkCycleOutcome.MODEL_SERVICE_PAUSED, str(outage), failure=outage)

    if stop_was_requested():  # a stop arrived during the health check: claim nothing new
        return WorkCycleReport(WorkCycleOutcome.STOPPED, "stop requested; no job claimed")
    claimed_job = claim_next_due_job(app.connection, lease_duration_seconds=app.config.worker.lease_seconds)
    if claimed_job is None:
        return WorkCycleReport(WorkCycleOutcome.NO_JOB_DUE, "no job is due")
    if claimed_job.paper_id and claimed_job.kind == REVIEW_PAPER_JOB_KIND:
        set_paper_status(app.connection, claimed_job.paper_id, "reviewing")
    LOGGER.info(
        "claimed %s (%s, attempt %d/%d)",
        claimed_job.job_id,
        claimed_job.kind,
        claimed_job.attempts_including_this_one,
        claimed_job.max_attempts,
    )
    job_detail = {
        "job_id": claimed_job.job_id,
        "kind": claimed_job.kind,
        "paper_id": claimed_job.paper_id,
        "attempt": claimed_job.attempts_including_this_one,
        "max_attempts": claimed_job.max_attempts,
    }
    with recorded_activity(app.connection, "job", job_detail, component=COMPONENT_WORKER) as job_activity:
        cycle_report = _run_claimed_job(app, claimed_job)
        job_activity.end_as(
            ACTIVITY_OUTCOME_BY_CYCLE_OUTCOME.get(cycle_report.outcome, cycle_report.outcome.value),
            note=None if cycle_report.outcome == WorkCycleOutcome.JOB_COMPLETED else cycle_report.message,
        )
    return cycle_report


# How a job's activity-log row ends (the monitor shows it next to the job).
ACTIVITY_OUTCOME_BY_CYCLE_OUTCOME = {
    WorkCycleOutcome.JOB_COMPLETED: "ok",
    WorkCycleOutcome.JOB_RETRY_SCHEDULED: "retry",
    WorkCycleOutcome.JOB_FAILED_PERMANENTLY: "failed",
    WorkCycleOutcome.JOB_RELEASED_UNCHARGED: "released",
    WorkCycleOutcome.MODEL_SERVICE_PAUSED: "paused",
    WorkCycleOutcome.LEASE_LOST: "lease_lost",
    WorkCycleOutcome.STOPPED: "stopped",
}


def _run_claimed_job(app: AppContext, claimed_job: ClaimedJob) -> WorkCycleReport:
    """Run the handler under a renewed lease, then commit its writes, or settle the failure."""
    try:
        job_handler = JOB_HANDLERS_BY_KIND.get(claimed_job.kind)
        if job_handler is None:
            raise UnexpectedWorkerError(f"no handler registered for job kind '{claimed_job.kind}'")
        with JobLeaseKeeper(
            app.config.database_path,
            claimed_job,
            lease_duration_seconds=app.config.worker.lease_seconds,
            renewal_interval_seconds=app.config.worker.lease_renewal_interval_seconds,
        ):
            handler_result = job_handler(app, claimed_job)
    except WorkerStopRequested as stop_request:
        return _release_job_after_stop_request(app, claimed_job, stop_request)
    except PaperwatchError as failure:
        return _settle_failed_job(app, claimed_job, failure)
    except Exception as unexpected_exception:  # a bug: charge the attempt so it cannot loop forever
        LOGGER.exception("unexpected error while handling %s", claimed_job.job_id)
        return _settle_failed_job(app, claimed_job, UnexpectedWorkerError(repr(unexpected_exception)))

    if not finish_job_with_writes(app.connection, claimed_job, handler_result.write_results):
        return WorkCycleReport(
            WorkCycleOutcome.LEASE_LOST,
            f"lease on {claimed_job.job_id} was lost; result discarded so a newer worker's result is not overwritten",
            job_id=claimed_job.job_id,
            paper_id=claimed_job.paper_id,
        )
    completion_message = f"completed {claimed_job.kind} {claimed_job.job_id}: {handler_result.describe_completion()}"
    if handler_result.after_commit is not None:
        try:
            handler_result.after_commit()
        except PaperwatchError as output_failure:  # the result is committed; `report` restores the note later
            completion_message += f" (note not written yet: {output_failure}; `report` will restore it)"
    return WorkCycleReport(
        WorkCycleOutcome.JOB_COMPLETED, completion_message, job_id=claimed_job.job_id, paper_id=claimed_job.paper_id
    )


class JobInterruptedByStopRequest(PaperwatchError):
    """Recorded as the job's last error when a stop request sent it back to the queue (no attempt is used)."""

    failed_component = "worker"


def _release_job_after_stop_request(
    app: AppContext, claimed_job: ClaimedJob, stop_request: WorkerStopRequested
) -> WorkCycleReport:
    interruption = JobInterruptedByStopRequest(str(stop_request))
    with write_transaction(app.connection):
        released = release_job_without_charging_attempt(app.connection, claimed_job, interruption)
        if released and claimed_job.paper_id and claimed_job.kind == REVIEW_PAPER_JOB_KIND:
            set_paper_status(app.connection, claimed_job.paper_id, "queued")
    message = (
        f"stopped during {claimed_job.kind} {claimed_job.job_id}; "
        + (
            "it is back in the queue without using an attempt"
            if released
            else "its lease was already lost, so the newer worker's state is left untouched"
        )
        + "; saved reading stages are kept"
    )
    return WorkCycleReport(WorkCycleOutcome.STOPPED, message, job_id=claimed_job.job_id, paper_id=claimed_job.paper_id)


def _settle_failed_job(app: AppContext, claimed_job: ClaimedJob, failure: PaperwatchError) -> WorkCycleReport:
    report_about_this_job = functools.partial(
        WorkCycleReport, job_id=claimed_job.job_id, paper_id=claimed_job.paper_id, failure=failure
    )
    lease_lost_report = report_about_this_job(
        WorkCycleOutcome.LEASE_LOST,
        f"{failure}; lease on {claimed_job.job_id} was already lost, so the newer worker's state is left untouched",
    )
    with write_transaction(app.connection):
        if failure.is_service_outage or isinstance(failure, InferenceLockTimeout):
            if not release_job_without_charging_attempt(app.connection, claimed_job, failure):
                return lease_lost_report
            paper_status, outcome = (
                "queued",
                (
                    WorkCycleOutcome.MODEL_SERVICE_PAUSED
                    if failure.is_service_outage
                    else WorkCycleOutcome.JOB_RELEASED_UNCHARGED
                ),
            )
            message = f"{failure}; job {claimed_job.job_id} returned to the queue without using an attempt"
        else:
            settlement = record_job_attempt_failure(
                app.connection, claimed_job, failure, app.config.worker.retry_delay_seconds
            )
            if settlement == ATTEMPT_SETTLEMENT_LEASE_LOST:
                return lease_lost_report
            if settlement == ATTEMPT_SETTLEMENT_FAILED_PERMANENTLY:
                paper_status, outcome = "failed", WorkCycleOutcome.JOB_FAILED_PERMANENTLY
                message = (
                    f"{failure}; job {claimed_job.job_id} failed permanently after "
                    f"{claimed_job.attempts_including_this_one} attempt(s)"
                )
            else:
                paper_status, outcome = "queued", WorkCycleOutcome.JOB_RETRY_SCHEDULED
                message = (
                    f"{failure}; job {claimed_job.job_id} will retry "
                    f"(attempt {claimed_job.attempts_including_this_one}/{claimed_job.max_attempts} used)"
                )
        if claimed_job.paper_id and claimed_job.kind == REVIEW_PAPER_JOB_KIND:
            set_paper_status(app.connection, claimed_job.paper_id, paper_status)
    if claimed_job.paper_id:
        message += f"; paper {claimed_job.paper_id} is preserved"
    return report_about_this_job(outcome, message)


# --------------------------------------------------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------------------------------------------------


EXIT_CODE_BY_OUTCOME = {
    WorkCycleOutcome.NO_JOB_DUE: 0,
    WorkCycleOutcome.JOB_COMPLETED: 0,
    WorkCycleOutcome.LEASE_LOST: 0,
    WorkCycleOutcome.STOPPED: 0,
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
    reset_worker_stop_request()

    def request_stop(signal_number: int, _frame: object) -> None:
        LOGGER.info("received signal %d; stopping now (a wait for the model is aborted)", signal_number)
        request_worker_stop()  # raises WorkerStopRequested if the main thread is waiting for the model

    previous_handlers = {
        signal_number: signal.signal(signal_number, request_stop) for signal_number in (signal.SIGTERM, signal.SIGINT)
    }
    print("paperwatch worker started; Ctrl+C or SIGTERM stops it (the current job goes back to the queue)", flush=True)
    try:
        with recorded_activity(app.connection, "run", {"mode": "loop"}, component=COMPONENT_WORKER) as run_activity:
            completed_job_count = 0
            while not stop_was_requested():
                cycle_report = run_single_work_cycle(app, probe_paused_model_now=False)
                if cycle_report.outcome != WorkCycleOutcome.NO_JOB_DUE:
                    _print_cycle_report(cycle_report)
                completed_job_count += cycle_report.outcome == WorkCycleOutcome.JOB_COMPLETED
                if cycle_report.outcome == WorkCycleOutcome.STOPPED:
                    break
                wait_for_stop_request(_seconds_until_next_cycle(app, cycle_report))
            run_activity.add_detail(completed_jobs=completed_job_count)
    finally:
        for signal_number, previous_handler in previous_handlers.items():
            signal.signal(signal_number, previous_handler)
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
        print(
            describe_failure_for_operator(cycle_report.failure, message_override=cycle_report.message),
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"work: {redact_service_log_text(cycle_report.message)}", flush=True)
