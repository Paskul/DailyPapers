"""Decide when a paper needs a (new) reading, and queue it as one logical job.

A reading is identified by (canonical paper, normalized source-text hash, reading generation); see review_inputs.py.
The job key embeds that identity, so the queue's UNIQUE(job_key) is the final guard against a duplicate reading even
when two collectors race. Consequences:
  * a changed title, another discovery route, a topic or scoring change, a restart, or an empty queue never creates
    another reading, because none of them changes the text hash or the generation;
  * a model or prompt upgrade alone never re-enqueues the archive, because neither is part of the key;
  * a new edition with identical text is not read again; a verified text change or newly available full text gets a
    new reading, if the paper was promoted by screening (paperwatch.screening.review_promotion);
  * an explicit operator reread gets a fresh generation id: a new logical job and, later, a separate review.
A job that failed permanently is NOT re-queued automatically for the same input; the operator can request a reread.

Which input a reading uses: the extracted full text of an edition when the sandboxed extractor produced
usable text, else the abstract of the preferred edition. When better text arrives while a reading is still queued
(the PDF was extracted after promotion), the new input is queued and the older queued job is retired, so the paper
is read once, on the better text.

Holding new readings for their source text: `collect` promotes papers and only then acquires their PDFs. A worker
running at the same moment would otherwise read the abstract seconds before the full text lands. So readings queued
by `collect` are held (due later) until its source-text stage releases them, at most
`collection.source_text_hold_minutes` if that stage never finishes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from paperwatch.clock import format_utc_timestamp, utc_now_timestamp
from paperwatch.dispatch_order import JobDispatchOrder, JobPriorityClass
from paperwatch.identifiers import READING_GENERATION_ID_PREFIX, new_internal_id
from paperwatch.jobs import (
    REVIEW_PAPER_JOB_KIND,
    build_review_job_key,
    enqueue_job_if_absent,
    upgrade_queued_job_dispatch_order_if_higher,
)
from paperwatch.review_inputs import (
    INITIAL_READING_GENERATION,
    TEXT_KIND_ABSTRACT,
    TEXT_KIND_EXTRACTED_TEXT,
    ReadingInputIdentity,
    build_abstract_reading_identity,
)

# Extracted text shorter than this (e.g. a scanned PDF without a text layer) is not worth a full-text reading.
MINIMUM_USABLE_EXTRACTED_TEXT_CHARACTERS = 1500


class ReadingEnqueueOutcome(StrEnum):
    QUEUED = "queued"
    ALREADY_REVIEWED = "already_reviewed"  # a review of exactly this input exists
    ALREADY_HAS_JOB = "already_has_job"  # a job for this input exists (queued, running, done or failed)
    NO_EDITION = "no_edition"  # nothing to read yet


@dataclass(frozen=True)
class ReadingJobInput:
    """What a review_paper job stores in jobs.input_json: exactly which edition, text, and generation to read."""

    edition_id: str
    reading_generation: str
    text_kind: str = TEXT_KIND_ABSTRACT
    operator_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "edition_id": self.edition_id,
            "reading_generation": self.reading_generation,
            "text_kind": self.text_kind,
        }
        if self.operator_reason:
            payload["operator_reason"] = self.operator_reason
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ReadingJobInput:
        """Jobs queued by earlier versions have no text_kind: they were abstract readings."""
        return cls(
            edition_id=payload["edition_id"],
            reading_generation=payload.get("reading_generation", INITIAL_READING_GENERATION),
            text_kind=payload.get("text_kind", TEXT_KIND_ABSTRACT),
            operator_reason=payload.get("operator_reason"),
        )


# The edition an abstract reading should use: one with an abstract if any, then the most recently first-seen, then
# the highest version number. A stale re-sighting of an old version never becomes "latest" (first_seen_at is kept).
# Shared with screening (paperwatch.screening.candidate_screening), so a paper is scored on the text it will be read on.
PREFERRED_EDITION_ORDER_SQL = """(abstract IS NOT NULL AND TRIM(abstract) != '') DESC, first_seen_at DESC,
                    COALESCE(version_number, 0) DESC, edition_id DESC"""
# The edition whose full text is preferred: arXiv first (newest version first), then OpenAlex, then the rest (newest
# sighting first). Shared with source-text acquisition (paperwatch.collection.source_text), so the edition whose PDF
# is fetched first is the edition that is read.
FULL_TEXT_EDITION_ORDER_SQL = """CASE WHEN edition_key LIKE 'arxiv:%' THEN 0 WHEN edition_key LIKE 'openalex:%'
                    THEN 1 ELSE 2 END, COALESCE(version_number, 0) DESC, first_seen_at DESC, edition_id"""


def select_edition_to_read(connection: sqlite3.Connection, paper_id: str) -> sqlite3.Row | None:
    edition_row: sqlite3.Row | None = connection.execute(
        f"SELECT * FROM paper_editions WHERE paper_id = ? ORDER BY {PREFERRED_EDITION_ORDER_SQL} LIMIT 1",  # noqa: S608  # interpolates module constants only; values are bound
        (paper_id,),
    ).fetchone()
    return edition_row


def select_reading_input(
    connection: sqlite3.Connection, paper_id: str, reading_generation: str = INITIAL_READING_GENERATION
) -> ReadingInputIdentity | None:
    """The best input available now: usable extracted full text, else the preferred edition's abstract.

    Uses only database columns (no file is read), so the collector can call it inside a write transaction. The
    worker verifies the text file itself before reading (paperwatch.reading.review_packet)."""
    extracted_row = connection.execute(
        f"""WITH readable_editions AS (
                SELECT edition.edition_id, edition.edition_key, edition.version_number, edition.first_seen_at,
                       source_text.normalized_text_sha256, source_text.extracted_text_file_name
                FROM paper_editions AS edition
                JOIN edition_source_texts AS source_text ON source_text.edition_id = edition.edition_id
                WHERE edition.paper_id = ? AND source_text.extraction_status = 'extracted'
                  AND source_text.normalized_text_sha256 IS NOT NULL
                  AND source_text.extracted_text_characters >= ?)
            SELECT * FROM readable_editions ORDER BY {FULL_TEXT_EDITION_ORDER_SQL} LIMIT 1""",  # noqa: S608  # interpolates module constants only; values are bound
        (paper_id, MINIMUM_USABLE_EXTRACTED_TEXT_CHARACTERS),
    ).fetchone()
    if extracted_row is not None:
        return ReadingInputIdentity(
            edition_id=extracted_row["edition_id"],
            text_kind=TEXT_KIND_EXTRACTED_TEXT,
            source_text_sha256=extracted_row["normalized_text_sha256"],
            reading_generation=reading_generation,
            extracted_text_file_name=extracted_row["extracted_text_file_name"],
        )
    edition_row = select_edition_to_read(connection, paper_id)
    return build_abstract_reading_identity(edition_row, reading_generation) if edition_row is not None else None


def review_exists_for_input(connection: sqlite3.Connection, paper_id: str, review_input_key: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM reviews WHERE paper_id = ? AND review_input_key = ?", (paper_id, review_input_key)
        ).fetchone()
        is not None
    )


def enqueue_reading_if_needed(
    connection: sqlite3.Connection,
    paper_id: str,
    *,
    dispatch_order: JobDispatchOrder,
    max_attempts: int,
    reading_generation: str = INITIAL_READING_GENERATION,
    operator_reason: str | None = None,
    hold_until: datetime | None = None,
) -> tuple[ReadingEnqueueOutcome, ReadingInputIdentity | None]:
    """Queue a reading of the paper's best available input unless this exact input was already read or queued.

    If the job already exists and is still queued, it is moved up to `dispatch_order` when that is a higher class
    (a paper qualifying for several classes runs in the highest one), never down. Older queued readings of the same
    paper and generation are retired: the new input supersedes them. `hold_until` delays a new job (see the module
    docstring). Call inside a write transaction. Does not change the paper's status; the caller decides that.
    """
    reading_input = select_reading_input(connection, paper_id, reading_generation)
    if reading_input is None:
        return ReadingEnqueueOutcome.NO_EDITION, None
    if review_exists_for_input(connection, paper_id, reading_input.review_input_key):
        return ReadingEnqueueOutcome.ALREADY_REVIEWED, reading_input
    review_job_key = build_review_job_key(paper_id, reading_input.review_input_key)
    job_was_added = enqueue_job_if_absent(
        connection,
        job_key=review_job_key,
        kind=REVIEW_PAPER_JOB_KIND,
        paper_id=paper_id,
        input_version=reading_input.review_input_key,
        input_payload=ReadingJobInput(
            edition_id=reading_input.edition_id,
            reading_generation=reading_generation,
            text_kind=reading_input.text_kind,
            operator_reason=operator_reason,
        ).to_payload(),
        dispatch_order=dispatch_order,
        max_attempts=max_attempts,
        due_at=format_utc_timestamp(hold_until) if hold_until is not None else None,
    )
    if not job_was_added:
        upgrade_queued_job_dispatch_order_if_higher(connection, review_job_key, dispatch_order)
    retire_superseded_queued_readings(connection, paper_id, reading_input)
    return (ReadingEnqueueOutcome.QUEUED if job_was_added else ReadingEnqueueOutcome.ALREADY_HAS_JOB), reading_input


def retire_superseded_queued_readings(
    connection: sqlite3.Connection, paper_id: str, current_input: ReadingInputIdentity
) -> int:
    """Mark still-QUEUED readings of the same paper and generation on another input as done (superseded). A running
    job is left alone: its worker notices at commit time that its input is no longer current."""
    retire_cursor = connection.execute(
        """UPDATE jobs SET status = 'done', last_error_component = NULL,
               last_error = 'superseded by a newer input: ' || ?, updated_at = ?
           WHERE kind = ? AND paper_id = ? AND status = 'queued' AND input_version != ?
             AND COALESCE(json_extract(input_json, '$.reading_generation'), ?) = ?""",
        (
            current_input.review_input_key,
            utc_now_timestamp(),
            REVIEW_PAPER_JOB_KIND,
            paper_id,
            current_input.review_input_key,
            INITIAL_READING_GENERATION,
            current_input.reading_generation,
        ),
    )
    return retire_cursor.rowcount


def compute_source_text_hold_until(moment: datetime, hold_minutes: int) -> datetime | None:
    return moment + timedelta(minutes=hold_minutes) if hold_minutes > 0 else None


def release_readings_held_for_source_text(connection: sqlite3.Connection) -> int:
    """Make every held reading due now. A held job is a queued review job that has never been attempted and whose due
    time is still in the future (retries always have attempts > 0). Returns how many were released."""
    now_timestamp = utc_now_timestamp()
    release_cursor = connection.execute(
        """UPDATE jobs SET due_at = ?, updated_at = ?
           WHERE kind = ? AND status = 'queued' AND attempts = 0 AND due_at > ?""",
        (now_timestamp, now_timestamp, REVIEW_PAPER_JOB_KIND, now_timestamp),
    )
    return release_cursor.rowcount


def dispatch_order_of_job_row(job_row: sqlite3.Row) -> JobDispatchOrder:
    """The dispatch order a queued job was given, so a superseding job can be queued in the same place."""
    return JobDispatchOrder(
        priority_class=JobPriorityClass(job_row["priority"]),
        within_class_order_primary=job_row["within_class_order_primary"],
        within_class_order_secondary=job_row["within_class_order_secondary"],
        new_release_fresh_until=job_row["new_release_fresh_until"],
    )


def new_reading_generation_id() -> str:
    """A fresh generation id for one explicit operator reread, e.g. 'g_3f9a1c0b7d2e4a51'."""
    return new_internal_id(READING_GENERATION_ID_PREFIX)


def refresh_paper_status_from_work(connection: sqlite3.Connection, paper_id: str) -> str:
    """Derive the paper status from its jobs and reviews (used when a job ends without its own clear outcome).

    running job -> reviewing; queued job -> queued; any review -> complete; failed job -> failed;
    otherwise the status is left unchanged (e.g. waiting_input or screened_out).
    """
    status_row = connection.execute(
        """SELECT
               EXISTS(SELECT 1 FROM jobs WHERE paper_id = :paper_id AND status = 'running') AS has_running_job,
               EXISTS(SELECT 1 FROM jobs WHERE paper_id = :paper_id AND status = 'queued') AS has_queued_job,
               EXISTS(SELECT 1 FROM reviews WHERE paper_id = :paper_id) AS has_review,
               EXISTS(SELECT 1 FROM jobs WHERE paper_id = :paper_id AND status = 'failed') AS has_failed_job,
               (SELECT status FROM papers WHERE paper_id = :paper_id) AS current_status""",
        {"paper_id": paper_id},
    ).fetchone()
    derived_status = (
        "reviewing"
        if status_row["has_running_job"]
        else "queued"
        if status_row["has_queued_job"]
        else "complete"
        if status_row["has_review"]
        else "failed"
        if status_row["has_failed_job"]
        else status_row["current_status"]
    )
    connection.execute(
        "UPDATE papers SET status = ?, updated_at = ? WHERE paper_id = ? AND status != ?",
        (derived_status, utc_now_timestamp(), paper_id, derived_status),
    )
    return derived_status
