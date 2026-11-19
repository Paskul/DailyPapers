"""Decide when a paper needs a (new) reading, and queue it as one logical job.

A reading is identified by (canonical paper, normalized source-text hash, reading generation); see review_inputs.py.
The job key embeds that identity, so the queue's UNIQUE(job_key) is the final guard against a duplicate reading even
when two collectors race. Consequences, matching plan part 2:
  * a changed title, another discovery route, a topic or scoring change, a restart, or an empty queue never creates
    another reading, because none of them changes the text hash or the generation;
  * a model or prompt upgrade alone never re-enqueues the archive, because neither is part of the key;
  * a new edition with identical text is not read again; a verified text change or newly available text gets a new
    reading (subject to promotion, plan part 6);
  * an explicit operator reread gets a fresh generation id: a new logical job and, later, a separate review.
A job that failed permanently is NOT re-queued automatically for the same input; the operator can request a reread.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from paperwatch.clock import utc_now_timestamp
from paperwatch.identifiers import READING_GENERATION_ID_PREFIX, new_internal_id
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND, build_review_job_key, enqueue_job_if_absent
from paperwatch.review_inputs import INITIAL_READING_GENERATION, ReviewPacket, build_abstract_review_packet


class ReadingEnqueueOutcome(StrEnum):
    QUEUED = "queued"
    ALREADY_REVIEWED = "already_reviewed"  # a review of exactly this input exists
    ALREADY_HAS_JOB = "already_has_job"  # a job for this input exists (queued, running, done or failed)
    NO_EDITION = "no_edition"  # nothing to read yet


@dataclass(frozen=True)
class ReadingJobInput:
    """What a review_paper job stores in jobs.input_json: exactly which edition and generation to read."""

    edition_id: str
    reading_generation: str
    coverage: str
    operator_reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {"edition_id": self.edition_id, "reading_generation": self.reading_generation,
                   "coverage": self.coverage}
        if self.operator_reason:
            payload["operator_reason"] = self.operator_reason
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ReadingJobInput:
        return cls(edition_id=payload["edition_id"],
                   reading_generation=payload.get("reading_generation", INITIAL_READING_GENERATION),
                   coverage=payload["coverage"], operator_reason=payload.get("operator_reason"))


def select_edition_to_read(connection: sqlite3.Connection, paper_id: str) -> sqlite3.Row | None:
    """The edition a new reading should use: one with an abstract if any, then the most recently first-seen, then the
    highest version number. A stale re-sighting of an old version never becomes "latest" (first_seen_at is kept)."""
    return connection.execute(
        """SELECT * FROM paper_editions WHERE paper_id = ?
           ORDER BY (abstract IS NOT NULL AND TRIM(abstract) != '') DESC, first_seen_at DESC,
                    COALESCE(version_number, 0) DESC, edition_id DESC
           LIMIT 1""", (paper_id,)).fetchone()


def review_exists_for_input(connection: sqlite3.Connection, paper_id: str, review_input_key: str) -> bool:
    return connection.execute("SELECT 1 FROM reviews WHERE paper_id = ? AND review_input_key = ?",
                              (paper_id, review_input_key)).fetchone() is not None


def enqueue_reading_if_needed(connection: sqlite3.Connection, paper_id: str, *, priority: int, max_attempts: int,
                              reading_generation: str = INITIAL_READING_GENERATION,
                              operator_reason: str | None = None) -> tuple[ReadingEnqueueOutcome, ReviewPacket | None]:
    """Queue a reading of the paper's preferred edition unless this exact input was already read or queued.

    Call inside a write transaction. Does not change the paper's status; the caller decides that.
    """
    edition_row = select_edition_to_read(connection, paper_id)
    if edition_row is None:
        return ReadingEnqueueOutcome.NO_EDITION, None
    review_packet = build_abstract_review_packet(edition_row, reading_generation)
    if review_exists_for_input(connection, paper_id, review_packet.review_input_key):
        return ReadingEnqueueOutcome.ALREADY_REVIEWED, review_packet
    job_was_added = enqueue_job_if_absent(
        connection, job_key=build_review_job_key(paper_id, review_packet.review_input_key),
        kind=REVIEW_PAPER_JOB_KIND, paper_id=paper_id, input_version=review_packet.review_input_key,
        input_payload=ReadingJobInput(edition_id=review_packet.edition_id, reading_generation=reading_generation,
                                      coverage=review_packet.coverage, operator_reason=operator_reason).to_payload(),
        priority=priority, max_attempts=max_attempts)
    return (ReadingEnqueueOutcome.QUEUED if job_was_added else ReadingEnqueueOutcome.ALREADY_HAS_JOB), review_packet


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
        {"paper_id": paper_id}).fetchone()
    derived_status = ("reviewing" if status_row["has_running_job"] else
                      "queued" if status_row["has_queued_job"] else
                      "complete" if status_row["has_review"] else
                      "failed" if status_row["has_failed_job"] else status_row["current_status"])
    connection.execute("UPDATE papers SET status = ?, updated_at = ? WHERE paper_id = ? AND status != ?",
                       (derived_status, utc_now_timestamp(), paper_id, derived_status))
    return derived_status
