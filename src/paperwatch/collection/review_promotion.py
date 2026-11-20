"""Which collected papers are queued for a reading, and at which priority class.

PLACEHOLDERS until the CPU screening score of plan part 6:
  * every collected paper is promoted, except papers in the user's active seed library (comparison anchors, never
    recommendations; plan parts 4 and 9) and papers a provider reported withdrawn or retracted (plan part 5);
  * the priority class follows the release date: a paper first published within the seven-day fresh window is a new
    release (class 1); older literature found by any route is class 2. Part 6 adds scores and the other classes.
"""

from __future__ import annotations

import sqlite3

from paperwatch.catalog import is_paper_withdrawn, load_paper_row, set_paper_status
from paperwatch.clock import parse_utc_timestamp
from paperwatch.db import write_transaction
from paperwatch.jobs import JobPriorityClass
from paperwatch.publication_dates import ReleaseClass, classify_release
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed


def should_promote_paper_for_review(connection: sqlite3.Connection, paper_row: sqlite3.Row,
                                    active_seed_paper_ids: frozenset[str]) -> bool:
    if paper_row["paper_id"] in active_seed_paper_ids:
        return False
    return not is_paper_withdrawn(connection, paper_row["paper_id"])


def choose_review_priority_class(paper_row: sqlite3.Row) -> JobPriorityClass:
    release_class = classify_release(paper_row["first_published_date"],
                                     parse_utc_timestamp(paper_row["discovered_at"]))
    return JobPriorityClass.HISTORICAL_SEARCH if release_class == ReleaseClass.OLDER_LITERATURE \
        else JobPriorityClass.NEW_RELEASE


def queue_review_if_promoted_in_transaction(connection: sqlite3.Connection, paper_id: str, *,
                                            active_seed_paper_ids: frozenset[str], max_attempts: int) -> bool:
    """Queue a reading of the paper unless it is not promoted or this exact input was already read or queued.
    Call inside a write transaction. Returns True if a job was added."""
    paper_row = load_paper_row(connection, paper_id)
    if not should_promote_paper_for_review(connection, paper_row, active_seed_paper_ids):
        return False
    enqueue_outcome, _review_packet = enqueue_reading_if_needed(
        connection, paper_id, priority=choose_review_priority_class(paper_row), max_attempts=max_attempts)
    if enqueue_outcome == ReadingEnqueueOutcome.QUEUED:
        set_paper_status(connection, paper_id, "queued")
        return True
    return False


def queue_review_if_promoted(connection: sqlite3.Connection, paper_id: str, *, max_attempts: int,
                             active_seed_paper_ids: frozenset[str] = frozenset()) -> bool:
    """The same decision for one paper, in its own short transaction (tests, operator tools)."""
    with write_transaction(connection):
        return queue_review_if_promoted_in_transaction(connection, paper_id,
                                                       active_seed_paper_ids=active_seed_paper_ids,
                                                       max_attempts=max_attempts)
