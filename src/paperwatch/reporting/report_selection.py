"""Choose the papers of one daily report from completed reviews (plan part 9). Pure reads; no model call.

Eligible: the LATEST review per canonical paper among those completed at or before the cutoff, whose best topic score
(as assessed by the cutoff) reaches report.minimum_reading_priority, except
  * withdrawn or retracted papers (plan part 5),
  * papers in the user's active seed library (they stay comparison anchors, never recommendations),
  * papers an earlier report already recommended, UNLESS the newer review is an update that changes a result,
    assumption, or conclusion (reporting/revision_changes.py); then it is labelled "update" and both reviews are
    linked.
Order: final score (best reading priority) descending, then evidence coverage (full text > partial > abstract only),
then the stable paper id. The first report.max_papers_per_report are chosen; the other eligible reviews are simply
not reported yet, so a later report picks them up (nothing is marked for them).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from enum import StrEnum

from paperwatch.clock import parse_utc_timestamp
from paperwatch.publication_dates import ReleaseClass, classify_release
from paperwatch.reporting.revision_changes import (
    RevisionChange,
    ReviewedVersion,
    describe_material_revision_changes,
    text_fingerprint_of_review,
)
from paperwatch.review_store import TopicAssessment, load_latest_topic_assessments

COVERAGE_ORDER = {"full_text": 3, "partial_text": 2, "abstract_only": 1}
REFERENCE_PROMOTION_REASON = "reference"


class EntryLabel(StrEnum):
    NEW_RELEASE = "new_release"
    OLDER = "older"
    REFERENCE = "reference"
    UPDATE = "update"
    UNDATED = "undated"


ENTRY_LABEL_TEXT = {
    EntryLabel.NEW_RELEASE: "new release",
    EntryLabel.OLDER: "older paper",
    EntryLabel.REFERENCE: "reference",
    EntryLabel.UPDATE: "update",
    EntryLabel.UNDATED: "publication date unknown",
    "legacy": "earlier entry",
}


@dataclass(frozen=True)
class ReportEntry:
    """One ranked paper of a report, exactly as frozen at the snapshot."""

    rank: int
    paper_id: str
    review_id: str
    entry_label: str  # an EntryLabel value
    reading_priority: int | None  # best topic score 0-100 (None: not rated)
    best_topic_id: str | None
    coverage: str
    previous_review_id: str | None = None  # update: the review the earlier report used
    previous_report_date: str | None = None
    revision_changes: tuple[RevisionChange, ...] = field(default_factory=tuple)

    @property
    def is_update(self) -> bool:
        return self.entry_label == EntryLabel.UPDATE


@dataclass(frozen=True)
class ReportSelection:
    entries: tuple[ReportEntry, ...]
    eligible_held_for_later_count: int  # above the floor, but beyond today's limit
    below_floor_count: int  # latest unreported reviews that do not reach the floor


def best_topic_assessment(topic_assessments: list[TopicAssessment]) -> TopicAssessment | None:
    """Highest reading priority; ties go to the lowest topic id (deterministic)."""
    return min(
        topic_assessments,
        key=lambda assessment: (-assessment.priority_score_out_of_100, assessment.topic_id),
        default=None,
    )


def select_report_entries(
    connection: sqlite3.Connection,
    *,
    report_date: str,
    cutoff_at: str,
    minimum_reading_priority: int,
    maximum_entries: int,
    excluded_paper_ids: frozenset[str],
) -> ReportSelection:
    """`cutoff_at` is a stored UTC timestamp; `excluded_paper_ids` are the active seed-library papers."""
    latest_review_rows = connection.execute(
        """SELECT review.*, paper.first_published_date, paper.discovered_at, promotion.promotion_reason
           FROM reviews AS review
           JOIN papers AS paper ON paper.paper_id = review.paper_id
           LEFT JOIN paper_promotions AS promotion ON promotion.paper_id = review.paper_id
           WHERE review.completed_at <= :cutoff
             AND review.paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
             AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                     WHERE newest.paper_id = review.paper_id AND newest.completed_at <= :cutoff
                                     ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)""",
        {"cutoff": cutoff_at},
    ).fetchall()

    candidates: list[ReportEntry] = []
    below_floor_count = 0
    for review_row in latest_review_rows:
        if review_row["paper_id"] in excluded_paper_ids:
            continue
        best_assessment = best_topic_assessment(
            load_latest_topic_assessments(connection, review_row["review_id"], assessed_no_later_than=cutoff_at)
        )
        reading_priority = None if best_assessment is None else round(best_assessment.priority_score_out_of_100)
        previous_item = connection.execute(
            """SELECT review_id, report_date, reading_priority FROM report_items
               WHERE paper_id = ? AND report_date != ? ORDER BY report_date DESC LIMIT 1""",
            (review_row["paper_id"], report_date),
        ).fetchone()
        if previous_item is not None and previous_item["review_id"] == review_row["review_id"]:
            continue  # already recommended with this very review
        if not _reaches_floor(reading_priority, minimum_reading_priority):
            below_floor_count += previous_item is None
            continue
        entry = ReportEntry(
            rank=0,
            paper_id=review_row["paper_id"],
            review_id=review_row["review_id"],
            entry_label=_label_for_new_entry(review_row),
            reading_priority=reading_priority,
            best_topic_id=None if best_assessment is None else best_assessment.topic_id,
            coverage=review_row["coverage"],
        )
        if previous_item is not None:
            revision_changes = _revision_changes_since_reported(
                connection, previous_item, review_row, reading_priority, cutoff_at
            )
            if not revision_changes:
                continue  # a newer reading that changes no result, assumption, or conclusion
            entry = replace(
                entry,
                entry_label=EntryLabel.UPDATE.value,
                previous_review_id=previous_item["review_id"],
                previous_report_date=previous_item["report_date"],
                revision_changes=tuple(revision_changes),
            )
        candidates.append(entry)

    candidates.sort(
        key=lambda candidate: (
            -(candidate.reading_priority if candidate.reading_priority is not None else -1),
            -COVERAGE_ORDER.get(candidate.coverage, 0),
            candidate.paper_id,
        )
    )
    chosen = [
        replace(candidate, rank=position) for position, candidate in enumerate(candidates[:maximum_entries], start=1)
    ]
    return ReportSelection(
        entries=tuple(chosen),
        eligible_held_for_later_count=max(0, len(candidates) - maximum_entries),
        below_floor_count=below_floor_count,
    )


def _reaches_floor(reading_priority: int | None, minimum_reading_priority: int) -> bool:
    """An unrated review (no topic assessment) qualifies only when the floor is 0."""
    if reading_priority is None:
        return minimum_reading_priority <= 0
    return reading_priority >= minimum_reading_priority


def _label_for_new_entry(review_row: sqlite3.Row) -> str:
    if review_row["promotion_reason"] == REFERENCE_PROMOTION_REASON:
        return EntryLabel.REFERENCE.value
    release_class = classify_release(
        review_row["first_published_date"], parse_utc_timestamp(review_row["discovered_at"])
    )
    return {
        ReleaseClass.NEW_RELEASE: EntryLabel.NEW_RELEASE,
        ReleaseClass.OLDER_LITERATURE: EntryLabel.OLDER,
        ReleaseClass.UNKNOWN: EntryLabel.UNDATED,
    }[release_class].value


def _revision_changes_since_reported(
    connection: sqlite3.Connection,
    previous_item: sqlite3.Row,
    newer_review_row: sqlite3.Row,
    newer_reading_priority: int | None,
    cutoff_at: str,
) -> list[RevisionChange]:
    reported_review_row = connection.execute(
        "SELECT * FROM reviews WHERE review_id = ?", (previous_item["review_id"],)
    ).fetchone()
    reported_priority = previous_item["reading_priority"]
    if reported_priority is None:  # reported before part 9 froze scores: use its assessments as of now
        reported_best = best_topic_assessment(
            load_latest_topic_assessments(
                connection, reported_review_row["review_id"], assessed_no_later_than=cutoff_at
            )
        )
        reported_priority = None if reported_best is None else round(reported_best.priority_score_out_of_100)
    return describe_material_revision_changes(
        ReviewedVersion(
            json.loads(reported_review_row["review_json"]),
            text_fingerprint_of_review(
                reported_review_row["source_text_sha256"], reported_review_row["review_input_key"]
            ),
            reported_priority,
        ),
        ReviewedVersion(
            json.loads(newer_review_row["review_json"]),
            text_fingerprint_of_review(newer_review_row["source_text_sha256"], newer_review_row["review_input_key"]),
            newer_reading_priority,
        ),
    )
