"""Which reviewed papers get their references followed, and how far each paper is from a screened paper.

Hop depth (plan part 8: "start with two citation hops"):
    0   a paper promoted by screening (or directly by the operator)
    n   a paper read because the research role chose it from the references of a hop n-1 paper
A paper's depth is the smallest depth under which it was admitted. References of a paper at depth max_citation_hops
are never followed, so following references always ends. A paper reached twice is read once (the reading job is
unique per input), and a paper that is already reviewed or queued is never admitted again, which also ends cycles
such as two papers that cite each other.

A reviewed paper is followed only if its latest review rated question_fit >= references.minimum_parent_question_fit
for at least one topic (owner decision 2026-11-20: the same "addresses part of the question" bar as comparison
anchors), it is not withdrawn, and its references were not fetched before.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from paperwatch.config import AppConfig
from paperwatch.review_store import best_question_fit_and_priority, load_latest_topic_assessments

PROMOTION_REASON_REFERENCE = "reference"
MAXIMUM_REFERENCE_FETCH_ATTEMPTS = 3
FETCH_STATUSES_THAT_ARE_FINAL = ("fetched", "no_provider_record")


@dataclass(frozen=True)
class ReviewedPaperToFollow:
    paper_id: str
    review_id: str
    citation_hop_depth: int
    best_question_fit: int
    best_reading_priority: float  # 0-100


def citation_hop_depth_of_paper(connection: sqlite3.Connection, paper_id: str) -> int:
    promotion_row = connection.execute(
        "SELECT promotion_reason FROM paper_promotions WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if promotion_row is not None and promotion_row["promotion_reason"] != PROMOTION_REASON_REFERENCE:
        return 0
    depth_row = connection.execute(
        """SELECT MIN(citation_hop_depth) AS depth FROM reference_selections
           WHERE cited_paper_id = ? AND admission_status = 'admitted'""",
        (paper_id,),
    ).fetchone()
    return int(depth_row["depth"]) if depth_row["depth"] is not None else 0


def latest_reviews_of_unwithdrawn_papers(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """The newest review of each paper that is not withdrawn, newest completion first."""
    return connection.execute(
        """SELECT review.review_id, review.paper_id, review.completed_at FROM reviews AS review
           WHERE review.paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
             AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                     WHERE newest.paper_id = review.paper_id
                                     ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)
           ORDER BY review.completed_at DESC, review.review_id"""
    ).fetchall()


def describe_followable_review(
    connection: sqlite3.Connection, app_config: AppConfig, paper_id: str, review_id: str
) -> ReviewedPaperToFollow | None:
    """The review as a parent whose references may be followed, or None if it does not qualify."""
    reference_settings = app_config.references
    best_question_fit, best_priority = best_question_fit_and_priority(
        load_latest_topic_assessments(connection, review_id)
    )
    if best_question_fit < reference_settings.minimum_parent_question_fit:
        return None
    hop_depth = citation_hop_depth_of_paper(connection, paper_id)
    if hop_depth >= reference_settings.max_citation_hops:
        return None
    return ReviewedPaperToFollow(paper_id, review_id, hop_depth, best_question_fit, best_priority)


def find_reviewed_papers_to_follow(
    connection: sqlite3.Connection, app_config: AppConfig, limit: int
) -> list[ReviewedPaperToFollow]:
    """Reviewed papers whose references have not been fetched yet (a blocked fetch is retried a few times), the
    highest reading priority first."""
    finished_or_given_up = {
        row["citing_paper_id"]
        for row in connection.execute(
            f"""SELECT citing_paper_id FROM reference_fetches
            WHERE fetch_status IN ({", ".join("?" * len(FETCH_STATUSES_THAT_ARE_FINAL))}) OR attempts >= ?""",
            (*FETCH_STATUSES_THAT_ARE_FINAL, MAXIMUM_REFERENCE_FETCH_ATTEMPTS),
        )
    }
    followable = []
    for review_row in latest_reviews_of_unwithdrawn_papers(connection):
        if review_row["paper_id"] in finished_or_given_up:
            continue
        parent = describe_followable_review(connection, app_config, review_row["paper_id"], review_row["review_id"])
        if parent is not None:
            followable.append(parent)
    followable.sort(key=lambda parent: (-parent.best_reading_priority, parent.paper_id))
    return followable[:limit]
