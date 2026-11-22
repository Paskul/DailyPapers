"""Queue comparisons of two completed reviews on the same question (plan part 8).

"Queue one comparison when two completed reviews address the same question and have comparable assumptions,
guarantees, or evaluation settings. Prefer the closest reviewed neighbor and cap new comparison jobs at five per day."
For a review R (the newest review of its paper):
  1. the questions R addresses: topics whose newest assessment of R rated question_fit >= 2, best score first;
  2. for each, R's neighbours: the newest reviews of other papers that also rated question_fit >= 2 for that topic,
     nearest first by the cached screening embeddings (paperwatch.reading.comparison_anchors.rank_comparable_reviews;
     nothing is embedded here, so the worker can do it);
  3. the first neighbour sharing at least comparisons.minimum_shared_evidence_dimensions (evidence_dimensions.py)
     is R's comparison partner. If that pair is already queued or compared, R is done: one comparison per review.
A review that already takes part in any comparison job (queued by itself or by a later neighbour) gets no further
one, even when a closer neighbour is reviewed later; that neighbour's own commit may still pair with it.
The job key is the comparison key (topic, question hash, sorted pair of review ids), so the same pair is compared
once whichever review found it first. At most comparisons.daily_comparison_jobs jobs are created per UTC day, counted
from the jobs table (so the cap holds across restarts). The worker queues for each new review in the review's own
commit; `collect` sweeps recent reviews the cap held back.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from paperwatch.clock import format_utc_timestamp
from paperwatch.comparisons.evidence_dimensions import shared_evidence_dimensions
from paperwatch.config import AppConfig
from paperwatch.dispatch_order import dispatch_order_for_comparison
from paperwatch.jobs import COMPARE_REVIEWS_JOB_KIND, enqueue_job_if_absent
from paperwatch.reading.comparison_anchors import CachedVectorLookup, rank_comparable_reviews
from paperwatch.readings import select_edition_to_read
from paperwatch.research_graph import build_comparison_key
from paperwatch.review_store import load_latest_review_id_of_paper, load_latest_topic_assessments
from paperwatch.screening.text_embedders import configured_embedder_identity

MINIMUM_QUESTION_FIT_FOR_COMPARISON = 2


def compute_question_sha256(question_text: str) -> str:
    return hashlib.sha256(question_text.encode("utf-8")).hexdigest()


def build_compare_reviews_job_key(comparison_key: str) -> str:
    return f"{COMPARE_REVIEWS_JOB_KIND}:{comparison_key}"


@dataclass(frozen=True)
class ComparisonJobInput:
    topic_id: str
    question_sha256: str
    review_id_a: str  # the lower review id of the pair
    review_id_b: str
    shared_dimensions: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "question_sha256": self.question_sha256,
            "review_id_a": self.review_id_a,
            "review_id_b": self.review_id_b,
            "shared_dimensions": list(self.shared_dimensions),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ComparisonJobInput:
        return cls(
            payload["topic_id"],
            payload["question_sha256"],
            payload["review_id_a"],
            payload["review_id_b"],
            tuple(payload.get("shared_dimensions", [])),
        )

    @property
    def comparison_key(self) -> str:
        return build_comparison_key(self.topic_id, self.question_sha256, self.review_id_a, self.review_id_b)


def count_comparison_jobs_created_today(connection: sqlite3.Connection, moment: datetime) -> int:
    day_start = format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM jobs WHERE kind = ? AND created_at >= ?", (COMPARE_REVIEWS_JOB_KIND, day_start)
        ).fetchone()[0]
    )


def review_has_comparison_job(connection: sqlite3.Connection, review_id: str) -> bool:
    return (
        connection.execute(
            """SELECT 1 FROM jobs WHERE kind = ? AND (json_extract(input_json, '$.review_id_a') = ?
                                                 OR json_extract(input_json, '$.review_id_b') = ?) LIMIT 1""",
            (COMPARE_REVIEWS_JOB_KIND, review_id, review_id),
        ).fetchone()
        is not None
    )


def queue_comparison_for_review(
    connection: sqlite3.Connection, app_config: AppConfig, review_id: str, moment: datetime
) -> str | None:
    """Queue the review's comparison with its closest comparable neighbour, if the daily limit allows. Call inside a
    write transaction. Returns the new job's comparison key, or None if nothing was queued."""
    if count_comparison_jobs_created_today(connection, moment) >= app_config.comparisons.daily_comparison_jobs:
        return None
    if review_has_comparison_job(connection, review_id):
        return None
    review_row = connection.execute(
        "SELECT review_id, paper_id, review_json FROM reviews WHERE review_id = ?", (review_id,)
    ).fetchone()
    if review_row is None or load_latest_review_id_of_paper(connection, review_row["paper_id"]) != review_id:
        return None
    topic_by_id = {topic.topic_id: topic for topic in app_config.topics}
    addressed_assessments = sorted(
        (
            assessment
            for assessment in load_latest_topic_assessments(connection, review_id)
            if assessment.question_fit >= MINIMUM_QUESTION_FIT_FOR_COMPARISON and assessment.topic_id in topic_by_id
        ),
        key=lambda assessment: (-assessment.priority_score_out_of_100, assessment.topic_id),
    )
    if not addressed_assessments:
        return None
    review_content = json.loads(review_row["review_json"])
    vector_lookup = CachedVectorLookup(connection, configured_embedder_identity(app_config))
    edition_row = select_edition_to_read(connection, review_row["paper_id"])
    paper_vector = (
        vector_lookup.vector_for_document(edition_row["title"], edition_row["abstract"])
        if edition_row is not None
        else None
    )
    for assessment in addressed_assessments:
        for neighbour_row in rank_comparable_reviews(
            connection, vector_lookup, paper_vector, review_row["paper_id"], assessment.topic_id
        ):
            shared_dimensions = shared_evidence_dimensions(review_content, json.loads(neighbour_row["review_json"]))
            if len(shared_dimensions) < app_config.comparisons.minimum_shared_evidence_dimensions:
                continue
            lower_review_id, higher_review_id = sorted((review_id, neighbour_row["review_id"]))
            job_input = ComparisonJobInput(
                topic_id=assessment.topic_id,
                question_sha256=compute_question_sha256(topic_by_id[assessment.topic_id].question_for_local_model),
                review_id_a=lower_review_id,
                review_id_b=higher_review_id,
                shared_dimensions=tuple(shared_dimensions),
            )
            was_added = enqueue_job_if_absent(
                connection,
                job_key=build_compare_reviews_job_key(job_input.comparison_key),
                kind=COMPARE_REVIEWS_JOB_KIND,
                paper_id=None,
                input_version=job_input.comparison_key,
                input_payload=job_input.to_payload(),
                dispatch_order=dispatch_order_for_comparison(len(shared_dimensions)),
                max_attempts=app_config.worker.max_attempts_per_job,
            )
            return job_input.comparison_key if was_added else None  # the closest comparable neighbour only
    return None


def queue_comparisons_for_recent_reviews(
    connection: sqlite3.Connection, app_config: AppConfig, moment: datetime
) -> int:
    """`collect`'s sweep: reviews completed in the last comparisons.sweep_reviews_completed_within_days that are in no
    comparison job yet, newest first, until today's limit is used. Call inside a write transaction."""
    since = format_utc_timestamp(moment - timedelta(days=app_config.comparisons.sweep_reviews_completed_within_days))
    review_ids = [
        row["review_id"]
        for row in connection.execute(
            """SELECT review.review_id FROM reviews AS review WHERE review.completed_at >= ?
             AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.kind = ?
                             AND (json_extract(jobs.input_json, '$.review_id_a') = review.review_id
                                  OR json_extract(jobs.input_json, '$.review_id_b') = review.review_id))
           ORDER BY review.completed_at DESC, review.review_id""",
            (since, COMPARE_REVIEWS_JOB_KIND),
        )
    ]
    queued_count = 0
    for review_id in review_ids:
        if count_comparison_jobs_created_today(connection, moment) >= app_config.comparisons.daily_comparison_jobs:
            break
        queued_count += queue_comparison_for_review(connection, app_config, review_id, moment) is not None
    return queued_count
