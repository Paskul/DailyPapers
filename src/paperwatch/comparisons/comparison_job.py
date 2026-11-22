"""The worker's side of a comparison job (plan part 8): build the comparison role's input from the two saved
reviews, and store the validated comparison once.

Nothing is reread: each review contributes its validated fields and the passages it cites (review_evidence_passages,
the same text its own evidence was checked against). If either review is gone, or the topic's question changed
since the job was queued (a different question hash), the job ends without a model call: a comparison is only valid
for the question it was asked under, and a changed question gets its own comparison through the normal queueing.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.comparisons.comparison_queue import ComparisonJobInput, compute_question_sha256
from paperwatch.config import AppConfig
from paperwatch.reading.reading_pipeline import summarize_review_for_assessment
from paperwatch.research_graph import record_comparison
from paperwatch.review_store import load_review_evidence_passages
from paperwatch.roles.base import RoleRunResult


@dataclass(frozen=True)
class PreparedComparison:
    job_input: ComparisonJobInput
    role_input: dict[str, Any] | None  # None: the job ends without a model call
    reason_without_model_call: str | None = None


def prepare_comparison(
    connection: sqlite3.Connection, app_config: AppConfig, job_input: ComparisonJobInput
) -> PreparedComparison:
    topic = next((topic for topic in app_config.topics if topic.topic_id == job_input.topic_id), None)
    if topic is None:
        return PreparedComparison(job_input, None, f"topic '{job_input.topic_id}' is no longer configured")
    if compute_question_sha256(topic.question_for_local_model) != job_input.question_sha256:
        return PreparedComparison(
            job_input, None, f"the question of topic '{job_input.topic_id}' changed since the comparison was queued"
        )
    reviews_input = []
    for review_id in (job_input.review_id_a, job_input.review_id_b):
        review_input = _review_as_comparison_input(connection, review_id)
        if review_input is None:
            return PreparedComparison(job_input, None, f"review {review_id} no longer exists")
        reviews_input.append(review_input)
    return PreparedComparison(
        job_input,
        {
            "question": {"topic_id": topic.topic_id, "question": topic.question_for_local_model},
            "reviews": reviews_input,
            "shared_dimensions": list(job_input.shared_dimensions),
        },
    )


def _review_as_comparison_input(connection: sqlite3.Connection, review_id: str) -> dict[str, Any] | None:
    review_row = connection.execute(
        """SELECT review.review_id, review.coverage, review.review_json, paper.title, paper.first_published_date
           FROM reviews AS review JOIN papers AS paper ON paper.paper_id = review.paper_id
           WHERE review.review_id = ?""",
        (review_id,),
    ).fetchone()
    if review_row is None:
        return None
    review_content = json.loads(review_row["review_json"])
    if not isinstance(review_content.get("method"), dict):
        return None  # a provisional step-1 review has no passage-level evidence to compare
    passages = load_review_evidence_passages(connection, review_id)
    return {
        "review_id": review_id,
        "title": review_row["title"],
        "year": (review_row["first_published_date"] or "")[:4] or None,
        "coverage": review_row["coverage"],
        **summarize_review_for_assessment(review_content),
        "passages": [
            {
                "passage_id": passage.passage_id,
                "section": passage.section_title,
                **({"page": passage.page_number} if passage.page_number is not None else {}),
                "text": passage.passage_text,
            }
            for passage in passages.values()
        ],
    }


def write_comparison(
    connection: sqlite3.Connection, prepared: PreparedComparison, role_result: RoleRunResult
) -> tuple[str, bool]:
    """Store the comparison once per key. Call inside the job's finish transaction.
    Returns (comparison id holding this key, was newly written)."""
    job_input = prepared.job_input
    return record_comparison(
        connection,
        topic_id=job_input.topic_id,
        question_sha256=job_input.question_sha256,
        review_id_a=job_input.review_id_a,
        review_id_b=job_input.review_id_b,
        comparison_output={**role_result.validated_output, "shared_dimensions": list(job_input.shared_dimensions)},
        role_name=role_result.role_name,
        prompt_version=role_result.prompt_version,
        model_identity=role_result.model_response.model_identity,
        server_build=role_result.model_response.server_build,
    )
