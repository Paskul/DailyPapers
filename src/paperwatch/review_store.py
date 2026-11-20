"""Persist completed readings and their topic assessments; load assessments for notes and reports.

A review row is the immutable reading (what was read, by which model and prompt, with which coverage). Topic
assessments live in their own table, keyed by review, topic, and the topic-profile hash, so a changed question or
scoring rule can get a new assessment while the reading itself is reused (plan part 2).
Call the write helpers inside the job's lease-checked finish transaction.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.clock import utc_now_timestamp
from paperwatch.config import TopicDefinition
from paperwatch.identifiers import REVIEW_ID_PREFIX, new_internal_id
from paperwatch.review_inputs import ReviewPacket
from paperwatch.roles.base import RoleRunResult

SCORE_SCALE_RATING_0_TO_4 = "rating_0_to_4"  # provisional reviewer; part 7 adds 'priority_0_to_100'


@dataclass(frozen=True)
class TopicAssessment:
    topic_id: str
    topic_score: int
    score_scale: str
    rationale: str
    topic_profile_sha256: str


def insert_review_with_topic_assessments(connection: sqlite3.Connection, *, paper_id: str, job_id: str,
                                         review_packet: ReviewPacket, role_result: RoleRunResult,
                                         topics: tuple[TopicDefinition, ...],
                                         topic_profile_sha256_by_topic_id: dict[str, str] | None = None) -> str:
    """Insert the reading once and its topic assessments. Returns the review id that now holds this input.

    `topic_profile_sha256_by_topic_id` (paperwatch.research_profile) includes each topic's active seeds; without it
    the hash covers the topic id and question only.

    UNIQUE(paper_id, review_input_key) keeps the first accepted review if the same input was somehow read twice.
    """
    new_review_id = new_internal_id(REVIEW_ID_PREFIX)
    completed_at = utc_now_timestamp()
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO reviews (review_id, paper_id, job_id, review_input_key, coverage, review_json,
               role_name, prompt_version, model_identity, server_build, completed_at, edition_id,
               source_text_sha256, reading_generation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_review_id, paper_id, job_id, review_packet.review_input_key, review_packet.coverage,
         json.dumps(role_result.validated_output, ensure_ascii=False), role_result.role_name,
         role_result.prompt_version, role_result.model_response.model_identity,
         role_result.model_response.server_build, completed_at, review_packet.edition_id,
         review_packet.source_text_sha256, review_packet.reading_generation))
    if insert_cursor.rowcount != 1:
        return connection.execute("SELECT review_id FROM reviews WHERE paper_id = ? AND review_input_key = ?",
                                  (paper_id, review_packet.review_input_key)).fetchone()["review_id"]

    topic_by_id = {topic.topic_id: topic for topic in topics}
    for topic_relevance in role_result.validated_output.get("topic_relevance", []):
        topic = topic_by_id[topic_relevance["topic_id"]]  # the reviewer role rejects unknown topic ids
        insert_topic_assessment(connection, review_id=new_review_id, topic=topic,
                                topic_score=topic_relevance["relevance_rating"],
                                score_scale=SCORE_SCALE_RATING_0_TO_4, assessment=topic_relevance,
                                role_name=role_result.role_name, prompt_version=role_result.prompt_version,
                                assessed_at=completed_at,
                                topic_profile_sha256=(topic_profile_sha256_by_topic_id or {}).get(topic.topic_id))
    return new_review_id


def insert_topic_assessment(connection: sqlite3.Connection, *, review_id: str, topic: TopicDefinition,
                            topic_score: int, score_scale: str, assessment: dict[str, Any], role_name: str,
                            prompt_version: str, assessed_at: str, topic_profile_sha256: str | None = None) -> bool:
    """One assessment per (review, topic, topic profile). Returns False if that exact assessment already exists."""
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO review_topic_assessments (review_id, topic_id, topic_profile_sha256, topic_score,
               score_scale, assessment_json, role_name, prompt_version, assessed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (review_id, topic.topic_id, topic_profile_sha256 or topic.compute_profile_sha256(), topic_score, score_scale,
         json.dumps(assessment, ensure_ascii=False), role_name, prompt_version, assessed_at))
    return insert_cursor.rowcount == 1


def load_latest_topic_assessments(connection: sqlite3.Connection, review_id: str) -> list[TopicAssessment]:
    """The newest assessment per topic for one review, ordered by topic id (deterministic rendering)."""
    assessment_rows = connection.execute(
        """SELECT * FROM review_topic_assessments AS assessment
           WHERE assessment.review_id = ?
             AND assessment.assessed_at = (SELECT MAX(newer.assessed_at) FROM review_topic_assessments AS newer
                                           WHERE newer.review_id = assessment.review_id
                                             AND newer.topic_id = assessment.topic_id)
           ORDER BY assessment.topic_id, assessment.topic_profile_sha256""", (review_id,)).fetchall()
    latest_by_topic: dict[str, TopicAssessment] = {}
    for assessment_row in assessment_rows:
        latest_by_topic.setdefault(assessment_row["topic_id"], TopicAssessment(
            topic_id=assessment_row["topic_id"], topic_score=assessment_row["topic_score"],
            score_scale=assessment_row["score_scale"],
            rationale=json.loads(assessment_row["assessment_json"]).get("rationale", ""),
            topic_profile_sha256=assessment_row["topic_profile_sha256"]))
    return list(latest_by_topic.values())
