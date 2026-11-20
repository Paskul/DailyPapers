"""Persist completed readings and their topic assessments; load them for notes and reports.

A review row is the immutable reading (what was read, by which model and prompt, with which coverage, and the
reading record: sections read and not read, why coverage is partial, parser recovery). The passages it cites are
copied into review_evidence_passages, so the evidence survives when the source cache is pruned (plan part 10).
Topic assessments live in their own table, keyed by review, topic, and the topic-profile hash, so a changed question
or seed set can get a new assessment while the reading itself is reused (plan part 2).
Call the write helpers inside the job's lease-checked finish transaction.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from paperwatch.clock import utc_now_timestamp
from paperwatch.identifiers import REVIEW_ID_PREFIX, new_internal_id
from paperwatch.reading.reading_pipeline import CompletedReading
from paperwatch.reading.reading_priority import SCORE_SCALE_PRIORITY_0_TO_100, question_fit_from_stored_assessment
from paperwatch.roles.reviewer import REVIEWER_ROLE_NAME
from paperwatch.roles.topic_assessor import TOPIC_ASSESSOR_ROLE_NAME

SCORE_SCALE_RATING_0_TO_4 = "rating_0_to_4"  # the provisional step-1 reviewer's relevance rating (legacy rows)


@dataclass(frozen=True)
class TopicAssessment:
    topic_id: str
    topic_score: int
    score_scale: str
    rationale: str
    topic_profile_sha256: str
    assessment_record: dict[str, Any] = field(default_factory=dict)  # ratings, explanations, anchors, scoring

    @property
    def priority_score_out_of_100(self) -> float:
        """Legacy 0-4 relevance and the 0-100 reading priority on one scale (for ordering only)."""
        return 25.0 * self.topic_score if self.score_scale == SCORE_SCALE_RATING_0_TO_4 else float(self.topic_score)

    @property
    def question_fit(self) -> int:
        """The applied 0-4 question_fit rating (a legacy relevance rating counts as question fit)."""
        return question_fit_from_stored_assessment(self.score_scale, self.topic_score, self.assessment_record)


@dataclass(frozen=True)
class EvidencePassage:
    passage_id: str
    page_number: int | None
    section_title: str
    passage_text: str

    @property
    def location_label(self) -> str:
        return (f"p. {self.page_number}, " if self.page_number is not None else "") + self.section_title


def insert_completed_reading(connection: sqlite3.Connection, *, paper_id: str, job_id: str,
                             completed_reading: CompletedReading) -> str:
    """Insert the reading once, its cited passages, and its topic assessments. Returns the review id that now holds
    this input. UNIQUE(paper_id, review_input_key) keeps the first accepted review if the same input was somehow
    read twice; the second reading's assessments are then not stored either."""
    packet = completed_reading.packet
    new_review_id = new_internal_id(REVIEW_ID_PREFIX)
    completed_at = utc_now_timestamp()
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO reviews (review_id, paper_id, job_id, review_input_key, coverage, review_json,
               role_name, prompt_version, model_identity, server_build, completed_at, edition_id,
               source_text_sha256, reading_generation, reading_record_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_review_id, paper_id, job_id, packet.review_input_key, packet.coverage,
         json.dumps(completed_reading.review_content, ensure_ascii=False), REVIEWER_ROLE_NAME,
         completed_reading.reviewer_prompt_version, completed_reading.model_identity,
         completed_reading.server_build, completed_at, packet.edition_id,
         packet.reading_input.source_text_sha256, packet.reading_input.reading_generation,
         json.dumps(completed_reading.reading_record, ensure_ascii=False)))
    if insert_cursor.rowcount != 1:
        return connection.execute("SELECT review_id FROM reviews WHERE paper_id = ? AND review_input_key = ?",
                                  (paper_id, packet.review_input_key)).fetchone()["review_id"]

    passage_by_id = packet.passage_by_id
    connection.executemany(
        """INSERT OR IGNORE INTO review_evidence_passages (review_id, passage_id, page_number, section_title,
               passage_text) VALUES (?, ?, ?, ?, ?)""",
        [(new_review_id, passage_id, passage_by_id[passage_id].page_number, passage_by_id[passage_id].section_title,
          passage_by_id[passage_id].text)
         for passage_id in completed_reading.cited_passage_ids if passage_id in passage_by_id])
    for completed_assessment in completed_reading.topic_assessments:
        insert_topic_assessment(
            connection, review_id=new_review_id, topic_id=completed_assessment.topic_id,
            topic_score=completed_assessment.scored.reading_priority_score,
            score_scale=SCORE_SCALE_PRIORITY_0_TO_100, assessment=completed_assessment.to_record(),
            role_name=TOPIC_ASSESSOR_ROLE_NAME, prompt_version=completed_reading.topic_assessor_prompt_version,
            assessed_at=completed_at, topic_profile_sha256=completed_assessment.topic_profile_sha256)
    return new_review_id


def insert_topic_assessment(connection: sqlite3.Connection, *, review_id: str, topic_id: str, topic_score: int,
                            score_scale: str, assessment: dict[str, Any], role_name: str, prompt_version: str,
                            assessed_at: str, topic_profile_sha256: str) -> bool:
    """One assessment per (review, topic, topic profile). Returns False if that exact assessment already exists."""
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO review_topic_assessments (review_id, topic_id, topic_profile_sha256, topic_score,
               score_scale, assessment_json, role_name, prompt_version, assessed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (review_id, topic_id, topic_profile_sha256, topic_score, score_scale,
         json.dumps(assessment, ensure_ascii=False), role_name, prompt_version, assessed_at))
    return insert_cursor.rowcount == 1


def load_latest_topic_assessments(connection: sqlite3.Connection, review_id: str, *,
                                  assessed_no_later_than: str | None = None) -> list[TopicAssessment]:
    """The newest assessment per topic for one review, ordered by topic id (deterministic rendering).

    `assessed_no_later_than` (a stored UTC timestamp) ignores assessments made after a report's cutoff (plan
    part 9), so a frozen report selection never depends on later re-assessments."""
    latest_allowed_time = assessed_no_later_than or "9999"
    assessment_rows = connection.execute(
        """SELECT * FROM review_topic_assessments AS assessment
           WHERE assessment.review_id = ? AND assessment.assessed_at <= ?
             AND assessment.assessed_at = (SELECT MAX(newer.assessed_at) FROM review_topic_assessments AS newer
                                           WHERE newer.review_id = assessment.review_id
                                             AND newer.topic_id = assessment.topic_id
                                             AND newer.assessed_at <= ?)
           ORDER BY assessment.topic_id, assessment.topic_profile_sha256""",
        (review_id, latest_allowed_time, latest_allowed_time)).fetchall()
    latest_by_topic: dict[str, TopicAssessment] = {}
    for assessment_row in assessment_rows:
        assessment_record = json.loads(assessment_row["assessment_json"])
        latest_by_topic.setdefault(assessment_row["topic_id"], TopicAssessment(
            topic_id=assessment_row["topic_id"], topic_score=assessment_row["topic_score"],
            score_scale=assessment_row["score_scale"], rationale=assessment_record.get("rationale", ""),
            topic_profile_sha256=assessment_row["topic_profile_sha256"], assessment_record=assessment_record))
    return list(latest_by_topic.values())


def load_review_evidence_passages(connection: sqlite3.Connection, review_id: str) -> dict[str, EvidencePassage]:
    return {row["passage_id"]: EvidencePassage(row["passage_id"], row["page_number"], row["section_title"],
                                               row["passage_text"])
            for row in connection.execute(
                "SELECT * FROM review_evidence_passages WHERE review_id = ? ORDER BY passage_id", (review_id,))}


def load_latest_review_id_of_paper(connection: sqlite3.Connection, paper_id: str) -> str | None:
    """The paper's newest completed review (the one later parts build on)."""
    review_row = connection.execute(
        """SELECT review_id FROM reviews WHERE paper_id = ? ORDER BY completed_at DESC, review_id DESC LIMIT 1""",
        (paper_id,)).fetchone()
    return review_row["review_id"] if review_row else None


def best_question_fit_and_priority(topic_assessments: list[TopicAssessment]) -> tuple[int, float]:
    """(highest question_fit over the topics, highest reading priority 0-100); (0, 0.0) without assessments."""
    if not topic_assessments:
        return 0, 0.0
    return (max(assessment.question_fit for assessment in topic_assessments),
            max(assessment.priority_score_out_of_100 for assessment in topic_assessments))
