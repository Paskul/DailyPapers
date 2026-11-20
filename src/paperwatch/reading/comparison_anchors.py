"""Comparison anchors for a topic assessment (plan part 7): "for each matched topic, attach the closest seed records
and available saved reviews". Their ids (seed ids, review ids) are distinct from the candidate's passage ids.

    seed anchors    the topic's active seeds closest to this paper: first the closest seed ids screening stored
                    (candidate_topic_scores), else the nearest by cached sentence embeddings; nothing is embedded
                    here, so the worker needs no embedding model
    review anchors  earlier completed reviews of OTHER papers whose assessment for this topic rated question_fit at
                    least 2 ("addresses part of the question"; legacy 0-4 relevance ratings count the same way),
                    nearest by cached embeddings, else by their topic score
Owner decision (2026-11-19): seeds and earlier reviews are both comparable saved papers, so relative_value is rated
whenever a topic has at least one anchor of either kind, and omitted ("comparison unavailable") otherwise.
Anchor texts are untrusted data and are bounded in length.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from paperwatch.config import AppConfig
from paperwatch.reading.reading_priority import question_fit_from_stored_assessment
from paperwatch.research_profile import TopicInputMode, TopicResearchProfile
from paperwatch.screening.embedding_cache import hash_model_input_text, load_cached_vectors
from paperwatch.screening.text_embedders import configured_embedder_identity
from paperwatch.screening.topic_screening_profiles import build_screening_document_text
from paperwatch.seeds.seed_store import StoredSeed

MINIMUM_QUESTION_FIT_OF_COMPARABLE_REVIEW = 2
MAXIMUM_ANCHOR_RESULTS = 3
MAXIMUM_ANCHOR_TEXT_CHARACTERS = 400
LEGACY_RELEVANCE_SCALE = "rating_0_to_4"


@dataclass(frozen=True)
class TopicAnchors:
    topic_id: str
    question: str
    seed_anchors: list[dict[str, Any]] = field(default_factory=list)
    review_anchors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def comparison_available(self) -> bool:
        return bool(self.seed_anchors or self.review_anchors)

    def as_model_input(self) -> dict[str, Any]:
        return {"topic_id": self.topic_id, "question": self.question,
                "comparison_available": self.comparison_available,
                "seed_anchors": self.seed_anchors, "review_anchors": self.review_anchors}


def assessable_topic_profiles(profiles: tuple[TopicResearchProfile, ...]) -> tuple[TopicResearchProfile, ...]:
    """Every topic is assessed (owner decision), except one with no relevance input at all."""
    return tuple(profile for profile in profiles if profile.input_mode != TopicInputMode.EMPTY)


def build_topic_anchors(connection: sqlite3.Connection, app_config: AppConfig, *, paper_id: str,
                        paper_title: str, paper_abstract: str | None,
                        profiles: tuple[TopicResearchProfile, ...]) -> list[TopicAnchors]:
    vector_lookup = CachedVectorLookup(connection, configured_embedder_identity(app_config))
    paper_vector = vector_lookup.vector_for_document(paper_title, paper_abstract)
    reading_settings = app_config.reading
    return [TopicAnchors(
        topic_id=profile.topic_id, question=profile.topic.question_for_local_model,
        seed_anchors=[_seed_anchor(seed, reading_settings.anchor_abstract_characters) for seed in
                      _closest_seeds(connection, vector_lookup, paper_vector, paper_id, profile,
                                     reading_settings.closest_seeds_per_topic)],
        review_anchors=_comparable_review_anchors(connection, vector_lookup, paper_vector, paper_id,
                                                  profile.topic_id, reading_settings.comparison_reviews_per_topic))
        for profile in profiles]


class CachedVectorLookup:
    """Read-only access to the screening embeddings already in SQLite (never computes a new one)."""

    def __init__(self, connection: sqlite3.Connection, embedder_identity: str) -> None:
        self._connection = connection
        self._embedder_identity = embedder_identity

    def vector_for_document(self, title: str, abstract: str | None) -> np.ndarray | None:
        return self.vectors_for_documents([(title, abstract)])[0]

    def vectors_for_documents(self, documents: list[tuple[str, str | None]]) -> list[np.ndarray | None]:
        text_hashes = [hash_model_input_text(build_screening_document_text(title, abstract))
                       for title, abstract in documents]
        cached_vectors = load_cached_vectors(self._connection, self._embedder_identity, set(text_hashes))
        return [cached_vectors.get(text_hash) for text_hash in text_hashes]


def _closest_seeds(connection: sqlite3.Connection, vector_lookup: CachedVectorLookup,
                   paper_vector: np.ndarray | None, paper_id: str, profile: TopicResearchProfile,
                   seed_limit: int) -> list[StoredSeed]:
    seed_by_id = {seed.seed_id: seed for seed in profile.active_seeds}
    if not seed_by_id:
        return []
    score_row = connection.execute(
        "SELECT closest_seed_ids_json FROM candidate_topic_scores WHERE paper_id = ? AND topic_id = ?",
        (paper_id, profile.topic_id)).fetchone()
    if score_row is not None:
        screened_closest = [seed_by_id[seed_id] for seed_id in json.loads(score_row["closest_seed_ids_json"])
                            if seed_id in seed_by_id]
        if screened_closest:
            return screened_closest[:seed_limit]
    if paper_vector is None:
        return []
    seeds = list(seed_by_id.values())
    seed_vectors = vector_lookup.vectors_for_documents([(seed.title, seed.abstract) for seed in seeds])
    scored_seeds = [(float(paper_vector @ seed_vector), seed.seed_id, seed)
                    for seed, seed_vector in zip(seeds, seed_vectors) if seed_vector is not None]
    scored_seeds.sort(key=lambda scored: (-scored[0], scored[1]))
    return [seed for _cosine, _seed_id, seed in scored_seeds[:seed_limit]]


def _seed_anchor(seed: StoredSeed, abstract_characters: int) -> dict[str, Any]:
    return {"seed_id": seed.seed_id, "title": seed.title, "year": seed.publication_year,
            "abstract": (seed.abstract or "(title only: no abstract in the library)")[:abstract_characters]}


def _comparable_review_anchors(connection: sqlite3.Connection, vector_lookup: CachedVectorLookup,
                               paper_vector: np.ndarray | None, paper_id: str, topic_id: str,
                               review_limit: int) -> list[dict[str, Any]]:
    ranked_rows = rank_comparable_reviews(connection, vector_lookup, paper_vector, paper_id, topic_id)
    return [_review_anchor(row) for row in ranked_rows[:review_limit]]


def rank_comparable_reviews(connection: sqlite3.Connection, vector_lookup: CachedVectorLookup,
                            paper_vector: np.ndarray | None, paper_id: str, topic_id: str) -> list[sqlite3.Row]:
    """The latest reviews of OTHER papers whose newest assessment for `topic_id` rated question_fit >= 2, nearest to
    `paper_vector` first (by cached embeddings), then by topic score. Also used by part-8 comparisons."""
    candidate_rows = connection.execute(
        """SELECT review.review_id, review.review_json, paper.title, paper.first_published_date,
                  edition.title AS edition_title, edition.abstract AS edition_abstract,
                  assessment.topic_score, assessment.score_scale, assessment.assessment_json
           FROM reviews AS review
           JOIN papers AS paper ON paper.paper_id = review.paper_id
           LEFT JOIN paper_editions AS edition ON edition.edition_id = review.edition_id
           JOIN review_topic_assessments AS assessment ON assessment.review_id = review.review_id
                AND assessment.topic_id = ?
           WHERE review.paper_id != ?
             AND review.paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
             AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                     WHERE newest.paper_id = review.paper_id
                                     ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)
             AND assessment.assessed_at = (SELECT MAX(newer.assessed_at) FROM review_topic_assessments AS newer
                                           WHERE newer.review_id = assessment.review_id
                                             AND newer.topic_id = assessment.topic_id)""",
        (topic_id, paper_id)).fetchall()
    comparable_rows = [row for row in candidate_rows
                       if _question_fit_of(row) >= MINIMUM_QUESTION_FIT_OF_COMPARABLE_REVIEW]
    comparable_rows = list({row["review_id"]: row for row in comparable_rows}.values())  # one per review
    review_vectors = vector_lookup.vectors_for_documents(
        [(row["edition_title"] or row["title"], row["edition_abstract"]) for row in comparable_rows])

    def ranking_key(indexed_row: tuple[int, sqlite3.Row]) -> tuple[float, float, str]:
        row_index, row = indexed_row
        review_vector = review_vectors[row_index]
        cosine = float(paper_vector @ review_vector) if paper_vector is not None and review_vector is not None \
            else -2.0  # rows without vectors rank after every row with one
        return (-cosine, -_normalized_topic_score(row), row["review_id"])

    return [row for _index, row in sorted(enumerate(comparable_rows), key=ranking_key)]


def _question_fit_of(assessment_row: sqlite3.Row) -> int:
    return question_fit_from_stored_assessment(assessment_row["score_scale"], assessment_row["topic_score"],
                                               json.loads(assessment_row["assessment_json"]))


def _normalized_topic_score(assessment_row: sqlite3.Row) -> float:
    """Legacy 0-4 relevance and 0-100 priority on one scale, for ordering only."""
    if assessment_row["score_scale"] == LEGACY_RELEVANCE_SCALE:
        return 25.0 * assessment_row["topic_score"]
    return float(assessment_row["topic_score"])


def _review_anchor(review_row: sqlite3.Row) -> dict[str, Any]:
    review_content = json.loads(review_row["review_json"])
    method = review_content.get("method", "")
    if isinstance(method, dict):  # part-7 review; a legacy review stores plain strings
        method = method.get("statement", "")
    results = [finding["claim"] for finding in review_content.get("results", [])] or \
        list(review_content.get("authors_main_claims", []))
    year = (review_row["first_published_date"] or "")[:4] or None
    return {"review_id": review_row["review_id"], "title": review_row["title"], "year": year,
            "summary": str(review_content.get("one_sentence_summary", ""))[:MAXIMUM_ANCHOR_TEXT_CHARACTERS],
            "method": str(method)[:MAXIMUM_ANCHOR_TEXT_CHARACTERS],
            "key_results": [str(claim)[:MAXIMUM_ANCHOR_TEXT_CHARACTERS]
                            for claim in results[:MAXIMUM_ANCHOR_RESULTS]]}
