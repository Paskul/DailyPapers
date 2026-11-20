"""Score every candidate for every topic, cheaply, on the CPU (plan part 6).

A CANDIDATE is a catalog paper that screening has not promoted and that has no job yet, excluding the user's active
seeds (comparison anchors, never recommendations) and withdrawn or retracted works. It is scored on the text of the
edition a reading would use (readings.PREFERRED_EDITION_ORDER_SQL), so the score and the later reading agree.

Each (candidate, topic) score is stored with the hash of its inputs (paper text + topic screening fingerprint). A run
recomputes only missing or stale scores, so an unchanged catalog costs one SQL pass and no embedding at all, while a
changed question, seed set, phrase list, weight, or embedding model rescores the unpromoted candidates locally. A
promoted paper keeps the scores it was promoted with.
What is stored are the RAW signals (and their raw weighted mean, `unnormalized_score`, for inspection). Promotion
ranks each signal within the topic's current pool first (signal_normalization.py); that score is never cached.

The slow part (embedding) happens outside any write transaction; the scores are then stored in one transaction.
If the embedder is unavailable, nothing is scored on partial signals: stale or new candidates wait, and candidates
whose stored scores are still current remain eligible for promotion.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from paperwatch.clock import utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.db import write_transaction
from paperwatch.errors import ScreeningUnavailable
from paperwatch.readings import PREFERRED_EDITION_ORDER_SQL
from paperwatch.screening.embedding_cache import embed_texts_with_cache
from paperwatch.screening.phrase_matching import find_matched_phrases, split_into_match_words
from paperwatch.screening.screening_score import (ScreeningSignals, SignalWeights, closest_seed_similarity,
                                                  combine_screening_signals, normalize_cosine_similarity)
from paperwatch.screening.text_embedders import TextEmbedder
from paperwatch.screening.topic_screening_profiles import (TopicScreeningProfile, build_screening_document_text,
                                                           compute_screening_input_sha256)


@dataclass(frozen=True)
class ScreeningCandidate:
    paper_id: str
    edition_id: str
    document_text: str
    title_only: bool
    first_published_date: str | None


@dataclass(frozen=True)
class TopicScore:
    """One stored (candidate, topic) score: the raw signals and their raw weighted mean (candidate_topic_scores)."""

    topic_id: str
    unnormalized_score: float
    signals: ScreeningSignals


@dataclass
class CandidateScreeningOutcome:
    candidates: dict[str, ScreeningCandidate] = field(default_factory=dict)
    # Raw signals of the up-to-date (candidate, topic) pairs only: {paper id: {topic id: signals}}.
    current_signals_by_paper: dict[str, dict[str, ScreeningSignals]] = field(default_factory=dict)
    pairs_scored_now: int = 0
    pairs_already_current: int = 0
    candidates_waiting_for_embedder: int = 0
    title_only_candidates: int = 0
    embedder_failure: ScreeningUnavailable | None = None

    def describe(self, topic_count: int) -> str:
        description = (f"Screening: {len(self.candidates)} candidate(s) across {topic_count} topic(s); "
                       f"{self.pairs_scored_now} topic score(s) computed now, {self.pairs_already_current} already "
                       f"current")
        if self.title_only_candidates:
            description += f"; {self.title_only_candidates} candidate(s) scored on the title only (no abstract)"
        if self.candidates_waiting_for_embedder:
            description += (f"; {self.candidates_waiting_for_embedder} candidate(s) WAIT unscored until the "
                            "embedding model is available")
        return description


def load_screening_candidates(connection: sqlite3.Connection,
                              active_seed_paper_ids: frozenset[str]) -> dict[str, ScreeningCandidate]:
    candidate_rows = connection.execute(
        f"""WITH preferred_editions AS (
                SELECT edition_id, paper_id, title, abstract,
                       ROW_NUMBER() OVER (PARTITION BY paper_id ORDER BY {PREFERRED_EDITION_ORDER_SQL})
                           AS preference_rank
                FROM paper_editions)
            SELECT paper.paper_id, paper.first_published_date, edition.edition_id, edition.title, edition.abstract
            FROM papers AS paper
            JOIN preferred_editions AS edition ON edition.paper_id = paper.paper_id AND edition.preference_rank = 1
            WHERE paper.paper_id NOT IN (SELECT paper_id FROM paper_promotions)
              AND paper.paper_id NOT IN (SELECT paper_id FROM jobs WHERE paper_id IS NOT NULL)
              AND paper.paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
            ORDER BY paper.paper_id""").fetchall()
    return {candidate_row["paper_id"]: ScreeningCandidate(
                paper_id=candidate_row["paper_id"], edition_id=candidate_row["edition_id"],
                document_text=build_screening_document_text(candidate_row["title"], candidate_row["abstract"]),
                title_only=not (candidate_row["abstract"] and candidate_row["abstract"].strip()),
                first_published_date=candidate_row["first_published_date"])
            for candidate_row in candidate_rows if candidate_row["paper_id"] not in active_seed_paper_ids}


def screen_candidates(connection: sqlite3.Connection, app_config: AppConfig,
                      topic_profiles: tuple[TopicScreeningProfile, ...], *,
                      active_seed_paper_ids: frozenset[str],
                      build_embedder: Callable[[], TextEmbedder]) -> CandidateScreeningOutcome:
    """Bring every candidate's topic scores up to date. `build_embedder` is called only if something must be scored."""
    outcome = CandidateScreeningOutcome(candidates=load_screening_candidates(connection, active_seed_paper_ids))
    outcome.title_only_candidates = sum(candidate.title_only for candidate in outcome.candidates.values())
    stored_rows = {(score_row["paper_id"], score_row["topic_id"]): score_row for score_row in connection.execute(
        """SELECT paper_id, topic_id, screening_input_sha256, question_similarity, seed_similarity,
                  phrase_match_fraction FROM candidate_topic_scores""")}

    stale_pairs: list[tuple[ScreeningCandidate, TopicScreeningProfile, str]] = []
    for candidate in outcome.candidates.values():
        for topic_profile in topic_profiles:
            input_sha256 = compute_screening_input_sha256(candidate.document_text, topic_profile.screening_fingerprint)
            stored_row = stored_rows.get((candidate.paper_id, topic_profile.topic_id))
            if stored_row is not None and stored_row["screening_input_sha256"] == input_sha256:
                outcome.current_signals_by_paper.setdefault(candidate.paper_id, {})[topic_profile.topic_id] = \
                    ScreeningSignals(stored_row["question_similarity"], stored_row["seed_similarity"],
                                     stored_row["phrase_match_fraction"])
                outcome.pairs_already_current += 1
            else:
                stale_pairs.append((candidate, topic_profile, input_sha256))
    if not stale_pairs:
        return outcome

    try:
        embedder = build_embedder()
        computed_scores = _compute_topic_scores(connection, embedder, stale_pairs, app_config)
    except ScreeningUnavailable as embedder_failure:
        outcome.embedder_failure = embedder_failure
        outcome.candidates_waiting_for_embedder = len({candidate.paper_id for candidate, _p, _h in stale_pairs})
        return outcome

    _store_topic_scores(connection, computed_scores, topic_ids={profile.topic_id for profile in topic_profiles})
    for candidate, topic_score, _input_sha256 in computed_scores:
        if topic_score is not None:
            outcome.current_signals_by_paper.setdefault(candidate.paper_id, {})[topic_score.topic_id] = \
                topic_score.signals
            outcome.pairs_scored_now += 1
    return outcome


def _compute_topic_scores(connection: sqlite3.Connection, embedder: TextEmbedder,
                          stale_pairs: list[tuple[ScreeningCandidate, TopicScreeningProfile, str]],
                          app_config: AppConfig) -> list[tuple[ScreeningCandidate, TopicScore | None, str]]:
    """Embed the candidates (cached) and score each stale pair with simple NumPy matrix products."""
    screening_settings = app_config.screening
    signal_weights = SignalWeights.from_settings(screening_settings)
    candidates_to_embed = list({candidate.paper_id: candidate for candidate, _p, _h in stale_pairs}.values())
    row_by_paper_id = {candidate.paper_id: row_index for row_index, candidate in enumerate(candidates_to_embed)}
    candidate_matrix = embed_texts_with_cache(connection, embedder,
                                              [candidate.document_text for candidate in candidates_to_embed])
    words_by_paper_id = {candidate.paper_id: split_into_match_words(candidate.document_text)
                         for candidate in candidates_to_embed}

    computed: list[tuple[ScreeningCandidate, TopicScore | None, str]] = []
    profiles_by_topic_id = {profile.topic_id: profile for _c, profile, _h in stale_pairs}
    for topic_id, topic_profile in profiles_by_topic_id.items():
        topic_pairs = [(candidate, input_sha256) for candidate, profile, input_sha256 in stale_pairs
                       if profile.topic_id == topic_id]
        topic_rows = candidate_matrix[[row_by_paper_id[candidate.paper_id] for candidate, _h in topic_pairs]]
        question_cosines = (topic_rows @ embed_texts_with_cache(
            connection, embedder, [embedder.prepare_query_text(topic_profile.question_text)])[0]
                            if topic_profile.question_text else None)
        seed_cosines = (topic_rows @ embed_texts_with_cache(
            connection, embedder, [seed.document_text for seed in topic_profile.seeds]).T
                        if topic_profile.seeds else np.zeros((len(topic_pairs), 0), dtype=np.float32))
        for pair_index, (candidate, input_sha256) in enumerate(topic_pairs):
            seed_similarity, closest_seed_ids = closest_seed_similarity(
                seed_cosines[pair_index], topic_profile.seed_ids, screening_settings.closest_seed_count)
            matched_phrases = find_matched_phrases(topic_profile.approved_phrases,
                                                   words_by_paper_id[candidate.paper_id])
            signals = ScreeningSignals(
                question_similarity=(normalize_cosine_similarity(question_cosines[pair_index])
                                     if question_cosines is not None else None),
                seed_similarity=seed_similarity,
                phrase_match_fraction=(len(matched_phrases) / len(topic_profile.approved_phrases)
                                       if topic_profile.approved_phrases else None),
                closest_seed_ids=closest_seed_ids, matched_phrases=matched_phrases)
            unnormalized_score = combine_screening_signals(signals, signal_weights)
            computed.append((candidate, None if unnormalized_score is None else
                             TopicScore(topic_id, unnormalized_score, signals), input_sha256))
    return computed


def _store_topic_scores(connection: sqlite3.Connection,
                        computed_scores: list[tuple[ScreeningCandidate, TopicScore | None, str]],
                        topic_ids: set[str]) -> None:
    scored_at = utc_now_timestamp()
    with write_transaction(connection):
        connection.executemany(
            """INSERT OR REPLACE INTO candidate_topic_scores (paper_id, topic_id, edition_id, unnormalized_score,
                   question_similarity, seed_similarity, phrase_match_fraction, closest_seed_ids_json,
                   matched_phrases_json, title_only, screening_input_sha256, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(candidate.paper_id, topic_score.topic_id, candidate.edition_id, topic_score.unnormalized_score,
              topic_score.signals.question_similarity, topic_score.signals.seed_similarity,
              topic_score.signals.phrase_match_fraction, json.dumps(list(topic_score.signals.closest_seed_ids)),
              json.dumps(list(topic_score.signals.matched_phrases)), int(candidate.title_only), input_sha256,
              scored_at)
             for candidate, topic_score, input_sha256 in computed_scores if topic_score is not None])
        # Scores of topics that were removed from topics.toml no longer mean anything for unpromoted candidates.
        placeholders = ", ".join("?" * len(topic_ids)) or "''"
        connection.execute(
            f"""DELETE FROM candidate_topic_scores WHERE topic_id NOT IN ({placeholders})
                AND paper_id NOT IN (SELECT paper_id FROM paper_promotions)""", tuple(topic_ids))


def load_topic_score_components(connection: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    """Every stored topic score of one paper, highest raw score first (for notes and part 7's matched seeds)."""
    return connection.execute(
        """SELECT * FROM candidate_topic_scores WHERE paper_id = ?
           ORDER BY unnormalized_score DESC, topic_id""", (paper_id,)).fetchall()
