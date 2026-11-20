"""What each topic is screened against, built from the research profile of part 4.

Per topic:
    question text   the private question; for a keyword topic without one, its approved phrases joined; a seeds-only
                    topic has none (the question signal is omitted, plan part 6). Stays on this machine.
    seeds           the topic's active seeds, embedded from title + abstract (title only for a title-only seed)
    phrases         the topic's approved public phrases, for the lexical signal
    fingerprint     a hash of everything above plus the scoring settings and the embedder identity. It is stored
                    with each score; a different fingerprint makes the score stale, so unpromoted candidates are
                    rescored locally whenever the profile changes.
A topic with no relevance input at all (input mode "no input") is not screened and gets no review slots.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass

from paperwatch.config import AppConfig
from paperwatch.research_profile import TopicInputMode, TopicResearchProfile, build_research_profiles
from paperwatch.seeds.seed_store import StoredSeed
from paperwatch.text_normalization import collapse_whitespace

SCREENING_FORMULA_VERSION = "screening-v1"  # bump when the formula or text preparation changes


def build_screening_document_text(title: str, abstract: str | None) -> str:
    """The text a paper or seed is embedded and phrase-matched on: "title. abstract", or the title alone."""
    title_text = collapse_whitespace(title or "")
    abstract_text = collapse_whitespace(abstract or "")
    return f"{title_text}. {abstract_text}" if abstract_text else title_text


@dataclass(frozen=True)
class ScreeningSeed:
    seed_id: str
    document_text: str
    title_only: bool


@dataclass(frozen=True)
class TopicScreeningProfile:
    topic_id: str
    question_text: str | None
    seeds: tuple[ScreeningSeed, ...]
    approved_phrases: tuple[str, ...]
    screening_fingerprint: str

    @property
    def seed_ids(self) -> tuple[str, ...]:
        return tuple(seed.seed_id for seed in self.seeds)


def build_topic_screening_profiles(connection: sqlite3.Connection, app_config: AppConfig,
                                   embedder_identity: str) -> tuple[TopicScreeningProfile, ...]:
    """One profile per topic that has any relevance input, in topics.toml order."""
    return tuple(_build_one_profile(research_profile, app_config, embedder_identity)
                 for research_profile in build_research_profiles(connection, app_config)
                 if research_profile.input_mode != TopicInputMode.EMPTY)


def _build_one_profile(research_profile: TopicResearchProfile, app_config: AppConfig,
                       embedder_identity: str) -> TopicScreeningProfile:
    topic = research_profile.topic
    if topic.has_written_question:
        question_text = collapse_whitespace(topic.question)
    elif topic.public_queries:
        question_text = "; ".join(topic.public_queries)
    else:
        question_text = None
    seeds = tuple(_screening_seed(stored_seed) for stored_seed in research_profile.active_seeds)
    screening_settings = app_config.screening
    fingerprint_fields = {
        "formula": SCREENING_FORMULA_VERSION,
        "embedder": embedder_identity,
        "topic_profile_sha256": research_profile.compute_profile_sha256(),
        "question_text": question_text,
        "seed_texts": sorted((seed.seed_id, seed.document_text) for seed in seeds),
        "phrases": list(topic.public_queries),
        "weights": [screening_settings.question_similarity_weight, screening_settings.seed_similarity_weight,
                    screening_settings.phrase_match_weight],
        "closest_seed_count": screening_settings.closest_seed_count,
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_fields, ensure_ascii=False, sort_keys=True)
                                 .encode("utf-8")).hexdigest()
    return TopicScreeningProfile(topic_id=topic.topic_id, question_text=question_text, seeds=seeds,
                                 approved_phrases=topic.public_queries, screening_fingerprint=fingerprint)


def _screening_seed(stored_seed: StoredSeed) -> ScreeningSeed:
    return ScreeningSeed(seed_id=stored_seed.seed_id,
                         document_text=build_screening_document_text(stored_seed.title, stored_seed.abstract),
                         title_only=not (stored_seed.abstract and stored_seed.abstract.strip()))


def compute_screening_input_sha256(document_text: str, screening_fingerprint: str) -> str:
    """What one stored score depends on: the paper's screened text and the topic's screening fingerprint."""
    return hashlib.sha256(f"{screening_fingerprint}\n{document_text}".encode("utf-8")).hexdigest()
