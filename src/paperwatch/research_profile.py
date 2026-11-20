"""The research profile: for every topic, what defines its relevance right now (plan part 4).

The question is the unit of relevance. A topic can be defined by
    keywords            a private question and/or approved public phrases,
    seeds               the active seed papers of its Zotero collections and CSL JSON files, or
    keywords and seeds  both.
A seeds-only topic has no written question: it searches its public arXiv categories and ranks the results against
its seeds locally (part 6). If a seed source is empty or offline, the topic keeps working from what remains; a
topic left with no input at all is reported by `status` with what to add.

The profile hash covers the topic id, the question, and the active seeds' metadata. It is stored with each topic
assessment, so a changed question or seed set makes old assessments visibly stale while the reading is reused.
Search terms are not part of the profile: only the approved public query catalog (public_query_catalog.py) may leave
the machine, and nothing here adds to it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from enum import StrEnum

from paperwatch.config import AppConfig, TopicDefinition
from paperwatch.public_query_catalog import build_public_query_catalog
from paperwatch.seeds.seed_items import MATCHING_TEXT_TITLE_ONLY
from paperwatch.seeds.seed_store import (SeedSourceReference, StoredSeed, load_active_seeds,
                                         seed_sources_of_all_topics, seed_sources_of_topic)


class TopicInputMode(StrEnum):
    KEYWORDS = "keywords"
    SEEDS = "seeds"
    KEYWORDS_AND_SEEDS = "keywords and seeds"
    EMPTY = "no input"


@dataclass(frozen=True)
class TopicResearchProfile:
    topic: TopicDefinition
    seed_sources: tuple[SeedSourceReference, ...]
    active_seeds: tuple[StoredSeed, ...]
    approved_public_query_ids: tuple[str, ...]  # catalog ids this topic may search (arXiv, OpenAlex)

    @property
    def topic_id(self) -> str:
        return self.topic.topic_id

    @property
    def input_mode(self) -> TopicInputMode:
        has_seeds = bool(self.active_seeds)
        if self.topic.has_keyword_input:
            return TopicInputMode.KEYWORDS_AND_SEEDS if has_seeds else TopicInputMode.KEYWORDS
        return TopicInputMode.SEEDS if has_seeds else TopicInputMode.EMPTY

    @property
    def title_only_seed_count(self) -> int:
        return sum(1 for seed in self.active_seeds if seed.matching_text_kind == MATCHING_TEXT_TITLE_ONLY)

    @property
    def can_search_providers(self) -> bool:
        return bool(self.approved_public_query_ids)

    def compute_profile_sha256(self) -> str:
        return self.topic.compute_profile_sha256(tuple(seed.profile_fingerprint for seed in self.active_seeds))

    def explain_missing_inputs(self) -> list[str]:
        """What the operator should add or fix, in plain words (empty when the topic is fully usable)."""
        explanations: list[str] = []
        if self.input_mode == TopicInputMode.EMPTY:
            if self.topic.has_configured_seed_sources:
                explanations.append("no relevance input yet: its seed sources hold no seeds (see the seed source "
                                     "lines). Add a private `question` or `public_queries`, or fix the seed source.")
            else:
                explanations.append("no relevance input: add a private `question`, `public_queries`, or seed "
                                    "sources (`zotero_collections` and/or `csl_json_files`).")
        if not self.can_search_providers:
            explanations.append("nothing can be searched for it: add `arxiv_categories` (enough for a seeds-only "
                                "topic) or `public_queries`.")
        if self.title_only_seed_count:
            explanations.append(f"{self.title_only_seed_count} seed(s) have no abstract and are matched by title "
                                "only.")
        return explanations


def build_research_profiles(connection: sqlite3.Connection, app_config: AppConfig) -> tuple[TopicResearchProfile, ...]:
    """One profile per configured topic, in topics.toml order. Read-only; safe in any command."""
    query_catalog = build_public_query_catalog(app_config.topics)
    profiles = []
    for topic in app_config.topics:
        topic_seed_sources = seed_sources_of_topic(topic)
        profiles.append(TopicResearchProfile(
            topic=topic, seed_sources=topic_seed_sources,
            active_seeds=tuple(load_active_seeds(connection, topic_seed_sources)),
            approved_public_query_ids=tuple(query_id for query_id, public_query in query_catalog.items()
                                            if public_query.topic_id == topic.topic_id)))
    return tuple(profiles)


def compute_topic_profile_hashes(connection: sqlite3.Connection, app_config: AppConfig) -> dict[str, str]:
    """{topic id: profile hash} for storing with topic assessments."""
    return {profile.topic_id: profile.compute_profile_sha256()
            for profile in build_research_profiles(connection, app_config)}


def load_active_seed_library_paper_ids(connection: sqlite3.Connection, app_config: AppConfig) -> frozenset[str]:
    """Catalog papers in the user's active seed library across all topics (kept out of review and recommendations)."""
    return frozenset(seed.paper_id for seed in load_active_seeds(connection,
                                                                 seed_sources_of_all_topics(app_config.topics)))


def explain_empty_research_profile(profiles: tuple[TopicResearchProfile, ...]) -> str | None:
    """The overall explanation when neither keywords nor seeds define any topic (plan part 4); else None."""
    if any(profile.input_mode != TopicInputMode.EMPTY for profile in profiles):
        return None
    return ("Nothing defines relevance yet. In topics.toml add, for at least one [[topics]] entry, a private "
            "`question` and `public_queries` (keyword mode), or seed papers via `zotero_collections` (with the read-"
            "only key in the collector credentials file) or `csl_json_files` (a CSL JSON export), plus "
            "`arxiv_categories` to search.")
