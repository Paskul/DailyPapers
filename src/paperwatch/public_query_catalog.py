"""The approved public query catalog: the ONLY search terms the collector may send to providers.

Built deterministically from topics.toml. Each entry has a stable id, for example
    team-safety:arxiv:keywords:3f9a1c0b7d     -> arXiv search for the phrase "multi-robot runtime assurance"
    team-safety:openalex:keywords:3f9a1c0b7d  -> OpenAlex search for the same phrase
    team-safety:arxiv:category:cs.RO          -> the arXiv category cs.RO
A provider search is requested by catalog id only (`ProviderGateway.search(query_id, cursor)`). The private question,
private notes, Zotero collection names, and model text therefore can never become an outbound query. The research
role may suggest another search only by naming one of these ids. Changing a phrase changes its id, which the
collector treats as a new search.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from paperwatch.config import TopicDefinition
from paperwatch.errors import SourceAccessDenied
from paperwatch.text_normalization import collapse_whitespace

QUERY_KIND_KEYWORDS = "keywords"
QUERY_KIND_CATEGORY = "category"
KEYWORD_SEARCH_PROVIDERS = ("arxiv", "openalex")
CATEGORY_SEARCH_PROVIDERS = ("arxiv",)
KEYWORD_ID_HASH_CHARACTERS = 10


@dataclass(frozen=True)
class PublicQuery:
    query_id: str
    topic_id: str
    provider_name: str
    query_kind: str  # keywords | category
    public_value: str  # exactly what may be sent: the normalized phrase, or the category name


def build_public_query_catalog(topics: tuple[TopicDefinition, ...]) -> dict[str, PublicQuery]:
    """All approved queries of all topics, keyed by query id (ordered by topic, then as written in topics.toml)."""
    catalog: dict[str, PublicQuery] = {}
    for topic in topics:
        for raw_phrase in topic.public_queries:
            normalized_phrase = collapse_whitespace(raw_phrase)
            phrase_hash = hashlib.sha256(normalized_phrase.lower().encode("utf-8")).hexdigest()
            for provider_name in KEYWORD_SEARCH_PROVIDERS:
                query_id = (
                    f"{topic.topic_id}:{provider_name}:{QUERY_KIND_KEYWORDS}:{phrase_hash[:KEYWORD_ID_HASH_CHARACTERS]}"
                )
                catalog[query_id] = PublicQuery(
                    query_id, topic.topic_id, provider_name, QUERY_KIND_KEYWORDS, normalized_phrase
                )
        for arxiv_category in topic.arxiv_categories:
            for provider_name in CATEGORY_SEARCH_PROVIDERS:
                query_id = f"{topic.topic_id}:{provider_name}:{QUERY_KIND_CATEGORY}:{arxiv_category}"
                catalog[query_id] = PublicQuery(
                    query_id, topic.topic_id, provider_name, QUERY_KIND_CATEGORY, arxiv_category
                )
    return catalog


def find_public_query(catalog: dict[str, PublicQuery], query_id: str) -> PublicQuery:
    """The approved entry for `query_id`, or SourceAccessDenied. Never builds a query from the id text itself."""
    if not isinstance(query_id, str) or query_id not in catalog:
        raise SourceAccessDenied(
            f"'{query_id}' is not an approved public query id; only ids from the query "
            "catalog (built from topics.toml public terms) may be searched"
        )
    return catalog[query_id]
