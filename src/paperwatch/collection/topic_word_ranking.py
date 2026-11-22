"""Which new records a capped arXiv category route admits first: the ones that share the most words with its topics.

A category feed (or its missed-release recovery search) that brings more new papers than its fair share of the
category-feed limit (collection/candidate_budget.py) cannot admit them all. Instead of the provider's order, it admits
the records whose title and abstract share the most distinct words with the questions and public phrases of the
topics that watch the category. This is a cheap, local pre-order only: screening still ranks every admitted paper
properly. The private question is used here on this machine only; nothing of it leaves the process.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from paperwatch.config import TopicDefinition
from paperwatch.sources.types import CollectedPaperRecord

WORD_PATTERN = re.compile(r"[a-z][a-z0-9]+")
MINIMUM_TOPIC_WORD_LENGTH = 3
COMMON_ENGLISH_WORDS = frozenset(
    [
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "what",
        "which",
        "when",
        "where",
        "while",
        "whose",
        "who",
        "how",
        "can",
        "could",
        "would",
        "should",
        "does",
        "did",
        "are",
        "was",
        "were",
        "been",
        "being",
        "has",
        "have",
        "had",
        "not",
        "but",
        "nor",
        "yet",
        "any",
        "all",
        "each",
        "every",
        "some",
        "such",
        "than",
        "then",
        "them",
        "they",
        "their",
        "there",
        "these",
        "those",
        "into",
        "onto",
        "over",
        "under",
        "about",
        "above",
        "after",
        "before",
        "between",
        "through",
        "during",
        "without",
        "within",
        "give",
        "gives",
        "given",
        "make",
        "makes",
        "made",
        "use",
        "uses",
        "used",
        "using",
        "our",
        "your",
        "its",
        "also",
        "more",
        "most",
        "less",
        "least",
        "very",
        "much",
        "many",
        "one",
        "two",
        "new",
        "via",
        "per",
        "based",
        "towards",
        "toward",
        "stop",
    ]
)


def comparable_words(text: str) -> set[str]:
    """Lowercase words of 3+ letters without common English words; a plural 's' is dropped ('agents' = 'agent')."""
    words = set()
    for word in WORD_PATTERN.findall(text.lower()):
        if len(word) < MINIMUM_TOPIC_WORD_LENGTH or word in COMMON_ENGLISH_WORDS:
            continue
        words.add(word[:-1] if word.endswith("s") and len(word) > 3 and not word.endswith("ss") else word)
    return words


def topic_words_for_category(topics: Iterable[TopicDefinition], arxiv_category: str) -> set[str]:
    """Words of the questions and public phrases of the topics that watch this category."""
    topic_words: set[str] = set()
    for topic in topics:
        if arxiv_category in topic.arxiv_categories:
            topic_words |= comparable_words(" ".join([topic.question, *topic.public_queries]))
    return topic_words


def rank_records_by_topic_words(
    collected_records: list[CollectedPaperRecord], topic_words: set[str]
) -> list[CollectedPaperRecord]:
    """Most shared topic words (title and abstract) first; ties keep the provider's order. Known papers cost no
    budget, so their position does not matter."""
    if not topic_words:
        return list(collected_records)
    return sorted(
        collected_records,
        key=lambda record: -len(topic_words & comparable_words(f"{record.title} {record.abstract or ''}")),
    )
