"""Build the exact packet the reviewer reads, and the key that says "this input was already read".

The review input key depends only on the text the model reads (not on topics, scores, or discovery routes), so a
topic change or another discovery route never causes a second reading. Step 1 builds abstract-only packets; part 7
adds full-text packets with passage ids, which get their own key from the normalized source-text hash.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.config import TopicDefinition

COVERAGE_ABSTRACT_ONLY = "abstract_only"


@dataclass(frozen=True)
class ReviewPacket:
    review_input_key: str
    coverage: str
    paper_payload: dict[str, Any]  # untrusted text, passed to the model as JSON data


def build_abstract_review_packet(paper_row: sqlite3.Row) -> ReviewPacket:
    paper_payload = {
        "title": paper_row["title"],
        "authors": json.loads(paper_row["authors_json"]),
        "abstract": paper_row["abstract"] or "(no abstract available: title-only record)",
        "first_published_date": paper_row["first_published_date"],
    }
    normalized_text = json.dumps([_normalize_whitespace(paper_row["title"]),
                                  _normalize_whitespace(paper_row["abstract"] or "")], ensure_ascii=False)
    text_fingerprint = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:32]
    return ReviewPacket(review_input_key=f"{COVERAGE_ABSTRACT_ONLY}:sha256:{text_fingerprint}",
                        coverage=COVERAGE_ABSTRACT_ONLY, paper_payload=paper_payload)


def build_reviewer_role_input(review_packet: ReviewPacket, topics: tuple[TopicDefinition, ...]) -> dict[str, Any]:
    # Only the topic id and the private question go to the LOCAL model; nothing here leaves the machine.
    return {
        "coverage": review_packet.coverage,
        "paper": review_packet.paper_payload,
        "research_topics": [{"topic_id": topic.topic_id, "question": topic.question} for topic in topics],
    }


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
