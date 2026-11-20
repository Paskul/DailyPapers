"""Build the exact packet the reviewer reads, and the key that says "this input was already read".

Reading identity (plan part 2): canonical paper + normalized source-text hash + reading generation, with the edition
whose text was read recorded alongside. The review input key therefore depends ONLY on the text the model reads and
on an explicit operator reread, never on topics, scores, discovery routes, the model, or the prompt version:
  * the text fingerprint covers the abstract (or, for a title-only record, the title), so a changed title alone is
    not a text change;
  * two editions whose texts normalize identically share one reading;
  * an explicit reread carries its own generation id, so it is a new logical input.
Step 2 builds abstract-only packets; part 7 adds full-text packets with passage ids under their own coverage label.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.config import TopicDefinition
from paperwatch.text_normalization import collapse_whitespace

COVERAGE_ABSTRACT_ONLY = "abstract_only"
INITIAL_READING_GENERATION = "initial"  # every automatic reading; operator rereads get a fresh "g_..." id
REVIEW_INPUT_KEY_HASH_CHARACTERS = 32


@dataclass(frozen=True)
class ReviewPacket:
    review_input_key: str
    coverage: str
    source_text_sha256: str
    edition_id: str
    reading_generation: str
    paper_payload: dict[str, Any]  # untrusted text, passed to the model as JSON data


def fingerprint_abstract_reading_text(title: str, abstract: str | None) -> str:
    """sha256 of the normalized text an abstract-only reading depends on (the title only for title-only records)."""
    if abstract and abstract.strip():
        normalized_reading_text = "abstract:" + collapse_whitespace(abstract)
    else:
        normalized_reading_text = "title-only:" + collapse_whitespace(title)
    return hashlib.sha256(normalized_reading_text.encode("utf-8")).hexdigest()


def build_review_input_key(coverage: str, source_text_sha256: str, reading_generation: str) -> str:
    """e.g. 'abstract_only:sha256:<32 hex>' or, for an operator reread, '...:generation:g_<hex>'."""
    review_input_key = f"{coverage}:sha256:{source_text_sha256[:REVIEW_INPUT_KEY_HASH_CHARACTERS]}"
    if reading_generation != INITIAL_READING_GENERATION:
        review_input_key += f":generation:{reading_generation}"
    return review_input_key


def build_abstract_review_packet(edition_row: sqlite3.Row,
                                 reading_generation: str = INITIAL_READING_GENERATION) -> ReviewPacket:
    """The packet for one edition. Edition rows are immutable snapshots, so the same edition gives the same key."""
    paper_payload = {
        "title": edition_row["title"],
        "authors": json.loads(edition_row["authors_json"]),
        "abstract": edition_row["abstract"] or "(no abstract available: title-only record)",
        "first_published_date": edition_row["published_date"],
        "version": edition_row["version_label"],
    }
    source_text_sha256 = fingerprint_abstract_reading_text(edition_row["title"], edition_row["abstract"])
    return ReviewPacket(
        review_input_key=build_review_input_key(COVERAGE_ABSTRACT_ONLY, source_text_sha256, reading_generation),
        coverage=COVERAGE_ABSTRACT_ONLY, source_text_sha256=source_text_sha256,
        edition_id=edition_row["edition_id"], reading_generation=reading_generation, paper_payload=paper_payload)


def build_reviewer_role_input(review_packet: ReviewPacket, topics: tuple[TopicDefinition, ...]) -> dict[str, Any]:
    # Only the topic id and the private question go to the LOCAL model; nothing here leaves the machine.
    return {
        "coverage": review_packet.coverage,
        "paper": review_packet.paper_payload,
        "research_topics": [{"topic_id": topic.topic_id, "question": topic.question_for_local_model}
                            for topic in topics],
    }
