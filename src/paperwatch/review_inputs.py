"""The identity of a reading's input: the key that says "this input was already read".

Reading identity: canonical paper + normalized source-text hash + reading generation, with the edition
whose text was read recorded alongside. The review input key therefore depends ONLY on the text the model reads and
on an explicit operator reread, never on topics, scores, discovery routes, reading limits, the model, or the prompt
version:
  * an abstract reading's fingerprint covers the abstract (or, for a title-only record, the title), so a changed
    title alone is not a text change;
  * a full-text reading's fingerprint covers the whitespace-normalized text the sandboxed extractor produced, so
    newly available full text is a new input ("newly available full text can create a new review");
  * two editions whose texts normalize identically share one reading;
  * an explicit reread carries its own generation id, so it is a new logical input.
This module is deliberately low-level (schema migrations import it). The packet the reviewer actually reads, with
passage ids and the reading plan, is built from this identity by paperwatch.reading.review_packet.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from paperwatch.text_normalization import collapse_whitespace

TEXT_KIND_ABSTRACT = "abstract"
TEXT_KIND_EXTRACTED_TEXT = "extracted_text"
COVERAGE_ABSTRACT_ONLY = "abstract_only"
# Key prefixes. The abstract prefix is the coverage label it always had, so keys stored by earlier versions stay valid.
REVIEW_INPUT_KEY_PREFIX_BY_TEXT_KIND = {
    TEXT_KIND_ABSTRACT: COVERAGE_ABSTRACT_ONLY,
    TEXT_KIND_EXTRACTED_TEXT: TEXT_KIND_EXTRACTED_TEXT,
}
INITIAL_READING_GENERATION = "initial"  # every automatic reading; operator rereads get a fresh "g_..." id
REVIEW_INPUT_KEY_HASH_CHARACTERS = 32


@dataclass(frozen=True)
class ReadingInputIdentity:
    """Which text of which edition a reading uses. Cheap to build (no file is read): the collector computes it when
    it queues a reading, and the worker recomputes it at dispatch to detect a superseded job."""

    edition_id: str
    text_kind: str  # TEXT_KIND_ABSTRACT or TEXT_KIND_EXTRACTED_TEXT
    source_text_sha256: str  # normalized-text fingerprint
    reading_generation: str
    extracted_text_file_name: str | None = None  # inside state/cache/sources/<edition_id>/ (extracted text only)

    @property
    def review_input_key(self) -> str:
        return build_review_input_key(
            REVIEW_INPUT_KEY_PREFIX_BY_TEXT_KIND[self.text_kind], self.source_text_sha256, self.reading_generation
        )


def fingerprint_abstract_reading_text(title: str, abstract: str | None) -> str:
    """sha256 of the normalized text an abstract-only reading depends on (the title only for title-only records)."""
    if abstract and abstract.strip():
        normalized_reading_text = "abstract:" + collapse_whitespace(abstract)
    else:
        normalized_reading_text = "title-only:" + collapse_whitespace(title)
    return hashlib.sha256(normalized_reading_text.encode("utf-8")).hexdigest()


def fingerprint_extracted_reading_text(extracted_text: str) -> str:
    """sha256 of the extractor's text with every whitespace run (page breaks included) collapsed to one space."""
    return hashlib.sha256(("extracted-text:" + collapse_whitespace(extracted_text)).encode("utf-8")).hexdigest()


def build_review_input_key(key_prefix: str, source_text_sha256: str, reading_generation: str) -> str:
    """e.g. 'abstract_only:sha256:<32 hex>', 'extracted_text:sha256:<32 hex>', or for an operator reread
    '...:generation:g_<hex>'."""
    review_input_key = f"{key_prefix}:sha256:{source_text_sha256[:REVIEW_INPUT_KEY_HASH_CHARACTERS]}"
    if reading_generation != INITIAL_READING_GENERATION:
        review_input_key += f":generation:{reading_generation}"
    return review_input_key


def build_abstract_reading_identity(
    edition_row: sqlite3.Row, reading_generation: str = INITIAL_READING_GENERATION
) -> ReadingInputIdentity:
    """The abstract (or title-only) input of one edition. Edition rows are immutable text snapshots."""
    return ReadingInputIdentity(
        edition_id=edition_row["edition_id"],
        text_kind=TEXT_KIND_ABSTRACT,
        source_text_sha256=fingerprint_abstract_reading_text(edition_row["title"], edition_row["abstract"]),
        reading_generation=reading_generation,
    )
