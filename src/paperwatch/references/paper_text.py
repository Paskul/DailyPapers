"""The passages of a paper's extracted full text, rebuilt from the source cache (shared by collector and worker).

Used to find bibliography identifiers (collector) and citation contexts (worker). Only text the collector already
extracted is used; nothing is downloaded. Returns None when the paper has no usable extracted text, or when the cached
file is gone or altered (the caller then works without the bibliography).
"""

from __future__ import annotations

import sqlite3

from paperwatch.config import AppConfig
from paperwatch.reading.passages import Passage, SectionKind, segment_extracted_text
from paperwatch.reading.review_packet import ExtractedTextUnavailable, load_verified_extracted_text
from paperwatch.readings import select_reading_input
from paperwatch.review_inputs import TEXT_KIND_EXTRACTED_TEXT


def load_extracted_text_passages(
    connection: sqlite3.Connection, app_config: AppConfig, paper_id: str
) -> tuple[Passage, ...] | None:
    reading_input = select_reading_input(connection, paper_id)
    if reading_input is None or reading_input.text_kind != TEXT_KIND_EXTRACTED_TEXT:
        return None
    try:
        extracted_text = load_verified_extracted_text(app_config, reading_input)
    except ExtractedTextUnavailable:
        return None
    return segment_extracted_text(
        extracted_text,
        passage_max_characters=app_config.reading.passage_max_characters,
        passage_min_characters=app_config.reading.passage_min_characters,
    )


def bibliography_passages(passages: tuple[Passage, ...]) -> list[Passage]:
    return [passage for passage in passages if passage.section_kind == SectionKind.REFERENCES]


def body_passages(passages: tuple[Passage, ...]) -> list[Passage]:
    """Passages that can cite something: everything but the bibliography and acknowledgments."""
    return [
        passage
        for passage in passages
        if passage.section_kind not in (SectionKind.REFERENCES, SectionKind.ACKNOWLEDGMENTS)
    ]
