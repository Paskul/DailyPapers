"""Build the edition-specific packet the reviewer reads (plan part 7), from a ReadingInputIdentity.

    abstract input        passages A1.. (or T1 for a title-only record), one reading section, coverage abstract_only
    extracted-text input  the cached text file is read and verified against the stored normalized fingerprint, cut
                          into passages P1.., and planned into bounded reading sections (full_text or partial_text)

The worker never downloads anything: the collector stored the text in state/cache/sources/<edition id>/ (plan part
6, "schedule source acquisition during collection"). If that file is missing or no longer matches its fingerprint,
ExtractedTextUnavailable is raised; the worker then marks the text failed and queues the abstract reading instead.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperwatch.catalog import load_edition_row
from paperwatch.config import AppConfig
from paperwatch.errors import SourceInputError
from paperwatch.reading.parser_recovery import ParserRecoveryAssessment, assess_parser_recovery
from paperwatch.reading.passages import Passage, build_abstract_passages, segment_extracted_text
from paperwatch.reading.reading_plan import ReadingPlan, plan_abstract_reading, plan_full_text_reading
from paperwatch.review_inputs import (TEXT_KIND_ABSTRACT, ReadingInputIdentity, fingerprint_abstract_reading_text,
                                      fingerprint_extracted_reading_text)
from paperwatch.sandboxed_extractor import BYTES_PER_MEGABYTE
from paperwatch.source_cache import cached_edition_directory

EXTRACTED_TEXT_FILE_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.txt$")


class ExtractedTextUnavailable(SourceInputError):
    """The cached extracted text is missing, unreadable, or no longer matches the fingerprint it was queued under."""


@dataclass(frozen=True)
class ReviewPacket:
    paper_id: str
    reading_input: ReadingInputIdentity
    orientation: dict[str, Any]  # title, authors, abstract, first_published_date, version (untrusted)
    passages: tuple[Passage, ...]
    reading_plan: ReadingPlan
    parser_recovery: ParserRecoveryAssessment | None  # None for an abstract reading

    @property
    def review_input_key(self) -> str:
        return self.reading_input.review_input_key

    @property
    def edition_id(self) -> str:
        return self.reading_input.edition_id

    @property
    def coverage(self) -> str:
        return self.reading_plan.coverage

    @property
    def passage_by_id(self) -> dict[str, Passage]:
        return {passage.passage_id: passage for passage in self.passages}

    def citable_passage_texts(self) -> dict[str, str]:
        """{passage id: text} of every passage the plan reads: what any answer may cite."""
        read_passage_ids = self.reading_plan.read_passage_ids
        return {passage.passage_id: passage.text for passage in self.passages
                if passage.passage_id in read_passage_ids}


def build_review_packet(connection: sqlite3.Connection, app_config: AppConfig, paper_id: str,
                        reading_input: ReadingInputIdentity) -> ReviewPacket:
    edition_row = load_edition_row(connection, reading_input.edition_id)
    orientation = {
        "title": edition_row["title"],
        "authors": json.loads(edition_row["authors_json"]),
        "abstract": edition_row["abstract"] or "(no abstract available: title-only record)",
        "first_published_date": edition_row["published_date"],
        "version": edition_row["version_label"],
    }
    if reading_input.text_kind == TEXT_KIND_ABSTRACT:
        if fingerprint_abstract_reading_text(edition_row["title"], edition_row["abstract"]) != \
                reading_input.source_text_sha256:
            raise SourceInputError(f"edition {reading_input.edition_id} no longer has the text this job was queued for")
        passages = build_abstract_passages(edition_row["title"], edition_row["abstract"])
        return ReviewPacket(paper_id, reading_input, orientation, passages, plan_abstract_reading(passages), None)

    extracted_text = load_verified_extracted_text(app_config, reading_input)
    reading_settings = app_config.reading
    passages = segment_extracted_text(extracted_text, passage_max_characters=reading_settings.passage_max_characters,
                                      passage_min_characters=reading_settings.passage_min_characters)
    reading_plan = plan_full_text_reading(passages, section_max_characters=reading_settings.section_max_characters,
                                          max_sections_per_paper=reading_settings.max_sections_per_paper)
    return ReviewPacket(paper_id, reading_input, orientation, passages, reading_plan,
                        assess_parser_recovery(extracted_text))


def load_verified_extracted_text(app_config: AppConfig, reading_input: ReadingInputIdentity) -> str:
    file_name = reading_input.extracted_text_file_name or ""
    if not EXTRACTED_TEXT_FILE_NAME_PATTERN.match(file_name):
        raise ExtractedTextUnavailable(f"edition {reading_input.edition_id}: '{file_name}' is not a cached text name")
    text_path: Path = cached_edition_directory(app_config.source_cache_dir, reading_input.edition_id) / file_name
    if text_path.is_symlink() or not text_path.is_file():
        raise ExtractedTextUnavailable(f"extracted text of edition {reading_input.edition_id} is missing from the "
                                       "source cache")
    maximum_bytes = app_config.extractor.max_output_megabytes * BYTES_PER_MEGABYTE
    if text_path.stat().st_size > maximum_bytes:
        raise ExtractedTextUnavailable(f"extracted text of edition {reading_input.edition_id} exceeds "
                                       "extractor.max_output_megabytes")
    try:
        extracted_text = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError as read_error:
        raise ExtractedTextUnavailable(f"cannot read the extracted text of edition {reading_input.edition_id}: "
                                       f"{read_error}") from None
    if fingerprint_extracted_reading_text(extracted_text) != reading_input.source_text_sha256:
        raise ExtractedTextUnavailable(f"extracted text of edition {reading_input.edition_id} no longer matches its "
                                       "stored fingerprint")
    return extracted_text
