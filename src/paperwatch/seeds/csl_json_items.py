"""Parse a CSL JSON export (a JSON array of CSL items, e.g. Zotero's "Export Collection... -> CSL JSON").

Read fields: id, type, title, author/editor, abstract, issued, DOI, URL, number / archive_location (arXiv ids).
The free-text `note` field (Zotero puts its Extra field there) is used only for structured "DOI: ..." and
"arXiv: ..." lines and is otherwise ignored, like private notes in the Zotero API route.
A malformed item is skipped and counted; a file that is not a JSON array is rejected as a whole.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from paperwatch.errors import SourceInputError
from paperwatch.seeds.seed_items import (MAXIMUM_SEED_ABSTRACT_CHARACTERS, MAXIMUM_SEED_TITLE_CHARACTERS,
                                         SEED_SOURCE_CSL_JSON, SeedItem, bound_author_names, clean_seed_text,
                                         find_arxiv_id, find_doi, find_publication_year)

MAXIMUM_CSL_ITEM_ID_CHARACTERS = 180  # keeps "<file key>#<id>" within the catalog's opaque-identifier limit
MAXIMUM_CSL_ITEMS_PER_FILE = 50_000

SKIP_REASON_MALFORMED = "malformed item"
SKIP_REASON_NO_ID = "missing or unusable id"
SKIP_REASON_NO_TITLE = "no title"
SKIP_REASON_DUPLICATE_ID = "duplicate id"


@dataclass
class ParsedCslJsonExport:
    seed_items: list[SeedItem]
    raw_item_count: int
    skipped_item_counts: Counter[str] = field(default_factory=Counter)


def parse_csl_json_export(file_bytes: bytes, csl_file_key: str) -> ParsedCslJsonExport:
    try:
        raw_items = json.loads(file_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as decode_error:
        raise SourceInputError(f"CSL JSON file is not valid JSON: {decode_error}") from None
    if not isinstance(raw_items, list):
        raise SourceInputError("CSL JSON file must be a JSON array of items")
    if len(raw_items) > MAXIMUM_CSL_ITEMS_PER_FILE:
        raise SourceInputError(f"CSL JSON file has {len(raw_items)} items; the limit is {MAXIMUM_CSL_ITEMS_PER_FILE}")

    parsed_export = ParsedCslJsonExport(seed_items=[], raw_item_count=len(raw_items))
    seen_item_keys: set[str] = set()
    for raw_item in raw_items:
        seed_item, skip_reason = _convert_csl_item(raw_item, csl_file_key)
        if seed_item is not None and seed_item.seed_item_key in seen_item_keys:
            seed_item, skip_reason = None, SKIP_REASON_DUPLICATE_ID
        if seed_item is None:
            parsed_export.skipped_item_counts[skip_reason] += 1
            continue
        seen_item_keys.add(seed_item.seed_item_key)
        parsed_export.seed_items.append(seed_item)
    return parsed_export


def _convert_csl_item(raw_item: Any, csl_file_key: str) -> tuple[SeedItem | None, str]:
    if not isinstance(raw_item, dict):
        return None, SKIP_REASON_MALFORMED
    raw_item_id = raw_item.get("id")
    if isinstance(raw_item_id, bool) or not isinstance(raw_item_id, (str, int)):
        return None, SKIP_REASON_NO_ID
    item_id = str(raw_item_id).strip()
    if not item_id or len(item_id) > MAXIMUM_CSL_ITEM_ID_CHARACTERS or \
            any(not character.isprintable() for character in item_id):
        return None, SKIP_REASON_NO_ID
    title = clean_seed_text(raw_item.get("title"), MAXIMUM_SEED_TITLE_CHARACTERS)
    if title is None:
        return None, SKIP_REASON_NO_TITLE

    note_field_text = raw_item.get("note")  # only its "DOI: ..." / "arXiv: ..." lines are used
    return SeedItem(
        seed_source=SEED_SOURCE_CSL_JSON,
        seed_item_key=f"{csl_file_key}#{item_id}",
        title=title,
        authors=_csl_author_names(raw_item.get("author")) or _csl_author_names(raw_item.get("editor")),
        abstract=clean_seed_text(raw_item.get("abstract"), MAXIMUM_SEED_ABSTRACT_CHARACTERS),
        publication_year=_csl_publication_year(raw_item.get("issued")),
        item_type=str(raw_item.get("type") or "unknown")[:64],
        item_version=None,
        doi=find_doi(raw_item.get("DOI"), raw_item.get("URL"), note_field_text),
        arxiv_id=find_arxiv_id(raw_item.get("number"), raw_item.get("archive_location"), raw_item.get("URL"),
                               note_field_text, raw_item.get("DOI")),
    ), ""


def _csl_author_names(raw_names: Any) -> tuple[str, ...]:
    if not isinstance(raw_names, list):
        return ()
    author_names: list[str] = []
    for raw_name in raw_names:
        if not isinstance(raw_name, dict):
            continue
        name_parts = [clean_seed_text(raw_name.get(part), 200) for part in ("given", "non-dropping-particle",
                                                                             "family")]
        full_name = " ".join(part for part in name_parts if part) or clean_seed_text(raw_name.get("literal"), 200)
        if full_name:
            author_names.append(full_name)
    return bound_author_names(author_names)


def _csl_publication_year(raw_issued: Any) -> int | None:
    """CSL dates: {"date-parts": [[2023, 5, 12]]}, or {"raw": "2023-05"}, or {"literal": "Spring 2019"}."""
    if not isinstance(raw_issued, dict):
        return find_publication_year(raw_issued)
    date_parts = raw_issued.get("date-parts")
    first_year = None
    if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
        first_year = date_parts[0][0]
        if isinstance(first_year, str) and first_year.isdigit():
            first_year = int(first_year)
    return find_publication_year(first_year, raw_issued.get("raw"), raw_issued.get("literal"))
