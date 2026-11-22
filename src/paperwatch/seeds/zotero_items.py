"""Parse one page of Zotero Web API items (format=json, include=data) into SeedItems.

Only bibliographic metadata is read: title, creators, abstract, date, DOI, URL, archive id, and identifier lines of the
Extra field. Notes, attachments, annotations, and trashed items are dropped here even though the request already asks
Zotero to leave them out (defence in depth: private notes must never become seeds). Collection names are never read.
A malformed item is skipped and counted; a malformed page is rejected as a whole.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from paperwatch.errors import SourceResponseRejected
from paperwatch.seeds.seed_items import (
    MAXIMUM_SEED_ABSTRACT_CHARACTERS,
    MAXIMUM_SEED_TITLE_CHARACTERS,
    SEED_SOURCE_ZOTERO,
    SeedItem,
    bound_author_names,
    clean_seed_text,
    find_arxiv_id,
    find_doi,
    find_publication_year,
)

ZOTERO_ITEM_KEY_PATTERN = re.compile(r"^[A-Z0-9]{8}$")
NON_BIBLIOGRAPHIC_ZOTERO_ITEM_TYPES = frozenset({"note", "attachment", "annotation"})
# Item types without a "title" field keep their title under another name.
ALTERNATIVE_TITLE_FIELDS = ("caseName", "nameOfAct", "subject")
AUTHOR_CREATOR_TYPES = frozenset({"author", "contributor", "programmer", "inventor", "presenter"})

SKIP_REASON_NOT_BIBLIOGRAPHIC = "note, attachment, or annotation"
SKIP_REASON_TRASHED = "in the trash"
SKIP_REASON_MALFORMED = "malformed item"
SKIP_REASON_NO_TITLE = "no title"


@dataclass
class ParsedZoteroPage:
    seed_items: list[SeedItem]
    raw_item_count: int  # every entry on the page, including skipped ones (used for paging)
    skipped_item_counts: Counter[str] = field(default_factory=Counter)  # reason -> count


def parse_zotero_items_page(body_bytes: bytes) -> ParsedZoteroPage:
    try:
        raw_items = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as decode_error:
        raise SourceResponseRejected(f"zotero: items page is not valid JSON: {decode_error}") from None
    if not isinstance(raw_items, list):
        raise SourceResponseRejected("zotero: items page is not a JSON array")

    parsed_page = ParsedZoteroPage(seed_items=[], raw_item_count=len(raw_items))
    for raw_item in raw_items:
        seed_item, skip_reason = _convert_zotero_item(raw_item)
        if seed_item is None:
            parsed_page.skipped_item_counts[skip_reason] += 1
        else:
            parsed_page.seed_items.append(seed_item)
    return parsed_page


def _convert_zotero_item(raw_item: Any) -> tuple[SeedItem | None, str]:
    if not isinstance(raw_item, dict) or not isinstance(raw_item.get("data"), dict):
        return None, SKIP_REASON_MALFORMED
    item_data: dict[str, Any] = raw_item["data"]
    item_key = raw_item.get("key", item_data.get("key"))
    if not isinstance(item_key, str) or not ZOTERO_ITEM_KEY_PATTERN.match(item_key):
        return None, SKIP_REASON_MALFORMED
    item_type = item_data.get("itemType")
    if not isinstance(item_type, str):
        return None, SKIP_REASON_MALFORMED
    if item_type in NON_BIBLIOGRAPHIC_ZOTERO_ITEM_TYPES:
        return None, SKIP_REASON_NOT_BIBLIOGRAPHIC
    if item_data.get("deleted"):
        return None, SKIP_REASON_TRASHED

    title = clean_seed_text(item_data.get("title"), MAXIMUM_SEED_TITLE_CHARACTERS)
    for alternative_title_field in ALTERNATIVE_TITLE_FIELDS:
        title = title or clean_seed_text(item_data.get(alternative_title_field), MAXIMUM_SEED_TITLE_CHARACTERS)
    if title is None:
        return None, SKIP_REASON_NO_TITLE

    item_version = raw_item.get("version", item_data.get("version"))
    raw_metadata = raw_item.get("meta")
    item_metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    # The Extra field is free text; only its structured "DOI: ..." / "arXiv: ..." lines are used, nothing else.
    extra_field_text = item_data.get("extra")
    return SeedItem(
        seed_source=SEED_SOURCE_ZOTERO,
        seed_item_key=item_key,
        title=title,
        authors=_zotero_author_names(item_data.get("creators")),
        abstract=clean_seed_text(item_data.get("abstractNote"), MAXIMUM_SEED_ABSTRACT_CHARACTERS),
        publication_year=find_publication_year(item_metadata.get("parsedDate"), item_data.get("date")),
        item_type=item_type,
        item_version=item_version if isinstance(item_version, int) and not isinstance(item_version, bool) else None,
        doi=find_doi(item_data.get("DOI"), item_data.get("url"), extra_field_text),
        arxiv_id=find_arxiv_id(
            item_data.get("archiveID"), item_data.get("url"), extra_field_text, item_data.get("DOI")
        ),
    ), ""


def _zotero_author_names(raw_creators: Any) -> tuple[str, ...]:
    """'First Last' (or the single-field name) of author-like creators; all creators if none are author-like."""
    if not isinstance(raw_creators, list):
        return ()
    named_creators: list[tuple[str, str]] = []
    for raw_creator in raw_creators:
        if not isinstance(raw_creator, dict):
            continue
        name_parts = [clean_seed_text(raw_creator.get(part), 200) for part in ("firstName", "lastName")]
        full_name = " ".join(part for part in name_parts if part) or clean_seed_text(raw_creator.get("name"), 200)
        if full_name:
            named_creators.append((str(raw_creator.get("creatorType", "")), full_name))
    author_names = [name for creator_type, name in named_creators if creator_type in AUTHOR_CREATOR_TYPES]
    return bound_author_names(author_names or [name for _, name in named_creators])
