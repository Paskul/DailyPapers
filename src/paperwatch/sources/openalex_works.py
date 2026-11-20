"""Parse OpenAlex works JSON (api.openalex.org/works) into CollectedPaperRecords (plan part 5).

Fields used (the request selects only these, see provider_requests.OPENALEX_SELECTED_FIELDS):
    id, display_name/title, authorships[].author.display_name, abstract_inverted_index, publication_date,
    publication_year, ids.doi, locations[] (landing_page_url, id/doi), is_retracted, type
Identifier links: OpenAlex groups the versions of one work (preprint, journal article) into one record and lists them
as locations. The work's DOI and its arXiv location (an arxiv.org/abs/... landing page, or arXiv's DataCite DOI
10.48550/arxiv.<id>) are clear identifier links stated by the provider, so the catalog may merge on them. Other
locations' DOIs (e.g. another preprint server) are not used for merging.
A single work (GET /works/<id>, plan part 8) also carries `referenced_works`: the OpenAlex ids of the works it cites.
Dates: publication_date is a day; without it, publication_year is kept as a year. `is_retracted` is recorded as a
retraction notice. Notices, errata, and similar non-papers are skipped.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from paperwatch.errors import SourceResponseRejected
from paperwatch.publication_dates import normalize_partial_date
from paperwatch.scholarly_identifiers import ARXIV_DATACITE_DOI_PREFIX, ARXIV_SCHEME, DOI_SCHEME, OPENALEX_SCHEME
from paperwatch.sources.record_cleaning import (MAXIMUM_ABSTRACT_CHARACTERS, MAXIMUM_TITLE_CHARACTERS,
                                                clean_author_names, clean_provider_text, keep_valid_identifiers)
from paperwatch.sources.types import (WITHDRAWAL_NOTICE_RETRACTED, CollectedPaperRecord, ParsedSearchPage,
                                      RawIdentifier)

SKIPPED_WORK_TYPES = frozenset({"paratext", "erratum", "retraction", "peer-review", "editorial", "letter"})
ARXIV_LANDING_PAGE_PATTERN = re.compile(r"^https?://(?:export\.)?arxiv\.org/abs/\S+$", re.IGNORECASE)
MAXIMUM_ABSTRACT_WORD_POSITIONS = 10_000
OPENALEX_WORK_ID_PATTERN = re.compile(r"^(?:https?://openalex\.org/)?W\d+$", re.IGNORECASE)


def parse_works_page(body_bytes: bytes, discovery_route: str) -> ParsedSearchPage:
    try:
        response_document = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
        raise SourceResponseRejected(f"OpenAlex: response is not valid JSON ({decode_error})") from None
    if not isinstance(response_document, dict) or not isinstance(response_document.get("results"), list):
        raise SourceResponseRejected("OpenAlex: response has no `results` list")
    meta = response_document.get("meta") if isinstance(response_document.get("meta"), dict) else {}

    records: list[CollectedPaperRecord] = []
    skipped_counts: Counter[str] = Counter()
    for raw_work in response_document["results"]:
        parsed_record, skip_reason = convert_openalex_work(raw_work, discovery_route)
        if parsed_record is None:
            skipped_counts[skip_reason] += 1
        else:
            records.append(parsed_record)
    next_cursor = meta.get("next_cursor") if isinstance(meta.get("next_cursor"), str) else None
    if not response_document["results"]:
        next_cursor = None  # OpenAlex keeps returning a cursor on the last, empty page
    total_results = meta.get("count") if isinstance(meta.get("count"), int) else None
    return ParsedSearchPage(records, len(response_document["results"]), next_cursor, total_results, skipped_counts)


MAXIMUM_REFERENCED_WORKS_READ = 2000


@dataclass(frozen=True)
class ParsedSingleWork:
    """One work from GET /works/<id>: its catalog record (None if skipped) and the works it cites."""

    record: CollectedPaperRecord | None
    skip_reason: str
    referenced_work_ids: tuple[str, ...]  # normalized OpenAlex ids, e.g. "W2741809807", in the provider's order


def parse_single_work(body_bytes: bytes, discovery_route: str) -> ParsedSingleWork:
    try:
        raw_work = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
        raise SourceResponseRejected(f"OpenAlex: response is not valid JSON ({decode_error})") from None
    if not isinstance(raw_work, dict):
        raise SourceResponseRejected("OpenAlex: a single work must be a JSON object")
    record, skip_reason = convert_openalex_work(raw_work, discovery_route)
    referenced_work_ids: list[str] = []
    raw_references = raw_work.get("referenced_works")
    for raw_reference in (raw_references if isinstance(raw_references, list) else [])[:MAXIMUM_REFERENCED_WORKS_READ]:
        if isinstance(raw_reference, str) and OPENALEX_WORK_ID_PATTERN.match(raw_reference):
            referenced_work_ids.append("W" + raw_reference.rsplit("/", 1)[-1][1:])
    return ParsedSingleWork(record, skip_reason, tuple(dict.fromkeys(referenced_work_ids)))


def convert_openalex_work(raw_work: Any, discovery_route: str) -> tuple[CollectedPaperRecord | None, str]:
    if not isinstance(raw_work, dict):
        return None, "not an object"
    work_id = raw_work.get("id")
    if not isinstance(work_id, str) or not OPENALEX_WORK_ID_PATTERN.match(work_id):
        return None, "invalid OpenAlex id"
    if raw_work.get("type") in SKIPPED_WORK_TYPES:
        return None, f"type {raw_work['type']}"
    title = clean_provider_text(raw_work.get("display_name") or raw_work.get("title"), MAXIMUM_TITLE_CHARACTERS)
    if title is None:
        return None, "no title"
    author_names = [(authorship.get("author") or {}).get("display_name")
                    for authorship in _list_of_objects(raw_work.get("authorships"))]
    first_published_date = normalize_partial_date(raw_work.get("publication_date"))
    if first_published_date is None and isinstance(raw_work.get("publication_year"), int):
        first_published_date = normalize_partial_date(f"{raw_work['publication_year']:04d}")
    is_retracted = raw_work.get("is_retracted") is True
    return CollectedPaperRecord(
        provider=OPENALEX_SCHEME, provider_paper_id=work_id, title=title, authors=clean_author_names(author_names),
        abstract=rebuild_abstract_from_inverted_index(raw_work.get("abstract_inverted_index")),
        first_published_date=first_published_date, latest_revision_date=None, source_version=None,
        discovery_route=discovery_route, linked_identifiers=keep_valid_identifiers(_stated_identifier_links(raw_work)),
        withdrawal_notice=WITHDRAWAL_NOTICE_RETRACTED if is_retracted else None,
        withdrawal_evidence="OpenAlex marks this work as retracted (is_retracted)" if is_retracted else None), ""


def rebuild_abstract_from_inverted_index(inverted_index: Any) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; put each word back at its positions (bounded)."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return None
    word_by_position: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and 0 <= position < MAXIMUM_ABSTRACT_WORD_POSITIONS:
                word_by_position[position] = word
    abstract_text = " ".join(word_by_position[position] for position in sorted(word_by_position))
    return clean_provider_text(abstract_text, MAXIMUM_ABSTRACT_CHARACTERS)


def _stated_identifier_links(raw_work: dict[str, Any]) -> list[RawIdentifier]:
    stated_links: list[RawIdentifier] = []
    work_identifiers = raw_work.get("ids") if isinstance(raw_work.get("ids"), dict) else {}
    for doi_value in (work_identifiers.get("doi"), raw_work.get("doi")):
        if isinstance(doi_value, str):
            stated_links.append(RawIdentifier(DOI_SCHEME, doi_value))
    for location in _list_of_objects(raw_work.get("locations")):
        landing_page_url = location.get("landing_page_url")
        if isinstance(landing_page_url, str) and ARXIV_LANDING_PAGE_PATTERN.match(landing_page_url):
            stated_links.append(RawIdentifier(ARXIV_SCHEME, landing_page_url))
        for location_doi in (location.get("doi"), location.get("id")):
            if isinstance(location_doi, str) and ARXIV_DATACITE_DOI_PREFIX in location_doi.lower():
                stated_links.append(RawIdentifier(DOI_SCHEME, location_doi.removeprefix("doi:")))
    return stated_links


def _list_of_objects(raw_value: Any) -> list[dict[str, Any]]:
    return [item for item in raw_value if isinstance(item, dict)] if isinstance(raw_value, list) else []
