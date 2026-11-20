"""Build provider requests from validated inputs only: catalog queries, normalized identifiers, internal ids.

Every function returns a ProviderRequest (provider, operation, path, query). The gateway then checks it against the
read-only provider rules before anything is sent, so a builder bug cannot widen access either. Nothing here accepts
free text from papers, notes, or the model; the only free text is a public phrase from the approved query catalog,
which is URL-encoded as a parameter value.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from datetime import date

from paperwatch.errors import SourceAccessDenied
from paperwatch.public_query_catalog import QUERY_KIND_CATEGORY, QUERY_KIND_KEYWORDS, PublicQuery
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME, OPENALEX_SCHEME, NormalizedIdentifier

OPERATION_SEARCH = "search"
OPERATION_GET_WORK = "get_work"
OPERATION_GET_TEXT = "get_text"
OPERATION_LIST_SEED_ITEMS = "list_seed_items"

MAXIMUM_ARXIV_START_OFFSET = 100_000
OPENALEX_CURSOR_PATTERN = re.compile(r"^(\*|[A-Za-z0-9+/=_-]{1,1024})$")
EARLIEST_PUBLICATION_YEAR = 1900


@dataclass(frozen=True)
class ProviderRequest:
    provider_name: str
    operation: str
    path: str  # percent-encoded
    query_parameters: tuple[tuple[str, str], ...] = ()
    request_headers: tuple[tuple[str, str], ...] = ()  # e.g. a conditional header; names must be allowed by the rules


@dataclass(frozen=True)
class PublicationWindow:
    """An inclusive date range for a dated search (missed releases, year bands of the historical search)."""

    first_date: date
    last_date: date

    def __post_init__(self) -> None:
        if not (EARLIEST_PUBLICATION_YEAR <= self.first_date.year and self.first_date <= self.last_date):
            raise SourceAccessDenied(f"invalid publication window {self.first_date} .. {self.last_date}")


# --------------------------------------------------------------------------------------------------------------------
# search(topic_query_id, cursor)
# --------------------------------------------------------------------------------------------------------------------


def build_search_request(public_query: PublicQuery, cursor: str | None, page_size: int,
                         publication_window: PublicationWindow | None = None) -> ProviderRequest:
    if public_query.provider_name == "arxiv":
        return _build_arxiv_search_request(public_query, cursor, page_size, publication_window)
    if public_query.provider_name == "openalex":
        return _build_openalex_search_request(public_query, cursor, page_size, publication_window)
    raise SourceAccessDenied(f"provider '{public_query.provider_name}' has no search operation")


def _build_arxiv_search_request(public_query: PublicQuery, cursor: str | None, page_size: int,
                                publication_window: PublicationWindow | None) -> ProviderRequest:
    if public_query.query_kind == QUERY_KIND_KEYWORDS:
        phrase_without_quotes = public_query.public_value.replace('"', " ").strip()
        search_expression = f'all:"{phrase_without_quotes}"'
    elif public_query.query_kind == QUERY_KIND_CATEGORY:
        search_expression = f"cat:{public_query.public_value}"
    else:
        raise SourceAccessDenied(f"arXiv cannot search query kind '{public_query.query_kind}'")
    if publication_window is not None:
        search_expression = (f"({search_expression}) AND submittedDate:"
                             f"[{publication_window.first_date:%Y%m%d}0000 TO {publication_window.last_date:%Y%m%d}2359]")
    start_offset = _parse_arxiv_start_offset(cursor)
    return ProviderRequest("arxiv", OPERATION_SEARCH, "/api/query", (
        ("search_query", search_expression), ("start", str(start_offset)), ("max_results", str(page_size)),
        ("sortBy", "submittedDate"), ("sortOrder", "descending")))


def _build_openalex_search_request(public_query: PublicQuery, cursor: str | None, page_size: int,
                                   publication_window: PublicationWindow | None) -> ProviderRequest:
    if public_query.query_kind != QUERY_KIND_KEYWORDS:
        raise SourceAccessDenied(f"OpenAlex cannot search query kind '{public_query.query_kind}'")
    openalex_cursor = cursor or "*"
    if not OPENALEX_CURSOR_PATTERN.match(openalex_cursor):
        raise SourceAccessDenied("OpenAlex cursor has an unexpected form; refusing to send it")
    query_parameters = [("search", public_query.public_value), ("per-page", str(page_size)),
                        ("cursor", openalex_cursor)]
    if publication_window is not None:
        query_parameters.append(("filter", f"from_publication_date:{publication_window.first_date.isoformat()},"
                                           f"to_publication_date:{publication_window.last_date.isoformat()}"))
    return ProviderRequest("openalex", OPERATION_SEARCH, "/works", tuple(query_parameters))


def _parse_arxiv_start_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    if not cursor.isdigit() or int(cursor) > MAXIMUM_ARXIV_START_OFFSET:
        raise SourceAccessDenied(f"arXiv cursor must be a start offset 0..{MAXIMUM_ARXIV_START_OFFSET}")
    return int(cursor)


# --------------------------------------------------------------------------------------------------------------------
# get_work(provider_id)
# --------------------------------------------------------------------------------------------------------------------


def build_get_work_request(identifier: NormalizedIdentifier) -> ProviderRequest:
    """Metadata of one work. arXiv ids go to arXiv; OpenAlex ids and DOIs go to OpenAlex."""
    if identifier.scheme == ARXIV_SCHEME:
        versioned_id = identifier.value + (f"v{identifier.version_number}" if identifier.version_number else "")
        return ProviderRequest("arxiv", OPERATION_GET_WORK, "/api/query", (("id_list", versioned_id),))
    if identifier.scheme == OPENALEX_SCHEME:
        return ProviderRequest("openalex", OPERATION_GET_WORK, f"/works/{identifier.value}")
    if identifier.scheme == DOI_SCHEME:
        return ProviderRequest("openalex", OPERATION_GET_WORK,
                               f"/works/doi:{urllib.parse.quote(identifier.value, safe='/')}")
    raise SourceAccessDenied(f"no provider lookup exists for '{identifier.scheme}' identifiers")


# --------------------------------------------------------------------------------------------------------------------
# get_text(known_edition_id)
# --------------------------------------------------------------------------------------------------------------------


def build_get_text_request(edition_identifier: NormalizedIdentifier) -> ProviderRequest:
    """The text of exactly one edition: a VERSIONED arXiv PDF, or an OpenAlex content PDF."""
    if edition_identifier.scheme == ARXIV_SCHEME:
        if edition_identifier.version_number is None:
            raise SourceAccessDenied(f"arXiv edition {edition_identifier.value} has no version; only a versioned "
                                     "PDF identifies the exact text that will be read")
        return ProviderRequest("arxiv", OPERATION_GET_TEXT,
                               f"/pdf/{edition_identifier.value}v{edition_identifier.version_number}")
    if edition_identifier.scheme == OPENALEX_SCHEME:
        return ProviderRequest("openalex", OPERATION_GET_TEXT, f"/works/{edition_identifier.value}.pdf")
    raise SourceAccessDenied(f"no permitted text route for '{edition_identifier.scheme}' editions")


# --------------------------------------------------------------------------------------------------------------------
# list_seed_items(collection_key, start)  (Zotero, plan part 4)
# --------------------------------------------------------------------------------------------------------------------


def build_seed_items_request(*, library_type: str, library_id: str, collection_key: str, start: int,
                             page_size: int, if_modified_since_library_version: int | None = None) -> ProviderRequest:
    """One page of the top-level items of one approved collection: bibliographic data only.

    Pages are ordered by date added (oldest first) so paging is stable while the library version stays the same.
    With `if_modified_since_library_version`, Zotero answers 304 Not Modified if nothing in the library changed.
    """
    if library_type not in ("user", "group") or not library_id.isdigit() or start < 0:
        raise SourceAccessDenied("invalid Zotero library or paging values")
    # "-attachment || note" asks Zotero to leave out attachments and notes; the seed parser
    # (paperwatch.seeds.zotero_items) still drops any note or attachment item it receives.
    query_parameters = (("format", "json"), ("include", "data"), ("itemType", "-attachment || note"),
                        ("sort", "dateAdded"), ("direction", "asc"), ("start", str(start)),
                        ("limit", str(page_size)))
    request_headers: tuple[tuple[str, str], ...] = ()
    if if_modified_since_library_version is not None:
        if isinstance(if_modified_since_library_version, bool) or int(if_modified_since_library_version) < 0:
            raise SourceAccessDenied("invalid Zotero library version")
        request_headers = (("If-Modified-Since-Version", str(int(if_modified_since_library_version))),)
    return ProviderRequest("zotero", OPERATION_LIST_SEED_ITEMS,
                           f"/{library_type}s/{library_id}/collections/{collection_key}/items/top",
                           query_parameters, request_headers)
