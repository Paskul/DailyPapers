"""Parse arXiv's two Atom formats into CollectedPaperRecords (plan part 5).

1. The category announcement feed (rss.arxiv.org/atom/<category>): one entry per announced item, with
   <arxiv:announce_type> "new" | "cross" | "replace" | "replace-cross" and the announcement date in <published>.
   The feed does not report the original submission date, so:
     * a new or cross-listed v1 is first published on its announcement date (day precision), unless its id shows it
       was first submitted well before (an old paper cross-listed late): then the id's month is used;
     * a revision ("replace") keeps the month encoded in its id (1806.01234 -> "2018-06") as its first publication,
       and its announcement date becomes the latest revision date. A 2018 paper revised today is older literature.
2. The legacy API (export.arxiv.org/api/query): <published> is the v1 submission time and <updated> this version's,
   and opensearch elements give the total for paging.

An entry that cannot be parsed is skipped and counted; one bad entry never costs the page. Withdrawals are recorded
when arXiv's own text says so: an author comment or abstract stating that the paper has been withdrawn.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from xml.etree import ElementTree

from paperwatch.errors import SourceResponseRejected
from paperwatch.publication_dates import normalize_partial_date
from paperwatch.scholarly_identifiers import ARXIV_ID_WITH_OPTIONAL_VERSION_PATTERN, ARXIV_SCHEME, DOI_SCHEME
from paperwatch.sources.record_cleaning import (
    MAXIMUM_ABSTRACT_CHARACTERS,
    MAXIMUM_TITLE_CHARACTERS,
    clean_author_names,
    clean_provider_text,
    keep_valid_identifiers,
    parse_untrusted_xml,
)
from paperwatch.sources.types import (
    ANNOUNCEMENT_KIND_CROSS_LIST,
    ANNOUNCEMENT_KIND_NEW,
    ANNOUNCEMENT_KIND_REVISION,
    WITHDRAWAL_NOTICE_WITHDRAWN,
    CollectedPaperRecord,
    ParsedSearchPage,
    RawIdentifier,
)

ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"
ARXIV_NAMESPACE = "{http://arxiv.org/schemas/atom}"
DUBLIN_CORE_NAMESPACE = "{http://purl.org/dc/elements/1.1/}"
OPENSEARCH_NAMESPACE = "{http://a9.com/-/spec/opensearch/1.1/}"

ANNOUNCEMENT_KIND_BY_FEED_LABEL = {
    "new": ANNOUNCEMENT_KIND_NEW,
    "cross": ANNOUNCEMENT_KIND_CROSS_LIST,
    "replace": ANNOUNCEMENT_KIND_REVISION,
    "replace-cross": ANNOUNCEMENT_KIND_REVISION,
}
FEED_ENTRY_ID_PREFIX = "oai:arXiv.org:"
API_ENTRY_ID_PATTERN = re.compile(r"^https?://arxiv\.org/abs/(?P<arxiv_id>.+)$")
API_ERROR_ENTRY_MARKER = "/api/errors"
# The feed's summary starts with "arXiv:2611.23499v1 Announce Type: new \nAbstract: ..."; keep only the abstract.
FEED_SUMMARY_PREFIX_PATTERN = re.compile(r"^\s*arXiv:\S+\s+Announce Type:\s*\S+\s*(Abstract:\s*)?", re.IGNORECASE)
WITHDRAWAL_STATEMENT_PATTERN = re.compile(
    r"\b(this (paper|article|submission|manuscript|preprint) (has been|is being|was) withdrawn"
    r"|withdrawn by (the )?authors?|paper (has been )?withdrawn)\b",
    re.IGNORECASE,
)
NEW_STYLE_ARXIV_ID_MONTH_PATTERN = re.compile(r"^(?P<yy>\d{2})(?P<mm>\d{2})\.\d{4,5}$")
OLD_STYLE_ARXIV_ID_MONTH_PATTERN = re.compile(r"^[a-z][a-z.-]*/(?P<yy>\d{2})(?P<mm>\d{2})\d{3}$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedFeed:
    records: list[CollectedPaperRecord]
    raw_entry_count: int
    announcement_date: str | None  # the announcement day shown by the feed (None for an empty feed)
    announcement_kind_counts: Counter[str] = field(default_factory=Counter)
    skipped_entry_counts: Counter[str] = field(default_factory=Counter)


# --------------------------------------------------------------------------------------------------------------------
# Announcement feed
# --------------------------------------------------------------------------------------------------------------------


def parse_category_feed(body_bytes: bytes, discovery_route: str) -> ParsedFeed:
    feed_element = parse_untrusted_xml(body_bytes, "arXiv feed")
    if feed_element.tag != f"{ATOM_NAMESPACE}feed":
        raise SourceResponseRejected("arXiv feed: the response is not an Atom feed")
    records: list[CollectedPaperRecord] = []
    kind_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    announcement_dates: list[str] = []
    entry_elements = feed_element.findall(f"{ATOM_NAMESPACE}entry")
    for entry_element in entry_elements:
        parsed_record, skip_reason = _convert_feed_entry(entry_element, discovery_route)
        if parsed_record is None:
            skipped_counts[skip_reason] += 1
            continue
        records.append(parsed_record)
        kind_counts[parsed_record.announcement_kind] += 1
        announced_on = parsed_record.latest_revision_date or parsed_record.first_published_date
        if announced_on and len(announced_on) == 10:
            announcement_dates.append(announced_on)
    return ParsedFeed(records, len(entry_elements), max(announcement_dates, default=None), kind_counts, skipped_counts)


def _convert_feed_entry(
    entry_element: ElementTree.Element,
    discovery_route: str,
) -> tuple[CollectedPaperRecord | None, str]:
    entry_id = _child_text(entry_element, f"{ATOM_NAMESPACE}id")
    if not entry_id.startswith(FEED_ENTRY_ID_PREFIX):
        return None, "unexpected entry id"
    arxiv_id_match = ARXIV_ID_WITH_OPTIONAL_VERSION_PATTERN.match(entry_id.removeprefix(FEED_ENTRY_ID_PREFIX))
    if arxiv_id_match is None:
        return None, "invalid arXiv id"
    announcement_kind = ANNOUNCEMENT_KIND_BY_FEED_LABEL.get(
        _child_text(entry_element, f"{ARXIV_NAMESPACE}announce_type").lower()
    )
    if announcement_kind is None:
        return None, "unknown announcement type"
    title = clean_provider_text(_child_text(entry_element, f"{ATOM_NAMESPACE}title"), MAXIMUM_TITLE_CHARACTERS)
    if title is None:
        return None, "no title"
    abstract = clean_provider_text(
        FEED_SUMMARY_PREFIX_PATTERN.sub("", _child_text(entry_element, f"{ATOM_NAMESPACE}summary")),
        MAXIMUM_ABSTRACT_CHARACTERS,
    )
    announcement_date = normalize_partial_date(_child_text(entry_element, f"{ATOM_NAMESPACE}published"))
    submission_month = arxiv_id_submission_month(arxiv_id_match["base_id"])

    if announcement_kind == ANNOUNCEMENT_KIND_REVISION:
        first_published_date, latest_revision_date = submission_month, announcement_date
    elif (
        announcement_date
        and submission_month
        and not _announced_soon_after_submission(announcement_date, submission_month)
    ):
        first_published_date, latest_revision_date = submission_month, None  # an old paper cross-listed late
    else:
        first_published_date, latest_revision_date = announcement_date or submission_month, None

    author_names = [
        author_name
        for creator in entry_element.findall(f"{DUBLIN_CORE_NAMESPACE}creator")
        for author_name in (creator.text or "").split(",")
    ]
    linked_identifiers = [
        RawIdentifier(DOI_SCHEME, doi_text)
        for doi_text in _child_texts(entry_element, f"{ARXIV_NAMESPACE}DOI", f"{ARXIV_NAMESPACE}doi")
    ]
    withdrawal_evidence = _find_withdrawal_statement(abstract)
    return CollectedPaperRecord(
        provider=ARXIV_SCHEME,
        provider_paper_id=arxiv_id_match.group(0),
        title=title,
        authors=clean_author_names(author_names),
        abstract=abstract,
        first_published_date=first_published_date,
        latest_revision_date=latest_revision_date,
        source_version=None,
        discovery_route=discovery_route,
        linked_identifiers=keep_valid_identifiers(linked_identifiers),
        announcement_kind=announcement_kind,
        withdrawal_notice=WITHDRAWAL_NOTICE_WITHDRAWN if withdrawal_evidence else None,
        withdrawal_evidence=withdrawal_evidence,
    ), ""


# --------------------------------------------------------------------------------------------------------------------
# Legacy API search page
# --------------------------------------------------------------------------------------------------------------------


def parse_api_search_page(body_bytes: bytes, discovery_route: str, start_offset: int) -> ParsedSearchPage:
    """One page of export.arxiv.org/api/query. The next cursor is the next start offset, or None at the end.

    arXiv occasionally answers an empty page before the reported end; that is refused (the cursor is kept and the
    page retried next run) rather than mistaken for the end of the results.
    """
    feed_element = parse_untrusted_xml(body_bytes, "arXiv API")
    if feed_element.tag != f"{ATOM_NAMESPACE}feed":
        raise SourceResponseRejected("arXiv API: the response is not an Atom feed")
    entry_elements = feed_element.findall(f"{ATOM_NAMESPACE}entry")
    for entry_element in entry_elements:
        if API_ERROR_ENTRY_MARKER in _child_text(entry_element, f"{ATOM_NAMESPACE}id"):
            error_text = clean_provider_text(_child_text(entry_element, f"{ATOM_NAMESPACE}summary"), 200)
            raise SourceResponseRejected(f"arXiv API reported an error: {error_text}")
    total_results = _optional_integer(_child_text(feed_element, f"{OPENSEARCH_NAMESPACE}totalResults"))

    records: list[CollectedPaperRecord] = []
    skipped_counts: Counter[str] = Counter()
    for entry_element in entry_elements:
        parsed_record, skip_reason = _convert_api_entry(entry_element, discovery_route)
        if parsed_record is None:
            skipped_counts[skip_reason] += 1
        else:
            records.append(parsed_record)

    end_offset = start_offset + len(entry_elements)
    if not entry_elements:
        if total_results is not None and start_offset < total_results:
            raise SourceResponseRejected(
                f"arXiv API returned an empty page at offset {start_offset} of "
                f"{total_results} results; the page will be retried"
            )
        next_cursor = None
    else:
        next_cursor = str(end_offset) if total_results is None or end_offset < total_results else None
    return ParsedSearchPage(records, len(entry_elements), next_cursor, total_results, skipped_counts)


def _convert_api_entry(
    entry_element: ElementTree.Element,
    discovery_route: str,
) -> tuple[CollectedPaperRecord | None, str]:
    entry_id_match = API_ENTRY_ID_PATTERN.match(_child_text(entry_element, f"{ATOM_NAMESPACE}id"))
    arxiv_id_match = (
        ARXIV_ID_WITH_OPTIONAL_VERSION_PATTERN.match(entry_id_match["arxiv_id"]) if entry_id_match else None
    )
    if arxiv_id_match is None:
        return None, "invalid arXiv id"
    title = clean_provider_text(_child_text(entry_element, f"{ATOM_NAMESPACE}title"), MAXIMUM_TITLE_CHARACTERS)
    if title is None:
        return None, "no title"
    abstract = clean_provider_text(_child_text(entry_element, f"{ATOM_NAMESPACE}summary"), MAXIMUM_ABSTRACT_CHARACTERS)
    version_number = int(arxiv_id_match["version_number"] or 1)
    first_published_date = normalize_partial_date(
        _child_text(entry_element, f"{ATOM_NAMESPACE}published")
    ) or arxiv_id_submission_month(arxiv_id_match["base_id"])
    latest_revision_date = (
        normalize_partial_date(_child_text(entry_element, f"{ATOM_NAMESPACE}updated")) if version_number > 1 else None
    )
    author_names = [
        _child_text(author_element, f"{ATOM_NAMESPACE}name")
        for author_element in entry_element.findall(f"{ATOM_NAMESPACE}author")
    ]
    linked_identifiers = [
        RawIdentifier(DOI_SCHEME, doi_text)
        for doi_text in _child_texts(entry_element, f"{ARXIV_NAMESPACE}doi", f"{ARXIV_NAMESPACE}DOI")
    ]
    comment = clean_provider_text(_child_text(entry_element, f"{ARXIV_NAMESPACE}comment"), 1_000)
    withdrawal_evidence = _find_withdrawal_statement(comment) or _find_withdrawal_statement(abstract)
    return CollectedPaperRecord(
        provider=ARXIV_SCHEME,
        provider_paper_id=arxiv_id_match.group(0),
        title=title,
        authors=clean_author_names(author_names),
        abstract=abstract,
        first_published_date=first_published_date,
        latest_revision_date=latest_revision_date,
        source_version=None,
        discovery_route=discovery_route,
        linked_identifiers=keep_valid_identifiers(linked_identifiers),
        withdrawal_notice=WITHDRAWAL_NOTICE_WITHDRAWN if withdrawal_evidence else None,
        withdrawal_evidence=withdrawal_evidence,
    ), ""


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def arxiv_id_submission_month(base_arxiv_id: str) -> str | None:
    """The month an arXiv id was first submitted, from the id itself: '1806.01234' -> '2018-06',
    'hep-th/9901001' -> '1999-01'. New-style ids began in April 2007; old-style ids use 91..99 for the 1990s."""
    new_style_match = NEW_STYLE_ARXIV_ID_MONTH_PATTERN.match(base_arxiv_id)
    if new_style_match:
        year = 2000 + int(new_style_match["yy"])
        month_number = int(new_style_match["mm"])
    else:
        old_style_match = OLD_STYLE_ARXIV_ID_MONTH_PATTERN.match(base_arxiv_id)
        if old_style_match is None:
            return None
        two_digit_year = int(old_style_match["yy"])
        year = (1900 if two_digit_year >= 91 else 2000) + two_digit_year
        month_number = int(old_style_match["mm"])
    return normalize_partial_date(f"{year:04d}-{month_number:02d}")


def _announced_soon_after_submission(announcement_date: str, submission_month: str) -> bool:
    """True if the announcement falls in the id's month or the month after (a v1 submitted near a month's end)."""
    announcement_year, announcement_month = int(announcement_date[:4]), int(announcement_date[5:7])
    submission_year, submission_month_number = int(submission_month[:4]), int(submission_month[5:7])
    months_between = (announcement_year - submission_year) * 12 + (announcement_month - submission_month_number)
    return 0 <= months_between <= 1


def _find_withdrawal_statement(text: str | None) -> str | None:
    if not text:
        return None
    statement_match = WITHDRAWAL_STATEMENT_PATTERN.search(text)
    if statement_match is None:
        return None
    return text[max(0, statement_match.start() - 40) : statement_match.end() + 120]


def _child_text(parent_element: ElementTree.Element, tag: str) -> str:
    child_element = parent_element.find(tag)
    return (child_element.text or "").strip() if child_element is not None else ""


def _child_texts(parent_element: ElementTree.Element, *tags: str) -> list[str]:
    return [
        child_element.text.strip()
        for tag in tags
        for child_element in parent_element.findall(tag)
        if child_element.text and child_element.text.strip()
    ]


def _optional_integer(text: str) -> int | None:
    return int(text) if text.isdigit() else None
