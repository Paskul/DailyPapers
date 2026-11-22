"""What every paper source hands to the catalog, whatever the provider (fixture, arXiv, OpenAlex, Zotero...)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol

# How an arXiv announcement presented an edition ("new, cross-listed, or revised").
ANNOUNCEMENT_KIND_NEW = "new"
ANNOUNCEMENT_KIND_CROSS_LIST = "cross_list"
ANNOUNCEMENT_KIND_REVISION = "revision"
ANNOUNCEMENT_KINDS = (ANNOUNCEMENT_KIND_NEW, ANNOUNCEMENT_KIND_CROSS_LIST, ANNOUNCEMENT_KIND_REVISION)

# A provider's report that a work should no longer be recommended.
WITHDRAWAL_NOTICE_WITHDRAWN = "withdrawn"  # arXiv: the authors withdrew the paper
WITHDRAWAL_NOTICE_RETRACTED = "retracted"  # OpenAlex: the publisher retracted the work
WITHDRAWAL_NOTICES = (WITHDRAWAL_NOTICE_WITHDRAWN, WITHDRAWAL_NOTICE_RETRACTED)


class RawIdentifier(NamedTuple):
    """An identifier exactly as a provider supplied it; the catalog normalizes it (paperwatch.scholarly_identifiers)."""

    scheme: str  # "doi", "arxiv", "openalex", or another provider name
    value: str  # e.g. "https://doi.org/10.1109/X", "arXiv:2401.01234v2", "W4400000001"


@dataclass(frozen=True)
class CollectedPaperRecord:
    """One sighting of one edition of a work.

    `provider` + `provider_paper_id` identify the edition this record describes (with the arXiv version inside the
    id, or in `source_version`). `linked_identifiers` are OTHER identifiers of the same work that the provider
    states explicitly, e.g. an OpenAlex work listing its DOI and arXiv id. Only put clear identifier links there:
    the catalog merges papers on them. Never put title or author look-alikes there.

    Dates are ISO forms whose length IS their precision (paperwatch.publication_dates): "2026" (year), "2026-11"
    (month), or "2026-11-17" (day). Parsers must never invent precision a provider did not report.
    """

    provider: str  # e.g. "fixture", "arxiv", "openalex"
    provider_paper_id: str  # the provider's own identifier for the work
    title: str
    authors: tuple[str, ...]
    abstract: str | None  # None when the provider supplied no abstract (title-only)
    first_published_date: str | None
    latest_revision_date: str | None
    source_version: str | None  # e.g. arXiv "v2"
    discovery_route: str  # how we found it, e.g. "fixture:recorded_paper.json", "arxiv-feed:cs.RO"
    linked_identifiers: tuple[RawIdentifier, ...] = ()
    # How an arXiv announcement presented this edition: "new", "cross_list", or "revision" (None for other routes).
    announcement_kind: str | None = None
    # A provider's own report that the work was withdrawn (arXiv) or retracted (OpenAlex); None when not reported.
    withdrawal_notice: str | None = None
    withdrawal_evidence: str | None = None  # short provider text that carried the notice, for the operator


class PaperSource(Protocol):
    source_name: str

    def fetch_records(self) -> list[CollectedPaperRecord]:
        """Return records; raise SourceInputError (or a subclass) naming what went wrong."""


@dataclass(frozen=True)
class ParsedSearchPage:
    """One parsed page of any provider's search results, plus what is needed to continue paging."""

    records: list[CollectedPaperRecord]
    raw_entry_count: int  # entries the provider returned, including skipped ones
    next_cursor: str | None  # None: the provider has no further page
    total_results: int | None
    skipped_entry_counts: Counter[str] = field(default_factory=Counter)
