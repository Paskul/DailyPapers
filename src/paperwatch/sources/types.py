"""What every paper source hands to the catalog, whatever the provider (fixture, arXiv, OpenAlex, Zotero...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol


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


class PaperSource(Protocol):
    source_name: str

    def fetch_records(self) -> list[CollectedPaperRecord]:
        """Return records; raise SourceInputError (or a subclass) naming what went wrong."""
