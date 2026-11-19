"""What every paper source hands to the catalog, whatever the provider (fixture, arXiv, OpenAlex, Zotero...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CollectedPaperRecord:
    provider: str  # e.g. "fixture", "arxiv", "openalex"
    provider_paper_id: str  # the provider's own identifier for the work
    title: str
    authors: tuple[str, ...]
    abstract: str | None  # None when the provider supplied no abstract (title-only)
    first_published_date: str | None
    latest_revision_date: str | None
    source_version: str | None  # e.g. arXiv "v2"
    discovery_route: str  # how we found it, e.g. "fixture:recorded_paper.json", "arxiv-feed:cs.RO"


class PaperSource(Protocol):
    source_name: str

    def fetch_records(self) -> list[CollectedPaperRecord]:
        """Return records; raise SourceInputError (or a subclass) naming what went wrong."""
