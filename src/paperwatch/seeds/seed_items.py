"""One seed record, whatever its source, plus the shared helpers both seed parsers use.

Identity of a seed:
    seed_source    "zotero" or "csl_json"
    seed_item_key  the Zotero item key (e.g. "X7K2P9QM"), or "<csl file key>#<CSL item id>" for a CSL JSON export
    seed_id        "<seed_source>:<seed_item_key>", the id a review cites as a matched seed

Every seed is also catalogued as a paper (so later parts can use it as a comparison anchor and keep it out of the
recommendations). Its catalog record uses the seed id as its own identifier, plus the DOI / arXiv id the library
states, so a seed and the same work found on arXiv land on one paper.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from paperwatch.errors import SourceInputError
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME, normalize_identifier
from paperwatch.sources.types import CollectedPaperRecord, RawIdentifier
from paperwatch.text_normalization import collapse_whitespace

SEED_SOURCE_ZOTERO = "zotero"
SEED_SOURCE_CSL_JSON = "csl_json"
SEED_SOURCES = (SEED_SOURCE_ZOTERO, SEED_SOURCE_CSL_JSON)

# Discovery routes of catalogued seeds. A paper found ONLY by these routes is private to the user's library.
ZOTERO_SEED_ROUTE_PREFIX = "zotero-seed:"
CSL_JSON_SEED_ROUTE_PREFIX = "csl-seed:"
SEED_DISCOVERY_ROUTE_PREFIXES = (ZOTERO_SEED_ROUTE_PREFIX, CSL_JSON_SEED_ROUTE_PREFIX)

# What text a seed offers for matching. A title-only seed is a recorded limitation.
MATCHING_TEXT_TITLE_AND_ABSTRACT = "title_and_abstract"
MATCHING_TEXT_TITLE_ONLY = "title_only"

MAXIMUM_SEED_TITLE_CHARACTERS = 1_000
MAXIMUM_SEED_ABSTRACT_CHARACTERS = 20_000
MAXIMUM_SEED_AUTHORS = 100
CSL_FILE_KEY_HASH_CHARACTERS = 10

# Identifier mentions inside structured metadata fields (URL, archive id, "DOI: ..." lines of Zotero's Extra field).
DOI_MENTION_PATTERN = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)")
ARXIV_MENTION_PATTERN = re.compile(
    r"(?:\barxiv\s*:\s*|arxiv\.org/(?:abs|pdf)/)(?P<arxiv_id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z.-]*/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
IDENTIFIER_LINE_PATTERN = re.compile(
    r"^\s*(?P<label>doi|arxiv(?: id)?)\s*:\s*(?P<value>\S+)\s*$", re.IGNORECASE | re.MULTILINE
)
PUBLICATION_YEAR_PATTERN = re.compile(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b")
TRAILING_PUNCTUATION = ".,;)]}"


@dataclass(frozen=True)
class SeedItem:
    """One bibliographic seed exactly as the library describes it (already cleaned and bounded)."""

    seed_source: str
    seed_item_key: str
    title: str
    authors: tuple[str, ...]
    abstract: str | None  # None: title-only seed
    publication_year: int | None
    item_type: str  # Zotero itemType or CSL type, e.g. "journalArticle", "article-journal"
    item_version: int | None  # Zotero item version; None for CSL JSON
    doi: str | None  # normalized DOI, if the library states one
    arxiv_id: str | None  # normalized arXiv id (with version if stated), if the library states one

    @property
    def seed_id(self) -> str:
        return build_seed_id(self.seed_source, self.seed_item_key)

    @property
    def matching_text_kind(self) -> str:
        return MATCHING_TEXT_TITLE_AND_ABSTRACT if self.abstract else MATCHING_TEXT_TITLE_ONLY

    @property
    def stated_identifiers(self) -> tuple[str, ...]:
        """'scheme:value' strings of the identifiers the library states (stored as seed_items.identifiers_json)."""
        return tuple(
            identifier
            for identifier in (
                f"{DOI_SCHEME}:{self.doi}" if self.doi else None,
                f"{ARXIV_SCHEME}:{self.arxiv_id}" if self.arxiv_id else None,
            )
            if identifier
        )

    def compute_metadata_sha256(self) -> str:
        """Fingerprint of the seed text used for matching. A changed fingerprint changes the topic profile hash."""
        metadata_text = json.dumps(
            {
                "title": self.title,
                "abstract": self.abstract or "",
                "authors": self.authors,
                "year": self.publication_year,
                "identifiers": self.stated_identifiers,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(metadata_text.encode("utf-8")).hexdigest()

    def to_catalog_record(self, discovery_route: str) -> CollectedPaperRecord:
        """The catalog sighting of this seed. Linked identifiers come only from fields the library states."""
        linked_identifiers = []
        if self.doi:
            linked_identifiers.append(RawIdentifier(DOI_SCHEME, self.doi))
        if self.arxiv_id:
            linked_identifiers.append(RawIdentifier(ARXIV_SCHEME, self.arxiv_id))
        return CollectedPaperRecord(
            provider=self.seed_source,
            provider_paper_id=self.seed_item_key,
            title=self.title,
            authors=self.authors,
            abstract=self.abstract,
            first_published_date=str(self.publication_year) if self.publication_year else None,
            latest_revision_date=None,
            source_version=None,
            discovery_route=discovery_route,
            linked_identifiers=tuple(linked_identifiers),
        )


def build_seed_id(seed_source: str, seed_item_key: str) -> str:
    return f"{seed_source}:{seed_item_key}"


def build_csl_json_file_key(configured_seed_file_path: str) -> str:
    """Short stable key of one configured CSL JSON file (its path as written in topics.toml), e.g. '3f9a1c0b7d'."""
    return hashlib.sha256(configured_seed_file_path.strip().encode("utf-8")).hexdigest()[:CSL_FILE_KEY_HASH_CHARACTERS]


def zotero_seed_discovery_route(collection_key: str) -> str:
    return f"{ZOTERO_SEED_ROUTE_PREFIX}{collection_key}"


def csl_json_seed_discovery_route(csl_file_key: str) -> str:
    return f"{CSL_JSON_SEED_ROUTE_PREFIX}{csl_file_key}"


# --------------------------------------------------------------------------------------------------------------------
# Cleaning helpers shared by the Zotero and CSL JSON parsers
# --------------------------------------------------------------------------------------------------------------------


def clean_seed_text(raw_value: object, maximum_characters: int) -> str | None:
    """Collapse whitespace and bound the length; None for anything that is not a non-empty string."""
    if not isinstance(raw_value, str):
        return None
    cleaned_text = collapse_whitespace(raw_value)
    return cleaned_text[:maximum_characters] if cleaned_text else None


def bound_author_names(author_names: list[str]) -> tuple[str, ...]:
    return tuple(author_names[:MAXIMUM_SEED_AUTHORS])


def find_publication_year(*candidate_texts: object) -> int | None:
    """The first plausible four-digit year in the candidates (e.g. '2023-05-12', 'May 2019')."""
    for candidate_text in candidate_texts:
        if isinstance(candidate_text, int) and not isinstance(candidate_text, bool) and 1500 <= candidate_text <= 2199:
            return candidate_text
        if isinstance(candidate_text, str):
            year_match = PUBLICATION_YEAR_PATTERN.search(candidate_text)
            if year_match:
                return int(year_match.group(1))
    return None


def find_doi(direct_doi_value: object, *texts_that_may_mention_a_doi: object) -> str | None:
    """A normalized DOI from the dedicated DOI field, else from a doi.org URL or a 'DOI: ...' line."""
    for candidate in (direct_doi_value, *_identifier_line_values(texts_that_may_mention_a_doi, "doi")):
        normalized_doi = _normalize_or_none(DOI_SCHEME, candidate)
        if normalized_doi:
            return normalized_doi
    for text in texts_that_may_mention_a_doi:
        if isinstance(text, str) and "doi.org/" in text.lower():
            doi_match = DOI_MENTION_PATTERN.search(text)
            if doi_match:
                normalized_doi = _normalize_or_none(DOI_SCHEME, doi_match.group(1).rstrip(TRAILING_PUNCTUATION))
                if normalized_doi:
                    return normalized_doi
    return None


def find_arxiv_id(*texts_that_may_mention_an_arxiv_id: object) -> str | None:
    """A normalized arXiv id ('2401.01234v2', 'hep-th/9901001') from 'arXiv:...' or arxiv.org/abs/... mentions."""
    for candidate in _identifier_line_values(texts_that_may_mention_an_arxiv_id, "arxiv"):
        normalized_arxiv_id = _normalize_or_none(ARXIV_SCHEME, candidate)
        if normalized_arxiv_id:
            return normalized_arxiv_id
    for text in texts_that_may_mention_an_arxiv_id:
        if isinstance(text, str):
            arxiv_match = ARXIV_MENTION_PATTERN.search(text)
            if arxiv_match:
                normalized_arxiv_id = _normalize_or_none(ARXIV_SCHEME, arxiv_match["arxiv_id"])
                if normalized_arxiv_id:
                    return normalized_arxiv_id
    return None


def _identifier_line_values(texts: tuple[object, ...], wanted_label_prefix: str) -> list[str]:
    """Values of structured 'DOI: ...' / 'arXiv: ...' lines. All other text in those fields is ignored."""
    values: list[str] = []
    for text in texts:
        if isinstance(text, str):
            for line_match in IDENTIFIER_LINE_PATTERN.finditer(text):
                if line_match["label"].lower().startswith(wanted_label_prefix):
                    values.append(line_match["value"].rstrip(TRAILING_PUNCTUATION))
    return values


def _normalize_or_none(scheme: str, raw_value: object) -> str | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        normalized_identifier = normalize_identifier(scheme, raw_value)
    except SourceInputError:
        return None
    version_suffix = f"v{normalized_identifier.version_number}" if normalized_identifier.version_number else ""
    return normalized_identifier.value + version_suffix
