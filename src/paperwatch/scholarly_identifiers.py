"""Normalize the public identifiers providers use for scholarly works: DOI, arXiv, OpenAlex, and opaque others.

Rules:
  * DOIs are case-insensitive: strip resolver prefixes ("https://doi.org/", "doi:"), percent-decode, lowercase.
  * arXiv ids lose their "arXiv:" / URL prefixes. The version ("v2") is kept SEPARATELY from the base id, so every
    version of a preprint shares one alias while each version can still be recorded as its own edition.
  * The DataCite DOI that arXiv mints for each preprint (10.48550/arXiv.<id>) is a clear identifier link to that
    arXiv id, so it implies the arXiv alias.
  * OpenAlex work ids are reduced to the bare "W123..." form.
  * Any other provider (recorded fixtures, future sources) keeps its id as given, minus surrounding whitespace.

A malformed identifier raises SourceInputError: it is a problem with the source, never a reason to create a paper.
These are also the only identifiers from which external links are built (see rendering/markdown_safety.py).
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass

from paperwatch.errors import SourceInputError

ARXIV_SCHEME = "arxiv"
DOI_SCHEME = "doi"
OPENALEX_SCHEME = "openalex"

IDENTIFIER_SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAXIMUM_OPAQUE_IDENTIFIER_LENGTH = 200
MAXIMUM_VERSION_LABEL_LENGTH = 32

_NEW_STYLE_ARXIV_ID = r"\d{4}\.\d{4,5}"  # 2401.01234 (since 2007)
_OLD_STYLE_ARXIV_ID = r"[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7}"  # hep-th/9901001, math.GT/0309136
ARXIV_ID_WITH_OPTIONAL_VERSION_PATTERN = re.compile(
    rf"^(?P<base_id>{_NEW_STYLE_ARXIV_ID}|{_OLD_STYLE_ARXIV_ID})(?:v(?P<version_number>[1-9]\d*))?$", re.IGNORECASE
)
ARXIV_PREFIX_PATTERN = re.compile(r"^(?:arxiv:|https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/)", re.IGNORECASE)
DOI_PREFIX_PATTERN = re.compile(r"^(?:doi:|https?://(?:dx\.)?doi\.org/)", re.IGNORECASE)
CANONICAL_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
OPENALEX_PREFIX_PATTERN = re.compile(r"^https?://openalex\.org/", re.IGNORECASE)
OPENALEX_WORK_ID_PATTERN = re.compile(r"^w(?P<digits>\d+)$", re.IGNORECASE)
ARXIV_DATACITE_DOI_PREFIX = "10.48550/arxiv."
VERSION_LABEL_PATTERN = re.compile(r"^v(?P<version_number>[1-9]\d*)$")


@dataclass(frozen=True)
class NormalizedIdentifier:
    """One canonical identifier. `value` never contains a version; the arXiv version lives in `version_number`."""

    scheme: str
    value: str
    version_number: int | None = None

    @property
    def scheme_and_value(self) -> str:
        """Readable 'scheme:value' form, e.g. 'arxiv:2401.01234'. Used in logs, evidence, and citation edges."""
        return f"{self.scheme}:{self.value}"

    def without_version(self) -> NormalizedIdentifier:
        return NormalizedIdentifier(self.scheme, self.value)


def normalize_identifier(scheme: str, raw_value: str) -> NormalizedIdentifier:
    """Return the canonical form of one identifier, or raise SourceInputError naming what is wrong."""
    normalized_scheme = (scheme or "").strip().lower()
    if not IDENTIFIER_SCHEME_PATTERN.match(normalized_scheme):
        raise SourceInputError(f"identifier scheme '{scheme}' must match {IDENTIFIER_SCHEME_PATTERN.pattern}")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise SourceInputError(f"empty {normalized_scheme} identifier")
    stripped_value = raw_value.strip()
    if normalized_scheme == ARXIV_SCHEME:
        return _normalize_arxiv_identifier(stripped_value)
    if normalized_scheme == DOI_SCHEME:
        return NormalizedIdentifier(DOI_SCHEME, _normalize_doi(stripped_value))
    if normalized_scheme == OPENALEX_SCHEME:
        return NormalizedIdentifier(OPENALEX_SCHEME, _normalize_openalex_work_id(stripped_value))
    return NormalizedIdentifier(normalized_scheme, _validate_opaque_identifier(normalized_scheme, stripped_value))


def identifiers_implied_by(identifier: NormalizedIdentifier) -> tuple[NormalizedIdentifier, ...]:
    """Identifiers that are the same work by definition, e.g. arXiv's own DataCite DOI implies the arXiv id."""
    if identifier.scheme == DOI_SCHEME and identifier.value.startswith(ARXIV_DATACITE_DOI_PREFIX):
        try:
            return (_normalize_arxiv_identifier(identifier.value.removeprefix(ARXIV_DATACITE_DOI_PREFIX)),)
        except SourceInputError:
            return ()
    return ()


def is_canonical_identifier(scheme: str, value: str) -> bool:
    """True if (scheme, value) is already in canonical, version-free form. Used before building external links."""
    try:
        return normalize_identifier(scheme, value) == NormalizedIdentifier(scheme, value)
    except SourceInputError:
        return False


def resolve_edition_version_label(
    primary_identifier: NormalizedIdentifier, provider_version_label: str | None
) -> str | None:
    """The edition's version: the arXiv version inside the id wins, else the provider's own label (or None)."""
    if primary_identifier.version_number is not None:
        return f"v{primary_identifier.version_number}"
    return normalize_version_label(provider_version_label)


def build_edition_key(primary_identifier: NormalizedIdentifier, version_label: str | None) -> str:
    """Unique key of one edition, e.g. 'arxiv:2401.01234@v2', 'doi:10.1109/x', 'fixture:demo-0001@v1'."""
    edition_key = primary_identifier.without_version().scheme_and_value
    return f"{edition_key}@{version_label}" if version_label else edition_key


def parse_version_number(version_label: str | None) -> int | None:
    """'v3' -> 3; anything else (None, 'published', '2nd') -> None."""
    if not version_label:
        return None
    version_match = VERSION_LABEL_PATTERN.match(version_label.strip().lower())
    return int(version_match["version_number"]) if version_match else None


def normalize_version_label(raw_version_label: str | None) -> str | None:
    """Trim a provider's free-text version label; 'V2' becomes 'v2'. Empty -> None."""
    if raw_version_label is None or not raw_version_label.strip():
        return None
    trimmed_label = raw_version_label.strip()
    if len(trimmed_label) > MAXIMUM_VERSION_LABEL_LENGTH or any(
        not character.isprintable() for character in trimmed_label
    ):
        raise SourceInputError(f"version label {raw_version_label!r} is too long or contains control characters")
    return trimmed_label.lower() if VERSION_LABEL_PATTERN.match(trimmed_label.lower()) else trimmed_label


# --------------------------------------------------------------------------------------------------------------------
# Scheme-specific normalizers
# --------------------------------------------------------------------------------------------------------------------


def _normalize_arxiv_identifier(stripped_value: str) -> NormalizedIdentifier:
    candidate_id = ARXIV_PREFIX_PATTERN.sub("", stripped_value)
    if candidate_id.lower().endswith(".pdf"):
        candidate_id = candidate_id[: -len(".pdf")]
    arxiv_match = ARXIV_ID_WITH_OPTIONAL_VERSION_PATTERN.match(candidate_id)
    if arxiv_match is None:
        raise SourceInputError(f"'{stripped_value}' is not a valid arXiv identifier")
    base_id = arxiv_match["base_id"]
    if "/" in base_id:  # old style: archive is lowercase, subject class uppercase (math.GT/0309136)
        archive_and_subject, number = base_id.split("/")
        archive, _, subject_class = archive_and_subject.partition(".")
        base_id = (
            f"{archive.lower()}.{subject_class.upper()}/{number}" if subject_class else f"{archive.lower()}/{number}"
        )
    version_number = int(arxiv_match["version_number"]) if arxiv_match["version_number"] else None
    return NormalizedIdentifier(ARXIV_SCHEME, base_id, version_number)


def _normalize_doi(stripped_value: str) -> str:
    candidate_doi = urllib.parse.unquote(DOI_PREFIX_PATTERN.sub("", stripped_value)).strip().lower()
    if not CANONICAL_DOI_PATTERN.match(candidate_doi):
        raise SourceInputError(f"'{stripped_value}' is not a valid DOI")
    return candidate_doi


def _normalize_openalex_work_id(stripped_value: str) -> str:
    work_match = OPENALEX_WORK_ID_PATTERN.match(OPENALEX_PREFIX_PATTERN.sub("", stripped_value))
    if work_match is None:
        raise SourceInputError(f"'{stripped_value}' is not an OpenAlex work id (W followed by digits)")
    return f"W{work_match['digits']}"


def _validate_opaque_identifier(scheme: str, stripped_value: str) -> str:
    if len(stripped_value) > MAXIMUM_OPAQUE_IDENTIFIER_LENGTH:
        raise SourceInputError(f"{scheme} identifier is longer than {MAXIMUM_OPAQUE_IDENTIFIER_LENGTH} characters")
    if any(not character.isprintable() for character in stripped_value):
        raise SourceInputError(f"{scheme} identifier {stripped_value!r} contains control characters")
    return stripped_value
