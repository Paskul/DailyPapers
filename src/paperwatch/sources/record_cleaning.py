"""Shared helpers that turn untrusted provider fields into bounded, clean values for a CollectedPaperRecord.

Provider text (titles, abstracts, author names) is untrusted (plan part 3): it is bounded in length, its whitespace
and control characters are collapsed, and it is only ever stored and later escaped, never executed or used to build
a request. XML is parsed only after refusing document type declarations, so no entity expansion can happen.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable

from paperwatch.errors import SourceInputError, SourceResponseRejected
from paperwatch.scholarly_identifiers import normalize_identifier
from paperwatch.sources.types import RawIdentifier
from paperwatch.text_normalization import collapse_whitespace

MAXIMUM_TITLE_CHARACTERS = 1_000
MAXIMUM_ABSTRACT_CHARACTERS = 20_000
MAXIMUM_AUTHOR_NAME_CHARACTERS = 200
MAXIMUM_AUTHORS_PER_RECORD = 100
XML_PROLOGUE_BYTES_CHECKED_FOR_DOCTYPE = 4_096


def clean_provider_text(raw_value: object, maximum_characters: int) -> str | None:
    """Collapse whitespace (control characters included) and bound the length; None if nothing printable is left."""
    if not isinstance(raw_value, str):
        return None
    printable_text = "".join(character if character.isprintable() else " " for character in raw_value)
    cleaned_text = collapse_whitespace(printable_text)[:maximum_characters].strip()
    return cleaned_text or None


def clean_author_names(raw_author_names: Iterable[object]) -> tuple[str, ...]:
    cleaned_names = []
    for raw_author_name in raw_author_names:
        cleaned_name = clean_provider_text(raw_author_name, MAXIMUM_AUTHOR_NAME_CHARACTERS)
        if cleaned_name:
            cleaned_names.append(cleaned_name)
        if len(cleaned_names) == MAXIMUM_AUTHORS_PER_RECORD:
            break
    return tuple(cleaned_names)


def keep_valid_identifiers(candidate_identifiers: Iterable[RawIdentifier]) -> tuple[RawIdentifier, ...]:
    """Drop linked identifiers that do not normalize, so one odd link never costs the whole record."""
    valid_identifiers: list[RawIdentifier] = []
    seen_normalized_forms: set[str] = set()
    for candidate_identifier in candidate_identifiers:
        try:
            normalized_form = normalize_identifier(candidate_identifier.scheme, candidate_identifier.value)
        except SourceInputError:
            continue
        if normalized_form.scheme_and_value not in seen_normalized_forms:
            seen_normalized_forms.add(normalized_form.scheme_and_value)
            valid_identifiers.append(candidate_identifier)
    return tuple(valid_identifiers)


def parse_untrusted_xml(body_bytes: bytes, source_label: str) -> ElementTree.Element:
    """Parse a provider's XML response; refuse a DOCTYPE (no entities, no external references)."""
    if b"<!DOCTYPE" in body_bytes[:XML_PROLOGUE_BYTES_CHECKED_FOR_DOCTYPE].upper():
        raise SourceResponseRejected(f"{source_label}: response declares a document type; refused")
    try:
        return ElementTree.fromstring(body_bytes)
    except ElementTree.ParseError as parse_error:
        raise SourceResponseRejected(f"{source_label}: response is not well-formed XML ({parse_error})") from None
