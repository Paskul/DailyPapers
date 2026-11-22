"""Identifiers in a paper's own bibliography, and the check that a provider record matches the cited entry.

Plan part 8: "When the graph is incomplete, accept an identifier found in the paper's bibliography only after a
provider record confirms the match." The bibliography text is untrusted, so:
  * only well-formed DOIs and arXiv ids are taken (normalized; anything else is ignored);
  * an identifier becomes a citation edge only if the provider's record for it has a title whose words appear in
    the bibliography entry around the identifier (TITLE_WORD_SHARE_FOR_MATCH of them);
  * otherwise it stays in bibliography_identifiers as an unresolved reference (mismatch / not_found).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from paperwatch.errors import SourceInputError
from paperwatch.reading.passages import Passage
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME, NormalizedIdentifier, normalize_identifier
from paperwatch.text_normalization import build_title_match_key

DOI_IN_TEXT_PATTERN = re.compile(r"(?i)\b10\.\d{4,9}/[^\s\"'<>\]\[(){}]+")
ARXIV_ID_IN_TEXT_PATTERN = re.compile(
    r"(?i)\barxiv(?:\.org/(?:abs|pdf)/|\s*:\s*|\s+)(?P<arxiv_id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z.-]*/\d{7}(?:v\d+)?)"
)
TRAILING_PUNCTUATION = ".,;:"
CHARACTERS_OF_ENTRY_BEFORE_IDENTIFIER = 500  # bibliography entries usually end with their DOI
CHARACTERS_OF_ENTRY_AFTER_IDENTIFIER = 120
MAXIMUM_STORED_EXCERPT_CHARACTERS = 700  # the whole entry window: confirmation reads it
TITLE_WORD_SHARE_FOR_MATCH = 0.8
MINIMUM_TITLE_WORDS_FOR_WORD_SHARE = 3
MAXIMUM_IDENTIFIERS_PER_PAPER = 100


@dataclass(frozen=True)
class BibliographyIdentifier:
    identifier: NormalizedIdentifier  # version-free
    passage_id: str
    entry_text: str  # the text around the identifier (untrusted); used to confirm a provider record

    @property
    def stored_excerpt(self) -> str:
        return self.entry_text[-MAXIMUM_STORED_EXCERPT_CHARACTERS:]


def find_bibliography_identifiers(bibliography_passages: list[Passage]) -> list[BibliographyIdentifier]:
    """Every distinct DOI or arXiv id in the bibliography passages, in document order (bounded)."""
    found: dict[NormalizedIdentifier, BibliographyIdentifier] = {}
    for passage in bibliography_passages:
        for scheme, match_start, match_end, raw_value in _identifier_matches(passage.text):
            try:
                identifier = normalize_identifier(scheme, raw_value.rstrip(TRAILING_PUNCTUATION)).without_version()
            except SourceInputError:
                continue
            entry_text = passage.text[
                max(0, match_start - CHARACTERS_OF_ENTRY_BEFORE_IDENTIFIER) : match_end
                + CHARACTERS_OF_ENTRY_AFTER_IDENTIFIER
            ]
            found.setdefault(identifier, BibliographyIdentifier(identifier, passage.passage_id, entry_text))
            if len(found) >= MAXIMUM_IDENTIFIERS_PER_PAPER:
                return list(found.values())
    return list(found.values())


def _identifier_matches(text: str) -> list[tuple[str, int, int, str]]:
    matches = [(DOI_SCHEME, match.start(), match.end(), match.group()) for match in DOI_IN_TEXT_PATTERN.finditer(text)]
    matches += [
        (ARXIV_SCHEME, match.start(), match.end(), match["arxiv_id"])
        for match in ARXIV_ID_IN_TEXT_PATTERN.finditer(text)
    ]
    return sorted(matches, key=lambda found_match: found_match[1])


def provider_title_matches_entry(provider_title: str, entry_text: str) -> bool:
    """True if the provider record's title is recognizably the cited entry: its whole title key occurs in the entry,
    or (for titles of 3+ words) at least TITLE_WORD_SHARE_FOR_MATCH of its words do."""
    title_key = build_title_match_key(provider_title)
    entry_key = build_title_match_key(entry_text)
    if not title_key:
        return False
    if f" {title_key} " in f" {entry_key} ":
        return True
    title_words = title_key.split()
    if len(title_words) < MINIMUM_TITLE_WORDS_FOR_WORD_SHARE:
        return False
    entry_words = set(entry_key.split())
    return sum(word in entry_words for word in title_words) / len(title_words) >= TITLE_WORD_SHARE_FOR_MATCH
