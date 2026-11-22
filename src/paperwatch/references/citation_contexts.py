"""Where the parent paper's text cites a given work: the "citation contexts" the research role reads (plan part 8).

OpenAlex supplies reference ids but no contexts, so they come from the parent's own extracted text:
  1. find the cited work's bibliography entry: a bibliography passage holding one of the work's identifiers, else the
     first words of its title (punctuation and case ignored);
  2. read the entry's marker: a numeric label "[12]" just before it; otherwise author-year style, matched by the
     first author's family name followed closely by the year ("Smith et al., 2019", "Smith (2019)");
  3. collect up to MAXIMUM_CONTEXTS_PER_WORK body sentences that carry the marker ("[3, 12]" and "[10-14]" count).
Each context keeps the id of the body passage it came from, so the research role can point at it and the host can
check that pointer. Paper text is untrusted: contexts are bounded and handed to the model as plain data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from paperwatch.reading.passages import Passage

MAXIMUM_CONTEXTS_PER_WORK = 2
MAXIMUM_CONTEXT_CHARACTERS = 320
TITLE_WORDS_USED_TO_FIND_ENTRY = 6
CHARACTERS_SEARCHED_BEFORE_ENTRY_FOR_LABEL = 400
MAXIMUM_CHARACTERS_BETWEEN_NAME_AND_YEAR = 40
NUMERIC_LABEL_PATTERN = re.compile(r"\[(\d{1,3})\]")
BRACKET_GROUP_PATTERN = re.compile(r"\[([\d\s,;–-]{1,60})\]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"“])")
WORD_PATTERN = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class CitedWorkDescription:
    """What is known about a cited work from the catalog (trusted metadata, not the parent's text)."""

    title: str
    first_author_family_name: str | None
    publication_year: str | None
    identifier_values: tuple[str, ...]  # e.g. a DOI or arXiv id, lowercase


@dataclass(frozen=True)
class CitationContext:
    passage_id: str
    text: str

    def as_model_input(self) -> dict[str, str]:
        return {"passage_id": self.passage_id, "text": self.text}


def find_citation_contexts(
    bibliography: list[Passage], body: list[Passage], cited_work: CitedWorkDescription
) -> list[CitationContext]:
    entry_location = locate_bibliography_entry(bibliography, cited_work)
    if entry_location is None:
        return []
    entry_passage, entry_start = entry_location
    numeric_label = _numeric_label_before(entry_passage.text, entry_start)
    contexts: list[CitationContext] = []
    for passage in body:
        for sentence in SENTENCE_SPLIT_PATTERN.split(passage.text):
            if _sentence_cites(sentence, numeric_label, cited_work):
                contexts.append(CitationContext(passage.passage_id, _bounded(sentence)))
                break  # one context per passage is enough
        if len(contexts) >= MAXIMUM_CONTEXTS_PER_WORK:
            break
    return contexts


def locate_bibliography_entry(
    bibliography: list[Passage], cited_work: CitedWorkDescription
) -> tuple[Passage, int] | None:
    """(passage, character offset of the entry's identifier or title) or None."""
    for passage in bibliography:
        lowered_text = passage.text.lower()
        for identifier_value in cited_work.identifier_values:
            identifier_offset = lowered_text.find(identifier_value.lower())
            if identifier_offset >= 0:
                return passage, _start_of_entry_holding(passage.text, identifier_offset)
    title_pattern = _title_search_pattern(cited_work.title)
    if title_pattern is None:
        return None
    for passage in bibliography:
        title_match = title_pattern.search(passage.text)
        if title_match is not None:
            return passage, title_match.start()
    return None


def _start_of_entry_holding(text: str, offset: int) -> int:
    """For an identifier found at `offset`, the start of its entry: just after the previous numeric label."""
    preceding_labels = list(NUMERIC_LABEL_PATTERN.finditer(text, 0, offset))
    return preceding_labels[-1].end() if preceding_labels else offset


def _title_search_pattern(title: str) -> re.Pattern | None:
    title_words = WORD_PATTERN.findall(title)[:TITLE_WORDS_USED_TO_FIND_ENTRY]
    if len(title_words) < 2:
        return None
    return re.compile(r"\W+".join(re.escape(word) for word in title_words), re.IGNORECASE)


def _numeric_label_before(text: str, entry_start: int) -> str | None:
    search_start = max(0, entry_start - CHARACTERS_SEARCHED_BEFORE_ENTRY_FOR_LABEL)
    labels = list(NUMERIC_LABEL_PATTERN.finditer(text, search_start, entry_start))
    return labels[-1].group(1) if labels else None


def _sentence_cites(sentence: str, numeric_label: str | None, cited_work: CitedWorkDescription) -> bool:
    if numeric_label is not None:
        return any(
            int(numeric_label) in _numbers_in_bracket_group(group.group(1))
            for group in BRACKET_GROUP_PATTERN.finditer(sentence)
        )
    family_name = cited_work.first_author_family_name
    if not family_name or not cited_work.publication_year:
        return False
    author_year_pattern = re.compile(
        rf"\b{re.escape(family_name)}\b.{{0,{MAXIMUM_CHARACTERS_BETWEEN_NAME_AND_YEAR}}}?\b"
        rf"{re.escape(cited_work.publication_year)}\b",
        re.IGNORECASE,
    )
    return author_year_pattern.search(sentence) is not None


def _numbers_in_bracket_group(group_text: str) -> set[int]:
    """'3, 12' -> {3, 12}; '10-14' -> {10..14}. Ranges are bounded so a garbled group cannot explode."""
    numbers: set[int] = set()
    for token in re.split(r"[,;]", group_text):
        bounds = [int(part) for part in re.split(r"[–-]", token.strip()) if part.strip().isdigit()]
        if len(bounds) == 1:
            numbers.add(bounds[0])
        elif len(bounds) == 2 and 0 <= bounds[1] - bounds[0] <= 50:
            numbers.update(range(bounds[0], bounds[1] + 1))
    return numbers


def _bounded(sentence: str) -> str:
    sentence = " ".join(sentence.split())
    if len(sentence) <= MAXIMUM_CONTEXT_CHARACTERS:
        return sentence
    return sentence[: MAXIMUM_CONTEXT_CHARACTERS - 1] + "…"
