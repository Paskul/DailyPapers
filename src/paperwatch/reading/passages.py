"""Turn a paper's text into passages with stable ids, page numbers, and section names (plan part 7).

Two kinds of text are read:
  * the abstract (abstract-only reading): one passage per sentence, ids A1, A2, ...; a title-only record gets one
    passage T1 holding the title, so the reviewer can say "insufficient evidence" while citing what it saw;
  * the text the sandboxed extractor produced (state/cache/sources/<edition>/<hash>.txt): pdftotext separates PDF
    pages with form feeds (\\f), so passages carry page numbers; the XML dumper marks headings with "## ".
    Body passages are ids P1, P2, ... in document order.
Ids depend only on the text and on PASSAGE_SEGMENTATION_VERSION, so the same saved edition always yields the same
passages: a review's evidence pointers stay valid, and validation can rebuild them from the cached file.

Section names come from heading lines (numbered "3.2 Monitor design", or known names such as "Results"). Each
section gets a kind (method, results, references, ...) that decides whether it is substantive (reading_plan.py).
Passages are untrusted text: they are data for the model and are escaped before they reach Markdown.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

PASSAGE_SEGMENTATION_VERSION = "passages-v1"  # bump when ids or boundaries could change for the same text
FRONT_MATTER_SECTION_TITLE = "Front matter"
ABSTRACT_SECTION_TITLE = "Abstract"
TITLE_SECTION_TITLE = "Title"
MAXIMUM_HEADING_CHARACTERS = 90
MAXIMUM_HEADING_WORDS = 12
XML_HEADING_PREFIX = "## "
FORM_FEED = "\f"


class SectionKind(StrEnum):
    """What a section is for. The reading plan reads method and results first and skips non-substantive kinds."""

    TITLE = "title"
    FRONT_MATTER = "front_matter"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHOD = "method"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    OTHER = "other"
    APPENDIX = "appendix"
    ACKNOWLEDGMENTS = "acknowledgments"
    REFERENCES = "references"


# Heading words -> kind. The first matching entry wins, so "Related work" is background, not "work".
SECTION_KIND_BY_HEADING_KEYWORD: tuple[tuple[str, SectionKind], ...] = (
    ("abstract", SectionKind.ABSTRACT),
    ("acknowledg", SectionKind.ACKNOWLEDGMENTS),
    ("funding", SectionKind.ACKNOWLEDGMENTS),
    ("author contribution", SectionKind.ACKNOWLEDGMENTS),
    ("conflict of interest", SectionKind.ACKNOWLEDGMENTS),
    ("competing interest", SectionKind.ACKNOWLEDGMENTS),
    ("references", SectionKind.REFERENCES),
    ("bibliography", SectionKind.REFERENCES),
    ("works cited", SectionKind.REFERENCES),
    ("appendix", SectionKind.APPENDIX),
    ("appendices", SectionKind.APPENDIX),
    ("supplementary", SectionKind.APPENDIX),
    ("supplemental", SectionKind.APPENDIX),
    ("introduction", SectionKind.INTRODUCTION),
    ("motivation", SectionKind.INTRODUCTION),
    ("related work", SectionKind.BACKGROUND),
    ("background", SectionKind.BACKGROUND),
    ("preliminar", SectionKind.BACKGROUND),
    ("prior work", SectionKind.BACKGROUND),
    ("literature", SectionKind.BACKGROUND),
    ("limitation", SectionKind.DISCUSSION),
    ("threats to validity", SectionKind.DISCUSSION),
    ("discussion", SectionKind.DISCUSSION),
    ("conclusion", SectionKind.CONCLUSION),
    ("concluding", SectionKind.CONCLUSION),
    ("summary", SectionKind.CONCLUSION),
    ("future work", SectionKind.CONCLUSION),
    ("result", SectionKind.RESULTS),
    ("experiment", SectionKind.RESULTS),
    ("evaluation", SectionKind.RESULTS),
    ("case stud", SectionKind.RESULTS),
    ("empirical", SectionKind.RESULTS),
    ("benchmark", SectionKind.RESULTS),
    ("analysis", SectionKind.RESULTS),
    ("findings", SectionKind.RESULTS),
    ("method", SectionKind.METHOD),
    ("approach", SectionKind.METHOD),
    ("model", SectionKind.METHOD),
    ("design", SectionKind.METHOD),
    ("algorithm", SectionKind.METHOD),
    ("framework", SectionKind.METHOD),
    ("formulation", SectionKind.METHOD),
    ("problem", SectionKind.METHOD),
    ("system", SectionKind.METHOD),
    ("architecture", SectionKind.METHOD),
    ("implementation", SectionKind.METHOD),
    ("proof", SectionKind.METHOD),
    ("theor", SectionKind.METHOD),
    ("setup", SectionKind.RESULTS),
    ("dataset", SectionKind.RESULTS),
)
# Unnumbered lines are headings only if they are one of these names (numbered headings need no list).
UNNUMBERED_HEADING_NAMES = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related work",
        "related works",
        "preliminaries",
        "method",
        "methods",
        "methodology",
        "approach",
        "our approach",
        "experiments",
        "experimental setup",
        "evaluation",
        "results",
        "experimental results",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "conclusion and future work",
        "conclusions and future work",
        "future work",
        "acknowledgments",
        "acknowledgements",
        "acknowledgment",
        "acknowledgement",
        "references",
        "bibliography",
        "appendix",
        "appendices",
        "supplementary material",
        "threats to validity",
        "case study",
        "funding",
        "summary",
    }
)
# "3", "3.2", "3.2.1.", "IV.", or an appendix letter WITH a dot ("A.", "B.1"), then a capitalized name. A bare
# "A " is not a number: "A Monte Carlo run ..." is the first line of a paragraph.
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?P<number>(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?)|(?:[IVX]{1,5}\.)|(?:[A-H]\.(?:\d{1,2}(?:\.\d{1,2})?\.?)?))\s+"
    r"(?P<name>[A-Z][^\n]*)$"
)
APPENDIX_LETTER_PATTERN = re.compile(r"^[A-H]\.")
# A numbered heading run into its paragraph (not a block of its own) must be this short and comma-free; otherwise
# it is more likely the first wrapped line of a paragraph that starts with a number.
MAXIMUM_WORDS_OF_HEADING_INSIDE_BLOCK = 6
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])")
PAGE_NUMBER_ONLY_PATTERN = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?$", re.IGNORECASE)
HYPHENATED_LINE_END_PATTERN = re.compile(r"[A-Za-z]-$")


@dataclass(frozen=True)
class Passage:
    passage_id: str  # A1.. (abstract), T1 (title-only record), P1.. (extracted text)
    page_number: int | None  # 1-based PDF page; None for an abstract or XML text
    section_title: str
    section_kind: SectionKind
    text: str

    def as_model_input(self) -> dict[str, object]:
        """The plain-data form the reviewer roles receive."""
        model_input: dict[str, object] = {
            "passage_id": self.passage_id,
            "section": self.section_title,
            "text": self.text,
        }
        if self.page_number is not None:
            model_input["page"] = self.page_number
        return model_input

    @property
    def location_label(self) -> str:
        """e.g. 'p. 4, 3.2 Monitor design' (rendered in notes; escaped there)."""
        return (f"p. {self.page_number}, " if self.page_number is not None else "") + self.section_title


# --------------------------------------------------------------------------------------------------------------------
# Abstract and title
# --------------------------------------------------------------------------------------------------------------------


def build_abstract_passages(title: str, abstract: str | None) -> tuple[Passage, ...]:
    """One passage per abstract sentence (A1..), or T1 = the title for a title-only record."""
    abstract_text = clean_passage_text(abstract or "")
    if not abstract_text:
        return (Passage("T1", None, TITLE_SECTION_TITLE, SectionKind.TITLE, clean_passage_text(title)),)
    sentences = [sentence for sentence in SENTENCE_BOUNDARY_PATTERN.split(abstract_text) if sentence.strip()]
    return tuple(
        Passage(f"A{sentence_index}", None, ABSTRACT_SECTION_TITLE, SectionKind.ABSTRACT, sentence.strip())
        for sentence_index, sentence in enumerate(sentences, start=1)
    )


# --------------------------------------------------------------------------------------------------------------------
# Extracted full text
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class _OpenSection:
    title: str
    kind: SectionKind
    top_level_kind: SectionKind  # a numbered subsection without its own keyword inherits its parent's kind


def segment_extracted_text(
    extracted_text: str, *, passage_max_characters: int, passage_min_characters: int
) -> tuple[Passage, ...]:
    """Passages P1..Pn of the extractor's text, in document order, with page numbers when pages are marked."""
    has_page_marks = FORM_FEED in extracted_text
    raw_paragraphs: list[tuple[int | None, _OpenSection, str]] = []
    open_section = _OpenSection(FRONT_MATTER_SECTION_TITLE, SectionKind.FRONT_MATTER, SectionKind.FRONT_MATTER)
    for page_index, page_text in enumerate(extracted_text.split(FORM_FEED)):
        page_number = page_index + 1 if has_page_marks else None
        for block_text in re.split(r"\n\s*\n", page_text):
            block_lines = [line.strip() for line in block_text.splitlines() if line.strip()]
            # A heading is the first line of a block (pdftotext sometimes runs it into the paragraph that follows).
            while block_lines:
                heading = recognize_heading_line(block_lines[0], is_whole_block=len(block_lines) == 1)
                if heading is None:
                    break
                open_section = _open_section_for_heading(heading, open_section)
                block_lines.pop(0)
            paragraph_text = clean_passage_text(join_extracted_lines(block_lines))
            if not paragraph_text or PAGE_NUMBER_ONLY_PATTERN.match(paragraph_text):
                continue
            raw_paragraphs.append((page_number, open_section, paragraph_text))

    bounded_paragraphs = _merge_and_split_paragraphs(raw_paragraphs, passage_max_characters, passage_min_characters)
    return tuple(
        Passage(f"P{passage_index}", page_number, section.title, section.kind, text)
        for passage_index, (page_number, section, text) in enumerate(bounded_paragraphs, start=1)
    )


@dataclass(frozen=True)
class RecognizedHeading:
    title: str  # as printed, e.g. "3.2 Monitor design"
    numbering_depth: int  # 1 for "3", 2 for "3.2"; 0 when unnumbered
    is_appendix_lettered: bool  # "A. Proofs", "B.1 Extra results"


def recognize_heading_line(line: str, *, is_whole_block: bool = True) -> RecognizedHeading | None:
    """A heading line, or None. Deliberately conservative: a missed heading only coarsens the section names, while
    a false one would split a paragraph. `is_whole_block` is False when more lines of the same block follow."""
    if line.startswith(XML_HEADING_PREFIX):
        heading_title = clean_passage_text(line[len(XML_HEADING_PREFIX) :])
        return RecognizedHeading(heading_title, 0, False) if heading_title else None
    candidate = clean_passage_text(line)
    if not candidate or len(candidate) > MAXIMUM_HEADING_CHARACTERS or len(candidate.split()) > MAXIMUM_HEADING_WORDS:
        return None
    if candidate.rstrip(":").casefold() in UNNUMBERED_HEADING_NAMES:
        return RecognizedHeading(candidate.rstrip(":"), 0, False)
    numbered_match = NUMBERED_HEADING_PATTERN.match(candidate)
    if numbered_match is None or candidate.endswith((".", ",", ";")):
        return None
    heading_name = numbered_match.group("name")
    if sum(character.isdigit() for character in heading_name) > 3 or "=" in heading_name:
        return None  # "3 Robots in 12 rooms", equations, or table rows
    if not is_whole_block and (len(candidate.split()) > MAXIMUM_WORDS_OF_HEADING_INSIDE_BLOCK or "," in heading_name):
        return None
    number_text = numbered_match.group("number").rstrip(".")
    is_appendix_lettered = bool(APPENDIX_LETTER_PATTERN.match(numbered_match.group("number")))
    return RecognizedHeading(candidate, number_text.count(".") + 1, is_appendix_lettered)


def classify_section_heading(heading_title: str) -> SectionKind | None:
    """The kind named by a heading's words, or None when the words say nothing (e.g. '3 Monitoring joint plans')."""
    lowered_title = heading_title.casefold()
    for heading_keyword, section_kind in SECTION_KIND_BY_HEADING_KEYWORD:
        if heading_keyword in lowered_title:
            return section_kind
    return None


def _open_section_for_heading(heading: RecognizedHeading, previous_section: _OpenSection) -> _OpenSection:
    named_kind = classify_section_heading(heading.title)
    if heading.is_appendix_lettered and previous_section.top_level_kind in (
        SectionKind.APPENDIX,
        SectionKind.REFERENCES,
    ):
        named_kind = SectionKind.APPENDIX  # lettered sections after the references are appendices
    if heading.numbering_depth >= 2:  # a subsection: its own keyword, else its parent's kind
        section_kind = named_kind or previous_section.top_level_kind
        return _OpenSection(heading.title, section_kind, previous_section.top_level_kind)
    section_kind = named_kind or SectionKind.OTHER
    return _OpenSection(heading.title, section_kind, section_kind)


def _merge_and_split_paragraphs(
    raw_paragraphs: list[tuple[int | None, _OpenSection, str]],
    passage_max_characters: int,
    passage_min_characters: int,
) -> list[tuple[int | None, _OpenSection, str]]:
    """Merge short paragraphs into the next one on the same page and section; split long ones at sentences."""
    merged: list[tuple[int | None, _OpenSection, str]] = []
    for page_number, section, paragraph_text in raw_paragraphs:
        if merged:
            previous_page, previous_section, previous_text = merged[-1]
            if (
                previous_page == page_number
                and previous_section is section
                and len(previous_text) < passage_min_characters
                and len(previous_text) + 1 + len(paragraph_text) <= passage_max_characters
            ):
                merged[-1] = (page_number, section, f"{previous_text} {paragraph_text}")
                continue
        merged.append((page_number, section, paragraph_text))
    bounded: list[tuple[int | None, _OpenSection, str]] = []
    for page_number, section, paragraph_text in merged:
        bounded.extend(
            (page_number, section, piece) for piece in split_text_at_sentences(paragraph_text, passage_max_characters)
        )
    return bounded


def split_text_at_sentences(text: str, maximum_characters: int) -> list[str]:
    """Pieces of at most `maximum_characters`, cut at sentence ends where possible (a very long sentence is cut at a
    word boundary)."""
    if len(text) <= maximum_characters:
        return [text]
    pieces: list[str] = []
    current_piece = ""
    for sentence in SENTENCE_BOUNDARY_PATTERN.split(text):
        while len(sentence) > maximum_characters:
            cut_index = sentence.rfind(" ", 0, maximum_characters)
            cut_index = cut_index if cut_index > maximum_characters // 2 else maximum_characters
            if current_piece:
                pieces.append(current_piece)
                current_piece = ""
            pieces.append(sentence[:cut_index].strip())
            sentence = sentence[cut_index:].strip()
        if current_piece and len(current_piece) + 1 + len(sentence) > maximum_characters:
            pieces.append(current_piece)
            current_piece = sentence
        else:
            current_piece = f"{current_piece} {sentence}".strip()
    if current_piece:
        pieces.append(current_piece)
    return pieces


def join_extracted_lines(lines: list[str]) -> str:
    """Join a paragraph's lines, undoing end-of-line hyphenation ("moni-" + "tor" -> "monitor")."""
    joined_text = ""
    for line in lines:
        if joined_text and HYPHENATED_LINE_END_PATTERN.search(joined_text) and line[:1].islower():
            joined_text = joined_text[:-1] + line
        else:
            joined_text = f"{joined_text} {line}" if joined_text else line
    return joined_text


def clean_passage_text(text: str) -> str:
    """NFKC (ligatures such as 'ﬁ' become 'fi'), soft hyphens and control characters removed, whitespace collapsed."""
    normalized_text = unicodedata.normalize("NFKC", text).replace("­", "")
    printable_text = "".join(
        character if character.isprintable() or character.isspace() else " " for character in normalized_text
    )
    return " ".join(printable_text.split())
