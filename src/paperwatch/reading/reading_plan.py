"""Cut a paper's passages into bounded reading sections, choose which ones to read, and decide the coverage label.

A reading section is one model call's worth of passages: at most `reading.section_max_characters`, so the prompt
leaves room in the context window for reasoning and the final answer (checked in config.py and again before each
call in roles/base.py). Sections are built in document order; a document section that does not fit into the
remainder of the current reading section starts a new one, so short sections stay whole.

Which sections are read (plan part 7: "use the title and abstract for orientation, then read methods, results, and
limitations"): every reading section gets the priority of the most important passage kind it holds,

    0 method, results       1 discussion / limitations, conclusion       2 introduction, other
    3 background, abstract  4 front matter                               5 appendix

and the best `reading.max_sections_per_paper` are read, in document order. References and acknowledgments are
never read.

Coverage: `full_text` only if every reading section with substantive content was read AND the text has a closing
section (conclusion, discussion, references, acknowledgments, or an appendix), so an extraction that stops early is
not mistaken for a complete paper. Otherwise `partial_text`, with the reasons recorded. Front matter and appendices
are read when the limit allows, but are not required for `full_text`. An abstract reading is `abstract_only`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from paperwatch.reading.passages import PASSAGE_SEGMENTATION_VERSION, Passage, SectionKind

READING_PLAN_VERSION = "reading-plan-v1"  # bump when sectioning or selection rules change

COVERAGE_FULL_TEXT = "full_text"
COVERAGE_PARTIAL_TEXT = "partial_text"
COVERAGE_ABSTRACT_ONLY = "abstract_only"

READING_PRIORITY_BY_SECTION_KIND: dict[SectionKind, int] = {
    SectionKind.METHOD: 0, SectionKind.RESULTS: 0,
    SectionKind.DISCUSSION: 1, SectionKind.CONCLUSION: 1,
    SectionKind.INTRODUCTION: 2, SectionKind.OTHER: 2,
    SectionKind.BACKGROUND: 3, SectionKind.ABSTRACT: 3, SectionKind.TITLE: 3,
    SectionKind.FRONT_MATTER: 4,
    SectionKind.APPENDIX: 5,
}
NEVER_READ_SECTION_KINDS = frozenset({SectionKind.REFERENCES, SectionKind.ACKNOWLEDGMENTS})
OPTIONAL_SECTION_KINDS = frozenset({SectionKind.FRONT_MATTER, SectionKind.APPENDIX})  # not needed for full_text
CLOSING_SECTION_KINDS = frozenset({SectionKind.CONCLUSION, SectionKind.DISCUSSION, SectionKind.REFERENCES,
                                   SectionKind.ACKNOWLEDGMENTS, SectionKind.APPENDIX})
PER_PASSAGE_PROMPT_OVERHEAD_CHARACTERS = 60  # JSON keys, id, page, and section name around each passage


@dataclass(frozen=True)
class ReadingSection:
    section_number: int  # 1-based, document order
    passages: tuple[Passage, ...]
    reading_priority: int
    has_required_content: bool  # holds a passage whose kind full_text requires

    @property
    def stage_key(self) -> str:
        """Key of this section's saved result (paperwatch.reading.stage_results)."""
        return f"section:{self.section_number:03d}"

    @property
    def section_titles(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(passage.section_title for passage in self.passages))

    @property
    def passage_ids(self) -> tuple[str, ...]:
        return tuple(passage.passage_id for passage in self.passages)

    @property
    def character_count(self) -> int:
        return sum(len(passage.text) + PER_PASSAGE_PROMPT_OVERHEAD_CHARACTERS for passage in self.passages)


@dataclass(frozen=True)
class ReadingPlan:
    coverage: str
    all_sections: tuple[ReadingSection, ...]
    sections_to_read: tuple[ReadingSection, ...]  # document order
    skipped_section_titles: tuple[str, ...]  # never read: references, acknowledgments
    partial_reasons: tuple[str, ...]  # why the coverage is partial_text (empty otherwise)
    plan_sha256: str

    @property
    def is_single_pass(self) -> bool:
        """One reading section: the reviewer reads the passages directly, without separate section notes."""
        return len(self.sections_to_read) == 1

    @property
    def unread_sections(self) -> tuple[ReadingSection, ...]:
        read_numbers = {section.section_number for section in self.sections_to_read}
        return tuple(section for section in self.all_sections if section.section_number not in read_numbers)

    @property
    def read_passage_ids(self) -> frozenset[str]:
        return frozenset(passage_id for section in self.sections_to_read for passage_id in section.passage_ids)

    def to_record(self) -> dict[str, object]:
        """What the review stores about its reading: which sections were read and why coverage is what it is."""
        return {
            "plan_version": READING_PLAN_VERSION,
            "segmentation_version": PASSAGE_SEGMENTATION_VERSION,
            "coverage": self.coverage,
            "sections_read": [{"section_number": section.section_number, "titles": list(section.section_titles),
                               "passages": f"{section.passage_ids[0]}-{section.passage_ids[-1]}"}
                              for section in self.sections_to_read],
            "sections_not_read": [{"section_number": section.section_number,
                                   "titles": list(section.section_titles)} for section in self.unread_sections],
            "skipped_not_substantive": list(self.skipped_section_titles),
            "partial_reasons": list(self.partial_reasons),
        }


def plan_abstract_reading(passages: tuple[Passage, ...]) -> ReadingPlan:
    reading_section = ReadingSection(1, passages, 0, True)
    return ReadingPlan(COVERAGE_ABSTRACT_ONLY, (reading_section,), (reading_section,), (), (),
                       _plan_sha256(COVERAGE_ABSTRACT_ONLY, (reading_section,), (reading_section,), 0, 0))


def plan_full_text_reading(passages: tuple[Passage, ...], *, section_max_characters: int,
                           max_sections_per_paper: int) -> ReadingPlan:
    if passages and all(passage.section_kind == SectionKind.FRONT_MATTER for passage in passages):
        has_recognized_structure = False  # no heading was recognized: everything is required, completeness unknown
    else:
        has_recognized_structure = True
    readable_passages = [passage for passage in passages if passage.section_kind not in NEVER_READ_SECTION_KINDS]
    skipped_titles = tuple(dict.fromkeys(passage.section_title for passage in passages
                                         if passage.section_kind in NEVER_READ_SECTION_KINDS))
    all_sections = _build_reading_sections(readable_passages, section_max_characters, has_recognized_structure)
    chosen_sections = sorted(all_sections, key=lambda section: (section.reading_priority,
                                                                section.section_number))[:max_sections_per_paper]
    sections_to_read = tuple(sorted(chosen_sections, key=lambda section: section.section_number))

    partial_reasons: list[str] = []
    read_numbers = {section.section_number for section in sections_to_read}
    unread_required = [section for section in all_sections
                       if section.has_required_content and section.section_number not in read_numbers]
    if unread_required:
        partial_reasons.append(f"{len(unread_required)} reading section(s) with substantive content not read "
                               f"(limit: {max_sections_per_paper} per paper)")
    if not has_recognized_structure:
        partial_reasons.append("no section headings recognized, so completeness cannot be confirmed")
    elif not any(passage.section_kind in CLOSING_SECTION_KINDS for passage in passages):
        partial_reasons.append("no closing section (conclusion, discussion, references) found: the extraction may "
                               "be truncated")
    if not sections_to_read:
        partial_reasons.append("the extracted text holds no readable passages")
    coverage = COVERAGE_PARTIAL_TEXT if partial_reasons else COVERAGE_FULL_TEXT
    return ReadingPlan(coverage, all_sections, sections_to_read, skipped_titles, tuple(partial_reasons),
                       _plan_sha256(coverage, all_sections, sections_to_read, section_max_characters,
                                    max_sections_per_paper))


def _build_reading_sections(readable_passages: list[Passage], section_max_characters: int,
                            has_recognized_structure: bool) -> tuple[ReadingSection, ...]:
    document_sections = _group_consecutive_passages_by_section(readable_passages)
    reading_sections: list[list[Passage]] = [[]]
    for document_section in document_sections:
        section_characters = sum(len(passage.text) + PER_PASSAGE_PROMPT_OVERHEAD_CHARACTERS
                                 for passage in document_section)
        current_characters = _characters(reading_sections[-1])
        if reading_sections[-1] and current_characters + section_characters > section_max_characters \
                and section_characters <= section_max_characters:
            reading_sections.append([])  # keep a section that fits a fresh reading section whole
        for passage in document_section:
            passage_characters = len(passage.text) + PER_PASSAGE_PROMPT_OVERHEAD_CHARACTERS
            if reading_sections[-1] and _characters(reading_sections[-1]) + passage_characters > section_max_characters:
                reading_sections.append([])
            reading_sections[-1].append(passage)
    return tuple(ReadingSection(
        section_number=section_index, passages=tuple(section_passages),
        reading_priority=min(_reading_priority(passage, has_recognized_structure) for passage in section_passages),
        has_required_content=any(_is_required(passage, has_recognized_structure) for passage in section_passages))
        for section_index, section_passages in enumerate((group for group in reading_sections if group), start=1))


def _group_consecutive_passages_by_section(passages: list[Passage]) -> list[list[Passage]]:
    groups: list[list[Passage]] = []
    for passage in passages:
        if groups and groups[-1][-1].section_title == passage.section_title and \
                groups[-1][-1].section_kind == passage.section_kind:
            groups[-1].append(passage)
        else:
            groups.append([passage])
    return groups


def _reading_priority(passage: Passage, has_recognized_structure: bool) -> int:
    if not has_recognized_structure:
        return READING_PRIORITY_BY_SECTION_KIND[SectionKind.OTHER]
    return READING_PRIORITY_BY_SECTION_KIND.get(passage.section_kind, READING_PRIORITY_BY_SECTION_KIND[SectionKind.OTHER])


def _is_required(passage: Passage, has_recognized_structure: bool) -> bool:
    return not has_recognized_structure or passage.section_kind not in OPTIONAL_SECTION_KINDS


def _characters(passages: list[Passage]) -> int:
    return sum(len(passage.text) + PER_PASSAGE_PROMPT_OVERHEAD_CHARACTERS for passage in passages)


def _plan_sha256(coverage: str, all_sections: tuple[ReadingSection, ...], sections_to_read: tuple[ReadingSection, ...],
                 section_max_characters: int, max_sections_per_paper: int) -> str:
    """Saved section results are reused only under the same plan, so changed limits never mix two cuttings."""
    plan_fields = {
        "plan_version": READING_PLAN_VERSION, "segmentation_version": PASSAGE_SEGMENTATION_VERSION,
        "coverage": coverage, "limits": [section_max_characters, max_sections_per_paper],
        "sections": [list(section.passage_ids) for section in all_sections],
        "read": [section.section_number for section in sections_to_read],
    }
    return hashlib.sha256(json.dumps(plan_fields, sort_keys=True).encode("utf-8")).hexdigest()
