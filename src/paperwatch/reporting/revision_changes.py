"""Does a newer review of an already reported paper change a result, an assumption, or a conclusion?

"Include a previously reported paper again only for a revision that changes a result, assumption, or conclusion."
The selection is frozen before the writer runs, so this is decided by a deterministic rule, without a model call
(owner decision 2026-11-20). A newer review counts as an UPDATE only if it was read from DIFFERENT text than the
reported review (a new version, or full text after the abstract; a plain re-read of the same text never qualifies)
AND at least one of these changed:
    numeric_results        the set of numbers in the result claims, their conditions, and their quoted excerpts
    assumptions            a supported assumption appears or disappears (word overlap, see ASSUMPTION_MATCH_OVERLAP)
    evidence_verdict       the reviewer's overall "insufficient evidence" verdict flipped
    reading_priority       the best topic score moved by at least READING_PRIORITY_CHANGE_FOR_UPDATE points
The descriptions are written here, by the host, from validated fields (numbers were checked against the cited
passages when each review was saved). The rule cannot see a change that is purely verbal; the report links both
notes so the reader can check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from paperwatch.comparisons.evidence_dimensions import content_words
from paperwatch.reading.evidence_checks import find_numbers_needing_evidence

READING_PRIORITY_CHANGE_FOR_UPDATE = 10  # points on the 0-100 scale
ASSUMPTION_MATCH_OVERLAP = 0.5  # shared words / words of the smaller statement, for "the same assumption"
ASSUMPTION_MATCH_MINIMUM_SHARED_WORDS = 2
MAXIMUM_LISTED_NUMBERS = 6
GENERATION_SUFFIX_MARKER = ":generation:"


@dataclass(frozen=True)
class RevisionChange:
    change_kind: str  # numeric_results | assumptions | evidence_verdict | reading_priority
    description: str  # host-written; numbers in it come from validated review fields

    def to_record(self) -> dict[str, str]:
        return {"change_kind": self.change_kind, "description": self.description}

    @classmethod
    def from_record(cls, record: dict[str, str]) -> RevisionChange:
        return cls(record["change_kind"], record["description"])


@dataclass(frozen=True)
class ReviewedVersion:
    """What the rule compares about one review."""

    review_content: dict[str, Any]
    text_fingerprint: str  # the normalized source-text hash the review read
    best_reading_priority: int | None


def text_fingerprint_of_review(source_text_sha256: str | None, review_input_key: str) -> str:
    """The source-text hash; older rows without it fall back to the input key minus its reading generation."""
    return source_text_sha256 or review_input_key.split(GENERATION_SUFFIX_MARKER)[0]


def describe_material_revision_changes(reported: ReviewedVersion, newer: ReviewedVersion) -> list[RevisionChange]:
    """The changes that make `newer` an update of the `reported` review; an empty list means "not an update"."""
    if reported.text_fingerprint == newer.text_fingerprint:
        return []
    changes: list[RevisionChange] = []

    reported_numbers, newer_numbers = _result_numbers(reported.review_content), _result_numbers(newer.review_content)
    if reported_numbers != newer_numbers:
        added = sorted(newer_numbers - reported_numbers)[:MAXIMUM_LISTED_NUMBERS]
        removed = sorted(reported_numbers - newer_numbers)[:MAXIMUM_LISTED_NUMBERS]
        changes.append(
            RevisionChange(
                "numeric_results",
                "numbers in the reported results changed"
                + (f"; now also {', '.join(added)}" if added else "")
                + (f"; no longer {', '.join(removed)}" if removed else ""),
            )
        )

    reported_assumptions = _supported_assumption_word_sets(reported.review_content)
    newer_assumptions = _supported_assumption_word_sets(newer.review_content)
    appeared = sum(1 for words in newer_assumptions if not _has_matching_statement(words, reported_assumptions))
    disappeared = sum(1 for words in reported_assumptions if not _has_matching_statement(words, newer_assumptions))
    if appeared or disappeared:
        changes.append(
            RevisionChange(
                "assumptions", f"supported assumptions changed: {appeared} new, {disappeared} no longer stated"
            )
        )

    reported_insufficient = bool(reported.review_content.get("insufficient_evidence"))
    newer_insufficient = bool(newer.review_content.get("insufficient_evidence"))
    if reported_insufficient != newer_insufficient:
        changes.append(
            RevisionChange(
                "evidence_verdict",
                "the reviewer now finds the evidence "
                + ("insufficient" if newer_insufficient else "sufficient for a review"),
            )
        )

    if (
        reported.best_reading_priority is not None
        and newer.best_reading_priority is not None
        and abs(newer.best_reading_priority - reported.best_reading_priority) >= READING_PRIORITY_CHANGE_FOR_UPDATE
    ):
        changes.append(
            RevisionChange(
                "reading_priority",
                f"reading priority moved from {reported.best_reading_priority} to {newer.best_reading_priority}/100",
            )
        )
    return changes


def _result_numbers(review_content: dict[str, Any]) -> set[str]:
    """Numbers stated in the results: evidence-review findings (claim, conditions, excerpt) or legacy main-claim strings."""
    result_texts: list[str] = []
    for finding in review_content.get("results", []):
        if isinstance(finding, dict):
            result_texts += [str(finding.get(name, "")) for name in ("claim", "conditions", "excerpt")]
    result_texts += [str(claim) for claim in review_content.get("authors_main_claims", [])]
    return {number for text in result_texts for number in find_numbers_needing_evidence(text)}


def _supported_assumption_word_sets(review_content: dict[str, Any]) -> list[set[str]]:
    return [
        content_words(str(statement.get("statement", "")))
        for statement in review_content.get("assumptions", [])
        if isinstance(statement, dict) and statement.get("evidence_status") == "supported"
    ]


def _has_matching_statement(words: set[str], candidates: list[set[str]]) -> bool:
    for candidate_words in candidates:
        if words == candidate_words:
            return True
        common_words = words & candidate_words
        smaller_size = min(len(words), len(candidate_words))
        if (
            smaller_size
            and len(common_words) >= min(ASSUMPTION_MATCH_MINIMUM_SHARED_WORDS, smaller_size)
            and len(common_words) / smaller_size >= ASSUMPTION_MATCH_OVERLAP
        ):
            return True
    return False
