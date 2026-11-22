"""Render the permanent per-paper reading note from validated, saved data (the daily report: daily_report.py).

Notes live at output/papers/<paper_id>/<review_id>.md: a stable, revision-specific file named only by internal ids.
A part-7 note shows (plan part 7: "render the permanent note from validated fields and trusted metadata"):
  * trusted metadata: source links built from identifiers, version, coverage, review date, model and prompt;
  * the review's fields, each with evidence pointers (passage id, page, section) and the quoted excerpt;
  * the scoring breakdown per topic: each rating with its weight and explanation, "comparison unavailable" when the
    value dimension was omitted, the evidence cap for abstract-only reviews, matched seeds, compared reviews;
  * the reading coverage: sections read and not read, why the coverage is partial, table/equation recovery;
  * the cited passages themselves, from the saved review (not the cache), so the note can always be rebuilt.
All paper and model text is escaped (rendering/markdown_safety.py). Reviews saved before part 7 keep their older,
simpler layout.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from paperwatch.reading.reading_priority import RUBRIC_DIMENSIONS, SCORE_SCALE_PRIORITY_0_TO_100
from paperwatch.rendering.markdown_safety import build_trusted_source_url, escape_untrusted_inline_text
from paperwatch.review_store import SCORE_SCALE_RATING_0_TO_4, EvidencePassage, TopicAssessment

MAXIMUM_RENDERED_PASSAGE_CHARACTERS = 700
SCORE_MEANING_NOTE = (
    "Scores are a reading priority for your research question, computed from the reviewer's "
    "ratings; they are not a judgement of the paper's novelty in its field."
)
ATTRIBUTION_NOTE = (
    "Each excerpt was checked to occur verbatim in the cited passage of the saved edition. That "
    "shows attribution; it cannot show that the interpretation is sound."
)


@dataclass(frozen=True)
class SelectedReview:
    """One completed review chosen for a report (or a note), with the trusted metadata needed to render it."""

    rank: int
    paper_row: sqlite3.Row
    review_row: sqlite3.Row
    review_content: dict[str, Any]
    paper_aliases: list[tuple[str, str]]
    topic_assessments: list[TopicAssessment]  # newest per topic, stored apart from the reading
    evidence_passages: dict[str, EvidencePassage] = field(default_factory=dict)
    seed_titles_by_id: dict[str, str] = field(default_factory=dict)

    @property
    def best_topic_assessment(self) -> TopicAssessment | None:
        """Highest reading priority; ties go to the lowest topic id so ordering is deterministic."""
        return min(
            self.topic_assessments,
            key=lambda assessment: (-assessment.priority_score_out_of_100, assessment.topic_id),
            default=None,
        )

    @property
    def reading_record(self) -> dict[str, Any]:
        return json.loads(self.review_row["reading_record_json"] or "{}")

    @property
    def is_evidence_review(self) -> bool:
        """Part-7 review (statements with passage ids), as opposed to the provisional step-1 layout."""
        return isinstance(self.review_content.get("method"), dict)

    @property
    def note_relative_path_from_daily_dir(self) -> str:
        return f"../papers/{self.paper_row['paper_id']}/{self.review_row['review_id']}.md"


# --------------------------------------------------------------------------------------------------------------------
# The permanent note
# --------------------------------------------------------------------------------------------------------------------


def render_paper_note(selected_review: SelectedReview) -> str:
    if not selected_review.is_evidence_review:
        return _render_legacy_paper_note(selected_review)
    paper_row, review_row, content = (
        selected_review.paper_row,
        selected_review.review_row,
        selected_review.review_content,
    )
    reading_record = selected_review.reading_record
    best_assessment = selected_review.best_topic_assessment
    note_lines = [
        f"# {escape_untrusted_inline_text(paper_row['title'])}",
        "",
        *_render_metadata_lines(selected_review),
        f"- Reading priority: {_describe_best_score(best_assessment)} · coverage **{review_row['coverage']}**"
        + (" (partially read)" if review_row["coverage"] == "partial_text" else ""),
        "",
        f"> {SCORE_MEANING_NOTE}",
        "",
        "## Summary",
        "",
        escape_untrusted_inline_text(content["one_sentence_summary"]) or "No summary.",
        "",
    ]
    if content["insufficient_evidence"]:
        note_lines += ["> The reviewer marked the available text as insufficient evidence for a full review.", ""]
    for heading, field_name in (
        ("Research question", "research_question"),
        ("Method", "method"),
        ("Evaluation setting", "evaluation_setting"),
    ):
        note_lines += [f"## {heading}", "", _render_statement(content[field_name], selected_review), ""]
    note_lines += ["## Assumptions", ""]
    note_lines += [f"- {_render_statement(statement, selected_review)}" for statement in content["assumptions"]] or [
        "- None stated."
    ]
    note_lines += ["", "## Results (the authors' claims)", ""]
    note_lines += [_render_finding(finding, selected_review) for finding in content["results"]] or [
        "- No result with supporting evidence."
    ]
    note_lines += ["", "## Limitations", ""]
    note_lines += [
        f"- ({'stated by the authors' if limitation['stated_by'] == 'authors' else 'reviewer inference'})"
        f" {escape_untrusted_inline_text(limitation['statement'])}"
        f"{_render_pointers(limitation['supporting_passage_ids'], selected_review)}"
        for limitation in content["limitations"]
    ] or ["- None stated."]
    if content["reviewer_inferences"]:
        note_lines += ["", "## Reviewer inferences (not the authors' claims)", ""]
        note_lines += [
            f"- {escape_untrusted_inline_text(inference['inference'])}"
            f"{_render_pointers(inference['supporting_passage_ids'], selected_review)}"
            for inference in content["reviewer_inferences"]
        ]
    if content["ambiguous_passages"]:
        note_lines += ["", "## Ambiguous passages", ""]
        note_lines += [
            f"- {_render_pointer(item['passage_id'], selected_review)} {escape_untrusted_inline_text(item['issue'])}"
            for item in content["ambiguous_passages"]
        ]
    note_lines += ["", "## Scores by topic", ""]
    note_lines += _render_topic_scores(selected_review)
    note_lines += ["", "## Reading coverage", ""]
    note_lines += _render_reading_coverage(reading_record, review_row["coverage"])
    note_lines += ["", "## Evidence passages", "", f"> {ATTRIBUTION_NOTE}", ""]
    note_lines += [
        f"- `{passage.passage_id}` ({escape_untrusted_inline_text(passage.location_label)}): "
        f"{escape_untrusted_inline_text(_shorten(passage.passage_text))}"
        for passage in _passages_in_id_order(selected_review.evidence_passages)
    ] or ["- No passage was cited."]
    return "\n".join(note_lines).rstrip() + "\n"


def _render_metadata_lines(selected_review: SelectedReview) -> list[str]:
    paper_row, review_row = selected_review.paper_row, selected_review.review_row
    return [
        f"- Authors: {escape_untrusted_inline_text(', '.join(json.loads(paper_row['authors_json']))) or 'unknown'}",
        (
            f"- First published: {escape_untrusted_inline_text(paper_row['first_published_date']) or 'unknown'}"
            f" · source version: {escape_untrusted_inline_text(paper_row['source_version']) or 'unknown'}"
        ),
        f"- Source: {_render_source_links(selected_review.paper_aliases)}",
        (
            f"- Paper id: `{paper_row['paper_id']}` · review id: `{review_row['review_id']}` · edition id: "
            f"`{review_row['edition_id']}`"
        ),
        (
            f"- Reviewed {review_row['completed_at']} · model "
            f"{escape_untrusted_inline_text(review_row['model_identity'])} · build "
            f"{escape_untrusted_inline_text(review_row['server_build'])} · prompt {review_row['prompt_version']}"
        ),
    ]


def _render_statement(statement: dict[str, Any], selected_review: SelectedReview) -> str:
    rendered_text = escape_untrusted_inline_text(statement["statement"]) or "Not stated."
    if statement["evidence_status"] != "supported":
        return f"{rendered_text} *(insufficient evidence in the text read)*"
    return rendered_text + _render_pointers(statement["supporting_passage_ids"], selected_review)


def _render_finding(finding: dict[str, Any], selected_review: SelectedReview) -> str:
    conditions = escape_untrusted_inline_text(finding["conditions"])
    return (
        f"- {escape_untrusted_inline_text(finding['claim'])}"
        + (f" *Conditions:* {conditions}." if conditions else "")
        + f"{_render_pointers([finding['passage_id']], selected_review)}"
        + f" Quote: “{escape_untrusted_inline_text(finding['excerpt'])}”"
    )


def _render_pointers(passage_ids: list[str], selected_review: SelectedReview) -> str:
    return (
        (" " + ", ".join(_render_pointer(passage_id, selected_review) for passage_id in passage_ids))
        if passage_ids
        else ""
    )


def _render_pointer(passage_id: str, selected_review: SelectedReview) -> str:
    """e.g. `P12` (p. 4, 3.2 Results). Passage ids were validated against the packet, so they are safe in code."""
    evidence_passage = selected_review.evidence_passages.get(passage_id)
    if evidence_passage is None:
        return render_code_span(passage_id)
    return f"`{passage_id}` ({escape_untrusted_inline_text(evidence_passage.location_label)})"


def _render_topic_scores(selected_review: SelectedReview) -> list[str]:
    if not selected_review.topic_assessments:
        return ["- No topic assessments (no research topic with a question, terms, or seeds is configured)."]
    score_lines: list[str] = []
    for topic_assessment in selected_review.topic_assessments:
        topic_label = escape_untrusted_inline_text(topic_assessment.topic_id)
        score_lines += [f"### {topic_label}: {_format_topic_score(topic_assessment)}", ""]
        record = topic_assessment.assessment_record
        if topic_assessment.score_scale != SCORE_SCALE_PRIORITY_0_TO_100:
            score_lines += [escape_untrusted_inline_text(topic_assessment.rationale), ""]
            continue
        scoring = record.get("scoring", {})
        applied_ratings = scoring.get("applied_ratings", {})
        score_lines += ["| Dimension | Weight | Rating | Explanation |", "| --- | ---: | ---: | --- |"]
        for dimension in RUBRIC_DIMENSIONS:
            explanation = escape_untrusted_inline_text(
                record.get("ratings", {}).get(dimension.dimension_name, {}).get("explanation", "")
            )
            if dimension.dimension_name not in applied_ratings:
                score_lines.append(f"| {dimension.label} | {dimension.weight} | — | comparison unavailable |")
                continue
            rating_text = f"{applied_ratings[dimension.dimension_name]}/4"
            proposed = scoring.get("proposed_ratings", {}).get(dimension.dimension_name)
            if proposed is not None and proposed != applied_ratings[dimension.dimension_name]:
                rating_text += f" (proposed {proposed}; capped at 1 for an abstract-only review)"
            score_lines.append(
                f"| {dimension.label} | {dimension.weight} | {rating_text} | {explanation} |"
            )  # '|' is already escaped
        matched_seeds = [
            f"{escape_untrusted_inline_text(selected_review.seed_titles_by_id.get(seed_id, ''))} "
            f"({render_code_span(seed_id)})"
            for seed_id in record.get("matched_seed_ids", [])
        ]
        compared_reviews = [render_code_span(review_id) for review_id in record.get("compared_review_ids", [])]
        score_lines += [
            "",
            f"- Why: {escape_untrusted_inline_text(topic_assessment.rationale) or 'no rationale given'}",
            f"- Matched seeds: {'; '.join(matched_seeds) or 'none'}",
            f"- Compared reviews: {', '.join(compared_reviews) or 'none'}",
            (
                f"- Topic profile `{topic_assessment.topic_profile_sha256[:12]}` (a changed question or seed "
                "set gets a new assessment; the reading is reused)"
            ),
            "",
        ]
    return score_lines


def _render_reading_coverage(reading_record: dict[str, Any], coverage: str) -> list[str]:
    if not reading_record:
        return [f"- Coverage: {coverage}."]
    coverage_lines = [
        (
            f"- Text read: {reading_record.get('text_kind', 'abstract')} · coverage {coverage} · "
            f"{reading_record.get('passage_count', 0)} passage(s)"
        )
    ]
    sections_read = reading_record.get("sections_read", [])
    if reading_record.get("text_kind") == "extracted_text":
        coverage_lines.append(
            "- Sections read: "
            + (
                "; ".join(
                    f"{section['section_number']}: {escape_untrusted_inline_text(', '.join(section['titles']))}"
                    for section in sections_read
                )
                or "none"
            )
        )
        not_read = reading_record.get("sections_not_read", [])
        if not_read:
            coverage_lines.append(
                "- Sections NOT read: "
                + "; ".join(
                    f"{section['section_number']}: {escape_untrusted_inline_text(', '.join(section['titles']))}"
                    for section in not_read
                )
            )
        skipped = reading_record.get("skipped_not_substantive", [])
        if skipped:
            coverage_lines.append("- Skipped as not substantive: " + escape_untrusted_inline_text(", ".join(skipped)))
    for reason in reading_record.get("partial_reasons", []):
        coverage_lines.append(f"- Partial because: {escape_untrusted_inline_text(reason)}")
    parser_recovery = reading_record.get("parser_recovery")
    if parser_recovery:
        coverage_lines += [
            f"- Tables: {escape_untrusted_inline_text(parser_recovery['tables'])}",
            f"- Equations: {escape_untrusted_inline_text(parser_recovery['equations'])}",
        ]
    for problem in reading_record.get("parser_problems_reported_by_reviewer", []):
        coverage_lines.append(
            f"- Reviewer flagged {render_code_span(problem['passage_id'])}: "
            f"{escape_untrusted_inline_text(problem['problem'].replace('_', ' '))}"
        )
    return coverage_lines


def _describe_best_score(best_assessment: TopicAssessment | None) -> str:
    if best_assessment is None:
        return "not rated"
    return f"**{_format_topic_score(best_assessment)}** for {escape_untrusted_inline_text(best_assessment.topic_id)}"


def _passages_in_id_order(evidence_passages: dict[str, EvidencePassage]) -> list[EvidencePassage]:
    def order_key(passage: EvidencePassage) -> tuple[str, int]:
        prefix, number = passage.passage_id[:1], passage.passage_id[1:]
        return (prefix, int(number) if number.isdigit() else 0)

    return sorted(evidence_passages.values(), key=order_key)


def _shorten(text: str) -> str:
    return (
        text
        if len(text) <= MAXIMUM_RENDERED_PASSAGE_CHARACTERS
        else text[:MAXIMUM_RENDERED_PASSAGE_CHARACTERS].rsplit(" ", 1)[0] + " …"
    )


def _render_legacy_paper_note(selected_review: SelectedReview) -> str:
    """Notes of reviews saved by the provisional step-1 reviewer (before part 7)."""
    review_content = selected_review.review_content
    note_lines = [
        f"# {escape_untrusted_inline_text(selected_review.paper_row['title'])}",
        "",
        *_render_metadata_lines(selected_review),
        f"- Coverage: {selected_review.review_row['coverage']}",
        "",
        "## Summary",
        "",
        escape_untrusted_inline_text(review_content.get("one_sentence_summary", "")),
        "",
        "## Research question",
        "",
        escape_untrusted_inline_text(str(review_content.get("research_question", ""))),
        "",
        "## Method",
        "",
        escape_untrusted_inline_text(str(review_content.get("method", ""))),
        "",
        "## Authors' main claims",
        "",
        *_render_bullet_list(review_content.get("authors_main_claims", [])),
        "",
        "## Stated limitations",
        "",
        *_render_bullet_list(review_content.get("stated_limitations", [])),
        "",
        "## Relevance to research topics",
        "",
    ]
    note_lines += [
        f"- **{escape_untrusted_inline_text(assessment.topic_id)}**: {_format_topic_score(assessment)}. "
        f"{escape_untrusted_inline_text(assessment.rationale)}"
        for assessment in selected_review.topic_assessments
    ] or ["- No topic assessments."]
    if review_content.get("insufficient_evidence"):
        note_lines += ["", "> The reviewer marked this record as having insufficient evidence."]
    return "\n".join(note_lines) + "\n"


# --------------------------------------------------------------------------------------------------------------------
# Shared helpers (also used by the daily report, rendering/daily_report.py, and the comparison notes)
# --------------------------------------------------------------------------------------------------------------------


def _format_topic_score(topic_assessment: TopicAssessment) -> str:
    if topic_assessment.score_scale == SCORE_SCALE_PRIORITY_0_TO_100:
        return f"{topic_assessment.topic_score}/100"
    if topic_assessment.score_scale == SCORE_SCALE_RATING_0_TO_4:
        return f"{topic_assessment.topic_score}/4"
    return f"{topic_assessment.topic_score} ({topic_assessment.score_scale})"


def render_code_span(identifier_text: str) -> str:
    """An id in a Markdown code span. Inside a code span backslash escapes are shown literally, so instead of
    escaping, the one character that could end the span (a backtick) and line breaks are removed."""
    return "`" + " ".join(identifier_text.replace("`", "'").split()) + "`"


def _render_bullet_list(untrusted_items: list[str]) -> list[str]:
    return [f"- {escape_untrusted_inline_text(item)}" for item in untrusted_items] or ["- None stated."]


def _render_source_links(paper_aliases: list[tuple[str, str]]) -> str:
    rendered_sources = []
    for provider, provider_paper_id in paper_aliases:
        trusted_url = build_trusted_source_url(provider, provider_paper_id)
        label = escape_untrusted_inline_text(f"{provider}:{provider_paper_id}")
        rendered_sources.append(f"[{label}](<{trusted_url}>)" if trusted_url else label)
    return ", ".join(rendered_sources) or "none"


def render_source_links(paper_aliases: list[tuple[str, str]]) -> str:
    """Links built from trusted identifiers only (shared with the comparison notes of part 8)."""
    return _render_source_links(paper_aliases)
