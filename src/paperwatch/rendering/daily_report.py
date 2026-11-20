"""Render the daily report (plan part 9) from the frozen snapshot, the saved reviews, and the synthesis.

| Section                     | Content                                                                              |
| Opening                     | date, cutoff and timezone, counts found/screened/reviewed, selected, backlog,        |
|                             | outages, time since the previous report, and who wrote the overview (or "fallback")  |
| What matters today          | the writer's overview, two or three connected findings, and what to read first,      |
|                             | each with review + passage citations; or the fallback / quiet-day / outage note      |
| Ranked papers               | title, label (new release / older / reference / update), score and coverage, authors, |
|                             | date, summary, a specific reason to read it, the cited sections, note and source     |
| Comparisons and corrections | saved comparisons involving today's papers; withdrawals of earlier recommendations   |

The renderer, never the model, supplies ranks, dates, numeric scores, URLs, and relative note links. Every paper or
model text is escaped (markdown_safety); external links are built only from trusted identifiers; note paths use
internal ids only. Rendering is a pure function of its inputs, so the same snapshot gives the same bytes.
A citation looks like `[#2 · P12 (p. 4, 3.2 Results)](<../papers/<paper>/<review>.md>)`: the ranked entry, the
passage and its section, and the note that quotes that passage under "Evidence passages".
"""

from __future__ import annotations

import json
from typing import Any
from zoneinfo import ZoneInfo

from paperwatch.clock import parse_utc_timestamp
from paperwatch.rendering.markdown_safety import escape_untrusted_inline_text
from paperwatch.rendering.notes import SCORE_MEANING_NOTE, SelectedReview, render_code_span, render_source_links
from paperwatch.reporting.report_comparisons import ComparisonSummary
from paperwatch.reporting.report_selection import ENTRY_LABEL_TEXT, ReportEntry
from paperwatch.reporting.report_snapshot import ReportSnapshot
from paperwatch.reporting.report_synthesis import SYNTHESIS_BY_MODEL, SYNTHESIS_FALLBACK, ReportSynthesis

COVERAGE_TEXT = {"full_text": "full text", "partial_text": "partial text (partially read)",
                 "abstract_only": "abstract only"}
COMPARABILITY_TEXT = {"comparable": "comparable conditions", "partially_comparable": "partially comparable conditions",
                      "incompatible": "incompatible conditions"}
HOURS_BEFORE_A_GAP_IS_DISCLOSED = 36
FALLBACK_CITED_RESULTS = 2


class _ReportContext:
    """Lookups shared by the section renderers."""

    def __init__(self, snapshot: ReportSnapshot, selected_reviews: list[SelectedReview],
                 comparisons: list[ComparisonSummary], synthesis: ReportSynthesis) -> None:
        self.snapshot = snapshot
        self.synthesis = synthesis
        self.comparisons = comparisons
        self.selected_review_by_id = {selected.review_row["review_id"]: selected for selected in selected_reviews}
        self.entry_by_review_id = {entry.review_id: entry for entry in snapshot.entries}
        self.writer_output: dict[str, Any] = synthesis.writer_output or {}

    def note_link(self, review_id: str) -> str:
        entry = self.entry_by_review_id[review_id]
        return f"../papers/{entry.paper_id}/{review_id}.md"

    def render_citations(self, evidence: list[dict[str, str]]) -> str:
        rendered = []
        for item in evidence:
            entry = self.entry_by_review_id.get(item["review_id"])
            selected_review = self.selected_review_by_id.get(item["review_id"])
            if entry is None or selected_review is None:
                continue
            passage = selected_review.evidence_passages.get(item["passage_id"])
            location = f" ({escape_untrusted_inline_text(passage.location_label)})" if passage else ""
            rendered.append(f"[#{entry.rank} · {render_code_span(item['passage_id'])}{location}]"
                            f"(<{self.note_link(item['review_id'])}>)")
        return (" — " + "; ".join(rendered)) if rendered else ""


def render_daily_report(snapshot: ReportSnapshot, selected_reviews: list[SelectedReview],
                        comparisons: list[ComparisonSummary], synthesis: ReportSynthesis) -> str:
    context = _ReportContext(snapshot, selected_reviews, comparisons, synthesis)
    report_lines = [f"# Paper report for {snapshot.report_date}", ""]
    report_lines += _render_opening(context)
    report_lines += ["", "## What matters today", ""]
    report_lines += _render_what_matters_today(context)
    report_lines += ["", "## Ranked papers", ""]
    report_lines += _render_ranked_papers(context, selected_reviews)
    report_lines += _render_comparisons_and_corrections(context)
    return "\n".join(report_lines).rstrip() + "\n"


# --------------------------------------------------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------------------------------------------------


def describe_moment_in_report_timezone(stored_timestamp: str, timezone_name: str) -> str:
    """'2026-11-20 07:00 America/New_York (12:00 UTC)'."""
    moment = parse_utc_timestamp(stored_timestamp)
    local_text = f"{moment.astimezone(ZoneInfo(timezone_name)):%Y-%m-%d %H:%M} {timezone_name}"
    return local_text if timezone_name == "UTC" else f"{local_text} ({moment:%H:%M} UTC)"


def _render_opening(context: _ReportContext) -> list[str]:
    snapshot, statistics = context.snapshot, context.snapshot.statistics
    cutoff_line = f"- Cutoff: {describe_moment_in_report_timezone(snapshot.cutoff_at, snapshot.timezone_name)}"
    if snapshot.published_early:
        cutoff_line += (f" · **published early** by the operator; the scheduled cutoff was "
                        f"{describe_moment_in_report_timezone(snapshot.scheduled_cutoff_at, snapshot.timezone_name)}")
    opening_lines = [cutoff_line]
    if statistics.previous_report_date is None:
        opening_lines.append("- Previous report: none (this is the first)")
    elif statistics.hours_since_previous_report is not None and \
            statistics.hours_since_previous_report > HOURS_BEFORE_A_GAP_IS_DISCLOSED:
        opening_lines.append(f"- Previous report: {statistics.previous_report_date}, "
                             f"**{statistics.hours_since_previous_report:g} h earlier** (a gap: no report was "
                             "published in between)")
    else:
        opening_lines.append(f"- Previous report: {statistics.previous_report_date}, "
                             f"{statistics.hours_since_previous_report:g} h earlier")
    opening_lines += [
        f"- Since then: {statistics.papers_found_in_window} paper(s) found, {statistics.papers_screened_in_window} "
        f"screened, {statistics.reviews_completed_in_window} review(s) completed · selected today: "
        f"**{len(snapshot.entries)}** (reading priority at least {snapshot.minimum_reading_priority}/100)",
        f"- Backlog: {statistics.eligible_held_for_later} qualifying review(s) held for later reports · "
        f"{statistics.jobs_queued} queued job(s) · {statistics.jobs_failed} failed",
        "- Source outages: " + ("; ".join(escape_untrusted_inline_text(line)
                                          for line in statistics.source_outages) or "none"),
    ]
    if statistics.model_outage:
        opening_lines.append(f"- Model server: {escape_untrusted_inline_text(statistics.model_outage)}")
    if statistics.topics_configured == 0:
        opening_lines.append("- No research topics are configured yet; relevance ratings are unavailable.")
    synthesis = context.synthesis
    if synthesis.synthesis_status == SYNTHESIS_BY_MODEL:
        opening_lines.append(f"- Overview and reasons written by the local model "
                             f"({escape_untrusted_inline_text(synthesis.model_identity)}, prompt "
                             f"{synthesis.prompt_version}) from the saved reviews; every claim cites its passages.")
    elif synthesis.synthesis_status == SYNTHESIS_FALLBACK:
        opening_lines.append(f"- **Fallback report:** the writer was unavailable "
                             f"({escape_untrusted_inline_text(synthesis.fallback_reason)}). The selection and order "
                             "are the same; entries show the saved summaries and topic rationales.")
    opening_lines += ["", f"> {SCORE_MEANING_NOTE}"]
    return opening_lines


# --------------------------------------------------------------------------------------------------------------------
# What matters today
# --------------------------------------------------------------------------------------------------------------------


def _render_what_matters_today(context: _ReportContext) -> list[str]:
    snapshot, statistics = context.snapshot, context.snapshot.statistics
    if not snapshot.entries:
        if statistics.model_outage or statistics.source_outages:
            return ["No paper qualified for this report. **This is not a quiet day:** see the outages above. "
                    "Queued papers are kept and will be reported once they are read."]
        if statistics.jobs_queued:
            return [f"No paper qualified for this report yet; {statistics.jobs_queued} job(s) are still queued."]
        return ["Quiet day: no newly completed review reached the reading-priority floor."]

    first_entry = snapshot.entries[0]
    if context.synthesis.synthesis_status != SYNTHESIS_BY_MODEL:
        first_title = context.selected_review_by_id[first_entry.review_id].paper_row["title"]
        return ["> Fallback: no overview was written today. The ranked list below is complete.", "",
                f"**Read first:** #1 {escape_untrusted_inline_text(first_title)} (the highest reading priority)."]

    writer_output = context.writer_output
    overview = writer_output["overview"]
    section_lines = [escape_untrusted_inline_text(overview["statement"])
                     + context.render_citations(overview["evidence"]), ""]
    comparison_ids = {comparison.comparison_id for comparison in context.comparisons}
    for finding in writer_output["connected_findings"]:
        comparison_links = [f"[comparison {render_code_span(comparison_id)}](<../comparisons/{comparison_id}.md>)"
                            for comparison_id in finding["comparison_ids"] if comparison_id in comparison_ids]
        section_lines.append(f"- {escape_untrusted_inline_text(finding['statement'])}"
                             + context.render_citations(finding["evidence"])
                             + (f" · see {', '.join(comparison_links)}" if comparison_links else ""))
    if writer_output["connected_findings"]:
        section_lines.append("")
    read_first = writer_output["read_first"]
    read_first_entry = context.entry_by_review_id[read_first["review_id"]]
    read_first_title = context.selected_review_by_id[read_first["review_id"]].paper_row["title"]
    section_lines.append(f"**Read first:** #{read_first_entry.rank} {escape_untrusted_inline_text(read_first_title)}"
                         f". {escape_untrusted_inline_text(read_first['explanation'])}"
                         + context.render_citations(read_first["evidence"]))
    return section_lines


# --------------------------------------------------------------------------------------------------------------------
# Ranked papers
# --------------------------------------------------------------------------------------------------------------------


def _render_ranked_papers(context: _ReportContext, selected_reviews: list[SelectedReview]) -> list[str]:
    if not selected_reviews:
        return ["None today."]
    reason_by_review_id = {reason["review_id"]: reason
                           for reason in context.writer_output.get("paper_reasons", [])}
    entry_lines: list[str] = []
    for selected_review in selected_reviews:
        review_id = selected_review.review_row["review_id"]
        entry = context.entry_by_review_id[review_id]
        paper_row = selected_review.paper_row
        score_text = f"{entry.reading_priority}/100" if entry.reading_priority is not None else "not rated"
        authors = escape_untrusted_inline_text(", ".join(json.loads(paper_row["authors_json"]))) or "authors unknown"
        published = escape_untrusted_inline_text(paper_row["first_published_date"]) or "date unknown"
        topic_text = f" · topic {render_code_span(entry.best_topic_id)}" if entry.best_topic_id else ""
        entry_lines += [
            f"### {entry.rank}. {escape_untrusted_inline_text(paper_row['title'])} — {score_text} · "
            f"{ENTRY_LABEL_TEXT.get(entry.entry_label, entry.entry_label)} · "
            f"{COVERAGE_TEXT.get(entry.coverage, entry.coverage)}",
            "",
            f"{authors} · {published}{topic_text}",
            "",
            f"**Summary.** {escape_untrusted_inline_text(selected_review.review_content.get('one_sentence_summary'))}",
            "",
        ]
        writer_reason = reason_by_review_id.get(review_id)
        if writer_reason is not None:
            entry_lines += [f"**Why read it.** {escape_untrusted_inline_text(writer_reason['reason_to_read'])}"
                            + context.render_citations(writer_reason["evidence"]), ""]
            cited_passage_ids = [item["passage_id"] for item in writer_reason["evidence"]]
        else:
            rationale = next((assessment.rationale for assessment in selected_review.topic_assessments
                              if assessment.topic_id == entry.best_topic_id), "")
            entry_lines += [f"**Why read it** (from the saved topic assessment). "
                            f"{escape_untrusted_inline_text(rationale) or 'No rationale was saved.'}", ""]
            cited_passage_ids = [finding["passage_id"] for finding in selected_review.review_content.get("results", [])
                                 if isinstance(finding, dict)][:FALLBACK_CITED_RESULTS]
        if entry.is_update:
            entry_lines += _render_update_lines(context, entry)
        entry_lines += [f"[Reading note](<{context.note_link(review_id)}>) · Source: "
                        f"{render_source_links(selected_review.paper_aliases)}"
                        + _render_cited_sections(selected_review, cited_passage_ids), ""]
    return entry_lines


def _render_update_lines(context: _ReportContext, entry: ReportEntry) -> list[str]:
    changes = "; ".join(escape_untrusted_inline_text(change.description) for change in entry.revision_changes)
    return [f"**Update.** Recommended on {entry.previous_report_date} from "
            f"[the earlier review](<../papers/{entry.paper_id}/{entry.previous_review_id}.md>); this newer review "
            f"read different text. What changed: {changes}.", ""]


def _render_cited_sections(selected_review: SelectedReview, passage_ids: list[str]) -> str:
    cited = []
    for passage_id in dict.fromkeys(passage_ids):
        passage = selected_review.evidence_passages.get(passage_id)
        if passage is not None:
            cited.append(f"{render_code_span(passage_id)} ({escape_untrusted_inline_text(passage.location_label)})")
    return (" · Cited: " + ", ".join(cited)) if cited else ""


# --------------------------------------------------------------------------------------------------------------------
# Comparisons and corrections
# --------------------------------------------------------------------------------------------------------------------


def _render_comparisons_and_corrections(context: _ReportContext) -> list[str]:
    comparisons, corrections = context.comparisons, context.snapshot.corrections
    if not comparisons and not corrections:
        return []
    section_lines = ["## Comparisons and corrections", ""]
    if comparisons:
        section_lines += ["### Comparisons", ""]
        for comparison in comparisons:
            paper_labels = []
            for review_id, title in zip(comparison.review_ids, comparison.paper_titles):
                entry = context.entry_by_review_id.get(review_id)
                rank_text = f"#{entry.rank} " if entry is not None else ""
                paper_labels.append(f"{rank_text}**{escape_untrusted_inline_text(title)}**")
            conclusion = ("the reviews were insufficient for a conclusion" if comparison.insufficient_evidence
                          else escape_untrusted_inline_text(comparison.reader_conclusion))
            section_lines.append(
                f"- [Comparison {render_code_span(comparison.comparison_id)}]"
                f"(<{comparison.note_relative_path_from_daily_dir}>) on {render_code_span(comparison.topic_id)}: "
                f"{' vs '.join(paper_labels)} — {COMPARABILITY_TEXT.get(comparison.conditions_comparability, 'unknown')}"
                f". {conclusion}")
        section_lines.append("")
    if corrections:
        section_lines += ["### Corrections", ""]
        for correction in corrections:
            section_lines.append(
                f"- **{escape_untrusted_inline_text(correction.title)}** (recommended on "
                f"{correction.first_reported_on}): the provider now reports it {correction.notice_kind} "
                f"(noticed {correction.detected_at[:10]}). It is no longer recommended.")
        section_lines.append("")
    return section_lines
