"""Render the permanent per-paper reading note and the daily report from validated, saved data.

Notes live at output/papers/<paper_id>/<review_id>.md: a stable, revision-specific file named only by internal ids.
Step-1 rendering is deterministic (no writer role yet); part 9 adds the model-written overview.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.rendering.markdown_safety import build_trusted_source_url, escape_untrusted_inline_text


@dataclass(frozen=True)
class SelectedReview:
    """One completed review chosen for a report, with the trusted metadata needed to render it."""

    rank: int
    paper_row: sqlite3.Row
    review_row: sqlite3.Row
    review_content: dict[str, Any]
    paper_aliases: list[tuple[str, str]]
    best_topic_rating: int | None
    best_topic_id: str | None

    @property
    def note_relative_path_from_daily_dir(self) -> str:
        return f"../papers/{self.paper_row['paper_id']}/{self.review_row['review_id']}.md"


def render_paper_note(selected_review: SelectedReview) -> str:
    paper_row, review_row, review_content = (selected_review.paper_row, selected_review.review_row,
                                             selected_review.review_content)
    note_lines = [
        f"# {escape_untrusted_inline_text(paper_row['title'])}",
        "",
        f"- Authors: {escape_untrusted_inline_text(', '.join(json.loads(paper_row['authors_json']))) or 'unknown'}",
        f"- First published: {escape_untrusted_inline_text(paper_row['first_published_date']) or 'unknown'}"
        f" · source version: {escape_untrusted_inline_text(paper_row['source_version']) or 'unknown'}",
        f"- Source: {_render_source_links(selected_review.paper_aliases)}",
        f"- Paper id: `{paper_row['paper_id']}` · review id: `{review_row['review_id']}`",
        f"- Coverage: {review_row['coverage']} · reviewed {review_row['completed_at']}",
        f"- Model: {escape_untrusted_inline_text(review_row['model_identity'])} · build "
        f"{escape_untrusted_inline_text(review_row['server_build'])} · prompt {review_row['prompt_version']}",
        "",
        "## Summary",
        "",
        escape_untrusted_inline_text(review_content["one_sentence_summary"]),
        "",
        "## Research question",
        "",
        escape_untrusted_inline_text(review_content["research_question"]),
        "",
        "## Method",
        "",
        escape_untrusted_inline_text(review_content["method"]),
        "",
        "## Authors' main claims",
        "",
        *_render_bullet_list(review_content["authors_main_claims"]),
        "",
        "## Stated limitations",
        "",
        *_render_bullet_list(review_content["stated_limitations"]),
        "",
        "## Relevance to research topics",
        "",
    ]
    if review_content["topic_relevance"]:
        for topic_assessment in review_content["topic_relevance"]:
            note_lines.append(f"- **{escape_untrusted_inline_text(topic_assessment['topic_id'])}**: "
                              f"{topic_assessment['relevance_rating']}/4. "
                              f"{escape_untrusted_inline_text(topic_assessment['rationale'])}")
    else:
        note_lines.append("- No topic assessments.")
    if review_content["insufficient_evidence"]:
        note_lines += ["", "> The reviewer marked this record as having insufficient evidence."]
    return "\n".join(note_lines) + "\n"


def render_daily_report(*, report_date: str, timezone_name: str, cutoff_at: str,
                        selected_reviews: list[SelectedReview], report_statistics: dict[str, Any]) -> str:
    report_lines = [
        f"# Paper report for {report_date}",
        "",
        f"- Cutoff: {cutoff_at} (UTC) · reporting timezone: {timezone_name}",
        f"- Papers in catalog: {report_statistics['papers_in_catalog']} · reviews completed so far: "
        f"{report_statistics['reviews_completed']} · selected today: {len(selected_reviews)}",
        f"- Backlog: {report_statistics['jobs_queued']} queued job(s), {report_statistics['jobs_failed']} failed",
    ]
    if report_statistics.get("model_outage_note"):
        report_lines.append(f"- Model server: {report_statistics['model_outage_note']}")
    if report_statistics.get("topics_configured", 0) == 0:
        report_lines.append("- No research topics are configured yet; relevance ratings are unavailable.")
    report_lines += ["", "## Ranked papers", ""]

    if not selected_reviews:
        report_lines.append("No newly completed reviews since the previous report." +
                            (" (Model server outage, see above.)" if report_statistics.get("model_outage_note")
                             else " Quiet day."))
    for selected_review in selected_reviews:
        paper_row, review_content = selected_review.paper_row, selected_review.review_content
        relevance_label = (f"relevance {selected_review.best_topic_rating}/4 for "
                           f"{escape_untrusted_inline_text(selected_review.best_topic_id)}"
                           if selected_review.best_topic_rating is not None else "relevance not rated")
        published_label = escape_untrusted_inline_text(paper_row["first_published_date"]) or "date unknown"
        report_lines += [
            f"### {selected_review.rank}. {escape_untrusted_inline_text(paper_row['title'])}",
            "",
            f"{escape_untrusted_inline_text(', '.join(json.loads(paper_row['authors_json'])))} · {published_label}"
            f" · {relevance_label} · {selected_review.review_row['coverage']}",
            "",
            escape_untrusted_inline_text(review_content["one_sentence_summary"]),
            "",
            f"[Reading note]({selected_review.note_relative_path_from_daily_dir}) · "
            f"Source: {_render_source_links(selected_review.paper_aliases)}",
            "",
        ]
    return "\n".join(report_lines).rstrip() + "\n"


def _render_bullet_list(untrusted_items: list[str]) -> list[str]:
    return [f"- {escape_untrusted_inline_text(item)}" for item in untrusted_items] or ["- None stated."]


def _render_source_links(paper_aliases: list[tuple[str, str]]) -> str:
    rendered_sources = []
    for provider, provider_paper_id in paper_aliases:
        trusted_url = build_trusted_source_url(provider, provider_paper_id)
        label = escape_untrusted_inline_text(f"{provider}:{provider_paper_id}")
        rendered_sources.append(f"[{label}](<{trusted_url}>)" if trusted_url else label)
    return ", ".join(rendered_sources) or "none"
