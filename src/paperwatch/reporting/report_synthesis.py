"""The report's synthesis: the writer role at the cutoff, or the labelled fallback (plan part 9).

"Reserve the next inference slot for the report at cutoff. Set a short deadline for obtaining the slot and
validating the synthesis. If Qwen is unavailable or misses the deadline, render the same selections from saved
summaries and label this fallback."

  1. The report reserves the inference slot (llm/inference_reservation.py) until the deadline; the worker yields at
     its next call boundary. A call already running is not interrupted, so it may use part of the deadline.
  2. A paused model server is probed once (cheap /health); if it is down, the fallback is used at once.
  3. The writer runs once, plus at most one repair, with every wait and call bounded by the time left
     (report.writer_deadline_minutes, owner decision 2026-11-20: 30 minutes at medium reasoning).
  4. Any failure (outage, lock wait, timeout, deadline, invalid answer after repair) gives the fallback with its
     reason. The selection is the same either way: it was frozen before the writer ran.
With no selected paper the writer is not called at all (zero model calls on a quiet day).

The writer's input is bounded to the model's context budget: each paper gets an equal share of the room left for
passages, filled with the passages its results cite first, then its other cited passages, each cut to
MAXIMUM_WRITER_PASSAGE_CHARACTERS (an excerpt quoted from a cut passage is still a quotation of the real passage).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from paperwatch.app_context import AppContext
from paperwatch.clock import utc_now
from paperwatch.config import AppConfig
from paperwatch.errors import PaperwatchError
from paperwatch.llm.inference_reservation import release_inference_reservation, reserve_inference_slot
from paperwatch.llm.service_guard import ensure_model_service_available
from paperwatch.rendering.notes import SelectedReview
from paperwatch.reporting.report_comparisons import ComparisonSummary
from paperwatch.reporting.report_selection import ReportEntry
from paperwatch.reporting.report_snapshot import ReportSnapshot
from paperwatch.roles.base import CHARACTERS_PER_TOKEN_ESTIMATE, estimate_prompt_tokens, run_role_with_one_repair
from paperwatch.roles.writer import WRITER_ROLE, WRITER_ROLE_NAME, WRITER_SYSTEM_PROMPT

SYNTHESIS_BY_MODEL = "model"
SYNTHESIS_FALLBACK = "fallback"
SYNTHESIS_NOT_NEEDED = "not_needed"  # nothing was selected
REPORT_RESERVATION_PREFIX = "report:"
MAXIMUM_WRITER_PASSAGE_CHARACTERS = 900
MAXIMUM_FALLBACK_REASON_CHARACTERS = 300
WRITER_PROMPT_MARGIN_TOKENS = 1500  # JSON indentation, headings, and the repair request's extra text


@dataclass(frozen=True)
class ReportSynthesis:
    synthesis_status: str  # model | fallback | not_needed
    writer_output: dict[str, Any] | None = None
    fallback_reason: str | None = None
    prompt_version: str | None = None
    model_identity: str | None = None

    @property
    def is_by_model(self) -> bool:
        return self.synthesis_status == SYNTHESIS_BY_MODEL

    def synthesis_record(self) -> dict[str, Any] | None:
        """What reports.synthesis_json stores (model synthesis only)."""
        if not self.is_by_model:
            return None
        return {
            "writer_output": self.writer_output,
            "prompt_version": self.prompt_version,
            "model_identity": self.model_identity,
        }


def write_report_synthesis(
    app: AppContext,
    snapshot: ReportSnapshot,
    selected_reviews: list[SelectedReview],
    comparisons: list[ComparisonSummary],
) -> ReportSynthesis:
    if not snapshot.entries:
        return ReportSynthesis(SYNTHESIS_NOT_NEEDED)
    deadline_seconds = app.config.report.writer_deadline_minutes * 60
    deadline_monotonic = time.monotonic() + deadline_seconds
    reservation_name = f"{REPORT_RESERVATION_PREFIX}{snapshot.report_date}"
    reserve_inference_slot(
        app.connection,
        reservation_name,
        purpose=f"daily report {snapshot.report_date}",
        reserved_until=utc_now() + timedelta(seconds=deadline_seconds),
    )
    try:
        ensure_model_service_available(
            app.connection, app.model_client, app.config.model, probe_now_even_if_backing_off=True
        )
        role_result = run_role_with_one_repair(
            role_definition=WRITER_ROLE,
            role_input=build_writer_role_input(app.config, snapshot, selected_reviews, comparisons),
            app_config=app.config,
            model_client=app.model_client,
            connection=app.connection,
            deadline_monotonic=deadline_monotonic,
            holds_inference_reservation=True,
        )
    except PaperwatchError as failure:
        return ReportSynthesis(
            SYNTHESIS_FALLBACK,
            fallback_reason=(f"{failure.failed_component}: {failure}")[:MAXIMUM_FALLBACK_REASON_CHARACTERS],
        )
    finally:
        release_inference_reservation(app.connection, reservation_name)
    return ReportSynthesis(
        SYNTHESIS_BY_MODEL,
        writer_output=role_result.validated_output,
        prompt_version=role_result.prompt_version,
        model_identity=role_result.model_response.model_identity,
    )


def load_saved_synthesis(report_row: sqlite3.Row) -> ReportSynthesis:
    """The synthesis a `pending` or `published` report was rendered with."""
    record = json.loads(report_row["synthesis_json"] or "null") or {}
    return ReportSynthesis(
        report_row["synthesis_status"] or SYNTHESIS_NOT_NEEDED,
        writer_output=record.get("writer_output"),
        fallback_reason=report_row["fallback_reason"],
        prompt_version=record.get("prompt_version"),
        model_identity=record.get("model_identity"),
    )


# --------------------------------------------------------------------------------------------------------------------
# The writer's input
# --------------------------------------------------------------------------------------------------------------------


def build_writer_role_input(
    app_config: AppConfig,
    snapshot: ReportSnapshot,
    selected_reviews: list[SelectedReview],
    comparisons: list[ComparisonSummary],
) -> dict[str, Any]:
    entry_by_review_id = {entry.review_id: entry for entry in snapshot.entries}
    papers = [
        _describe_paper_without_passages(selected_review, entry_by_review_id[selected_review.review_row["review_id"]])
        for selected_review in selected_reviews
    ]
    role_input: dict[str, Any] = {
        "report_date": snapshot.report_date,
        "topics": [
            {"topic_id": topic.topic_id, "question": topic.question_for_local_model} for topic in app_config.topics
        ],
        "papers": papers,
        "comparisons": [comparison.to_writer_input() for comparison in comparisons],
    }
    passage_characters_per_paper = _passage_characters_per_paper(app_config, role_input, len(papers))
    for paper, selected_review in zip(papers, selected_reviews):
        paper["passages"] = _choose_passages(selected_review, passage_characters_per_paper)
    return role_input


def _describe_paper_without_passages(selected_review: SelectedReview, entry: ReportEntry) -> dict[str, Any]:
    content = selected_review.review_content
    paper_row = selected_review.paper_row
    best_rationale = next(
        (
            assessment.rationale
            for assessment in selected_review.topic_assessments
            if assessment.topic_id == entry.best_topic_id
        ),
        "",
    )
    if not selected_review.is_evidence_review:  # reviews saved before part 7: a simpler layout, no passages
        return {
            "review_id": entry.review_id,
            "entry_label": entry.entry_label,
            "title": paper_row["title"],
            "year": (paper_row["first_published_date"] or "")[:4],
            "coverage": entry.coverage,
            "best_topic_id": entry.best_topic_id,
            "summary": content.get("one_sentence_summary", ""),
            "research_question": str(content.get("research_question", "")),
            "method": str(content.get("method", "")),
            "evaluation_setting": "",
            "assumptions": [],
            "results": [
                {"claim": str(claim), "conditions": "", "passage_id": ""}
                for claim in content.get("authors_main_claims", [])
            ],
            "limitations": [
                {"statement": str(limitation), "stated_by": "authors"}
                for limitation in content.get("stated_limitations", [])
            ],
            "topic_rationale": best_rationale,
            "passages": [],
        }

    def statement_text(statement: dict[str, Any]) -> str:
        suffix = "" if statement["evidence_status"] == "supported" else " (insufficient evidence)"
        return str(statement["statement"]) + suffix

    return {
        "review_id": entry.review_id,
        "entry_label": entry.entry_label,
        "title": paper_row["title"],
        "year": (paper_row["first_published_date"] or "")[:4],
        "coverage": entry.coverage,
        "best_topic_id": entry.best_topic_id,
        "summary": content["one_sentence_summary"],
        "research_question": statement_text(content["research_question"]),
        "method": statement_text(content["method"]),
        "evaluation_setting": statement_text(content["evaluation_setting"]),
        "assumptions": [statement_text(statement) for statement in content["assumptions"]],
        "results": [
            {"claim": finding["claim"], "conditions": finding["conditions"], "passage_id": finding["passage_id"]}
            for finding in content["results"]
        ],
        "limitations": [
            {"statement": limitation["statement"], "stated_by": limitation["stated_by"]}
            for limitation in content["limitations"]
        ],
        "topic_rationale": best_rationale,
        "passages": [],
    }


def _passage_characters_per_paper(
    app_config: AppConfig, role_input_without_passages: dict[str, Any], paper_count: int
) -> int:
    """Equal share of the context left after the system prompt, the rest of the input, and the output reserve."""
    writer_limits = app_config.limits_for_role(WRITER_ROLE_NAME)
    used_tokens = estimate_prompt_tokens(WRITER_SYSTEM_PROMPT, json.dumps(role_input_without_passages, indent=1))
    free_tokens = (
        app_config.model.required_context_tokens
        - writer_limits.max_output_tokens
        - used_tokens
        - WRITER_PROMPT_MARGIN_TOKENS
    )
    return max(0, int(free_tokens * CHARACTERS_PER_TOKEN_ESTIMATE) // max(1, paper_count))


def _choose_passages(selected_review: SelectedReview, character_budget: int) -> list[dict[str, str]]:
    """Result passages first, then the review's other cited passages, while the budget lasts."""
    content = selected_review.review_content
    result_passage_ids = [finding["passage_id"] for finding in content.get("results", []) if isinstance(finding, dict)]
    ordered_passage_ids = list(dict.fromkeys(result_passage_ids + list(selected_review.evidence_passages)))
    chosen_passages, characters_used = [], 0
    for passage_id in ordered_passage_ids:
        passage = selected_review.evidence_passages.get(passage_id)
        if passage is None:
            continue
        passage_text = passage.passage_text[:MAXIMUM_WRITER_PASSAGE_CHARACTERS]
        entry_characters = len(passage_text) + len(passage.location_label) + 60  # JSON keys around it
        if characters_used + entry_characters > character_budget:
            break
        chosen_passages.append({"passage_id": passage_id, "location": passage.location_label, "text": passage_text})
        characters_used += entry_characters
    return chosen_passages
