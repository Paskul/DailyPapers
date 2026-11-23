"""The report's synthesis: the writer role at the cutoff, or the labelled fallback.

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
passages, filled first with the passages that state the numbers of the paper's summary (the writer restates them:
third pilot, 2026-11-23, "a systematic review of 38 studies" was rejected because no passage it was shown said 38),
then the passages its results cite (with the passages that state their numbers), then its other cited passages.
Each is cut to MAXIMUM_WRITER_PASSAGE_CHARACTERS at a word boundary, around the summary's numbers when they lie past
the cut. The host checks the writer's quotations and numbers against the FULL saved passages (`citable_passages`,
not shown to the model): an excerpt quoted from a cut passage is still a quotation of the real passage, and a model
that completes a word cut off at the boundary is not rejected for it (third pilot: "beyond the fi" -> "first").
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
from paperwatch.reading.evidence_checks import (
    NUMBER_PATTERN,
    find_numbers_backed_by_passage,
    find_numbers_needing_evidence,
    find_words_around_number,
    normalize_number_text,
)
from paperwatch.rendering.notes import SelectedReview
from paperwatch.reporting.report_comparisons import ComparisonSummary
from paperwatch.reporting.report_selection import ReportEntry
from paperwatch.reporting.report_snapshot import ReportSnapshot
from paperwatch.roles.base import (
    CHARACTERS_PER_TOKEN_ESTIMATE,
    MAXIMUM_REJECTED_ANSWER_CHARACTERS_IN_REPAIR,
    estimate_prompt_tokens,
    run_role_with_one_repair,
)
from paperwatch.roles.keyed_evidence import evidence_passage_key
from paperwatch.roles.writer import WRITER_ROLE, WRITER_ROLE_NAME, WRITER_SYSTEM_PROMPT

SYNTHESIS_BY_MODEL = "model"
SYNTHESIS_FALLBACK = "fallback"
SYNTHESIS_NOT_NEEDED = "not_needed"  # nothing was selected
REPORT_RESERVATION_PREFIX = "report:"
MAXIMUM_WRITER_PASSAGE_CHARACTERS = 900
MAXIMUM_SUMMARY_NUMBER_PASSAGES = 3  # per paper, chosen before the result passages
SUMMARY_NUMBER_CONTEXT_WORDS = 4
CHARACTERS_KEPT_BEFORE_A_NUMBER = 200  # when a passage is cut around a number past the first cut
MAXIMUM_FALLBACK_REASON_CHARACTERS = 300
# JSON indentation and headings, plus the repair request: the rejected answer (up to
# MAXIMUM_REJECTED_ANSWER_CHARACTERS_IN_REPAIR characters) and its list of problems.
WRITER_PROMPT_MARGIN_TOKENS = 1500 + int(
    (MAXIMUM_REJECTED_ANSWER_CHARACTERS_IN_REPAIR + 3000) / CHARACTERS_PER_TOKEN_ESTIMATE
)


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
    # The full text of every passage the selected reviews cite: what the host checks quotations and numbers
    # against (never rendered into the prompt; roles/writer.py).
    role_input["citable_passages"] = {
        evidence_passage_key(selected_review.review_row["review_id"], passage_id): passage.passage_text
        for selected_review in selected_reviews
        for passage_id, passage in selected_review.evidence_passages.items()
    }
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
    if not selected_review.is_evidence_review:  # legacy reviews: a simpler layout, no passages
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
            {
                "claim": finding["claim"],
                "conditions": finding["conditions"],
                "passage_id": finding["passage_id"],
                **({"number_passage_ids": finding["number_passage_ids"]} if finding.get("number_passage_ids") else {}),
            }
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
    """The passages that state the summary's numbers first, then the result passages (each followed by the passages
    that state its numbers), then the review's other cited passages, while the budget lasts."""
    content = selected_review.review_content
    summary = str(content.get("one_sentence_summary", ""))
    summary_numbers = find_numbers_needing_evidence(summary)
    result_passage_ids: list[str] = []
    for finding in content.get("results", []):
        if isinstance(finding, dict):
            result_passage_ids += [finding["passage_id"], *finding.get("number_passage_ids", [])]
    ordered_passage_ids = list(
        dict.fromkeys(
            _passages_stating_summary_numbers(selected_review, summary, summary_numbers)
            + result_passage_ids
            + list(selected_review.evidence_passages)
        )
    )
    chosen_passages, characters_used = [], 0
    for passage_id in ordered_passage_ids:
        passage = selected_review.evidence_passages.get(passage_id)
        if passage is None:
            continue
        passage_text = cut_passage_for_writer(passage.passage_text, summary_numbers)
        entry_characters = len(passage_text) + len(passage.location_label) + 60  # JSON keys around it
        if characters_used + entry_characters > character_budget:
            break
        chosen_passages.append({"passage_id": passage_id, "location": passage.location_label, "text": passage_text})
        characters_used += entry_characters
    return chosen_passages


def _passages_stating_summary_numbers(
    selected_review: SelectedReview, summary: str, summary_numbers: list[str]
) -> list[str]:
    """For each number of the summary, the cited passage that states it with the most words in common around it
    (then the earliest); at most MAXIMUM_SUMMARY_NUMBER_PASSAGES passages."""
    chosen_passage_ids: list[str] = []
    for number in summary_numbers:
        summary_words = find_words_around_number(summary, number, SUMMARY_NUMBER_CONTEXT_WORDS)
        stating_passages = [
            (passage_id, passage)
            for passage_id, passage in selected_review.evidence_passages.items()
            if number.lstrip("+-")
            in {value.lstrip("+-") for value in find_numbers_backed_by_passage(passage.passage_text)}
        ]
        if not stating_passages:
            continue
        best_passage_id = max(
            enumerate(stating_passages),
            key=lambda position_and_passage: (
                len(
                    summary_words
                    & find_words_around_number(
                        position_and_passage[1][1].passage_text, number, SUMMARY_NUMBER_CONTEXT_WORDS
                    )
                ),
                -position_and_passage[0],
            ),
        )[1][0]
        if best_passage_id not in chosen_passage_ids:
            chosen_passage_ids.append(best_passage_id)
        if len(chosen_passage_ids) == MAXIMUM_SUMMARY_NUMBER_PASSAGES:
            break
    return chosen_passage_ids


def cut_passage_for_writer(passage_text: str, numbers_to_keep: list[str]) -> str:
    """At most MAXIMUM_WRITER_PASSAGE_CHARACTERS of the passage, ending at a word boundary (a word cut in half
    invites the model to complete it inside a quotation). When the first place the passage states one of
    `numbers_to_keep` lies near or past the end of the first cut, the window starts at the sentence that states it
    (or CHARACTERS_KEPT_BEFORE_A_NUMBER before it) instead."""
    if len(passage_text) <= MAXIMUM_WRITER_PASSAGE_CHARACTERS:
        return passage_text
    window_start = 0
    number_position = _first_position_of_any_number(passage_text, numbers_to_keep)
    if (
        number_position is not None
        and number_position > MAXIMUM_WRITER_PASSAGE_CHARACTERS - CHARACTERS_KEPT_BEFORE_A_NUMBER
    ):
        sentence_end_before = passage_text.rfind(". ", 0, number_position)
        if (
            sentence_end_before != -1
            and number_position - sentence_end_before <= MAXIMUM_WRITER_PASSAGE_CHARACTERS // 2
        ):
            window_start = sentence_end_before + 2
        else:
            window_start = number_position - CHARACTERS_KEPT_BEFORE_A_NUMBER
            while window_start < number_position and not passage_text[window_start - 1].isspace():
                window_start += 1  # start at the beginning of a word
    window_end = min(len(passage_text), window_start + MAXIMUM_WRITER_PASSAGE_CHARACTERS)
    if window_end < len(passage_text) and not passage_text[window_end].isspace():
        last_space = passage_text.rfind(" ", window_start, window_end)
        if last_space > window_start:
            window_end = last_space  # end at the end of a word
    return passage_text[window_start:window_end].strip()


def _first_position_of_any_number(passage_text: str, numbers: list[str]) -> int | None:
    wanted_values = {number.lstrip("+-") for number in numbers}
    for number_match in NUMBER_PATTERN.finditer(passage_text):
        if normalize_number_text(number_match.group()).lstrip("+-") in wanted_values:
            return number_match.start()
    return None
