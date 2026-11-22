"""Run one evidence-backed reading of a packet (plan part 7). Called by the worker OUTSIDE any write transaction;
the worker commits the result together with `job done` under its lease check.

    1. section notes   only when the plan has several reading sections: one section_reader call per section, each
                       saved at once (stage_results.py), so a retry continues at the first missing section
    2. synthesis       one reviewer call over the passages (single section) or over the section notes; also saved
    3. assessment      one topic_assessor call over the validated review and every assessable topic with its
                       comparison anchors; the 0-100 score is computed here, not by the model
Every model answer is validated (schema, passage ids, excerpts, numbers) and gets one bounded repair attempt.
A topic list with no assessable topic skips step 3 (the reading is still saved).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from paperwatch.activity_log import recorded_activity
from paperwatch.app_context import AppContext
from paperwatch.reading.comparison_anchors import TopicAnchors, assessable_topic_profiles, build_topic_anchors
from paperwatch.reading.reading_priority import RELATIVE_VALUE, ScoredTopicAssessment, score_topic_assessment
from paperwatch.reading.reading_plan import COVERAGE_ABSTRACT_ONLY
from paperwatch.reading.review_packet import ReviewPacket
from paperwatch.reading.stage_results import (
    SYNTHESIS_STAGE_KEY,
    ReadingStageIdentity,
    SavedStageResult,
    load_saved_stage_results,
    save_stage_result,
)
from paperwatch.readings import select_edition_to_read
from paperwatch.research_profile import build_research_profiles
from paperwatch.errors import ModelOutputInvalid
from paperwatch.roles.base import (
    RoleDefinition,
    RoleInput,
    estimate_prompt_tokens,
    parse_and_validate_role_output,
    run_role_with_one_repair,
)
from paperwatch.roles.reviewer import (
    READING_MODE_PASSAGES,
    READING_MODE_SECTION_NOTES,
    REVIEWER_ROLE,
    collect_cited_passage_ids,
)
from paperwatch.roles.section_reader import SECTION_READER_ROLE
from paperwatch.roles.topic_assessor import TOPIC_ASSESSOR_ROLE

NOTE_FIELDS_DROPPED_FIRST_WHEN_TOO_LONG = ("reviewer_inferences", "ambiguous_passages", "parser_problems")


@dataclass(frozen=True)
class CompletedTopicAssessment:
    topic_id: str
    model_assessment: dict[str, Any]  # validated: ratings with explanations, rationale, matched/compared ids
    scored: ScoredTopicAssessment
    topic_profile_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {**self.model_assessment, "scoring": self.scored.to_record()}


@dataclass(frozen=True)
class CompletedReading:
    packet: ReviewPacket
    review_content: dict[str, Any]
    reviewer_prompt_version: str
    model_identity: str
    server_build: str
    reading_record: dict[str, Any]
    topic_assessments: list[CompletedTopicAssessment]
    topic_assessor_prompt_version: str

    @property
    def cited_passage_ids(self) -> list[str]:
        return collect_cited_passage_ids(self.review_content)


def read_paper(app: AppContext, packet: ReviewPacket, *, job_id: str | None) -> CompletedReading:
    stage_identity = ReadingStageIdentity(packet.paper_id, packet.review_input_key, packet.reading_plan.plan_sha256)
    section_notes: list[dict[str, Any]] = []
    if not packet.reading_plan.is_single_pass:
        section_notes = _read_sections(app, packet, stage_identity, job_id)
    synthesis = _run_or_load_stage(
        app,
        stage_identity,
        SYNTHESIS_STAGE_KEY,
        REVIEWER_ROLE,
        _build_reviewer_input(app, packet, section_notes),
        job_id,
        stage_detail={"stage": "synthesis", "section_notes": len(section_notes), "coverage": packet.coverage},
    )
    review_content = synthesis.validated_output
    topic_assessments = _assess_topics(app, packet, review_content, job_id)
    return CompletedReading(
        packet=packet,
        review_content=review_content,
        reviewer_prompt_version=synthesis.prompt_version,
        model_identity=synthesis.model_identity,
        server_build=synthesis.server_build,
        reading_record=_build_reading_record(packet, section_notes, review_content),
        topic_assessments=topic_assessments,
        topic_assessor_prompt_version=TOPIC_ASSESSOR_ROLE.prompt_version,
    )


# --------------------------------------------------------------------------------------------------------------------
# Steps 1 and 2: section notes, synthesis
# --------------------------------------------------------------------------------------------------------------------


def _read_sections(
    app: AppContext, packet: ReviewPacket, stage_identity: ReadingStageIdentity, job_id: str | None
) -> list[dict[str, Any]]:
    sections_to_read = packet.reading_plan.sections_to_read
    section_notes = []
    for position, reading_section in enumerate(sections_to_read, start=1):
        section_input = {
            "paper": packet.orientation,
            "reading_section": {
                "section_number": reading_section.section_number,
                "section_count": len(packet.reading_plan.all_sections),
                "section_titles": list(reading_section.section_titles),
            },
            "passages": [passage.as_model_input() for passage in reading_section.passages],
        }
        stage_result = _run_or_load_stage(
            app,
            stage_identity,
            reading_section.stage_key,
            SECTION_READER_ROLE,
            section_input,
            job_id,
            stage_detail={
                "stage": "section",
                "position": position,
                "sections_to_read": len(sections_to_read),
                "section_number": reading_section.section_number,
                "section_count": len(packet.reading_plan.all_sections),
            },
        )
        section_notes.append(
            {
                "section_number": reading_section.section_number,
                "section_titles": list(reading_section.section_titles),
                "notes": stage_result.validated_output,
            }
        )
    return section_notes


def _run_or_load_stage(
    app: AppContext,
    stage_identity: ReadingStageIdentity,
    stage_key: str,
    role_definition: RoleDefinition,
    role_input: RoleInput,
    job_id: str | None,
    stage_detail: dict[str, Any] | None = None,
) -> SavedStageResult:
    """A saved result (an earlier attempt finished this stage) is reused without a model call or activity row. A result
    of a reusable earlier prompt version is used only if it passes the role's current checks on this input."""
    for saved_result in load_saved_stage_results(
        app.connection,
        stage_identity,
        stage_key,
        (role_definition.prompt_version, *role_definition.reusable_earlier_prompt_versions),
    ):
        if saved_result.prompt_version == role_definition.prompt_version or _passes_current_checks(
            role_definition, saved_result, role_input
        ):
            return saved_result
    with recorded_activity(app.connection, "reading_stage", stage_detail or {"stage": stage_key}):
        role_result = run_role_with_one_repair(
            role_definition=role_definition,
            role_input=role_input,
            app_config=app.config,
            model_client=app.model_client,
            connection=app.connection,
            job_id=job_id,
        )
    save_stage_result(app.connection, stage_identity, stage_key, role_result, job_id)
    return SavedStageResult(
        role_result.validated_output,
        role_result.model_response.model_identity,
        role_result.model_response.server_build,
        role_result.prompt_version,
    )


def _passes_current_checks(
    role_definition: RoleDefinition, saved_result: SavedStageResult, role_input: RoleInput
) -> bool:
    try:
        parse_and_validate_role_output(role_definition, json.dumps(saved_result.validated_output), role_input)
    except ModelOutputInvalid:
        return False
    return True


def _build_reviewer_input(app: AppContext, packet: ReviewPacket, section_notes: list[dict[str, Any]]) -> dict[str, Any]:
    reviewer_input: dict[str, Any] = {
        "paper": packet.orientation,
        "coverage": packet.coverage,
        "citable_passages": packet.citable_passage_texts(),
    }
    if packet.reading_plan.is_single_pass:
        reviewer_input["reading_mode"] = READING_MODE_PASSAGES
        reviewer_input["passages"] = [
            passage.as_model_input() for passage in packet.reading_plan.sections_to_read[0].passages
        ]
        return reviewer_input
    reviewer_input["reading_mode"] = READING_MODE_SECTION_NOTES
    reviewer_input["section_notes"] = _fit_section_notes_into_context(app, packet, section_notes)
    return reviewer_input


def _fit_section_notes_into_context(
    app: AppContext, packet: ReviewPacket, section_notes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Section notes are bounded per section, but ten of them can still crowd out the reviewer's reasoning. Drop the
    secondary note fields first; the findings (which carry the evidence) are kept whole whenever possible."""
    available_tokens = (
        app.config.model.required_context_tokens - app.config.limits_for_role(REVIEWER_ROLE.role_name).max_output_tokens
    )

    def fits(candidate_notes: list[dict[str, Any]]) -> bool:
        rendered_size = json.dumps(
            {"paper": packet.orientation, "section_notes": candidate_notes}, ensure_ascii=False, indent=1
        )
        return estimate_prompt_tokens(REVIEWER_ROLE.system_prompt, rendered_size) <= available_tokens

    if fits(section_notes):
        return section_notes
    compact_notes = [
        {
            **entry,
            "notes": {
                field_name: value
                for field_name, value in entry["notes"].items()
                if field_name not in NOTE_FIELDS_DROPPED_FIRST_WHEN_TOO_LONG
            },
        }
        for entry in section_notes
    ]
    return compact_notes  # if even these do not fit, run_role_once refuses the call with a clear reason


# --------------------------------------------------------------------------------------------------------------------
# Step 3: topic assessment
# --------------------------------------------------------------------------------------------------------------------


def _assess_topics(
    app: AppContext, packet: ReviewPacket, review_content: dict[str, Any], job_id: str | None
) -> list[CompletedTopicAssessment]:
    profiles = assessable_topic_profiles(build_research_profiles(app.connection, app.config))
    if not profiles:
        return []
    # Anchors are found by the text screening embedded: the preferred edition's title and abstract.
    screened_edition_row = select_edition_to_read(app.connection, packet.paper_id)
    topic_anchors = build_topic_anchors(
        app.connection,
        app.config,
        paper_id=packet.paper_id,
        paper_title=screened_edition_row["title"],
        paper_abstract=screened_edition_row["abstract"],
        profiles=profiles,
    )
    assessor_input = {
        "paper": {
            "title": packet.orientation["title"],
            "first_published_date": packet.orientation["first_published_date"],
            "coverage": packet.coverage,
        },
        "reading": summarize_review_for_assessment(review_content),
        "topics": [anchors.as_model_input() for anchors in topic_anchors],
    }
    with recorded_activity(
        app.connection, "reading_stage", {"stage": "topic_assessment", "topic_count": len(topic_anchors)}
    ):
        role_result = run_role_with_one_repair(
            role_definition=TOPIC_ASSESSOR_ROLE,
            role_input=assessor_input,
            app_config=app.config,
            model_client=app.model_client,
            connection=app.connection,
            job_id=job_id,
        )
    anchors_by_topic_id: dict[str, TopicAnchors] = {anchors.topic_id: anchors for anchors in topic_anchors}
    profile_hash_by_topic_id = {profile.topic_id: profile.compute_profile_sha256() for profile in profiles}
    completed_assessments = []
    for model_assessment in role_result.validated_output["topic_assessments"]:
        topic_id = model_assessment["topic_id"]
        proposed_ratings = {name: rating["rating"] for name, rating in model_assessment["ratings"].items()}
        proposed_ratings.setdefault(RELATIVE_VALUE.dimension_name, 0)  # ignored when comparison is unavailable
        completed_assessments.append(
            CompletedTopicAssessment(
                topic_id=topic_id,
                model_assessment=model_assessment,
                scored=score_topic_assessment(
                    topic_id,
                    proposed_ratings,
                    comparison_available=anchors_by_topic_id[topic_id].comparison_available,
                    is_abstract_only=packet.coverage == COVERAGE_ABSTRACT_ONLY,
                ),
                topic_profile_sha256=profile_hash_by_topic_id[topic_id],
            )
        )
    return sorted(completed_assessments, key=lambda assessment: assessment.topic_id)


def summarize_review_for_assessment(review_content: dict[str, Any]) -> dict[str, Any]:
    """The validated review as the assessor sees it: statements and claims, without excerpts."""
    return {
        "summary": review_content["one_sentence_summary"],
        "research_question": review_content["research_question"]["statement"],
        "method": review_content["method"]["statement"],
        "evaluation_setting": review_content["evaluation_setting"]["statement"],
        "assumptions": [assumption["statement"] for assumption in review_content["assumptions"]],
        "results": [
            {"claim": finding["claim"], "conditions": finding["conditions"], "passage_id": finding["passage_id"]}
            for finding in review_content["results"]
        ],
        "limitations": [
            {"statement": limitation["statement"], "stated_by": limitation["stated_by"]}
            for limitation in review_content["limitations"]
        ],
        "insufficient_evidence": review_content["insufficient_evidence"],
    }


# --------------------------------------------------------------------------------------------------------------------
# What the review records about its own reading
# --------------------------------------------------------------------------------------------------------------------


def _build_reading_record(
    packet: ReviewPacket, section_notes: list[dict[str, Any]], review_content: dict[str, Any]
) -> dict[str, Any]:
    reported_parser_problems = {
        (problem["passage_id"], problem["problem"])
        for entry in section_notes
        for problem in entry["notes"].get("parser_problems", [])
    }
    reported_parser_problems |= {
        (problem["passage_id"], problem["problem"]) for problem in review_content["parser_problems"]
    }
    reading_record: dict[str, Any] = {
        "text_kind": packet.reading_input.text_kind,
        "edition_id": packet.edition_id,
        "passage_count": len(packet.passages),
        "reading_mode": READING_MODE_PASSAGES if packet.reading_plan.is_single_pass else READING_MODE_SECTION_NOTES,
        **packet.reading_plan.to_record(),
        "parser_problems_reported_by_reviewer": [
            {"passage_id": passage_id, "problem": problem} for passage_id, problem in sorted(reported_parser_problems)
        ],
    }
    if packet.parser_recovery is not None:
        reading_record["parser_recovery"] = packet.parser_recovery.to_record()
    return reading_record
