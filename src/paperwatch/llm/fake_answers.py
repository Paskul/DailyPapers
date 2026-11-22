"""Valid, deterministic answers for the evidence-reading roles, built from the role input (offline fake backend).

The fake model has no weights, so it cannot read. These builders produce the smallest answers that pass every host
check: they quote the first words of the first passage they were given, cite it, and rate every topic 2. They let
the offline vertical slice and the tests drive the real pipeline (packets, evidence checks, stage results, scoring,
notes) end to end. Text containing digits is avoided except inside quotations, so the number check always passes.
Keyed by role name; a role without a builder gets the schema's minimal instance (paperwatch.llm.fake_client).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

FAKE_EXCERPT_WORDS = 8


def _excerpt_of(passage_text: str) -> str | None:
    words = passage_text.split()
    return " ".join(words[:FAKE_EXCERPT_WORDS]) if len(words) >= 3 else None


def _first_quotable_passage(passages: list[Mapping[str, Any]]) -> tuple[str, str] | None:
    for passage in passages:
        excerpt = _excerpt_of(passage["text"])
        if excerpt:
            return passage["passage_id"], excerpt
    return None


def _insufficient(statement: str = "insufficient evidence (offline fake model)") -> dict[str, Any]:
    return {"statement": statement, "evidence_status": "insufficient_evidence", "supporting_passage_ids": []}


def build_fake_section_notes(role_input: Mapping[str, Any]) -> dict[str, Any]:
    quotable = _first_quotable_passage(role_input["passages"])
    findings = (
        []
        if quotable is None
        else [
            {
                "claim": quotable[1],
                "conditions": "",
                "passage_id": quotable[0],
                "excerpt": quotable[1],
                "finding_kind": "other",
            }
        ]
    )
    return {
        "section_summary": "Offline fake notes of this section.",
        "findings": findings,
        "reviewer_inferences": [],
        "ambiguous_passages": [],
        "parser_problems": [],
        "insufficient_evidence": quotable is None,
    }


def build_fake_review(role_input: Mapping[str, Any]) -> dict[str, Any]:
    if role_input["reading_mode"] == "passages":
        quotable = _first_quotable_passage(role_input["passages"])
    else:
        quoted_findings = [finding for entry in role_input["section_notes"] for finding in entry["notes"]["findings"]]
        quotable = (quoted_findings[0]["passage_id"], quoted_findings[0]["excerpt"]) if quoted_findings else None
    review = {
        "one_sentence_summary": "Offline fake review: the fake model quotes the opening words of the text.",
        "research_question": _insufficient(),
        "method": _insufficient(),
        "evaluation_setting": _insufficient(),
        "assumptions": [],
        "results": [],
        "limitations": [],
        "reviewer_inferences": [],
        "ambiguous_passages": [],
        "parser_problems": [],
        "insufficient_evidence": quotable is None,
    }
    if quotable is not None:
        passage_id, excerpt = quotable
        review["research_question"] = {
            "statement": "The opening passage states what the paper is about.",
            "evidence_status": "supported",
            "supporting_passage_ids": [passage_id],
        }
        review["results"] = [{"claim": excerpt, "conditions": "", "passage_id": passage_id, "excerpt": excerpt}]
    return review


def build_fake_topic_assessments(role_input: Mapping[str, Any]) -> dict[str, Any]:
    assessments = []
    for topic in role_input["topics"]:
        ratings = {
            dimension_name: {"rating": 2, "explanation": "Offline fake rating."}
            for dimension_name in ("question_fit", "method_fit", "evidence_strength")
        }
        if topic["comparison_available"]:
            ratings["relative_value"] = {"rating": 2, "explanation": "Offline fake rating."}
        assessments.append(
            {
                "topic_id": topic["topic_id"],
                "ratings": ratings,
                "rationale": "Offline fake assessment.",
                "matched_seed_ids": [anchor["seed_id"] for anchor in topic["seed_anchors"][:1]],
                "compared_review_ids": [anchor["review_id"] for anchor in topic["review_anchors"][:1]],
            }
        )
    return {"topic_assessments": assessments}


def build_fake_reference_selection(role_input: Mapping[str, Any]) -> dict[str, Any]:
    """Part 8 research role: choose the first candidate as a method source (citing its first context), no gap."""
    selections = []
    for candidate in role_input["candidates"][:1]:
        selections.append(
            {
                "candidate_id": candidate["candidate_id"],
                "role_in_parent": "method",
                "importance": 3,
                "explanation": "Offline fake choice: the first candidate offered.",
                "context_passage_ids": [context["passage_id"] for context in candidate["citation_contexts"][:1]],
            }
        )
    return {"selected_references": selections[: role_input["maximum_selections"]], "search_gap": []}


def build_fake_comparison(role_input: Mapping[str, Any]) -> dict[str, Any]:
    """Part 8 comparison role: quote the first passage of each review; conditions only partially comparable, so it
    names a condition difference and ranks nothing."""
    evidence = []
    for review in role_input["reviews"]:
        quotable = _first_quotable_passage(review["passages"])
        if quotable is not None:
            evidence.append({"review_id": review["review_id"], "passage_id": quotable[0], "excerpt": quotable[1]})
    both_cited = len(evidence) == len(role_input["reviews"])
    return {
        "conditions_comparability": "partially_comparable",
        "condition_differences": [
            {
                "dimension": "evaluation_setting",
                "statement": "Offline fake: the two evaluation settings are not the same.",
                "evidence": evidence[:1],
            }
        ]
        if evidence
        else [],
        "agreements": [{"statement": "Offline fake: both papers address the same question.", "evidence": evidence}]
        if both_cited
        else [],
        "differences": [],
        "result_ranking": [],
        "reader_conclusion": {
            "statement": "Offline fake conclusion: read both with their conditions in mind.",
            "evidence": evidence if both_cited else [],
        },
        "insufficient_evidence": not both_cited,
    }


def build_fake_report_synthesis(role_input: Mapping[str, Any]) -> dict[str, Any]:
    """Part 9 writer: quote the first passage of each paper; one connected finding over the first two papers."""
    evidence_by_review_id: dict[str, list[dict[str, str]]] = {}
    for paper in role_input["papers"]:
        quotable = _first_quotable_passage(paper["passages"])
        evidence_by_review_id[paper["review_id"]] = (
            []
            if quotable is None
            else [{"review_id": paper["review_id"], "passage_id": quotable[0], "excerpt": quotable[1]}]
        )
    all_evidence = [item for items in evidence_by_review_id.values() for item in items]
    first_review_id = role_input["papers"][0]["review_id"]
    return {
        "overview": {"statement": "Offline fake overview of today's papers.", "evidence": all_evidence[:2]},
        "connected_findings": [
            {
                "statement": "Offline fake finding that connects the first papers.",
                "evidence": all_evidence[:2],
                "comparison_ids": [comparison["comparison_id"] for comparison in role_input["comparisons"][:1]],
            }
        ]
        if all_evidence
        else [],
        "read_first": {
            "review_id": first_review_id,
            "explanation": "Offline fake: read the first paper first.",
            "evidence": evidence_by_review_id[first_review_id],
        },
        "paper_reasons": [
            {"review_id": review_id, "reason_to_read": "Offline fake reason to read this paper.", "evidence": evidence}
            for review_id, evidence in evidence_by_review_id.items()
        ],
    }


FAKE_ANSWER_BUILDERS_BY_ROLE: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "section_reader": build_fake_section_notes,
    "reviewer": build_fake_review,
    "topic_assessor": build_fake_topic_assessments,
    "research": build_fake_reference_selection,
    "comparison": build_fake_comparison,
    "writer": build_fake_report_synthesis,
}
