"""Host corrections added after the second pilot (2026-11-22), one group per problem the pilot showed:

joined ids     the reviewer joined passage ids into one entry ("P18P19", "'P19', 'P20'", "P35-P37") when its method
               statement wanted more than 6 passages: the entry is split, and the single statements may cite 10
numbers        a statement stated a number from a passage it did not cite: the ONE passage that states it is added;
               a model name such as "GPT-6" is not a stated number
writer         the writer's answer gets "fix, then drop": miscited excerpts move, failing reasons and findings drop
saved answers  a rejected or corrected answer is kept in model_calls.answer_text for offline replay
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import build_in_process_app
from test_daily_report import WRITER_INPUT, validate_writer_answer, writer_answer
from test_model_contract import INSUFFICIENT

from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.reading.evidence_checks import find_numbers_needing_evidence
from paperwatch.roles.answer_salvage import split_joined_passage_ids
from paperwatch.roles.base import parse_and_validate_role_output, run_role_with_one_repair
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.roles.writer import WRITER_ROLE

PASSAGE_TEXTS = {
    "P1": "We train the guard with LoRA on the agent traces collected from the benchmark suite.",
    "P2": "Training uses rank 16 and alpha 32 with a learning rate chosen on the validation split.",
    "P3": "The guard blocked 412 of 450 unsafe actions in the held-out evaluation episodes.",
    "P4": "Each configuration was evaluated with 3 random seeds and 3 repetitions per task.",
    "P5": "Ablations remove the memory module and the trajectory summarizer in turn.",
    "P6": "We compare against GPT-6 and the prompting baseline from the prior work.",
}
REVIEWER_INPUT: dict[str, Any] = {
    "coverage": "full_text",
    "paper": {"title": "A guard"},
    "reading_mode": "passages",
    "passages": [
        {"passage_id": passage_id, "section": "Method", "text": text} for passage_id, text in PASSAGE_TEXTS.items()
    ],
    "citable_passages": PASSAGE_TEXTS,
}
RESULT = {
    "claim": "The guard blocked 412 of 450 unsafe actions.",
    "conditions": "held-out evaluation episodes",
    "passage_id": "P3",
    "excerpt": "The guard blocked 412 of 450 unsafe actions",
}


def reviewer_answer(method_statement: str, method_passage_ids: list[str], **overrides) -> dict:
    answer = {
        "one_sentence_summary": "A trained guard blocks most unsafe agent actions.",
        "research_question": INSUFFICIENT,
        "method": {
            "statement": method_statement,
            "evidence_status": "supported",
            "supporting_passage_ids": method_passage_ids,
        },
        "evaluation_setting": INSUFFICIENT,
        "assumptions": [],
        "results": [RESULT],
        "limitations": [],
        "reviewer_inferences": [],
        "ambiguous_passages": [],
        "parser_problems": [],
        "insufficient_evidence": False,
    }
    answer.update(overrides)
    return answer


def rejection_of(role_definition, answer: dict, role_input) -> ModelOutputInvalid:
    with pytest.raises(ModelOutputInvalid) as rejection:
        parse_and_validate_role_output(role_definition, json.dumps(answer), role_input)
    return rejection.value


# --------------------------------------------------------------------------------------------------------------------
# Joined passage ids
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cited_entry, expected_ids",
    [
        ("P2", ["P2"]),
        ("P1P2", ["P1", "P2"]),
        ("P1', 'P2', 'P4", ["P1", "P2", "P4"]),
        ("['P1', 'P2']", ["P1", "P2"]),
        ("P1, P3 and P5", ["P1", "P3", "P5"]),
        ("P2-P4", ["P2", "P3", "P4"]),
        ("P2–4", ["P2", "P3", "P4"]),
        ("P12", None),  # one unknown id; never read as P1 + P2
        ("P1P9", None),  # a part that is not in the input
        ("P1-P300", None),  # a range is a short run of neighbours, not the whole paper
        ("P4-P2", None),
        ("P1P2 joined as P1-P2", None),  # prose is not an id list
    ],
)
def test_a_joined_passage_id_entry_is_split_only_into_known_ids(cited_entry, expected_ids):
    assert split_joined_passage_ids(cited_entry, PASSAGE_TEXTS) == expected_ids


def test_a_method_with_joined_ids_is_corrected_without_a_repair():
    answer = reviewer_answer("LoRA with rank 16 and alpha 32 on agent traces.", ["P1P2", "P3-P5"])
    salvage = rejection_of(REVIEWER_ROLE, answer, REVIEWER_INPUT).salvage
    assert salvage is not None and salvage.dropped_items == ()
    assert salvage.salvaged_output["method"]["supporting_passage_ids"] == ["P1", "P2", "P3", "P4", "P5"]
    assert "split 2 joined passage id(s)" in salvage.describe()


def test_the_single_statements_may_cite_ten_passages_but_not_more():
    ten_ids = list(PASSAGE_TEXTS) * 2  # duplicates are harmless for the schema check; the count is what matters
    validated = parse_and_validate_role_output(
        REVIEWER_ROLE, json.dumps(reviewer_answer("LoRA on agent traces.", ten_ids[:10])), REVIEWER_INPUT
    )
    assert len(validated["method"]["supporting_passage_ids"]) == 10
    assert "violates its schema" in str(
        rejection_of(REVIEWER_ROLE, reviewer_answer("LoRA on agent traces.", ten_ids[:11]), REVIEWER_INPUT)
    )


# --------------------------------------------------------------------------------------------------------------------
# Numbers from an uncited passage
# --------------------------------------------------------------------------------------------------------------------


def test_the_one_passage_that_states_an_uncited_number_is_added_to_the_statement():
    answer = reviewer_answer("LoRA with rank 16 and alpha 32 on agent traces.", ["P1"])
    salvage = rejection_of(REVIEWER_ROLE, answer, REVIEWER_INPUT).salvage
    assert salvage is not None
    assert salvage.salvaged_output["method"]["supporting_passage_ids"] == ["P1", "P2"]
    assert "method: + P2 (states 16, 32)" in salvage.describe()
    assert salvage.salvaged_output["method"]["statement"] == answer["method"]["statement"]  # the text is never changed


def test_a_number_stated_by_several_passages_or_by_none_is_not_given_a_citation():
    in_two_passages = reviewer_answer("Each run used 3 seeds and 450 episodes.", ["P1"])  # 3: P4 only; 450: P3 only
    assert rejection_of(REVIEWER_ROLE, in_two_passages, REVIEWER_INPUT).salvage is not None
    ambiguous_input = {**REVIEWER_INPUT, "citable_passages": {**PASSAGE_TEXTS, "P7": "We repeat it 3 times."}}
    assert rejection_of(REVIEWER_ROLE, in_two_passages, ambiguous_input).salvage is None  # 3 is in P4 and P7
    invented = reviewer_answer("LoRA with rank 64 on agent traces.", ["P1"])
    assert rejection_of(REVIEWER_ROLE, invented, REVIEWER_INPUT).salvage is None  # 64 is nowhere: still a repair


def test_an_uncited_number_in_a_droppable_item_is_cited_instead_of_dropped():
    assumption = {
        "statement": "The guard is evaluated with 3 random seeds.",
        "evidence_status": "supported",
        "supporting_passage_ids": ["P5"],
    }
    answer = reviewer_answer("LoRA on agent traces.", ["P1"], assumptions=[assumption])
    salvage = rejection_of(REVIEWER_ROLE, answer, REVIEWER_INPUT).salvage
    assert salvage is not None and salvage.dropped_items == ()
    assert salvage.salvaged_output["assumptions"][0]["supporting_passage_ids"] == ["P5", "P4"]


@pytest.mark.parametrize(
    "stated_text, expected_numbers",
    [
        ("compared with GPT-6 and Llama-3", []),
        ("COVID-19 records and ResNet-50 features", []),
        ("from 10-20 runs at -3 dB", ["10", "20", "-3"]),
        ("GPT-6 reached 92.4%", ["92.4"]),
    ],
)
def test_a_digit_joined_to_a_word_by_a_hyphen_names_a_thing(stated_text, expected_numbers):
    assert find_numbers_needing_evidence(stated_text) == expected_numbers


# --------------------------------------------------------------------------------------------------------------------
# The writer
# --------------------------------------------------------------------------------------------------------------------

EVIDENCE_A = {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}
EVIDENCE_B = {"review_id": "r_b", "passage_id": "P1", "excerpt": "guarantees exit clearance for exact"}


def test_a_writer_excerpt_cited_from_the_wrong_passage_of_its_review_is_moved():
    two_passage_input = json.loads(json.dumps(WRITER_INPUT))
    two_passage_input["papers"][0]["passages"].append(
        {"passage_id": "P2", "location": "p. 2", "text": "Plans were checked before dispatch to the robots."}
    )
    miscited = {"review_id": "r_a", "passage_id": "P1", "excerpt": "checked before dispatch to the robots"}
    answer = writer_answer(
        read_first={"review_id": "r_a", "explanation": "It checks plans first.", "evidence": [miscited]}
    )
    rejection = rejection_of(WRITER_ROLE, answer, two_passage_input)
    assert rejection.salvage is not None
    assert rejection.salvage.salvaged_output["read_first"]["evidence"][0]["passage_id"] == "P2"
    assert "read_first.evidence[1]: P1 -> P2" in rejection.salvage.describe()


def test_a_failing_paper_reason_is_dropped_and_the_rest_of_the_writer_answer_kept():
    answer = writer_answer(
        paper_reasons=[
            {"review_id": "r_a", "reason_to_read": "Simulation evidence.", "evidence": [EVIDENCE_A]},
            {"review_id": "r_b", "reason_to_read": "It proves 99 cases.", "evidence": [EVIDENCE_B]},  # 99: invented
        ]
    )
    salvage = rejection_of(WRITER_ROLE, answer, WRITER_INPUT).salvage
    assert salvage is not None and salvage.dropped_at_most_half
    assert [reason["review_id"] for reason in salvage.salvaged_output["paper_reasons"]] == ["r_a"]
    assert salvage.salvaged_output["overview"] == answer["overview"]


def test_a_failing_connected_finding_and_an_unknown_comparison_id_are_dropped():
    answer = writer_answer(
        connected_findings=[
            {"statement": "Both address exits.", "evidence": [EVIDENCE_A, EVIDENCE_B], "comparison_ids": ["c_404"]},
            {"statement": "Both reject 50 plans.", "evidence": [EVIDENCE_A], "comparison_ids": []},  # 50: invented
        ]
    )
    salvage = rejection_of(WRITER_ROLE, answer, WRITER_INPUT).salvage
    assert salvage is not None
    [kept_finding] = salvage.salvaged_output["connected_findings"]
    assert kept_finding["statement"] == "Both address exits." and kept_finding["comparison_ids"] == []


def test_a_writer_overview_that_still_fails_needs_the_repair():
    answer = writer_answer(overview={"statement": "One rejected 97 of 99 plans.", "evidence": [EVIDENCE_A]})
    assert rejection_of(WRITER_ROLE, answer, WRITER_INPUT).salvage is None
    invented_address = writer_answer(
        paper_reasons=[
            {"review_id": "r_a", "reason_to_read": "See https://evil.example/x.", "evidence": [EVIDENCE_A]},
            {"review_id": "r_b", "reason_to_read": "The exact-position assumption.", "evidence": [EVIDENCE_B]},
        ]
    )
    assert rejection_of(WRITER_ROLE, invented_address, WRITER_INPUT).salvage is None


def test_a_valid_writer_answer_is_unchanged():
    assert validate_writer_answer(writer_answer()) == writer_answer()


# --------------------------------------------------------------------------------------------------------------------
# Saved answers
# --------------------------------------------------------------------------------------------------------------------


def test_rejected_and_corrected_answers_are_kept_for_replay(tmp_path):
    invented = json.dumps(reviewer_answer("LoRA with rank 64 on agent traces.", ["P1"]))
    corrected = json.dumps(reviewer_answer("LoRA with rank 16 on agent traces.", ["P1"]))
    fake_model = FakeModelClient()
    fake_model._canned_final_answers_by_role["reviewer"] = [invented, corrected]
    app = build_in_process_app(tmp_path / "home", model_client=fake_model)
    try:
        result = run_role_with_one_repair(
            role_definition=REVIEWER_ROLE,
            role_input=REVIEWER_INPUT,
            app_config=app.config,
            model_client=fake_model,
            connection=app.connection,
        )
        assert result.validated_output["method"]["supporting_passage_ids"] == ["P1", "P2"]
        call_rows = app.connection.execute(
            "SELECT outcome, answer_text FROM model_calls ORDER BY started_at, rowid"
        ).fetchall()
        assert [(row["outcome"], row["answer_text"]) for row in call_rows] == [
            ("invalid_output", invented),
            ("ok", corrected),
        ]
    finally:
        app.close()
