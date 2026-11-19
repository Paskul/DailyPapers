"""Unit tests for the model contract: reasoning separation, answer validation, outage pause, and worker failures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paperwatch.app_context import AppContext
from paperwatch.catalog import record_collected_paper
from paperwatch.commands.collect_command import _queue_review_if_promoted
from paperwatch.commands.init_command import run_init_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.reasoning import separate_reasoning_from_final_answer
from paperwatch.roles.base import parse_and_validate_role_output
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.sources.fixture_source import RecordedFixtureSource

from conftest import EXAMPLE_TOPIC_TOML, RECORDED_PAPER_FIXTURE


@pytest.mark.parametrize("content, reasoning_content, expected_answer, expected_reasoning", [
    ('{"a": 1}', "thinking", '{"a": 1}', "thinking"),
    ('<think>inline</think>\n{"a": 1}', None, '{"a": 1}', "inline"),
    ('template opened it</think>{"a": 1}', None, '{"a": 1}', "template opened it"),
    ("<think>never finished", None, "", "never finished"),
])
def test_reasoning_is_separated_from_final_answer(content, reasoning_content, expected_answer, expected_reasoning):
    assert separate_reasoning_from_final_answer(content, reasoning_content) == (expected_answer, expected_reasoning)


REVIEWER_ROLE_INPUT = {"coverage": "abstract_only", "paper": {"title": "t"},
                       "research_topics": [{"topic_id": "team-safety", "question": "q"}]}
VALID_REVIEWER_ANSWER = {"one_sentence_summary": "s", "research_question": "q", "method": "m",
                         "authors_main_claims": ["c"], "stated_limitations": [],
                         "topic_relevance": [{"topic_id": "team-safety", "relevance_rating": 3, "rationale": "r"}],
                         "insufficient_evidence": False}


def test_valid_reviewer_answer_passes():
    validated = parse_and_validate_role_output(REVIEWER_ROLE, json.dumps(VALID_REVIEWER_ANSWER), REVIEWER_ROLE_INPUT)
    assert validated == VALID_REVIEWER_ANSWER


@pytest.mark.parametrize("bad_answer_text, expected_message", [
    ("not json", "not valid JSON"),
    (json.dumps({**VALID_REVIEWER_ANSWER, "invented_field": 1}), "violates its schema"),
    (json.dumps({**VALID_REVIEWER_ANSWER, "topic_relevance": [
        {"topic_id": "team-safety", "relevance_rating": 9, "rationale": "r"}]}), "violates its schema"),
    (json.dumps({**VALID_REVIEWER_ANSWER, "topic_relevance": [
        {"topic_id": "invented-topic", "relevance_rating": 2, "rationale": "r"}]}), "unknown topic id"),
])
def test_invalid_reviewer_answers_are_rejected(bad_answer_text, expected_message):
    with pytest.raises(ModelOutputInvalid, match=expected_message):
        parse_and_validate_role_output(REVIEWER_ROLE, bad_answer_text, REVIEWER_ROLE_INPUT)


def _load_app_with_one_queued_paper(home_dir: Path, model_client) -> AppContext:
    run_init_command(home_dir, None)
    (home_dir / "topics.toml").write_text(EXAMPLE_TOPIC_TOML)
    config_path = home_dir / "config.toml"  # retries become due immediately so the test can drain them
    config_path.write_text(config_path.read_text().replace("retry_delay_seconds = 600", "retry_delay_seconds = 0"))
    app = AppContext.load(home_dir, model_client=model_client)
    [record] = RecordedFixtureSource(RECORDED_PAPER_FIXTURE).fetch_records()
    _queue_review_if_promoted(app, record_collected_paper(app.connection, record).paper_id)
    return app


def test_invalid_model_output_charges_attempts_then_fails_permanently(tmp_path):
    fake_model = FakeModelClient({"reviewer": json.dumps({"wrong": "shape"})})
    app = _load_app_with_one_queued_paper(tmp_path, fake_model)

    outcomes = [run_single_work_cycle(app, probe_paused_model_now=False).outcome for _ in range(3)]
    assert outcomes == [WorkCycleOutcome.JOB_RETRY_SCHEDULED, WorkCycleOutcome.JOB_RETRY_SCHEDULED,
                        WorkCycleOutcome.JOB_FAILED_PERMANENTLY]
    assert app.connection.execute("SELECT status FROM papers").fetchone()[0] == "failed"
    assert app.connection.execute(
        "SELECT COUNT(*) FROM model_calls WHERE outcome = 'invalid_output'").fetchone()[0] == 3


def test_empty_queue_issues_zero_inference_calls(tmp_path):
    fake_model = FakeModelClient()
    app = _load_app_with_one_queued_paper(tmp_path, fake_model)
    assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_COMPLETED
    for _ in range(5):
        assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.NO_JOB_DUE
    assert len(fake_model.received_requests) == 1


@pytest.mark.live_model
@pytest.mark.skipif(os.environ.get("PAPERWATCH_LIVE_MODEL") != "1", reason="set PAPERWATCH_LIVE_MODEL=1")
def test_live_server_reviews_recorded_paper(tmp_path):
    """Optional: the same vertical slice against the real local llama.cpp server."""
    from paperwatch.llm.llama_cpp_client import LlamaCppModelClient
    from paperwatch.config import ModelSettings
    app = _load_app_with_one_queued_paper(tmp_path, LlamaCppModelClient(ModelSettings()))
    cycle_report = run_single_work_cycle(app, probe_paused_model_now=True)
    assert cycle_report.outcome == WorkCycleOutcome.JOB_COMPLETED, cycle_report.message
