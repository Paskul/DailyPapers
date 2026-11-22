"""Unit tests for the model contract: reasoning separation, answer validation, outage pause, and worker failures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from conftest import EXAMPLE_TOPIC_TOML, RECORDED_PAPER_FIXTURE, build_in_process_app
from fake_llama_server import FakeLlamaServer

from paperwatch.app_context import AppContext
from paperwatch.catalog import record_collected_paper
from paperwatch.commands.init_command import run_init_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.llama_cpp_client import parse_server_timings
from paperwatch.llm.reasoning import separate_reasoning_from_final_answer
from paperwatch.llm.types import ServerTimings
from paperwatch.roles.base import parse_and_validate_role_output, run_role_once
from paperwatch.roles.healthcheck import HEALTHCHECK_EXPECTED_FINAL_ANSWER, HEALTHCHECK_ROLE
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.sources.fixture_source import RecordedFixtureSource


@pytest.mark.parametrize(
    "content, reasoning_content, expected_answer, expected_reasoning",
    [
        ('{"a": 1}', "thinking", '{"a": 1}', "thinking"),
        ('<think>inline</think>\n{"a": 1}', None, '{"a": 1}', "inline"),
        ('template opened it</think>{"a": 1}', None, '{"a": 1}', "template opened it"),
        ("<think>never finished", None, "", "never finished"),
    ],
)
def test_reasoning_is_separated_from_final_answer(content, reasoning_content, expected_answer, expected_reasoning):
    assert separate_reasoning_from_final_answer(content, reasoning_content) == (expected_answer, expected_reasoning)


PASSAGE_TEXT = "In 200 simulated warehouse episodes the monitor rejected all 37 plans that blocked an exit."
REVIEWER_ROLE_INPUT = {
    "coverage": "abstract_only",
    "paper": {"title": "t"},
    "reading_mode": "passages",
    "passages": [{"passage_id": "A1", "section": "Abstract", "text": PASSAGE_TEXT}],
    "citable_passages": {"A1": PASSAGE_TEXT},
}
INSUFFICIENT = {
    "statement": "insufficient evidence",
    "evidence_status": "insufficient_evidence",
    "supporting_passage_ids": [],
}
VALID_REVIEWER_ANSWER: dict[str, Any] = {
    "one_sentence_summary": "A monitor rejected all 37 exit-blocking plans.",
    "research_question": {
        "statement": "Can plans be checked before dispatch?",
        "evidence_status": "supported",
        "supporting_passage_ids": ["A1"],
    },
    "method": INSUFFICIENT,
    "evaluation_setting": INSUFFICIENT,
    "assumptions": [],
    "results": [
        {
            "claim": "The monitor rejected all 37 blocking plans.",
            "conditions": "200 simulated episodes",
            "passage_id": "A1",
            "excerpt": "the monitor rejected all 37 plans",
        }
    ],
    "limitations": [],
    "reviewer_inferences": [],
    "ambiguous_passages": [],
    "parser_problems": [],
    "insufficient_evidence": False,
}


def test_valid_reviewer_answer_passes():
    validated = parse_and_validate_role_output(REVIEWER_ROLE, json.dumps(VALID_REVIEWER_ANSWER), REVIEWER_ROLE_INPUT)
    assert validated == VALID_REVIEWER_ANSWER


@pytest.mark.parametrize(
    "bad_answer_text, expected_message",
    [
        ("not json", "not valid JSON"),
        (json.dumps({**VALID_REVIEWER_ANSWER, "invented_field": 1}), "violates its schema"),
        (
            json.dumps(
                {**VALID_REVIEWER_ANSWER, "results": [{**VALID_REVIEWER_ANSWER["results"][0], "passage_id": "P99"}]}
            ),
            "not in the input packet",
        ),
        (
            json.dumps(
                {
                    **VALID_REVIEWER_ANSWER,
                    "results": [{**VALID_REVIEWER_ANSWER["results"][0], "claim": "Accuracy was 97.3%."}],
                }
            ),
            "97.3",
        ),
    ],
)
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
    promote_paper_directly_and_queue_review(
        app.connection, record_collected_paper(app.connection, record).paper_id, max_attempts=3
    )
    return app


def test_invalid_model_output_gets_one_repair_then_fails_the_job_with_its_reason(tmp_path):
    """One bounded repair attempt; on repeated failure retain a reason and move on."""
    fake_model = FakeModelClient({"reviewer": json.dumps({"wrong": "shape"})})
    app = _load_app_with_one_queued_paper(tmp_path, fake_model)

    cycle_report = run_single_work_cycle(app, probe_paused_model_now=False)
    assert cycle_report.outcome == WorkCycleOutcome.JOB_FAILED_PERMANENTLY
    assert "after one repair attempt" in cycle_report.message
    assert app.connection.execute("SELECT status FROM papers").fetchone()[0] == "failed"
    [job_row] = app.connection.execute("SELECT status, last_error FROM jobs").fetchall()
    assert job_row["status"] == "failed" and "violates its schema" in job_row["last_error"]
    assert [
        row[0]
        for row in app.connection.execute(
            "SELECT prompt_version FROM model_calls WHERE outcome = 'invalid_output' ORDER BY started_at, rowid"
        )
    ] == [REVIEWER_ROLE.prompt_version, REVIEWER_ROLE.prompt_version + "+repair"]
    assert "REPAIR REQUEST" in fake_model.received_requests[1].user_message
    assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.NO_JOB_DUE


def test_a_repaired_answer_is_accepted(tmp_path):
    from paperwatch.llm.fake_answers import build_fake_review

    fake_model = FakeModelClient(
        {
            "reviewer": [
                json.dumps({"wrong": "shape"}),
                lambda request: json.dumps(build_fake_review(request.role_input or {})),
            ]
        }
    )
    app = _load_app_with_one_queued_paper(tmp_path, fake_model)
    assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_COMPLETED
    assert app.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1


def test_empty_queue_issues_zero_inference_calls(tmp_path):
    fake_model = FakeModelClient()
    app = _load_app_with_one_queued_paper(tmp_path, fake_model)
    assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_COMPLETED
    requests_after_the_job = len(fake_model.received_requests)  # reviewer + topic assessor
    assert requests_after_the_job == 2
    for _ in range(5):
        assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.NO_JOB_DUE
    assert len(fake_model.received_requests) == requests_after_the_job


@pytest.mark.live_model
@pytest.mark.skipif(os.environ.get("PAPERWATCH_LIVE_MODEL") != "1", reason="set PAPERWATCH_LIVE_MODEL=1")
def test_live_server_reviews_recorded_paper(tmp_path):
    """Optional: the same vertical slice against the real local llama.cpp server."""
    from paperwatch.config import ModelSettings
    from paperwatch.llm.llama_cpp_client import LlamaCppModelClient

    app = _load_app_with_one_queued_paper(tmp_path, LlamaCppModelClient(ModelSettings()))
    cycle_report = run_single_work_cycle(app, probe_paused_model_now=True)
    assert cycle_report.outcome == WorkCycleOutcome.JOB_COMPLETED, cycle_report.message


# --------------------------------------------------------------------------------------------------------------------
# llama.cpp's timings, for `paperwatch capacity`
# --------------------------------------------------------------------------------------------------------------------


def test_llama_cpp_timings_are_read_and_malformed_values_are_left_out():
    timings = parse_server_timings(
        {
            "cache_n": 2048,
            "prompt_n": 9000,
            "prompt_ms": 46153.8,
            "predicted_n": 6182,
            "predicted_ms": 272321.6,
            "draft_n": 4100,
            "draft_n_accepted": 2911,
        }
    )
    assert timings.prompt_seconds == pytest.approx(46.1538)
    assert timings.generation_seconds == pytest.approx(272.3216)
    assert (timings.cached_prompt_tokens, timings.draft_tokens, timings.draft_tokens_accepted) == (2048, 4100, 2911)
    assert parse_server_timings(None) == ServerTimings()  # a server that reports no timings
    malformed = {"prompt_ms": "fast", "predicted_ms": -1, "cache_n": True, "draft_n": 1.5, "draft_n_accepted": None}
    assert parse_server_timings(malformed) == ServerTimings()


def test_a_model_call_through_llama_cpp_stores_its_timings(tmp_path):
    with FakeLlamaServer(lambda chat_payload: HEALTHCHECK_EXPECTED_FINAL_ANSWER) as fake_server:
        app = build_in_process_app(
            tmp_path / "home",
            use_configured_model_backend=True,
            config_replacements={'"http://127.0.0.1:8080/v1"': f'"{fake_server.base_url}"'},
        )
        run_role_once(
            role_definition=HEALTHCHECK_ROLE,
            role_input={},
            app_config=app.config,
            model_client=app.model_client,
            connection=app.connection,
        )
    stored_call = app.connection.execute(
        "SELECT outcome, prompt_seconds, generation_seconds, cached_prompt_tokens, draft_tokens FROM model_calls"
    ).fetchone()
    assert tuple(stored_call) == ("ok", 0.002, 0.001, 0, None)  # the fake server's timings; it drafts nothing
    app.close()
