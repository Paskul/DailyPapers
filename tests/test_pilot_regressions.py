"""Fixes after the first pilot (2026-11-21), one group per problem the pilot showed:

citations    numbers spelled differently, a quotation cited from the wrong passage, and "fix, then drop": the
             host corrects or drops failing items instead of throwing the whole answer away (roles/answer_salvage)
stages       saved stages of an earlier prompt version are reused only if they pass today's checks
stopping     SIGTERM aborts a model call and puts the job back without using an attempt (worker_stop.py)
files        reports and notes are readable by the shared group (rendering/file_output.py)
budget       arXiv category feeds share a capped part of the daily budget fairly, best topic matches first
retry        `retry-failed` requeues failed jobs and keeps their saved stages
"""

from __future__ import annotations

import argparse
import json
import signal
import stat
import time

import pytest
from conftest import RECORDED_PAPER_FIXTURE, PaperwatchHome, build_in_process_app
from test_evidence_reading import add_paper
from test_model_contract import REVIEWER_ROLE_INPUT, VALID_REVIEWER_ANSWER, _load_app_with_one_queued_paper

from paperwatch.collection.candidate_budget import CandidateAdmissionBudget, water_fill
from paperwatch.collection.topic_word_ranking import rank_records_by_topic_words
from paperwatch.commands.retry_failed_command import run_retry_failed_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.config import CollectionSettings
from paperwatch.errors import ModelOutputInvalid, ModelOutputRejectedAfterRepair
from paperwatch.llm import FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.types import StructuredChatResponse
from paperwatch.reading.evidence_checks import check_numbers_occur_in_passages
from paperwatch.reading.reading_pipeline import _run_or_load_stage
from paperwatch.reading.stage_results import ReadingStageIdentity, save_stage_result
from paperwatch.rendering.file_output import OUTPUT_FILE_MODE, write_output_file_atomically
from paperwatch.roles.base import RoleRunResult, parse_and_validate_role_output, run_role_with_one_repair
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.roles.section_reader import SECTION_READER_ROLE
from paperwatch.sources.discovery_routes import RouteKind
from paperwatch.sources.types import CollectedPaperRecord
from paperwatch.worker_stop import (
    WorkerStopRequested,
    interruptible_model_wait,
    request_worker_stop,
    reset_worker_stop_request,
)

# --------------------------------------------------------------------------------------------------------------------
# Citations
# --------------------------------------------------------------------------------------------------------------------

PASSAGES = [
    {
        "passage_id": "P1",
        "section": "3 Results",
        "text": "Table 1 reports the rejection rate of the monitor across all benchmarks we study.",
    },
    {
        "passage_id": "P2",
        "section": "3 Results",
        "text": "Rejection rate 0.924 on the warehouse benchmark and 0.871 on the depot benchmark.",
    },
    {
        "passage_id": "P3",
        "section": "3 Results",
        "text": "The monitor was evaluated on three benchmarks with five robots each.",
    },
    {
        "passage_id": "P4",
        "section": "4 Discussion",
        "text": "The monitor was evaluated on three benchmarks with five robots each, as described above.",
    },
]
SECTION_INPUT = {
    "paper": {"title": "A monitor"},
    "passages": PASSAGES,
    "reading_section": {"section_number": 1, "section_count": 1, "section_titles": ["3 Results"]},
}
PASSAGE_TEXT_BY_ID = {passage["passage_id"]: passage["text"] for passage in PASSAGES}


def finding(claim: str, passage_id: str, excerpt: str) -> dict:
    return {"claim": claim, "conditions": "", "passage_id": passage_id, "excerpt": excerpt, "finding_kind": "result"}


GOOD_FINDING = finding("The rejection rate is reported for all benchmarks.", "P1", "reports the rejection rate")
MISCITED_FINDING = finding(
    "The rejection rate was 92.4% on the warehouse benchmark.", "P1", "Rejection rate 0.924 on the warehouse benchmark"
)  # the quote and number are in P2
INVENTED_FINDING = finding("Accuracy was 99.1%.", "P1", "rejection rate of the monitor")  # 99.1 is nowhere
INVENTED_QUOTE = finding("The monitor never fails.", "P2", "the monitor never fails in any setting")


def section_answer(*findings: dict, inferences: tuple[dict, ...] = ()) -> str:
    return json.dumps(
        {
            "section_summary": "Results of the monitor.",
            "findings": list(findings),
            "reviewer_inferences": list(inferences),
            "ambiguous_passages": [],
            "parser_problems": [],
            "insufficient_evidence": False,
        }
    )


@pytest.mark.parametrize(
    "stated_text, passage_text, is_backed",
    [
        ("evaluated on 3 benchmarks with 5 robots", "evaluated on three benchmarks with five robots", True),
        ("a rejection rate of 92.4%", "rejection rate 0.924", True),
        ("a rejection rate of 0.924", "a rejection rate of 92.4%", True),
        ("a rejection rate of 92.5%", "rejection rate 0.924", False),
        ("evaluated on 4 benchmarks", "evaluated on three benchmarks", False),
    ],
)
def test_the_same_value_written_differently_counts_as_occurring(stated_text, passage_text, is_backed):
    problems = check_numbers_occur_in_passages(stated_text, ["P1"], {"P1": passage_text}, "claim")
    assert (problems == []) == is_backed


@pytest.fixture
def role_app(tmp_path):
    fake_model = FakeModelClient()
    app = build_in_process_app(tmp_path / "home", model_client=fake_model)
    yield app, fake_model
    app.close()


def run_section_reader(app, fake_model, *answers: str):
    fake_model._canned_final_answers_by_role["section_reader"] = list(answers)
    return run_role_with_one_repair(
        role_definition=SECTION_READER_ROLE,
        role_input=SECTION_INPUT,
        app_config=app.config,
        model_client=fake_model,
        connection=app.connection,
    )


def test_a_quote_cited_from_the_wrong_passage_is_moved_and_an_invented_number_is_dropped_without_a_repair(role_app):
    app, fake_model = role_app
    result = run_section_reader(app, fake_model, section_answer(GOOD_FINDING, MISCITED_FINDING, INVENTED_FINDING))

    assert len(fake_model.requests_for_role("section_reader")) == 1  # 1 of 3 dropped: no repair call
    assert [kept["passage_id"] for kept in result.validated_output["findings"]] == ["P1", "P2"]
    assert result.validated_output["findings"][1]["claim"] == MISCITED_FINDING["claim"]  # only the id changed
    assert "findings[2]: P1 -> P2" in result.host_corrections and "dropped 1 of 3" in result.host_corrections
    assert "99.1" in result.host_corrections
    [call_row] = app.connection.execute("SELECT outcome, error_message FROM model_calls").fetchall()
    assert call_row["outcome"] == "ok" and call_row["error_message"].startswith("host corrections:")
    # The corrected answer passes the complete checks on its own.
    assert parse_and_validate_role_output(SECTION_READER_ROLE, json.dumps(result.validated_output), SECTION_INPUT)


def test_a_quote_found_in_two_passages_is_not_moved_but_dropped(role_app):
    app, fake_model = role_app
    ambiguous = finding("It was evaluated on 3 benchmarks.", "P1", "evaluated on three benchmarks")  # P3 and P4
    result = run_section_reader(app, fake_model, section_answer(GOOD_FINDING, GOOD_FINDING, ambiguous))
    assert [kept["passage_id"] for kept in result.validated_output["findings"]] == ["P1", "P1"]
    assert "P1 ->" not in result.host_corrections


def test_when_more_than_half_would_be_dropped_the_model_is_asked_to_repair_first(role_app):
    app, fake_model = role_app
    result = run_section_reader(
        app,
        fake_model,
        section_answer(GOOD_FINDING, INVENTED_FINDING, INVENTED_QUOTE),
        section_answer(GOOD_FINDING, MISCITED_FINDING),
    )
    requests = fake_model.requests_for_role("section_reader")
    assert len(requests) == 2 and "REPAIR REQUEST" in requests[1].user_message
    assert "cite THAT passage" in requests[1].user_message
    assert len(result.validated_output["findings"]) == 2  # the repair, with its citation moved to P2


def test_a_failed_repair_falls_back_to_the_corrected_first_answer(role_app):
    app, fake_model = role_app
    result = run_section_reader(
        app, fake_model, section_answer(GOOD_FINDING, INVENTED_FINDING, INVENTED_QUOTE), "not json"
    )
    assert result.validated_output["findings"] == [GOOD_FINDING]
    assert "used the corrected first answer because the repair was rejected" in result.host_corrections


def test_an_invented_web_address_rejects_the_whole_answer(role_app):
    app, fake_model = role_app
    lure = finding("See https://attacker.example/x for the data.", "P1", "reports the rejection rate")
    with pytest.raises(ModelOutputRejectedAfterRepair, match="invented reference"):
        run_section_reader(app, fake_model, section_answer(GOOD_FINDING, GOOD_FINDING, lure))
    assert len(fake_model.requests_for_role("section_reader")) == 2


def test_a_failing_statement_that_cannot_be_dropped_still_rejects_the_review():
    bad_method = {
        **VALID_REVIEWER_ANSWER,
        "method": {"statement": "It runs in 4 ms.", "evidence_status": "supported", "supporting_passage_ids": ["A1"]},
    }
    with pytest.raises(ModelOutputInvalid) as rejection:
        parse_and_validate_role_output(REVIEWER_ROLE, json.dumps(bad_method), REVIEWER_ROLE_INPUT)
    assert rejection.value.salvage is None
    bad_result = {
        **VALID_REVIEWER_ANSWER,
        "results": [
            *VALID_REVIEWER_ANSWER["results"],
            {
                "claim": "It ran 1000 episodes.",
                "conditions": "",
                "passage_id": "A1",
                "excerpt": "the monitor rejected all",
            },
        ],
    }
    with pytest.raises(ModelOutputInvalid) as rejection:
        parse_and_validate_role_output(REVIEWER_ROLE, json.dumps(bad_result), REVIEWER_ROLE_INPUT)
    assert rejection.value.salvage.salvaged_output["results"] == VALID_REVIEWER_ANSWER["results"]


# --------------------------------------------------------------------------------------------------------------------
# Saved stages of an earlier prompt version
# --------------------------------------------------------------------------------------------------------------------


def save_section_stage(app, output: dict, prompt_version: str) -> ReadingStageIdentity:
    paper_id = add_paper(app, "2611.09999", "A monitor", queue_now=False)  # stage rows reference a real paper
    identity = ReadingStageIdentity(paper_id, "input-key", "plan-hash")
    response = StructuredChatResponse(
        final_answer_text="",
        reasoning_text="",
        finish_reason="stop",
        model_identity="m",
        server_build="b",
        prompt_tokens=0,
        completion_tokens=0,
        duration_seconds=0.0,
    )
    save_stage_result(
        app.connection,
        identity,
        "section:001",
        RoleRunResult(output, response, "section_reader", prompt_version),
        job_id=None,
    )
    return identity


@pytest.mark.parametrize("saved_findings, model_is_called", [([GOOD_FINDING], False), ([INVENTED_FINDING], True)])
def test_an_earlier_prompt_versions_stage_is_reused_only_if_it_passes_todays_checks(
    role_app, saved_findings, model_is_called
):
    app, fake_model = role_app
    saved_output = json.loads(section_answer(*saved_findings))
    identity = save_section_stage(app, saved_output, "section-reader-v1")
    fake_model._canned_final_answers_by_role["section_reader"] = section_answer(GOOD_FINDING)
    stage = _run_or_load_stage(app, identity, "section:001", SECTION_READER_ROLE, SECTION_INPUT, None)
    assert bool(fake_model.requests_for_role("section_reader")) == model_is_called
    assert stage.prompt_version == (SECTION_READER_ROLE.prompt_version if model_is_called else "section-reader-v1")


# --------------------------------------------------------------------------------------------------------------------
# Stopping the worker
# --------------------------------------------------------------------------------------------------------------------


def test_a_stop_request_during_a_model_call_puts_the_job_back_without_using_an_attempt(tmp_path):
    def stop_while_the_model_is_thinking(_request):
        request_worker_stop()  # what the SIGTERM handler does; we are inside the interruptible wait
        raise AssertionError("not reached")

    reset_worker_stop_request()
    app = _load_app_with_one_queued_paper(tmp_path, FakeModelClient({"reviewer": stop_while_the_model_is_thinking}))
    try:
        cycle_report = run_single_work_cycle(app, probe_paused_model_now=False)
        assert cycle_report.outcome == WorkCycleOutcome.STOPPED
        [job_row] = app.connection.execute("SELECT status, attempts, lease_token FROM jobs").fetchall()
        assert (job_row["status"], job_row["attempts"], job_row["lease_token"]) == ("queued", 0, None)
        assert app.connection.execute("SELECT status FROM papers").fetchone()[0] == "queued"
        assert app.connection.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] == 0
        # Later calls refuse to start until the worker exits.
        with pytest.raises(WorkerStopRequested), interruptible_model_wait():
            pass
    finally:
        reset_worker_stop_request()
        app.close()


def test_sigterm_stops_a_real_worker_within_seconds_during_a_long_model_call(initialized_home: PaperwatchHome):
    initialized_home.use_fake_model()
    initialized_home.add_example_topic()
    assert initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE)).exit_code == 0
    worker = initialized_home.start_cli("work", environment_overrides={FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE: "120"})
    deadline = time.monotonic() + 30
    while initialized_home.query("SELECT status FROM jobs")[0]["status"] != "running":
        assert time.monotonic() < deadline, "the worker never claimed the job"
        time.sleep(0.1)
    time.sleep(1.0)  # inside the (120 s) model call
    stop_sent_at = time.monotonic()
    worker.send_signal(signal.SIGTERM)
    exit_code = worker.wait(timeout=20)
    assert exit_code == 0 and time.monotonic() - stop_sent_at < 10
    [job_row] = initialized_home.query("SELECT status, attempts, lease_token FROM jobs")
    assert (job_row["status"], job_row["attempts"], job_row["lease_token"]) == ("queued", 0, None)
    assert worker.stdout is not None
    assert "back in the queue without using an attempt" in worker.stdout.read()


# --------------------------------------------------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------------------------------------------------


def test_reports_and_notes_are_readable_by_the_shared_group(tmp_path):
    report_path = tmp_path / "daily" / "2026-11-22.md"
    write_output_file_atomically(report_path, b"# report\n")
    assert stat.S_IMODE(report_path.stat().st_mode) == OUTPUT_FILE_MODE == 0o640


# --------------------------------------------------------------------------------------------------------------------
# Candidate budget
# --------------------------------------------------------------------------------------------------------------------


def test_water_fill_gives_what_a_small_demand_leaves_to_the_others():
    assert water_fill(250, {"cs.AI": 360, "cs.RO": 140, "cs.LO": 21}) == {"cs.AI": 115, "cs.RO": 114, "cs.LO": 21}
    assert water_fill(250, {"cs.AI": None, "cs.RO": 0}) == {"cs.AI": 250, "cs.RO": 0}
    assert water_fill(2, {"a": None, "b": None, "c": None}) == {"a": 1, "b": 1, "c": 0}


def test_feeds_share_their_cap_and_keyword_routes_keep_the_rest():
    budget = CandidateAdmissionBudget(daily_limit=500, reserved_for_current_releases=200, category_feed_limit=250)
    budget.plan_category_feed_shares({"cs.AI": 360, "cs.RO": 140, "cs.LO": 21})
    assert budget.remaining_for_route("arxiv-feed:cs.AI") == 115
    assert budget.remaining_for_route("openalex-recent:q") == 250  # the feeds' unused 250 stay reserved
    budget.record_route_admissions("arxiv-feed:cs.AI", 115)
    assert budget.remaining_for_route("arxiv-feed:cs.AI") == 0
    assert budget.remaining_for_route("arxiv-recovery:cs.LO") == 21
    budget.record_route_admissions("openalex-recent:q", 250)
    assert budget.remaining_for_route("openalex-recent:q") == 0
    assert budget.remaining_for_route("arxiv-feed:cs.RO") == 114  # the feeds' share survives the keyword routes
    assert "arXiv category feeds 115 of 250 (cs.AI 115)" in budget.describe()


def test_the_historical_search_keeps_its_reserve_while_it_has_work():
    settings = CollectionSettings()
    assert (
        settings.daily_new_candidate_budget,
        settings.category_feed_candidate_limit,
        settings.historical_reserved_candidates,
    ) == (1000, 400, 100)
    budget = CandidateAdmissionBudget(
        daily_limit=1000,
        reserved_for_current_releases=200,
        category_feed_limit=400,
        historical_reserve=100,
        historical_reserve_applies=True,
    )
    assert budget.remaining_for_route("openalex-recent:q") == 500  # 1000 - 400 (feeds) - 100 (history)
    budget.record_route_admissions("arxiv-feed:cs.AI", 400)
    budget.record_route_admissions("openalex-recent:q", 500)
    assert budget.remaining_for_route("openalex-recent:q") == 0
    assert budget.remaining_for(RouteKind.HISTORICAL_SEARCH) == 100
    budget.historical_reserve_applies = False  # every plan is complete: the reserve goes back to the others
    assert budget.remaining_for_route("openalex-recent:q") == 100


def test_a_feed_over_its_share_admits_its_best_topic_matches_first():
    def record(title: str) -> CollectedPaperRecord:
        return CollectedPaperRecord(
            provider="arxiv",
            provider_paper_id="2611.00001",
            title=title,
            authors=(),
            abstract=None,
            first_published_date="2026-11-20",
            latest_revision_date=None,
            source_version=None,
            discovery_route="arxiv-feed:cs.AI",
        )

    records = [
        record("Protein folding at scale"),
        record("Runtime monitoring of LLM agents"),
        record("Guardrails for agents"),
        record("Image captioning"),
    ]
    ranked = rank_records_by_topic_words(records, {"runtime", "monitoring", "agent", "guardrail"})
    assert [ranked_record.title for ranked_record in ranked] == [
        "Runtime monitoring of LLM agents",
        "Guardrails for agents",
        "Protein folding at scale",
        "Image captioning",
    ]


# --------------------------------------------------------------------------------------------------------------------
# retry-failed
# --------------------------------------------------------------------------------------------------------------------


def test_retry_failed_requeues_model_output_failures_only(tmp_path, capsys):
    app = _load_app_with_one_queued_paper(tmp_path, FakeModelClient({"reviewer": json.dumps({"wrong": "shape"})}))
    try:
        assert (
            run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_FAILED_PERMANENTLY
        )
        arguments = argparse.Namespace(components=None, job_ids=None, dry_run=True)
        assert run_retry_failed_command(app, arguments) == 0
        assert "would requeue" in capsys.readouterr().out
        assert app.connection.execute("SELECT status FROM jobs").fetchone()[0] == "failed"

        assert (
            run_retry_failed_command(app, argparse.Namespace(components=["database"], job_ids=None, dry_run=False)) == 0
        )
        assert app.connection.execute("SELECT status FROM jobs").fetchone()[0] == "failed"

        assert run_retry_failed_command(app, argparse.Namespace(components=None, job_ids=None, dry_run=False)) == 0
        [job_row] = app.connection.execute("SELECT status, attempts, last_error FROM jobs").fetchall()
        assert (job_row["status"], job_row["attempts"]) == ("queued", 0)
        assert job_row["last_error"].startswith("requeued by the operator after: [model-output]")
        assert app.connection.execute("SELECT status FROM papers").fetchone()[0] == "queued"
        fake_model_client = app.model_client
        assert isinstance(fake_model_client, FakeModelClient)
        fake_model_client._canned_final_answers_by_role.pop("reviewer")
        assert run_single_work_cycle(app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_COMPLETED
    finally:
        app.close()


# --------------------------------------------------------------------------------------------------------------------
# Provider pacing
# --------------------------------------------------------------------------------------------------------------------


def test_openalex_pdf_downloads_are_spaced_more_widely_than_metadata_requests():
    from paperwatch.network.provider_rules import load_provider_access_rules

    rules = load_provider_access_rules()
    assert rules.operation_rule("openalex", "get_text").minimum_seconds_between_requests == 1.5
    assert rules.operation_rule("openalex", "search").minimum_seconds_between_requests is None
    assert rules.provider_rule("openalex").minimum_seconds_between_requests == 0.2
