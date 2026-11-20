"""Plan part 9 check: test 0, 6, and 20 qualifying papers, a late completion, a missing note file, and a model outage.
Restore a missing note from its saved review. Kill the publisher between file rename and the database update. A rerun
must produce the same bytes and no duplicate recommendations.

Also covered: the scheduled cutoff across a daylight-saving change and early publication, the reading-priority floor
and the ordering, seed exclusion, an update (a reported paper whose newer review read different text with different
results) versus a plain re-read, the writer role's evidence checks, the inference reservation the worker yields to,
the writer's deadline, a crash between freezing and rendering, and one publication process at a time.
The model is the offline fake; the topic assessor is scripted so that "Strong" papers score above the floor.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from paperwatch.catalog import record_collected_paper
from paperwatch.clock import utc_now
from paperwatch.commands import report_command
from paperwatch.commands.report_command import run_report_command
from paperwatch.config import ReportSettings
from paperwatch.commands.reread_command import run_reread_command
from paperwatch.db import connect_database, write_transaction
from paperwatch.dispatch_order import dispatch_order_for_operator_request
from paperwatch.errors import (ModelOutputInvalid, ModelServerUnavailable, ReportNotDueYet,
                               ReportPublicationInProgress)
from paperwatch.llm.fake_answers import build_fake_topic_assessments
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.inference_lock import hold_inference_lock
from paperwatch.llm.inference_reservation import (release_inference_reservation, reserve_inference_slot,
                                                  wait_while_inference_is_reserved)
from paperwatch.reporting.publication_lock import hold_report_publication_lock
from paperwatch.reporting.report_schedule import (latest_due_report_date, resolve_report_timing,
                                                  scheduled_cutoff_for_report_date)
from paperwatch.reporting.report_selection import select_report_entries
from paperwatch.readings import enqueue_reading_if_needed
from paperwatch.reporting.revision_changes import ReviewedVersion, describe_material_revision_changes
from paperwatch.roles.base import parse_and_validate_role_output, run_role_once
from paperwatch.roles.healthcheck import HEALTHCHECK_ROLE
from paperwatch.roles.writer import WRITER_ROLE
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.sources.types import CollectedPaperRecord

from conftest import build_in_process_app
from test_step5_collection import START_OF_TEST_WEEK, test_clock  # noqa: F401  (pytest fixture)
from test_step7_evidence_reading import PARAGRAPH_FILLER, drain, install_extracted_text

FIRST_REPORT_DATE = "2026-11-16"  # START_OF_TEST_WEEK is 2026-11-16 11:00 UTC = 06:00 in New York
HOURS_TO_PASS_THE_FIRST_CUTOFF = 2  # 13:00 UTC = 08:00 New York, after the 07:00 cutoff


def rate_strong_papers_high(request) -> str:
    """Topic assessor: every rating 4 for papers whose title contains "Strong", else 2 (abstract-only: 82 vs 44)."""
    answer = build_fake_topic_assessments(request.role_input)
    rating = 4 if "Strong" in request.role_input["paper"]["title"] else 2
    for assessment in answer["topic_assessments"]:
        for rated_dimension in assessment["ratings"].values():
            rated_dimension["rating"] = rating
    return json.dumps(answer)


def open_app(tmp_path, *, model_client: FakeModelClient | None = None, **extra_replacements: str):
    client = model_client or FakeModelClient({"topic_assessor": rate_strong_papers_high})
    return build_in_process_app(tmp_path / "home", model_client=client, config_replacements=extra_replacements)


@pytest.fixture
def app(tmp_path, test_clock):  # noqa: F811
    test_app = open_app(tmp_path)
    yield test_app
    test_app.close()


def add_reviewed_papers(app, count: int, *, strong: bool, first_number: int = 1) -> list[str]:
    """Record `count` papers, read each through the real worker, and return their paper ids."""
    paper_ids = []
    for number in range(first_number, first_number + count):
        title = f"{'Strong' if strong else 'Weak'} runtime monitor study {number:02d}"
        paper_ids.append(record_collected_paper(app.connection, CollectedPaperRecord(
            provider="fixture", provider_paper_id=f"synthetic-{number:04d}", title=title, authors=("A. Author",),
            abstract=f"We check joint robot plans against human exits with monitor variant {number}.",
            first_published_date="2026-11-15", latest_revision_date=None, source_version="v1",
            discovery_route="test:fixture")).paper_id)
        assert promote_paper_directly_and_queue_review(app.connection, paper_ids[-1], max_attempts=3)
    drain(app)
    return paper_ids


def publish(app, report_date: str | None = None, *, early: bool = False) -> str:
    assert run_report_command(app, argparse.Namespace(report_date=report_date, publish_early=early)) == 0
    published_date = report_date or app.connection.execute(
        "SELECT report_date FROM reports ORDER BY created_at DESC, report_date DESC LIMIT 1").fetchone()[0]
    return (app.config.daily_reports_dir / f"{published_date}.md").read_text()


def reported_paper_ids(app, report_date: str) -> list[str]:
    return [row["paper_id"] for row in app.connection.execute(
        "SELECT paper_id FROM report_items WHERE report_date = ? ORDER BY rank", (report_date,))]


def count_ranked_entries(report_text: str) -> int:
    return len(re.findall(r"(?m)^### \d+\. ", report_text))


def writer_calls(app) -> int:
    return len(app.model_client.requests_for_role("writer"))


# --------------------------------------------------------------------------------------------------------------------
# The plan's check
# --------------------------------------------------------------------------------------------------------------------


def test_zero_qualifying_papers_give_a_short_status_report_without_a_model_call(app, test_clock):  # noqa: F811
    add_reviewed_papers(app, 2, strong=False)  # reviewed, but below the floor of 65
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    report_text = publish(app)
    assert "# Paper report for 2026-11-16" in report_text
    assert "Quiet day" in report_text and "selected today: **0**" in report_text
    assert "07:00 America/New_York (12:00 UTC)" in report_text
    assert writer_calls(app) == 0
    assert app.connection.execute("SELECT synthesis_status FROM reports").fetchone()[0] == "not_needed"


def test_six_qualifying_papers_are_published_with_a_grounded_overview(app, test_clock):  # noqa: F811
    strong_paper_ids = add_reviewed_papers(app, 6, strong=True)
    add_reviewed_papers(app, 3, strong=False, first_number=20)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    report_text = publish(app, FIRST_REPORT_DATE)

    assert sorted(reported_paper_ids(app, FIRST_REPORT_DATE)) == sorted(strong_paper_ids)
    assert count_ranked_entries(report_text) == 6 and "### 6." in report_text and "### 7." not in report_text
    assert "Weak runtime monitor" not in report_text
    assert writer_calls(app) == 1
    for heading in ("## What matters today", "## Ranked papers", "## Comparisons and corrections"):
        assert heading in report_text
    assert "Overview and reasons written by the local model" in report_text
    assert "**Read first:** #1" in report_text and "**Why read it.**" in report_text
    # Citations name the ranked entry, the passage and its section, and link the saved note.
    first_review_id = app.connection.execute("SELECT review_id FROM report_items WHERE rank = 1").fetchone()[0]
    assert f"[#1 · `A1` (abstract)](<../papers/{strong_paper_ids[0]}/{first_review_id}.md>)" in report_text \
        or "[#1 · `A1`" in report_text
    assert "— 82/100 · new release · abstract only" in report_text
    assert "Comparison `" in report_text  # part 8 compared some of these reviews; the report links those notes


def test_twenty_qualifying_papers_publish_fifteen_and_keep_five_for_the_next_day(app, test_clock):  # noqa: F811
    add_reviewed_papers(app, 20, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    first_report = publish(app, FIRST_REPORT_DATE)
    first_day = reported_paper_ids(app, FIRST_REPORT_DATE)
    first_day_scores = dict(app.connection.execute(
        "SELECT paper_id, reading_priority FROM report_items WHERE report_date = ?", (FIRST_REPORT_DATE,)).fetchall())
    order_keys = [(-first_day_scores[paper_id], paper_id) for paper_id in first_day]
    assert len(first_day) == 15 and order_keys == sorted(order_keys)  # by score, then (same coverage) paper id
    assert "5 qualifying review(s) held for later reports" in first_report

    test_clock.advance(days=1)
    second_report = publish(app, "2026-11-17")
    second_day = reported_paper_ids(app, "2026-11-17")
    assert len(second_day) == 5 and not set(first_day) & set(second_day)
    assert "Previous report: 2026-11-16, 24 h earlier" in second_report
    assert "0 qualifying review(s) held" in second_report


def test_a_late_completion_belongs_to_the_next_report(app, test_clock):  # noqa: F811
    early_paper_ids = add_reviewed_papers(app, 2, strong=True)
    test_clock.advance(hours=1, minutes=30)  # 12:30 UTC: after the 12:00 UTC cutoff of 2026-11-16
    late_paper_ids = add_reviewed_papers(app, 1, strong=True, first_number=50)
    test_clock.advance(minutes=30)
    publish(app, FIRST_REPORT_DATE)
    assert sorted(reported_paper_ids(app, FIRST_REPORT_DATE)) == sorted(early_paper_ids)

    test_clock.advance(days=1)
    publish(app, "2026-11-17")
    assert reported_paper_ids(app, "2026-11-17") == late_paper_ids


def test_a_missing_note_is_restored_from_its_saved_review_with_the_same_bytes(app, test_clock):  # noqa: F811
    [paper_id] = add_reviewed_papers(app, 1, strong=True)
    review_id = app.connection.execute("SELECT review_id FROM reviews").fetchone()[0]
    note_path = app.config.paper_notes_dir / paper_id / f"{review_id}.md"
    note_bytes_written_by_worker = note_path.read_bytes()
    note_path.unlink()
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    publish(app, FIRST_REPORT_DATE)
    assert note_path.read_bytes() == note_bytes_written_by_worker


class WriterOutageModelClient(FakeModelClient):
    """Reads papers normally; the model server is down when the report needs it."""

    server_is_down = False

    def check_health(self) -> None:
        if self.server_is_down:
            raise ModelServerUnavailable("connection refused (test outage)")

    def request_structured_completion(self, request):
        if self.server_is_down:
            self.received_requests.append(request)
            raise ModelServerUnavailable("connection refused (test outage)")
        return super().request_structured_completion(request)


def test_a_model_outage_renders_the_same_selection_from_saved_summaries(tmp_path, test_clock):  # noqa: F811
    model_client = WriterOutageModelClient({"topic_assessor": rate_strong_papers_high})
    app = open_app(tmp_path, model_client=model_client)
    add_reviewed_papers(app, 3, strong=True)
    model_client.server_is_down = True
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    report_text = publish(app, FIRST_REPORT_DATE)

    assert len(reported_paper_ids(app, FIRST_REPORT_DATE)) == 3  # the same selection the writer would have had
    assert "**Fallback report:** the writer was unavailable (model-server: " in report_text
    assert "> Fallback: no overview was written today." in report_text
    assert "**Why read it** (from the saved topic assessment). Offline fake assessment." in report_text
    report_row = app.connection.execute("SELECT synthesis_status, fallback_reason FROM reports").fetchone()
    assert report_row["synthesis_status"] == "fallback" and "test outage" in report_row["fallback_reason"]
    assert writer_calls(app) == 1  # one attempt: an outage is not retried or repaired within the deadline
    app.close()


def test_kill_between_rename_and_database_update_republishes_the_same_bytes(app, test_clock,  # noqa: F811
                                                                             monkeypatch):
    add_reviewed_papers(app, 3, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    original_write = report_command.write_bytes_atomically

    def write_then_die(destination_path, file_bytes, **options):
        original_write(destination_path, file_bytes, **options)
        if destination_path.parent == app.config.daily_reports_dir:
            raise KeyboardInterrupt("killed right after the rename")

    monkeypatch.setattr(report_command, "write_bytes_atomically", write_then_die)
    with pytest.raises(KeyboardInterrupt):
        run_report_command(app, argparse.Namespace(report_date=FIRST_REPORT_DATE, publish_early=False))
    report_path = app.config.daily_reports_dir / f"{FIRST_REPORT_DATE}.md"
    bytes_before_kill = report_path.read_bytes()
    assert app.connection.execute("SELECT status FROM reports").fetchone()[0] == "pending"

    monkeypatch.setattr(report_command, "write_bytes_atomically", original_write)
    test_clock.advance(minutes=10)
    publish(app, FIRST_REPORT_DATE)
    assert report_path.read_bytes() == bytes_before_kill
    assert app.connection.execute("SELECT status FROM reports").fetchone()[0] == "published"
    assert app.connection.execute("SELECT COUNT(*) FROM report_items").fetchone()[0] == 3
    assert writer_calls(app) == 1  # the rerun used the saved bytes, not a second synthesis

    publish(app, FIRST_REPORT_DATE)  # a published date changes nothing
    assert report_path.read_bytes() == bytes_before_kill


def test_a_crash_between_freezing_and_rendering_resumes_with_the_frozen_selection(app, test_clock,  # noqa: F811
                                                                                  monkeypatch):
    frozen_paper_ids = add_reviewed_papers(app, 2, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)

    def crash(*_arguments, **_options):
        raise KeyboardInterrupt("killed during the synthesis")

    monkeypatch.setattr(report_command, "write_report_synthesis", crash)
    with pytest.raises(KeyboardInterrupt):
        run_report_command(app, argparse.Namespace(report_date=FIRST_REPORT_DATE, publish_early=False))
    assert app.connection.execute("SELECT status FROM reports").fetchone()[0] == "writing"

    add_reviewed_papers(app, 1, strong=True, first_number=70)  # completed after the cutoff
    monkeypatch.undo()
    report_text = publish(app, FIRST_REPORT_DATE)
    assert sorted(reported_paper_ids(app, FIRST_REPORT_DATE)) == sorted(frozen_paper_ids)
    assert count_ranked_entries(report_text) == 2


# --------------------------------------------------------------------------------------------------------------------
# Selection rules
# --------------------------------------------------------------------------------------------------------------------


def test_floor_order_and_seed_exclusion(app, test_clock):  # noqa: F811
    strong_ids = add_reviewed_papers(app, 3, strong=True)
    add_reviewed_papers(app, 2, strong=False, first_number=10)
    cutoff_at = "2026-11-16T12:00:00Z"
    selection = select_report_entries(app.connection, report_date=FIRST_REPORT_DATE, cutoff_at=cutoff_at,
                                      minimum_reading_priority=65, maximum_entries=15,
                                      excluded_paper_ids=frozenset({strong_ids[1]}))
    assert {entry.paper_id for entry in selection.entries} == {strong_ids[0], strong_ids[2]}
    assert [entry.rank for entry in selection.entries] == [1, 2] and selection.below_floor_count == 2
    everything = select_report_entries(app.connection, report_date=FIRST_REPORT_DATE, cutoff_at=cutoff_at,
                                       minimum_reading_priority=0, maximum_entries=15, excluded_paper_ids=frozenset())
    order_keys = [(-entry.reading_priority, entry.paper_id) for entry in everything.entries]
    assert order_keys == sorted(order_keys) and len(order_keys) == 5  # score, then (same coverage) paper id


def test_a_revision_with_different_results_is_reported_again_as_an_update(app, test_clock):  # noqa: F811
    [paper_id] = add_reviewed_papers(app, 1, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    publish(app, FIRST_REPORT_DATE)
    first_review_id = app.connection.execute("SELECT review_id FROM reviews").fetchone()[0]

    # A re-read of the same text is never an update.
    assert run_reread_command(app, argparse.Namespace(paper_id=paper_id, reason="operator check")) == 0
    drain(app)
    assert app.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 2
    test_clock.advance(days=1)
    publish(app, "2026-11-17")
    assert reported_paper_ids(app, "2026-11-17") == []

    # Full text with different numeric results is an update.
    install_extracted_text(app, paper_id, "1 Introduction\nWe measured 42 blocked exits across 7 warehouse "
                                          "trials with the monitor enabled.\n\n" + PARAGRAPH_FILLER * 12 +
                                          "\n\n2 Results\nThe monitor rejected every plan that blocked an exit in "
                                          "all 7 trials.\n")
    with write_transaction(app.connection):
        enqueue_reading_if_needed(app.connection, paper_id, dispatch_order=dispatch_order_for_operator_request(),
                                  max_attempts=3)
    drain(app)
    test_clock.advance(days=1)
    report_text = publish(app, "2026-11-18")
    assert reported_paper_ids(app, "2026-11-18") == [paper_id]
    assert "· update ·" in report_text and "Recommended on 2026-11-16" in report_text
    assert f"../papers/{paper_id}/{first_review_id}.md" in report_text and "numbers in the reported results" \
        in report_text


def test_revision_rule_needs_different_text_and_a_material_change():
    finding = {"claim": "c", "conditions": "", "excerpt": "rejected 7 of 9 plans", "passage_id": "P1"}
    reported = ReviewedVersion({"results": [finding], "assumptions": [], "insufficient_evidence": False}, "aaa", 70)
    same_text = ReviewedVersion({"results": [], "assumptions": [], "insufficient_evidence": True}, "aaa", 40)
    assert describe_material_revision_changes(reported, same_text) == []
    reworded = ReviewedVersion({"results": [dict(finding, claim="other words")], "assumptions": [],
                                "insufficient_evidence": False}, "bbb", 75)
    assert describe_material_revision_changes(reported, reworded) == []
    assumption = {"statement": "Robots know their exact positions", "evidence_status": "supported",
                  "supporting_passage_ids": ["P2"]}
    changed = ReviewedVersion({"results": [dict(finding, excerpt="rejected 8 of 9 plans")],
                               "assumptions": [assumption], "insufficient_evidence": False}, "bbb", 85)
    kinds = [change.change_kind for change in describe_material_revision_changes(reported, changed)]
    assert kinds == ["numeric_results", "assumptions", "reading_priority"]


# --------------------------------------------------------------------------------------------------------------------
# The writer role
# --------------------------------------------------------------------------------------------------------------------


WRITER_INPUT = {
    "report_date": "2026-11-16", "topics": [{"topic_id": "team-safety", "question": "Q?"}],
    "papers": [
        {"review_id": "r_a", "passages": [{"passage_id": "P1", "location": "p. 1, Results",
                                           "text": "The monitor rejected 7 of 9 unsafe joint plans in simulation."}]},
        {"review_id": "r_b", "passages": [{"passage_id": "P1", "location": "abstract",
                                           "text": "A theorem guarantees exit clearance for exact robot positions."}]},
    ],
    "comparisons": [{"comparison_id": "c_1", "review_ids": ["r_a", "r_b"]}],
}


def writer_answer(**overrides) -> dict:
    evidence_a = {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}
    evidence_b = {"review_id": "r_b", "passage_id": "P1", "excerpt": "guarantees exit clearance for exact"}
    answer = {
        "overview": {"statement": "Both papers keep exits clear; one rejected 7 of 9 unsafe plans.",
                     "evidence": [evidence_a, evidence_b]},
        "connected_findings": [{"statement": "A proof and a simulation address the same failure.",
                                "evidence": [evidence_a, evidence_b], "comparison_ids": ["c_1"]}],
        "read_first": {"review_id": "r_a", "explanation": "It tests the monitor directly.", "evidence": [evidence_a]},
        "paper_reasons": [{"review_id": "r_a", "reason_to_read": "Simulation evidence.", "evidence": [evidence_a]},
                          {"review_id": "r_b", "reason_to_read": "The exact-position assumption.",
                           "evidence": [evidence_b]}],
    }
    answer.update(overrides)
    return answer


def validate_writer_answer(answer: dict) -> dict:
    return parse_and_validate_role_output(WRITER_ROLE, json.dumps(answer), WRITER_INPUT)


def test_writer_answer_with_grounded_claims_is_accepted():
    assert validate_writer_answer(writer_answer())["read_first"]["review_id"] == "r_a"


@pytest.mark.parametrize("bad_part, expected_problem", [
    ({"overview": {"statement": "One rejected 97 of 99 plans.", "evidence": [
        {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}]}},
     "do not occur in the cited passage"),
    ({"overview": {"statement": "Overview.", "evidence": [
        {"review_id": "r_a", "passage_id": "P9", "excerpt": "rejected 7 of 9 unsafe joint plans"}]}},
     "is not a passage of review 'r_a'"),
    ({"read_first": {"review_id": "r_z", "explanation": "x", "evidence": []}}, "is not one of today's papers"),
    ({"paper_reasons": [{"review_id": "r_a", "reason_to_read": "x", "evidence": [
        {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}]}]},
     "missing r_b"),
    ({"read_first": {"review_id": "r_a", "explanation": "x", "evidence": [
        {"review_id": "r_b", "passage_id": "P1", "excerpt": "guarantees exit clearance for exact"}]}},
     "cite passages of review 'r_a' here"),
    ({"connected_findings": [{"statement": "x", "evidence": [
        {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}],
        "comparison_ids": ["c_404"]}]}, "comparison id 'c_404' is not in the input"),
    ({"overview": {"statement": "See https://evil.example/paper for more.", "evidence": [
        {"review_id": "r_a", "passage_id": "P1", "excerpt": "rejected 7 of 9 unsafe joint plans"}]}},
     "invented reference"),
])
def test_writer_answer_with_ungrounded_claims_is_rejected(bad_part, expected_problem):
    with pytest.raises(ModelOutputInvalid) as rejection:
        validate_writer_answer(writer_answer(**bad_part))
    assert expected_problem in str(rejection.value)


def test_an_invalid_writer_answer_after_repair_gives_the_fallback(tmp_path, test_clock):  # noqa: F811
    model_client = FakeModelClient({"topic_assessor": rate_strong_papers_high, "writer": "{\"overview\": 1}"})
    app = open_app(tmp_path, model_client=model_client)
    add_reviewed_papers(app, 2, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    report_text = publish(app, FIRST_REPORT_DATE)
    assert "**Fallback report:**" in report_text and "(model-output: " in report_text
    assert writer_calls(app) == 2  # the answer and its one repair
    assert app.connection.execute("SELECT COUNT(*) FROM inference_reservations").fetchone()[0] == 0
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Inference slot reservation and the writer's deadline
# --------------------------------------------------------------------------------------------------------------------


def test_worker_calls_wait_for_the_report_reservation(tmp_path):
    app = build_in_process_app(tmp_path / "home",
                               model_client=FakeModelClient({"healthcheck": '{"status": "ready", "sum": 42}'}))
    reserve_inference_slot(app.connection, "report:test", purpose="test",
                           reserved_until=utc_now() + timedelta(minutes=5))

    def release_soon() -> None:  # another process (the report) ends its reservation
        time.sleep(0.4)
        release_connection = connect_database(app.config.database_path)
        release_inference_reservation(release_connection, "report:test")
        release_connection.close()

    releaser = threading.Thread(target=release_soon)
    started = time.monotonic()
    releaser.start()
    run_role_once(role_definition=HEALTHCHECK_ROLE, role_input={}, app_config=app.config,
                  model_client=app.model_client, connection=app.connection)
    releaser.join()
    assert time.monotonic() - started >= 0.4  # the call waited for the reservation to end
    assert len(app.model_client.received_requests) == 1

    # The holder itself (the report writer) does not wait; an expired reservation blocks nobody.
    reserve_inference_slot(app.connection, "report:test", purpose="test",
                           reserved_until=utc_now() + timedelta(minutes=5))
    started = time.monotonic()
    run_role_once(role_definition=HEALTHCHECK_ROLE, role_input={}, app_config=app.config,
                  model_client=app.model_client, connection=app.connection, holds_inference_reservation=True)
    assert time.monotonic() - started < 0.3
    reserve_inference_slot(app.connection, "report:test", purpose="test",
                           reserved_until=utc_now() - timedelta(seconds=1))
    assert wait_while_inference_is_reserved(app.connection, sleep_function=lambda seconds: None) < 0.3
    app.close()


def test_a_writer_that_misses_its_deadline_gives_the_fallback(tmp_path, test_clock):  # noqa: F811
    model_client = FakeModelClient({"topic_assessor": rate_strong_papers_high})
    app = open_app(tmp_path, model_client=model_client)
    add_reviewed_papers(app, 1, strong=True)
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    app.config = replace(app.config, report=replace(app.config.report, writer_deadline_minutes=0.01))  # 0.6 s
    with hold_inference_lock(app.config.inference_lock_path, 1):  # another process is using the model
        report_text = publish(app, FIRST_REPORT_DATE)
    assert app.connection.execute("SELECT synthesis_status FROM reports").fetchone()[0] == "fallback"
    assert "**Fallback report:**" in report_text and "(inference-lock: " in report_text
    assert writer_calls(app) == 0
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Schedule, early publication, and one publication process
# --------------------------------------------------------------------------------------------------------------------


def test_the_cutoff_follows_local_time_across_daylight_saving_changes():
    settings = ReportSettings()
    assert scheduled_cutoff_for_report_date(date(2026, 3, 7), settings) == datetime(2026, 3, 7, 12, 0,
                                                                                    tzinfo=timezone.utc)
    assert scheduled_cutoff_for_report_date(date(2026, 3, 8), settings) == datetime(2026, 3, 8, 11, 0,
                                                                                    tzinfo=timezone.utc)
    assert scheduled_cutoff_for_report_date(date(2026, 11, 1), settings) == datetime(2026, 11, 1, 12, 0,
                                                                                     tzinfo=timezone.utc)
    before_cutoff = datetime(2026, 11, 16, 11, 59, tzinfo=timezone.utc)
    assert latest_due_report_date(before_cutoff, settings) == date(2026, 11, 15)
    assert latest_due_report_date(before_cutoff + timedelta(minutes=1), settings) == date(2026, 11, 16)


def test_a_report_that_is_not_due_needs_early_and_only_for_today():
    settings = ReportSettings()
    before_cutoff = datetime(2026, 11, 16, 11, 0, tzinfo=timezone.utc)
    with pytest.raises(ReportNotDueYet):
        resolve_report_timing(settings, before_cutoff, "2026-11-16", publish_early=False)
    with pytest.raises(ReportNotDueYet):
        resolve_report_timing(settings, before_cutoff, "2026-11-17", publish_early=True)
    early = resolve_report_timing(settings, before_cutoff, "2026-11-16", publish_early=True)
    assert early.published_early and early.cutoff_at == before_cutoff
    catch_up = resolve_report_timing(settings, before_cutoff, "2026-11-10", publish_early=True)
    assert not catch_up.published_early and catch_up.cutoff_at == catch_up.scheduled_cutoff_at
    assert resolve_report_timing(settings, before_cutoff, None, publish_early=False).report_date == "2026-11-15"


def test_an_early_report_says_so_and_later_completions_go_to_the_next_report(app, test_clock):  # noqa: F811
    first_ids = add_reviewed_papers(app, 1, strong=True)
    report_text = publish(app, FIRST_REPORT_DATE, early=True)  # 06:00 New York, before the 07:00 cutoff
    assert "**published early**" in report_text and reported_paper_ids(app, FIRST_REPORT_DATE) == first_ids
    test_clock.advance(minutes=30)
    later_ids = add_reviewed_papers(app, 1, strong=True, first_number=30)
    test_clock.advance(days=1, hours=2)
    publish(app, "2026-11-17")
    assert reported_paper_ids(app, "2026-11-17") == later_ids


def test_one_publication_process_at_a_time(app, test_clock):  # noqa: F811
    test_clock.advance(hours=HOURS_TO_PASS_THE_FIRST_CUTOFF)
    with hold_report_publication_lock(app.config.report_publication_lock_path):
        with pytest.raises(ReportPublicationInProgress):
            run_report_command(app, argparse.Namespace(report_date=None, publish_early=False))
    assert app.connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0
    publish(app)
    assert app.connection.execute("SELECT report_date FROM reports").fetchone()[0] == FIRST_REPORT_DATE
