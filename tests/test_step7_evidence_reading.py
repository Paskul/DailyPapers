"""Plan part 7 check: run a full paper, a truncated extraction, and an abstract-only paper through the same schema.
Reject an invented metric and an unknown passage id. Confirm that an unchanged second run reads none of them again.

Also covered: stable passages with pages and sections, bounded reading sections with saved results (a timeout
resumes instead of restarting), one bounded repair, the rubric (4, 3, 2, 3 -> 80/100; evidence capped for abstract
only; "comparison unavailable"), comparison anchors, the note, holding readings for their source text, switching a
queued reading to newly extracted full text, and falling back to the abstract when the cached text is gone.
The fake model (paperwatch.llm.fake_answers) quotes the first words of what it is given, so every answer passes the
real evidence checks; extracted text is installed into the cache directly (no Podman needed).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from conftest import build_in_process_app
from jsonschema import Draft202012Validator

from paperwatch.catalog import record_collected_paper
from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import dispatch_order_for_older_paper
from paperwatch.errors import ConfigurationError, ModelCallTimedOut, ModelOutputInvalid
from paperwatch.jobs import claim_next_due_job
from paperwatch.llm.fake_answers import build_fake_section_notes
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.reading.evidence_checks import find_numbers_needing_evidence
from paperwatch.reading.passages import SectionKind, build_abstract_passages, segment_extracted_text
from paperwatch.reading.reading_plan import plan_full_text_reading
from paperwatch.reading.reading_priority import score_topic_assessment
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed, release_readings_held_for_source_text
from paperwatch.review_inputs import fingerprint_extracted_reading_text
from paperwatch.roles.base import parse_and_validate_role_output
from paperwatch.roles.reviewer import REVIEWER_OUTPUT_SCHEMA
from paperwatch.roles.section_reader import SECTION_READER_ROLE
from paperwatch.screening.review_promotion import (
    promote_paper_directly_and_queue_review,
    queue_new_reading_for_promoted_paper_in_transaction,
)
from paperwatch.source_cache import store_extracted_text
from paperwatch.sources.types import CollectedPaperRecord

SMALL_READING_LIMITS = {  # several reading sections from a short synthetic paper
    "section_max_characters = 24000": "section_max_characters = 1500",
    "passage_max_characters = 1500": "passage_max_characters = 700",
    "retry_delay_seconds = 600": "retry_delay_seconds = 0",
}
PARAGRAPH_FILLER = (
    "The monitor is described in enough detail for a reader to reimplement it, and the authors "
    "explain each design choice with reference to the warehouse setting they study. "
)


def synthetic_full_text(*, truncated: bool = False) -> str:
    """A paper as pdftotext prints it: form feeds between pages, numbered headings, a hyphenated line break."""
    pages = [
        "Runtime Monitoring of Joint Robot Plans\nA. Example\n\nAbstract\n\nWe check joint plans before dispatch.\n\n"
        "1 Introduction\n\nHuman exits must stay clear while robot teams move goods. " + PARAGRAPH_FILLER * 3,
        "2 Method\n\nThe monitor predicts robot positions along each joint plan and rejects a plan whose predicted "
        "footprint overlaps a marked exit corridor. The moni-\ntor assumes bounded travel time between waypoints. "
        + PARAGRAPH_FILLER * 3
        + "\n\n2.1 Corridor model\n\nEach exit corridor is a polygon. "
        + PARAGRAPH_FILLER * 3,
        "3 Results\n\nIn 200 simulated warehouse episodes the monitor rejected all 37 plans that blocked an exit, "
        "with a median check time of 12 ms. " + PARAGRAPH_FILLER * 2 + "\n\nTable 1: Check times\nPlans 37 12 9 4\n"
        "Robots 5 8 11 14\n\n4 Limitations\n\nThe guarantee does not hold under noisy localization. "
        + PARAGRAPH_FILLER
        * 2,
        "5 Conclusion\n\nChecking joint plans before dispatch keeps exits clear in simulation. "
        + PARAGRAPH_FILLER
        + "\n\nReferences\n\n[1] A. Author. A cited paper. 2020.\n\n[2] B. Author. Another paper. 2021.",
    ]
    return "\f".join(pages[:2] if truncated else pages)


def paper_record(identifier: str, title: str, abstract: str | None = None) -> CollectedPaperRecord:
    return CollectedPaperRecord(
        provider="arxiv",
        provider_paper_id=identifier,
        title=title,
        authors=("A. Example",),
        abstract=abstract
        if abstract is not None
        else (
            "We present a runtime monitor for joint robot plans. In 200 simulated episodes it rejected all 37 "
            "plans that blocked an exit."
        ),
        first_published_date="2026-11-17",
        latest_revision_date=None,
        source_version=None,
        discovery_route="test:arxiv-feed",
    )


def add_paper(app, identifier: str, title: str, *, extracted_text: str | None = None, queue_now: bool = True) -> str:
    paper_id = record_collected_paper(app.connection, paper_record(identifier, title)).paper_id
    if extracted_text is not None:
        install_extracted_text(app, paper_id, extracted_text)
    if queue_now:
        assert promote_paper_directly_and_queue_review(app.connection, paper_id, max_attempts=3)
    return paper_id


def install_extracted_text(app, paper_id: str, extracted_text: str) -> str:
    """What the collector's source-text stage records after a successful sandboxed extraction."""
    edition_id = app.connection.execute(
        "SELECT edition_id FROM paper_editions WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    source_sha256 = "ab" * 32
    text_path = store_extracted_text(app.config.source_cache_dir, edition_id, source_sha256, extracted_text)
    with write_transaction(app.connection):
        app.connection.execute(
            """INSERT INTO edition_source_texts (edition_id, paper_id, acquisition_status, text_route,
                   source_file_name, source_sha256, content_type, source_bytes, extraction_status,
                   extracted_text_file_name, extracted_text_sha256, normalized_text_sha256,
                   extracted_text_characters, attempts, acquired_at, updated_at)
               VALUES (?, ?, 'acquired', 'local_file', ?, ?, 'application/pdf', 1, 'extracted', ?, ?, ?, ?, 1, ?, ?)""",
            (
                edition_id,
                paper_id,
                f"{source_sha256[:32]}.pdf",
                source_sha256,
                text_path.name,
                "cd" * 32,
                fingerprint_extracted_reading_text(extracted_text),
                len(extracted_text.strip()),
                utc_now_timestamp(),
                utc_now_timestamp(),
            ),
        )
    return edition_id


def drain(app) -> list[WorkCycleOutcome]:
    outcomes: list[WorkCycleOutcome] = []
    while (outcome := run_single_work_cycle(app, probe_paused_model_now=True).outcome) != WorkCycleOutcome.NO_JOB_DUE:
        outcomes.append(outcome)
        assert len(outcomes) < 50
    return outcomes


@pytest.fixture
def reading_app(tmp_path):
    app = build_in_process_app(tmp_path / "home", config_replacements=SMALL_READING_LIMITS)
    yield app
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# The plan's check
# --------------------------------------------------------------------------------------------------------------------


def test_full_truncated_and_abstract_only_papers_share_one_schema_and_are_not_read_again(reading_app):
    app = reading_app
    full_paper_id = add_paper(app, "2611.00001v1", "Full paper (synthetic)", extracted_text=synthetic_full_text())
    truncated_paper_id = add_paper(
        app, "2611.00002v1", "Truncated extraction (synthetic)", extracted_text=synthetic_full_text(truncated=True)
    )
    abstract_paper_id = add_paper(app, "2611.00003v1", "Abstract only (synthetic)")

    first_run_outcomes = drain(app)  # three readings, plus any part-8 comparison of them
    assert set(first_run_outcomes) == {WorkCycleOutcome.JOB_COMPLETED} and len(first_run_outcomes) >= 3
    reviews = {row["paper_id"]: row for row in app.connection.execute("SELECT * FROM reviews")}
    assert {paper_id: reviews[paper_id]["coverage"] for paper_id in reviews} == {
        full_paper_id: "full_text",
        truncated_paper_id: "partial_text",
        abstract_paper_id: "abstract_only",
    }

    schema_validator = Draft202012Validator(REVIEWER_OUTPUT_SCHEMA)
    for review_row in reviews.values():  # one schema for all three
        assert not list(schema_validator.iter_errors(json.loads(review_row["review_json"])))
        cited_rows = app.connection.execute(
            "SELECT * FROM review_evidence_passages WHERE review_id = ?", (review_row["review_id"],)
        ).fetchall()
        assert cited_rows, "every review keeps the passages it cites"

    full_record = json.loads(reviews[full_paper_id]["reading_record_json"])
    assert full_record["text_kind"] == "extracted_text" and full_record["reading_mode"] == "section_notes"
    assert full_record["partial_reasons"] == [] and full_record["skipped_not_substantive"] == ["References"]
    assert "rows recovered" in full_record["parser_recovery"]["tables"]
    truncated_record = json.loads(reviews[truncated_paper_id]["reading_record_json"])
    assert any("may be truncated" in reason for reason in truncated_record["partial_reasons"])
    full_cited = app.connection.execute(
        "SELECT page_number, section_title FROM review_evidence_passages WHERE review_id = ?",
        (reviews[full_paper_id]["review_id"],),
    ).fetchone()
    assert full_cited["page_number"] == 1  # the fake quotes the first passage of the first section

    section_calls = app.model_client.requests_for_role("section_reader")
    assert len(section_calls) == len(full_record["sections_read"]) + len(truncated_record["sections_read"])
    calls_after_first_run = len(app.model_client.received_requests)

    # An unchanged second run reads none of them again.
    with write_transaction(app.connection):
        for paper_id in (full_paper_id, truncated_paper_id, abstract_paper_id):
            outcome, _input = enqueue_reading_if_needed(
                app.connection, paper_id, dispatch_order=dispatch_order_for_older_paper(0.5), max_attempts=3
            )
            assert outcome == ReadingEnqueueOutcome.ALREADY_REVIEWED
            assert not queue_new_reading_for_promoted_paper_in_transaction(
                app.connection, paper_id, active_seed_paper_ids=frozenset(), max_attempts=3
            )
    assert drain(app) == []
    assert len(app.model_client.received_requests) == calls_after_first_run

    for paper_id, review_row in reviews.items():  # the worker wrote each permanent note after its commit
        note_text = (app.config.paper_notes_dir / paper_id / f"{review_row['review_id']}.md").read_text()
        assert "## Scores by topic" in note_text and "## Evidence passages" in note_text
    full_note = (app.config.paper_notes_dir / full_paper_id / f"{reviews[full_paper_id]['review_id']}.md").read_text()
    assert "`P1` (p. 1, Front matter)" in full_note and "Skipped as not substantive: References" in full_note


VALID_SECTION_PASSAGES = [
    {
        "passage_id": "P7",
        "section": "3 Results",
        "page": 3,
        "text": "In 200 simulated warehouse episodes the monitor rejected all 37 plans that blocked an exit.",
    }
]
SECTION_INPUT = {
    "paper": {"title": "t"},
    "reading_section": {"section_number": 1, "section_count": 1, "section_titles": ["3 Results"]},
    "passages": VALID_SECTION_PASSAGES,
}


def section_answer(**finding_overrides) -> str:
    finding = {
        "claim": "The monitor rejected all 37 blocking plans.",
        "conditions": "200 simulated episodes",
        "passage_id": "P7",
        "excerpt": "the monitor rejected all 37 plans",
        "finding_kind": "result",
        **finding_overrides,
    }
    return json.dumps(
        {
            "section_summary": "Results.",
            "findings": [finding],
            "reviewer_inferences": [],
            "ambiguous_passages": [],
            "parser_problems": [],
            "insufficient_evidence": False,
        }
    )


def test_a_valid_finding_passes_and_an_invented_metric_or_unknown_passage_is_rejected():
    assert parse_and_validate_role_output(SECTION_READER_ROLE, section_answer(), SECTION_INPUT)["findings"]
    with pytest.raises(ModelOutputInvalid, match=r"the number\(s\) 97.3 do not occur in the cited passage"):
        parse_and_validate_role_output(
            SECTION_READER_ROLE, section_answer(claim="The monitor reached 97.3% accuracy."), SECTION_INPUT
        )
    with pytest.raises(ModelOutputInvalid, match=r"the number\(s\) 500 do not occur"):
        parse_and_validate_role_output(SECTION_READER_ROLE, section_answer(conditions="500 episodes"), SECTION_INPUT)
    with pytest.raises(ModelOutputInvalid, match="passage id 'P99' is not in the input packet"):
        parse_and_validate_role_output(SECTION_READER_ROLE, section_answer(passage_id="P99"), SECTION_INPUT)
    with pytest.raises(ModelOutputInvalid, match="excerpt does not occur in passage P7"):
        parse_and_validate_role_output(
            SECTION_READER_ROLE, section_answer(excerpt="the monitor rejected all 38 plans"), SECTION_INPUT
        )
    with pytest.raises(ModelOutputInvalid, match="contiguous quotation"):
        parse_and_validate_role_output(
            SECTION_READER_ROLE, section_answer(excerpt="the monitor ... blocked an exit"), SECTION_INPUT
        )


def test_number_labels_are_exempt_and_number_forms_are_normalized():
    assert find_numbers_needing_evidence("As Table 2 and Section 3.1 show, P12 reports 1,000 runs at 0.50 s") == [
        "1000",
        "0.5",
    ]


# --------------------------------------------------------------------------------------------------------------------
# Long papers: bounded sections that are saved one by one
# --------------------------------------------------------------------------------------------------------------------


def test_a_timeout_resumes_from_the_saved_sections_instead_of_restarting(reading_app):
    app = reading_app
    section_call_count = 0

    def time_out_on_the_third_section_once(request):
        nonlocal section_call_count
        section_call_count += 1
        if section_call_count == 3:
            raise ModelCallTimedOut("role 'section_reader' exceeded its time limit (simulated)")
        return json.dumps(build_fake_section_notes(request.role_input))

    app.model_client = FakeModelClient({"section_reader": time_out_on_the_third_section_once})
    paper_id = add_paper(app, "2611.00004v1", "Long paper (synthetic)", extracted_text=synthetic_full_text())

    assert drain(app) == [WorkCycleOutcome.JOB_RETRY_SCHEDULED, WorkCycleOutcome.JOB_COMPLETED]
    sections_read = json.loads(app.connection.execute("SELECT reading_record_json FROM reviews").fetchone()[0])[
        "sections_read"
    ]
    # Sections 1 and 2 were saved before the timeout; the retry read only section 3 onward.
    assert section_call_count == len(sections_read) + 1
    sections_requested = [
        request.role_input["reading_section"]["section_number"]
        for request in app.model_client.requests_for_role("section_reader")
    ]
    assert sections_requested[:3] == [1, 2, 3] and sections_requested[3] == 3
    assert len(set(sections_requested)) == len(sections_read)
    assert (
        app.connection.execute("SELECT COUNT(*) FROM review_stage_results WHERE paper_id = ?", (paper_id,)).fetchone()[
            0
        ]
        == len(sections_read) + 1
    )  # + synthesis


def test_reading_plan_reads_method_and_results_first_and_a_capped_paper_is_partial():
    passages = segment_extracted_text(synthetic_full_text(), passage_max_characters=700, passage_min_characters=200)
    full_plan = plan_full_text_reading(passages, section_max_characters=1500, max_sections_per_paper=10)
    assert full_plan.coverage == "full_text" and len(full_plan.all_sections) >= 3
    capped_plan = plan_full_text_reading(passages, section_max_characters=1500, max_sections_per_paper=2)
    assert capped_plan.coverage == "partial_text" and "not read (limit: 2 per paper)" in capped_plan.partial_reasons[0]
    read_kinds = {passage.section_kind for section in capped_plan.sections_to_read for passage in section.passages}
    assert SectionKind.METHOD in read_kinds or SectionKind.RESULTS in read_kinds
    assert all(
        section.section_number < later.section_number
        for section, later in zip(capped_plan.sections_to_read, capped_plan.sections_to_read[1:])
    )  # document order


def test_passages_have_stable_ids_pages_sections_and_dehyphenated_text():
    passages = segment_extracted_text(synthetic_full_text(), passage_max_characters=700, passage_min_characters=200)
    assert passages == segment_extracted_text(
        synthetic_full_text(), passage_max_characters=700, passage_min_characters=200
    )
    assert [passage.passage_id for passage in passages] == [f"P{index}" for index in range(1, len(passages) + 1)]
    method_passage = next(passage for passage in passages if "monitor assumes bounded" in passage.text)
    assert (method_passage.page_number, method_passage.section_title, method_passage.section_kind) == (
        2,
        "2 Method",
        SectionKind.METHOD,
    )
    corridor_passage = next(passage for passage in passages if "polygon" in passage.text)
    assert corridor_passage.section_title == "2.1 Corridor model"
    assert corridor_passage.section_kind == SectionKind.METHOD  # inherited from its parent section
    assert {passage.section_kind for passage in passages if "cited paper" in passage.text} == {SectionKind.REFERENCES}
    assert [passage.passage_id for passage in build_abstract_passages("T", "First. Second one.")] == ["A1", "A2"]
    assert [(passage.passage_id, passage.text) for passage in build_abstract_passages("Only a title", None)] == [
        ("T1", "Only a title")
    ]


# --------------------------------------------------------------------------------------------------------------------
# The rubric, comparison anchors, and the context budget
# --------------------------------------------------------------------------------------------------------------------


def test_rubric_example_gives_80_and_applies_the_caps():
    ratings = {"question_fit": 4, "method_fit": 3, "evidence_strength": 2, "relative_value": 3}
    assert (
        score_topic_assessment("t", ratings, comparison_available=True, is_abstract_only=False).reading_priority_score
        == 80
    )
    without_comparison = score_topic_assessment("t", ratings, comparison_available=False, is_abstract_only=False)
    assert "relative_value" not in without_comparison.applied_ratings
    assert without_comparison.reading_priority_score == 81  # 100 * (40 + 18.75 + 10) / 85 = 80.88
    abstract_only = score_topic_assessment("t", ratings, comparison_available=True, is_abstract_only=True)
    assert abstract_only.applied_ratings["evidence_strength"] == 1 and abstract_only.evidence_strength_capped
    assert abstract_only.reading_priority_score == 75


def test_an_earlier_review_on_the_same_question_becomes_a_comparison_anchor(reading_app):
    app = reading_app
    first_paper_id = add_paper(app, "2611.00005v1", "First paper (synthetic)")
    drain(app)
    first_assessor_input = app.model_client.requests_for_role("topic_assessor")[0].role_input
    assert first_assessor_input["topics"][0]["comparison_available"] is False

    add_paper(app, "2611.00006v1", "Second paper (synthetic)")
    drain(app)
    second_topic_input = app.model_client.requests_for_role("topic_assessor")[1].role_input["topics"][0]
    first_review_id = app.connection.execute(
        "SELECT review_id FROM reviews WHERE paper_id = ?", (first_paper_id,)
    ).fetchone()[0]
    assert [anchor["review_id"] for anchor in second_topic_input["review_anchors"]] == [first_review_id]
    assert second_topic_input["comparison_available"] is True
    assert "P1" not in json.dumps(second_topic_input["review_anchors"])  # anchor ids are review ids, not passages
    scorings = [
        json.loads(row[0])["scoring"]
        for row in app.connection.execute(
            "SELECT assessment_json FROM review_topic_assessments ORDER BY assessed_at, rowid"
        )
    ]
    assert [scoring["comparison_available"] for scoring in scorings] == [False, True]
    second_note = next(
        path for path in app.config.paper_notes_dir.rglob("*.md") if first_paper_id not in str(path)
    ).read_text()
    assert f"Compared reviews: `{first_review_id}`" in second_note  # an id in a code span is never escaped


def test_a_reading_section_that_cannot_fit_the_context_budget_is_refused_at_config_load(tmp_path):
    with pytest.raises(ConfigurationError, match="exceeds model.required_context_tokens"):
        build_in_process_app(
            tmp_path / "home", config_replacements={"section_max_characters = 24000": "section_max_characters = 90000"}
        )
    assert "section_max_characters = 90000" in (tmp_path / "home" / "config.toml").read_text()


# --------------------------------------------------------------------------------------------------------------------
# Which text is read: holding for source text, switching to full text, falling back
# --------------------------------------------------------------------------------------------------------------------


def test_held_readings_wait_for_the_source_text_stage(reading_app):
    app = reading_app
    paper_id = record_collected_paper(app.connection, paper_record("2611.00007v1", "Held (synthetic)")).paper_id
    with write_transaction(app.connection):
        enqueue_reading_if_needed(
            app.connection,
            paper_id,
            dispatch_order=dispatch_order_for_older_paper(0.5),
            max_attempts=3,
            hold_until=utc_now() + timedelta(minutes=60),
        )
    assert claim_next_due_job(app.connection, lease_duration_seconds=60) is None
    with write_transaction(app.connection):
        assert release_readings_held_for_source_text(app.connection) == 1
    assert claim_next_due_job(app.connection, lease_duration_seconds=60) is not None


def test_full_text_extracted_after_promotion_replaces_the_queued_abstract_reading(reading_app):
    app = reading_app
    paper_id = add_paper(app, "2611.00008v1", "Promoted before its PDF (synthetic)")
    install_extracted_text(app, paper_id, synthetic_full_text())
    with write_transaction(app.connection):  # what the source-text stage does after a successful extraction
        assert queue_new_reading_for_promoted_paper_in_transaction(
            app.connection, paper_id, active_seed_paper_ids=frozenset(), max_attempts=3
        )
    job_rows = app.connection.execute(
        "SELECT status, input_version, last_error FROM jobs ORDER BY created_at, rowid"
    ).fetchall()
    assert [(row["status"], row["input_version"].split(":")[0]) for row in job_rows] == [
        ("done", "abstract_only"),
        ("queued", "extracted_text"),
    ]
    assert drain(app) == [WorkCycleOutcome.JOB_COMPLETED]
    assert app.connection.execute("SELECT coverage FROM reviews").fetchone()[0] == "full_text"


def test_missing_cached_text_falls_back_to_the_abstract_without_a_model_call(reading_app):
    app = reading_app
    paper_id = add_paper(app, "2611.00009v1", "Lost cache (synthetic)", extracted_text=synthetic_full_text())
    for cached_text in Path(app.config.source_cache_dir).rglob("*.txt"):
        cached_text.unlink()
    outcomes = drain(app)
    assert outcomes == [WorkCycleOutcome.JOB_COMPLETED, WorkCycleOutcome.JOB_COMPLETED]
    assert (
        app.connection.execute("SELECT coverage FROM reviews WHERE paper_id = ?", (paper_id,)).fetchone()[0]
        == "abstract_only"
    )
    assert app.connection.execute("SELECT extraction_status FROM edition_source_texts").fetchone()[0] == "failed"
    assert app.model_client.requests_for_role("section_reader") == []
