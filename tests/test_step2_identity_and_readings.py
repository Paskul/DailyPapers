"""Plan part 2: paper identity, editions, and the reading key.

One canonical paper per work (merged only on identifier links), similar titles flagged but never merged, editions
kept apart, and a reading repeated only for changed text or an explicit operator reread.
"""

from __future__ import annotations

import functools
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import EXAMPLE_TOPIC_TOML

from paperwatch.app_context import AppContext
from paperwatch.catalog import record_collected_paper
from paperwatch.commands.init_command import run_init_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.db import apply_pending_migrations, connect_database
from paperwatch.db_migrations import MIGRATION_0001_INITIAL_SCHEMA
from paperwatch.errors import SourceInputError
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.research_graph import (
    build_comparison_key,
    record_citation_edge,
    record_comparison,
    resolve_citation_edges,
)
from paperwatch.scholarly_identifiers import NormalizedIdentifier, normalize_identifier
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.source_checkpoints import (
    SearchCoverageStatus,
    SourceCheckpoint,
    load_source_checkpoint,
    save_source_checkpoint,
)
from paperwatch.sources.types import CollectedPaperRecord, RawIdentifier

BASE_RECORD = CollectedPaperRecord(
    provider="arxiv",
    provider_paper_id="2411.05555v1",
    title="Checking Joint Robot Plans Against Human Exits (synthetic)",
    authors=("C. Synthetic", "D. Fixture"),
    abstract="SYNTHETIC. A monitor rejects joint robot plans that would block a marked human exit.",
    first_published_date="2026-11-01",
    latest_revision_date=None,
    source_version=None,
    discovery_route="test:arxiv-feed",
)


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def load_test_app(home_dir: Path, model_client: FakeModelClient | None = None) -> AppContext:
    run_init_command(home_dir, None)
    (home_dir / "topics.toml").write_text(EXAMPLE_TOPIC_TOML)
    return AppContext.load(home_dir, model_client=model_client or FakeModelClient())


def collect(app: AppContext, collected_record: CollectedPaperRecord) -> tuple[str, bool]:
    """Catalog the record and queue a reading if needed, exactly like `paperwatch collect`."""
    paper_id = record_collected_paper(app.connection, collected_record).paper_id
    return paper_id, promote_paper_directly_and_queue_review(app.connection, paper_id, max_attempts=3)


def drain_queue(app: AppContext) -> list[WorkCycleOutcome]:
    outcomes: list[WorkCycleOutcome] = []
    while (outcome := run_single_work_cycle(app, probe_paused_model_now=False).outcome) != WorkCycleOutcome.NO_JOB_DUE:
        outcomes.append(outcome)
    return outcomes


def count_rows(connection: sqlite3.Connection, table_name: str) -> int:
    return connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # noqa: S608  # table names come from COUNTED_TABLES


# --------------------------------------------------------------------------------------------------------------------
# Identifier normalization
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scheme, raw_value, expected",
    [
        ("doi", "https://doi.org/10.1109/LRA.2026.0001", NormalizedIdentifier("doi", "10.1109/lra.2026.0001")),
        ("DOI", "doi:10.1109/LRA.2026.0001", NormalizedIdentifier("doi", "10.1109/lra.2026.0001")),
        ("doi", "http://dx.doi.org/10.1000%2Fabc", NormalizedIdentifier("doi", "10.1000/abc")),
        ("arxiv", "arXiv:2411.01234v3", NormalizedIdentifier("arxiv", "2411.01234", 3)),
        ("arxiv", "https://arxiv.org/pdf/2411.01234v2.pdf", NormalizedIdentifier("arxiv", "2411.01234", 2)),
        ("arxiv", "2411.01234", NormalizedIdentifier("arxiv", "2411.01234", None)),
        ("arxiv", "math.gt/0309136v1", NormalizedIdentifier("arxiv", "math.GT/0309136", 1)),
        ("arxiv", "hep-th/9901001", NormalizedIdentifier("arxiv", "hep-th/9901001", None)),
        ("openalex", "https://openalex.org/w4400000001", NormalizedIdentifier("openalex", "W4400000001")),
        ("fixture", "  demo-0001 ", NormalizedIdentifier("fixture", "demo-0001")),
    ],
)
def test_identifiers_are_normalized(scheme, raw_value, expected):
    assert normalize_identifier(scheme, raw_value) == expected


@pytest.mark.parametrize(
    "scheme, raw_value",
    [
        ("doi", "not-a-doi"),
        ("arxiv", "2411.1"),
        ("openalex", "A123"),
        ("bad scheme", "x"),
        ("fixture", "  "),
    ],
)
def test_malformed_identifiers_are_rejected(scheme, raw_value):
    with pytest.raises(SourceInputError):
        normalize_identifier(scheme, raw_value)


# --------------------------------------------------------------------------------------------------------------------
# Identity: two routes, linked identifiers, possible matches
# --------------------------------------------------------------------------------------------------------------------


def test_same_work_by_two_routes_is_one_paper_with_two_editions(tmp_path):
    app = load_test_app(tmp_path)
    arxiv_paper_id, _ = collect(app, BASE_RECORD)
    openalex_sighting = replace(
        BASE_RECORD,
        provider="openalex",
        provider_paper_id="W4400000009",
        discovery_route="test:openalex-search",
        title=BASE_RECORD.title.upper(),  # a different title must not matter
        linked_identifiers=(
            RawIdentifier("doi", "10.48550/arXiv.2411.05555"),
            RawIdentifier("doi", "10.1109/LRA.2026.5555"),
        ),
    )
    openalex_outcome = record_collected_paper(app.connection, openalex_sighting)

    assert openalex_outcome.paper_id == arxiv_paper_id and not openalex_outcome.was_newly_created
    assert count_rows(app.connection, "papers") == 1
    assert sorted(
        tuple(row) for row in app.connection.execute("SELECT provider, provider_paper_id FROM paper_aliases")
    ) == [
        ("arxiv", "2411.05555"),
        ("doi", "10.1109/lra.2026.5555"),
        ("doi", "10.48550/arxiv.2411.05555"),
        ("openalex", "W4400000009"),
    ]
    assert sorted(row[0] for row in app.connection.execute("SELECT edition_key FROM paper_editions")) == [
        "arxiv:2411.05555@v1",
        "openalex:W4400000009",
    ]
    assert sorted(row[0] for row in app.connection.execute("SELECT route FROM paper_discoveries")) == [
        "test:arxiv-feed",
        "test:openalex-search",
    ]
    paper_row = app.connection.execute("SELECT title, origin_route FROM papers").fetchone()
    assert (paper_row["title"], paper_row["origin_route"]) == (BASE_RECORD.title, "test:arxiv-feed")

    # A later sighting of the publisher DOI alone (another route) still finds the same paper.
    journal_sighting = replace(
        BASE_RECORD,
        provider="doi",
        provider_paper_id="https://doi.org/10.1109/LRA.2026.5555",
        discovery_route="test:journal-toc",
    )
    assert record_collected_paper(app.connection, journal_sighting).paper_id == arxiv_paper_id


def test_similar_title_without_identifier_link_is_flagged_not_merged(tmp_path):
    app = load_test_app(tmp_path)
    first_paper_id, _ = collect(app, BASE_RECORD)
    look_alike = replace(
        BASE_RECORD,
        provider="openalex",
        provider_paper_id="W1",
        title=BASE_RECORD.title + "  ",
        authors=("D. Fixture",),
        discovery_route="test:openalex-search",
    )
    look_alike_outcome = record_collected_paper(app.connection, look_alike)

    assert look_alike_outcome.was_newly_created and look_alike_outcome.paper_id != first_paper_id
    assert look_alike_outcome.possible_match_paper_ids == (first_paper_id,)
    [match_row] = app.connection.execute("SELECT * FROM paper_possible_matches").fetchall()
    assert match_row["match_reason"] == "same_title_shared_author" and match_row["resolution"] == "unresolved"
    assert json.loads(match_row["evidence_json"])["shared_family_names"] == ["fixture"]


def test_record_linking_two_existing_papers_is_flagged_not_merged(tmp_path):
    app = load_test_app(tmp_path)
    arxiv_paper_id, _ = collect(app, BASE_RECORD)
    journal_paper_id, _ = collect(
        app,
        replace(
            BASE_RECORD,
            provider="doi",
            provider_paper_id="10.1109/x.1",
            title="A Journal Paper That Looks Different",
            abstract="Other text.",
        ),
    )
    bridging_record = replace(
        BASE_RECORD,
        provider="openalex",
        provider_paper_id="W77",
        linked_identifiers=(RawIdentifier("arxiv", "2411.05555"), RawIdentifier("doi", "10.1109/X.1")),
    )
    bridging_outcome = record_collected_paper(app.connection, bridging_record)

    assert bridging_outcome.paper_id == arxiv_paper_id  # the oldest paper; nothing is merged
    assert bridging_outcome.possible_match_paper_ids == (journal_paper_id,)
    assert count_rows(app.connection, "papers") == 2
    [match_row] = app.connection.execute("SELECT match_reason FROM paper_possible_matches").fetchall()
    assert match_row["match_reason"] == "conflicting_identifier_link"
    doi_owner = app.connection.execute(
        "SELECT paper_id FROM paper_aliases WHERE provider = 'doi' AND provider_paper_id = '10.1109/x.1'"
    ).fetchone()
    assert doi_owner["paper_id"] == journal_paper_id  # the alias was not stolen


# --------------------------------------------------------------------------------------------------------------------
# Reading identity: what does and does not cause another reading
# --------------------------------------------------------------------------------------------------------------------


def test_title_route_topic_and_model_changes_never_trigger_another_reading(tmp_path):
    fake_model = FakeModelClient()
    app = load_test_app(tmp_path, fake_model)
    paper_id, was_queued = collect(app, BASE_RECORD)
    assert was_queued and drain_queue(app) == [WorkCycleOutcome.JOB_COMPLETED]

    # A new revision whose only change is the title, the same version seen by another route, and a restart.
    assert not collect(app, replace(BASE_RECORD, provider_paper_id="2411.05555v2", title="Renamed (synthetic)"))[1]
    assert not collect(app, replace(BASE_RECORD, discovery_route="test:openalex-search"))[1]
    app.close()

    # A topic change and a model upgrade (new identity) on restart.
    (tmp_path / "topics.toml").write_text(EXAMPLE_TOPIC_TOML.replace("Can a runtime monitor", "Could a monitor"))
    upgraded_model = FakeModelClient()
    app = AppContext.load(tmp_path, model_client=upgraded_model)
    assert not collect(app, BASE_RECORD)[1]
    assert drain_queue(app) == []  # empty queue: no inference call
    # One reading = one reviewer call + one topic-assessment call (plan part 7); nothing after the model upgrade.
    assert [request.role_name for request in fake_model.received_requests] == ["reviewer", "topic_assessor"]
    assert upgraded_model.received_requests == []
    assert count_rows(app.connection, "reviews") == 1 and count_rows(app.connection, "jobs") == 1
    assert (
        app.connection.execute("SELECT status FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()[0] == "complete"
    )


def test_changed_text_gets_a_new_reading_and_identical_text_does_not(tmp_path):
    app = load_test_app(tmp_path)
    collect(app, BASE_RECORD)
    drain_queue(app)
    whitespace_only_change = replace(
        BASE_RECORD, provider_paper_id="2411.05555v2", abstract=str(BASE_RECORD.abstract).replace(" ", "   ")
    )
    assert not collect(app, whitespace_only_change)[1]

    revised_text = replace(
        BASE_RECORD,
        provider_paper_id="2411.05555v3",
        latest_revision_date="2026-11-15",
        abstract=f"{BASE_RECORD.abstract} Version 3 adds a hardware experiment.",
    )
    assert collect(app, revised_text)[1]
    assert drain_queue(app) == [WorkCycleOutcome.JOB_COMPLETED]
    review_rows = app.connection.execute(
        """SELECT edition.edition_key FROM reviews JOIN paper_editions AS edition USING (edition_id)
           ORDER BY reviews.completed_at, edition.edition_key"""
    ).fetchall()
    assert [row[0] for row in review_rows] == ["arxiv:2411.05555@v1", "arxiv:2411.05555@v3"]
    paper_row = app.connection.execute("SELECT source_version, latest_revision_date FROM papers").fetchone()
    assert (paper_row["source_version"], paper_row["latest_revision_date"]) == ("v3", "2026-11-15")


def test_abstract_arriving_for_title_only_record_supersedes_the_queued_title_reading(tmp_path):
    fake_model = FakeModelClient()
    app = load_test_app(tmp_path, fake_model)
    _paper_id, was_queued = collect(app, replace(BASE_RECORD, abstract=None))
    assert was_queued
    assert collect(app, BASE_RECORD)[1]  # same edition, now with its abstract: newly available text

    outcomes = drain_queue(app)
    assert outcomes == [WorkCycleOutcome.JOB_COMPLETED]  # the title-only job was retired when the abstract arrived
    assert len(fake_model.requests_for_role("reviewer")) == 1
    [retired_job] = app.connection.execute(
        "SELECT status, last_error FROM jobs WHERE input_version != (SELECT review_input_key FROM reviews)"
    ).fetchall()
    assert retired_job["status"] == "done" and retired_job["last_error"].startswith("superseded by a newer input")
    [review_row] = app.connection.execute("SELECT source_text_sha256, review_input_key FROM reviews").fetchall()
    assert review_row["review_input_key"].startswith("abstract_only:sha256:")
    assert app.connection.execute("SELECT status FROM papers").fetchone()[0] == "complete"
    assert app.connection.execute("SELECT abstract FROM papers").fetchone()[0] == BASE_RECORD.abstract


def test_operator_reread_gets_its_own_generation_and_review(initialized_home):
    initialized_home.use_fake_model()
    initialized_home.add_example_topic()
    from conftest import RECORDED_PAPER_FIXTURE

    initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert initialized_home.run_cli("work", "--once").exit_code == 0
    [paper_row] = initialized_home.query("SELECT paper_id FROM papers")

    reread_result = initialized_home.run_cli("reread", paper_row["paper_id"], "--reason", "check new prompt")
    assert reread_result.exit_code == 0, reread_result.stderr
    assert "generation g_" in reread_result.stdout
    assert initialized_home.run_cli("work", "--once").exit_code == 0

    review_rows = initialized_home.query("SELECT reading_generation, review_input_key FROM reviews ORDER BY rowid")
    assert review_rows[0]["reading_generation"] == "initial"
    assert review_rows[1]["reading_generation"].startswith("g_")
    assert review_rows[1]["review_input_key"].endswith(f":generation:{review_rows[1]['reading_generation']}")
    [reread_job] = initialized_home.query("SELECT priority, input_json FROM jobs WHERE priority = 0")
    assert json.loads(reread_job["input_json"])["operator_reason"] == "check new prompt"

    # The ordinary collector still sees the original reading and queues nothing.
    recollect_result = initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert "0 new reading(s) of promoted papers queued" in recollect_result.stdout
    assert "Promotion: 0 paper(s) queued for review now" in recollect_result.stdout
    assert initialized_home.run_cli("reread", "p_0000000000000000").exit_code == 2


def test_topic_assessments_are_stored_apart_from_the_reading(tmp_path):
    """Part 7: the topic assessor proposes ratings; the host caps evidence for an abstract-only reading and
    computes the 0-100 reading priority. No anchors exist, so relative_value is omitted."""
    assessor_answer = {
        "topic_assessments": [
            {
                "topic_id": "team-safety",
                "rationale": "fits",
                "ratings": {
                    "question_fit": {"rating": 4, "explanation": "direct"},
                    "method_fit": {"rating": 3, "explanation": "close"},
                    "evidence_strength": {"rating": 2, "explanation": "partial"},
                },
                "matched_seed_ids": [],
                "compared_review_ids": [],
            }
        ]
    }
    app = load_test_app(tmp_path, FakeModelClient({"topic_assessor": json.dumps(assessor_answer)}))
    collect(app, BASE_RECORD)
    drain_queue(app)
    [assessment_row] = app.connection.execute("SELECT * FROM review_topic_assessments").fetchall()
    # round(100 * (40*4 + 25*3 + 20*1) / (4 * 85)) = 75: evidence capped from 2 to 1 (abstract only).
    assert (assessment_row["topic_id"], assessment_row["topic_score"], assessment_row["score_scale"]) == (
        "team-safety",
        75,
        "priority_0_to_100",
    )
    scoring = json.loads(assessment_row["assessment_json"])["scoring"]
    assert scoring["proposed_ratings"]["evidence_strength"] == 2 and scoring["applied_ratings"] == {
        "question_fit": 4,
        "method_fit": 3,
        "evidence_strength": 1,
    }
    assert scoring["comparison_available"] is False
    assert assessment_row["topic_profile_sha256"] == app.config.topics[0].compute_profile_sha256()


# --------------------------------------------------------------------------------------------------------------------
# Records used by later parts: citation edges, comparisons, checkpoints
# --------------------------------------------------------------------------------------------------------------------


def test_citation_edges_and_comparisons_are_unique(tmp_path):
    app = load_test_app(tmp_path)
    citing_paper_id, _ = collect(app, BASE_RECORD)
    cited_identifier = normalize_identifier("doi", "10.1000/CITED.1")
    with app.connection:
        assert record_citation_edge(
            app.connection,
            citing_paper_id=citing_paper_id,
            cited_identifier=cited_identifier,
            provenance="openalex:referenced_works",
        )
        assert not record_citation_edge(
            app.connection,
            citing_paper_id=citing_paper_id,
            cited_identifier=cited_identifier,
            provenance="openalex:referenced_works",
        )
    cited_paper_id, _ = collect(
        app,
        replace(
            BASE_RECORD, provider="doi", provider_paper_id="10.1000/cited.1", title="The cited work", abstract="Cited."
        ),
    )
    assert resolve_citation_edges(app.connection) == 1
    assert app.connection.execute("SELECT cited_paper_id FROM citation_edges").fetchone()[0] == cited_paper_id

    drain_queue(app)
    review_id_a, review_id_b = [row[0] for row in app.connection.execute("SELECT review_id FROM reviews")]
    assert build_comparison_key("team-safety", "ab" * 32, review_id_a, review_id_b) == build_comparison_key(
        "team-safety", "ab" * 32, review_id_b, review_id_a
    )
    record_this_comparison = functools.partial(
        record_comparison,
        app.connection,
        topic_id="team-safety",
        question_sha256="ab" * 32,
        comparison_output={"x": 1},
        role_name="comparison",
        prompt_version="v0",
        model_identity="m",
        server_build="b",
    )
    first_id, first_is_new = record_this_comparison(review_id_a=review_id_a, review_id_b=review_id_b)
    second_id, second_is_new = record_this_comparison(review_id_a=review_id_b, review_id_b=review_id_a)
    assert first_is_new and not second_is_new and first_id == second_id
    assert count_rows(app.connection, "comparisons") == 1


def test_source_checkpoint_round_trip_and_coverage(tmp_path):
    app = load_test_app(tmp_path)
    checkpoint = SourceCheckpoint(
        search_key="openalex-history:team-safety:q0:2015-2019",
        source_name="openalex",
        topic_id="team-safety",
        query_parameters={"search": "runtime assurance"},
        candidate_budget=300,
    )
    after_first_page = checkpoint.after_page(next_cursor="cursor-2", newly_admitted_candidates=200)
    assert after_first_page.coverage_status == SearchCoverageStatus.ACTIVE
    after_second_page = after_first_page.after_page(next_cursor="cursor-3", newly_admitted_candidates=120)
    assert after_second_page.coverage_status == SearchCoverageStatus.CAPPED
    assert after_second_page.remaining_candidate_budget == 0
    save_source_checkpoint(app.connection, after_second_page)
    assert load_source_checkpoint(app.connection, checkpoint.search_key) == after_second_page
    assert (
        checkpoint.after_page(next_cursor=None, newly_admitted_candidates=3).coverage_status
        == SearchCoverageStatus.EXHAUSTED
    )


# --------------------------------------------------------------------------------------------------------------------
# Migration from a part-1 database
# --------------------------------------------------------------------------------------------------------------------


def test_schema_v1_database_migrates_without_queueing_a_second_reading(tmp_path):
    run_init_command(tmp_path, None)  # creates config files; then replace the DB with a v1 one holding data
    database_path = tmp_path / "state" / "paperwatch.sqlite3"
    database_path.unlink()
    legacy_connection = connect_database(database_path)
    legacy_connection.executescript(f"BEGIN;\n{MIGRATION_0001_INITIAL_SCHEMA}\nPRAGMA user_version = 1;\nCOMMIT;")
    legacy_connection.executescript("""
        INSERT INTO papers VALUES ('p_00000000000000aa', 'Legacy title', '["A. Old"]', 'Legacy abstract.',
            '2026-11-01', NULL, 'v1', 'fixture:old.json', '2026-11-18T10:00:00Z', 'complete', '2026-11-18T10:00:00Z');
        INSERT INTO paper_aliases VALUES ('fixture', 'legacy-1', 'p_00000000000000aa');
        INSERT INTO paper_discoveries VALUES ('p_00000000000000aa', 'fixture:old.json', '2026-11-18T10:00:00Z');
        INSERT INTO jobs VALUES ('j_00000000000000aa', 'review_paper:p_00000000000000aa:abstract_only:sha256:old',
            'review_paper', 'p_00000000000000aa', 'abstract_only:sha256:old', '{"coverage": "abstract_only"}', 1,
            'done', 1, 3, '2026-11-18T10:00:00Z', NULL, NULL, NULL, NULL, '2026-11-18T10:00:00Z',
            '2026-11-18T10:01:00Z');
        INSERT INTO reviews VALUES ('r_00000000000000aa', 'p_00000000000000aa', 'j_00000000000000aa',
            'abstract_only:sha256:old', 'abstract_only',
            '{"topic_relevance": [{"topic_id": "team-safety", "relevance_rating": 2, "rationale": "r"}]}',
            'reviewer', 'reviewer-abstract-v0', 'fake', 'fake', '2026-11-18T10:01:00Z');
    """)
    assert apply_pending_migrations(legacy_connection)[0] == 1
    legacy_connection.close()

    app = load_test_app(tmp_path)  # init again is a no-op for the migrated database
    [edition_row] = app.connection.execute("SELECT * FROM paper_editions").fetchall()
    assert edition_row["edition_key"] == "fixture:legacy-1@v1" and edition_row["version_number"] == 1
    [review_row] = app.connection.execute("SELECT edition_id, source_text_sha256 FROM reviews").fetchall()
    assert review_row["edition_id"] == edition_row["edition_id"] and review_row["source_text_sha256"]
    assert count_rows(app.connection, "review_topic_assessments") == 1

    legacy_record = CollectedPaperRecord(
        provider="fixture",
        provider_paper_id="legacy-1",
        title="Legacy title",
        authors=("A. Old",),
        abstract="Legacy abstract.",
        first_published_date="2026-11-01",
        latest_revision_date=None,
        source_version="v1",
        discovery_route="fixture:old.json",
    )
    assert collect(app, legacy_record) == ("p_00000000000000aa", False)
    assert count_rows(app.connection, "paper_editions") == 1 and count_rows(app.connection, "jobs") == 1
