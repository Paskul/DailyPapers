"""Plan part 8 check: use a citation graph in which two papers cite each other and share a third reference. Verify
that the search ends, the third paper has one review, and the same comparison is written once. Reject a comparison
that ranks two results obtained under incompatible conditions without explaining the difference.

Also covered: the hop limit (a reference of a reference is read, its own references are not followed), the daily
reference limit with choices waiting for the next day, bibliography identifiers confirmed (or not) by a provider
record, citation contexts, research-role id checks, a gap search over an unused approved branch, and the comparison
note. The providers are the fakes of tests/fake_paper_providers.py; the model is the offline fake.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import pytest

from paperwatch.catalog import record_collected_paper
from paperwatch.clock import utc_now
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.comparisons.comparison_queue import queue_comparison_for_review
from paperwatch.comparisons.evidence_dimensions import shared_evidence_dimensions
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import JobPriorityClass
from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.network.collector_credentials import CollectorCredentials
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.network.provider_rules import load_provider_access_rules, packaged_provider_rules_path
from paperwatch.reading.passages import Passage, SectionKind
from paperwatch.references.citation_contexts import CitedWorkDescription, find_citation_contexts
from paperwatch.references.reference_selection import admit_waiting_reference_selections
from paperwatch.roles.base import parse_and_validate_role_output
from paperwatch.roles.comparison import COMPARISON_ROLE
from paperwatch.roles.research import RESEARCH_ROLE
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.sources.types import CollectedPaperRecord

from conftest import build_in_process_app
from fake_paper_providers import PUBLIC_ADDRESS, FakeOpenAlexWork, FakePaperProviders
from test_step7_evidence_reading import install_extracted_text

TOPICS = """
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
public_queries = ["joint robot plans"]
"""
# Screening promotes nothing here, so every cited work is read only because the research role chose it; the
# historical search creates its plan but fetches no page (its windows stay unused, i.e. offerable gap branches).
NO_SCREENING_PROMOTIONS = {
    "daily_promotion_limit = 30": "daily_promotion_limit = 0",
    "near_cutoff_promotions = 3": "near_cutoff_promotions = 0",
    "historical_pages_per_topic_per_run = 6": "historical_pages_per_topic_per_run = 0",
}
ABSTRACT_A = "We verify joint robot plans with a runtime monitor before dispatch in warehouses."
ABSTRACT_B = "We verify joint robot plans with a theorem prover that assumes exact robot positions."
ABSTRACT_C = "We verify joint robot plans with corridor polygons that model each human exit."
ABSTRACT_D = "We verify joint robot plans with a timing model of bounded travel between waypoints."


def open_app(tmp_path, *, model_client=None, extra_replacements: dict[str, str] | None = None):
    return build_in_process_app(tmp_path / "home", topics_toml=TOPICS,
                                config_replacements={**NO_SCREENING_PROMOTIONS, **(extra_replacements or {})},
                                model_client=model_client)


def make_gateway(app, providers: FakePaperProviders) -> ProviderGateway:
    return ProviderGateway(app.connection, app.config, credentials=CollectorCredentials(),
                           http_transport=providers.transport,
                           provider_rules=load_provider_access_rules(packaged_provider_rules_path().read_text()),
                           address_resolver=lambda host, port: [PUBLIC_ADDRESS], sleep_function=lambda seconds: None)


def collect(app, providers: FakePaperProviders) -> None:
    gateway = make_gateway(app, providers)
    assert run_collect_command(app, argparse.Namespace(fixture_paths=None, skip_searches=False),
                               build_gateway=lambda: gateway) == 0


def drain(app) -> list[WorkCycleOutcome]:
    outcomes = []
    while (outcome := run_single_work_cycle(app, probe_paused_model_now=True).outcome) != WorkCycleOutcome.NO_JOB_DUE:
        assert outcome == WorkCycleOutcome.JOB_COMPLETED, outcome
        outcomes.append(outcome)
        assert len(outcomes) < 50
    return outcomes


def add_openalex_paper(app, work: FakeOpenAlexWork) -> str:
    """A paper found by a public OpenAlex route, read at once (hop 0)."""
    paper_id = record_collected_paper(app.connection, CollectedPaperRecord(
        provider="openalex", provider_paper_id=work.work_id, title=work.title, authors=("C. Synthetic",),
        abstract=work.abstract, first_published_date=work.publication_date.isoformat(), latest_revision_date=None,
        source_version=None, discovery_route="test:openalex")).paper_id
    assert promote_paper_directly_and_queue_review(app.connection, paper_id, max_attempts=3)
    return paper_id


def paper_id_of(app, work_id: str) -> str:
    return app.connection.execute("SELECT paper_id FROM paper_aliases WHERE provider = 'openalex' AND "
                                  "provider_paper_id = ?", (work_id,)).fetchone()["paper_id"]


def count(app, sql: str, parameters: tuple = ()) -> int:
    return app.connection.execute(sql, parameters).fetchone()[0]


# --------------------------------------------------------------------------------------------------------------------
# The plan's check
# --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def citation_graph() -> FakePaperProviders:
    """A and B cite each other and share C; C cites D and A; D cites A (a cycle back to the start)."""
    providers = FakePaperProviders()
    providers.openalex_works = [
        FakeOpenAlexWork("W1", "Runtime monitoring of joint robot plans", ABSTRACT_A, date(2024, 3, 1),
                         referenced_work_ids=("W2", "W3")),
        FakeOpenAlexWork("W2", "Proving joint robot plans exit-safe", ABSTRACT_B, date(2023, 5, 1),
                         referenced_work_ids=("W1", "W3")),
        FakeOpenAlexWork("W3", "Corridor models of human exits", ABSTRACT_C, date(2020, 1, 1),
                         referenced_work_ids=("W4", "W1")),
        FakeOpenAlexWork("W4", "Bounded travel time for robot fleets", ABSTRACT_D, date(2016, 1, 1),
                         referenced_work_ids=("W1",)),
    ]
    return providers


def test_mutual_citations_with_a_shared_reference_end_with_one_review_each_and_one_comparison(
        tmp_path, citation_graph):
    app = open_app(tmp_path)
    paper_a = add_openalex_paper(app, citation_graph.openalex_works[0])
    paper_b = add_openalex_paper(app, citation_graph.openalex_works[1])
    drain(app)  # A and B are read; B's commit queues the A-B comparison (same question, shared guarantees)
    assert count(app, "SELECT COUNT(*) FROM reviews") == 2
    assert count(app, "SELECT COUNT(*) FROM comparisons") == 1

    for _round in range(6):  # collect fetches references and queues choices; the worker reads what was chosen
        collect(app, citation_graph)
        drain(app)
    assert not count(app, "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"), "the search ended"

    paper_c, paper_d = paper_id_of(app, "W3"), paper_id_of(app, "W4")
    reviews_per_paper = {row["paper_id"]: row["review_count"] for row in app.connection.execute(
        "SELECT paper_id, COUNT(*) AS review_count FROM reviews GROUP BY paper_id")}
    assert reviews_per_paper == {paper_a: 1, paper_b: 1, paper_c: 1, paper_d: 1}, "the shared reference read once"

    selections = {row["cited_paper_id"]: row for row in app.connection.execute("SELECT * FROM reference_selections")}
    assert selections[paper_c]["admission_status"] == "admitted" and selections[paper_c]["citation_hop_depth"] == 1
    assert selections[paper_d]["admission_status"] == "admitted" and selections[paper_d]["citation_hop_depth"] == 2
    assert paper_a not in selections and paper_b not in selections, "a cycle back to a reviewed paper is never chosen"
    fetched_papers = {row["citing_paper_id"] for row in app.connection.execute("SELECT * FROM reference_fetches")}
    assert fetched_papers == {paper_a, paper_b, paper_c}, "references of a hop-2 paper are not followed"
    promotion_reasons = {row["paper_id"]: row["promotion_reason"]
                         for row in app.connection.execute("SELECT * FROM paper_promotions")}
    assert promotion_reasons[paper_c] == promotion_reasons[paper_d] == "reference"
    reference_jobs = app.connection.execute(
        "SELECT priority FROM jobs WHERE kind = 'review_paper' AND paper_id IN (?, ?)", (paper_c, paper_d)).fetchall()
    assert {row["priority"] for row in reference_jobs} == {int(JobPriorityClass.IMPORTANT_REFERENCE)}

    # One decision per parent; B's came after C was chosen for A (or the other way round), so it had nothing left.
    runs = app.connection.execute("SELECT * FROM reference_selection_runs").fetchall()
    assert len(runs) == 3 and sum(row["role_name"] is None for row in runs) == 1

    # The A-B comparison exists once, and asking again for either review queues nothing new.
    lower_review, higher_review = sorted(row["review_id"] for row in app.connection.execute(
        "SELECT review_id FROM reviews WHERE paper_id IN (?, ?)", (paper_a, paper_b)))
    assert count(app, "SELECT COUNT(*) FROM comparisons WHERE lower_review_id = ? AND higher_review_id = ?",
                 (lower_review, higher_review)) == 1
    with write_transaction(app.connection):
        assert queue_comparison_for_review(app.connection, app.config, lower_review, utc_now()) is None
        assert queue_comparison_for_review(app.connection, app.config, higher_review, utc_now()) is None
    comparison_jobs = app.connection.execute("SELECT job_key FROM jobs WHERE kind = 'compare_reviews'").fetchall()
    assert len(comparison_jobs) == len({row["job_key"] for row in comparison_jobs}) == \
        count(app, "SELECT COUNT(*) FROM comparisons")
    for comparison_row in app.connection.execute("SELECT comparison_id FROM comparisons"):
        note_text = (app.config.comparison_notes_dir / f"{comparison_row['comparison_id']}.md").read_text()
        assert "../papers/" in note_text and "## What the evidence permits the reader to conclude" in note_text

    model_requests_before = len(app.model_client.received_requests)
    collect(app, citation_graph)
    assert drain(app) == [] and len(app.model_client.received_requests) == model_requests_before


def test_a_choice_over_the_daily_limit_waits_and_is_admitted_the_next_day(tmp_path, citation_graph):
    def choose_every_candidate(request) -> str:
        return json.dumps({"selected_references": [
            {"candidate_id": candidate["candidate_id"], "role_in_parent": "baseline", "importance": 2,
             "explanation": "Compared against it.", "context_passage_ids": []}
            for candidate in request.role_input["candidates"][:3]], "search_gap": []})

    app = open_app(tmp_path, model_client=FakeModelClient({"research": choose_every_candidate}),
                   extra_replacements={"daily_reference_readings = 20": "daily_reference_readings = 1"})
    citation_graph.openalex_works[0].referenced_work_ids = ("W3", "W4")  # A cites C and D, both unread
    add_openalex_paper(app, citation_graph.openalex_works[0])
    drain(app)
    collect(app, citation_graph)
    drain(app)
    statuses = sorted(row["admission_status"] for row in app.connection.execute("SELECT * FROM reference_selections"))
    assert statuses == ["admitted", "waiting_daily_cap"]

    with write_transaction(app.connection):
        assert admit_waiting_reference_selections(app.connection, app.config, active_seed_paper_ids=frozenset(),
                                                  moment=utc_now()) == (0, 1, 0)
        assert admit_waiting_reference_selections(app.connection, app.config, active_seed_paper_ids=frozenset(),
                                                  moment=utc_now() + timedelta(days=1)) == (1, 0, 0)


# --------------------------------------------------------------------------------------------------------------------
# Comparison validation
# --------------------------------------------------------------------------------------------------------------------


def comparison_input() -> dict:
    return {"question": {"topic_id": "team-safety", "question": "Can a monitor keep exits clear?"},
            "shared_dimensions": ["guarantees"],
            "reviews": [
                {"review_id": "r_exact", "passages": [{"passage_id": "P3", "section": "Theorem",
                                                       "text": "The plan is exit-safe for exact robot positions."}]},
                {"review_id": "r_noisy", "passages": [{"passage_id": "P7", "section": "Results",
                                                       "text": "In simulation with noisy positions 37 of 37 "
                                                               "blocking plans were rejected."}]}]}


EXACT = {"review_id": "r_exact", "passage_id": "P3", "excerpt": "exit-safe for exact robot positions"}
NOISY = {"review_id": "r_noisy", "passage_id": "P7", "excerpt": "simulation with noisy positions"}


def comparison_answer(**overrides) -> str:
    answer = {"conditions_comparability": "incompatible", "condition_differences": [],
              "agreements": [{"statement": "Both reject plans that block an exit.", "evidence": [EXACT, NOISY]}],
              "differences": [], "result_ranking": [],
              "reader_conclusion": {"statement": "The guarantee and the simulation answer different conditions.",
                                    "evidence": [EXACT, NOISY]},
              "insufficient_evidence": False}
    return json.dumps({**answer, **overrides})


def test_ranking_results_under_incompatible_conditions_needs_the_difference_explained():
    ranking = {"preferred_review_id": "r_exact", "criterion": "strength of guarantee",
               "justification": "A proof beats a simulation.", "conditions_caveat": "", "evidence": [EXACT]}
    with pytest.raises(ModelOutputInvalid, match="list the condition differences"):
        parse_and_validate_role_output(COMPARISON_ROLE, comparison_answer(result_ranking=[ranking]),
                                       comparison_input())

    explained_ranking = {**ranking, "conditions_caveat": "The proof assumes exact positions, which the noisy "
                                                         "simulation does not, so it says nothing about noise."}
    difference = {"dimension": "assumptions", "statement": "Exact versus noisy robot positions.",
                  "evidence": [EXACT, NOISY]}
    validated = parse_and_validate_role_output(
        COMPARISON_ROLE, comparison_answer(result_ranking=[explained_ranking], condition_differences=[difference]),
        comparison_input())
    assert validated["result_ranking"][0]["preferred_review_id"] == "r_exact"


@pytest.mark.parametrize("bad_answer, expected_problem", [
    (comparison_answer(agreements=[{"statement": "Both agree.", "evidence": [EXACT]}]), "cite evidence from both"),
    (comparison_answer(agreements=[{"statement": "Both agree.", "evidence": [
        EXACT, {**NOISY, "passage_id": "P99"}]}]), "is not a passage of review"),
    (comparison_answer(agreements=[{"statement": "Both agree.", "evidence": [
        EXACT, {**NOISY, "review_id": "r_other"}]}]), "is not one of the two reviews"),
    (comparison_answer(differences=[{"statement": "The simulation rejected 99 plans.", "evidence": [EXACT, NOISY]}]),
     "do not occur in the cited passage"),
])
def test_a_comparison_must_cite_both_reviews_with_real_passages_and_numbers(bad_answer, expected_problem):
    with pytest.raises(ModelOutputInvalid, match=expected_problem):
        parse_and_validate_role_output(COMPARISON_ROLE, bad_answer, comparison_input())


def test_shared_evidence_dimensions_ignore_statements_without_evidence():
    review_a = {"assumptions": [{"statement": "Robot positions are exact.", "evidence_status": "supported"}],
                "evaluation_setting": {"statement": "insufficient evidence", "evidence_status":
                                       "insufficient_evidence"},
                "results": [{"claim": "The monitor rejects blocking joint plans.", "conditions": "warehouse robots"}]}
    review_b = {"assumptions": [{"statement": "Robot positions are noisy.", "evidence_status": "supported"}],
                "evaluation_setting": {"statement": "insufficient evidence", "evidence_status":
                                       "insufficient_evidence"},
                "results": [{"claim": "The prover rejects blocking joint plans.", "conditions": "exact positions"}]}
    assert shared_evidence_dimensions(review_a, review_b) == ["assumptions", "guarantees"]


# --------------------------------------------------------------------------------------------------------------------
# Research role validation and citation contexts
# --------------------------------------------------------------------------------------------------------------------


def research_input() -> dict:
    return {"parent": {"review_id": "r_parent"}, "topics": [], "maximum_selections": 2,
            "candidates": [{"candidate_id": "p_cited", "title": "Corridor models", "authors": [], "year": "2020",
                            "abstract": "", "citation_contexts": [{"passage_id": "P4", "text": "We use [3]."}]}],
            "branches": [{"branch_id": "B1", "topic_id": "team-safety", "provider": "openalex",
                          "search_terms": "joint robot plans", "publication_years": "1940-2008"}]}


@pytest.mark.parametrize("selection, gap, expected_problem", [
    ({"candidate_id": "p_invented"}, [], "is not in CANDIDATES"),
    ({"context_passage_ids": ["P9"]}, [], "is not one of this candidate's citation contexts"),
    ({}, [{"branch_id": "B7", "gap": "Older work is missing.", "explanation": "x"}], "is not in BRANCHES"),
    ({"explanation": "See https://evil.example/upload for details."}, [], "invented reference"),
])
def test_the_research_role_may_only_name_ids_it_was_given(selection, gap, expected_problem):
    chosen = {"candidate_id": "p_cited", "role_in_parent": "method", "importance": 3,
              "explanation": "The corridor model is reused.", "context_passage_ids": ["P4"], **selection}
    with pytest.raises(ModelOutputInvalid, match=expected_problem):
        parse_and_validate_role_output(RESEARCH_ROLE, json.dumps({"selected_references": [chosen],
                                                                  "search_gap": gap}), research_input())


def passage(passage_id: str, text: str, kind: SectionKind = SectionKind.METHOD) -> Passage:
    return Passage(passage_id, 1, "References" if kind == SectionKind.REFERENCES else "2 Method", kind, text)


def test_citation_contexts_follow_numeric_labels_and_author_year_citations():
    bibliography = [passage("P20", "[2] B. Other. Unrelated work. 2018. [3] C. Synthetic. Corridor models of human "
                                   "exits. Robotics Letters, 2020. doi:10.5555/corridor.2020", SectionKind.REFERENCES)]
    body = [passage("P4", "We reuse the corridor polygons of [1, 3]. They are convex."),
            passage("P5", "Other monitors exist [2]."), passage("P6", "Ranges also count [2-4] here.")]
    cited = CitedWorkDescription("Corridor models of human exits", "synthetic", "2020", ("10.5555/corridor.2020",))
    contexts = find_citation_contexts(bibliography, body, cited)
    assert [(context.passage_id, context.text) for context in contexts] == [
        ("P4", "We reuse the corridor polygons of [1, 3]."), ("P6", "Ranges also count [2-4] here.")]

    author_year_bibliography = [passage("P20", "Synthetic, C. (2020). Corridor models of human exits.",
                                        SectionKind.REFERENCES)]
    author_year_body = [passage("P4", "Exits are polygons (Synthetic et al., 2020). Nothing else.")]
    assert [context.passage_id for context in find_citation_contexts(author_year_bibliography, author_year_body,
                                                                     cited)] == ["P4"]


# --------------------------------------------------------------------------------------------------------------------
# Bibliography identifiers, when the graph is incomplete
# --------------------------------------------------------------------------------------------------------------------

FILLER = ("The monitor is described in enough detail for a reader to reimplement it, and each design choice is "
          "explained with reference to the warehouse setting. ")


def parent_full_text() -> str:
    return ("Runtime Monitoring of Joint Robot Plans\n\nAbstract\n\nWe check joint plans before dispatch.\n\n"
            "1 Introduction\n\n" + FILLER * 4 + "\n\n2 Method\n\nWe build on the corridor model of [1] to check "
            "every plan before dispatch. " + FILLER * 4 + "\n\n3 Conclusion\n\nExits stay clear. " + FILLER
            + "\n\nReferences\n\n[1] C. Synthetic. Exit clearance monitors for robot fleets. Journal of Synthetic "
            "Results, 2019. doi:10.5555/exit.2019\n\n[2] D. Other. Planning under uncertainty for fleets. 2018. "
            "https://doi.org/10.5555/mismatch.2018\n")


def test_bibliography_identifiers_become_references_only_when_a_provider_record_matches(tmp_path):
    providers = FakePaperProviders()
    providers.openalex_works = [
        FakeOpenAlexWork("W10", "Runtime monitoring of joint robot plans", ABSTRACT_A, date(2024, 3, 1)),  # no refs
        FakeOpenAlexWork("W11", "Exit clearance monitors for robot fleets", ABSTRACT_C, date(2019, 1, 1),
                         doi="10.5555/exit.2019"),
        FakeOpenAlexWork("W12", "A title that matches nothing cited", ABSTRACT_D, date(2018, 1, 1),
                         doi="10.5555/mismatch.2018"),
    ]
    app = open_app(tmp_path)
    parent_id = record_collected_paper(app.connection, CollectedPaperRecord(
        provider="openalex", provider_paper_id="W10", title="Runtime monitoring of joint robot plans",
        authors=("C. Synthetic",), abstract=ABSTRACT_A, first_published_date="2024-03-01", latest_revision_date=None,
        source_version=None, discovery_route="test:openalex")).paper_id
    install_extracted_text(app, parent_id, parent_full_text())
    assert promote_paper_directly_and_queue_review(app.connection, parent_id, max_attempts=3)
    drain(app)
    collect(app, providers)

    statuses = {row["identifier_value"]: row["confirmation_status"]
                for row in app.connection.execute("SELECT * FROM bibliography_identifiers")}
    assert statuses == {"10.5555/exit.2019": "confirmed", "10.5555/mismatch.2018": "mismatch"}
    edges = app.connection.execute("SELECT * FROM citation_edges WHERE citing_paper_id = ?", (parent_id,)).fetchall()
    assert [(edge["cited_value"], edge["provenance"]) for edge in edges] == [
        ("10.5555/exit.2019", "bibliography:confirmed_by_provider")]
    assert edges[0]["cited_paper_id"] == paper_id_of(app, "W11")
    assert count(app, "SELECT COUNT(*) FROM paper_aliases WHERE provider_paper_id = '10.5555/mismatch.2018'") == 0

    drain(app)  # the research role sees the confirmed work with the sentence that cites it
    research_request = app.model_client.requests_for_role("research")[0]
    offered = research_request.role_input["candidates"]
    assert [candidate["candidate_id"] for candidate in offered] == [paper_id_of(app, "W11")]
    assert "corridor model of [1]" in offered[0]["citation_contexts"][0]["text"]
    selection = app.connection.execute("SELECT * FROM reference_selections").fetchone()
    assert selection["role_in_parent"] == "method" and selection["admission_status"] == "admitted"


# --------------------------------------------------------------------------------------------------------------------
# A gap search over an unused approved branch
# --------------------------------------------------------------------------------------------------------------------


def test_the_research_role_can_request_one_unused_approved_search_branch(tmp_path, citation_graph):
    def choose_an_openalex_branch(request) -> str:
        branch = next(branch for branch in request.role_input["branches"] if branch["provider"] == "openalex")
        return json.dumps({"selected_references": [], "search_gap": [
            {"branch_id": branch["branch_id"], "gap": "The parent builds on decades-old exit planning work.",
             "explanation": "The oldest window of the approved phrase has not been searched."}]})

    app = open_app(tmp_path, model_client=FakeModelClient({"research": choose_an_openalex_branch}))
    citation_graph.openalex_works.append(
        FakeOpenAlexWork("W9", "Joint robot plans in early warehouse automation", "An early study of joint robot "
                         "plans for warehouse exits.", date(1995, 6, 1), cited_by_count=40))
    add_openalex_paper(app, citation_graph.openalex_works[0])
    drain(app)
    collect(app, citation_graph)  # fetches A's references and creates the historical plan (no pages fetched)
    drain(app)
    branch = app.connection.execute("SELECT * FROM research_branches").fetchone()
    assert branch["branch_status"] == "requested" and branch["query_id"].startswith("team-safety:openalex:")
    assert branch["gap_statement"].startswith("The parent builds on decades-old exit planning work.")

    collect(app, citation_graph)  # runs the branch: one page of that approved query in that window
    branch = app.connection.execute("SELECT * FROM research_branches").fetchone()
    assert branch["branch_status"] == "done" and branch["new_paper_count"] == 1
    assert json.loads(branch["returned_paper_ids_json"]) == [paper_id_of(app, "W9")]
    branch_requests = [request for request in citation_graph.requests_to("api.openalex.org", "/works")
                       if "title_and_abstract.search" in request.url.params.get("filter", "")]
    assert len(branch_requests) == 1, "the branch is searched once, never repeated"
