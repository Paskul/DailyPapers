"""`paperwatch collect`: sync seeds, collect current and older papers, acquire source text, and queue review jobs.

The collector is the only component allowed to contact paper providers (its process arms the collector egress
policy), and it does so only through paperwatch.network.provider_gateway.ProviderGateway's fixed operations.

Stages of one run:
  1. Seeds (plan part 4): sync every configured Zotero collection and CSL JSON file (paperwatch.seeds.seed_sync).
     A source that cannot be read keeps its last successful seed set; collection continues either way.
  2. Papers (plan part 5, paperwatch.collection): current releases first (arXiv category feeds, recovery of releases
     missed during downtime, recent OpenAlex windows), then each topic's finite historical search, all within the
     daily budget of new metadata candidates. `--fixture FILE` replaces these live searches with recorded records
     (a replay); `--skip-searches` skips them.
  2b. References and gap searches (plan part 8, live runs only): fetch the outgoing references of reviewed papers
     that address a topic's question (paperwatch.references.reference_fetch), and run the gap searches the research
     role requested (paperwatch.references.research_branches). Cited works and branch results join the catalog and
     are screened in step 3 like every other candidate.
  3. Screening (plan part 6, paperwatch.screening): score every candidate for every topic on the CPU with the pinned
     local embedding model plus the phrase signal, then promote up to the daily limit (slots divided across topics
     first) and queue each reading in its priority class. Nothing leaves the machine and no model call is made.
  3b. Follow-up work (plan part 8, no network): queue the research role's choice of references for each fetched
     parent, admit choices that waited for the daily reference limit, and queue comparisons the daily comparison
     limit held back for recent reviews.
  4. Source text: for papers queued for review (reference readings included), a local file, else (live runs only) a
     capped download; then text extraction in the sandbox when the extractor image is configured. A paper whose full
     text was extracted gets its reading switched to the full text (plan part 7). The readings promoted in step 3,
     and the reference readings chosen since the last run, are held until this stage ends, so the worker does not
     read an abstract seconds before the PDF arrives.
  5. Research profile: print each topic's input mode (keywords, seeds, or both) and anything missing.

Re-collecting is always safe: the catalog resolves each record to its canonical paper, every page commits with its
checkpoint, and a reading is queued only for a promoted paper whose preferred edition has text that was never read.
If the embedding model is unavailable, every other stage still runs, candidates wait unscored, and the command ends
with the screening failure (exit code 12) so the problem is visible.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from paperwatch.activity_log import COMPONENT_COLLECT, prune_old_activity, recorded_activity
from paperwatch.app_context import AppContext
from paperwatch.clock import utc_now
from paperwatch.db import write_transaction
from paperwatch.readings import compute_source_text_hold_until, release_readings_held_for_source_text
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.current_releases import collect_current_releases
from paperwatch.collection.historical_search import historical_search_has_work, run_historical_searches
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.collection.source_text import acquire_source_texts
from paperwatch.comparisons.comparison_queue import queue_comparisons_for_recent_reviews
from paperwatch.references.reference_fetch import fetch_references_of_reviewed_papers
from paperwatch.references.reference_selection import (admit_waiting_reference_selections,
                                                       queue_reference_selection_jobs)
from paperwatch.references.research_branches import run_requested_research_branches
from paperwatch.errors import describe_failure_for_operator
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.research_profile import (build_research_profiles, explain_empty_research_profile,
                                         load_active_seed_library_paper_ids)
from paperwatch.screening.screening_stage import run_screening_and_promotion
from paperwatch.seeds.seed_store import seed_sources_of_all_topics
from paperwatch.seeds.seed_sync import synchronize_all_seed_sources
from paperwatch.sources.fixture_source import RecordedFixtureSource


def run_collect_command(app: AppContext, arguments: argparse.Namespace, *,
                        build_gateway: Callable[[], ProviderGateway] | None = None) -> int:
    """`build_gateway` lets tests inject a gateway with a recorded transport; production uses the collector's.
    One gateway serves the whole run; it (and the credentials file) is created only if a stage needs a provider."""
    shared_gateway: list[ProviderGateway] = []

    def build_gateway_once() -> ProviderGateway:
        if not shared_gateway:
            shared_gateway.append(build_gateway() if build_gateway else
                                  ProviderGateway.for_collector(app.connection, app.config))
        return shared_gateway[0]

    fixture_paths = getattr(arguments, "fixture_paths", None) or []
    run_live_searches = not fixture_paths and not getattr(arguments, "skip_searches", False)
    run_mode = "fixture" if fixture_paths else ("live" if run_live_searches else "skip_searches")

    prune_old_activity(app.connection)
    with recorded_activity(app.connection, "run", {"mode": run_mode}, component=COMPONENT_COLLECT) as run_activity:
        exit_code = _run_collect_stages(app, build_gateway_once, fixture_paths, run_live_searches)
        if exit_code != 0:
            run_activity.end_as("failed", note=f"exit code {exit_code} (see the service log)")
    return exit_code


def _run_collect_stages(app: AppContext, build_gateway_once: Callable[[], ProviderGateway], fixture_paths: list,
                        run_live_searches: bool) -> int:
    """The stages of one run, each recorded in the activity log (the monitor shows which one is running)."""
    def stage(step_name: str):
        return recorded_activity(app.connection, step_name)

    with stage("seeds"):
        _synchronize_seed_sources(app, build_gateway_once)
    collection_context = CollectionContext.start(
        app.connection, app.config, build_gateway_once,
        active_seed_paper_ids=load_active_seed_library_paper_ids(app.connection, app.config))
    if fixture_paths:
        with stage("fixtures"):
            _import_recorded_fixtures(collection_context, fixture_paths)
    elif run_live_searches:
        # Keep the day's historical reserve away from the current releases while some plan still has work.
        collection_context.candidate_budget.historical_reserve_applies = \
            historical_search_has_work(collection_context)
        with stage("current_releases"):
            _print_lines(collect_current_releases(collection_context))
        with stage("historical_search"):
            _print_lines(run_historical_searches(collection_context))
        with stage("gap_searches"):
            _print_lines(run_requested_research_branches(collection_context))
        print(f"Candidates: {collection_context.candidate_budget.describe()}")
        with stage("references"):
            _print_lines(fetch_references_of_reviewed_papers(collection_context))
    else:
        print("Papers: live arXiv / OpenAlex searches skipped (--skip-searches)")
    with stage("screening") as screening_activity:
        screening_result = run_screening_and_promotion(
            app.connection, app.config, active_seed_paper_ids=collection_context.active_seed_paper_ids,
            moment=collection_context.started_at,
            hold_new_readings_until=compute_source_text_hold_until(utc_now(),
                                                                   app.config.collection.source_text_hold_minutes))
        if screening_result.embedder_failure is not None:
            screening_activity.end_as("failed", note=str(screening_result.embedder_failure))
    _print_lines(screening_result.report_lines)
    with stage("follow_up"):
        _print_lines(_queue_follow_up_work(app, collection_context))
    try:
        with stage("source_text"):
            _print_lines(acquire_source_texts(collection_context, allow_network=run_live_searches))
    finally:  # the held readings become due now, whatever the source-text stage achieved
        with write_transaction(app.connection):
            released_count = release_readings_held_for_source_text(app.connection)
        if released_count:
            print(f"Readings: {released_count} held reading(s) released to the worker")
    _print_research_profile_summary(app)
    if screening_result.embedder_failure is not None:
        print(describe_failure_for_operator(screening_result.embedder_failure), file=sys.stderr)
        return screening_result.embedder_failure.process_exit_code
    return 0


def _queue_follow_up_work(app: AppContext, collection_context: CollectionContext) -> list[str]:
    """Plan part 8 queueing that needs no network: reference choices, waiting admissions, comparisons."""
    with write_transaction(app.connection):
        selection_jobs = queue_reference_selection_jobs(
            app.connection, app.config, active_seed_paper_ids=collection_context.active_seed_paper_ids)
        admitted, waiting, _skipped = admit_waiting_reference_selections(
            app.connection, app.config, active_seed_paper_ids=collection_context.active_seed_paper_ids,
            moment=utc_now())
        comparison_jobs = queue_comparisons_for_recent_reviews(app.connection, app.config, utc_now())
    return [f"Follow-up: {selection_jobs} reference choice(s) queued for the research role, {admitted} waiting "
            f"reference reading(s) admitted ({waiting} still waiting for the daily limit), {comparison_jobs} "
            "comparison(s) queued"]


def _synchronize_seed_sources(app: AppContext, build_gateway: Callable[[], ProviderGateway]) -> None:
    if not seed_sources_of_all_topics(app.config.topics):
        print("Seeds: no zotero_collections or csl_json_files configured in topics.toml")
        return
    for seed_sync_outcome in synchronize_all_seed_sources(app.connection, app.config, build_gateway=build_gateway):
        print(f"Seeds: {seed_sync_outcome.describe()}")


def _import_recorded_fixtures(collection_context: CollectionContext, fixture_paths: list) -> None:
    """A replay: recorded records go through the same catalog, screening, promotion, and queueing as live pages."""
    for fixture_path in fixture_paths:
        paper_source = RecordedFixtureSource(fixture_path)
        page_result = ingest_records_with_checkpoints(
            collection_context.connection, paper_source.fetch_records(), admission_limit=None,
            active_seed_paper_ids=collection_context.active_seed_paper_ids,
            max_attempts_per_job=collection_context.app_config.worker.max_attempts_per_job)
        print(page_result.describe(paper_source.source_name))


def _print_research_profile_summary(app: AppContext) -> None:
    profiles = build_research_profiles(app.connection, app.config)
    for profile in profiles:
        print(f"Topic {profile.topic_id}: {profile.input_mode}; {len(profile.active_seeds)} active seed(s); "
              f"{len(profile.approved_public_query_ids)} approved public quer(y/ies)")
        for explanation in profile.explain_missing_inputs():
            print(f"  note: {explanation}")
    empty_profile_explanation = explain_empty_research_profile(profiles)
    if empty_profile_explanation:
        print(empty_profile_explanation)


def _print_lines(report_lines: list[str]) -> None:
    for report_line in report_lines:
        print(report_line)
