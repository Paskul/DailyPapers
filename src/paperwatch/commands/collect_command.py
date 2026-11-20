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
  3. Source text: for papers queued for review, a local file, else (live runs only) a capped download; then text
     extraction in the sandbox when the extractor image is configured.
  4. Research profile: print each topic's input mode (keywords, seeds, or both) and anything missing.

Re-collecting is always safe: the catalog resolves each record to its canonical paper, every page commits with its
checkpoint, and a reading is queued only when the paper's preferred edition has text that was never read.
The CPU screening score of part 6 replaces the placeholder promotion in paperwatch.collection.review_promotion.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from paperwatch.app_context import AppContext
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.current_releases import collect_current_releases
from paperwatch.collection.historical_search import run_historical_searches
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.collection.source_text import acquire_source_texts
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.research_profile import (build_research_profiles, explain_empty_research_profile,
                                         load_active_seed_library_paper_ids)
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

    _synchronize_seed_sources(app, build_gateway_once)
    collection_context = CollectionContext.start(
        app.connection, app.config, build_gateway_once,
        active_seed_paper_ids=load_active_seed_library_paper_ids(app.connection, app.config))
    if fixture_paths:
        _import_recorded_fixtures(collection_context, fixture_paths)
    elif run_live_searches:
        _print_lines(collect_current_releases(collection_context))
        _print_lines(run_historical_searches(collection_context))
        print(f"Candidates: {collection_context.candidate_budget.describe()}")
    else:
        print("Papers: live arXiv / OpenAlex searches skipped (--skip-searches)")
    _print_lines(acquire_source_texts(collection_context, allow_network=run_live_searches))
    _print_research_profile_summary(app)
    return 0


def _synchronize_seed_sources(app: AppContext, build_gateway: Callable[[], ProviderGateway]) -> None:
    if not seed_sources_of_all_topics(app.config.topics):
        print("Seeds: no zotero_collections or csl_json_files configured in topics.toml")
        return
    for seed_sync_outcome in synchronize_all_seed_sources(app.connection, app.config, build_gateway=build_gateway):
        print(f"Seeds: {seed_sync_outcome.describe()}")


def _import_recorded_fixtures(collection_context: CollectionContext, fixture_paths: list) -> None:
    """A replay: recorded records go through the same catalog, promotion, and queueing as live pages."""
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
