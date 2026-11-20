"""`paperwatch collect`: sync seed papers, pull records from sources into the catalog, and queue review jobs.

The collector is the only component allowed to contact paper providers (its process arms the collector egress
policy), and it does so only through paperwatch.network.provider_gateway.ProviderGateway's fixed operations.

Stages of one run:
  1. Seeds (plan part 4): sync every configured Zotero collection and CSL JSON file (paperwatch.seeds.seed_sync).
     A source that cannot be read keeps its last successful seed set; collection continues either way.
  2. Papers: recorded fixtures for now; arXiv and OpenAlex sources (plan part 5) will call the gateway and parse its
     responses into the same PaperSource protocol. The CPU screening score (part 6) replaces
     `should_promote_paper_for_review`.
  3. Research profile: print each topic's input mode (keywords, seeds, or both) and anything missing.

Re-collecting is always safe: the catalog resolves each record to its canonical paper, and a reading is queued only
when the paper's preferred edition has text that was never read (see paperwatch.readings).
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from paperwatch.app_context import AppContext
from paperwatch.catalog import CatalogUpsertOutcome, load_paper_row, record_collected_paper, set_paper_status
from paperwatch.db import write_transaction
from paperwatch.jobs import JobPriorityClass
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed
from paperwatch.research_profile import (build_research_profiles, explain_empty_research_profile,
                                         load_active_seed_library_paper_ids)
from paperwatch.seeds.seed_store import seed_sources_of_all_topics
from paperwatch.seeds.seed_sync import synchronize_all_seed_sources
from paperwatch.sources.fixture_source import RecordedFixtureSource
from paperwatch.sources.types import PaperSource


@dataclass
class SourceCollectionCounts:
    records: int = 0
    new_papers: int = 0
    already_known: int = 0
    new_editions_of_known_papers: int = 0
    review_jobs_queued: int = 0
    possible_duplicates_flagged: int = 0

    def describe(self, source_name: str) -> str:
        return (f"{source_name}: {self.records} record(s), {self.new_papers} new, {self.already_known} already known, "
                f"{self.review_jobs_queued} review job(s) queued, {self.new_editions_of_known_papers} new edition(s) "
                f"of known papers, {self.possible_duplicates_flagged} possible duplicate(s) flagged")


def run_collect_command(app: AppContext, arguments: argparse.Namespace, *,
                        build_gateway: Callable[[], ProviderGateway] | None = None) -> int:
    """`build_gateway` lets tests inject a gateway with a recorded transport; production uses the collector's."""
    _synchronize_seed_sources(app, build_gateway or (lambda: ProviderGateway.for_collector(app.connection,
                                                                                            app.config)))
    _collect_papers_from_sources(app, arguments)
    _print_research_profile_summary(app)
    return 0


def _synchronize_seed_sources(app: AppContext, build_gateway: Callable[[], ProviderGateway]) -> None:
    if not seed_sources_of_all_topics(app.config.topics):
        print("Seeds: no zotero_collections or csl_json_files configured in topics.toml")
        return
    for seed_sync_outcome in synchronize_all_seed_sources(app.connection, app.config, build_gateway=build_gateway):
        print(f"Seeds: {seed_sync_outcome.describe()}")


def _collect_papers_from_sources(app: AppContext, arguments: argparse.Namespace) -> None:
    paper_sources: list[PaperSource] = [RecordedFixtureSource(fixture_path)
                                        for fixture_path in (getattr(arguments, "fixture_paths", None) or [])]
    if not paper_sources:
        print("Papers: no paper sources to collect from. Live arXiv / OpenAlex searches arrive in plan part 5; "
              "pass --fixture FILE to import recorded papers.")
        return

    active_seed_paper_ids = load_active_seed_library_paper_ids(app.connection, app.config)
    for paper_source in paper_sources:
        collected_records = paper_source.fetch_records()
        source_counts = SourceCollectionCounts(records=len(collected_records))
        for collected_record in collected_records:
            catalog_outcome = record_collected_paper(app.connection, collected_record)
            _count_catalog_outcome(source_counts, catalog_outcome)
            if _queue_review_if_promoted(app, catalog_outcome.paper_id, active_seed_paper_ids):
                source_counts.review_jobs_queued += 1
        print(source_counts.describe(paper_source.source_name))


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


def should_promote_paper_for_review(paper_row: sqlite3.Row, active_seed_paper_ids: frozenset[str]) -> bool:
    """PLACEHOLDER until the screening score of plan part 6: every collected paper is promoted, except papers already
    in the user's active seed library (they are comparison anchors, never recommendations; plan parts 4 and 9)."""
    return paper_row["paper_id"] not in active_seed_paper_ids


def _count_catalog_outcome(source_counts: SourceCollectionCounts, catalog_outcome: CatalogUpsertOutcome) -> None:
    if catalog_outcome.was_newly_created:
        source_counts.new_papers += 1
    else:
        source_counts.already_known += 1
        if catalog_outcome.was_new_edition:
            source_counts.new_editions_of_known_papers += 1
    source_counts.possible_duplicates_flagged += len(catalog_outcome.possible_match_paper_ids)


def _queue_review_if_promoted(app: AppContext, paper_id: str,
                              active_seed_paper_ids: frozenset[str] = frozenset()) -> bool:
    paper_row = load_paper_row(app.connection, paper_id)
    if not should_promote_paper_for_review(paper_row, active_seed_paper_ids):
        return False
    with write_transaction(app.connection):
        enqueue_outcome, _review_packet = enqueue_reading_if_needed(
            app.connection, paper_id, priority=JobPriorityClass.NEW_RELEASE,
            max_attempts=app.config.worker.max_attempts_per_job)
        if enqueue_outcome == ReadingEnqueueOutcome.QUEUED:
            set_paper_status(app.connection, paper_id, "queued")
    return enqueue_outcome == ReadingEnqueueOutcome.QUEUED
