"""`paperwatch collect`: pull records from sources into the catalog and queue review jobs.

The collector is the only component allowed to contact paper providers (its process arms the collector egress
policy), and it does so only through paperwatch.network.provider_gateway.ProviderGateway's fixed operations.
Recorded fixtures are the only source so far; arXiv, OpenAlex and Zotero sources (plan parts 4-5) will call the
gateway and parse its responses into the same PaperSource protocol, and the CPU screening score (part 6) replaces
`should_promote_paper_for_review`.

Re-collecting is always safe: the catalog resolves each record to its canonical paper, and a reading is queued only
when the paper's preferred edition has text that was never read (see paperwatch.readings).
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

from paperwatch.app_context import AppContext
from paperwatch.catalog import CatalogUpsertOutcome, load_paper_row, record_collected_paper, set_paper_status
from paperwatch.db import write_transaction
from paperwatch.jobs import JobPriorityClass
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed
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


def run_collect_command(app: AppContext, arguments: argparse.Namespace) -> int:
    paper_sources: list[PaperSource] = [RecordedFixtureSource(fixture_path)
                                        for fixture_path in (arguments.fixture_paths or [])]
    if not paper_sources:
        print("No sources to collect from. Live arXiv / OpenAlex / Zotero sources are not implemented yet "
              "(plan parts 4-5); pass --fixture FILE to import recorded papers.")
        return 0

    for paper_source in paper_sources:
        collected_records = paper_source.fetch_records()
        source_counts = SourceCollectionCounts(records=len(collected_records))
        for collected_record in collected_records:
            catalog_outcome = record_collected_paper(app.connection, collected_record)
            _count_catalog_outcome(source_counts, catalog_outcome)
            if _queue_review_if_promoted(app, catalog_outcome.paper_id):
                source_counts.review_jobs_queued += 1
        print(source_counts.describe(paper_source.source_name))
    return 0


def should_promote_paper_for_review(paper_row: sqlite3.Row) -> bool:
    """PLACEHOLDER until the screening score of plan part 6: every collected paper is promoted."""
    return True


def _count_catalog_outcome(source_counts: SourceCollectionCounts, catalog_outcome: CatalogUpsertOutcome) -> None:
    if catalog_outcome.was_newly_created:
        source_counts.new_papers += 1
    else:
        source_counts.already_known += 1
        if catalog_outcome.was_new_edition:
            source_counts.new_editions_of_known_papers += 1
    source_counts.possible_duplicates_flagged += len(catalog_outcome.possible_match_paper_ids)


def _queue_review_if_promoted(app: AppContext, paper_id: str) -> bool:
    paper_row = load_paper_row(app.connection, paper_id)
    if not should_promote_paper_for_review(paper_row):
        return False
    with write_transaction(app.connection):
        enqueue_outcome, _review_packet = enqueue_reading_if_needed(
            app.connection, paper_id, priority=JobPriorityClass.NEW_RELEASE,
            max_attempts=app.config.worker.max_attempts_per_job)
        if enqueue_outcome == ReadingEnqueueOutcome.QUEUED:
            set_paper_status(app.connection, paper_id, "queued")
    return enqueue_outcome == ReadingEnqueueOutcome.QUEUED
