"""Store one page of collected records and its checkpoint(s) in ONE write transaction (plan parts 2 and 5).

So a crash, a kill, or a failure mid-search leaves either the page AND its cursor, or neither: the next run resumes
at the first page that was not stored, and no page is skipped. Records are normalized before the transaction; a
malformed record is skipped and counted, never allowed to roll back its neighbours.

The admission limit caps how many papers NEW to the catalog this page may add (the daily candidate budget, a topic's
historical budget). Records of papers already in the catalog are always stored: they add editions, routes, and
revisions, and cost no candidate budget. When the limit stops a page part-way, the caller keeps the page's cursor, so
the page is fetched again later and its remaining new papers are admitted then.

Ingestion never decides relevance: a new paper waits for the screening stage of part 6 (paperwatch.screening). Only a
paper screening already promoted gets a reading queued here, when this page brings it new text.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from paperwatch.catalog import prepare_catalog_record, record_prepared_paper_in_transaction, would_create_new_paper
from paperwatch.db import write_transaction
from paperwatch.errors import SourceInputError
from paperwatch.screening.review_promotion import queue_new_reading_for_promoted_paper_in_transaction
from paperwatch.source_checkpoints import SourceCheckpoint, save_source_checkpoint
from paperwatch.sources.types import CollectedPaperRecord


@dataclass
class PageIngestionResult:
    records_received: int = 0
    new_papers_admitted: int = 0
    already_known: int = 0
    new_editions_of_known_papers: int = 0
    review_jobs_queued: int = 0  # new readings of already promoted papers (new papers wait for screening)
    possible_duplicates_flagged: int = 0
    withdrawals_reported: int = 0
    skipped_malformed_records: int = 0
    skipped_over_admission_limit: int = 0
    queued_paper_ids: list[str] = field(default_factory=list)
    catalogued_paper_ids: list[str] = field(default_factory=list)  # every paper a stored record landed on, in order

    @property
    def was_stopped_by_admission_limit(self) -> bool:
        return self.skipped_over_admission_limit > 0

    def add(self, other: PageIngestionResult) -> None:
        for counter_name in (
            "records_received",
            "new_papers_admitted",
            "already_known",
            "new_editions_of_known_papers",
            "review_jobs_queued",
            "possible_duplicates_flagged",
            "withdrawals_reported",
            "skipped_malformed_records",
            "skipped_over_admission_limit",
        ):
            setattr(self, counter_name, getattr(self, counter_name) + getattr(other, counter_name))
        self.queued_paper_ids += other.queued_paper_ids
        self.catalogued_paper_ids += other.catalogued_paper_ids

    def describe(self, source_label: str) -> str:
        description = (
            f"{source_label}: {self.records_received} record(s), {self.new_papers_admitted} new, "
            f"{self.already_known} already known, {self.review_jobs_queued} new reading(s) of promoted "
            f"papers queued, {self.new_editions_of_known_papers} new edition(s) of known papers, "
            f"{self.possible_duplicates_flagged} possible duplicate(s) flagged"
        )
        extras = [
            f"{self.withdrawals_reported} withdrawn/retracted" if self.withdrawals_reported else "",
            f"{self.skipped_malformed_records} malformed record(s) skipped" if self.skipped_malformed_records else "",
            f"{self.skipped_over_admission_limit} new record(s) left for later (candidate budget)"
            if self.skipped_over_admission_limit
            else "",
        ]
        return description + "".join(f", {extra}" for extra in extras if extra)


CheckpointsAfterPage = Callable[[PageIngestionResult], list[SourceCheckpoint]]


def ingest_records_with_checkpoints(
    connection: sqlite3.Connection,
    collected_records: list[CollectedPaperRecord],
    *,
    admission_limit: int | None,
    active_seed_paper_ids: frozenset[str],
    max_attempts_per_job: int,
    checkpoints_after_page: CheckpointsAfterPage | None = None,
) -> PageIngestionResult:
    """Catalog the records, queue promoted papers, and save the checkpoints the callback returns, atomically.

    `checkpoints_after_page` runs INSIDE the transaction after the records, so it may read rows (e.g. a topic's
    historical plan) and base the new cursors on what this page actually admitted.
    """
    result = PageIngestionResult(records_received=len(collected_records))
    prepared_records = []
    for collected_record in collected_records:
        try:
            prepared_records.append(prepare_catalog_record(collected_record))
        except SourceInputError:
            result.skipped_malformed_records += 1

    with write_transaction(connection):
        for prepared_record in prepared_records:
            if (
                admission_limit is not None
                and result.new_papers_admitted >= admission_limit
                and would_create_new_paper(connection, prepared_record)
            ):
                result.skipped_over_admission_limit += 1
                continue
            catalog_outcome = record_prepared_paper_in_transaction(connection, prepared_record)
            result.catalogued_paper_ids.append(catalog_outcome.paper_id)
            if catalog_outcome.was_newly_created:
                result.new_papers_admitted += 1
            else:
                result.already_known += 1
                result.new_editions_of_known_papers += int(catalog_outcome.was_new_edition)
            result.possible_duplicates_flagged += len(catalog_outcome.possible_match_paper_ids)
            result.withdrawals_reported += int(prepared_record.collected_record.withdrawal_notice is not None)
            if queue_new_reading_for_promoted_paper_in_transaction(
                connection,
                catalog_outcome.paper_id,
                active_seed_paper_ids=active_seed_paper_ids,
                max_attempts=max_attempts_per_job,
            ):
                result.review_jobs_queued += 1
                result.queued_paper_ids.append(catalog_outcome.paper_id)
        for checkpoint in checkpoints_after_page(result) if checkpoints_after_page else []:
            save_source_checkpoint(connection, checkpoint)
    return result
