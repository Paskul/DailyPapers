"""`paperwatch collect`: pull records from sources into the catalog and queue review jobs.

The collector is the only component allowed to contact paper providers. Step 1 supports recorded fixtures only;
arXiv, OpenAlex and Zotero sources (plan parts 4-5) plug in through the same PaperSource protocol, and the
CPU screening score (part 6) replaces `should_promote_paper_for_review`.
"""

from __future__ import annotations

import argparse
import sqlite3

from paperwatch.app_context import AppContext
from paperwatch.catalog import load_paper_row, record_collected_paper, set_paper_status
from paperwatch.db import write_transaction
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND, JobPriorityClass, build_review_job_key, enqueue_job_if_absent
from paperwatch.review_inputs import build_abstract_review_packet
from paperwatch.sources.fixture_source import RecordedFixtureSource
from paperwatch.sources.types import PaperSource


def run_collect_command(app: AppContext, arguments: argparse.Namespace) -> int:
    paper_sources: list[PaperSource] = [RecordedFixtureSource(fixture_path)
                                        for fixture_path in (arguments.fixture_paths or [])]
    if not paper_sources:
        print("No sources to collect from. Live arXiv / OpenAlex / Zotero sources are not implemented yet "
              "(plan parts 4-5); pass --fixture FILE to import recorded papers.")
        return 0

    for paper_source in paper_sources:
        collected_records = paper_source.fetch_records()
        newly_created_count = already_known_count = review_jobs_queued_count = 0
        for collected_record in collected_records:
            catalog_outcome = record_collected_paper(app.connection, collected_record)
            if catalog_outcome.was_newly_created:
                newly_created_count += 1
            else:
                already_known_count += 1
            if _queue_review_if_promoted(app, catalog_outcome.paper_id):
                review_jobs_queued_count += 1
        print(f"{paper_source.source_name}: {len(collected_records)} record(s), {newly_created_count} new, "
              f"{already_known_count} already known, {review_jobs_queued_count} review job(s) queued")
    return 0


def should_promote_paper_for_review(paper_row: sqlite3.Row) -> bool:
    """PLACEHOLDER until the screening score of plan part 6: every collected paper is promoted."""
    return True


def _queue_review_if_promoted(app: AppContext, paper_id: str) -> bool:
    paper_row = load_paper_row(app.connection, paper_id)
    if not should_promote_paper_for_review(paper_row):
        return False
    review_packet = build_abstract_review_packet(paper_row)
    with write_transaction(app.connection):
        # The job key contains the review input key, so an unchanged paper never gets a second reading.
        job_was_added = enqueue_job_if_absent(
            app.connection,
            job_key=build_review_job_key(paper_id, review_packet.review_input_key),
            kind=REVIEW_PAPER_JOB_KIND, paper_id=paper_id, input_version=review_packet.review_input_key,
            input_payload={"coverage": review_packet.coverage},
            priority=JobPriorityClass.NEW_RELEASE, max_attempts=app.config.worker.max_attempts_per_job)
        if job_was_added:
            set_paper_status(app.connection, paper_id, "queued")
    return job_was_added
