"""`paperwatch report [--date YYYY-MM-DD] [--early]`: publish one coherent Markdown report per reporting date.

Publication runs in recoverable stages, one process at a time (state/report-publication.lock) and one row per date:
  1. FREEZE    at the cutoff, one short transaction saves the snapshot: the selected reviews (with labels, frozen
               scores, and update details), the opening's statistics, the comparisons involved, and the corrections
               due. The report is now `writing`. (reporting/report_snapshot.py)
  2. SYNTHESIZE  reserve the next inference slot and run the writer under the deadline, or fall back to the saved
               summaries with a label. Not called when nothing was selected. (reporting/report_synthesis.py)
  3. RENDER    the Markdown from the snapshot, the saved reviews, and the synthesis; save the exact bytes and the
               synthesis in one transaction. The report is now `pending`. (rendering/daily_report.py)
  4. PUBLISH   restore any missing reading or comparison note from the database, write the report through a temp
               file and an atomic rename, then mark the report `published`.
A rerun resumes where the previous run stopped: `writing` -> synthesize again over the same frozen selection;
`pending` -> write the saved bytes (a kill between the rename and the database update republishes identical bytes);
`published` -> restore the file from the saved bytes if it was deleted, otherwise change nothing.
Eligibility and the default date are explained in reporting/report_schedule.py and reporting/report_selection.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from paperwatch.activity_log import COMPONENT_REPORT, recorded_activity
from paperwatch.app_context import AppContext
from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.rendering.comparison_notes import restore_missing_comparison_notes
from paperwatch.rendering.daily_report import render_daily_report
from paperwatch.rendering.file_output import write_bytes_atomically
from paperwatch.rendering.notes import SelectedReview
from paperwatch.rendering.review_notes import load_selected_review, write_paper_note_if_missing
from paperwatch.reporting.publication_lock import hold_report_publication_lock
from paperwatch.reporting.report_comparisons import load_comparison_summaries
from paperwatch.reporting.report_schedule import resolve_report_timing
from paperwatch.reporting.report_snapshot import (REPORT_STATUS_PENDING, REPORT_STATUS_PUBLISHED,
                                                  REPORT_STATUS_WRITING, ReportSnapshot, freeze_report_snapshot,
                                                  load_frozen_report_snapshot)
from paperwatch.reporting.report_synthesis import ReportSynthesis, write_report_synthesis
from paperwatch.research_profile import load_active_seed_library_paper_ids


def run_report_command(app: AppContext, arguments: argparse.Namespace) -> int:
    timing = resolve_report_timing(app.config.report, utc_now(), arguments.report_date,
                                   publish_early=getattr(arguments, "publish_early", False))
    report_date = timing.report_date
    report_output_path = app.config.daily_reports_dir / f"{report_date}.md"

    with (hold_report_publication_lock(app.config.report_publication_lock_path),
          recorded_activity(app.connection, "run", {"report_date": report_date, "early": timing.published_early},
                            component=COMPONENT_REPORT) as run_activity):
        report_row = _load_report_row(app.connection, report_date)
        if report_row is not None and report_row["status"] == REPORT_STATUS_PUBLISHED:
            if not report_output_path.exists():  # restore a deleted file from the saved bytes
                write_bytes_atomically(report_output_path, report_row["rendered_bytes"])
                print(f"restored missing report file {report_output_path}")
            print(f"report for {report_date} already published: {report_output_path}")
            run_activity.end_as("already_published")
            return 0

        if report_row is None:  # stage 1
            with recorded_activity(app.connection, "freeze"):
                freeze_report_snapshot(app.connection, app.config, timing, report_output_path,
                                       active_seed_paper_ids=load_active_seed_library_paper_ids(app.connection,
                                                                                                app.config))
            report_row = _load_report_row(app.connection, report_date)
        if report_row["status"] == REPORT_STATUS_WRITING:  # stages 2 and 3 (also after a crash in them)
            with recorded_activity(app.connection, "synthesize"):
                _synthesize_and_render(app, report_date)
        with recorded_activity(app.connection, "publish"):
            published_paper_count = _publish_saved_report(app, report_date, report_output_path)  # stage 4
        run_activity.add_detail(papers=published_paper_count)

    synthesis_status = _load_report_row(app.connection, report_date)["synthesis_status"]
    print(f"published report for {report_date} with {published_paper_count} paper(s) "
          f"(synthesis: {synthesis_status}): {report_output_path}")
    return 0


def _load_report_row(connection: sqlite3.Connection, report_date: str) -> sqlite3.Row | None:
    return connection.execute("SELECT * FROM reports WHERE report_date = ?", (report_date,)).fetchone()


def _synthesize_and_render(app: AppContext, report_date: str) -> None:
    snapshot = load_frozen_report_snapshot(app.connection, report_date)
    selected_reviews = load_selected_reviews_as_frozen(app.connection, snapshot)
    comparisons = load_comparison_summaries(app.connection, snapshot.comparison_ids)
    synthesis = write_report_synthesis(app, snapshot, selected_reviews, comparisons)
    report_bytes = render_daily_report(snapshot, selected_reviews, comparisons, synthesis).encode("utf-8")
    _save_rendered_report(app.connection, report_date, report_bytes, synthesis)


def load_selected_reviews_as_frozen(connection: sqlite3.Connection,
                                    snapshot: ReportSnapshot) -> list[SelectedReview]:
    """The selected reviews in rank order, with topic assessments as of the cutoff."""
    return [load_selected_review(connection, entry.review_id, rank=entry.rank,
                                 assessed_no_later_than=snapshot.cutoff_at) for entry in snapshot.entries]


def _save_rendered_report(connection: sqlite3.Connection, report_date: str, report_bytes: bytes,
                          synthesis: ReportSynthesis) -> None:
    synthesis_record = synthesis.synthesis_record()
    with write_transaction(connection):
        connection.execute(
            """UPDATE reports SET status = ?, rendered_bytes = ?, rendered_sha256 = ?, synthesis_status = ?,
                   synthesis_json = ?, fallback_reason = ?
               WHERE report_date = ? AND status = ?""",
            (REPORT_STATUS_PENDING, report_bytes, hashlib.sha256(report_bytes).hexdigest(),
             synthesis.synthesis_status,
             None if synthesis_record is None else json.dumps(synthesis_record, ensure_ascii=False),
             synthesis.fallback_reason, report_date, REPORT_STATUS_WRITING))


def _publish_saved_report(app: AppContext, report_date: str, report_output_path: Path) -> int:
    """Restore missing notes first, then write the saved bytes atomically, then mark the report published."""
    report_row = _load_report_row(app.connection, report_date)
    item_rows = app.connection.execute("SELECT review_id FROM report_items WHERE report_date = ? ORDER BY rank",
                                       (report_date,)).fetchall()
    for item_row in item_rows:  # notes render from the saved review, exactly as the worker wrote them
        write_paper_note_if_missing(app.config.paper_notes_dir, load_selected_review(app.connection,
                                                                                     item_row["review_id"]))
    for previous_row in app.connection.execute(  # an update links the earlier note too
            "SELECT previous_review_id FROM report_items WHERE report_date = ? AND previous_review_id IS NOT NULL",
            (report_date,)):
        write_paper_note_if_missing(app.config.paper_notes_dir,
                                    load_selected_review(app.connection, previous_row["previous_review_id"]))
    restore_missing_comparison_notes(app.connection, app.config.comparison_notes_dir)
    write_bytes_atomically(report_output_path, report_row["rendered_bytes"])
    app.connection.execute("UPDATE reports SET status = ?, published_at = ? WHERE report_date = ?",
                           (REPORT_STATUS_PUBLISHED, utc_now_timestamp(), report_date))
    return len(item_rows)
