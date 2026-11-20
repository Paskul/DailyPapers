"""`paperwatch report [--date YYYY-MM-DD]`: publish one Markdown report per local reporting date.

Publication order (so a crash at any point is recoverable and a rerun produces the same bytes):
  1. freeze the selection and render notes + report bytes,
  2. save the selection and the exact report bytes in the database as `pending`,
  3. write the notes, then the report, each via temp file + atomic rename,
  4. mark the report `published`.
A rerun for a `pending` date republishes the saved bytes; a rerun for a `published` date changes nothing.
Step-1 selection: newest completed review per unreported paper, ordered by its best topic assessment. Plan part 9
adds the reading-priority floor, the writer role, and the scheduled cutoff.
Plan part 5: papers a provider reported withdrawn or retracted are never selected. If such a paper was recommended
by an earlier report, the next report carries a correction (that earlier file is never altered), once.
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
from datetime import date

from paperwatch.app_context import AppContext
from paperwatch.clock import format_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.errors import ConfigurationError
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, read_service_health_state
from paperwatch.rendering.file_output import write_bytes_atomically
from paperwatch.rendering.notes import ReportCorrection, SelectedReview, render_daily_report
from paperwatch.rendering.review_notes import load_selected_review, write_paper_note_if_missing


def run_report_command(app: AppContext, arguments: argparse.Namespace) -> int:
    report_date = _resolve_report_date(app, arguments.report_date)
    report_output_path = app.config.daily_reports_dir / f"{report_date}.md"

    existing_report_row = app.connection.execute("SELECT * FROM reports WHERE report_date = ?",
                                                 (report_date,)).fetchone()
    if existing_report_row is not None and existing_report_row["status"] == "published":
        if not report_output_path.exists():  # restore a deleted file from the saved bytes
            write_bytes_atomically(report_output_path, existing_report_row["rendered_bytes"])
            print(f"restored missing report file {report_output_path}")
        print(f"report for {report_date} already published: {report_output_path}")
        return 0

    if existing_report_row is not None:  # pending: a previous run crashed before marking it published
        selected_reviews = _load_frozen_selection(app.connection, report_date)
        report_bytes = existing_report_row["rendered_bytes"]
    else:
        cutoff_at = format_utc_timestamp(utc_now())
        selected_reviews = _select_reviews_for_report(app, cutoff_at)
        corrections = _load_pending_corrections(app.connection)
        report_markdown = render_daily_report(
            report_date=report_date, timezone_name=app.config.report.timezone, cutoff_at=cutoff_at,
            selected_reviews=selected_reviews, report_statistics=_collect_report_statistics(app),
            corrections=corrections)
        report_bytes = report_markdown.encode("utf-8")
        _save_pending_report(app.connection, report_date, cutoff_at, report_output_path, report_bytes,
                             selected_reviews, corrections)

    for selected_review in selected_reviews:  # restore any missing note from its saved review first
        write_paper_note_if_missing(app.config.paper_notes_dir, selected_review)
    write_bytes_atomically(report_output_path, report_bytes)
    app.connection.execute("UPDATE reports SET status = 'published', published_at = ? WHERE report_date = ?",
                           (utc_now_timestamp(), report_date))
    print(f"published report for {report_date} with {len(selected_reviews)} paper(s): {report_output_path}")
    return 0


def _resolve_report_date(app: AppContext, requested_report_date: str | None) -> str:
    if requested_report_date is None:
        return utc_now().astimezone(app.config.report_timezone).date().isoformat()
    try:
        return date.fromisoformat(requested_report_date).isoformat()
    except ValueError:
        raise ConfigurationError(f"--date must be YYYY-MM-DD, got '{requested_report_date}'") from None


def _select_reviews_for_report(app: AppContext, cutoff_at: str) -> list[SelectedReview]:
    candidate_rows = app.connection.execute(
        """SELECT review.* FROM reviews AS review
           WHERE review.completed_at <= ?
             AND review.paper_id NOT IN (SELECT paper_id FROM report_items)
             AND review.paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
             AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                     WHERE newest.paper_id = review.paper_id
                                     ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)""",
        (cutoff_at,)).fetchall()
    unranked_reviews = [_build_selected_review(app.connection, rank=0, review_row=row) for row in candidate_rows]
    unranked_reviews.sort(key=lambda candidate: (  # legacy 0-4 and 0-100 priorities on one scale
        -(candidate.best_topic_assessment.priority_score_out_of_100
          if candidate.best_topic_assessment is not None else -1),
        candidate.paper_row["paper_id"]))
    chosen_reviews = unranked_reviews[:app.config.report.max_papers_per_report]
    return [_build_selected_review(app.connection, rank=position + 1, review_row=chosen.review_row)
            for position, chosen in enumerate(chosen_reviews)]


def _load_pending_corrections(connection: sqlite3.Connection) -> tuple[ReportCorrection, ...]:
    """Withdrawn or retracted papers that an earlier report recommended and no report has corrected yet."""
    return tuple(ReportCorrection(paper_id=row["paper_id"], title=row["title"], notice_kind=row["notice_kind"],
                                  first_reported_on=row["first_reported_on"], detected_at=row["detected_at"])
                 for row in connection.execute(
                     """SELECT withdrawal.paper_id, paper.title, withdrawal.notice_kind, withdrawal.detected_at,
                               MIN(item.report_date) AS first_reported_on
                        FROM paper_withdrawals AS withdrawal
                        JOIN report_items AS item ON item.paper_id = withdrawal.paper_id
                        JOIN papers AS paper ON paper.paper_id = withdrawal.paper_id
                        WHERE withdrawal.correction_report_date IS NULL
                        GROUP BY withdrawal.paper_id ORDER BY first_reported_on, withdrawal.paper_id"""))


def _load_frozen_selection(connection: sqlite3.Connection, report_date: str) -> list[SelectedReview]:
    item_rows = connection.execute(
        "SELECT rank, review_id FROM report_items WHERE report_date = ? ORDER BY rank", (report_date,)).fetchall()
    return [_build_selected_review(connection, rank=item_row["rank"], review_row=connection.execute(
        "SELECT * FROM reviews WHERE review_id = ?", (item_row["review_id"],)).fetchone()) for item_row in item_rows]


def _build_selected_review(connection: sqlite3.Connection, *, rank: int, review_row: sqlite3.Row) -> SelectedReview:
    return load_selected_review(connection, review_row["review_id"], rank=rank)


def _collect_report_statistics(app: AppContext) -> dict:
    def count(sql: str) -> int:
        return app.connection.execute(sql).fetchone()[0]

    model_health = read_service_health_state(app.connection, MODEL_SERVICE_NAME)
    model_outage_note = None
    if model_health.is_paused_at(utc_now()):
        model_outage_note = (f"unavailable ({model_health.consecutive_failures} consecutive failure(s), last error: "
                             f"{model_health.last_error})")
    return {
        "papers_in_catalog": count("SELECT COUNT(*) FROM papers"),
        "reviews_completed": count("SELECT COUNT(*) FROM reviews"),
        "jobs_queued": count("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"),
        "jobs_failed": count("SELECT COUNT(*) FROM jobs WHERE status = 'failed'"),
        "topics_configured": len(app.config.topics),
        "model_outage_note": model_outage_note,
    }


def _save_pending_report(connection: sqlite3.Connection, report_date: str, cutoff_at: str, report_output_path,
                         report_bytes: bytes, selected_reviews: list[SelectedReview],
                         corrections: tuple[ReportCorrection, ...] = ()) -> None:
    with write_transaction(connection):
        connection.executemany(
            "UPDATE paper_withdrawals SET correction_report_date = ? WHERE paper_id = ? AND "
            "correction_report_date IS NULL", [(report_date, correction.paper_id) for correction in corrections])
        connection.execute(
            """INSERT INTO reports (report_date, cutoff_at, status, output_path, rendered_sha256, rendered_bytes,
                   created_at) VALUES (?, ?, 'pending', ?, ?, ?, ?)""",
            (report_date, cutoff_at, str(report_output_path), hashlib.sha256(report_bytes).hexdigest(),
             report_bytes, utc_now_timestamp()))
        connection.executemany(
            "INSERT INTO report_items (report_date, rank, paper_id, review_id) VALUES (?, ?, ?, ?)",
            [(report_date, selected.rank, selected.paper_row["paper_id"], selected.review_row["review_id"])
             for selected in selected_reviews])
