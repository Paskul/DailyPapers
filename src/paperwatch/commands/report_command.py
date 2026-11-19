"""`paperwatch report [--date YYYY-MM-DD]`: publish one Markdown report per local reporting date.

Publication order (so a crash at any point is recoverable and a rerun produces the same bytes):
  1. freeze the selection and render notes + report bytes,
  2. save the selection and the exact report bytes in the database as `pending`,
  3. write the notes, then the report, each via temp file + atomic rename,
  4. mark the report `published`.
A rerun for a `pending` date republishes the saved bytes; a rerun for a `published` date changes nothing.
Step-1 selection: newest completed review per unreported paper, ordered by best topic rating. Plan part 9 adds the
reading-priority floor, the writer role, and the scheduled cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import date

from paperwatch.app_context import AppContext
from paperwatch.catalog import load_paper_aliases
from paperwatch.clock import format_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.errors import ConfigurationError
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, read_service_health_state
from paperwatch.rendering.file_output import write_bytes_atomically
from paperwatch.rendering.notes import SelectedReview, render_daily_report, render_paper_note


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
        report_markdown = render_daily_report(
            report_date=report_date, timezone_name=app.config.report.timezone, cutoff_at=cutoff_at,
            selected_reviews=selected_reviews, report_statistics=_collect_report_statistics(app))
        report_bytes = report_markdown.encode("utf-8")
        _save_pending_report(app.connection, report_date, cutoff_at, report_output_path, report_bytes,
                             selected_reviews)

    for selected_review in selected_reviews:
        note_path = (app.config.paper_notes_dir / selected_review.paper_row["paper_id"]
                     / f"{selected_review.review_row['review_id']}.md")
        if not note_path.exists():  # notes are immutable per review revision; only write or restore
            write_bytes_atomically(note_path, render_paper_note(selected_review).encode("utf-8"))
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
             AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                     WHERE newest.paper_id = review.paper_id
                                     ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)""",
        (cutoff_at,)).fetchall()
    unranked_reviews = [_build_selected_review(app.connection, rank=0, review_row=row) for row in candidate_rows]
    unranked_reviews.sort(key=lambda candidate: (
        -(candidate.best_topic_rating if candidate.best_topic_rating is not None else -1),
        candidate.paper_row["paper_id"]))
    chosen_reviews = unranked_reviews[:app.config.report.max_papers_per_report]
    return [_build_selected_review(app.connection, rank=position + 1, review_row=chosen.review_row)
            for position, chosen in enumerate(chosen_reviews)]


def _load_frozen_selection(connection: sqlite3.Connection, report_date: str) -> list[SelectedReview]:
    item_rows = connection.execute(
        "SELECT rank, review_id FROM report_items WHERE report_date = ? ORDER BY rank", (report_date,)).fetchall()
    return [_build_selected_review(connection, rank=item_row["rank"], review_row=connection.execute(
        "SELECT * FROM reviews WHERE review_id = ?", (item_row["review_id"],)).fetchone()) for item_row in item_rows]


def _build_selected_review(connection: sqlite3.Connection, *, rank: int, review_row: sqlite3.Row) -> SelectedReview:
    review_content = json.loads(review_row["review_json"])
    topic_assessments = review_content.get("topic_relevance", [])
    best_assessment = max(topic_assessments, key=lambda assessment: assessment["relevance_rating"], default=None)
    return SelectedReview(
        rank=rank,
        paper_row=connection.execute("SELECT * FROM papers WHERE paper_id = ?", (review_row["paper_id"],)).fetchone(),
        review_row=review_row,
        review_content=review_content,
        paper_aliases=load_paper_aliases(connection, review_row["paper_id"]),
        best_topic_rating=best_assessment["relevance_rating"] if best_assessment else None,
        best_topic_id=best_assessment["topic_id"] if best_assessment else None)


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
                         report_bytes: bytes, selected_reviews: list[SelectedReview]) -> None:
    with write_transaction(connection):
        connection.execute(
            """INSERT INTO reports (report_date, cutoff_at, status, output_path, rendered_sha256, rendered_bytes,
                   created_at) VALUES (?, ?, 'pending', ?, ?, ?, ?)""",
            (report_date, cutoff_at, str(report_output_path), hashlib.sha256(report_bytes).hexdigest(),
             report_bytes, utc_now_timestamp()))
        connection.executemany(
            "INSERT INTO report_items (report_date, rank, paper_id, review_id) VALUES (?, ?, ?, ?)",
            [(report_date, selected.rank, selected.paper_row["paper_id"], selected.review_row["review_id"])
             for selected in selected_reviews])
