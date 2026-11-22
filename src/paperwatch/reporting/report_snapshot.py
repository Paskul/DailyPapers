"""The frozen snapshot of one daily report: "At the cutoff, take a short database snapshot of completed,
eligible reviews ... Save the snapshot and the final Markdown bytes before publishing."

`freeze_report_snapshot` runs ONE short write transaction that reads everything the report needs and saves it:
  * the selection (report_items, with labels, frozen scores and update details),
  * the opening's statistics for the window since the previous report (found, screened, reviewed, backlog, source and
    model outages, time since the previous report),
  * the ids of completed comparisons that involve a selected review,
  * the corrections due for previously reported papers (marked as announced by this date),
and records the report as `writing`. From then on the report is rendered only from this snapshot, so a rerun after a
crash renders the same selection, and reviews completed later stay for the next report.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.db import write_transaction
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, ServiceHealthState, read_service_health_state
from paperwatch.network.provider_gateway import PROVIDER_SERVICE_NAME_PREFIX
from paperwatch.reporting.report_schedule import ReportTiming
from paperwatch.reporting.report_selection import ReportEntry, select_report_entries
from paperwatch.reporting.revision_changes import RevisionChange

REPORT_STATUS_WRITING = "writing"  # snapshot frozen; synthesis and rendering not finished
REPORT_STATUS_PENDING = "pending"  # exact bytes saved; the file may not be written yet
REPORT_STATUS_PUBLISHED = "published"
BEGINNING_OF_TIME = "0000"  # sorts before every stored timestamp


@dataclass(frozen=True)
class ReportCorrection:
    """A previously reported paper that a provider now reports withdrawn or retracted.

    The historical report file is never altered; the next report carries this correction instead."""

    paper_id: str
    title: str
    notice_kind: str  # withdrawn | retracted
    first_reported_on: str  # report date (YYYY-MM-DD) that recommended it
    detected_at: str  # UTC timestamp of the provider notice


@dataclass(frozen=True)
class ReportStatistics:
    """The opening's numbers. "Window" means after the previous report's cutoff, up to this report's cutoff."""

    previous_report_date: str | None
    hours_since_previous_report: float | None
    papers_found_in_window: int
    papers_screened_in_window: int
    reviews_completed_in_window: int
    papers_in_catalog: int
    reviews_in_total: int
    eligible_held_for_later: int
    below_floor: int
    jobs_queued: int
    jobs_failed: int
    topics_configured: int
    source_outages: tuple[str, ...]  # host-written lines
    model_outage: str | None  # set when the model server was unavailable at the snapshot or failed in the window


@dataclass(frozen=True)
class ReportSnapshot:
    report_date: str
    timezone_name: str
    publish_time: str
    scheduled_cutoff_at: str  # stored UTC timestamps
    cutoff_at: str
    published_early: bool
    minimum_reading_priority: int
    entries: tuple[ReportEntry, ...]
    statistics: ReportStatistics
    comparison_ids: tuple[str, ...]
    corrections: tuple[ReportCorrection, ...]

    @property
    def selected_review_ids(self) -> list[str]:
        return [entry.review_id for entry in self.entries]

    def snapshot_record(self) -> dict[str, Any]:
        """What reports.snapshot_json stores (entries live in report_items, timing in the reports columns)."""
        return {
            "timezone_name": self.timezone_name,
            "publish_time": self.publish_time,
            "minimum_reading_priority": self.minimum_reading_priority,
            "statistics": asdict(self.statistics),
            "comparison_ids": list(self.comparison_ids),
            "corrections": [asdict(correction) for correction in self.corrections],
        }


# --------------------------------------------------------------------------------------------------------------------
# Freezing
# --------------------------------------------------------------------------------------------------------------------


def freeze_report_snapshot(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    timing: ReportTiming,
    report_output_path: Path,
    *,
    active_seed_paper_ids: frozenset[str],
) -> ReportSnapshot:
    """Read and save the snapshot in one transaction; the report row becomes `writing`."""
    cutoff_at = format_utc_timestamp(timing.cutoff_at)
    with write_transaction(connection):
        selection = select_report_entries(
            connection,
            report_date=timing.report_date,
            cutoff_at=cutoff_at,
            minimum_reading_priority=app_config.report.minimum_reading_priority,
            maximum_entries=app_config.report.max_papers_per_report,
            excluded_paper_ids=active_seed_paper_ids,
        )
        snapshot = ReportSnapshot(
            report_date=timing.report_date,
            timezone_name=app_config.report.timezone,
            publish_time=app_config.report.publish_time,
            scheduled_cutoff_at=format_utc_timestamp(timing.scheduled_cutoff_at),
            cutoff_at=cutoff_at,
            published_early=timing.published_early,
            minimum_reading_priority=app_config.report.minimum_reading_priority,
            entries=selection.entries,
            statistics=_collect_statistics(
                connection,
                app_config,
                timing,
                cutoff_at,
                eligible_held_for_later=selection.eligible_held_for_later_count,
                below_floor=selection.below_floor_count,
            ),
            comparison_ids=_comparisons_involving(
                connection, [entry.review_id for entry in selection.entries], cutoff_at
            ),
            corrections=_pending_corrections(connection, timing.report_date),
        )
        _save_frozen_snapshot(connection, snapshot, report_output_path)
    return snapshot


def _save_frozen_snapshot(connection: sqlite3.Connection, snapshot: ReportSnapshot, report_output_path: Path) -> None:
    connection.executemany(
        "UPDATE paper_withdrawals SET correction_report_date = ? WHERE paper_id = ? AND correction_report_date IS NULL",
        [(snapshot.report_date, correction.paper_id) for correction in snapshot.corrections],
    )
    connection.execute(
        """INSERT INTO reports (report_date, scheduled_cutoff_at, cutoff_at, published_early, status, output_path,
               snapshot_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot.report_date,
            snapshot.scheduled_cutoff_at,
            snapshot.cutoff_at,
            int(snapshot.published_early),
            REPORT_STATUS_WRITING,
            str(report_output_path),
            json.dumps(snapshot.snapshot_record(), ensure_ascii=False),
            utc_now_timestamp(),
        ),
    )
    connection.executemany(
        """INSERT INTO report_items (report_date, rank, paper_id, review_id, entry_label, reading_priority,
               best_topic_id, previous_review_id, previous_report_date, revision_changes_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                snapshot.report_date,
                entry.rank,
                entry.paper_id,
                entry.review_id,
                entry.entry_label,
                entry.reading_priority,
                entry.best_topic_id,
                entry.previous_review_id,
                entry.previous_report_date,
                json.dumps([change.to_record() for change in entry.revision_changes], ensure_ascii=False),
            )
            for entry in snapshot.entries
        ],
    )


# --------------------------------------------------------------------------------------------------------------------
# Loading a frozen snapshot (rerun after a crash)
# --------------------------------------------------------------------------------------------------------------------


def load_frozen_report_snapshot(connection: sqlite3.Connection, report_date: str) -> ReportSnapshot:
    report_row = connection.execute("SELECT * FROM reports WHERE report_date = ?", (report_date,)).fetchone()
    snapshot_record = json.loads(report_row["snapshot_json"] or "{}")
    entries = tuple(
        ReportEntry(
            rank=item_row["rank"],
            paper_id=item_row["paper_id"],
            review_id=item_row["review_id"],
            entry_label=item_row["entry_label"],
            reading_priority=item_row["reading_priority"],
            best_topic_id=item_row["best_topic_id"],
            coverage=item_row["coverage"],
            previous_review_id=item_row["previous_review_id"],
            previous_report_date=item_row["previous_report_date"],
            revision_changes=tuple(
                RevisionChange.from_record(record) for record in json.loads(item_row["revision_changes_json"])
            ),
        )
        for item_row in connection.execute(
            """SELECT item.*, review.coverage FROM report_items AS item
               JOIN reviews AS review ON review.review_id = item.review_id
               WHERE item.report_date = ? ORDER BY item.rank""",
            (report_date,),
        )
    )
    statistics_record = snapshot_record["statistics"]
    return ReportSnapshot(
        report_date=report_date,
        timezone_name=snapshot_record["timezone_name"],
        publish_time=snapshot_record["publish_time"],
        scheduled_cutoff_at=report_row["scheduled_cutoff_at"],
        cutoff_at=report_row["cutoff_at"],
        published_early=bool(report_row["published_early"]),
        minimum_reading_priority=snapshot_record["minimum_reading_priority"],
        entries=entries,
        statistics=ReportStatistics(
            **{**statistics_record, "source_outages": tuple(statistics_record["source_outages"])}
        ),
        comparison_ids=tuple(snapshot_record["comparison_ids"]),
        corrections=tuple(ReportCorrection(**record) for record in snapshot_record["corrections"]),
    )


# --------------------------------------------------------------------------------------------------------------------
# What the snapshot reads
# --------------------------------------------------------------------------------------------------------------------


def _collect_statistics(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    timing: ReportTiming,
    cutoff_at: str,
    *,
    eligible_held_for_later: int,
    below_floor: int,
) -> ReportStatistics:
    previous_report = connection.execute(
        "SELECT report_date, cutoff_at FROM reports WHERE report_date < ? ORDER BY report_date DESC LIMIT 1",
        (timing.report_date,),
    ).fetchone()
    window_start = previous_report["cutoff_at"] if previous_report else BEGINNING_OF_TIME
    window = (window_start, cutoff_at)

    def count(sql: str, parameters: tuple[Any, ...] = ()) -> int:
        return int(connection.execute(sql, parameters).fetchone()[0])

    hours_since_previous = None
    if previous_report is not None:
        hours_since_previous = round(
            (timing.cutoff_at - parse_utc_timestamp(previous_report["cutoff_at"])).total_seconds() / 3600, 1
        )
    return ReportStatistics(
        previous_report_date=previous_report["report_date"] if previous_report else None,
        hours_since_previous_report=hours_since_previous,
        papers_found_in_window=count(
            "SELECT COUNT(*) FROM papers WHERE discovered_at > ? AND discovered_at <= ?", window
        ),
        papers_screened_in_window=count(
            "SELECT COUNT(DISTINCT paper_id) FROM candidate_topic_scores WHERE scored_at > ? AND scored_at <= ?", window
        ),
        reviews_completed_in_window=count(
            "SELECT COUNT(*) FROM reviews WHERE completed_at > ? AND completed_at <= ?", window
        ),
        papers_in_catalog=count("SELECT COUNT(*) FROM papers"),
        reviews_in_total=count("SELECT COUNT(*) FROM reviews WHERE completed_at <= ?", (cutoff_at,)),
        eligible_held_for_later=eligible_held_for_later,
        below_floor=below_floor,
        jobs_queued=count("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"),
        jobs_failed=count("SELECT COUNT(*) FROM jobs WHERE status = 'failed'"),
        topics_configured=len(app_config.topics),
        source_outages=_describe_source_outages(connection, window),
        model_outage=_describe_service_outage(
            read_service_health_state(connection, MODEL_SERVICE_NAME), window, include_error=True
        ),
    )


def _describe_service_outage(health: ServiceHealthState, window: tuple[str, str], *, include_error: bool) -> str | None:
    """A description when the service is paused now, or failed inside the window; None when it was healthy."""
    moment = utc_now()
    if (paused_until := health.pause_end_at(moment)) is not None:
        description = (
            f"unavailable at the cutoff ({health.consecutive_failures} consecutive failure(s), "
            f"paused until {format_utc_timestamp(paused_until)})"
        )
    elif health.last_failure_at and window[0] < health.last_failure_at <= window[1]:
        description = f"failed at least once since the previous report (last at {health.last_failure_at})"
    else:
        return None
    if include_error and health.last_error:
        description += f"; last error: {health.last_error[:200]}"
    return description


def _describe_source_outages(connection: sqlite3.Connection, window: tuple[str, str]) -> tuple[str, ...]:
    """Paper providers that were paused or failed, and searches that were blocked, inside the window. Provider error
    texts are left out: they can contain request details (keys are redacted in the ledger, not here)."""
    outage_lines = []
    for service_row in connection.execute(
        "SELECT service FROM service_state WHERE service LIKE ? || '%' ORDER BY service",
        (PROVIDER_SERVICE_NAME_PREFIX,),
    ):
        provider_name = service_row["service"].removeprefix(PROVIDER_SERVICE_NAME_PREFIX)
        description = _describe_service_outage(
            read_service_health_state(connection, service_row["service"]), window, include_error=False
        )
        if description:
            outage_lines.append(f"provider {provider_name} {description}")
    for checkpoint_row in connection.execute(
        """SELECT search_key, stop_reason FROM source_checkpoints WHERE coverage_status = 'blocked'
               AND updated_at > ? AND updated_at <= ? ORDER BY search_key""",
        window,
    ):
        outage_lines.append(
            f"search {checkpoint_row['search_key']} blocked: {checkpoint_row['stop_reason'] or 'no reason recorded'}"
        )
    return tuple(outage_lines)


def _comparisons_involving(connection: sqlite3.Connection, review_ids: list[str], cutoff_at: str) -> tuple[str, ...]:
    """Completed comparisons (as of the cutoff) in which a selected review takes part."""
    if not review_ids:
        return ()
    placeholders = ", ".join("?" for _ in review_ids)
    return tuple(
        row["comparison_id"]
        for row in connection.execute(
            f"""SELECT comparison_id FROM comparisons WHERE completed_at <= ?
            AND (lower_review_id IN ({placeholders}) OR higher_review_id IN ({placeholders}))
            ORDER BY completed_at, comparison_id""",  # noqa: S608  # only ? placeholders are interpolated
            (cutoff_at, *review_ids, *review_ids),
        )
    )


def _pending_corrections(connection: sqlite3.Connection, report_date: str) -> tuple[ReportCorrection, ...]:
    """Withdrawn or retracted papers that an earlier report recommended and no report has corrected yet."""
    return tuple(
        ReportCorrection(
            paper_id=row["paper_id"],
            title=row["title"],
            notice_kind=row["notice_kind"],
            first_reported_on=row["first_reported_on"],
            detected_at=row["detected_at"],
        )
        for row in connection.execute(
            """SELECT withdrawal.paper_id, paper.title, withdrawal.notice_kind, withdrawal.detected_at,
                               MIN(item.report_date) AS first_reported_on
                        FROM paper_withdrawals AS withdrawal
                        JOIN report_items AS item ON item.paper_id = withdrawal.paper_id
                        JOIN papers AS paper ON paper.paper_id = withdrawal.paper_id
                        WHERE withdrawal.correction_report_date IS NULL AND item.report_date < ?
                        GROUP BY withdrawal.paper_id ORDER BY first_reported_on, withdrawal.paper_id""",
            (report_date,),
        )
    )
