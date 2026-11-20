"""The monitor's plan panel: what will happen between now and the next report, and what limits the day.

  schedule   every collect / report / pilot-finish moment of the installed timers (or, with none installed, the
             configured times marked as "not installed": nothing then runs on its own);
  queue      the jobs in the order the worker will claim them (the same ORDER BY as jobs.claim_next_due_job);
  estimate   how many of them fit before the next report, from the model minutes measured per completed reading;
  budgets    today's (UTC) counters against their daily limits.
"""

from __future__ import annotations

import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp
from paperwatch.collection.candidate_budget import CandidateAdmissionBudget
from paperwatch.comparisons.comparison_queue import count_comparison_jobs_created_today
from paperwatch.config import AppConfig
from paperwatch.jobs import EFFECTIVE_PRIORITY_CLASS_SQL
from paperwatch.monitor.current_activity import JOB_KIND_LABELS, priority_class_label
from paperwatch.monitor.database_access import load_paper_titles
from paperwatch.monitor.schedule_state import ScheduleState
from paperwatch.network.provider_rules import load_provider_access_rules
from paperwatch.references.reference_selection import count_reference_admissions_today
from paperwatch.reporting.report_schedule import parse_clock_time, scheduled_cutoff_for_report_date
from paperwatch.screening.review_promotion import count_screening_promotions_today, utc_day_start_timestamp

NEXT_JOBS_SHOWN = 12
ESTIMATE_WINDOW = timedelta(days=7)
MOMENT_LABELS = {"collect": "Collect papers", "report": "Publish the daily report",
                 "pilot_finish": "Pilot ends (its units are removed)"}


def next_report_cutoff_after(moment: datetime, app_config: AppConfig) -> datetime:
    local_date = moment.astimezone(app_config.report_timezone).date()
    cutoff = scheduled_cutoff_for_report_date(local_date, app_config.report)
    return cutoff if cutoff > moment else scheduled_cutoff_for_report_date(local_date + timedelta(days=1),
                                                                           app_config.report)


def describe_plan(connection: sqlite3.Connection, app_config: AppConfig, *, now: datetime, window_start: datetime,
                  horizon_end: datetime, schedule: ScheduleState, next_report_cutoff: datetime) -> dict[str, Any]:
    return {
        "schedule_source": schedule.schedule_source,
        "systemd_available": schedule.systemd_available,
        "moments": describe_scheduled_moments(app_config, now=now, window_start=window_start,
                                              horizon_end=horizon_end, schedule=schedule),
        "queue": describe_queue(connection, now=now),
        "estimate": estimate_until_cutoff(connection, now=now, next_report_cutoff=next_report_cutoff),
        "budgets": describe_daily_budgets(connection, app_config, now=now),
    }


def describe_scheduled_moments(app_config: AppConfig, *, now: datetime, window_start: datetime,
                               horizon_end: datetime, schedule: ScheduleState) -> list[dict[str, Any]]:
    if schedule.moments:
        return [{"at": format_utc_timestamp(moment.at), "kind": moment.kind, "label": MOMENT_LABELS[moment.kind],
                 "source": moment.schedule_source, "installed": True, "past": moment.at <= now}
                for moment in schedule.moments if window_start <= moment.at <= horizon_end]
    # Nothing installed: show the configured times so the operator sees what a schedule WOULD do.
    configured = []
    local_day = window_start.astimezone(app_config.report_timezone).date()
    while True:
        day_moments = [(clock_text, "collect") for clock_text in app_config.schedule.collect_times] + [
            (app_config.report.publish_time, "report")]
        for clock_text, kind in day_moments:
            moment = datetime.combine(local_day, parse_clock_time(clock_text), tzinfo=app_config.report_timezone)
            if window_start <= moment <= horizon_end:
                configured.append({"at": format_utc_timestamp(moment), "kind": kind, "label": MOMENT_LABELS[kind],
                                   "source": "config", "installed": False, "past": moment <= now})
        if datetime.combine(local_day, datetime.min.time(), tzinfo=app_config.report_timezone) > horizon_end:
            return sorted(configured, key=lambda entry: entry["at"])
        local_day += timedelta(days=1)


def describe_queue(connection: sqlite3.Connection, *, now: datetime) -> dict[str, Any]:
    now_text = format_utc_timestamp(now)
    class_rows = connection.execute(
        f"""SELECT {EFFECTIVE_PRIORITY_CLASS_SQL} AS effective_class, COUNT(*) AS job_count, MIN(created_at) AS oldest,
                   SUM(due_at > :now) AS held_count
            FROM jobs WHERE status = 'queued' GROUP BY effective_class ORDER BY effective_class""",
        {"now": now_text}).fetchall()
    next_rows = connection.execute(
        f"""SELECT job_id, kind, paper_id, priority, attempts, max_attempts, due_at, last_error,
                   {EFFECTIVE_PRIORITY_CLASS_SQL} AS effective_class
            FROM jobs WHERE status = 'queued'
            ORDER BY due_at > :now, {EFFECTIVE_PRIORITY_CLASS_SQL}, within_class_order_primary DESC,
                     within_class_order_secondary DESC, created_at, job_id
            LIMIT {NEXT_JOBS_SHOWN}""", {"now": now_text}).fetchall()
    # Due jobs first, in claim order; a held job (e.g. waiting for its PDF) waits for its due time whatever its class.
    titles = load_paper_titles(connection, [row["paper_id"] for row in next_rows])
    return {
        "total": sum(row["job_count"] for row in class_rows),
        "by_class": [{"label": priority_class_label(row["effective_class"]), "count": row["job_count"],
                      "held": row["held_count"] or 0,
                      "oldest_hours": round((now - parse_utc_timestamp(row["oldest"])).total_seconds() / 3600, 1)}
                     for row in class_rows],
        "next_jobs": [{"job_id": row["job_id"], "kind_label": JOB_KIND_LABELS.get(row["kind"], row["kind"]),
                       "paper_id": row["paper_id"], "title": titles.get(row["paper_id"]),
                       "class_label": priority_class_label(row["effective_class"]), "due_at": row["due_at"],
                       "held": row["due_at"] > now_text, "attempts": row["attempts"],
                       "max_attempts": row["max_attempts"], "retrying": bool(row["attempts"])}
                      for row in next_rows],
    }


def estimate_until_cutoff(connection: sqlite3.Connection, *, now: datetime,
                          next_report_cutoff: datetime) -> dict[str, Any]:
    """Readings that fit before the next report = model minutes left / mean model minutes per finished reading
    (all its calls, repairs and rejected answers included), measured over the last ESTIMATE_WINDOW."""
    window_start_text = format_utc_timestamp(now - ESTIMATE_WINDOW)
    review_minutes = [row["model_seconds"] / 60 for row in connection.execute(
        """SELECT (SELECT COALESCE(SUM(call.duration_seconds), 0) FROM model_calls AS call
                   WHERE call.job_id = review.job_id) AS model_seconds
           FROM reviews AS review WHERE review.completed_at >= ? AND review.job_id IS NOT NULL""",
        (window_start_text,))]
    call_rows = connection.execute(
        """SELECT outcome, duration_seconds FROM model_calls WHERE started_at >= ?
             AND role_name IN ('section_reader', 'reviewer', 'topic_assessor')""", (window_start_text,)).fetchall()
    minutes_left = max(0.0, (next_report_cutoff - now).total_seconds() / 60)
    queued_reviews = connection.execute(
        "SELECT COUNT(*) FROM jobs WHERE kind = 'review_paper' AND status IN ('queued', 'running')").fetchone()[0]
    mean_minutes_per_review = statistics.fmean(review_minutes) if review_minutes else None
    estimate = {
        "minutes_until_cutoff": round(minutes_left),
        "cutoff": format_utc_timestamp(next_report_cutoff),
        "queued_readings": queued_reviews,
        "completed_readings_measured": len(review_minutes),
        "mean_minutes_per_reading": round(mean_minutes_per_review, 1) if mean_minutes_per_review else None,
        "reading_calls_measured": len(call_rows),
        "mean_minutes_per_call": round(statistics.fmean(row["duration_seconds"] for row in call_rows) / 60, 1)
        if call_rows else None,
        "rejected_call_share": round(sum(row["outcome"] != "ok" for row in call_rows) / len(call_rows), 2)
        if call_rows else None,
        "readings_before_cutoff": None,
    }
    if mean_minutes_per_review:
        estimate["readings_before_cutoff"] = min(queued_reviews, int(minutes_left // mean_minutes_per_review))
    return estimate


def describe_daily_budgets(connection: sqlite3.Connection, app_config: AppConfig, *,
                           now: datetime) -> list[dict[str, Any]]:
    """Today's counters (UTC day, like the limits themselves). A counter that cannot be read is left out."""
    budgets: list[dict[str, Any]] = []

    def add(label: str, compute_used, limit: int | None, note: str | None = None) -> None:
        try:
            budgets.append({"label": label, "used": int(compute_used()), "limit": limit, "note": note})
        except (sqlite3.Error, KeyError, TypeError):
            pass

    candidate_budget = None
    try:
        candidate_budget = CandidateAdmissionBudget.load_for_day(connection, app_config.collection, now)
    except sqlite3.Error:
        pass
    if candidate_budget is not None:
        budgets.append({"label": "New candidates", "used": candidate_budget.admitted_today,
                        "limit": candidate_budget.daily_limit,
                        "note": f"{candidate_budget.admitted_today_by_current_releases} current, "
                                f"{candidate_budget.admitted_today_by_historical_search} historical"})

    def promotions_today() -> int:
        counts = count_screening_promotions_today(connection, now)
        return counts["regular_count"] + counts["near_cutoff_count"]

    add("Readings promoted", promotions_today, app_config.screening.daily_promotion_limit)
    add("Full-text downloads", lambda: connection.execute(
        "SELECT COUNT(*) FROM edition_source_texts WHERE downloaded_from_network = 1 AND acquired_at >= ?",
        (utc_day_start_timestamp(now),)).fetchone()[0], app_config.collection.daily_source_text_budget)
    add("Reference readings", lambda: count_reference_admissions_today(connection, now),
        app_config.references.daily_reference_readings)
    add("Comparisons queued", lambda: count_comparison_jobs_created_today(connection, now),
        app_config.comparisons.daily_comparison_jobs)
    try:
        provider_rules = load_provider_access_rules()
        utc_date_text = now.astimezone(timezone.utc).date().isoformat()
        for row in connection.execute(
                "SELECT provider_name, requests_reserved FROM provider_request_budgets WHERE budget_date = ? "
                "ORDER BY provider_name", (utc_date_text,)):
            rule = provider_rules.provider_rule(row["provider_name"])
            budgets.append({"label": f"{row['provider_name']} requests", "used": row["requests_reserved"],
                            "limit": rule.daily_request_budget,
                            "note": None if rule.daily_request_budget_without_credential is None else
                            f"{rule.daily_request_budget_without_credential} without an API key"})
    except (sqlite3.Error, KeyError, ValueError):
        pass
    return budgets
