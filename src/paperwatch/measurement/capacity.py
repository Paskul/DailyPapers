"""How many papers per day the local model can really read (plan part 10, "Measure capacity").

Everything comes from what the pipeline already records, so it measures the actual model, reasoning settings, and
context on this workstation:
  model_calls   one row per inference request: role, duration, outcome, tokens, job (repairs and failed attempts
                included, so retries cost what they really cost);
  reviews       one row per completed reading, with its job and coverage (abstract_only / partial_text / full_text);
  jobs          the queue right now.
The plan's estimate: daily capacity = available inference minutes / measured minutes per review, after reserving
time for research, comparisons, and the report. Text extraction runs in the collector, on the CPU, in parallel with
the model, so it is not part of the model's minutes (the collect run's duration shows it in the service log).
"""

from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

from paperwatch.clock import format_utc_timestamp
from paperwatch.config import AppConfig

MINUTES_PER_DAY = 24 * 60
READING_ROLE_NAMES = ("section_reader", "reviewer", "topic_assessor")  # the roles one paper reading uses
RESERVED_ROLE_NAMES = ("research", "comparison", "writer")  # the plan's reservations before readings


@dataclass(frozen=True)
class DurationSummary:
    count: int
    median_minutes: float
    p90_minutes: float
    maximum_minutes: float
    total_minutes: float

    @classmethod
    def of_seconds(cls, durations_in_seconds: list[float]) -> DurationSummary:
        minutes = sorted(duration / 60.0 for duration in durations_in_seconds)
        if not minutes:
            return cls(0, 0.0, 0.0, 0.0, 0.0)
        p90_index = min(len(minutes) - 1, max(0, round(0.9 * len(minutes)) - 1))
        return cls(len(minutes), statistics.median(minutes), minutes[p90_index], minutes[-1], sum(minutes))

    def describe(self) -> str:
        return (f"n={self.count}, median {self.median_minutes:.1f} min, p90 {self.p90_minutes:.1f} min, "
                f"max {self.maximum_minutes:.1f} min")


@dataclass(frozen=True)
class RoleCallSummary:
    role_name: str
    outcomes: dict[str, int]  # outcome -> count
    successful_calls: DurationSummary
    all_calls_total_minutes: float  # failed and invalid calls cost model time too
    completion_tokens_per_second: float | None


@dataclass(frozen=True)
class CapacityMeasurement:
    window_start: datetime
    window_end: datetime
    roles: list[RoleCallSummary]
    minutes_per_review_by_coverage: dict[str, DurationSummary]  # coverage -> model minutes per completed review
    model_calls_per_review_median: float | None
    reviews_completed: int
    review_jobs_created: int
    review_jobs_queued_now: int
    oldest_queued_review_hours: float | None
    reserved_minutes_per_day: float
    estimated_reviews_per_day: float | None
    configured_readings_per_day: int  # screening promotions + reference readings

    @property
    def window_minutes(self) -> float:
        return (self.window_end - self.window_start).total_seconds() / 60.0

    @property
    def model_busy_fraction(self) -> float:
        busy_minutes = sum(role.all_calls_total_minutes for role in self.roles)
        return busy_minutes / self.window_minutes if self.window_minutes > 0 else 0.0


def measure_capacity(connection: sqlite3.Connection, app_config: AppConfig, *, window_end: datetime,
                     window_hours: float) -> CapacityMeasurement:
    window_start = window_end - timedelta(hours=window_hours)
    start_text, end_text = format_utc_timestamp(window_start), format_utc_timestamp(window_end)
    roles = _summarize_roles(connection, start_text, end_text)

    review_rows = connection.execute(
        """SELECT review.coverage, review.job_id,
                  (SELECT COALESCE(SUM(call.duration_seconds), 0) FROM model_calls AS call
                   WHERE call.job_id = review.job_id) AS model_seconds,
                  (SELECT COUNT(*) FROM model_calls AS call WHERE call.job_id = review.job_id) AS call_count
           FROM reviews AS review WHERE review.completed_at >= ? AND review.completed_at <= ?
             AND review.job_id IS NOT NULL""", (start_text, end_text)).fetchall()
    seconds_by_coverage: dict[str, list[float]] = {}
    for review_row in review_rows:
        seconds_by_coverage.setdefault(review_row["coverage"], []).append(review_row["model_seconds"])
    minutes_per_review_by_coverage = {coverage: DurationSummary.of_seconds(seconds)
                                      for coverage, seconds in sorted(seconds_by_coverage.items())}

    reserved_minutes_per_day = _reserved_minutes_per_day(roles, window_hours, app_config)
    all_review_minutes = [row["model_seconds"] / 60.0 for row in review_rows]
    mean_minutes_per_review = statistics.fmean(all_review_minutes) if all_review_minutes else None
    estimated_reviews_per_day = (max(0.0, MINUTES_PER_DAY - reserved_minutes_per_day) / mean_minutes_per_review
                                 if mean_minutes_per_review else None)

    queued_row = connection.execute(
        """SELECT COUNT(*) AS queued, MIN(created_at) AS oldest FROM jobs
           WHERE kind = 'review_paper' AND status IN ('queued', 'running')""").fetchone()
    oldest_queued_review_hours = None
    if queued_row["oldest"]:
        oldest_moment = datetime.fromisoformat(queued_row["oldest"].replace("Z", "+00:00"))
        oldest_queued_review_hours = (window_end - oldest_moment).total_seconds() / 3600.0
    return CapacityMeasurement(
        window_start=window_start, window_end=window_end, roles=roles,
        minutes_per_review_by_coverage=minutes_per_review_by_coverage,
        model_calls_per_review_median=statistics.median([row["call_count"] for row in review_rows])
        if review_rows else None,
        reviews_completed=len(review_rows),
        review_jobs_created=connection.execute(
            # distinct papers: a queued abstract reading replaced by its full-text reading is one paper, not two
            "SELECT COUNT(DISTINCT paper_id) FROM jobs WHERE kind = 'review_paper' AND created_at >= ? "
            "AND created_at <= ?",
            (start_text, end_text)).fetchone()[0],
        review_jobs_queued_now=queued_row["queued"], oldest_queued_review_hours=oldest_queued_review_hours,
        reserved_minutes_per_day=reserved_minutes_per_day, estimated_reviews_per_day=estimated_reviews_per_day,
        configured_readings_per_day=app_config.screening.daily_promotion_limit
        + app_config.references.daily_reference_readings)


def _summarize_roles(connection: sqlite3.Connection, start_text: str, end_text: str) -> list[RoleCallSummary]:
    call_rows = connection.execute(
        """SELECT role_name, outcome, duration_seconds, completion_tokens FROM model_calls
           WHERE started_at >= ? AND started_at <= ? ORDER BY role_name""", (start_text, end_text)).fetchall()
    roles: list[RoleCallSummary] = []
    for role_name in sorted({row["role_name"] for row in call_rows}):
        role_rows = [row for row in call_rows if row["role_name"] == role_name]
        successful_rows = [row for row in role_rows if row["outcome"] == "ok"]
        outcomes: dict[str, int] = {}
        for row in role_rows:
            outcomes[row["outcome"]] = outcomes.get(row["outcome"], 0) + 1
        token_seconds = sum(row["duration_seconds"] for row in successful_rows if row["completion_tokens"])
        tokens = sum(row["completion_tokens"] or 0 for row in successful_rows)
        roles.append(RoleCallSummary(
            role_name=role_name, outcomes=outcomes,
            successful_calls=DurationSummary.of_seconds([row["duration_seconds"] for row in successful_rows]),
            all_calls_total_minutes=sum(row["duration_seconds"] for row in role_rows) / 60.0,
            completion_tokens_per_second=tokens / token_seconds if token_seconds > 0 else None))
    return roles


def _reserved_minutes_per_day(roles: list[RoleCallSummary], window_hours: float, app_config: AppConfig) -> float:
    """Research and comparison minutes scaled to a day, plus the report: its measured writer time, or its whole
    deadline if no writer call was measured yet."""
    scale_to_day = 24.0 / window_hours if window_hours > 0 else 1.0
    minutes_by_role = {role.role_name: role.all_calls_total_minutes for role in roles}
    reserved_minutes = sum(minutes_by_role.get(role_name, 0.0) for role_name in ("research", "comparison")) \
        * scale_to_day
    writer_minutes = minutes_by_role.get("writer")
    reserved_minutes += min(writer_minutes, app_config.report.writer_deadline_minutes) if writer_minutes \
        else app_config.report.writer_deadline_minutes
    return reserved_minutes


def describe_capacity(measurement: CapacityMeasurement) -> list[str]:
    lines = [f"Capacity over {measurement.window_minutes / 60:.1f} h "
             f"({format_utc_timestamp(measurement.window_start)} .. {format_utc_timestamp(measurement.window_end)}): "
             f"model busy {measurement.model_busy_fraction:.0%} of the time"]
    if not measurement.roles:
        return lines + ["  no model calls in this window"]
    lines.append("Model calls by role (successful-call durations; total includes failed and repaired calls):")
    for role in measurement.roles:
        speed = f", {role.completion_tokens_per_second:.1f} tokens/s" if role.completion_tokens_per_second else ""
        outcome_text = ", ".join(f"{outcome} {count}" for outcome, count in sorted(role.outcomes.items()))
        lines.append(f"  {role.role_name:<15} {role.successful_calls.describe()}; total "
                     f"{role.all_calls_total_minutes:.0f} min{speed}; outcomes: {outcome_text}")
    lines.append(f"Completed reviews: {measurement.reviews_completed}"
                 + (f" (median {measurement.model_calls_per_review_median:.0f} model calls each)"
                    if measurement.model_calls_per_review_median is not None else ""))
    for coverage, summary in measurement.minutes_per_review_by_coverage.items():
        lines.append(f"  {coverage:<14} model minutes per review: {summary.describe()}")
    lines.append(f"Queue: {measurement.review_jobs_created} reading job(s) created in the window, "
                 f"{measurement.reviews_completed} completed; {measurement.review_jobs_queued_now} waiting now"
                 + (f", oldest {measurement.oldest_queued_review_hours:.1f} h"
                    if measurement.oldest_queued_review_hours is not None else ""))
    if measurement.estimated_reviews_per_day is None:
        lines.append("Estimate: no completed review in this window yet")
        return lines
    lines.append(f"Estimate: ({MINUTES_PER_DAY} min - {measurement.reserved_minutes_per_day:.0f} min reserved for "
                 f"research, comparisons, and the report) / mean minutes per review = "
                 f"about {measurement.estimated_reviews_per_day:.0f} reading(s) per day; configured: up to "
                 f"{measurement.configured_readings_per_day} (screening.daily_promotion_limit + "
                 "references.daily_reference_readings)")
    if measurement.estimated_reviews_per_day < measurement.configured_readings_per_day:
        lines.append("  The configured limits exceed the measured capacity: the queue will grow. If it grows over "
                     "consecutive days, lower screening.daily_promotion_limit (or reasoning effort) toward "
                     f"{max(1, int(measurement.estimated_reviews_per_day))} minus the reference readings you want.")
    return lines
