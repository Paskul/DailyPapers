"""The monitor's history: timeline lanes, a feed of notable events, and per-day totals for the recent window.

Lanes (bars with a start and an end):
  collect  collect runs: from the activity log (with their stages), else reconstructed from what a run leaves behind
           (outbound requests, promotions, acquired full texts) by grouping moments closer than COLLECT_GAP;
  jobs     one bar per worker job: from the activity log, else the span of the job's model calls;
  model    one bar per model call (the log of every inference request), plus the call in progress;
  report   report runs: from the activity log, else each report's creation (freeze) to its publication.
The window usually reaches back into the previous day, so the overnight work stays visible in the morning.
"""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp
from paperwatch.config import AppConfig
from paperwatch.monitor.activity_rows import ActivityStep, ActivityTree
from paperwatch.monitor.current_activity import JOB_KIND_LABELS, describe_activity_step
from paperwatch.monitor.database_access import load_job_papers, load_paper_titles

COLLECT_GAP = timedelta(minutes=20)
MAXIMUM_EVENTS = 150
MAXIMUM_ERROR_CHARACTERS = 240
QUIET_OUTCOMES = ("ok", "redirect", "not_modified", "not_found")


def describe_history(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    *,
    now: datetime,
    window_start: datetime,
    activity: ActivityTree | None,
    running_job_ids: list[str],
) -> dict[str, Any]:
    window_text = format_utc_timestamp(window_start)
    model_lane = _model_call_lane(
        connection, now=now, window_text=window_text, activity=activity, running_job_ids=running_job_ids
    )
    collect_lane = _collect_lane(connection, now=now, window_text=window_text, activity=activity)
    job_lane = _job_lane(connection, now=now, activity=activity, model_lane=model_lane)
    report_lane = _report_lane(connection, now=now, window_text=window_text, activity=activity)
    events = _events(
        connection,
        app_config,
        now=now,
        window_text=window_text,
        activity=activity,
        collect_lane=collect_lane,
        model_lane=model_lane,
    )
    return {
        "lanes": {"collect": collect_lane, "jobs": job_lane, "model": model_lane, "report": report_lane},
        "events": events,
        "days": _day_totals(connection, app_config, window_start=window_start, now=now),
    }


def latest_collect_run(collect_lane: list[dict[str, Any]]) -> dict[str, Any] | None:
    finished_runs = [run for run in collect_lane if not run.get("running")]
    return finished_runs[-1] if finished_runs else None


# --------------------------------------------------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------------------------------------------------


def _model_call_lane(
    connection: sqlite3.Connection,
    *,
    now: datetime,
    window_text: str,
    activity: ActivityTree | None,
    running_job_ids: list[str],
) -> list[dict[str, Any]]:
    call_rows = connection.execute(
        """SELECT call_id, role_name, prompt_version, job_id, started_at, duration_seconds, outcome,
                  completion_tokens, error_message FROM model_calls WHERE started_at >= ? ORDER BY started_at""",
        (window_text,),
    ).fetchall()
    job_rows = load_job_papers(connection, [row["job_id"] for row in call_rows] + running_job_ids)
    titles = load_paper_titles(connection, [row["paper_id"] for row in job_rows.values()])

    def title_of(job_id: str | None) -> str | None:
        job_row = job_rows.get(job_id) if job_id else None
        return titles.get(job_row["paper_id"]) if job_row else None

    lane = []
    for row in call_rows:
        started = parse_utc_timestamp(row["started_at"])
        lane.append(
            {
                "start": row["started_at"],
                "end": format_utc_timestamp(started + timedelta(seconds=row["duration_seconds"])),
                "role": row["role_name"],
                "repair": row["prompt_version"].endswith("+repair"),
                "outcome": row["outcome"],
                "job_id": row["job_id"],
                "title": title_of(row["job_id"]),
                "minutes": round(row["duration_seconds"] / 60, 1),
                "tokens": row["completion_tokens"],
                "error": _shorten(row["error_message"]),
                "running": False,
            }
        )
    running_calls = [step for step in (activity.all_steps("model_call") if activity else []) if step.is_running]
    for step in running_calls:
        lane.append(
            {
                "start": format_utc_timestamp(step.started_at),
                "end": format_utc_timestamp(now),
                "role": step.detail.get("role"),
                "repair": bool(step.detail.get("repair")),
                "outcome": "running",
                "job_id": step.detail.get("job_id"),
                "title": title_of(step.detail.get("job_id")),
                "running": True,
                "inferred": False,
                "minutes": round((now - step.started_at).total_seconds() / 60, 1),
            }
        )
    if not running_calls and running_job_ids:
        # One inference at a time: the call in progress began when the previous call ended.
        last_end = max((parse_utc_timestamp(call["end"]) for call in lane), default=None)
        if last_end is not None and last_end <= now:
            lane.append(
                {
                    "start": format_utc_timestamp(last_end),
                    "end": format_utc_timestamp(now),
                    "role": None,
                    "repair": False,
                    "outcome": "running",
                    "job_id": running_job_ids[0],
                    "title": title_of(running_job_ids[0]),
                    "running": True,
                    "inferred": True,
                    "minutes": round((now - last_end).total_seconds() / 60, 1),
                }
            )
    return lane


def _collect_lane(
    connection: sqlite3.Connection, *, now: datetime, window_text: str, activity: ActivityTree | None
) -> list[dict[str, Any]]:
    activity_runs = [run for run in (activity.roots_of("collect", "run") if activity else [])]
    if activity_runs:
        return [_collect_run_from_activity(connection, run, now) for run in activity_runs]
    return _collect_runs_inferred(connection, now=now, window_text=window_text)


def _collect_run_from_activity(connection: sqlite3.Connection, run: ActivityStep, now: datetime) -> dict[str, Any]:
    entry = run.as_json(now=now)
    entry.update(_collect_run_counts(connection, entry["start"], entry["end"]))
    entry["stages"] = [{**stage.as_json(now=now), "label": describe_activity_step(stage)} for stage in run.children]
    entry["inferred"] = False
    return entry


def _collect_runs_inferred(connection: sqlite3.Connection, *, now: datetime, window_text: str) -> list[dict[str, Any]]:
    """Group the traces a collect run leaves (requests, promotions, acquired text) into runs."""
    moments: list[tuple[datetime, datetime]] = []
    for row in connection.execute(
        "SELECT started_at, duration_seconds FROM outbound_requests WHERE started_at >= ?", (window_text,)
    ):
        started = parse_utc_timestamp(row["started_at"])
        moments.append((started, started + timedelta(seconds=row["duration_seconds"])))
    for sql in (
        "SELECT promoted_at AS at FROM paper_promotions WHERE promoted_at >= ?",
        "SELECT acquired_at AS at FROM edition_source_texts WHERE acquired_at >= ?",
    ):
        for row in connection.execute(sql, (window_text,)):
            if row["at"]:
                moment = parse_utc_timestamp(row["at"])
                moments.append((moment, moment))
    moments.sort()
    groups: list[list[datetime]] = []
    for started, ended in moments:
        if groups and started - groups[-1][1] <= COLLECT_GAP:
            groups[-1][1] = max(groups[-1][1], ended)
        else:
            groups.append([started, ended])
    runs = []
    for started, ended in groups:
        start_text, end_text = format_utc_timestamp(started), format_utc_timestamp(ended)
        entry = {
            "start": start_text,
            "end": end_text,
            "step": "run",
            "component": "collect",
            "inferred": True,
            "running": now - ended < timedelta(minutes=3),
            "outcome": "ok",
            "stages": [],
            "note": None,
        }
        entry.update(_collect_run_counts(connection, start_text, end_text))
        if entry["failed_requests"]:
            entry["outcome"] = "partial"
        runs.append(entry)
    return runs


def _collect_run_counts(connection: sqlite3.Connection, start_text: str, end_text: str) -> dict[str, Any]:
    requests = connection.execute(
        """SELECT provider_name, outcome, COUNT(*) AS request_count FROM outbound_requests
           WHERE started_at >= ? AND started_at <= ? GROUP BY provider_name, outcome""",
        (start_text, end_text),
    ).fetchall()
    by_provider: Counter[str] = Counter()
    failed = 0
    for row in requests:
        by_provider[row["provider_name"]] += row["request_count"]
        if row["outcome"] not in QUIET_OUTCOMES:
            failed += row["request_count"]
    return {
        "requests": dict(by_provider),
        "failed_requests": failed,
        "new_papers": connection.execute(
            "SELECT COUNT(*) FROM papers WHERE discovered_at >= ? AND discovered_at <= ?", (start_text, end_text)
        ).fetchone()[0],
        "promoted": connection.execute(
            "SELECT COUNT(*) FROM paper_promotions WHERE promoted_at >= ? AND promoted_at <= ?", (start_text, end_text)
        ).fetchone()[0],
        "full_texts": connection.execute(
            "SELECT COUNT(*) FROM edition_source_texts WHERE acquired_at >= ? AND acquired_at <= ?",
            (start_text, end_text),
        ).fetchone()[0],
    }


def _job_lane(
    connection: sqlite3.Connection, *, now: datetime, activity: ActivityTree | None, model_lane: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    job_steps = activity.all_steps("job") if activity else []
    if job_steps:
        job_rows = load_job_papers(connection, [step.detail.get("job_id") for step in job_steps])
        titles = load_paper_titles(connection, [row["paper_id"] for row in job_rows.values()])
        lane = []
        for step in sorted(job_steps, key=lambda step: step.started_at):
            entry = step.as_json(now=now)
            job_row = job_rows.get(step.detail.get("job_id"))
            entry.update(
                {
                    "job_id": step.detail.get("job_id"),
                    "kind_label": JOB_KIND_LABELS.get(step.detail.get("kind"), step.detail.get("kind")),
                    "title": titles.get(job_row["paper_id"]) if job_row else None,
                    "inferred": False,
                }
            )
            lane.append(entry)
        return lane
    # Older databases: a job's bar spans its model calls (claim and commit take seconds, not minutes).
    spans: dict[str, dict[str, Any]] = {}
    for call in model_lane:
        if not call.get("job_id"):
            continue
        span = spans.setdefault(
            call["job_id"],
            {
                "job_id": call["job_id"],
                "start": call["start"],
                "end": call["end"],
                "title": call["title"],
                "calls": 0,
                "running": False,
                "inferred": True,
            },
        )
        span["start"], span["end"] = min(span["start"], call["start"]), max(span["end"], call["end"])
        span["calls"] += 1
        span["running"] = span["running"] or call["running"]
    for job_id, span in spans.items():
        status = connection.execute("SELECT kind, status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        span["kind_label"] = JOB_KIND_LABELS.get(status["kind"], status["kind"]) if status else None
        span["outcome"] = (
            "running"
            if span["running"]
            else {"done": "ok", "failed": "failed"}.get(status["status"] if status else "", "retry")
        )
    return sorted(spans.values(), key=lambda span: span["start"])


def _report_lane(
    connection: sqlite3.Connection, *, now: datetime, window_text: str, activity: ActivityTree | None
) -> list[dict[str, Any]]:
    activity_runs = activity.roots_of("report", "run") if activity else []
    if activity_runs:
        return [
            {
                **run.as_json(now=now),
                "label": f"Report {run.detail.get('report_date', '')}",
                "inferred": False,
                "stages": [
                    {**stage.as_json(now=now), "label": describe_activity_step(stage)} for stage in run.children
                ],
            }
            for run in activity_runs
        ]
    lane = []
    for row in connection.execute(
        """SELECT report_date, status, synthesis_status, created_at, published_at FROM reports
               WHERE created_at >= ? OR published_at >= ? ORDER BY created_at""",
        (window_text, window_text),
    ):
        lane.append(
            {
                "start": row["created_at"],
                "end": row["published_at"] or format_utc_timestamp(now),
                "label": f"Report {row['report_date']}",
                "outcome": "ok" if row["status"] == "published" else row["status"],
                "running": row["status"] != "published",
                "inferred": True,
                "stages": [],
            }
        )
    return lane


# --------------------------------------------------------------------------------------------------------------------
# Events and day totals
# --------------------------------------------------------------------------------------------------------------------


def _events(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    *,
    now: datetime,
    window_text: str,
    activity: ActivityTree | None,
    collect_lane: list[dict[str, Any]],
    model_lane: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add(at: str, kind: str, severity: str, text: str, **extra: Any) -> None:
        events.append({"at": at, "kind": kind, "severity": severity, "text": text, **extra})

    floor = app_config.report.minimum_reading_priority
    for row in connection.execute(
        """SELECT review.review_id, review.paper_id, review.coverage, review.completed_at, paper.title,
                      (SELECT MAX(topic_score) FROM review_topic_assessments AS assessment
                       WHERE assessment.review_id = review.review_id
                         AND assessment.score_scale = 'priority_0_to_100') AS best_score
               FROM reviews AS review JOIN papers AS paper ON paper.paper_id = review.paper_id
               WHERE review.completed_at >= ?""",
        (window_text,),
    ):
        score = row["best_score"]
        recommended = score is not None and score >= floor
        add(
            row["completed_at"],
            "review",
            "good" if recommended else "info",
            f"Read ({row['coverage'].replace('_', ' ')})"
            + (f", priority {score:.0f}/100" if score is not None else "")
            + (" — recommended" if recommended else ""),
            title=row["title"],
            paper_id=row["paper_id"],
        )
    for row in connection.execute(
        """SELECT job.job_id, job.kind, job.updated_at, job.last_error_component, job.last_error, paper.title
               FROM jobs AS job LEFT JOIN papers AS paper ON paper.paper_id = job.paper_id
               WHERE job.status = 'failed' AND job.updated_at >= ?""",
        (window_text,),
    ):
        add(
            row["updated_at"],
            "job_failed",
            "bad",
            f"{JOB_KIND_LABELS.get(row['kind'], row['kind'])} failed for good "
            f"[{row['last_error_component']}]: {_shorten(row['last_error'])}",
            title=row["title"],
        )
    for call in model_lane:
        if call["outcome"] not in ("ok", "running"):
            add(
                call["end"],
                "model_rejected",
                "warn",
                f"{call['role']} answer rejected ({call['outcome'].replace('_', ' ')}) after {call['minutes']} min"
                + (" (repair attempt)" if call["repair"] else " — one repair follows"),
                title=call["title"],
                detail=call.get("error"),
            )
    for run in collect_lane:
        if run.get("running"):
            continue
        requests = ", ".join(f"{provider} {count}" for provider, count in sorted(run["requests"].items()))
        minutes = (parse_utc_timestamp(run["end"]) - parse_utc_timestamp(run["start"])).total_seconds() / 60
        add(
            run["end"],
            "collect",
            "bad" if run["outcome"] in ("failed", "interrupted") else ("warn" if run["failed_requests"] else "info"),
            f"Collect {'finished' if run['outcome'] in ('ok', 'partial') else run['outcome']} ({minutes:.0f} min): "
            f"{run['new_papers']} new paper(s), {run['promoted']} promoted, {run['full_texts']} full text(s)"
            + (f"; requests: {requests}" if requests else "")
            + (f"; {run['failed_requests']} failed request(s)" if run["failed_requests"] else ""),
            detail=run.get("note"),
        )
    for row in connection.execute(
        """SELECT report_date, status, synthesis_status, published_at, fallback_reason,
                      (SELECT COUNT(*) FROM report_items AS item WHERE item.report_date = report.report_date)
                          AS papers FROM reports AS report WHERE published_at >= ?""",
        (window_text,),
    ):
        add(
            row["published_at"],
            "report",
            "good" if row["synthesis_status"] != "fallback" else "warn",
            f"Report {row['report_date']} published: {row['papers']} paper(s), overview "
            f"{row['synthesis_status'] or 'none'}",
            detail=_shorten(row["fallback_reason"]),
        )
    for row in connection.execute(
        "SELECT service, last_failure_at, last_error FROM service_state WHERE last_failure_at >= ?", (window_text,)
    ):
        add(row["last_failure_at"], "outage", "bad", f"{row['service']} unavailable: {_shorten(row['last_error'])}")
    if activity is not None:
        for step in activity.steps_by_id.values():
            if step.step == "job" and step.outcome in ("retry", "released", "paused"):
                add(
                    format_utc_timestamp(step.finished_at),
                    "job_retry",
                    "warn",
                    f"Job {step.detail.get('job_id')} ended with '{step.outcome}'",
                    detail=step.note,
                )
            if step.is_interrupted and step.parent_activity_id is None:
                add(
                    format_utc_timestamp(step.latest_moment()),
                    "interrupted",
                    "bad",
                    f"{step.component} run (process {step.process_id}) stopped without finishing (killed or "
                    "crashed); its work resumes on the next run",
                )
            if step.step == "run" and step.component == "work":
                add(
                    format_utc_timestamp(step.started_at),
                    "worker",
                    "info",
                    f"Worker started (process {step.process_id})",
                )
                if step.finished_at:
                    add(
                        format_utc_timestamp(step.finished_at),
                        "worker",
                        "info",
                        f"Worker stopped after {step.detail.get('completed_jobs', 0)} completed job(s)",
                    )
    events.sort(key=lambda event: event["at"], reverse=True)
    return events[:MAXIMUM_EVENTS]


def _day_totals(
    connection: sqlite3.Connection, app_config: AppConfig, *, window_start: datetime, now: datetime
) -> list[dict[str, Any]]:
    """Totals per local day (report timezone) for every day the window touches, newest first."""
    report_timezone = app_config.report_timezone
    first_day, last_day = window_start.astimezone(report_timezone).date(), now.astimezone(report_timezone).date()
    days = []
    day = last_day
    while day >= first_day:
        day_start = datetime.combine(day, datetime.min.time(), tzinfo=report_timezone)
        start_text = format_utc_timestamp(day_start)
        end_text = format_utc_timestamp(day_start + timedelta(days=1))
        call_counts = defaultdict(int)
        model_seconds = 0.0
        for row in connection.execute(
            "SELECT outcome, COUNT(*) AS call_count, SUM(duration_seconds) AS seconds FROM model_calls "
            "WHERE started_at >= ? AND started_at < ? GROUP BY outcome",
            (start_text, end_text),
        ):
            call_counts[row["outcome"]] += row["call_count"]
            model_seconds += row["seconds"] or 0
        days.append(
            {
                "date": day.isoformat(),
                "readings": connection.execute(
                    "SELECT COUNT(*) FROM reviews WHERE completed_at >= ? AND completed_at < ?", (start_text, end_text)
                ).fetchone()[0],
                "model_hours": round(model_seconds / 3600, 1),
                "model_calls": sum(call_counts.values()),
                "rejected_calls": sum(count for outcome, count in call_counts.items() if outcome != "ok"),
                "new_papers": connection.execute(
                    "SELECT COUNT(*) FROM papers WHERE discovered_at >= ? AND discovered_at < ?", (start_text, end_text)
                ).fetchone()[0],
                "promoted": connection.execute(
                    "SELECT COUNT(*) FROM paper_promotions WHERE promoted_at >= ? AND promoted_at < ?",
                    (start_text, end_text),
                ).fetchone()[0],
                "requests": connection.execute(
                    "SELECT COUNT(*) FROM outbound_requests WHERE started_at >= ? AND started_at < ?",
                    (start_text, end_text),
                ).fetchone()[0],
            }
        )
        day -= timedelta(days=1)
    return days


def _shorten(text: str | None) -> str | None:
    if not text:
        return None
    return text if len(text) <= MAXIMUM_ERROR_CHARACTERS else text[: MAXIMUM_ERROR_CHARACTERS - 1] + "…"
