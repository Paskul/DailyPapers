"""The monitor's "now" panel: what the worker, the collector, the report, and the model server are doing.

Each part prefers the activity log (exact steps, schema v10) and falls back to what older databases already record:
  worker     a running job with a fresh lease, the reading stages it has saved, and the end of the last model call
             (one inference at a time, so the current call started when the previous one ended);
  collector  systemd's state of the collect service, else outbound requests in the last few minutes;
  report     the report row and the inference reservation the writer takes;
  model      the outage bookkeeping (service_state) and the model-call log.
Every part has a `source` ("activity", "inferred", or "systemd") so the page can say how sure it is.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp
from paperwatch.config import AppConfig
from paperwatch.deployment.one_day_pilot import COLLECT_SERVICE as PILOT_COLLECT_SERVICE
from paperwatch.deployment.one_day_pilot import REPORT_SERVICE as PILOT_REPORT_SERVICE
from paperwatch.deployment.one_day_pilot import WORKER_SERVICE as PILOT_WORKER_SERVICE
from paperwatch.deployment.unit_names import DAILY_COLLECT_SERVICE, DAILY_REPORT_SERVICE, DAILY_WORKER_SERVICE
from paperwatch.dispatch_order import PRIORITY_CLASS_LABELS, JobPriorityClass
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME
from paperwatch.monitor.activity_rows import ActivityStep, ActivityTree
from paperwatch.monitor.database_access import load_paper_titles
from paperwatch.monitor.schedule_state import ScheduleState, ServiceState

RECENT_REQUEST_WINDOW = timedelta(minutes=3)  # requests this recent mean a collect is probably running
JOB_KIND_LABELS = {
    "review_paper": "Reading",
    "select_references": "Choosing references",
    "compare_reviews": "Comparing two reviews",
}
COLLECT_STAGE_LABELS = {
    "seeds": "Syncing seed papers (Zotero / CSL JSON)",
    "fixtures": "Importing recorded papers",
    "current_releases": "Polling arXiv feeds and recent OpenAlex",
    "historical_search": "Historical search",
    "gap_searches": "Gap searches requested by the research role",
    "references": "Fetching references",
    "screening": "Screening and promoting candidates (CPU)",
    "follow_up": "Queueing follow-up work",
    "source_text": "Downloading and extracting full text",
}
REPORT_STAGE_LABELS = {
    "freeze": "Freezing the report's selection",
    "synthesize": "Writing the overview",
    "publish": "Publishing the report file",
}


def describe_now(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    *,
    now: datetime,
    activity: ActivityTree | None,
    schedule: ScheduleState,
    next_report_cutoff: datetime,
    last_collect_run: dict[str, Any] | None,
) -> dict[str, Any]:
    model_part = describe_model_server(connection, now=now)
    worker_part = describe_worker(
        connection, now=now, activity=activity, schedule=schedule, model_paused=model_part["state"] == "paused"
    )
    model_part["state"] = (
        "busy" if worker_part["state"] == "working" or _report_is_writing(activity) else model_part["state"]
    )
    return {
        "worker": worker_part,
        "collector": describe_collector(
            connection, now=now, activity=activity, schedule=schedule, last_collect_run=last_collect_run
        ),
        "report": describe_report(
            connection, app_config, now=now, activity=activity, schedule=schedule, next_report_cutoff=next_report_cutoff
        ),
        "model": model_part,
    }


# --------------------------------------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------------------------------------


def describe_worker(
    connection: sqlite3.Connection,
    *,
    now: datetime,
    activity: ActivityTree | None,
    schedule: ScheduleState,
    model_paused: bool,
) -> dict[str, Any]:
    now_text = format_utc_timestamp(now)
    running_job_rows = connection.execute(
        """SELECT job_id, kind, paper_id, priority, attempts, max_attempts, input_version, updated_at
           FROM jobs WHERE status = 'running' AND lease_expires_at > ? ORDER BY updated_at""",
        (now_text,),
    ).fetchall()
    titles = load_paper_titles(connection, [row["paper_id"] for row in running_job_rows])
    service_state = schedule.service(PILOT_WORKER_SERVICE, DAILY_WORKER_SERVICE)
    worker_runs = activity.running_roots("work") if activity else []
    jobs_in_activity: dict[str, list[ActivityStep]] = {}
    for worker_run in worker_runs:
        chain = worker_run.running_chain()
        job_steps = [step for step in chain if step.step == "job"]
        if job_steps:
            jobs_in_activity[job_steps[0].detail.get("job_id", "")] = chain
    # A `work --once` job has no run row: its job row is a root of its own.
    for job_root in activity.roots_of("work", "job") if activity else []:
        if job_root.is_running:
            jobs_in_activity.setdefault(job_root.detail.get("job_id", ""), job_root.running_chain())

    current_jobs = []
    for job_row in running_job_rows:
        job_chain = jobs_in_activity.get(job_row["job_id"])
        current_jobs.append(
            _describe_job_from_activity(job_row, job_chain, titles, now)
            if job_chain
            else _describe_job_inferred(connection, job_row, titles, now)
        )

    if current_jobs:
        state = "working"
    elif model_paused:
        state = "paused"
    elif worker_runs or (service_state is not None and service_state.is_running):
        state = "idle"
    elif service_state is not None:
        state = "stopped"
    else:
        state = "not_running"
    next_due_row = connection.execute("SELECT MIN(due_at) AS next_due FROM jobs WHERE status = 'queued'").fetchone()
    return {
        "state": state,
        "jobs": current_jobs,
        "processes": [
            {"process_id": run.process_id, "since": format_utc_timestamp(run.started_at)} for run in worker_runs
        ],
        "service": _describe_service(service_state),
        "next_due_at": next_due_row["next_due"],
        "source": "activity" if jobs_in_activity or worker_runs else ("inferred" if current_jobs else "systemd"),
    }


def _describe_job_from_activity(
    job_row: sqlite3.Row, chain: list[ActivityStep], titles: dict[str, str], now: datetime
) -> dict[str, Any]:
    steps = []
    progress = None
    for activity_step in chain:
        if activity_step.step == "run":
            continue
        label = describe_activity_step(activity_step)
        step_entry = {"label": label, "since": format_utc_timestamp(activity_step.started_at)}
        if activity_step.step == "model_call" and activity_step.detail.get("timeout_seconds"):
            step_entry["deadline"] = format_utc_timestamp(
                activity_step.started_at + timedelta(seconds=activity_step.detail["timeout_seconds"])
            )
        if activity_step.step == "reading_stage" and activity_step.detail.get("stage") == "section":
            progress = {
                "done": activity_step.detail.get("position", 1) - 1,
                "total": activity_step.detail.get("sections_to_read", 0) + 2,  # + synthesis + assessment
                "label": "sections, then synthesis and topic assessment",
            }
        elif activity_step.step == "reading_stage" and activity_step.detail.get("stage") == "synthesis":
            notes = activity_step.detail.get("section_notes", 0)
            progress = {"done": notes, "total": notes + 2, "label": "synthesis, then topic assessment"}
        elif activity_step.step == "reading_stage" and activity_step.detail.get("stage") == "topic_assessment":
            progress = {"done": 1, "total": 2, "label": "topic assessment (last step)"}
        steps.append(step_entry)
    job_step = next((step for step in chain if step.step == "job"), None)
    return _job_summary(
        job_row,
        titles,
        steps=steps,
        progress=progress,
        source="activity",
        since=format_utc_timestamp(job_step.started_at) if job_step else None,
    )


def _describe_job_inferred(
    connection: sqlite3.Connection, job_row: sqlite3.Row, titles: dict[str, str], now: datetime
) -> dict[str, Any]:
    """Without the activity log: saved stages tell how far the reading got; the last model call's end tells
    when the current call began (one inference at a time)."""
    saved_stage_keys = (
        [
            row["stage_key"]
            for row in connection.execute(
                "SELECT stage_key FROM review_stage_results WHERE paper_id = ? AND review_input_key = ?",
                (job_row["paper_id"], job_row["input_version"]),
            )
        ]
        if job_row["paper_id"]
        else []
    )
    saved_section_count = sum(key.startswith("section:") for key in saved_stage_keys)
    synthesis_saved = "synthesis" in saved_stage_keys
    job_calls = connection.execute(
        """SELECT role_name, outcome, started_at, duration_seconds FROM model_calls WHERE job_id = ?
           ORDER BY started_at""",
        (job_row["job_id"],),
    ).fetchall()
    last_call_overall = connection.execute(
        "SELECT started_at, duration_seconds FROM model_calls ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    call_since = None
    if last_call_overall is not None:
        call_since = parse_utc_timestamp(last_call_overall["started_at"]) + timedelta(
            seconds=last_call_overall["duration_seconds"]
        )
    if job_row["kind"] != "review_paper":
        stage_label = "Model call in progress"
    elif synthesis_saved:
        stage_label = "Topic assessment (the synthesis is saved)"
    elif saved_section_count:
        stage_label = f"{saved_section_count} section note(s) saved; now the next section or the synthesis"
    elif str(job_row["input_version"]).startswith("abstract_only"):
        stage_label = "Review from the abstract"
    else:
        stage_label = "First reading stage"
    last_job_call = job_calls[-1] if job_calls else None
    if last_job_call is not None and last_job_call["outcome"] != "ok":
        stage_label += f" — repair after a rejected answer ({last_job_call['outcome'].replace('_', ' ')})"
    steps = [{"label": stage_label, "since": format_utc_timestamp(call_since) if call_since else None}]
    ok_calls = sum(call["outcome"] == "ok" for call in job_calls)
    return _job_summary(
        job_row,
        titles,
        steps=steps,
        source="inferred",
        progress={
            "done": saved_section_count + synthesis_saved,
            "total": None,
            "label": f"{len(job_calls)} model call(s) so far, {ok_calls} accepted",
        },
        since=job_calls[0]["started_at"] if job_calls else None,
    )


def _job_summary(
    job_row: sqlite3.Row,
    titles: dict[str, str],
    *,
    steps: list[dict[str, Any]],
    progress: dict[str, Any] | None,
    source: str,
    since: str | None,
) -> dict[str, Any]:
    return {
        "job_id": job_row["job_id"],
        "kind": job_row["kind"],
        "kind_label": JOB_KIND_LABELS.get(job_row["kind"], job_row["kind"]),
        "paper_id": job_row["paper_id"],
        "title": titles.get(job_row["paper_id"]),
        "class_label": priority_class_label(job_row["priority"]),
        "attempt": job_row["attempts"],
        "max_attempts": job_row["max_attempts"],
        "since": since,
        "steps": steps,
        "progress": progress,
        "source": source,
    }


def describe_activity_step(activity_step: ActivityStep) -> str:
    """One line for a step, e.g. 'Section 3 of 7' or 'Model call: section_reader (repair)'."""
    detail = activity_step.detail
    if activity_step.step == "job":
        return (
            f"{JOB_KIND_LABELS.get(str(detail.get('kind')), detail.get('kind', 'job'))} — attempt "
            f"{detail.get('attempt', '?')}/{detail.get('max_attempts', '?')}"
        )
    if activity_step.step == "reading_stage":
        stage = detail.get("stage")
        if stage == "section":
            label = f"Section {detail.get('position', '?')} of {detail.get('sections_to_read', '?')}"
            if detail.get("section_count") and detail.get("section_count") != detail.get("sections_to_read"):
                label += f" (paper has {detail['section_count']} sections)"
            return label
        if stage == "synthesis":
            notes = detail.get("section_notes", 0)
            return (
                f"Synthesis from {notes} section note(s)"
                if notes
                else f"Review ({str(detail.get('coverage', 'text')).replace('_', ' ')})"
            )
        if stage == "topic_assessment":
            return f"Topic assessment ({detail.get('topic_count', '?')} topic(s))"
        return f"Reading stage {stage}"
    if activity_step.step == "model_call":
        return f"Model call: {detail.get('role', '?')}" + (" (repair)" if detail.get("repair") else "")
    if activity_step.component == "collect":
        return COLLECT_STAGE_LABELS.get(activity_step.step, activity_step.step.replace("_", " "))
    if activity_step.component == "report":
        return REPORT_STAGE_LABELS.get(activity_step.step, activity_step.step.replace("_", " "))
    return activity_step.step.replace("_", " ")


# --------------------------------------------------------------------------------------------------------------------
# Collector, report, model server
# --------------------------------------------------------------------------------------------------------------------


def describe_collector(
    connection: sqlite3.Connection,
    *,
    now: datetime,
    activity: ActivityTree | None,
    schedule: ScheduleState,
    last_collect_run: dict[str, Any] | None,
) -> dict[str, Any]:
    service_state = schedule.service(PILOT_COLLECT_SERVICE, DAILY_COLLECT_SERVICE)
    running_runs = activity.running_roots("collect") if activity else []
    if running_runs:
        chain = running_runs[-1].running_chain()
        return {
            "state": "running",
            "since": format_utc_timestamp(chain[0].started_at),
            "source": "activity",
            "steps": [
                {"label": describe_activity_step(step), "since": format_utc_timestamp(step.started_at)}
                for step in chain[1:]
            ],
            "service": _describe_service(service_state),
            "last_run": last_collect_run,
        }
    recent_request = connection.execute(
        "SELECT MAX(started_at) AS latest FROM outbound_requests WHERE started_at >= ?",
        (format_utc_timestamp(now - RECENT_REQUEST_WINDOW),),
    ).fetchone()["latest"]
    if service_state is not None and service_state.active_state == "activating":
        state, source = "running", "systemd"
    elif recent_request:
        state, source = "running", "inferred"
    else:
        state, source = "idle", "activity" if activity is not None else "inferred"
    return {
        "state": state,
        "since": service_state.main_started_at if service_state and state == "running" else None,
        "source": source,
        "steps": [{"label": f"Requests in progress (latest {recent_request})", "since": None}]
        if recent_request and state == "running"
        else [],
        "service": _describe_service(service_state),
        "last_run": last_collect_run,
    }


def describe_report(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    *,
    now: datetime,
    activity: ActivityTree | None,
    schedule: ScheduleState,
    next_report_cutoff: datetime,
) -> dict[str, Any]:
    latest_report = connection.execute(
        """SELECT report_date, status, synthesis_status, published_at, output_path,
                  (SELECT COUNT(*) FROM report_items AS item WHERE item.report_date = report.report_date) AS papers
           FROM reports AS report ORDER BY report_date DESC LIMIT 1"""
    ).fetchone()
    reservation = connection.execute(
        "SELECT reservation_name, reserved_until FROM inference_reservations WHERE reserved_until > ? LIMIT 1",
        (format_utc_timestamp(now),),
    ).fetchone()
    running_runs = activity.running_roots("report") if activity else []
    service_state = schedule.service(PILOT_REPORT_SERVICE, DAILY_REPORT_SERVICE)
    steps = []
    if running_runs:
        steps = [
            {"label": describe_activity_step(step), "since": format_utc_timestamp(step.started_at)}
            for step in running_runs[-1].running_chain()[1:]
        ]
    if running_runs or (service_state is not None and service_state.active_state == "activating"):
        state = "publishing"
    elif latest_report is not None and latest_report["status"] != "published":
        state = "unfinished"  # a crash left it writing/pending; the next `report` run resumes it
    else:
        state = "waiting"
    return {
        "state": state,
        "steps": steps,
        "next_cutoff": format_utc_timestamp(next_report_cutoff),
        "next_report_date": next_report_cutoff.astimezone(app_config.report_timezone).date().isoformat(),
        "writer_reservation_until": reservation["reserved_until"] if reservation else None,
        "latest": dict(latest_report) if latest_report else None,
        "service": _describe_service(service_state),
    }


def describe_model_server(connection: sqlite3.Connection, *, now: datetime) -> dict[str, Any]:
    health = connection.execute("SELECT * FROM service_state WHERE service = ?", (MODEL_SERVICE_NAME,)).fetchone()
    paused_until = health["paused_until"] if health else None
    is_paused = bool(paused_until and paused_until > format_utc_timestamp(now))
    last_ok = connection.execute(
        "SELECT started_at, duration_seconds, role_name FROM model_calls WHERE outcome = 'ok' "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    hour_rows = connection.execute(
        "SELECT outcome, COUNT(*) AS call_count FROM model_calls WHERE started_at >= ? GROUP BY outcome",
        (format_utc_timestamp(now - timedelta(hours=24)),),
    ).fetchall()
    return {
        "state": "paused" if is_paused else "idle",
        "paused_until": paused_until if is_paused else None,
        "consecutive_failures": health["consecutive_failures"] if health else 0,
        "last_error": health["last_error"] if health and is_paused else None,
        "last_ok_call": {
            "at": last_ok["started_at"],
            "role": last_ok["role_name"],
            "minutes": round(last_ok["duration_seconds"] / 60, 1),
        }
        if last_ok
        else None,
        "calls_last_24h": {row["outcome"]: row["call_count"] for row in hour_rows},
    }


def _report_is_writing(activity: ActivityTree | None) -> bool:
    return bool(
        activity
        and any(step.step == "synthesize" for run in activity.running_roots("report") for step in run.running_chain())
    )


def _describe_service(service_state: ServiceState | None) -> dict[str, Any] | None:
    if service_state is None:
        return None
    return {
        "unit": service_state.unit_name,
        "scope": service_state.scope,
        "active_state": service_state.active_state,
        "sub_state": service_state.sub_state,
        "since": service_state.main_started_at,
    }


def priority_class_label(priority: int) -> str:
    try:
        return PRIORITY_CLASS_LABELS[JobPriorityClass(priority)]
    except ValueError:
        return f"class {priority}"
