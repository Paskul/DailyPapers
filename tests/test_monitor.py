"""The monitor: the activity log the pipeline writes, the read-only snapshot (exact and inferred), the schedule it
reads from systemd unit files, and the web server's safeguards."""

from __future__ import annotations

import http.client
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from conftest import RECORDED_PAPER_FIXTURE, build_in_process_app

from paperwatch.activity_log import finish_activity, recorded_activity, start_activity
from paperwatch.clock import format_utc_timestamp, utc_now
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.work_command import run_single_work_cycle
from paperwatch.config import load_app_config
from paperwatch.deployment.daily_schedule import SCOPE_SYSTEM, render_daily_units
from paperwatch.deployment.one_day_pilot import plan_one_day_pilot, render_pilot_units
from paperwatch.errors import ConfigurationError
from paperwatch.monitor.activity_rows import load_activity_tree
from paperwatch.monitor.database_access import open_read_only_database
from paperwatch.monitor.monitor_server import SnapshotProvider, create_monitor_server, validate_listen_address
from paperwatch.monitor.schedule_state import (
    ScheduleState,
    expand_calendar_value,
    parse_systemctl_show_output,
    read_schedule_state,
)
from paperwatch.monitor.snapshot import build_monitor_snapshot
from paperwatch.network.egress_guard import EgressBlocked, build_monitor_policy, temporarily_armed_egress_policy

NO_SCHEDULE = ScheduleState(schedule_source="none", moments=())


class Arguments:
    """The parsed CLI arguments the command functions read."""

    def __init__(self, **values):
        self.__dict__.update(values)


def run_one_full_day(app) -> None:
    run_collect_command(app, Arguments(fixture_paths=[RECORDED_PAPER_FIXTURE], skip_searches=False))
    report = run_single_work_cycle(app, probe_paused_model_now=True)
    assert report.outcome.value == "job_completed", report.message


def snapshot_of(app, *, schedule: ScheduleState = NO_SCHEDULE, hours: float = 24.0) -> dict:
    connection = open_read_only_database(app.config.database_path)
    try:
        return build_monitor_snapshot(connection, app.config, now=utc_now(), history_hours=hours, schedule=schedule)
    finally:
        connection.close()


# --------------------------------------------------------------------------------------------------------------------
# The activity log written by collect, work, and report
# --------------------------------------------------------------------------------------------------------------------


def test_collect_work_and_report_record_nested_finished_steps(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    run_one_full_day(app)
    rows = app.connection.execute("SELECT * FROM activity_log ORDER BY started_at, rowid").fetchall()
    assert rows and all(row["finished_at"] and row["outcome"] == "ok" for row in rows)
    by_id = {row["activity_id"]: row for row in rows}

    collect_run = next(row for row in rows if row["component"] == "collect" and row["step"] == "run")
    collect_stages = [row["step"] for row in rows if row["parent_activity_id"] == collect_run["activity_id"]]
    assert collect_stages == ["seeds", "fixtures", "screening", "follow_up", "source_text"]

    model_calls = [row for row in rows if row["step"] == "model_call" and row["component"] == "work"]
    assert {json.loads(row["detail_json"])["role"] for row in model_calls} == {"reviewer", "topic_assessor"}
    for model_call in model_calls:  # job -> reading stage -> model call
        stage = by_id[model_call["parent_activity_id"]]
        assert stage["step"] == "reading_stage"
        assert by_id[stage["parent_activity_id"]]["step"] == "job"
    job_row = next(row for row in rows if row["step"] == "job")
    assert json.loads(job_row["detail_json"])["kind"] == "review_paper"


def test_report_run_records_its_stages(tmp_path):
    app = build_in_process_app(
        tmp_path / "home",
        config_replacements={
            'timezone = "America/New_York"': 'timezone = "UTC"',
            'publish_time = "07:00"': 'publish_time = "23:59"',
        },
    )
    run_one_full_day(app)
    if utc_now().hour == 23 and utc_now().minute >= 58:
        pytest.skip("too close to the test's report cutoff")
    run_report_command(app, Arguments(report_date=None, publish_early=True))
    report_run = app.connection.execute(
        "SELECT * FROM activity_log WHERE component = 'report' AND step = 'run'"
    ).fetchone()
    stages = [
        row["step"]
        for row in app.connection.execute(
            "SELECT step FROM activity_log WHERE parent_activity_id = ? ORDER BY started_at, rowid",
            (report_run["activity_id"],),
        )
    ]
    assert stages == ["freeze", "synthesize", "publish"]
    assert report_run["outcome"] == "ok"


def test_a_failing_step_records_failed_and_reraises_and_a_broken_log_never_fails_the_step(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    with (
        pytest.raises(ValueError),
        recorded_activity(app.connection, "run", component="collect"),
        recorded_activity(app.connection, "screening"),
    ):
        raise ValueError("embedding exploded https://example.org/x?api_key=SECRET")
    rows = {row["step"]: row for row in app.connection.execute("SELECT * FROM activity_log")}
    assert rows["screening"]["outcome"] == "failed" and rows["run"]["outcome"] == "failed"
    assert "SECRET" not in rows["screening"]["note"]  # notes are redacted like service logs

    read_only_connection = open_read_only_database(app.config.database_path)
    ran = []
    with recorded_activity(read_only_connection, "run", component="collect") as handle:  # cannot write the log
        ran.append(True)
    assert ran == [True] and handle.activity_id is None
    assert start_activity(read_only_connection, "anything") is None
    finish_activity(read_only_connection, None, "ok")  # no error either


def test_unfinished_steps_of_dead_processes_are_interrupted_and_live_ones_running(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    live_run = start_activity(app.connection, "run", component="work")
    app.connection.execute(
        """INSERT INTO activity_log (activity_id, component, step, detail_json, process_id, boot_id, started_at)
           VALUES ('a_dead', 'collect', 'run', '{}', ?, 'an-older-boot', ?)""",
        (os.getpid(), format_utc_timestamp(utc_now() - timedelta(minutes=5))),
    )
    tree = load_activity_tree(app.connection, utc_now() - timedelta(hours=1), current_boot_id="this-boot")
    assert tree.steps_by_id["a_dead"].is_interrupted  # same pid, but recorded in another boot
    del live_run
    tree = load_activity_tree(app.connection, utc_now() - timedelta(hours=1))
    assert [run.component for run in tree.running_roots("work")] == ["work"]


# --------------------------------------------------------------------------------------------------------------------
# The snapshot: exact (activity log) and inferred (older databases)
# --------------------------------------------------------------------------------------------------------------------


def _claim_review_job(app):
    run_collect_command(app, Arguments(fixture_paths=[RECORDED_PAPER_FIXTURE], skip_searches=False))
    job_id = app.connection.execute("SELECT job_id FROM jobs WHERE kind = 'review_paper'").fetchone()[0]
    app.connection.execute(
        "UPDATE jobs SET status = 'running', attempts = 1, lease_token = 't', lease_expires_at = ?, due_at = ? "
        "WHERE job_id = ?",
        (
            format_utc_timestamp(utc_now() + timedelta(minutes=5)),
            format_utc_timestamp(utc_now() - timedelta(minutes=1)),
            job_id,
        ),
    )
    return job_id


def test_snapshot_shows_the_exact_current_step_from_the_activity_log(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    job_id = _claim_review_job(app)
    with (
        recorded_activity(app.connection, "run", {"mode": "loop"}, component="work"),
        recorded_activity(
            app.connection, "job", {"job_id": job_id, "kind": "review_paper", "attempt": 1, "max_attempts": 3}
        ),
        recorded_activity(
            app.connection,
            "reading_stage",
            {"stage": "section", "position": 2, "sections_to_read": 5, "section_count": 5},
        ),
    ):
        call_id = start_activity(
            app.connection,
            "model_call",
            {"role": "section_reader", "repair": True, "job_id": job_id, "timeout_seconds": 1800},
        )
        snapshot = snapshot_of(app)
        finish_activity(app.connection, call_id, "ok")
    worker = snapshot["now"]["worker"]
    assert worker["state"] == "working" and worker["source"] == "activity"
    [job] = worker["jobs"]
    paper_title = app.connection.execute("SELECT title FROM papers WHERE paper_id = ?", (job["paper_id"],)).fetchone()[
        0
    ]
    assert job["title"] == paper_title
    labels = [step["label"] for step in job["steps"]]
    assert labels == ["Reading — attempt 1/3", "Section 2 of 5", "Model call: section_reader (repair)"]
    assert job["progress"] == {"done": 1, "total": 7, "label": "sections, then synthesis and topic assessment"}
    assert job["steps"][2]["deadline"]
    running_calls = [call for call in snapshot["history"]["lanes"]["model"] if call["running"]]
    assert running_calls and running_calls[0]["role"] == "section_reader" and not running_calls[0]["inferred"]
    assert snapshot["meta"]["activity_log"] is True


def test_snapshot_infers_the_current_step_on_a_database_without_the_activity_log(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    job_id = _claim_review_job(app)
    earlier = utc_now() - timedelta(minutes=30)
    app.connection.execute(
        """INSERT INTO model_calls (call_id, role_name, prompt_version, job_id, started_at, duration_seconds, outcome)
           VALUES ('c_1', 'section_reader', 'section-reader-v1', ?, ?, 600, 'invalid_output')""",
        (job_id, format_utc_timestamp(earlier)),
    )
    app.connection.executescript("DROP TABLE activity_log; PRAGMA user_version = 9;")  # as the deployed pilot
    snapshot = snapshot_of(app)
    assert snapshot["meta"]["activity_log"] is False and snapshot["meta"]["schema_version"] == 9
    [job] = snapshot["now"]["worker"]["jobs"]
    assert job["source"] == "inferred"
    assert "repair after a rejected answer (invalid output)" in job["steps"][0]["label"]
    assert job["steps"][0]["since"] == format_utc_timestamp(earlier + timedelta(seconds=600))
    in_progress = [call for call in snapshot["history"]["lanes"]["model"] if call["running"]]
    assert in_progress[0]["inferred"] and in_progress[0]["start"] == job["steps"][0]["since"]
    assert [event["kind"] for event in snapshot["history"]["events"]].count("model_rejected") == 1
    assert snapshot["history"]["lanes"]["collect"][0]["inferred"] is True


def test_snapshot_history_plan_and_budgets_after_a_day(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    run_one_full_day(app)
    snapshot = snapshot_of(app)
    assert snapshot["now"]["worker"]["state"] in ("not_running", "idle")
    assert {"review", "collect"} <= {event["kind"] for event in snapshot["history"]["events"]}
    assert snapshot["history"]["days"][0]["readings"] == 1
    assert {budget["label"] for budget in snapshot["plan"]["budgets"]} >= {"Readings promoted", "New candidates"}
    # Nothing installed: the configured times are shown, marked as not installed.
    moments = snapshot["plan"]["moments"]
    assert moments and not any(moment["installed"] for moment in moments)
    assert snapshot["plan"]["schedule_source"] == "none"
    assert "question" not in json.dumps(snapshot["meta"])  # topic ids only, never the private questions


def test_building_a_snapshot_never_writes(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    run_one_full_day(app)
    app.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database_bytes_before = app.config.database_path.read_bytes()
    snapshot_of(app)
    assert app.config.database_path.read_bytes() == database_bytes_before
    read_only = open_read_only_database(app.config.database_path)
    with pytest.raises(sqlite3.OperationalError):
        read_only.execute("DELETE FROM jobs")


# --------------------------------------------------------------------------------------------------------------------
# Schedule from unit files and systemctl
# --------------------------------------------------------------------------------------------------------------------


def test_schedule_reads_the_pilot_and_the_daily_timers(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    installed_at = datetime(2026, 11, 20, 14, 20, tzinfo=UTC)
    pilot_dir = tmp_path / "pilot-units"
    pilot_dir.mkdir()
    for name, text in render_pilot_units(plan_one_day_pilot(app.config, now=installed_at, collector_uid=999)).items():
        (pilot_dir / name).write_text(text)
    pilot = read_schedule_state(
        window_start=installed_at,
        window_end=installed_at + timedelta(days=2),
        unit_dirs=(pilot_dir,),
        stamp_dirs=(),
        query_systemd=False,
    )
    assert pilot.schedule_source == "pilot"
    assert [moment.kind for moment in pilot.moments][-2:] == ["report", "pilot_finish"]
    assert {moment.kind for moment in pilot.moments} == {"collect", "report", "pilot_finish"}

    daily_dir = tmp_path / "daily-units"
    daily_dir.mkdir()
    for name, text in render_daily_units(
        app.config, scope=SCOPE_SYSTEM, paperwatch_executable=Path("/x/paperwatch"), collector_uid=999
    ).items():
        (daily_dir / name).write_text(text)
    daily = read_schedule_state(
        window_start=installed_at,
        window_end=installed_at + timedelta(days=2),
        unit_dirs=(daily_dir,),
        stamp_dirs=(),
        query_systemd=False,
    )
    assert daily.schedule_source == "daily"
    collects = [moment for moment in daily.moments if moment.kind == "collect"]
    reports = [moment for moment in daily.moments if moment.kind == "report"]
    assert len(collects) == 8 and len(reports) == 2  # 4 times a day, 2 days; one report a day


def test_calendar_expressions_and_systemctl_output_are_parsed():
    start = datetime(2026, 11, 20, 12, tzinfo=UTC)
    assert expand_calendar_value("*-*-* 06:30:00 America/New_York", start, start + timedelta(days=1)) == [
        datetime(2026, 11, 21, 6, 30, tzinfo=__import__("zoneinfo").ZoneInfo("America/New_York"))
    ]
    assert expand_calendar_value("Mon *-*-* 06:30:00", start, start + timedelta(days=7)) == []  # not ours: ignored
    states = parse_systemctl_show_output(
        "Id=paperwatch-worker.service\nLoadState=loaded\nActiveState=active\nSubState=running\nResult=success\n"
        "ExecMainPID=4242\nUnitFileState=enabled\n\nId=paperwatch-collect.service\nLoadState=not-found\n"
        "ActiveState=inactive\nExecMainPID=0\n",
        "system",
    )
    assert (
        states["paperwatch-worker.service"].is_running and states["paperwatch-worker.service"].main_process_id == 4242
    )
    assert not states["paperwatch-collect.service"].is_installed


# --------------------------------------------------------------------------------------------------------------------
# Web server safeguards
# --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def running_monitor(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    run_one_full_day(app)
    provider = SnapshotProvider(
        load_app_config(app.config.home_dir), default_history_hours=24, read_schedule=lambda **_window: NO_SCHEDULE
    )
    server = create_monitor_server(app.config, port=0, snapshot_provider=provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _request(server, method: str, path: str, host: str | None = None) -> http.client.HTTPResponse:
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    headers = {"Host": host} if host else {}
    connection.request(method, path, headers=headers)
    return connection.getresponse()


def test_server_serves_the_page_and_the_snapshot_with_strict_headers(running_monitor):
    page = _request(running_monitor, "GET", "/")
    assert page.status == 200 and b"/monitor.js" in page.read()
    assert "script-src 'self'" in page.getheader("Content-Security-Policy")
    assert page.getheader("X-Frame-Options") == "DENY"
    for asset in ("/monitor.js", "/monitor.css"):
        assert _request(running_monitor, "GET", asset).status == 200
    response = _request(running_monitor, "GET", "/api/snapshot?hours=12")
    snapshot = json.loads(response.read())
    assert response.status == 200 and snapshot["history"]["days"][0]["readings"] == 1


def test_server_refuses_foreign_hosts_writes_and_unknown_paths(running_monitor):
    assert _request(running_monitor, "GET", "/api/snapshot", host="attacker.example").status == 421  # DNS rebinding
    assert _request(running_monitor, "GET", "/", host=f"localhost:{running_monitor.server_port}").status == 200
    assert _request(running_monitor, "POST", "/api/snapshot").status == 405
    assert _request(running_monitor, "GET", "/../config.toml").status == 404


@pytest.mark.parametrize("address", ["0.0.0.0", "192.168.1.20", "8.8.8.8", "::", "not-an-ip"])
def test_server_listens_only_on_loopback_or_tailscale(address):
    with pytest.raises(ConfigurationError):
        validate_listen_address(address)


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "100.74.19.96", "fd7a:115c:a1e0::9f01:13c8"])
def test_loopback_and_tailscale_addresses_are_accepted(address):
    validate_listen_address(address)


def test_monitor_policy_connects_nowhere_and_runs_only_systemctl():
    import socket
    import subprocess

    with temporarily_armed_egress_policy(build_monitor_policy()):
        with pytest.raises(EgressBlocked):
            socket.create_connection(("127.0.0.1", 8080), timeout=1)
        with pytest.raises(EgressBlocked):
            subprocess.run(["/usr/bin/curl", "https://example.org"], check=False)
        if Path("/usr/bin/systemctl").exists():
            subprocess.run(["/usr/bin/systemctl", "--version"], capture_output=True, check=False)


def test_monitor_json_command_runs_against_a_home(initialized_home):
    initialized_home.use_fake_model()
    result = initialized_home.run_cli("monitor", "--json", "--history-hours", "6")
    assert result.exit_code == 0, result.stderr
    assert json.loads(result.stdout)["meta"]["schema_version"] == 10
