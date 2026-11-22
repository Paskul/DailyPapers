"""Plan part 10, owner decision 2026-11-20: the pilot's timers run ONCE (one day), never again.

The rendered unit files are checked for every layer that makes a second run impossible: absolute calendar times
only (and systemd itself computes no second elapse), no relative or boot triggers, no service enabled for boot, an
expiry condition in every service that really skips after the deadline, and a finish step that disables and deletes
every pilot unit. Also checked: the worker/report sandbox, the accounts, and the report command.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from paperwatch.config import load_app_config
from paperwatch.deployment.one_day_pilot import (
    ALL_PILOT_UNIT_NAMES,
    COLLECT_SERVICE,
    COLLECT_TIMER,
    FINISH_SERVICE,
    FINISH_TIMER,
    PILOT_TIMER_NAMES,
    REPORT_SERVICE,
    REPORT_TIMER,
    WORKER_SERVICE,
    describe_pilot_plan,
    plan_one_day_pilot,
    render_pilot_units,
)

NEW_YORK = ZoneInfo("America/New_York")
INSTALLED_AT = datetime(2026, 11, 20, 14, 20, tzinfo=NEW_YORK)
RELATIVE_OR_BOOT_TRIGGERS = (
    "OnBootSec",
    "OnStartupSec",
    "OnActiveSec",
    "OnUnitActiveSec",
    "OnUnitInactiveSec",
    "OnClockChange",
    "OnTimezoneChange",
)
ABSOLUTE_CALENDAR_TIME = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} America/New_York$")


def parse_unit(unit_text: str) -> dict[str, list[tuple[str, str]]]:
    """{section: [(key, value), ...]} keeping repeated keys in order."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current_section = None
    for line in unit_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("["):
            current_section = line.strip("[]")
            sections[current_section] = []
            continue
        key, _, value = line.partition("=")
        sections[current_section].append((key.strip(), value.strip()))
    return sections


def values(unit: dict, section: str, key: str) -> list[str]:
    return [value for found_key, value in unit.get(section, []) if found_key == key]


@pytest.fixture
def pilot(initialized_home):
    app_config = load_app_config(initialized_home.home_dir)
    plan = plan_one_day_pilot(app_config, now=INSTALLED_AT, collector_uid=997)
    units = {name: parse_unit(text) for name, text in render_pilot_units(plan).items()}
    return plan, units, render_pilot_units(plan)


def test_the_schedule_covers_one_day_and_ends_with_the_next_morning_report(pilot):
    plan, _units, _texts = pilot
    assert [moment.strftime("%Y-%m-%d %H:%M") for moment in plan.collect_moments] == [
        "2026-11-20 18:30",
        "2026-11-21 00:30",
        "2026-11-21 06:30",
    ]  # 12:30 had passed; 12:30 tomorrow is after
    assert plan.report_date.isoformat() == "2026-11-21"
    assert plan.report_cutoff == datetime(2026, 11, 21, 7, 0, tzinfo=NEW_YORK)
    assert plan.finish_moment - plan.report_cutoff == timedelta(minutes=100)
    assert "finish" in describe_pilot_plan(plan)[-1]


def test_every_timer_fires_at_fixed_instants_only_inside_the_pilot(pilot):
    plan, units, _texts = pilot
    for timer_name in PILOT_TIMER_NAMES:
        timer = units[timer_name]
        assert not [key for key, _value in timer["Timer"] if key in RELATIVE_OR_BOOT_TRIGGERS], timer_name
        calendar_times = values(timer, "Timer", "OnCalendar")
        assert calendar_times, timer_name
        for calendar_time in calendar_times:
            assert ABSOLUTE_CALENDAR_TIME.match(calendar_time), calendar_time  # no '*', '..', '/', or weekday
            instant = datetime.strptime(calendar_time.removesuffix(" America/New_York"), "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=NEW_YORK
            )
            assert INSTALLED_AT < instant <= plan.finish_moment
        assert values(timer, "Install", "WantedBy") == ["timers.target"]
    assert len(values(units[REPORT_TIMER], "Timer", "OnCalendar")) == 1
    assert len(values(units[FINISH_TIMER], "Timer", "OnCalendar")) == 1
    assert len(values(units[COLLECT_TIMER], "Timer", "OnCalendar")) == 3


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze is not installed")
def test_systemd_itself_computes_no_second_elapse_for_any_timer(pilot):
    _plan, units, _texts = pilot
    for timer_name in PILOT_TIMER_NAMES:
        for calendar_time in values(units[timer_name], "Timer", "OnCalendar"):
            analysis = subprocess.run(
                ["systemd-analyze", "calendar", "--iterations=5", calendar_time],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert f"Normalized form: {calendar_time}" in analysis.stdout, analysis.stdout + analysis.stderr
            assert "Iteration #2" not in analysis.stdout, f"{calendar_time} would elapse more than once"


def test_no_service_starts_at_boot_and_every_service_expires(pilot):
    plan, units, _texts = pilot
    expected_deadline_by_service = {
        COLLECT_SERVICE: plan.report_cutoff,
        WORKER_SERVICE: plan.report_cutoff,
        REPORT_SERVICE: plan.report_window_end,
    }
    for unit_name in ALL_PILOT_UNIT_NAMES:
        if unit_name.endswith(".service"):
            assert "Install" not in units[unit_name], f"{unit_name} must not be enabled for boot"
    for service_name, deadline in expected_deadline_by_service.items():
        conditions = values(units[service_name], "Service", "ExecCondition")
        assert conditions == [f"/usr/bin/sh -c 'test \"$$(date +%%s)\" -lt {int(deadline.timestamp())}'"]
        assert deadline <= plan.finish_moment


@pytest.mark.parametrize("deadline_offset_seconds, expected_exit_code", [(3600, 0), (-1, 1), (-86400 * 30, 1)])
def test_the_expiry_condition_really_skips_after_its_deadline(pilot, deadline_offset_seconds, expected_exit_code):
    """Run the ExecCondition command line exactly as systemd would (after its `$$`/`%%` unescaping)."""
    _plan, units, _texts = pilot
    rendered_condition = values(units[WORKER_SERVICE], "Service", "ExecCondition")[0]
    deadline_epoch = int(datetime.now(timezone.utc).timestamp()) + deadline_offset_seconds
    command_line = re.sub(r"-lt \d+", f"-lt {deadline_epoch}", rendered_condition)
    shell_script = command_line.removeprefix("/usr/bin/sh -c '").removesuffix("'")
    shell_script = shell_script.replace("$$", "$").replace("%%", "%")
    assert subprocess.run(["sh", "-c", shell_script], timeout=10).returncode == expected_exit_code


def test_the_finish_step_disables_and_deletes_every_pilot_unit(pilot):
    _plan, units, _texts = pilot
    commands = values(units[FINISH_SERVICE], "Service", "ExecStart")
    removal_command = next(command for command in commands if command.startswith("/usr/bin/rm -f "))
    removed_paths = set(removal_command.split()[2:])
    assert {f"/etc/systemd/system/{unit_name}" for unit_name in ALL_PILOT_UNIT_NAMES} <= removed_paths
    assert all(f"/var/lib/systemd/timers/stamp-{timer_name}" in removed_paths for timer_name in PILOT_TIMER_NAMES)
    disabled_units = " ".join(command for command in commands if " disable " in command)
    assert all(timer_name in disabled_units for timer_name in PILOT_TIMER_NAMES)
    assert any(command.endswith(f"stop {WORKER_SERVICE}") for command in commands)
    assert commands.index(removal_command) < commands.index("/usr/bin/systemctl daemon-reload")
    # Nothing stays in systemd's memory: the finish timer is stopped and failed states are reset before deletion.
    assert commands.index(f"-/usr/bin/systemctl stop --no-block {FINISH_TIMER}") < commands.index(removal_command)
    reset_command = next(command for command in commands if command.startswith("-/usr/bin/systemctl reset-failed "))
    assert set(reset_command.split()[2:]) == set(ALL_PILOT_UNIT_NAMES)


def test_paperwatch_runs_only_as_the_service_accounts_and_the_model_side_is_sandboxed(pilot):
    plan, units, _texts = pilot
    assert values(units[COLLECT_SERVICE], "Service", "User") == ["paperwatch-collector"]
    for service_name in (WORKER_SERVICE, REPORT_SERVICE):
        service = units[service_name]
        assert values(service, "Service", "User") == ["paperwatch-worker"]
        assert values(service, "Service", "IPAddressDeny") == ["any"]
        assert values(service, "Service", "IPAddressAllow") == ["localhost"]
        assert values(service, "Service", "ReadWritePaths") == [f"{plan.state_dir} {plan.output_dir}"]
        for hardening_setting, expected_value in (
            ("NoNewPrivileges", "yes"),
            ("ProtectSystem", "strict"),
            ("ProtectHome", "yes"),
            ("PrivateTmp", "yes"),
            ("CapabilityBoundingSet", ""),
        ):
            assert values(service, "Service", hardening_setting) == [expected_value], hardening_setting
    report_command = values(units[REPORT_SERVICE], "Service", "ExecStart")[0]
    assert report_command.endswith("report --date 2026-11-21") and "--early" not in report_command
    assert values(units[REPORT_SERVICE], "Unit", "Conflicts") == [WORKER_SERVICE]
    assert values(units[COLLECT_SERVICE], "Unit", "Wants") == [WORKER_SERVICE]
    assert "User" not in dict(units[FINISH_SERVICE]["Service"])  # root, but it runs only systemctl and rm


def test_a_pilot_cannot_be_planned_after_its_report_cutoff(initialized_home):
    app_config = load_app_config(initialized_home.home_dir)
    late_evening = datetime(2026, 11, 20, 23, 59, tzinfo=NEW_YORK)
    plan = plan_one_day_pilot(app_config, now=late_evening, collector_uid=997)
    assert plan.report_date.isoformat() == "2026-11-21"
    assert [moment.strftime("%H:%M") for moment in plan.collect_moments] == ["00:30", "06:30"]
