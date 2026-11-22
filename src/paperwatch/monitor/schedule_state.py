"""What is scheduled to run, and which paperwatch services are running now, as far as systemd knows.

Two sources, both read-only:
  * The installed timer unit files (the one-day pilot's, or the daily schedule's) give every planned moment. The
    pilot's OnCalendar= lines are absolute ("2026-11-21 00:30:00 America/New_York"); the daily schedule's repeat
    ("*-*-* 06:30:00 America/New_York"). Both forms are expanded here into the moments inside the monitor's horizon.
    A timer's stamp file (written by systemd for Persistent= timers) gives the moment it last fired.
  * `systemctl show` (one fixed, read-only query per scope) gives each service's state: a oneshot collect that is
    running is `activating`, the worker is `active`.
Without systemd (a plain `local` setup run by hand) both are simply empty and the monitor says nothing is scheduled.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paperwatch.deployment.unit_names import (
    DAILY_COLLECT_TIMER,
    DAILY_REPORT_TIMER,
    DAILY_SERVICE_NAMES,
    DAILY_TIMER_NAMES,
    MONITOR_SERVICE,
)
from paperwatch.deployment.one_day_pilot import (
    ALL_PILOT_UNIT_NAMES,
    COLLECT_TIMER as PILOT_COLLECT_TIMER,
    FINISH_TIMER as PILOT_FINISH_TIMER,
    REPORT_TIMER as PILOT_REPORT_TIMER,
)

SYSTEM_UNIT_DIR = Path("/etc/systemd/system")
SYSTEM_TIMER_STAMP_DIR = Path("/var/lib/systemd/timers")
SYSTEMCTL_PATH = "/usr/bin/systemctl"
SYSTEMCTL_TIMEOUT_SECONDS = 5.0
SHOWN_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "Result",
    "ExecMainStartTimestamp",
    "ExecMainPID",
    "UnitFileState",
)

ABSOLUTE_CALENDAR = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})(?: (\S+))?$")
DAILY_CALENDAR = re.compile(r"^\*-\*-\* (\d{2}):(\d{2}):(\d{2})(?: (\S+))?$")

# What each timer does, as shown in the plan.
TIMER_KINDS = {
    PILOT_COLLECT_TIMER: ("collect", "pilot"),
    PILOT_REPORT_TIMER: ("report", "pilot"),
    PILOT_FINISH_TIMER: ("pilot_finish", "pilot"),
    DAILY_COLLECT_TIMER: ("collect", "daily"),
    DAILY_REPORT_TIMER: ("report", "daily"),
}
PILOT_SERVICE_NAMES = tuple(name for name in ALL_PILOT_UNIT_NAMES if name.endswith(".service"))


@dataclass(frozen=True)
class ScheduledMoment:
    at: datetime
    kind: str  # collect | report | pilot_finish
    schedule_source: str  # pilot | daily
    timer_name: str


@dataclass(frozen=True)
class ServiceState:
    unit_name: str
    scope: str  # system | user
    load_state: str  # loaded | not-found | ...
    active_state: str  # active | activating | inactive | failed | ...
    sub_state: str
    result: str
    main_started_at: str | None  # as systemd prints it
    main_process_id: int | None
    unit_file_state: str  # enabled | disabled | static | ...

    @property
    def is_installed(self) -> bool:
        return self.load_state == "loaded"

    @property
    def is_running(self) -> bool:
        return self.active_state in ("active", "activating", "reloading", "deactivating")


@dataclass(frozen=True)
class ScheduleState:
    schedule_source: str  # pilot | daily | none
    moments: tuple[ScheduledMoment, ...]  # inside the requested window, in time order
    last_fired: dict[str, datetime] = field(default_factory=dict)  # timer name -> last trigger
    services: dict[str, ServiceState] = field(default_factory=dict)
    systemd_available: bool = False

    def service(self, *unit_names: str) -> ServiceState | None:
        """The first of `unit_names` that is installed (pilot and daily units do the same jobs)."""
        for unit_name in unit_names:
            state = self.services.get(unit_name)
            if state is not None and state.is_installed:
                return state
        return None


def read_schedule_state(
    *,
    window_start: datetime,
    window_end: datetime,
    unit_dirs: tuple[Path, ...] | None = None,
    stamp_dirs: tuple[Path, ...] | None = None,
    query_systemd: bool = True,
) -> ScheduleState:
    """Everything the monitor needs from systemd. `unit_dirs`/`stamp_dirs`/`query_systemd` exist for tests."""
    unit_dirs = unit_dirs if unit_dirs is not None else (SYSTEM_UNIT_DIR, user_unit_dir())
    stamp_dirs = stamp_dirs if stamp_dirs is not None else (SYSTEM_TIMER_STAMP_DIR, user_timer_stamp_dir())
    moments: list[ScheduledMoment] = []
    last_fired: dict[str, datetime] = {}
    for timer_name, (kind, schedule_source) in TIMER_KINDS.items():
        timer_path = next((unit_dir / timer_name for unit_dir in unit_dirs if (unit_dir / timer_name).is_file()), None)
        if timer_path is None:
            continue
        for calendar_value in read_on_calendar_values(timer_path):
            moments.extend(
                ScheduledMoment(moment, kind, schedule_source, timer_name)
                for moment in expand_calendar_value(calendar_value, window_start, window_end)
            )
        for stamp_dir in stamp_dirs:
            stamp_path = stamp_dir / f"stamp-{timer_name}"
            try:
                last_fired[timer_name] = datetime.fromtimestamp(stamp_path.stat().st_mtime, timezone.utc)
                break
            except OSError:
                continue
    sources = {moment.schedule_source for moment in moments} | {TIMER_KINDS[name][1] for name in last_fired}
    schedule_source = "pilot" if "pilot" in sources else ("daily" if "daily" in sources else "none")
    services: dict[str, ServiceState] = {}
    systemd_available = False
    if query_systemd and Path(SYSTEMCTL_PATH).is_file():
        system_states = query_service_states("system", PILOT_SERVICE_NAMES + DAILY_SERVICE_NAMES + DAILY_TIMER_NAMES)
        user_states = query_service_states("user", DAILY_SERVICE_NAMES + DAILY_TIMER_NAMES + (MONITOR_SERVICE,))
        systemd_available = system_states is not None
        for states in (system_states or {}, user_states or {}):
            for unit_name, state in states.items():
                if state.is_installed or unit_name not in services:
                    services[unit_name] = state
    return ScheduleState(
        schedule_source=schedule_source,
        moments=tuple(sorted(moments, key=lambda m: m.at)),
        last_fired=last_fired,
        services=services,
        systemd_available=systemd_available,
    )


def user_unit_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "systemd" / "user"


def user_timer_stamp_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "systemd" / "timers"


def read_on_calendar_values(timer_path: Path) -> list[str]:
    values = []
    try:
        for line in timer_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.strip().partition("=")
            if separator and key.strip() == "OnCalendar":
                values.append(value.strip())
    except OSError:
        return []
    return values


def expand_calendar_value(calendar_value: str, window_start: datetime, window_end: datetime) -> list[datetime]:
    """The moments of one OnCalendar= value inside [window_start, window_end]. Only the two forms paperwatch
    writes are understood; anything else is ignored (shown as nothing rather than guessed)."""
    if absolute_match := ABSOLUTE_CALENDAR.match(calendar_value):
        year, month, day, hour, minute, second = (int(part) for part in absolute_match.groups()[:6])
        zone = _zone_or_none(absolute_match.group(7))
        if zone is None:
            return []
        moment = datetime(year, month, day, hour, minute, second, tzinfo=zone)
        return [moment] if window_start <= moment <= window_end else []
    if daily_match := DAILY_CALENDAR.match(calendar_value):
        hour, minute, second = (int(part) for part in daily_match.groups()[:3])
        zone = _zone_or_none(daily_match.group(4))
        if zone is None:
            return []
        moments = []
        local_day: date = window_start.astimezone(zone).date() - timedelta(days=1)
        while True:
            moment = datetime.combine(local_day, time(hour, minute, second), tzinfo=zone)
            if moment > window_end:
                return moments
            if moment >= window_start:
                moments.append(moment)
            local_day += timedelta(days=1)
    return []


def _zone_or_none(zone_name: str | None):
    if zone_name is None:
        return datetime.now().astimezone().tzinfo  # systemd's default: the machine's local time
    try:
        return ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def query_service_states(scope: str, unit_names: tuple[str, ...]) -> dict[str, ServiceState] | None:
    """One `systemctl [--user] show` for all units. None if systemd cannot be asked (no bus, no user manager)."""
    command = (
        [SYSTEMCTL_PATH]
        + (["--user"] if scope == "user" else [])
        + ["show", "--no-pager", f"--property={','.join(SHOWN_PROPERTIES)}", "--", *unit_names]
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_TIMEOUT_SECONDS,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return parse_systemctl_show_output(completed.stdout, scope)


def parse_systemctl_show_output(output_text: str, scope: str) -> dict[str, ServiceState]:
    """`systemctl show` prints one block of KEY=value lines per unit, separated by blank lines."""
    states: dict[str, ServiceState] = {}
    for block in re.split(r"\n\s*\n", output_text.strip()):
        properties = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        unit_name = properties.get("Id")
        if not unit_name:
            continue
        main_process_id = properties.get("ExecMainPID", "0")
        states[unit_name] = ServiceState(
            unit_name=unit_name,
            scope=scope,
            load_state=properties.get("LoadState", ""),
            active_state=properties.get("ActiveState", ""),
            sub_state=properties.get("SubState", ""),
            result=properties.get("Result", ""),
            main_started_at=properties.get("ExecMainStartTimestamp") or None,
            main_process_id=int(main_process_id) if main_process_id.isdigit() and main_process_id != "0" else None,
            unit_file_state=properties.get("UnitFileState", ""),
        )
    return states
