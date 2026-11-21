"""The ONE-DAY pilot schedule (plan part 10, "run a 24-hour pilot ... before enabling the regular timers").

Owner decision, 2026-11-20: the timers of this pilot run exactly once and never again. They are rendered here as
systemd units with ABSOLUTE calendar times, and several independent layers make a second run impossible:
  1. Every OnCalendar= is one fixed date and time with an explicit timezone ("2026-11-21 00:30:00 America/New_York"):
     no wildcard, no weekday, no repetition, and no OnBootSec=/OnUnitActiveSec= style relative trigger.
  2. Every service has an ExecCondition= that compares the current time with a fixed end of the pilot (a Unix
     epoch rendered into the unit). After it, the service is skipped even if something starts it.
  3. No service is enabled for boot (no [Install] section); only the three timers are, and their times are past
     after the pilot.
  4. The finish timer (90+ minutes after the report cutoff) stops the worker, disables every pilot timer, deletes
     every pilot unit file, and reloads systemd, so nothing of the pilot is left to trigger.

Schedule for pilot day D (times from config.toml, in report.timezone):
  collect   at each schedule.collect_times that is still ahead and before the report cutoff (and once at install)
  worker    started with each collect (Wants=), runs `paperwatch work` until the cutoff
  report    at the cutoff on D+1 plus REPORT_START_DELAY: stops the worker (Conflicts=), then publishes D+1's
            report; retried on failure until REPORT_RETRY_WINDOW after the cutoff
  finish    at the cutoff plus FINISH_DELAY: removes the pilot

The worker and the report run in a tight systemd sandbox on top of the nftables rules (read-only system, private
/tmp and /dev, no home directories, write access only to state/ and output/, IP traffic only to localhost, no new
privileges). The collector keeps a lighter sandbox, because rootless Podman (the PDF extractor) needs setuid id
mapping and /run/user; it relies on nftables, the egress guard, and file permissions.

Run as root by deploy/pilot/install_one_day_pilot.sh:
    python -m paperwatch.deployment.one_day_pilot --home /srv/paperwatch --unit-dir /etc/systemd/system
"""

from __future__ import annotations

import argparse
import pwd
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from paperwatch.config import AppConfig, load_app_config
from paperwatch.deployment.systemd_units import worker_sandbox_lines
from paperwatch.reporting.report_schedule import parse_clock_time

PILOT_UNIT_PREFIX = "paperwatch-pilot-"
COLLECT_SERVICE = f"{PILOT_UNIT_PREFIX}collect.service"
COLLECT_TIMER = f"{PILOT_UNIT_PREFIX}collect.timer"
WORKER_SERVICE = f"{PILOT_UNIT_PREFIX}worker.service"
REPORT_SERVICE = f"{PILOT_UNIT_PREFIX}report.service"
REPORT_TIMER = f"{PILOT_UNIT_PREFIX}report.timer"
FINISH_SERVICE = f"{PILOT_UNIT_PREFIX}finish.service"
FINISH_TIMER = f"{PILOT_UNIT_PREFIX}finish.timer"
ALL_PILOT_UNIT_NAMES = (COLLECT_SERVICE, COLLECT_TIMER, WORKER_SERVICE, REPORT_SERVICE, REPORT_TIMER, FINISH_SERVICE,
                        FINISH_TIMER)
PILOT_TIMER_NAMES = (COLLECT_TIMER, REPORT_TIMER, FINISH_TIMER)

REPORT_START_DELAY = timedelta(seconds=30)  # start just after the cutoff, so the report date is surely due
REPORT_RETRY_WINDOW = timedelta(minutes=35)  # a failed report is retried every 5 minutes until cutoff + this
REPORT_TIMEOUT = timedelta(minutes=50)  # writer deadline (30 min) plus snapshot, notes, and publication
FINISH_DELAY = timedelta(minutes=100)  # after the last possible report retry has ended (35 + 50 < 100)
COLLECT_TIMEOUT = timedelta(hours=2)
SYSTEMD_TIMER_STAMP_DIR = Path("/var/lib/systemd/timers")
DEFAULT_PAPERWATCH_EXECUTABLE = Path("/opt/paperwatch/venv/bin/paperwatch")


@dataclass(frozen=True)
class OneDayPilotPlan:
    """Everything the unit files depend on. All moments are aware datetimes."""
    pilot_date: date  # the day the pilot starts; its report is the one for pilot_date + 1
    timezone_name: str
    collect_moments: tuple[datetime, ...]  # future collect times before the cutoff
    report_cutoff: datetime  # publish_time on pilot_date + 1
    paperwatch_home: Path
    state_dir: Path
    output_dir: Path
    paperwatch_executable: Path
    collector_account: str
    collector_uid: int
    worker_account: str
    shared_state_group: str  # both accounts use SQLite in state/ through this group

    @property
    def report_date(self) -> date:
        return self.pilot_date + timedelta(days=1)

    @property
    def report_start(self) -> datetime:
        return self.report_cutoff + REPORT_START_DELAY

    @property
    def report_window_end(self) -> datetime:
        return self.report_cutoff + REPORT_RETRY_WINDOW

    @property
    def finish_moment(self) -> datetime:
        return self.report_cutoff + FINISH_DELAY


def plan_one_day_pilot(app_config: AppConfig, *, now: datetime, collector_uid: int,
                       paperwatch_executable: Path = DEFAULT_PAPERWATCH_EXECUTABLE) -> OneDayPilotPlan:
    """The pilot that starts today (in report.timezone) and reports tomorrow at report.publish_time."""
    pilot_timezone = ZoneInfo(app_config.report.timezone)
    pilot_date = now.astimezone(pilot_timezone).date()
    report_cutoff = datetime.combine(pilot_date + timedelta(days=1),
                                     parse_clock_time(app_config.report.publish_time), tzinfo=pilot_timezone)
    collect_moments = sorted(
        moment for day in (pilot_date, pilot_date + timedelta(days=1))
        for clock_text in app_config.schedule.collect_times
        if now < (moment := datetime.combine(day, parse_clock_time(clock_text), tzinfo=pilot_timezone))
        < report_cutoff)
    if report_cutoff <= now:
        raise ValueError("the report cutoff of this pilot is not in the future")
    return OneDayPilotPlan(
        pilot_date=pilot_date, timezone_name=app_config.report.timezone, collect_moments=tuple(collect_moments),
        report_cutoff=report_cutoff, paperwatch_home=app_config.home_dir, state_dir=app_config.state_dir,
        output_dir=app_config.output_dir, paperwatch_executable=paperwatch_executable,
        collector_account=app_config.access.collector_account, collector_uid=collector_uid,
        worker_account=app_config.access.worker_account, shared_state_group=app_config.access.shared_state_group)


# --------------------------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------------------------


def systemd_calendar_time(moment: datetime, timezone_name: str) -> str:
    """One absolute instant, e.g. '2026-11-21 00:30:00 America/New_York'. Never a repeating expression."""
    return f"{moment.astimezone(ZoneInfo(timezone_name)):%Y-%m-%d %H:%M:%S} {timezone_name}"


def only_before_condition(deadline: datetime) -> str:
    """ExecCondition: exit 0 (run) only before `deadline`. `$$` and `%%` are systemd's escapes for `$` and `%`."""
    deadline_epoch = int(deadline.timestamp())
    return (f"# Skip (never run) at or after {deadline.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%SZ}, whoever starts "
            f"this unit.\nExecCondition=/usr/bin/sh -c 'test \"$$(date +%%s)\" -lt {deadline_epoch}'")


def _pilot_notice(plan: OneDayPilotPlan) -> str:
    return (f"# ONE-DAY PILOT of paperwatch (plan part 10), {plan.pilot_date} -> report {plan.report_date}.\n"
            f"# Generated by paperwatch.deployment.one_day_pilot. Runs once; {FINISH_SERVICE} deletes this file at\n"
            f"# {systemd_calendar_time(plan.finish_moment, plan.timezone_name)}.\n")


def _paperwatch_command(plan: OneDayPilotPlan, *arguments: str) -> str:
    return " ".join([str(plan.paperwatch_executable), "--home", str(plan.paperwatch_home), *arguments])


def _worker_sandbox_lines(plan: OneDayPilotPlan) -> str:
    """systemd sandbox for `work` and `report` (the processes that talk to the model)."""
    return worker_sandbox_lines(worker_account=plan.worker_account, shared_state_group=plan.shared_state_group,
                                state_dir=plan.state_dir, output_dir=plan.output_dir)


def render_pilot_units(plan: OneDayPilotPlan) -> dict[str, str]:
    """{unit file name: content} for every pilot unit."""
    notice = _pilot_notice(plan)
    collect_calendar_lines = "\n".join(f"OnCalendar={systemd_calendar_time(moment, plan.timezone_name)}"
                                       for moment in plan.collect_moments)
    unit_paths = " ".join(f"/etc/systemd/system/{unit_name}" for unit_name in ALL_PILOT_UNIT_NAMES)
    unit_names = " ".join(ALL_PILOT_UNIT_NAMES)
    stamp_paths = " ".join(str(SYSTEMD_TIMER_STAMP_DIR / f"stamp-{timer_name}") for timer_name in PILOT_TIMER_NAMES)
    units = {
        COLLECT_SERVICE: f"""{notice}
[Unit]
Description=paperwatch pilot: collect papers (one-day pilot, ends {plan.report_cutoff:%Y-%m-%d %H:%M})
Wants={WORKER_SERVICE}

[Service]
Type=oneshot
User={plan.collector_account}
Group={plan.collector_account}
SupplementaryGroups={plan.shared_state_group}
UMask=0007
WorkingDirectory=/
# Rootless Podman (the sandboxed PDF extractor) keeps its runtime files in the account's lingering session.
Environment=XDG_RUNTIME_DIR=/run/user/{plan.collector_uid}
{only_before_condition(plan.report_cutoff)}
ExecStart={_paperwatch_command(plan, "collect")}
TimeoutStartSec={int(COLLECT_TIMEOUT.total_seconds())}
Nice=10
ProtectSystem=full
""",
        COLLECT_TIMER: f"""{notice}
[Unit]
Description=paperwatch pilot: collection times (absolute; none after {plan.report_cutoff:%Y-%m-%d %H:%M})

[Timer]
{collect_calendar_lines}
AccuracySec=1s
# Persistent: a collection missed while the machine was off runs once at boot (still only before the cutoff).
Persistent=true
Unit={COLLECT_SERVICE}

[Install]
WantedBy=timers.target
""",
        WORKER_SERVICE: f"""{notice}
[Unit]
Description=paperwatch pilot: worker (reads queued papers with the local model until the report cutoff)

[Service]
Type=simple
{_worker_sandbox_lines(plan)}{only_before_condition(plan.report_cutoff)}
ExecStart={_paperwatch_command(plan, "work")}
Restart=on-failure
RestartSec=60
# SIGTERM aborts a model call at once and puts the job back in the queue without using an attempt; this is only
# the backstop if that ever hangs.
TimeoutStopSec=60
""",
        REPORT_SERVICE: f"""{notice}
[Unit]
Description=paperwatch pilot: publish the report for {plan.report_date}
# Starting the report stops the worker first, so the writer gets the model at once.
Conflicts={WORKER_SERVICE}
After={WORKER_SERVICE}

[Service]
Type=oneshot
{_worker_sandbox_lines(plan)}{only_before_condition(plan.report_window_end)}
ExecStart={_paperwatch_command(plan, "report", "--date", plan.report_date.isoformat())}
TimeoutStartSec={int(REPORT_TIMEOUT.total_seconds())}
Restart=on-failure
RestartSec=300
""",
        REPORT_TIMER: f"""{notice}
[Unit]
Description=paperwatch pilot: report time (absolute, once)

[Timer]
OnCalendar={systemd_calendar_time(plan.report_start, plan.timezone_name)}
AccuracySec=1s
Persistent=true
Unit={REPORT_SERVICE}

[Install]
WantedBy=timers.target
""",
        FINISH_SERVICE: f"""{notice}
[Unit]
Description=paperwatch pilot: finish (stop the worker, disable and DELETE every pilot unit)

[Service]
Type=oneshot
ExecStart=-/usr/bin/systemctl stop {WORKER_SERVICE}
ExecStart=-/usr/bin/systemctl disable --now {COLLECT_TIMER} {REPORT_TIMER}
ExecStart=-/usr/bin/systemctl disable {FINISH_TIMER}
# Leave no state in systemd's memory either (first pilot, 2026-11-21: the finish timer stayed "active (running)" and
# the killed worker "failed" after their files were gone). Stopping the timer does not stop this service.
ExecStart=-/usr/bin/systemctl stop --no-block {FINISH_TIMER}
ExecStart=-/usr/bin/systemctl reset-failed {unit_names}
ExecStart=/usr/bin/rm -f {unit_paths} {stamp_paths}
ExecStart=/usr/bin/systemctl daemon-reload
ExecStart=-/usr/bin/systemctl reset-failed
""",
        FINISH_TIMER: f"""{notice}
[Unit]
Description=paperwatch pilot: finish time (absolute, once)

[Timer]
OnCalendar={systemd_calendar_time(plan.finish_moment, plan.timezone_name)}
AccuracySec=1s
# Persistent: if the machine was off at that time, the pilot is removed at the next boot.
Persistent=true
Unit={FINISH_SERVICE}

[Install]
WantedBy=timers.target
""",
    }
    assert set(units) == set(ALL_PILOT_UNIT_NAMES)
    return units


def describe_pilot_plan(plan: OneDayPilotPlan) -> list[str]:
    describe = lambda moment: systemd_calendar_time(moment, plan.timezone_name)  # noqa: E731
    return ([f"Pilot {plan.pilot_date} -> report {plan.report_date}"] +
            [f"  collect  {describe(moment)}" for moment in plan.collect_moments] +
            [f"  worker   with each collect, until {describe(plan.report_cutoff)}",
             f"  report   {describe(plan.report_start)} (retried until {describe(plan.report_window_end)})",
             f"  finish   {describe(plan.finish_moment)}: worker stopped, every pilot unit disabled and deleted"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the one-day pilot's systemd units (plan part 10).")
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--unit-dir", type=Path, required=True, help="e.g. /etc/systemd/system")
    parser.add_argument("--paperwatch-executable", type=Path, default=DEFAULT_PAPERWATCH_EXECUTABLE)
    arguments = parser.parse_args(argv)
    app_config = load_app_config(arguments.home)
    plan = plan_one_day_pilot(app_config, now=datetime.now(timezone.utc),
                              collector_uid=pwd.getpwnam(app_config.access.collector_account).pw_uid,
                              paperwatch_executable=arguments.paperwatch_executable)
    for unit_name, unit_text in render_pilot_units(plan).items():
        unit_path = arguments.unit_dir / unit_name
        unit_path.write_text(unit_text)
        unit_path.chmod(0o644)
    print("\n".join(describe_pilot_plan(plan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
