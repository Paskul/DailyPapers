"""When a report is due and which reviews it may contain (plan part 9).

Each reporting date D (a calendar date in report.timezone) has one SCHEDULED CUTOFF: report.publish_time on D in that
timezone, converted to UTC. zoneinfo applies the daylight-saving offset of that very day, so 07:00 stays 07:00 local
across a DST change. Reviews completed at or before the cutoff are eligible; later completions belong to a later
report, however long the publication itself takes.

Which date `report` publishes:
  * no --date: the latest due date (today if the cutoff has passed, else yesterday). After a long shutdown this is
    the latest due report, and its opening shows the gap since the previous one (plan part 10).
  * --date D whose cutoff has passed: D, with its scheduled cutoff (a catch-up of a missed date).
  * --date D whose cutoff is still ahead: refused (ReportNotDueYet), unless the operator adds --early for TODAY's
    date. An early report uses "now" as its eligibility cutoff and says so in its opening; reviews completed after
    it belong to the next report. The timer never uses --early.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from paperwatch.config import ReportSettings
from paperwatch.errors import ConfigurationError, ReportNotDueYet


@dataclass(frozen=True)
class ReportTiming:
    report_date: str  # YYYY-MM-DD in the report timezone
    scheduled_cutoff_at: datetime  # UTC
    cutoff_at: datetime  # UTC eligibility cutoff: the scheduled one, or the moment of an early publication
    published_early: bool


def parse_clock_time(clock_text: str) -> time:
    hour_text, minute_text = clock_text.split(":")
    return time(int(hour_text), int(minute_text))


def scheduled_cutoff_for_report_date(report_date: date, report_settings: ReportSettings) -> datetime:
    """publish_time on `report_date` in the report timezone, as an aware UTC datetime.

    A publish time inside a spring-forward gap (e.g. 02:30 on the DST day) resolves the way zoneinfo does (fold=0:
    the offset before the change), which is a valid, unique instant."""
    local_cutoff = datetime.combine(report_date, parse_clock_time(report_settings.publish_time),
                                    tzinfo=ZoneInfo(report_settings.timezone))
    return local_cutoff.astimezone(timezone.utc)


def local_date_at(moment: datetime, report_settings: ReportSettings) -> date:
    return moment.astimezone(ZoneInfo(report_settings.timezone)).date()


def latest_due_report_date(moment: datetime, report_settings: ReportSettings) -> date:
    """Today (local) if today's cutoff has passed at `moment`, otherwise yesterday."""
    today = local_date_at(moment, report_settings)
    if scheduled_cutoff_for_report_date(today, report_settings) <= moment:
        return today
    return today - timedelta(days=1)


def resolve_report_timing(report_settings: ReportSettings, moment: datetime, requested_report_date: str | None,
                          *, publish_early: bool) -> ReportTiming:
    if requested_report_date is None:
        report_date = local_date_at(moment, report_settings) if publish_early \
            else latest_due_report_date(moment, report_settings)
    else:
        try:
            report_date = date.fromisoformat(requested_report_date)
        except ValueError:
            raise ConfigurationError(f"--date must be YYYY-MM-DD, got '{requested_report_date}'") from None

    scheduled_cutoff_at = scheduled_cutoff_for_report_date(report_date, report_settings)
    if scheduled_cutoff_at <= moment:
        return ReportTiming(report_date.isoformat(), scheduled_cutoff_at, scheduled_cutoff_at, published_early=False)
    local_publish_time = f"{report_settings.publish_time} {report_settings.timezone}"
    if not publish_early:
        raise ReportNotDueYet(
            f"the report for {report_date.isoformat()} is not due until {local_publish_time}",
            operator_hint="omit --date to publish the latest due report, or add --early to publish today's report "
                          "now (its cutoff becomes the current time)")
    if report_date != local_date_at(moment, report_settings):
        raise ReportNotDueYet(f"--early publishes only today's report ({local_date_at(moment, report_settings)}); "
                              f"{report_date.isoformat()} is not today")
    return ReportTiming(report_date.isoformat(), scheduled_cutoff_at, moment.astimezone(timezone.utc).replace(
        microsecond=0), published_early=True)
