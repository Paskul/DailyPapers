"""Single source of "now" plus the one timestamp format stored in SQLite.

All stored timestamps are UTC strings in the fixed-width form 2026-11-19T14:05:00Z, so lexicographic order in SQL
equals chronological order. Conversion to the reporting timezone happens only at the reporting boundary.
Tests can monkeypatch `utc_now` to control time.
"""

from __future__ import annotations

from datetime import datetime, timezone

STORED_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(STORED_TIMESTAMP_FORMAT)


def utc_now_timestamp() -> str:
    return format_utc_timestamp(utc_now())


def parse_utc_timestamp(stored_timestamp: str) -> datetime:
    return datetime.strptime(stored_timestamp, STORED_TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
