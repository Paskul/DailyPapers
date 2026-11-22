"""Publication dates as providers report them, with their precision kept visible (plan part 5).

A stored publication date is a partial ISO date whose LENGTH is its precision:
    "2018"        year   (e.g. a CSL JSON seed, or a provider that reports only the year)
    "2018-06"     month  (e.g. derived from a new-style arXiv id: 1806.01234 was first submitted in June 2018)
    "2018-06-15"  day    (e.g. the arXiv API's submission timestamp, or OpenAlex's publication_date)
So the precision of every date is visible without another column, and no code may pad a coarse date into a fine one.

Three dates are stored independently for every paper: first publication, latest revision, and local discovery time
(a UTC timestamp, paperwatch.clock). A paper first published in 2018 but first found today is OLDER literature, not a
new release: the release class depends on the publication date, never on the discovery time.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from enum import StrEnum

PRECISION_YEAR = "year"
PRECISION_MONTH = "month"
PRECISION_DAY = "day"
PARTIAL_ISO_DATE_PATTERN = re.compile(r"^(?P<year>\d{4})(?:-(?P<month>\d{2})(?:-(?P<day>\d{2}))?)?$")
EARLIEST_PLAUSIBLE_PUBLICATION_YEAR = 1600
NEW_RELEASE_FRESH_WINDOW_DAYS = 7  # plan part 6: "a seven-day fresh window" from the original release date


class ReleaseClass(StrEnum):
    NEW_RELEASE = "new_release"  # first published within the fresh window before it was found
    OLDER_LITERATURE = "older_literature"  # first published earlier, however recently it was found
    UNKNOWN = "unknown"  # no publication date reported


def normalize_partial_date(raw_date_text: object) -> str | None:
    """Return the date in partial ISO form ("YYYY", "YYYY-MM", "YYYY-MM-DD"), or None if it is not a valid date.

    Accepts a full timestamp ("2026-11-18T18:59:38Z", "2026-11-19T00:00:00-05:00") and keeps only its calendar date
    exactly as written (the provider's own day), without converting time zones.
    """
    if not isinstance(raw_date_text, str):
        return None
    candidate_text = raw_date_text.strip()[:10] if "T" in raw_date_text else raw_date_text.strip()
    date_match = PARTIAL_ISO_DATE_PATTERN.match(candidate_text)
    if date_match is None:
        return None
    year = int(date_match["year"])
    if year < EARLIEST_PLAUSIBLE_PUBLICATION_YEAR:
        return None
    try:
        if date_match["day"]:
            date(year, int(date_match["month"]), int(date_match["day"]))
        elif date_match["month"] and not 1 <= int(date_match["month"]) <= 12:
            return None
    except ValueError:
        return None
    return candidate_text


def describe_date_precision(partial_date: str | None) -> str | None:
    """'year', 'month', 'day', or None for a missing or malformed date."""
    if normalize_partial_date(partial_date) != partial_date or partial_date is None:
        return None
    return {4: PRECISION_YEAR, 7: PRECISION_MONTH, 10: PRECISION_DAY}[len(partial_date)]


def is_more_precise_refinement(existing_date: str | None, candidate_date: str | None) -> bool:
    """True if `candidate_date` says the same thing as `existing_date`, only more precisely ("2018-06" -> "2018-06-15").

    The catalog uses this to sharpen a coarse first-publication date without ever contradicting it.
    """
    if not existing_date or not candidate_date or normalize_partial_date(candidate_date) is None:
        return False
    return len(candidate_date) > len(existing_date) and candidate_date.startswith(existing_date)


def earliest_calendar_day(partial_date: str) -> date:
    """The first day the partial date could mean: "2018" -> 2018-01-01, "2018-06" -> 2018-06-01."""
    year_text, _, rest = partial_date.partition("-")
    month_text, _, day_text = rest.partition("-")
    return date(int(year_text), int(month_text or 1), int(day_text or 1))


def classify_release(
    first_published_date: str | None, discovered_at: datetime, fresh_window_days: int = NEW_RELEASE_FRESH_WINDOW_DAYS
) -> ReleaseClass:
    """New release if first published no more than `fresh_window_days` before discovery; else older literature.

    A coarse date counts from its earliest possible day: a paper known only as "2026" is new only if found in the
    first week of 2026. This errs toward "older" rather than calling an old paper new.
    """
    if normalize_partial_date(first_published_date) is None:
        return ReleaseClass.UNKNOWN
    published_day = earliest_calendar_day(first_published_date)
    if discovered_at.date() - published_day <= timedelta(days=fresh_window_days):
        return ReleaseClass.NEW_RELEASE
    return ReleaseClass.OLDER_LITERATURE
