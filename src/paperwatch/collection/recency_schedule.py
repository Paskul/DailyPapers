"""Recency-weighted time windows for the historical search (owner decision 2026-11-19).

The goal: spend the topic's historical candidate budget so that the density of candidates per publication year
DECAYS EXPONENTIALLY with age, down to an absolute floor year (default 1940), while every era keeps a guaranteed
share, so influential old papers can still be found.

The weighting curve
    weight(age) = 2 ** (-age / half_life_years),   for 0 <= age <= span (plan date back to the floor year)
A paper 6 years old (with the default half-life of 6) carries half the weight of a paper from this year; one 12 years
old a quarter; one 60 years old about a thousandth.

From curve to requests: equal-mass quantile windows
Providers can only be asked for date ranges, so the curve is cut into `window_count` windows that each hold the SAME
share of the curve's area (its quantiles). The boundary ages are the inverse of the truncated exponential CDF:
    age_k = -half_life * log2(1 - (k / K) * (1 - 2 ** (-span / half_life))),   k = 0 .. K
With the defaults (half-life 6, K = 8, floor 1940, plan date 2026-11) the windows are about 1.2, 1.3, 1.6, 1.9, 2.5,
3.5, 6 and 68 years wide, newest first. Giving each window an equal quota then reproduces the exponential density
exactly in expectation: narrow recent windows get as many candidates as the 68-year window 1940-2008.
Window boundaries are fixed when a topic's historical plan is created and stored in the plan, so cursors stay valid
while the plan runs. New work after the plan date comes from the current-release routes.

Inside a window: most-cited first for old windows
A window whose NEWEST day is at least `citation_ordering_min_age_years` old asks OpenAlex for its most-cited matches
first (sort=cited_by_count:desc). Its small quota then goes to the influential papers of that era rather than
arbitrary ones. Recent windows use relevance, because young papers have not had time to collect citations.
arXiv has no citation data and uses relevance everywhere; it also starts in August 1991, so arXiv is not asked about
older windows at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from paperwatch.network.provider_requests import PublicationWindow, SearchResultOrder

DAYS_PER_YEAR = 365.2425
ARXIV_FIRST_ANNOUNCEMENT_DATE = date(1991, 8, 14)  # arXiv holds nothing older


@dataclass(frozen=True)
class RecencyScheduleSettings:
    earliest_year: int = 1940
    half_life_years: float = 6.0
    window_count: int = 8
    citation_ordering_min_age_years: float = 5.0

    def to_parameters(self) -> dict[str, Any]:
        return {
            "earliest_year": self.earliest_year,
            "half_life_years": self.half_life_years,
            "window_count": self.window_count,
            "citation_ordering_min_age_years": self.citation_ordering_min_age_years,
        }


@dataclass(frozen=True)
class HistoricalWindow:
    first_date: date
    last_date: date
    weight: float  # this window's share of the topic's historical budget; the shares sum to 1
    result_order: SearchResultOrder

    @property
    def label(self) -> str:
        return f"{self.first_date.isoformat()}..{self.last_date.isoformat()}"

    @property
    def publication_window(self) -> PublicationWindow:
        return PublicationWindow(self.first_date, self.last_date)

    def is_searchable_on(self, provider_name: str) -> bool:
        return provider_name != "arxiv" or self.last_date >= ARXIV_FIRST_ANNOUNCEMENT_DATE

    def to_parameters(self) -> dict[str, Any]:
        return {
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "weight": round(self.weight, 9),
            "result_order": str(self.result_order),
        }

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> HistoricalWindow:
        return cls(
            date.fromisoformat(parameters["first_date"]),
            date.fromisoformat(parameters["last_date"]),
            float(parameters["weight"]),
            SearchResultOrder(parameters["result_order"]),
        )


def exponential_boundary_ages(span_years: float, half_life_years: float, window_count: int) -> list[float]:
    """Ages (in years) that cut the truncated exponential over [0, span] into `window_count` equal-mass parts."""
    tail_weight = 2.0 ** (-span_years / half_life_years)
    boundary_ages = [
        -half_life_years * math.log2(1.0 - (index / window_count) * (1.0 - tail_weight))
        for index in range(window_count)
    ]
    return boundary_ages + [span_years]  # the last boundary is exactly the floor (avoids rounding drift)


def build_recency_weighted_windows(plan_date: date, settings: RecencyScheduleSettings) -> list[HistoricalWindow]:
    """Contiguous, non-overlapping windows from `plan_date` back to 1 January of the floor year, newest first."""
    earliest_date = date(settings.earliest_year, 1, 1)
    span_years = (plan_date - earliest_date).days / DAYS_PER_YEAR
    if span_years <= 0:
        return [HistoricalWindow(plan_date, plan_date, 1.0, SearchResultOrder.RELEVANCE)]
    boundary_dates = [
        plan_date - timedelta(days=round(age * DAYS_PER_YEAR))
        for age in exponential_boundary_ages(span_years, settings.half_life_years, settings.window_count)
    ]
    boundary_dates[-1] = earliest_date

    windows: list[HistoricalWindow] = []
    equal_share = 1.0 / settings.window_count
    for index in range(settings.window_count):
        last_date = plan_date if index == 0 else boundary_dates[index] - timedelta(days=1)
        first_date = boundary_dates[index + 1]
        if first_date > last_date:  # thinner than a day (only with extreme settings): fold into the previous one
            if windows:
                previous = windows[-1]
                windows[-1] = HistoricalWindow(
                    previous.first_date, previous.last_date, previous.weight + equal_share, previous.result_order
                )
            continue
        newest_age_years = (plan_date - last_date).days / DAYS_PER_YEAR
        result_order = (
            SearchResultOrder.MOST_CITED
            if newest_age_years >= settings.citation_ordering_min_age_years
            else SearchResultOrder.RELEVANCE
        )
        windows.append(HistoricalWindow(first_date, last_date, equal_share, result_order))
    return windows
