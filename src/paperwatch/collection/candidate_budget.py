"""The daily budget of new metadata candidates (plan part 5: start with 500 per day, current releases first).

A candidate is a paper the catalog did not know before, found by an arXiv or OpenAlex route (seeds and fixtures do not
count). The count is derived from the catalog itself (papers.origin_route and papers.discovered_at for the current
UTC day), so it survives restarts and is shared by concurrent collectors without another table.

How the day's budget is shared:
  * category feeds   arXiv category feeds and their missed-release recovery searches together admit at most
                     `category_feed_candidate_limit` (owner decision 2026-11-21: in the first pilot the cs.AI feed
                     alone used the whole budget and cs.LO got nothing). That limit is split FAIRLY across the
                     approved categories: `plan_category_feed_shares` water-fills it over what each category can use
                     today, so a category that needs less leaves its share to the others. Until the shares are
                     planned (the feed stage plans them each run), the limit applies to the feeds as a whole.
  * keyword routes   recent OpenAlex windows may use what remains after the part of the category-feed limit that the
                     feeds have not used yet today, and after the unused historical reserve.
  * historical       `historical_reserved_candidates` per day are kept for the historical search while some topic's
                     plan still has work (`historical_reserve_applies`, set by `collect`). Owner decision 2026-11-21:
                     in the first pilot the current releases used the whole budget, so the history got 0 on every
                     run. Beyond its reserve, the history may use what remains after both current-release
                     reservations: the category-feed one and `current_release_reserved_candidates` (the feeds count
                     toward the latter too, so the larger outstanding one is held back).
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from paperwatch.clock import format_utc_timestamp
from paperwatch.config import CollectionSettings
from paperwatch.sources.discovery_routes import RouteKind, arxiv_category_of_route, classify_discovery_route

UNLIMITED_ADMISSIONS = sys.maxsize


@dataclass
class CandidateAdmissionBudget:
    daily_limit: int
    reserved_for_current_releases: int
    category_feed_limit: int = UNLIMITED_ADMISSIONS
    historical_reserve: int = 0
    historical_reserve_applies: bool = False  # True while a historical plan still has work (set by `collect`)
    admitted_today_by_current_releases: int = 0  # every current route, the category feeds included
    admitted_today_by_historical_search: int = 0
    admitted_today_by_category: dict[str, int] = field(default_factory=dict)  # feeds + recovery, per category
    # Each category's target for the whole UTC day (planned by the feed stage); None until planned.
    category_daily_targets: dict[str, int] | None = None

    @classmethod
    def load_for_day(
        cls, connection: sqlite3.Connection, collection_settings: CollectionSettings, moment: datetime
    ) -> CandidateAdmissionBudget:
        day_start_timestamp = format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))
        budget = cls(
            daily_limit=collection_settings.daily_new_candidate_budget,
            reserved_for_current_releases=collection_settings.current_release_reserved_candidates,
            category_feed_limit=collection_settings.category_feed_candidate_limit,
            historical_reserve=collection_settings.historical_reserved_candidates,
        )
        for paper_row in connection.execute(
            "SELECT origin_route FROM papers WHERE discovered_at >= ?", (day_start_timestamp,)
        ):
            budget.record_route_admissions(paper_row["origin_route"], 1)
        return budget

    # ---- what was admitted --------------------------------------------------------------------------------------

    @property
    def admitted_today(self) -> int:
        return self.admitted_today_by_current_releases + self.admitted_today_by_historical_search

    @property
    def admitted_today_by_category_feeds(self) -> int:
        return sum(self.admitted_today_by_category.values())

    def record_route_admissions(self, discovery_route: str, admitted_count: int) -> None:
        self.record_admissions(classify_discovery_route(discovery_route), admitted_count)
        arxiv_category = arxiv_category_of_route(discovery_route)
        if arxiv_category is not None:
            self.admitted_today_by_category[arxiv_category] = (
                self.admitted_today_by_category.get(arxiv_category, 0) + admitted_count
            )

    def record_admissions(self, route_kind: RouteKind, admitted_count: int) -> None:
        if route_kind == RouteKind.CURRENT_RELEASE:
            self.admitted_today_by_current_releases += admitted_count
        elif route_kind == RouteKind.HISTORICAL_SEARCH:
            self.admitted_today_by_historical_search += admitted_count

    # ---- what may still be admitted -----------------------------------------------------------------------------

    @property
    def remaining_today(self) -> int:
        return max(0, self.daily_limit - self.admitted_today)

    @property
    def unused_historical_reserve(self) -> int:
        if not self.historical_reserve_applies:
            return 0
        return max(0, min(self.historical_reserve, self.daily_limit) - self.admitted_today_by_historical_search)

    @property
    def effective_category_feed_limit(self) -> int:
        return min(self.category_feed_limit, self.daily_limit)

    @property
    def unused_category_feed_reserve(self) -> int:
        return max(0, self.effective_category_feed_limit - self.admitted_today_by_category_feeds)

    def remaining_for_route(self, discovery_route: str) -> int:
        """How many new papers `discovery_route` may still admit today."""
        arxiv_category = arxiv_category_of_route(discovery_route)
        if arxiv_category is not None:
            return min(
                max(0, self.remaining_today - self.unused_historical_reserve),
                self.remaining_share_of_category(arxiv_category),
            )
        return self.remaining_for(classify_discovery_route(discovery_route))

    def remaining_for(self, route_kind: RouteKind) -> int:
        """For a route that is NOT a category feed (those use `remaining_for_route`)."""
        if route_kind == RouteKind.CURRENT_RELEASE:
            return max(0, self.remaining_today - self.unused_category_feed_reserve - self.unused_historical_reserve)
        if route_kind == RouteKind.HISTORICAL_SEARCH:
            unused_current_reserve = max(
                0, self.reserved_for_current_releases - self.admitted_today_by_current_releases
            )
            beyond_its_reserve = self.remaining_today - max(unused_current_reserve, self.unused_category_feed_reserve)
            return max(0, min(self.remaining_today, self.unused_historical_reserve), beyond_its_reserve)
        return UNLIMITED_ADMISSIONS  # seeds and fixtures are not provider candidates

    def remaining_share_of_category(self, arxiv_category: str) -> int:
        if self.category_daily_targets is None:
            return self.unused_category_feed_reserve
        return max(
            0,
            min(
                self.category_daily_targets.get(arxiv_category, 0)
                - self.admitted_today_by_category.get(arxiv_category, 0),
                self.unused_category_feed_reserve,
            ),
        )

    def plan_category_feed_shares(self, new_records_waiting_by_category: Mapping[str, int | None]) -> None:
        """Split the category-feed limit fairly for today. A category's demand is what it already admitted today
        plus the new records waiting for it now (None = unknown, e.g. an open recovery search: it takes whatever
        share it gets)."""
        demands = {
            arxiv_category: None
            if waiting is None
            else self.admitted_today_by_category.get(arxiv_category, 0) + waiting
            for arxiv_category, waiting in new_records_waiting_by_category.items()
        }
        self.category_daily_targets = water_fill(self.effective_category_feed_limit, demands)

    def describe(self) -> str:
        description = (
            f"{self.admitted_today} of {self.daily_limit} new candidate(s) admitted today (UTC): "
            f"{self.admitted_today_by_current_releases} current, {self.admitted_today_by_historical_search} "
            "historical"
        )
        if self.category_feed_limit < UNLIMITED_ADMISSIONS:
            per_category = ", ".join(
                f"{arxiv_category} {count}" for arxiv_category, count in sorted(self.admitted_today_by_category.items())
            )
            description += (
                f"; arXiv category feeds {self.admitted_today_by_category_feeds} of "
                f"{self.category_feed_limit}" + (f" ({per_category})" if per_category else "")
            )
        return description


def water_fill(limit: int, demands: Mapping[str, int | None]) -> dict[str, int]:
    """Split `limit` as evenly as possible; nobody gets more than their demand (None = no bound), and what a small
    demand leaves over goes to the others. Integer shares; leftover units go to the first names in order."""
    targets = {name: 0 for name in demands}
    unsatisfied = [name for name, demand in demands.items() if demand is None or demand > 0]
    remaining = max(0, limit)
    while unsatisfied and remaining > 0:
        equal_share = remaining // len(unsatisfied)
        if equal_share == 0:
            for name in unsatisfied[:remaining]:
                targets[name] += 1
            break
        still_unsatisfied = []
        for name in unsatisfied:
            demand = demands[name]
            grant = equal_share if demand is None else min(equal_share, demand - targets[name])
            targets[name] += grant
            remaining -= grant
            if demand is None or targets[name] < demand:
                still_unsatisfied.append(name)
        unsatisfied = still_unsatisfied
    return targets
