"""The daily budget of new metadata candidates (plan part 5: start with 500 per day, current releases first).

A candidate is a paper the catalog did not know before, found by an arXiv or OpenAlex route (seeds and fixtures do not
count). The count is derived from the catalog itself (papers.origin_route and papers.discovered_at for the current
UTC day), so it survives restarts and is shared by concurrent collectors without another table.
Current releases may use the whole remaining budget. The historical search may use only what remains after the part
of `current_release_reserved_candidates` that current releases have not used yet today.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime

from paperwatch.clock import format_utc_timestamp
from paperwatch.config import CollectionSettings
from paperwatch.sources.discovery_routes import RouteKind, classify_discovery_route

UNLIMITED_ADMISSIONS = sys.maxsize


@dataclass
class CandidateAdmissionBudget:
    daily_limit: int
    reserved_for_current_releases: int
    admitted_today_by_current_releases: int = 0
    admitted_today_by_historical_search: int = 0

    @classmethod
    def load_for_day(cls, connection: sqlite3.Connection, collection_settings: CollectionSettings,
                     moment: datetime) -> CandidateAdmissionBudget:
        day_start_timestamp = format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))
        budget = cls(daily_limit=collection_settings.daily_new_candidate_budget,
                     reserved_for_current_releases=collection_settings.current_release_reserved_candidates)
        for paper_row in connection.execute("SELECT origin_route FROM papers WHERE discovered_at >= ?",
                                            (day_start_timestamp,)):
            budget.record_admissions(classify_discovery_route(paper_row["origin_route"]), 1)
        return budget

    @property
    def admitted_today(self) -> int:
        return self.admitted_today_by_current_releases + self.admitted_today_by_historical_search

    def remaining_for(self, route_kind: RouteKind) -> int:
        remaining_today = max(0, self.daily_limit - self.admitted_today)
        if route_kind == RouteKind.CURRENT_RELEASE:
            return remaining_today
        if route_kind == RouteKind.HISTORICAL_SEARCH:
            unused_reserve = max(0, self.reserved_for_current_releases - self.admitted_today_by_current_releases)
            return max(0, remaining_today - unused_reserve)
        return UNLIMITED_ADMISSIONS  # seeds and fixtures are not provider candidates

    def record_admissions(self, route_kind: RouteKind, admitted_count: int) -> None:
        if route_kind == RouteKind.CURRENT_RELEASE:
            self.admitted_today_by_current_releases += admitted_count
        elif route_kind == RouteKind.HISTORICAL_SEARCH:
            self.admitted_today_by_historical_search += admitted_count

    def describe(self) -> str:
        return (f"{self.admitted_today} of {self.daily_limit} new candidate(s) admitted today (UTC): "
                f"{self.admitted_today_by_current_releases} current, {self.admitted_today_by_historical_search} "
                "historical")
