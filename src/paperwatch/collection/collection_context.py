"""What every stage of one collection run shares: the database, settings, one gateway, the clock, and the budget."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from paperwatch import clock
from paperwatch.collection.candidate_budget import CandidateAdmissionBudget
from paperwatch.config import AppConfig
from paperwatch.network.provider_gateway import ProviderGateway


@dataclass
class CollectionContext:
    connection: sqlite3.Connection
    app_config: AppConfig
    build_gateway: Callable[[], ProviderGateway]
    started_at: datetime  # one "now" for the whole run: windows, gaps, and interval ends all agree
    candidate_budget: CandidateAdmissionBudget
    active_seed_paper_ids: frozenset[str] = frozenset()
    _gateway: ProviderGateway | None = field(default=None, repr=False)

    @classmethod
    def start(cls, connection: sqlite3.Connection, app_config: AppConfig, build_gateway: Callable[[], ProviderGateway],
              active_seed_paper_ids: frozenset[str] = frozenset()) -> CollectionContext:
        started_at = clock.utc_now()
        return cls(connection=connection, app_config=app_config, build_gateway=build_gateway, started_at=started_at,
                   candidate_budget=CandidateAdmissionBudget.load_for_day(connection, app_config.collection,
                                                                          started_at),
                   active_seed_paper_ids=active_seed_paper_ids)

    @property
    def gateway(self) -> ProviderGateway:
        """Built on first use, so a run that needs no provider never loads the credentials file."""
        if self._gateway is None:
            self._gateway = self.build_gateway()
        return self._gateway

    @property
    def started_at_timestamp(self) -> str:
        return clock.format_utc_timestamp(self.started_at)
