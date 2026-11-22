"""Gap searches chosen by the research role (plan part 8): "select an unused approved query or year band when reviews
expose a gap. Record the gap, selected branch, parent review, and returned candidates."

A branch is one unit of a topic's historical search plan (paperwatch.collection.historical_search): one APPROVED
catalog query over one recency window. A unit is unused when it has no checkpoint yet, or its checkpoint has admitted
nothing and can still continue. So the model can only point at a search that already exists in the approved plan; it
never writes a query, and the host checks the branch id against the list it offered.
  * Offered (worker): the unused units of the topics the parent review addresses, oldest windows first, at most
    references.branches_offered_per_selection, and none once references.research_branches_per_day were requested
    today (UTC). A unit requested before is never offered again (no repeated searches).
  * Run (collector, `collect`): one page of each requested branch through the unit's own checkpoint, under the route
    research-branch:<query id>. It counts against the day's historical allowance like the historical search; the
    returned candidates are recorded on the branch and then screened like every other candidate.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from paperwatch.clock import format_utc_timestamp, utc_now_timestamp
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.historical_search import (
    HISTORY_WINDOW_CHECKPOINT_PREFIX,
    HistoricalSearchUnit,
    build_search_units,
    historical_queries_for_topic,
    history_plan_key,
    plan_windows,
    recency_schedule_settings,
)
from paperwatch.collection.paged_search import SearchTarget, is_search_resumable, run_search_pages, search_parameters
from paperwatch.config import AppConfig
from paperwatch.db import write_transaction
from paperwatch.errors import SourceAccessDenied
from paperwatch.network.provider_requests import PublicationWindow, SearchResultOrder
from paperwatch.public_query_catalog import build_public_query_catalog, find_public_query
from paperwatch.source_checkpoints import SourceCheckpoint, load_source_checkpoint
from paperwatch.sources.discovery_routes import RouteKind, research_branch_route

BRANCH_ID_PREFIX = "B"
BRANCH_STATUS_REQUESTED = "requested"
BRANCH_STATUS_DONE = "done"
BRANCH_STATUS_BLOCKED = "blocked"  # the query left the approved catalog; the branch cannot run
MAXIMUM_STORED_TEXT_CHARACTERS = 600


@dataclass(frozen=True)
class ResearchBranch:
    branch_id: str  # short id shown to the model, e.g. "B3"
    topic_id: str
    query_id: str
    provider_name: str
    public_value: str  # the approved public phrase or category: exactly what may be sent
    window_first_date: date
    window_last_date: date
    result_order: SearchResultOrder

    @property
    def branch_key(self) -> str:
        return f"{self.query_id}@{self.window_first_date.isoformat()}..{self.window_last_date.isoformat()}"

    def as_model_input(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "topic_id": self.topic_id,
            "provider": self.provider_name,
            "search_terms": self.public_value,
            "publication_years": f"{self.window_first_date.year}-{self.window_last_date.year}",
        }


def count_branches_requested_today(connection: sqlite3.Connection, moment: datetime) -> int:
    day_start = format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))
    return connection.execute(
        "SELECT COUNT(*) FROM research_branches WHERE requested_at >= ?", (day_start,)
    ).fetchone()[0]


def offer_unused_branches(
    connection: sqlite3.Connection, app_config: AppConfig, topic_ids: list[str], moment: datetime
) -> list[ResearchBranch]:
    reference_settings = app_config.references
    if count_branches_requested_today(connection, moment) >= reference_settings.research_branches_per_day:
        return []
    requested_branch_keys = {
        row["branch_key"] for row in connection.execute("SELECT branch_key FROM research_branches")
    }
    query_catalog = build_public_query_catalog(app_config.topics)
    schedule_settings = recency_schedule_settings(app_config.collection)
    unused_units: list[HistoricalSearchUnit] = []
    for topic_id in topic_ids:
        historical_queries = historical_queries_for_topic(topic_id, query_catalog)
        plan_checkpoint = load_source_checkpoint(
            connection, history_plan_key(topic_id, historical_queries, schedule_settings)
        )
        if plan_checkpoint is None:
            continue  # the topic's historical search has not started; nothing to extend yet
        for unit in build_search_units(historical_queries, plan_windows(plan_checkpoint)):
            unit_checkpoint = load_source_checkpoint(connection, unit.checkpoint_key)
            is_unused = unit_checkpoint is None or (
                unit_checkpoint.candidates_admitted == 0 and is_search_resumable(unit_checkpoint)
            )
            if is_unused and _branch_key_of_unit(unit) not in requested_branch_keys:
                unused_units.append(unit)
    unused_units.sort(key=lambda unit: (-unit.window_position, unit.public_query.query_id))  # oldest windows first
    return [
        ResearchBranch(
            branch_id=f"{BRANCH_ID_PREFIX}{position}",
            topic_id=unit.public_query.topic_id,
            query_id=unit.public_query.query_id,
            provider_name=unit.public_query.provider_name,
            public_value=unit.public_query.public_value,
            window_first_date=unit.window.first_date,
            window_last_date=unit.window.last_date,
            result_order=unit.window.result_order,
        )
        for position, unit in enumerate(unused_units[: reference_settings.branches_offered_per_selection], start=1)
    ]


def _branch_key_of_unit(unit: HistoricalSearchUnit) -> str:
    return f"{unit.public_query.query_id}@{unit.window.first_date.isoformat()}..{unit.window.last_date.isoformat()}"


def record_branch_request(
    connection: sqlite3.Connection, branch: ResearchBranch, *, parent_review_id: str, gap_statement: str
) -> bool:
    """Store the chosen branch once (INSERT OR IGNORE on its key). Call inside the job's finish transaction."""
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO research_branches (branch_key, topic_id, query_id, window_first_date,
               window_last_date, result_order, parent_review_id, gap_statement, branch_status, requested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            branch.branch_key,
            branch.topic_id,
            branch.query_id,
            branch.window_first_date.isoformat(),
            branch.window_last_date.isoformat(),
            str(branch.result_order),
            parent_review_id,
            gap_statement[:MAXIMUM_STORED_TEXT_CHARACTERS],
            BRANCH_STATUS_REQUESTED,
            utc_now_timestamp(),
        ),
    )
    return insert_cursor.rowcount == 1


# --------------------------------------------------------------------------------------------------------------------
# Collector: run the requested branches
# --------------------------------------------------------------------------------------------------------------------


def run_requested_research_branches(context: CollectionContext) -> list[str]:
    connection = context.connection
    requested_rows = connection.execute(
        "SELECT * FROM research_branches WHERE branch_status = ? ORDER BY requested_at, branch_key",
        (BRANCH_STATUS_REQUESTED,),
    ).fetchall()
    if not requested_rows:
        return []
    query_catalog = build_public_query_catalog(context.app_config.topics)
    report_lines = []
    for branch_row in requested_rows:
        try:
            public_query = find_public_query(query_catalog, branch_row["query_id"])
        except SourceAccessDenied:
            _settle_branch(
                connection,
                branch_row["branch_key"],
                BRANCH_STATUS_BLOCKED,
                stop_reason="its query is no longer in the approved catalog (topics.toml changed)",
            )
            report_lines.append(f"Research branch {branch_row['branch_key']}: blocked (query no longer approved)")
            continue
        target = SearchTarget(
            public_query.provider_name,
            public_query.query_id,
            PublicationWindow(
                date.fromisoformat(branch_row["window_first_date"]), date.fromisoformat(branch_row["window_last_date"])
            ),
            research_branch_route(public_query.query_id),
            RouteKind.HISTORICAL_SEARCH,
            SearchResultOrder(branch_row["result_order"]),
        )
        unit_checkpoint_key = (
            f"{HISTORY_WINDOW_CHECKPOINT_PREFIX}{public_query.query_id}:"
            f"{branch_row['window_first_date']}..{branch_row['window_last_date']}"
        )
        unit_checkpoint = load_source_checkpoint(connection, unit_checkpoint_key) or SourceCheckpoint(
            search_key=unit_checkpoint_key,
            source_name=public_query.provider_name,
            topic_id=branch_row["topic_id"],
            query_parameters=search_parameters(target, public_query.public_value),
        )
        summary = run_search_pages(context, unit_checkpoint, target, maximum_pages=1)
        final_checkpoint = summary.final_checkpoint
        if summary.pages_fetched == 0 and final_checkpoint is not None and is_search_resumable(final_checkpoint):
            # Blocked for now (outage, request budget) or no daily allowance left: it stays requested for next run.
            report_lines.append(
                f"Research branch {branch_row['branch_key']}: waiting "
                f"({final_checkpoint.stop_reason or final_checkpoint.coverage_status})"
            )
            continue
        returned_paper_ids = list(dict.fromkeys(summary.ingestion.catalogued_paper_ids))
        _settle_branch(
            connection,
            branch_row["branch_key"],
            BRANCH_STATUS_DONE,
            stop_reason=final_checkpoint.stop_reason if final_checkpoint else None,
            returned_record_count=summary.ingestion.records_received,
            new_paper_count=summary.ingestion.new_papers_admitted,
            returned_paper_ids=returned_paper_ids,
        )
        report_lines.append(
            f"Research branch {branch_row['branch_key']}: {summary.ingestion.records_received} "
            f"record(s), {summary.ingestion.new_papers_admitted} new candidate(s)"
        )
    return report_lines


def _settle_branch(
    connection: sqlite3.Connection,
    branch_key: str,
    branch_status: str,
    *,
    stop_reason: str | None,
    returned_record_count: int = 0,
    new_paper_count: int = 0,
    returned_paper_ids: list[str] | None = None,
) -> None:
    with write_transaction(connection):
        connection.execute(
            """UPDATE research_branches SET branch_status = ?, stop_reason = ?, returned_record_count = ?,
                   new_paper_count = ?, returned_paper_ids_json = ?, completed_at = ? WHERE branch_key = ?""",
            (
                branch_status,
                (stop_reason or "")[:MAXIMUM_STORED_TEXT_CHARACTERS] or None,
                returned_record_count,
                new_paper_count,
                json.dumps(returned_paper_ids or []),
                utc_now_timestamp(),
                branch_key,
            ),
        )
