"""One cursor and candidate budget per search, so collection resumes where it stopped (plan parts 2 and 5).

A search is anything the collector pages through: an arXiv category feed, an OpenAlex query for one topic and year
band, a Zotero collection sync. Its coverage status makes gaps visible:
    active     more pages remain and budget is left
    capped     the candidate budget was reached; the cursor is kept for later
    exhausted  the provider has no more results
    blocked    the provider refused or is unaffordable for now (see stop_reason)
Save a checkpoint in the SAME write transaction as the records from that page, so a crash mid-page either keeps
both or neither. `query_parameters` must only contain approved public terms (never a private question).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from paperwatch.clock import utc_now_timestamp


class SearchCoverageStatus(StrEnum):
    ACTIVE = "active"
    CAPPED = "capped"
    EXHAUSTED = "exhausted"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class SourceCheckpoint:
    search_key: str  # stable name, e.g. "arxiv-feed:cs.RO" or "openalex-history:team-safety:q0:2015-2019"
    source_name: str  # "arxiv", "openalex", "zotero", ...
    topic_id: str | None = None
    query_parameters: dict[str, Any] = field(default_factory=dict)
    cursor: str | None = None
    coverage_status: SearchCoverageStatus = SearchCoverageStatus.ACTIVE
    stop_reason: str | None = None
    candidates_admitted: int = 0
    candidate_budget: int | None = None  # None = unlimited
    last_completed_interval_end: str | None = None  # UTC timestamp up to which this search is known complete

    @property
    def remaining_candidate_budget(self) -> int | None:
        if self.candidate_budget is None:
            return None
        return max(0, self.candidate_budget - self.candidates_admitted)

    def after_page(self, *, next_cursor: str | None, newly_admitted_candidates: int) -> SourceCheckpoint:
        """The checkpoint after one completed page: advance the cursor, count candidates, and set coverage."""
        candidates_admitted = self.candidates_admitted + newly_admitted_candidates
        if next_cursor is None:
            coverage_status, stop_reason = SearchCoverageStatus.EXHAUSTED, "provider returned no further page"
        elif self.candidate_budget is not None and candidates_admitted >= self.candidate_budget:
            coverage_status, stop_reason = SearchCoverageStatus.CAPPED, "candidate budget reached"
        else:
            coverage_status, stop_reason = SearchCoverageStatus.ACTIVE, None
        return replace(self, cursor=next_cursor, candidates_admitted=candidates_admitted,
                       coverage_status=coverage_status, stop_reason=stop_reason)


def load_source_checkpoint(connection: sqlite3.Connection, search_key: str) -> SourceCheckpoint | None:
    checkpoint_row = connection.execute("SELECT * FROM source_checkpoints WHERE search_key = ?",
                                        (search_key,)).fetchone()
    if checkpoint_row is None:
        return None
    return SourceCheckpoint(
        search_key=checkpoint_row["search_key"], source_name=checkpoint_row["source_name"],
        topic_id=checkpoint_row["topic_id"], query_parameters=json.loads(checkpoint_row["query_json"]),
        cursor=checkpoint_row["cursor"], coverage_status=SearchCoverageStatus(checkpoint_row["coverage_status"]),
        stop_reason=checkpoint_row["stop_reason"], candidates_admitted=checkpoint_row["candidates_admitted"],
        candidate_budget=checkpoint_row["candidate_budget"],
        last_completed_interval_end=checkpoint_row["last_completed_interval_end"])


def load_source_checkpoints_with_prefix(connection: sqlite3.Connection, search_key_prefix: str,
                                        ) -> list[SourceCheckpoint]:
    """All checkpoints whose key starts with the prefix, e.g. every 'arxiv-recovery:' search, ordered by key."""
    escaped_prefix = search_key_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    key_rows = connection.execute(
        "SELECT search_key FROM source_checkpoints WHERE search_key LIKE ? ESCAPE '\\' ORDER BY search_key",
        (escaped_prefix + "%",)).fetchall()
    return [load_source_checkpoint(connection, key_row["search_key"]) for key_row in key_rows]


def save_source_checkpoint(connection: sqlite3.Connection, checkpoint: SourceCheckpoint) -> None:
    """Insert or replace the checkpoint row. Call inside the transaction that stores the page's records."""
    connection.execute(
        """INSERT INTO source_checkpoints (search_key, source_name, topic_id, query_json, cursor, coverage_status,
               stop_reason, candidates_admitted, candidate_budget, last_completed_interval_end, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(search_key) DO UPDATE SET source_name = excluded.source_name, topic_id = excluded.topic_id,
               query_json = excluded.query_json, cursor = excluded.cursor,
               coverage_status = excluded.coverage_status, stop_reason = excluded.stop_reason,
               candidates_admitted = excluded.candidates_admitted, candidate_budget = excluded.candidate_budget,
               last_completed_interval_end = excluded.last_completed_interval_end, updated_at = excluded.updated_at""",
        (checkpoint.search_key, checkpoint.source_name, checkpoint.topic_id,
         json.dumps(checkpoint.query_parameters, ensure_ascii=False, sort_keys=True), checkpoint.cursor,
         str(checkpoint.coverage_status), checkpoint.stop_reason, checkpoint.candidates_admitted,
         checkpoint.candidate_budget, checkpoint.last_completed_interval_end, utc_now_timestamp()))
