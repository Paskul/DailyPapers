"""The paper catalog: one internal paper per provider identity, plus every route that discovered it.

Step-1 identity rule: a paper is matched only by an exact (provider, provider_paper_id) alias. Part 2 of the plan
adds DOI/arXiv normalization, editions, and trusted cross-provider links. A second sighting never rewrites the
stored record (a changed title must not trigger another reading); it only records the extra discovery route.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from paperwatch.clock import utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.identifiers import PAPER_ID_PREFIX, new_internal_id
from paperwatch.sources.types import CollectedPaperRecord


@dataclass(frozen=True)
class CatalogUpsertOutcome:
    paper_id: str
    was_newly_created: bool


def record_collected_paper(connection: sqlite3.Connection, collected_record: CollectedPaperRecord,
                           ) -> CatalogUpsertOutcome:
    now_timestamp = utc_now_timestamp()
    with write_transaction(connection):
        existing_alias_row = connection.execute(
            "SELECT paper_id FROM paper_aliases WHERE provider = ? AND provider_paper_id = ?",
            (collected_record.provider, collected_record.provider_paper_id)).fetchone()
        if existing_alias_row is not None:
            existing_paper_id = existing_alias_row["paper_id"]
            _record_discovery_route(connection, existing_paper_id, collected_record.discovery_route, now_timestamp)
            return CatalogUpsertOutcome(existing_paper_id, was_newly_created=False)

        new_paper_id = new_internal_id(PAPER_ID_PREFIX)
        connection.execute(
            """INSERT INTO papers (paper_id, title, authors_json, abstract, first_published_date,
                   latest_revision_date, source_version, origin_route, discovered_at, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'waiting_input', ?)""",
            (new_paper_id, collected_record.title, json.dumps(list(collected_record.authors)),
             collected_record.abstract, collected_record.first_published_date,
             collected_record.latest_revision_date, collected_record.source_version,
             collected_record.discovery_route, now_timestamp, now_timestamp))
        connection.execute("INSERT INTO paper_aliases (provider, provider_paper_id, paper_id) VALUES (?, ?, ?)",
                           (collected_record.provider, collected_record.provider_paper_id, new_paper_id))
        _record_discovery_route(connection, new_paper_id, collected_record.discovery_route, now_timestamp)
        return CatalogUpsertOutcome(new_paper_id, was_newly_created=True)


def set_paper_status(connection: sqlite3.Connection, paper_id: str, new_status: str) -> None:
    connection.execute("UPDATE papers SET status = ?, updated_at = ? WHERE paper_id = ?",
                       (new_status, utc_now_timestamp(), paper_id))


def load_paper_row(connection: sqlite3.Connection, paper_id: str) -> sqlite3.Row:
    paper_row = connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if paper_row is None:
        raise LookupError(f"paper {paper_id} not found")
    return paper_row


def load_paper_aliases(connection: sqlite3.Connection, paper_id: str) -> list[tuple[str, str]]:
    return [(row["provider"], row["provider_paper_id"]) for row in connection.execute(
        "SELECT provider, provider_paper_id FROM paper_aliases WHERE paper_id = ? ORDER BY provider", (paper_id,))]


def _record_discovery_route(connection: sqlite3.Connection, paper_id: str, route: str, now_timestamp: str) -> None:
    connection.execute("INSERT OR IGNORE INTO paper_discoveries (paper_id, route, discovered_at) VALUES (?, ?, ?)",
                       (paper_id, route, now_timestamp))
