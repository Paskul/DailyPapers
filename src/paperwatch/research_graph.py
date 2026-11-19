"""Citation edges and paper comparisons: the uniqueness rules from plan part 2, used by part 8.

  * A citation edge is directed (citing paper -> cited work) and carries its provenance (which provider or check
    supplied it). The cited work is stored by normalized identifier, so an edge can exist before the cited paper is
    in the catalog ("unresolved"); `resolve_citation_edges` links it once the paper arrives.
  * A comparison is keyed by the question (topic id + question hash) and the SORTED pair of review ids, so the same
    two readings are compared at most once per question, whichever order they are presented in.
Call the write helpers inside a write transaction (the comparison: inside the job's lease-checked finish).
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from paperwatch.catalog import find_paper_id_by_identifier
from paperwatch.clock import utc_now_timestamp
from paperwatch.identifiers import COMPARISON_ID_PREFIX, new_internal_id
from paperwatch.scholarly_identifiers import NormalizedIdentifier


def record_citation_edge(connection: sqlite3.Connection, *, citing_paper_id: str,
                         cited_identifier: NormalizedIdentifier, provenance: str,
                         citation_context: str | None = None) -> bool:
    """Store one directed edge once per provenance. Returns True if it is new."""
    version_free_identifier = cited_identifier.without_version()
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO citation_edges (citing_paper_id, cited_scheme, cited_value, cited_paper_id,
               provenance, citation_context, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (citing_paper_id, version_free_identifier.scheme, version_free_identifier.value,
         find_paper_id_by_identifier(connection, version_free_identifier), provenance, citation_context,
         utc_now_timestamp()))
    return insert_cursor.rowcount == 1


def resolve_citation_edges(connection: sqlite3.Connection) -> int:
    """Link unresolved edges whose cited identifier is now a catalog alias. Returns how many were resolved."""
    resolve_cursor = connection.execute(
        """UPDATE citation_edges SET cited_paper_id = (
               SELECT alias.paper_id FROM paper_aliases AS alias
               WHERE alias.provider = citation_edges.cited_scheme
                 AND alias.provider_paper_id = citation_edges.cited_value)
           WHERE cited_paper_id IS NULL
             AND EXISTS (SELECT 1 FROM paper_aliases AS alias
                         WHERE alias.provider = citation_edges.cited_scheme
                           AND alias.provider_paper_id = citation_edges.cited_value)""")
    return resolve_cursor.rowcount


def build_comparison_key(topic_id: str, question_sha256: str, review_id_a: str, review_id_b: str) -> str:
    """'<topic>:<question hash prefix>:<lower review id>:<higher review id>'. Order of the two reviews is irrelevant."""
    if review_id_a == review_id_b:
        raise ValueError("a comparison needs two different reviews")
    lower_review_id, higher_review_id = sorted((review_id_a, review_id_b))
    return f"{topic_id}:{question_sha256[:16]}:{lower_review_id}:{higher_review_id}"


def record_comparison(connection: sqlite3.Connection, *, topic_id: str, question_sha256: str, review_id_a: str,
                      review_id_b: str, comparison_output: dict[str, Any], role_name: str, prompt_version: str,
                      model_identity: str, server_build: str) -> tuple[str, bool]:
    """Store the comparison once per key. Returns (comparison_id holding this key, was_newly_written)."""
    comparison_key = build_comparison_key(topic_id, question_sha256, review_id_a, review_id_b)
    lower_review_id, higher_review_id = sorted((review_id_a, review_id_b))
    new_comparison_id = new_internal_id(COMPARISON_ID_PREFIX)
    insert_cursor = connection.execute(
        """INSERT OR IGNORE INTO comparisons (comparison_id, comparison_key, topic_id, question_sha256,
               lower_review_id, higher_review_id, comparison_json, role_name, prompt_version, model_identity,
               server_build, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_comparison_id, comparison_key, topic_id, question_sha256, lower_review_id, higher_review_id,
         json.dumps(comparison_output, ensure_ascii=False), role_name, prompt_version, model_identity, server_build,
         utc_now_timestamp()))
    if insert_cursor.rowcount == 1:
        return new_comparison_id, True
    return connection.execute("SELECT comparison_id FROM comparisons WHERE comparison_key = ?",
                              (comparison_key,)).fetchone()["comparison_id"], False
