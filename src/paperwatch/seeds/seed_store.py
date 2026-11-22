"""The seed cache in SQLite: which seeds each configured source holds now, and their last synced metadata.

Tables (see db_migrations.py):
    seed_items               one row per seed: metadata as last synced, the catalog paper it maps to
    paper_seed_memberships   one row per (seed, collection or CSL file); `removed_at` set when it leaves the source
    source_checkpoints       one row per seed source: library version or file hash, last complete sync, coverage

A source's new state is written by `commit_seed_snapshot` in ONE transaction, only after the whole source was read,
so an interrupted sync leaves the previous seed set (the cache) in place. Removed seeds keep their seed_items row as
history; only an active membership in a source that a topic lists makes a seed active.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, replace

from paperwatch.catalog import record_collected_paper
from paperwatch.clock import utc_now_timestamp
from paperwatch.config import TopicDefinition
from paperwatch.db import write_transaction
from paperwatch.errors import SourceInputError
from paperwatch.seeds.seed_items import (
    SEED_SOURCE_CSL_JSON,
    SEED_SOURCE_ZOTERO,
    SeedItem,
    build_csl_json_file_key,
    build_seed_id,
    csl_json_seed_discovery_route,
    zotero_seed_discovery_route,
)
from paperwatch.source_checkpoints import (
    SearchCoverageStatus,
    SourceCheckpoint,
    load_source_checkpoint,
    save_source_checkpoint,
)

ZOTERO_SEED_CHECKPOINT_PREFIX = "zotero-seeds:"
CSL_JSON_SEED_CHECKPOINT_PREFIX = "csl-seeds:"


# --------------------------------------------------------------------------------------------------------------------
# Which seed sources are configured
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedSourceReference:
    """One configured seed source: a Zotero collection, or one CSL JSON file."""

    seed_source: str  # "zotero" | "csl_json"
    collection_key: str  # Zotero collection key, or the CSL file key (hash of its configured path)
    display_label: str  # for operators, e.g. "Zotero collection ABCD2345", "CSL JSON seeds/robots.json"
    configured_path: str | None = None  # CSL JSON only: the path as written in topics.toml

    @property
    def checkpoint_key(self) -> str:
        prefix = (
            ZOTERO_SEED_CHECKPOINT_PREFIX if self.seed_source == SEED_SOURCE_ZOTERO else CSL_JSON_SEED_CHECKPOINT_PREFIX
        )
        return f"{prefix}{self.collection_key}"

    @property
    def discovery_route(self) -> str:
        return (
            zotero_seed_discovery_route(self.collection_key)
            if self.seed_source == SEED_SOURCE_ZOTERO
            else csl_json_seed_discovery_route(self.collection_key)
        )

    @classmethod
    def for_zotero_collection(cls, collection_key: str) -> SeedSourceReference:
        return cls(SEED_SOURCE_ZOTERO, collection_key, f"Zotero collection {collection_key}")

    @classmethod
    def for_csl_json_file(cls, configured_path: str) -> SeedSourceReference:
        return cls(
            SEED_SOURCE_CSL_JSON,
            build_csl_json_file_key(configured_path),
            f"CSL JSON {configured_path}",
            configured_path,
        )


def seed_sources_of_topic(topic: TopicDefinition) -> tuple[SeedSourceReference, ...]:
    return tuple(
        [SeedSourceReference.for_zotero_collection(key) for key in topic.zotero_collections]
        + [SeedSourceReference.for_csl_json_file(path) for path in topic.csl_json_files]
    )


def seed_sources_of_all_topics(topics: Iterable[TopicDefinition]) -> tuple[SeedSourceReference, ...]:
    """Every configured source once, in topic order (a collection shared by two topics is synced once)."""
    unique_sources: dict[tuple[str, str], SeedSourceReference] = {}
    for topic in topics:
        for seed_source_reference in seed_sources_of_topic(topic):
            unique_sources.setdefault(
                (seed_source_reference.seed_source, seed_source_reference.collection_key), seed_source_reference
            )
    return tuple(unique_sources.values())


# --------------------------------------------------------------------------------------------------------------------
# Reading the cache
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredSeed:
    """An active seed as the pipeline uses it: matching text for screening, anchors for reviews."""

    seed_id: str
    seed_source: str
    seed_item_key: str
    paper_id: str
    title: str
    authors: tuple[str, ...]
    abstract: str | None
    publication_year: int | None
    matching_text_kind: str
    metadata_sha256: str
    stated_identifiers: tuple[str, ...]

    @property
    def profile_fingerprint(self) -> str:
        return f"{self.seed_id}:{self.metadata_sha256}"


def load_active_seeds(
    connection: sqlite3.Connection, seed_source_references: Iterable[SeedSourceReference]
) -> list[StoredSeed]:
    """Seeds with an active membership in any of the given sources, de-duplicated, ordered by seed id."""
    seeds_by_id: dict[str, StoredSeed] = {}
    for seed_source_reference in seed_source_references:
        for seed_row in connection.execute(
            """SELECT seed.* FROM paper_seed_memberships AS membership
                   JOIN seed_items AS seed ON seed.seed_source = membership.seed_source
                                          AND seed.seed_item_key = membership.seed_item_key
                   WHERE membership.seed_source = ? AND membership.seed_collection_key = ?
                     AND membership.removed_at IS NULL""",
            (seed_source_reference.seed_source, seed_source_reference.collection_key),
        ):
            stored_seed = _stored_seed_from_row(seed_row)
            seeds_by_id.setdefault(stored_seed.seed_id, stored_seed)
    return [seeds_by_id[seed_id] for seed_id in sorted(seeds_by_id)]


def count_active_seeds_by_source(
    connection: sqlite3.Connection, seed_source_reference: SeedSourceReference
) -> tuple[int, int]:
    """(active seeds, of which title-only) in one source."""
    count_row = connection.execute(
        """SELECT COUNT(*) AS active_count,
                  COALESCE(SUM(seed.matching_text_kind = 'title_only'), 0) AS title_only_count
           FROM paper_seed_memberships AS membership
           JOIN seed_items AS seed ON seed.seed_source = membership.seed_source
                                  AND seed.seed_item_key = membership.seed_item_key
           WHERE membership.seed_source = ? AND membership.seed_collection_key = ? AND membership.removed_at IS NULL""",
        (seed_source_reference.seed_source, seed_source_reference.collection_key),
    ).fetchone()
    return count_row["active_count"], count_row["title_only_count"]


def _stored_seed_from_row(seed_row: sqlite3.Row) -> StoredSeed:
    return StoredSeed(
        seed_id=build_seed_id(seed_row["seed_source"], seed_row["seed_item_key"]),
        seed_source=seed_row["seed_source"],
        seed_item_key=seed_row["seed_item_key"],
        paper_id=seed_row["paper_id"],
        title=seed_row["title"],
        authors=tuple(json.loads(seed_row["authors_json"])),
        abstract=seed_row["abstract"],
        publication_year=seed_row["publication_year"],
        matching_text_kind=seed_row["matching_text_kind"],
        metadata_sha256=seed_row["metadata_sha256"],
        stated_identifiers=tuple(json.loads(seed_row["identifiers_json"])),
    )


# --------------------------------------------------------------------------------------------------------------------
# Writing a complete snapshot of one source
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class SeedSnapshotChanges:
    added: int = 0  # new to this source, or back after a removal
    updated: int = 0  # metadata changed
    removed: int = 0  # left the source (deleted, trashed, moved to another collection, dropped from the file)
    unchanged: int = 0
    not_catalogued: int = 0  # the catalog refused the record (reported, never fatal)
    active: int = 0
    title_only: int = 0


def catalog_seed_items(
    connection: sqlite3.Connection, seed_source_reference: SeedSourceReference, seed_items: list[SeedItem]
) -> tuple[dict[str, str], int]:
    """Catalog new or changed seeds (one short transaction each, idempotent). Returns ({item key: paper id},
    count not catalogued). Unchanged seeds keep their paper without touching the catalog."""
    known_rows = {
        row["seed_item_key"]: row
        for row in connection.execute(
            "SELECT seed_item_key, paper_id, metadata_sha256 FROM seed_items WHERE seed_source = ?",
            (seed_source_reference.seed_source,),
        )
    }
    paper_id_by_item_key: dict[str, str] = {}
    not_catalogued_count = 0
    for seed_item in seed_items:
        known_row = known_rows.get(seed_item.seed_item_key)
        if known_row is not None and known_row["metadata_sha256"] == seed_item.compute_metadata_sha256():
            paper_id_by_item_key[seed_item.seed_item_key] = known_row["paper_id"]
            continue
        try:
            catalog_outcome = record_collected_paper(
                connection, seed_item.to_catalog_record(seed_source_reference.discovery_route)
            )
        except SourceInputError:
            not_catalogued_count += 1
            continue
        paper_id_by_item_key[seed_item.seed_item_key] = catalog_outcome.paper_id
    return paper_id_by_item_key, not_catalogued_count


def commit_seed_snapshot(
    connection: sqlite3.Connection,
    seed_source_reference: SeedSourceReference,
    seed_items: list[SeedItem],
    paper_id_by_item_key: dict[str, str],
    completed_checkpoint: SourceCheckpoint,
) -> SeedSnapshotChanges:
    """Replace the source's seed set with this complete snapshot and save its checkpoint, in one transaction."""
    now_timestamp = utc_now_timestamp()
    changes = SeedSnapshotChanges()
    catalogued_seed_items = [seed_item for seed_item in seed_items if seed_item.seed_item_key in paper_id_by_item_key]
    with write_transaction(connection):
        active_item_keys_before = {
            row["seed_item_key"]
            for row in connection.execute(
                """SELECT seed_item_key FROM paper_seed_memberships
               WHERE seed_source = ? AND seed_collection_key = ? AND removed_at IS NULL""",
                (seed_source_reference.seed_source, seed_source_reference.collection_key),
            )
        }
        for seed_item in catalogued_seed_items:
            paper_id = paper_id_by_item_key[seed_item.seed_item_key]
            metadata_changed = _upsert_seed_item(connection, seed_item, paper_id, now_timestamp)
            _activate_membership(connection, seed_source_reference, seed_item, paper_id, now_timestamp)
            if seed_item.seed_item_key not in active_item_keys_before:
                changes.added += 1
            elif metadata_changed:
                changes.updated += 1
            else:
                changes.unchanged += 1
        current_item_keys = {seed_item.seed_item_key for seed_item in catalogued_seed_items}
        for removed_item_key in sorted(active_item_keys_before - current_item_keys):
            connection.execute(
                """UPDATE paper_seed_memberships SET removed_at = ?
                   WHERE seed_source = ? AND seed_item_key = ? AND seed_collection_key = ?""",
                (
                    now_timestamp,
                    seed_source_reference.seed_source,
                    removed_item_key,
                    seed_source_reference.collection_key,
                ),
            )
            changes.removed += 1
        changes.active = len(catalogued_seed_items)
        changes.title_only = sum(1 for seed_item in catalogued_seed_items if not seed_item.abstract)
        changes.not_catalogued = len(seed_items) - len(catalogued_seed_items)
        save_source_checkpoint(
            connection,
            replace(
                completed_checkpoint, candidates_admitted=changes.active, last_completed_interval_end=now_timestamp
            ),
        )
    return changes


def load_or_start_seed_checkpoint(
    connection: sqlite3.Connection, seed_source_reference: SeedSourceReference
) -> SourceCheckpoint:
    existing_checkpoint = load_source_checkpoint(connection, seed_source_reference.checkpoint_key)
    if existing_checkpoint is not None:
        return existing_checkpoint
    query_parameters = {"collection_key": seed_source_reference.collection_key}
    if seed_source_reference.configured_path is not None:
        query_parameters["path"] = seed_source_reference.configured_path  # local only; never sent anywhere
    return SourceCheckpoint(
        search_key=seed_source_reference.checkpoint_key,
        source_name=seed_source_reference.seed_source,
        query_parameters=query_parameters,
    )


def record_seed_source_unavailable(
    connection: sqlite3.Connection, seed_source_reference: SeedSourceReference, stop_reason: str
) -> None:
    """Mark the source blocked with a reason, keeping its cursor and last seed set (the cache stays usable)."""
    blocked_checkpoint = replace(
        load_or_start_seed_checkpoint(connection, seed_source_reference),
        coverage_status=SearchCoverageStatus.BLOCKED,
        stop_reason=stop_reason[:500],
    )
    with write_transaction(connection):
        save_source_checkpoint(connection, blocked_checkpoint)


def _upsert_seed_item(connection: sqlite3.Connection, seed_item: SeedItem, paper_id: str, now_timestamp: str) -> bool:
    """Insert or refresh the seed's cached metadata. Returns True if the stored metadata changed."""
    metadata_sha256 = seed_item.compute_metadata_sha256()
    previous_row = connection.execute(
        "SELECT metadata_sha256 FROM seed_items WHERE seed_source = ? AND seed_item_key = ?",
        (seed_item.seed_source, seed_item.seed_item_key),
    ).fetchone()
    connection.execute(
        """INSERT INTO seed_items (seed_source, seed_item_key, paper_id, item_type, item_version, title, authors_json,
               abstract, publication_year, identifiers_json, matching_text_kind, metadata_sha256, first_seen_at,
               last_synced_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(seed_source, seed_item_key) DO UPDATE SET
               paper_id = excluded.paper_id, item_type = excluded.item_type, item_version = excluded.item_version,
               title = excluded.title, authors_json = excluded.authors_json, abstract = excluded.abstract,
               publication_year = excluded.publication_year, identifiers_json = excluded.identifiers_json,
               matching_text_kind = excluded.matching_text_kind, metadata_sha256 = excluded.metadata_sha256,
               last_synced_at = excluded.last_synced_at""",
        (
            seed_item.seed_source,
            seed_item.seed_item_key,
            paper_id,
            seed_item.item_type,
            seed_item.item_version,
            seed_item.title,
            json.dumps(list(seed_item.authors), ensure_ascii=False),
            seed_item.abstract,
            seed_item.publication_year,
            json.dumps(list(seed_item.stated_identifiers)),
            seed_item.matching_text_kind,
            metadata_sha256,
            now_timestamp,
            now_timestamp,
        ),
    )
    return previous_row is not None and previous_row["metadata_sha256"] != metadata_sha256


def _activate_membership(
    connection: sqlite3.Connection,
    seed_source_reference: SeedSourceReference,
    seed_item: SeedItem,
    paper_id: str,
    now_timestamp: str,
) -> None:
    connection.execute(
        """INSERT INTO paper_seed_memberships (paper_id, seed_source, seed_item_key, seed_collection_key,
               first_seen_at, removed_at)
           VALUES (?, ?, ?, ?, ?, NULL)
           ON CONFLICT(seed_source, seed_item_key, seed_collection_key) DO UPDATE SET
               paper_id = excluded.paper_id, removed_at = NULL""",
        (paper_id, seed_item.seed_source, seed_item.seed_item_key, seed_source_reference.collection_key, now_timestamp),
    )
