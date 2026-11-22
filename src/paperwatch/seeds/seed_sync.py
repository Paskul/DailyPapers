"""Synchronize every configured seed source: Zotero collections (live Web API) and CSL JSON files (offline).

Both kinds follow the same contract:
  * read the WHOLE source first, then commit its new seed set and checkpoint in one transaction
    (seed_store.commit_seed_snapshot); an interrupted or failed read changes nothing;
  * a source that cannot be read (no Zotero key, provider outage, missing or invalid file) is marked blocked with its
    reason, and its last successful seed set stays in use; other sources and all topics keep processing;
  * an unchanged source costs almost nothing: Zotero answers 304 to If-Modified-Since-Version, and a CSL file whose
    SHA-256 matches the last import is skipped.

Zotero sync uses library versions. Every page of one sync must report the same Last-Modified-Version; if the library
changes mid-sync the sync is retried once, then left for the next run. Because each sync lists the collection's
complete current membership, deleted items, trashed items, and items moved to another collection simply drop out
and are marked removed. The collection's items are requested with only its key, the library id, and paging values.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Literal, overload

from paperwatch.clock import utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.db import write_transaction
from paperwatch.errors import (
    PaperwatchError,
    ProviderUnavailable,
    SourceAccessDenied,
    SourceInputError,
    SourceResponseRejected,
)
from paperwatch.network.provider_gateway import ProviderGateway, ProviderResponse
from paperwatch.seeds.csl_json_items import parse_csl_json_export
from paperwatch.seeds.seed_items import SEED_SOURCE_ZOTERO, SeedItem
from paperwatch.seeds.seed_store import (
    SeedSnapshotChanges,
    SeedSourceReference,
    catalog_seed_items,
    commit_seed_snapshot,
    count_active_seeds_by_source,
    load_or_start_seed_checkpoint,
    record_seed_source_unavailable,
    seed_sources_of_all_topics,
)
from paperwatch.seeds.zotero_items import parse_zotero_items_page
from paperwatch.source_checkpoints import SearchCoverageStatus, save_source_checkpoint

MAXIMUM_SEED_ITEMS_PER_ZOTERO_COLLECTION = 10_000
MAXIMUM_CSL_JSON_FILE_BYTES = 50 * 1024 * 1024
RETRIES_WHEN_LIBRARY_CHANGES_DURING_SYNC = 1
ZOTERO_LIBRARY_VERSION_HEADER = "last-modified-version"
ZOTERO_TOTAL_RESULTS_HEADER = "total-results"


class SeedSyncResult(StrEnum):
    SYNCED = "synced"  # a new complete snapshot was committed
    UNCHANGED = "unchanged"  # the source has not changed since the last complete sync
    UNAVAILABLE = "unavailable"  # not read this time; the last successful seed set stays in use


@dataclass(frozen=True)
class SeedSyncOutcome:
    seed_source_reference: SeedSourceReference
    result: SeedSyncResult
    active_seed_count: int
    title_only_seed_count: int
    detail: str  # e.g. "library version 1234" or the reason the source was unavailable
    changes: SeedSnapshotChanges | None = None
    skipped_item_counts: Counter[str] | None = None
    operator_hint: str | None = None

    def describe(self) -> str:
        label = self.seed_source_reference.display_label
        seed_counts = f"{self.active_seed_count} active seed(s), {self.title_only_seed_count} title-only"
        if self.result == SeedSyncResult.UNAVAILABLE:
            description = f"{label}: NOT synced ({self.detail}); using the last successful seed set: {seed_counts}"
            return description + (f"\n  hint: {self.operator_hint}" if self.operator_hint else "")
        if self.result == SeedSyncResult.UNCHANGED:
            return f"{label}: unchanged ({self.detail}); {seed_counts}"
        changes = self.changes or SeedSnapshotChanges()
        description = (
            f"{label}: synced ({self.detail}); {seed_counts}; {changes.added} added, {changes.updated} "
            f"updated, {changes.removed} removed"
        )
        skipped_parts = [f"{count} {reason}" for reason, count in sorted((self.skipped_item_counts or {}).items())]
        if changes.not_catalogued:
            skipped_parts.append(f"{changes.not_catalogued} not catalogued")
        return description + (f"; skipped: {', '.join(skipped_parts)}" if skipped_parts else "")


class LibraryChangedDuringSync(SourceResponseRejected):
    """Two pages of one Zotero sync reported different library versions."""


def synchronize_all_seed_sources(
    connection: sqlite3.Connection, app_config: AppConfig, *, build_gateway: Callable[[], ProviderGateway]
) -> list[SeedSyncOutcome]:
    """Sync each configured source once. The gateway (and the credentials file) is loaded only if Zotero is used."""
    gateway: ProviderGateway | None = None
    outcomes: list[SeedSyncOutcome] = []
    for seed_source_reference in seed_sources_of_all_topics(app_config.topics):
        if seed_source_reference.seed_source == SEED_SOURCE_ZOTERO:
            gateway = gateway or build_gateway()
            outcomes.append(synchronize_zotero_collection(connection, gateway, seed_source_reference))
        else:
            outcomes.append(import_csl_json_seed_file(connection, app_config, seed_source_reference))
    return outcomes


# --------------------------------------------------------------------------------------------------------------------
# Zotero (live Web API)
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchedZoteroCollection:
    seed_items: list[SeedItem]
    library_version: int
    skipped_item_counts: Counter[str]


def synchronize_zotero_collection(
    connection: sqlite3.Connection, gateway: ProviderGateway, seed_source_reference: SeedSourceReference
) -> SeedSyncOutcome:
    checkpoint = load_or_start_seed_checkpoint(connection, seed_source_reference)
    known_library_version = int(checkpoint.cursor) if checkpoint.cursor and checkpoint.cursor.isdigit() else None
    try:
        fetched_collection = _fetch_with_retry_on_library_change(
            gateway, seed_source_reference.collection_key, known_library_version
        )
    except (SourceAccessDenied, ProviderUnavailable, SourceInputError) as problem:
        return _mark_unavailable(connection, seed_source_reference, problem)

    if fetched_collection is None:  # 304: nothing in the library changed since the last complete sync
        _record_unchanged(connection, seed_source_reference)
        return _outcome_from_cache(
            connection, seed_source_reference, SeedSyncResult.UNCHANGED, f"library version {known_library_version}"
        )

    paper_id_by_item_key, _ = catalog_seed_items(connection, seed_source_reference, fetched_collection.seed_items)
    completed_checkpoint = replace(
        checkpoint,
        cursor=str(fetched_collection.library_version),
        coverage_status=SearchCoverageStatus.EXHAUSTED,
        stop_reason="complete snapshot of the collection",
    )
    changes = commit_seed_snapshot(
        connection, seed_source_reference, fetched_collection.seed_items, paper_id_by_item_key, completed_checkpoint
    )
    return SeedSyncOutcome(
        seed_source_reference,
        SeedSyncResult.SYNCED,
        changes.active,
        changes.title_only,
        f"library version {fetched_collection.library_version}",
        changes,
        fetched_collection.skipped_item_counts,
    )


def _fetch_with_retry_on_library_change(
    gateway: ProviderGateway, collection_key: str, known_library_version: int | None
) -> FetchedZoteroCollection | None:
    for attempt_number in range(RETRIES_WHEN_LIBRARY_CHANGES_DURING_SYNC + 1):
        try:
            return fetch_complete_zotero_collection(gateway, collection_key, known_library_version)
        except LibraryChangedDuringSync:
            if attempt_number == RETRIES_WHEN_LIBRARY_CHANGES_DURING_SYNC:
                raise
    return None  # unreachable


def fetch_complete_zotero_collection(
    gateway: ProviderGateway, collection_key: str, known_library_version: int | None
) -> FetchedZoteroCollection | None:
    """Every top-level item of the collection, all at one library version. None if unchanged since
    `known_library_version`. No database writes happen here (the outbound ledger aside)."""
    seed_items_by_key: dict[str, SeedItem] = {}
    skipped_item_counts: Counter[str] = Counter()
    snapshot_library_version: int | None = None
    next_start = 0
    while True:
        response = gateway.list_seed_items(
            collection_key,
            next_start,
            if_modified_since_library_version=known_library_version if next_start == 0 else None,
        )
        if response.was_not_modified:
            return None
        page_library_version = _integer_header(response, ZOTERO_LIBRARY_VERSION_HEADER, required=True)
        if snapshot_library_version is None:
            snapshot_library_version = page_library_version
        elif page_library_version != snapshot_library_version:
            raise LibraryChangedDuringSync(
                f"zotero: the library changed during the sync (version "
                f"{snapshot_library_version} -> {page_library_version})"
            )
        parsed_page = parse_zotero_items_page(response.body_bytes)
        for seed_item in parsed_page.seed_items:
            seed_items_by_key[seed_item.seed_item_key] = seed_item
        skipped_item_counts.update(parsed_page.skipped_item_counts)
        next_start += parsed_page.raw_item_count
        total_results = _integer_header(response, ZOTERO_TOTAL_RESULTS_HEADER, required=False)
        if parsed_page.raw_item_count == 0 or (total_results is not None and next_start >= total_results):
            return FetchedZoteroCollection(
                list(seed_items_by_key.values()),
                page_library_version,
                skipped_item_counts,  # == the snapshot's
            )
        if next_start >= MAXIMUM_SEED_ITEMS_PER_ZOTERO_COLLECTION:
            raise SourceResponseRejected(
                f"zotero: collection {collection_key} has more than "
                f"{MAXIMUM_SEED_ITEMS_PER_ZOTERO_COLLECTION} items; use a smaller collection"
            )


@overload
def _integer_header(response: ProviderResponse, header_name: str, *, required: Literal[True]) -> int: ...
@overload
def _integer_header(response: ProviderResponse, header_name: str, *, required: Literal[False]) -> int | None: ...
def _integer_header(response: ProviderResponse, header_name: str, *, required: bool) -> int | None:
    header_value = response.response_headers.get(header_name)
    if header_value is not None and header_value.strip().isdigit():
        return int(header_value.strip())
    if required:
        raise SourceResponseRejected(f"zotero: response lacks a valid {header_name} header")
    return None


# --------------------------------------------------------------------------------------------------------------------
# CSL JSON files (offline export)
# --------------------------------------------------------------------------------------------------------------------


def import_csl_json_seed_file(
    connection: sqlite3.Connection, app_config: AppConfig, seed_source_reference: SeedSourceReference
) -> SeedSyncOutcome:
    checkpoint = load_or_start_seed_checkpoint(connection, seed_source_reference)
    seed_file_path = app_config.resolve_seed_file_path(seed_source_reference.configured_path or "")
    try:
        file_bytes = _read_bounded_seed_file(seed_file_path)
        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        if checkpoint.cursor == file_sha256 and checkpoint.coverage_status == SearchCoverageStatus.EXHAUSTED:
            _record_unchanged(connection, seed_source_reference)
            return _outcome_from_cache(
                connection, seed_source_reference, SeedSyncResult.UNCHANGED, "same file contents as the last import"
            )
        parsed_export = parse_csl_json_export(file_bytes, seed_source_reference.collection_key)
    except SourceInputError as problem:
        return _mark_unavailable(connection, seed_source_reference, problem)

    paper_id_by_item_key, _ = catalog_seed_items(connection, seed_source_reference, parsed_export.seed_items)
    completed_checkpoint = replace(
        checkpoint,
        cursor=file_sha256,
        coverage_status=SearchCoverageStatus.EXHAUSTED,
        stop_reason="complete import of the file",
    )
    changes = commit_seed_snapshot(
        connection, seed_source_reference, parsed_export.seed_items, paper_id_by_item_key, completed_checkpoint
    )
    return SeedSyncOutcome(
        seed_source_reference,
        SeedSyncResult.SYNCED,
        changes.active,
        changes.title_only,
        f"{parsed_export.raw_item_count} item(s) in the file",
        changes,
        parsed_export.skipped_item_counts,
    )


def _read_bounded_seed_file(seed_file_path: Path) -> bytes:
    try:
        with seed_file_path.open("rb") as seed_file:
            file_bytes = seed_file.read(MAXIMUM_CSL_JSON_FILE_BYTES + 1)
    except FileNotFoundError:
        raise SourceInputError(f"file {seed_file_path} does not exist") from None
    except OSError as read_error:
        raise SourceInputError(f"cannot read {seed_file_path}: {read_error.strerror or read_error}") from None
    if len(file_bytes) > MAXIMUM_CSL_JSON_FILE_BYTES:
        raise SourceInputError(f"{seed_file_path} is larger than {MAXIMUM_CSL_JSON_FILE_BYTES // (1024 * 1024)} MB")
    return file_bytes


# --------------------------------------------------------------------------------------------------------------------
# Shared outcome helpers
# --------------------------------------------------------------------------------------------------------------------


def _mark_unavailable(
    connection: sqlite3.Connection, seed_source_reference: SeedSourceReference, problem: PaperwatchError
) -> SeedSyncOutcome:
    record_seed_source_unavailable(connection, seed_source_reference, str(problem))
    return _outcome_from_cache(
        connection, seed_source_reference, SeedSyncResult.UNAVAILABLE, str(problem), operator_hint=problem.operator_hint
    )


def _record_unchanged(connection: sqlite3.Connection, seed_source_reference: SeedSourceReference) -> None:
    """The cached seed set is confirmed current: coverage complete, and the check time recorded."""
    confirmed_checkpoint = replace(
        load_or_start_seed_checkpoint(connection, seed_source_reference),
        coverage_status=SearchCoverageStatus.EXHAUSTED,
        stop_reason="unchanged since the last complete sync",
        last_completed_interval_end=utc_now_timestamp(),
    )
    with write_transaction(connection):
        save_source_checkpoint(connection, confirmed_checkpoint)


def _outcome_from_cache(
    connection: sqlite3.Connection,
    seed_source_reference: SeedSourceReference,
    result: SeedSyncResult,
    detail: str,
    *,
    operator_hint: str | None = None,
) -> SeedSyncOutcome:
    active_count, title_only_count = count_active_seeds_by_source(connection, seed_source_reference)
    return SeedSyncOutcome(
        seed_source_reference, result, active_count, title_only_count, detail, operator_hint=operator_hint
    )
