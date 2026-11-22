"""Bound the source cache's disk use (plan part 10): "Bound disk growth with a source-cache limit. Keep validated
reviews, cited evidence, and report snapshots when removing replaceable downloads."

What lives in state/cache/sources/<edition id>/ (paperwatch.source_cache):
  <hash>.pdf | <hash>.xml   the downloaded (or locally supplied) source bytes: REPLACEABLE. They are needed only to
                            extract text; once extraction succeeded, the reading uses the .txt file.
  <hash>.txt                the extracted text: KEPT. It is the text every full-text review cited (its passages are
                            re-segmented from it for notes, references, and comparisons), and its fingerprint is
                            what decides whether a paper needs another reading.
Reviews, evidence passages, and report snapshots are in SQLite and are never touched here.

When the raw source files together exceed collection.source_cache_limit_megabytes, the oldest raw files are removed
first, but only files whose text was extracted, and never while the paper still has a queued or running reading.
A removed file is only re-downloaded if the paper ever needs a fresh extraction (no step does this automatically).
Runs inside `collect`, i.e. as the collector account, which owns the cache.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperwatch.config import AppConfig
from paperwatch.errors import SourceAccessDenied
from paperwatch.source_cache import (
    CACHED_SOURCE_FILE_NAME_PATTERN,
    EXTRACTED_TEXT_SUFFIX,
    SOURCE_FILES_SUBDIRECTORY,
    require_cached_source_file,
)

BYTES_PER_MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class SourceCacheUsage:
    raw_source_bytes: int  # replaceable downloads (.pdf / .xml)
    extracted_text_bytes: int  # kept (.txt)
    limit_bytes: int

    def describe(self) -> str:
        return (
            f"Source cache: {self.raw_source_bytes / BYTES_PER_MEGABYTE:.1f} MB of downloads (limit "
            f"{self.limit_bytes / BYTES_PER_MEGABYTE:.0f} MB, removable once extracted), "
            f"{self.extracted_text_bytes / BYTES_PER_MEGABYTE:.1f} MB of extracted text (kept)"
        )


@dataclass(frozen=True)
class SourceCachePruneSummary:
    files_removed: int
    bytes_removed: int
    usage_after: SourceCacheUsage

    def describe(self) -> str:
        removed_text = (
            (f"removed {self.files_removed} extracted download(s) ({self.bytes_removed / BYTES_PER_MEGABYTE:.1f} MB); ")
            if self.files_removed
            else ""
        )
        return f"{self.usage_after.describe()}" + (f"; {removed_text.rstrip('; ')}" if removed_text else "")


def measure_source_cache(app_config: AppConfig) -> SourceCacheUsage:
    raw_source_bytes = extracted_text_bytes = 0
    sources_dir = app_config.source_cache_dir / SOURCE_FILES_SUBDIRECTORY
    if sources_dir.is_dir():
        for cached_path in sources_dir.glob("*/*"):
            if cached_path.is_symlink() or not cached_path.is_file():
                continue
            if CACHED_SOURCE_FILE_NAME_PATTERN.match(cached_path.name):
                raw_source_bytes += cached_path.stat().st_size
            elif cached_path.name.endswith(EXTRACTED_TEXT_SUFFIX):
                extracted_text_bytes += cached_path.stat().st_size
    return SourceCacheUsage(
        raw_source_bytes, extracted_text_bytes, app_config.collection.source_cache_limit_megabytes * BYTES_PER_MEGABYTE
    )


def prune_replaceable_downloads(connection: sqlite3.Connection, app_config: AppConfig) -> SourceCachePruneSummary:
    """Remove the oldest extracted raw downloads until the raw files fit the limit."""
    usage = measure_source_cache(app_config)
    bytes_over_limit = usage.raw_source_bytes - usage.limit_bytes
    files_removed = bytes_removed = 0
    if bytes_over_limit > 0:
        for removable_row in connection.execute(
            """SELECT text.edition_id, text.source_file_name FROM edition_source_texts AS text
                   WHERE text.acquisition_status = 'acquired' AND text.extraction_status = 'extracted'
                     AND text.source_file_name IS NOT NULL
                     AND NOT EXISTS (SELECT 1 FROM jobs WHERE jobs.paper_id = text.paper_id
                                       AND jobs.kind = 'review_paper' AND jobs.status IN ('queued', 'running'))
                   ORDER BY text.acquired_at, text.edition_id"""
        ).fetchall():
            if bytes_removed >= bytes_over_limit:
                break
            removed_byte_count = _remove_cached_source_file(
                app_config.source_cache_dir,
                app_config.source_cache_dir
                / SOURCE_FILES_SUBDIRECTORY
                / removable_row["edition_id"]
                / removable_row["source_file_name"],
            )
            if removed_byte_count:
                files_removed += 1
                bytes_removed += removed_byte_count
    return SourceCachePruneSummary(
        files_removed, bytes_removed, measure_source_cache(app_config) if files_removed else usage
    )


def _remove_cached_source_file(cache_dir: Path, candidate_path: Path) -> int:
    """Delete one raw source file after checking it is exactly a cache-named .pdf/.xml (never a .txt)."""
    try:
        verified_path = require_cached_source_file(cache_dir, candidate_path)
    except SourceAccessDenied:
        return 0  # already gone, or not a cache-generated source file name: leave it alone
    byte_count = verified_path.stat().st_size
    verified_path.unlink()
    return byte_count
