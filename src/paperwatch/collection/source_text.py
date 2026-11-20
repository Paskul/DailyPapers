"""Acquire full text for papers queued for review, then extract it in the sandbox (plan part 5: "Source text").

Shortlisted work = papers with a queued review job, i.e. those screening promoted (plan part 6). For each, in
queue order (priority class, then queue age), the first route that works wins:
  1. local file: a PDF/XML the user placed in the local sources folder (local_source_files.py). Free; no network.
  2. a VERSIONED arXiv PDF through arXiv's programmatic host (export.arxiv.org), for an arXiv edition.
  3. an OpenAlex content PDF (content.openalex.org), for an OpenAlex edition; needs an OpenAlex key.
Downloads (2 and 3) are limited to `collection.daily_source_text_budget` per UTC day, and go through the gateway's
fixed get_text(known_edition_id) operation, which re-checks the edition against its source records at dispatch.
When no route has the text, the outcome is recorded (`unavailable`, with the reasons) and the review uses the
abstract. It is retried after RETRY_UNAVAILABLE_AFTER, at most MAXIMUM_ACQUISITION_ATTEMPTS times. A provider outage
or a used-up request budget is temporary: nothing is recorded, and the next run tries again.

Extraction runs the pinned extractor in rootless Podman (paperwatch.sandboxed_extractor) on the CACHED copy and
stores the text next to it (state/cache/sources/<edition id>/<source hash>.txt). Without a configured image the row
says `not_configured`, and extraction happens on the first run after the image is set. Part 7 turns the text into
passages; the worker never downloads anything itself.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import timedelta

from paperwatch.catalog import load_paper_aliases
from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.local_source_files import (LocalSourceFolder, file_stems_for_edition, file_stems_for_paper,
                                                      read_local_source_bytes)
from paperwatch.db import write_transaction
from paperwatch.errors import (ExtractionError, ProviderUnavailable, SourceAccessDenied, SourceInputError,
                               SourceRecordNotFound, SourceRequestBudgetExhausted)
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND
from paperwatch.sandboxed_extractor import BYTES_PER_MEGABYTE, extract_text_in_sandbox
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, OPENALEX_SCHEME
from paperwatch.source_cache import (CachedSourceFile, cached_edition_directory, store_extracted_text,
                                     store_source_bytes)

TEXT_ROUTE_LOCAL_FILE = "local_file"
TEXT_ROUTE_ARXIV_PDF = "arxiv_pdf"
TEXT_ROUTE_OPENALEX_CONTENT = "openalex_content"
NETWORK_TEXT_ROUTE_BY_SCHEME = {ARXIV_SCHEME: (TEXT_ROUTE_ARXIV_PDF, "arxiv"),
                                OPENALEX_SCHEME: (TEXT_ROUTE_OPENALEX_CONTENT, "openalex")}
RETRY_UNAVAILABLE_AFTER = timedelta(days=1)
MAXIMUM_ACQUISITION_ATTEMPTS = 3
MAXIMUM_PAPERS_CONSIDERED_PER_RUN = 200
MAXIMUM_EXTRACTIONS_PER_RUN = 50
MAXIMUM_STORED_ERROR_CHARACTERS = 300


@dataclass
class SourceTextRunCounts:
    from_local_files: int = 0
    downloaded: int = 0
    unavailable: int = 0
    waiting_for_download_budget: int = 0
    extracted: int = 0
    extraction_failed: int = 0
    extraction_not_configured: int = 0

    def describe(self, downloads_remaining: int) -> str:
        return (f"Source text: {self.from_local_files} from local files, {self.downloaded} downloaded "
                f"({downloads_remaining} download(s) left today), {self.unavailable} unavailable (abstract only), "
                f"{self.waiting_for_download_budget} waiting for download budget; extraction: {self.extracted} "
                f"extracted, {self.extraction_failed} failed, {self.extraction_not_configured} waiting for the "
                "extractor image (set extractor.image)")


def acquire_source_texts(context: CollectionContext, *, allow_network: bool) -> list[str]:
    collection_settings = context.app_config.collection
    local_folder = LocalSourceFolder(context.app_config.local_sources_dir)
    downloads_remaining = max(0, collection_settings.daily_source_text_budget
                              - count_network_downloads_today(context.connection, context))
    unavailable_providers: set[str] = set()  # paused or out of request budget for the rest of this run
    run_counts = SourceTextRunCounts()

    for paper_id in papers_waiting_for_source_text(context.connection, MAXIMUM_PAPERS_CONSIDERED_PER_RUN):
        if not _is_due_for_another_attempt(context, paper_id):
            continue
        edition_rows = editions_in_text_preference_order(context.connection, paper_id)
        if not edition_rows:
            continue
        if _try_local_file(context, local_folder, paper_id, edition_rows):
            run_counts.from_local_files += 1
            continue
        if not allow_network:
            continue
        if downloads_remaining == 0:
            run_counts.waiting_for_download_budget += 1
            continue
        download_outcome = _try_downloads(context, edition_rows, unavailable_providers)
        if download_outcome == "downloaded":
            downloads_remaining -= 1
            run_counts.downloaded += 1
        elif download_outcome == "unavailable":
            run_counts.unavailable += 1

    _extract_pending_texts(context, run_counts)
    return [run_counts.describe(downloads_remaining)]


# --------------------------------------------------------------------------------------------------------------------
# Which papers, which editions, and whether it is time to try again
# --------------------------------------------------------------------------------------------------------------------


def papers_waiting_for_source_text(connection: sqlite3.Connection, limit: int) -> list[str]:
    """Papers with a queued review job and no acquired text yet, in queue order."""
    return [paper_row["paper_id"] for paper_row in connection.execute(
        """SELECT paper_id, MIN(priority) AS best_priority, MIN(created_at) AS first_queued_at FROM jobs
           WHERE kind = ? AND status = 'queued' AND paper_id IS NOT NULL
             AND paper_id NOT IN (SELECT paper_id FROM edition_source_texts WHERE acquisition_status = 'acquired')
           GROUP BY paper_id ORDER BY best_priority, first_queued_at, paper_id LIMIT ?""",
        (REVIEW_PAPER_JOB_KIND, limit))]


def editions_in_text_preference_order(connection: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    """arXiv editions first (newest version first), then OpenAlex, then the rest (newest sighting first)."""
    return connection.execute(
        """SELECT * FROM paper_editions WHERE paper_id = ?
           ORDER BY CASE WHEN edition_key LIKE 'arxiv:%' THEN 0 WHEN edition_key LIKE 'openalex:%' THEN 1 ELSE 2 END,
                    COALESCE(version_number, 0) DESC, first_seen_at DESC, edition_id""", (paper_id,)).fetchall()


def count_network_downloads_today(connection: sqlite3.Connection, context: CollectionContext) -> int:
    day_start = format_utc_timestamp(context.started_at.replace(hour=0, minute=0, second=0, microsecond=0))
    return connection.execute(
        "SELECT COUNT(*) FROM edition_source_texts WHERE downloaded_from_network = 1 AND acquired_at >= ?",
        (day_start,)).fetchone()[0]


def _is_due_for_another_attempt(context: CollectionContext, paper_id: str) -> bool:
    last_attempt_row = context.connection.execute(
        """SELECT MAX(attempts) AS attempts, MAX(updated_at) AS last_attempt_at FROM edition_source_texts
           WHERE paper_id = ? AND acquisition_status IN ('unavailable', 'failed')""", (paper_id,)).fetchone()
    if last_attempt_row["last_attempt_at"] is None:
        return True
    if last_attempt_row["attempts"] >= MAXIMUM_ACQUISITION_ATTEMPTS:
        return False
    return context.started_at - parse_utc_timestamp(last_attempt_row["last_attempt_at"]) >= RETRY_UNAVAILABLE_AFTER


# --------------------------------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------------------------------


def _try_local_file(context: CollectionContext, local_folder: LocalSourceFolder, paper_id: str,
                    edition_rows: list[sqlite3.Row]) -> bool:
    if local_folder.is_empty:
        return False
    candidates = [(edition_row, file_stems_for_edition(edition_row["edition_key"])) for edition_row in edition_rows]
    candidates.append((edition_rows[0], file_stems_for_paper(load_paper_aliases(context.connection, paper_id))))
    maximum_bytes = context.app_config.extractor.max_input_megabytes * BYTES_PER_MEGABYTE
    for edition_row, file_stems in candidates:
        local_source_file = local_folder.find(file_stems)
        if local_source_file is None:
            continue
        try:
            file_bytes = read_local_source_bytes(local_source_file, maximum_bytes)
        except (OSError, ValueError) as unusable_file:
            _record_not_acquired(context, edition_row, "failed",
                                 f"local file {local_source_file.file_path.name} {unusable_file}")
            continue
        cached_file = store_source_bytes(context.app_config.source_cache_dir, edition_row["edition_id"], file_bytes,
                                         local_source_file.content_type)
        _record_acquired(context, edition_row, cached_file, TEXT_ROUTE_LOCAL_FILE, downloaded_from_network=False)
        return True
    return False


def _try_downloads(context: CollectionContext, edition_rows: list[sqlite3.Row],
                   unavailable_providers: set[str]) -> str:
    """'downloaded', 'unavailable' (every route answered no), or 'temporary' (try again next run)."""
    reasons: list[str] = []
    had_temporary_problem = False
    for edition_row in edition_rows:
        scheme = edition_row["edition_key"].partition(":")[0]
        if scheme not in NETWORK_TEXT_ROUTE_BY_SCHEME:
            continue
        text_route, provider_name = NETWORK_TEXT_ROUTE_BY_SCHEME[scheme]
        if provider_name in unavailable_providers:
            had_temporary_problem = True
            continue
        try:
            cached_file = context.gateway.get_text(edition_row["edition_id"])
        except (SourceRequestBudgetExhausted, ProviderUnavailable) as temporary_problem:
            unavailable_providers.add(provider_name)
            had_temporary_problem = True
            reasons.append(f"{text_route}: {temporary_problem}")
            continue
        except (SourceRecordNotFound, SourceAccessDenied, SourceInputError) as refusal:
            reasons.append(f"{text_route}: {refusal}")
            continue
        _record_acquired(context, edition_row, cached_file, text_route, downloaded_from_network=True)
        return "downloaded"
    if had_temporary_problem:
        return "temporary"
    _record_not_acquired(context, edition_rows[0], "unavailable",
                         "; ".join(reasons) or "no permitted text route for this paper's editions")
    return "unavailable"


# --------------------------------------------------------------------------------------------------------------------
# Extraction in the sandbox
# --------------------------------------------------------------------------------------------------------------------


def _extract_pending_texts(context: CollectionContext, run_counts: SourceTextRunCounts) -> None:
    extractor_settings = context.app_config.extractor
    pending_rows = context.connection.execute(
        """SELECT * FROM edition_source_texts WHERE acquisition_status = 'acquired'
               AND extraction_status IN ('pending', 'not_configured')
           ORDER BY acquired_at, edition_id LIMIT ?""", (MAXIMUM_EXTRACTIONS_PER_RUN,)).fetchall()
    for text_row in pending_rows:
        if not extractor_settings.image:
            _update_extraction(context, text_row["edition_id"], extraction_status="not_configured")
            run_counts.extraction_not_configured += 1
            continue
        source_file_path = cached_edition_directory(context.app_config.source_cache_dir, text_row["edition_id"]) \
            / text_row["source_file_name"]
        try:
            extracted = extract_text_in_sandbox(context.app_config.source_cache_dir, source_file_path,
                                                extractor_settings)
        except (ExtractionError, SourceAccessDenied) as extraction_problem:
            _update_extraction(context, text_row["edition_id"], extraction_status="failed",
                               last_error=str(extraction_problem)[:MAXIMUM_STORED_ERROR_CHARACTERS])
            run_counts.extraction_failed += 1
            continue
        text_path = store_extracted_text(context.app_config.source_cache_dir, text_row["edition_id"],
                                         text_row["source_sha256"], extracted.text)
        _update_extraction(context, text_row["edition_id"], extraction_status="extracted",
                           extracted_text_file_name=text_path.name,
                           extracted_text_sha256=hashlib.sha256(extracted.text.encode("utf-8")).hexdigest(),
                           extractor_image=extracted.extractor_image, last_error=None)
        run_counts.extracted += 1


# --------------------------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------------------------


def _record_acquired(context: CollectionContext, edition_row: sqlite3.Row, cached_file: CachedSourceFile,
                     text_route: str, *, downloaded_from_network: bool) -> None:
    now_timestamp = context.started_at_timestamp
    with write_transaction(context.connection):
        context.connection.execute(
            """INSERT INTO edition_source_texts (edition_id, paper_id, acquisition_status, text_route,
                   source_file_name, source_sha256, content_type, source_bytes, extraction_status, attempts,
                   last_error, acquired_at, downloaded_from_network, updated_at)
               VALUES (?, ?, 'acquired', ?, ?, ?, ?, ?, 'pending', 1, NULL, ?, ?, ?)
               ON CONFLICT(edition_id) DO UPDATE SET acquisition_status = 'acquired', text_route = excluded.text_route,
                   source_file_name = excluded.source_file_name, source_sha256 = excluded.source_sha256,
                   content_type = excluded.content_type, source_bytes = excluded.source_bytes,
                   extraction_status = 'pending', attempts = attempts + 1, last_error = NULL,
                   acquired_at = excluded.acquired_at, downloaded_from_network = excluded.downloaded_from_network,
                   updated_at = excluded.updated_at""",
            (edition_row["edition_id"], edition_row["paper_id"], text_route, cached_file.file_path.name,
             cached_file.content_sha256, cached_file.content_type, cached_file.byte_count, now_timestamp,
             int(downloaded_from_network), now_timestamp))


def _record_not_acquired(context: CollectionContext, edition_row: sqlite3.Row, acquisition_status: str,
                         reason: str) -> None:
    now_timestamp = context.started_at_timestamp
    with write_transaction(context.connection):
        context.connection.execute(
            """INSERT INTO edition_source_texts (edition_id, paper_id, acquisition_status, attempts, last_error,
                   updated_at) VALUES (?, ?, ?, 1, ?, ?)
               ON CONFLICT(edition_id) DO UPDATE SET acquisition_status = excluded.acquisition_status,
                   attempts = attempts + 1, last_error = excluded.last_error, updated_at = excluded.updated_at
               WHERE acquisition_status != 'acquired'""",
            (edition_row["edition_id"], edition_row["paper_id"], acquisition_status,
             reason[:MAXIMUM_STORED_ERROR_CHARACTERS], now_timestamp))


def _update_extraction(context: CollectionContext, edition_id: str, **column_values: object) -> None:
    assignments = ", ".join(f"{column_name} = ?" for column_name in column_values)
    with write_transaction(context.connection):
        context.connection.execute(
            f"UPDATE edition_source_texts SET {assignments}, updated_at = ? WHERE edition_id = ?",
            (*column_values.values(), context.started_at_timestamp, edition_id))
