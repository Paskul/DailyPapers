"""Acquire full text for papers queued for review, then extract it in the sandbox ("Source text").

Shortlisted work = papers with a queued review job, i.e. those screening promoted. For each, in
queue order (priority class, then queue age), the first route that works wins:
  1. local file: a PDF/XML the user placed in the local sources folder (local_source_files.py). Free; no network.
  2. a VERSIONED arXiv PDF through arXiv's programmatic host (export.arxiv.org), for an arXiv edition.
  3. an OpenAlex content PDF (content.openalex.org), for an OpenAlex edition; needs an OpenAlex key.
Downloads (2 and 3) are limited to `collection.daily_source_text_budget` per UTC day, and go through the gateway's
fixed get_text(known_edition_id) operation, which re-checks the edition against its source records at dispatch.
A paper known by an arXiv id but catalogued only from another source (typically a cited work found through
OpenAlex) first gets its arXiv version looked up: one arXiv API call, used only when the arXiv title matches the
paper's, so its PDF can come from arXiv (free, and not limited like the OpenAlex PDF host).
When no route has the text, the outcome is recorded (`unavailable`, with the reasons) and the review uses the
abstract. It is retried after RETRY_UNAVAILABLE_AFTER, at most MAXIMUM_ACQUISITION_ATTEMPTS times.

A provider outage, a used-up request budget, or the day's used-up download budget is temporary: nothing is
recorded for the paper. Third pilot (2026-11-23): arXiv was paused for 5 minutes after a gap search got HTTP 429,
and the OpenAlex PDF host answered 429 after exactly 10 downloads in every run; the papers skipped then were read
from their abstracts minutes later, though their PDFs came in the next run. So the stage now
  * waits out a short pause and tries those papers again (at most `collection.source_text_pause_wait_seconds` of
    waiting per run), and
  * returns the papers still waiting, whose readings `collect` keeps held for the next run instead of releasing them
    to the worker (paperwatch.readings.settle_readings_held_for_source_text; at most
    `collection.source_text_maximum_hold_hours` after they were queued, after which the abstract is read).

Extraction runs the pinned extractor in rootless Podman (paperwatch.sandboxed_extractor) on the CACHED copy and
stores the text next to it (state/cache/sources/<edition id>/<source hash>.txt). Without a configured image the row
says `not_configured`, and extraction happens on the first run after the image is set. After a successful extraction
the text's normalized fingerprint is stored and a promoted paper's queued reading switches to the full text (the
reading turns the text into passages); the worker never downloads anything itself.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from paperwatch.catalog import load_paper_aliases, load_paper_row
from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp, utc_now
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.local_source_files import (
    LocalSourceFolder,
    file_stems_for_edition,
    file_stems_for_paper,
    read_local_source_bytes,
)
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.collection.source_cache_limit import prune_replaceable_downloads
from paperwatch.db import write_transaction
from paperwatch.errors import (
    ExtractionError,
    ProviderUnavailable,
    SourceAccessDenied,
    SourceInputError,
    SourceRecordNotFound,
    SourceRequestBudgetExhausted,
)
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND
from paperwatch.llm.service_guard import read_service_health_state
from paperwatch.network.provider_gateway import provider_service_name
from paperwatch.readings import FULL_TEXT_EDITION_ORDER_SQL
from paperwatch.references.bibliography_identifiers import provider_title_matches_entry
from paperwatch.review_inputs import fingerprint_extracted_reading_text
from paperwatch.sandboxed_extractor import BYTES_PER_MEGABYTE, extract_text_in_sandbox
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, OPENALEX_SCHEME
from paperwatch.screening.review_promotion import queue_new_reading_for_promoted_paper_in_transaction
from paperwatch.source_cache import CachedSourceFile, cached_edition_directory, store_extracted_text, store_source_bytes
from paperwatch.sources.arxiv_atom import parse_api_search_page

TEXT_ROUTE_LOCAL_FILE = "local_file"
TEXT_ROUTE_ARXIV_PDF = "arxiv_pdf"
TEXT_ROUTE_OPENALEX_CONTENT = "openalex_content"
ARXIV_PROVIDER_NAME = "arxiv"
NETWORK_TEXT_ROUTE_BY_SCHEME = {
    ARXIV_SCHEME: (TEXT_ROUTE_ARXIV_PDF, ARXIV_PROVIDER_NAME),
    OPENALEX_SCHEME: (TEXT_ROUTE_OPENALEX_CONTENT, "openalex"),
}
RETRY_UNAVAILABLE_AFTER = timedelta(days=1)
MAXIMUM_ACQUISITION_ATTEMPTS = 3
MAXIMUM_PAPERS_CONSIDERED_PER_RUN = 200
MAXIMUM_EXTRACTIONS_PER_RUN = 50
MAXIMUM_STORED_ERROR_CHARACTERS = 300
MAXIMUM_WAITING_ROUNDS_PER_RUN = 10  # a bound on retry rounds, whatever the providers answer
ARXIV_VERSION_LOOKUP_ROUTE_PREFIX = "arxiv-version-lookup:"  # discovery route of an arXiv version found this way


@dataclass
class SourceTextRunCounts:
    from_local_files: int = 0
    downloaded: int = 0
    unavailable: int = 0
    waiting_for_download_budget: int = 0
    waiting_for_a_provider: int = 0
    arxiv_versions_found: int = 0
    seconds_waited_for_providers: float = 0.0
    extracted: int = 0
    extraction_failed: int = 0
    extraction_not_configured: int = 0
    readings_switched_to_full_text: int = 0

    def describe(self, downloads_remaining: int) -> str:
        return (
            f"Source text: {self.from_local_files} from local files, {self.downloaded} downloaded "
            f"({downloads_remaining} download(s) left today), {self.unavailable} unavailable (abstract only), "
            f"{self.waiting_for_download_budget} waiting for download budget, {self.waiting_for_a_provider} "
            f"waiting for a paused provider (waited {self.seconds_waited_for_providers:.0f} s), "
            f"{self.arxiv_versions_found} arXiv version(s) found for papers known by an arXiv id; extraction: "
            f"{self.extracted} extracted, {self.extraction_failed} failed, {self.extraction_not_configured} waiting "
            f"for the extractor image (set extractor.image); {self.readings_switched_to_full_text} reading(s) "
            "switched to full text"
        )


@dataclass(frozen=True)
class SourceTextRunResult:
    report_lines: list[str]
    # Papers whose text a provider (or the day's download budget) could not deliver in this run: `collect` keeps
    # their readings held for a later run (paperwatch.readings.settle_readings_held_for_source_text).
    papers_waiting_for_text: frozenset[str]


def wait_for_provider_pause(seconds: float) -> None:
    """Sleep through a provider's short pause (a module function, so tests can advance their clock instead)."""
    time.sleep(seconds)


def acquire_source_texts(context: CollectionContext, *, allow_network: bool) -> SourceTextRunResult:
    collection_settings = context.app_config.collection
    source_text_run = _SourceTextRun(
        context,
        LocalSourceFolder(context.app_config.local_sources_dir),
        allow_network=allow_network,
        downloads_remaining=max(
            0,
            collection_settings.daily_source_text_budget - count_network_downloads_today(context.connection, context),
        ),
    )
    for paper_id in papers_waiting_for_source_text(context.connection, MAXIMUM_PAPERS_CONSIDERED_PER_RUN):
        source_text_run.acquire_text_of_paper(paper_id)
    source_text_run.wait_out_short_provider_pauses(
        timedelta(seconds=collection_settings.source_text_pause_wait_seconds)
    )
    run_counts = source_text_run.counts
    run_counts.waiting_for_a_provider = len(source_text_run.papers_waiting_for_a_provider)
    _extract_pending_texts(context, run_counts)
    cache_summary = prune_replaceable_downloads(context.connection, context.app_config)
    return SourceTextRunResult(
        [run_counts.describe(source_text_run.downloads_remaining), cache_summary.describe()],
        frozenset(source_text_run.papers_waiting_for_a_provider + source_text_run.papers_waiting_for_budget),
    )


@dataclass
class _SourceTextRun:
    """The state of one source-text stage: the download budget left, the providers that are paused or out of
    request budget for now, and the papers waiting for one of them."""

    context: CollectionContext
    local_folder: LocalSourceFolder
    allow_network: bool
    downloads_remaining: int
    counts: SourceTextRunCounts = field(default_factory=SourceTextRunCounts)
    unavailable_providers: set[str] = field(default_factory=set)
    papers_waiting_for_a_provider: list[str] = field(default_factory=list)  # in queue order
    papers_waiting_for_budget: list[str] = field(default_factory=list)
    arxiv_versions_looked_up: set[str] = field(default_factory=set)  # one lookup per paper and run

    def acquire_text_of_paper(self, paper_id: str) -> None:
        if not _is_due_for_another_attempt(self.context, paper_id):
            return
        edition_rows = editions_in_text_preference_order(self.context.connection, paper_id)
        if not edition_rows:
            return
        if _try_local_file(self.context, self.local_folder, paper_id, edition_rows):
            self.counts.from_local_files += 1
            return
        if not self.allow_network:
            return
        if self.downloads_remaining == 0:
            self.counts.waiting_for_download_budget += 1
            self.papers_waiting_for_budget.append(paper_id)
            return
        if not self._add_arxiv_version_if_known(paper_id, edition_rows):
            self.papers_waiting_for_a_provider.append(paper_id)  # arXiv is paused: its free PDF is worth the wait
            return
        edition_rows = editions_in_text_preference_order(self.context.connection, paper_id)
        download_outcome = _try_downloads(self.context, edition_rows, self.unavailable_providers)
        if download_outcome == "downloaded":
            self.downloads_remaining -= 1
            self.counts.downloaded += 1
        elif download_outcome == "unavailable":
            self.counts.unavailable += 1
        else:
            self.papers_waiting_for_a_provider.append(paper_id)

    def wait_out_short_provider_pauses(self, wait_budget: timedelta) -> None:
        """While papers wait for a paused provider whose pause ends within what is left of `wait_budget`: sleep
        until then and try those papers again."""
        for _round in range(MAXIMUM_WAITING_ROUNDS_PER_RUN):
            if not self.papers_waiting_for_a_provider or self.downloads_remaining == 0:
                return
            resume_at = self._earliest_end_of_a_provider_pause()
            if resume_at is None:
                return  # out of request budget rather than paused: nothing to wait for in this run
            wait_seconds = max(0.0, (resume_at - utc_now()).total_seconds()) + 1.0
            if wait_seconds > wait_budget.total_seconds():
                return
            wait_for_provider_pause(wait_seconds)
            wait_budget -= timedelta(seconds=wait_seconds)
            self.counts.seconds_waited_for_providers += wait_seconds
            self.unavailable_providers.clear()
            papers_to_retry, self.papers_waiting_for_a_provider = self.papers_waiting_for_a_provider, []
            for paper_id in papers_to_retry:
                self.acquire_text_of_paper(paper_id)

    def _earliest_end_of_a_provider_pause(self) -> datetime | None:
        pause_ends = []
        for provider_name in self.unavailable_providers:
            health_state = read_service_health_state(self.context.connection, provider_service_name(provider_name))
            pause_end = health_state.pause_end_at(utc_now())
            if pause_end is not None:
                pause_ends.append(pause_end)
        return min(pause_ends, default=None)

    def _add_arxiv_version_if_known(self, paper_id: str, edition_rows: list[sqlite3.Row]) -> bool:
        """For a paper known by an arXiv id but without an arXiv edition, look the arXiv version up (once per run)
        and catalog it when its title matches. False only when arXiv is paused or out of budget right now."""
        if any(row["edition_key"].startswith(f"{ARXIV_SCHEME}:") for row in edition_rows):
            return True
        arxiv_ids = [
            value for scheme, value in load_paper_aliases(self.context.connection, paper_id) if scheme == ARXIV_SCHEME
        ]
        if not arxiv_ids or paper_id in self.arxiv_versions_looked_up:
            return True
        if ARXIV_PROVIDER_NAME in self.unavailable_providers:
            return False
        self.arxiv_versions_looked_up.add(paper_id)
        try:
            if catalog_arxiv_version_of_paper(self.context, paper_id, arxiv_ids[0]):
                self.counts.arxiv_versions_found += 1
        except (ProviderUnavailable, SourceRequestBudgetExhausted):
            self.unavailable_providers.add(ARXIV_PROVIDER_NAME)
            self.arxiv_versions_looked_up.discard(paper_id)  # try again after the pause
            return False
        return True


def catalog_arxiv_version_of_paper(context: CollectionContext, paper_id: str, arxiv_id: str) -> bool:
    """Look `arxiv_id` up on arXiv and catalog the record as an edition of `paper_id` when its title matches the
    paper's (an arXiv id that another source linked to the paper may be wrong). True if an arXiv edition was added.
    Provider-wide problems (a pause, a used-up request budget) propagate."""
    try:
        provider_response = context.gateway.get_work(f"{ARXIV_SCHEME}:{arxiv_id}")
        parsed_page = parse_api_search_page(
            provider_response.body_bytes, f"{ARXIV_VERSION_LOOKUP_ROUTE_PREFIX}{paper_id}", start_offset=0
        )
    except (SourceRecordNotFound, SourceAccessDenied, SourceInputError):
        return False
    paper_title = load_paper_row(context.connection, paper_id)["title"]
    matching_records = [
        record for record in parsed_page.records if provider_title_matches_entry(record.title, paper_title)
    ]
    if not matching_records:
        return False
    ingest_records_with_checkpoints(
        context.connection,
        matching_records[:1],
        admission_limit=None,
        active_seed_paper_ids=context.active_seed_paper_ids,
        max_attempts_per_job=context.app_config.worker.max_attempts_per_job,
    )
    return any(
        row["edition_key"].startswith(f"{ARXIV_SCHEME}:")
        for row in editions_in_text_preference_order(context.connection, paper_id)
    )


# --------------------------------------------------------------------------------------------------------------------
# Which papers, which editions, and whether it is time to try again
# --------------------------------------------------------------------------------------------------------------------


def papers_waiting_for_source_text(connection: sqlite3.Connection, limit: int) -> list[str]:
    """Papers with a queued review job and no acquired text yet, in queue order."""
    return [
        paper_row["paper_id"]
        for paper_row in connection.execute(
            """SELECT paper_id, MIN(priority) AS best_priority, MIN(created_at) AS first_queued_at FROM jobs
           WHERE kind = ? AND status = 'queued' AND paper_id IS NOT NULL
             AND paper_id NOT IN (SELECT paper_id FROM edition_source_texts WHERE acquisition_status = 'acquired')
           GROUP BY paper_id ORDER BY best_priority, first_queued_at, paper_id LIMIT ?""",
            (REVIEW_PAPER_JOB_KIND, limit),
        )
    ]


def editions_in_text_preference_order(connection: sqlite3.Connection, paper_id: str) -> list[sqlite3.Row]:
    """arXiv editions first (newest version first), then OpenAlex, then the rest (newest sighting first). The same
    order picks the edition a full-text reading uses (paperwatch.readings.select_reading_input)."""
    return connection.execute(
        f"SELECT * FROM paper_editions WHERE paper_id = ? ORDER BY {FULL_TEXT_EDITION_ORDER_SQL}",  # noqa: S608  # interpolates module constants only; values are bound
        (paper_id,),
    ).fetchall()


def count_network_downloads_today(connection: sqlite3.Connection, context: CollectionContext) -> int:
    day_start = format_utc_timestamp(context.started_at.replace(hour=0, minute=0, second=0, microsecond=0))
    return int(
        connection.execute(
            "SELECT COUNT(*) FROM edition_source_texts WHERE downloaded_from_network = 1 AND acquired_at >= ?",
            (day_start,),
        ).fetchone()[0]
    )


def _is_due_for_another_attempt(context: CollectionContext, paper_id: str) -> bool:
    last_attempt_row = context.connection.execute(
        """SELECT MAX(attempts) AS attempts, MAX(updated_at) AS last_attempt_at FROM edition_source_texts
           WHERE paper_id = ? AND acquisition_status IN ('unavailable', 'failed')""",
        (paper_id,),
    ).fetchone()
    if last_attempt_row["last_attempt_at"] is None:
        return True
    if last_attempt_row["attempts"] >= MAXIMUM_ACQUISITION_ATTEMPTS:
        return False
    return context.started_at - parse_utc_timestamp(last_attempt_row["last_attempt_at"]) >= RETRY_UNAVAILABLE_AFTER


# --------------------------------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------------------------------


def _try_local_file(
    context: CollectionContext, local_folder: LocalSourceFolder, paper_id: str, edition_rows: list[sqlite3.Row]
) -> bool:
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
            _record_not_acquired(
                context, edition_row, "failed", f"local file {local_source_file.file_path.name} {unusable_file}"
            )
            continue
        cached_file = store_source_bytes(
            context.app_config.source_cache_dir, edition_row["edition_id"], file_bytes, local_source_file.content_type
        )
        _record_acquired(context, edition_row, cached_file, TEXT_ROUTE_LOCAL_FILE, downloaded_from_network=False)
        return True
    return False


def _try_downloads(context: CollectionContext, edition_rows: list[sqlite3.Row], unavailable_providers: set[str]) -> str:
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
    _record_not_acquired(
        context,
        edition_rows[0],
        "unavailable",
        "; ".join(reasons) or "no permitted text route for this paper's editions",
    )
    return "unavailable"


# --------------------------------------------------------------------------------------------------------------------
# Extraction in the sandbox
# --------------------------------------------------------------------------------------------------------------------


def _extract_pending_texts(context: CollectionContext, run_counts: SourceTextRunCounts) -> None:
    extractor_settings = context.app_config.extractor
    pending_rows = context.connection.execute(
        """SELECT * FROM edition_source_texts WHERE acquisition_status = 'acquired'
               AND extraction_status IN ('pending', 'not_configured')
           ORDER BY acquired_at, edition_id LIMIT ?""",
        (MAXIMUM_EXTRACTIONS_PER_RUN,),
    ).fetchall()
    for text_row in pending_rows:
        if not extractor_settings.image:
            _update_extraction(context, text_row["edition_id"], extraction_status="not_configured")
            run_counts.extraction_not_configured += 1
            continue
        source_file_path = (
            cached_edition_directory(context.app_config.source_cache_dir, text_row["edition_id"])
            / text_row["source_file_name"]
        )
        try:
            extracted = extract_text_in_sandbox(
                context.app_config.source_cache_dir, source_file_path, extractor_settings
            )
        except (ExtractionError, SourceAccessDenied) as extraction_problem:
            _update_extraction(
                context,
                text_row["edition_id"],
                extraction_status="failed",
                last_error=str(extraction_problem)[:MAXIMUM_STORED_ERROR_CHARACTERS],
            )
            run_counts.extraction_failed += 1
            continue
        text_path = store_extracted_text(
            context.app_config.source_cache_dir, text_row["edition_id"], text_row["source_sha256"], extracted.text
        )
        _update_extraction(
            context,
            text_row["edition_id"],
            extraction_status="extracted",
            extracted_text_file_name=text_path.name,
            extracted_text_sha256=hashlib.sha256(extracted.text.encode("utf-8")).hexdigest(),
            normalized_text_sha256=fingerprint_extracted_reading_text(extracted.text),
            extracted_text_characters=len(extracted.text.strip()),
            extractor_image=extracted.extractor_image,
            last_error=None,
        )
        run_counts.extracted += 1
        if _switch_reading_to_full_text(context, text_row["paper_id"]):
            run_counts.readings_switched_to_full_text += 1


def _switch_reading_to_full_text(context: CollectionContext, paper_id: str) -> bool:
    """Queue the full-text reading of a promoted paper (its queued abstract reading is retired): newly
    available full text is a new input. Returns True if a job was added."""
    with write_transaction(context.connection):
        return queue_new_reading_for_promoted_paper_in_transaction(
            context.connection,
            paper_id,
            active_seed_paper_ids=context.active_seed_paper_ids,
            max_attempts=context.app_config.worker.max_attempts_per_job,
        )


# --------------------------------------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------------------------------------


def _record_acquired(
    context: CollectionContext,
    edition_row: sqlite3.Row,
    cached_file: CachedSourceFile,
    text_route: str,
    *,
    downloaded_from_network: bool,
) -> None:
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
            (
                edition_row["edition_id"],
                edition_row["paper_id"],
                text_route,
                cached_file.file_path.name,
                cached_file.content_sha256,
                cached_file.content_type,
                cached_file.byte_count,
                now_timestamp,
                int(downloaded_from_network),
                now_timestamp,
            ),
        )


def _record_not_acquired(
    context: CollectionContext, edition_row: sqlite3.Row, acquisition_status: str, reason: str
) -> None:
    now_timestamp = context.started_at_timestamp
    with write_transaction(context.connection):
        context.connection.execute(
            """INSERT INTO edition_source_texts (edition_id, paper_id, acquisition_status, attempts, last_error,
                   updated_at) VALUES (?, ?, ?, 1, ?, ?)
               ON CONFLICT(edition_id) DO UPDATE SET acquisition_status = excluded.acquisition_status,
                   attempts = attempts + 1, last_error = excluded.last_error, updated_at = excluded.updated_at
               WHERE acquisition_status != 'acquired'""",
            (
                edition_row["edition_id"],
                edition_row["paper_id"],
                acquisition_status,
                reason[:MAXIMUM_STORED_ERROR_CHARACTERS],
                now_timestamp,
            ),
        )


def _update_extraction(context: CollectionContext, edition_id: str, **column_values: object) -> None:
    assignments = ", ".join(f"{column_name} = ?" for column_name in column_values)
    with write_transaction(context.connection):
        context.connection.execute(
            f"UPDATE edition_source_texts SET {assignments}, updated_at = ? WHERE edition_id = ?",  # noqa: S608  # column names come from code; values are bound
            (*column_values.values(), context.started_at_timestamp, edition_id),
        )
