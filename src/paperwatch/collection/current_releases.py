"""Current releases: arXiv category feeds, missed-release recovery, and recent OpenAlex windows (plan part 5).

arXiv category feeds (checkpoint `arxiv-feed:<category>`)
  * Every approved category is polled once per run with If-None-Match, so an unchanged feed costs one 304.
    The feed changes once per announcement day; a later poll of the same feed finds the same records (no-ops).
  * Each item is labelled new, cross-listed, or revised (paper_discoveries / paper_editions.announcement_kind).
  * The checkpoint's `last_completed_interval_end` is the last poll whose items were all stored.

Missed releases (checkpoints `arxiv-recovery:<category>:<first day>..<last day>`)
  A feed shows only the latest announcement, which stays up at least a day. So a release can be missed only if no poll
  completed for longer than ARXIV_FEED_SAFE_GAP, or if the daily candidate budget cut a poll short. In both cases a
  dated arXiv API search over submission dates (from ARXIV_SUBMISSION_LAG before the last complete poll up to now)
  is recorded as its own checkpoint: an open coverage gap until it is exhausted. It pages from saved cursors across
  runs, within the daily candidate budget. Revisions of papers never seen are not recovered (the API selects by first
  submission date); revisions of known papers arrive with their next sighting.

Recent OpenAlex windows (checkpoint `openalex-recent:<query id>`)
  Journal and conference work outside arXiv: each approved OpenAlex phrase is searched over publication dates from
  `openalex_recent_window_days` before the last completed sweep up to today, at most once per
  `openalex_recent_sweep_interval_hours`. The overlap re-checks works OpenAlex indexed late. An interrupted window keeps
  its window and cursor and resumes before a new window starts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

from paperwatch.clock import parse_utc_timestamp
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.page_ingestion import PageIngestionResult, ingest_records_with_checkpoints
from paperwatch.collection.paged_search import (PARAMETER_PUBLIC_VALUE, SearchTarget, is_search_resumable,
                                                mark_search_blocked, run_search_pages, search_parameters,
                                                stopped_for_a_missing_credential, window_from_parameters)
from paperwatch.errors import ProviderUnavailable, SourceAccessDenied, SourceInputError
from paperwatch.network.provider_requests import PublicationWindow
from paperwatch.public_query_catalog import QUERY_KIND_CATEGORY, QUERY_KIND_KEYWORDS, build_public_query_catalog
from paperwatch.source_checkpoints import (SearchCoverageStatus, SourceCheckpoint, load_source_checkpoint,
                                           load_source_checkpoints_with_prefix)
from paperwatch.sources.arxiv_atom import ParsedFeed, parse_category_feed
from paperwatch.sources.discovery_routes import (RouteKind, arxiv_feed_route, arxiv_recovery_route,
                                                 openalex_recent_route)

ARXIV_FEED_SAFE_GAP = timedelta(hours=24)  # a feed shows each announcement for at least this long
ARXIV_SUBMISSION_LAG = timedelta(days=4)  # submissions are announced up to ~4 days later (weekends)
ARXIV_FEED_CHECKPOINT_PREFIX = "arxiv-feed:"
ARXIV_RECOVERY_CHECKPOINT_PREFIX = "arxiv-recovery:"
OPENALEX_RECENT_CHECKPOINT_PREFIX = "openalex-recent:"
FEED_ETAG_RESPONSE_HEADER = "etag"


def collect_current_releases(context: CollectionContext) -> list[str]:
    """Feeds first, then any open recovery windows, then recent OpenAlex windows. Returns lines for the operator."""
    report_lines: list[str] = []
    for arxiv_category in approved_arxiv_categories(context):
        report_lines.append(poll_category_feed(context, arxiv_category))
    report_lines += run_open_recovery_searches(context)
    report_lines += sweep_recent_openalex_windows(context)
    return report_lines


def approved_arxiv_categories(context: CollectionContext) -> list[str]:
    """Each approved category once, in topics.toml order (several topics may share one)."""
    categories = [public_query.public_value for public_query in build_public_query_catalog(
        context.app_config.topics).values() if public_query.query_kind == QUERY_KIND_CATEGORY]
    return list(dict.fromkeys(categories))


# --------------------------------------------------------------------------------------------------------------------
# arXiv category feeds
# --------------------------------------------------------------------------------------------------------------------


def poll_category_feed(context: CollectionContext, arxiv_category: str) -> str:
    feed_key = f"{ARXIV_FEED_CHECKPOINT_PREFIX}{arxiv_category}"
    feed_checkpoint = load_source_checkpoint(context.connection, feed_key) or SourceCheckpoint(
        search_key=feed_key, source_name="arxiv", query_parameters={"category": arxiv_category})
    previous_complete_poll = feed_checkpoint.last_completed_interval_end
    stored_etag = feed_checkpoint.cursor if previous_complete_poll else None
    try:
        feed_response = context.gateway.poll_category_feed(arxiv_category, if_none_match_etag=stored_etag)
        parsed_feed = None if feed_response.was_not_modified else parse_category_feed(
            feed_response.body_bytes, arxiv_feed_route(arxiv_category))
    except (ProviderUnavailable, SourceAccessDenied, SourceInputError) as problem:
        blocked_checkpoint = mark_search_blocked(context, feed_checkpoint, problem)
        return f"arXiv feed {arxiv_category}: NOT polled ({blocked_checkpoint.stop_reason}); the last complete poll " \
               f"was {previous_complete_poll or 'never'}"

    gap_is_too_long = previous_complete_poll is not None and \
        context.started_at - parse_utc_timestamp(previous_complete_poll) > ARXIV_FEED_SAFE_GAP
    response_etag = feed_response.response_headers.get(FEED_ETAG_RESPONSE_HEADER) or stored_etag
    feed_records = parsed_feed.records if parsed_feed else []
    scheduled_recovery_keys: list[str] = []

    def feed_checkpoints_after_page(page_result: PageIngestionResult) -> list[SourceCheckpoint]:
        """Inside the page transaction: the feed's own checkpoint, plus a recovery search if coverage has a gap."""
        checkpoints: list[SourceCheckpoint] = []
        if gap_is_too_long or page_result.was_stopped_by_admission_limit:
            recovery_checkpoint = _new_recovery_checkpoint_if_absent(context, arxiv_category, previous_complete_poll)
            if recovery_checkpoint is not None:
                checkpoints.append(recovery_checkpoint)
                scheduled_recovery_keys.append(recovery_checkpoint.search_key)
        if page_result.was_stopped_by_admission_limit:
            # The recovery search now owns the uncovered window; forget the ETag so the next poll re-reads the feed.
            completed_feed = replace(feed_checkpoint, cursor=None, coverage_status=SearchCoverageStatus.CAPPED,
                                     stop_reason=f"daily candidate budget reached after part of the feed; the rest "
                                                 f"is left to the recovery search",
                                     last_completed_interval_end=context.started_at_timestamp)
        else:
            completed_feed = replace(feed_checkpoint, cursor=response_etag,
                                     coverage_status=SearchCoverageStatus.EXHAUSTED,
                                     stop_reason=_describe_feed_poll(parsed_feed),
                                     last_completed_interval_end=context.started_at_timestamp)
        return checkpoints + [replace(completed_feed, candidates_admitted=feed_checkpoint.candidates_admitted
                                      + page_result.new_papers_admitted)]

    page_result = ingest_records_with_checkpoints(
        context.connection, feed_records, admission_limit=context.candidate_budget.remaining_for(
            RouteKind.CURRENT_RELEASE), active_seed_paper_ids=context.active_seed_paper_ids,
        max_attempts_per_job=context.app_config.worker.max_attempts_per_job,
        checkpoints_after_page=feed_checkpoints_after_page)
    if parsed_feed is not None:
        page_result.skipped_malformed_records += sum(parsed_feed.skipped_entry_counts.values())
    context.candidate_budget.record_admissions(RouteKind.CURRENT_RELEASE, page_result.new_papers_admitted)

    label = f"arXiv feed {arxiv_category}"
    report_line = f"{label}: unchanged since the last poll (HTTP 304)" if parsed_feed is None else \
        page_result.describe(f"{label} ({_describe_feed_poll(parsed_feed)})")
    for recovery_key in scheduled_recovery_keys:
        report_line += f"\n  coverage gap recorded; recovery search {recovery_key} scheduled"
    return report_line


def _describe_feed_poll(parsed_feed: ParsedFeed | None) -> str:
    if parsed_feed is None:
        return "unchanged (HTTP 304)"
    kind_counts = parsed_feed.announcement_kind_counts
    return (f"announcement {parsed_feed.announcement_date or 'none'}: {kind_counts.get('new', 0)} new, "
            f"{kind_counts.get('cross_list', 0)} cross-listed, {kind_counts.get('revision', 0)} revised")


def _new_recovery_checkpoint_if_absent(context: CollectionContext, arxiv_category: str,
                                       previous_complete_poll: str | None) -> SourceCheckpoint | None:
    """A dated search from the submission lag before the last complete poll (or before now) up to today."""
    window_anchor = parse_utc_timestamp(previous_complete_poll) if previous_complete_poll else context.started_at
    recovery_window = PublicationWindow((window_anchor - ARXIV_SUBMISSION_LAG).date(), context.started_at.date())
    recovery_key = (f"{ARXIV_RECOVERY_CHECKPOINT_PREFIX}{arxiv_category}:{recovery_window.first_date.isoformat()}.."
                    f"{recovery_window.last_date.isoformat()}")
    if load_source_checkpoint(context.connection, recovery_key) is not None:
        return None  # the same window is already being recovered
    category_query_id = _category_query_id(context, arxiv_category)
    if category_query_id is None:
        return None
    target = SearchTarget("arxiv", category_query_id, recovery_window, arxiv_recovery_route(arxiv_category),
                          RouteKind.CURRENT_RELEASE)
    return SourceCheckpoint(search_key=recovery_key, source_name="arxiv",
                            query_parameters=search_parameters(target, arxiv_category),
                            stop_reason="scheduled after a feed coverage gap")


# --------------------------------------------------------------------------------------------------------------------
# Missed-release recovery
# --------------------------------------------------------------------------------------------------------------------


def run_open_recovery_searches(context: CollectionContext) -> list[str]:
    report_lines = []
    approved_categories = set(approved_arxiv_categories(context))
    for recovery_checkpoint in load_source_checkpoints_with_prefix(context.connection,
                                                                   ARXIV_RECOVERY_CHECKPOINT_PREFIX):
        arxiv_category = recovery_checkpoint.query_parameters.get(PARAMETER_PUBLIC_VALUE)
        category_query_id = _category_query_id(context, arxiv_category) if arxiv_category in approved_categories \
            else None
        if category_query_id is None or not is_search_resumable(recovery_checkpoint):
            continue  # finished, or its category is no longer approved
        target = SearchTarget("arxiv", category_query_id, window_from_parameters(
            recovery_checkpoint.query_parameters), arxiv_recovery_route(arxiv_category), RouteKind.CURRENT_RELEASE)
        summary = run_search_pages(context, recovery_checkpoint, target,
                                   maximum_pages=context.app_config.collection.pages_per_search_per_run)
        report_lines.append(f"Recovery {summary.describe()}")
    return report_lines


def _category_query_id(context: CollectionContext, arxiv_category: str | None) -> str | None:
    for query_id, public_query in build_public_query_catalog(context.app_config.topics).items():
        if public_query.query_kind == QUERY_KIND_CATEGORY and public_query.public_value == arxiv_category:
            return query_id
    return None


# --------------------------------------------------------------------------------------------------------------------
# Recent OpenAlex windows
# --------------------------------------------------------------------------------------------------------------------


def sweep_recent_openalex_windows(context: CollectionContext) -> list[str]:
    report_lines = []
    collection_settings = context.app_config.collection
    for query_id, public_query in build_public_query_catalog(context.app_config.topics).items():
        if public_query.provider_name != "openalex" or public_query.query_kind != QUERY_KIND_KEYWORDS:
            continue
        recent_key = f"{OPENALEX_RECENT_CHECKPOINT_PREFIX}{query_id}"
        recent_checkpoint = load_source_checkpoint(context.connection, recent_key)
        if recent_checkpoint is None or not _has_unfinished_window(recent_checkpoint):
            sweep_interval_hours = collection_settings.openalex_recent_sweep_interval_hours
            if recent_checkpoint is not None and _swept_recently(recent_checkpoint, context.started_at,
                                                                 sweep_interval_hours):
                continue
            recent_checkpoint = _start_new_recent_window(context, query_id, public_query.public_value,
                                                         public_query.topic_id, recent_checkpoint)
        target = SearchTarget("openalex", query_id, window_from_parameters(recent_checkpoint.query_parameters),
                              openalex_recent_route(query_id), RouteKind.CURRENT_RELEASE)
        summary = run_search_pages(context, recent_checkpoint, target,
                                   maximum_pages=collection_settings.pages_per_search_per_run)
        report_lines.append(f"OpenAlex recent {summary.describe()}")
    return report_lines


def _has_unfinished_window(checkpoint: SourceCheckpoint) -> bool:
    """A window to resume first. One that served its first page and needs a key for more counts as finished."""
    return window_from_parameters(checkpoint.query_parameters) is not None and is_search_resumable(checkpoint) \
        and not stopped_for_a_missing_credential(checkpoint)


def _swept_recently(checkpoint: SourceCheckpoint, now: datetime, sweep_interval_hours: int) -> bool:
    if not checkpoint.last_completed_interval_end:
        return False
    return now - parse_utc_timestamp(checkpoint.last_completed_interval_end) < timedelta(hours=sweep_interval_hours)


def _start_new_recent_window(context: CollectionContext, query_id: str, public_value: str, topic_id: str,
                             previous_checkpoint: SourceCheckpoint | None) -> SourceCheckpoint:
    """From `openalex_recent_window_days` before the last completed sweep (or before today) up to today."""
    window_days = timedelta(days=context.app_config.collection.openalex_recent_window_days)
    anchor = parse_utc_timestamp(previous_checkpoint.last_completed_interval_end) \
        if previous_checkpoint and previous_checkpoint.last_completed_interval_end else context.started_at
    recent_window = PublicationWindow((anchor - window_days).date(), context.started_at.date())
    target = SearchTarget("openalex", query_id, recent_window, openalex_recent_route(query_id),
                          RouteKind.CURRENT_RELEASE)
    base_checkpoint = previous_checkpoint or SourceCheckpoint(search_key=f"{OPENALEX_RECENT_CHECKPOINT_PREFIX}"
                                                                         f"{query_id}", source_name="openalex",
                                                              topic_id=topic_id)
    return replace(base_checkpoint, query_parameters=search_parameters(target, public_value), cursor=None,
                   coverage_status=SearchCoverageStatus.ACTIVE, stop_reason=None)
