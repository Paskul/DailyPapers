"""Current releases: arXiv category feeds, missed-release recovery, and recent OpenAlex windows.

arXiv category feeds (checkpoint `arxiv-feed:<category>`)
  * Every approved category is polled once per run with If-None-Match, so an unchanged feed costs one 304.
    The feed changes once per announcement day; a later poll of the same feed finds the same records (no-ops).
  * Each item is labelled new, cross-listed, or revised (paper_discoveries / paper_editions.announcement_kind).
  * The checkpoint's `last_completed_interval_end` is the last poll whose items were all stored.
  * Fair shares (owner decision 2026-11-21): ALL feeds are fetched first, then the category-feed limit is split
    fairly over what each category can use (collection/candidate_budget.py), then each feed is stored. A feed with
    more new records than its share admits the ones that share the most words with its topics (question and public
    phrases, `rank_records_by_topic_words`), so a broad feed such as cs.AI fills its share with its most relevant
    papers. The other new records of that announcement are deliberately not admitted: this is a chosen limit, not a
    coverage gap, so no recovery search is scheduled for it (only the whole daily budget running out schedules one).

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

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from paperwatch.catalog import prepare_catalog_record, would_create_new_paper
from paperwatch.clock import parse_utc_timestamp
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.page_ingestion import PageIngestionResult, ingest_records_with_checkpoints
from paperwatch.collection.paged_search import (
    PARAMETER_PUBLIC_VALUE,
    SearchTarget,
    is_search_resumable,
    mark_search_blocked,
    run_search_pages,
    search_parameters,
    stopped_for_a_missing_credential,
    window_from_parameters,
)
from paperwatch.collection.topic_word_ranking import rank_records_by_topic_words, topic_words_for_category
from paperwatch.errors import ProviderUnavailable, SourceAccessDenied, SourceInputError
from paperwatch.network.provider_gateway import ProviderResponse
from paperwatch.network.provider_requests import PublicationWindow
from paperwatch.public_query_catalog import QUERY_KIND_CATEGORY, QUERY_KIND_KEYWORDS, build_public_query_catalog
from paperwatch.source_checkpoints import (
    SearchCoverageStatus,
    SourceCheckpoint,
    load_source_checkpoint,
    load_source_checkpoints_with_prefix,
)
from paperwatch.sources.arxiv_atom import ParsedFeed, parse_category_feed
from paperwatch.sources.discovery_routes import RouteKind, arxiv_feed_route, arxiv_recovery_route, openalex_recent_route
from paperwatch.sources.types import CollectedPaperRecord

ARXIV_FEED_SAFE_GAP = timedelta(hours=24)  # a feed shows each announcement for at least this long
ARXIV_SUBMISSION_LAG = timedelta(days=4)  # submissions are announced up to ~4 days later (weekends)
ARXIV_FEED_CHECKPOINT_PREFIX = "arxiv-feed:"
ARXIV_RECOVERY_CHECKPOINT_PREFIX = "arxiv-recovery:"
OPENALEX_RECENT_CHECKPOINT_PREFIX = "openalex-recent:"
FEED_ETAG_RESPONSE_HEADER = "etag"


def collect_current_releases(context: CollectionContext) -> list[str]:
    """Feeds first (fetched, shared fairly, then stored), then any open recovery windows, then recent OpenAlex
    windows. Returns lines for the operator."""
    report_lines: list[str] = []
    polled_feeds = [
        fetch_category_feed(context, arxiv_category) for arxiv_category in approved_arxiv_categories(context)
    ]
    plan_fair_category_shares(context, polled_feeds)
    for polled_feed in polled_feeds:
        report_lines.append(store_category_feed(context, polled_feed))
    report_lines += run_open_recovery_searches(context)
    report_lines += sweep_recent_openalex_windows(context)
    return report_lines


def poll_category_feed(context: CollectionContext, arxiv_category: str) -> str:
    """One feed on its own (fetch, share, store). The run uses collect_current_releases, which shares across feeds."""
    polled_feed = fetch_category_feed(context, arxiv_category)
    plan_fair_category_shares(context, [polled_feed])
    return store_category_feed(context, polled_feed)


def approved_arxiv_categories(context: CollectionContext) -> list[str]:
    """Each approved category once, in topics.toml order (several topics may share one)."""
    categories = [
        public_query.public_value
        for public_query in build_public_query_catalog(context.app_config.topics).values()
        if public_query.query_kind == QUERY_KIND_CATEGORY
    ]
    return list(dict.fromkeys(categories))


# --------------------------------------------------------------------------------------------------------------------
# arXiv category feeds
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class PolledCategoryFeed:
    """One category feed after its request: the parsed feed (None = HTTP 304), or the line saying why it was not
    polled (blocked)."""

    arxiv_category: str
    feed_checkpoint: SourceCheckpoint
    previous_complete_poll: str | None
    stored_etag: str | None
    feed_response: ProviderResponse | None = None
    parsed_feed: ParsedFeed | None = None
    blocked_report_line: str | None = None

    @property
    def feed_records(self) -> list[CollectedPaperRecord]:
        return self.parsed_feed.records if self.parsed_feed else []

    def has_coverage_gap(self, run_started_at: datetime) -> bool:
        """No complete poll for longer than a feed shows an announcement: a recovery search will be scheduled."""
        return (
            self.previous_complete_poll is not None
            and run_started_at - parse_utc_timestamp(self.previous_complete_poll) > ARXIV_FEED_SAFE_GAP
        )


def fetch_category_feed(context: CollectionContext, arxiv_category: str) -> PolledCategoryFeed:
    feed_key = f"{ARXIV_FEED_CHECKPOINT_PREFIX}{arxiv_category}"
    feed_checkpoint = load_source_checkpoint(context.connection, feed_key) or SourceCheckpoint(
        search_key=feed_key, source_name="arxiv", query_parameters={"category": arxiv_category}
    )
    previous_complete_poll = feed_checkpoint.last_completed_interval_end
    polled_feed = PolledCategoryFeed(
        arxiv_category,
        feed_checkpoint,
        previous_complete_poll,
        feed_checkpoint.cursor if previous_complete_poll else None,
    )
    try:
        polled_feed.feed_response = context.gateway.poll_category_feed(
            arxiv_category, if_none_match_etag=polled_feed.stored_etag
        )
        polled_feed.parsed_feed = (
            None
            if polled_feed.feed_response.was_not_modified
            else parse_category_feed(polled_feed.feed_response.body_bytes, arxiv_feed_route(arxiv_category))
        )
    except (ProviderUnavailable, SourceAccessDenied, SourceInputError) as problem:
        blocked_checkpoint = mark_search_blocked(context, feed_checkpoint, problem)
        polled_feed.blocked_report_line = (
            f"arXiv feed {arxiv_category}: NOT polled ({blocked_checkpoint.stop_reason}"
            f"); the last complete poll was {previous_complete_poll or 'never'}"
        )
    return polled_feed


def plan_fair_category_shares(context: CollectionContext, polled_feeds: list[PolledCategoryFeed]) -> None:
    """Water-fill the category-feed limit over every approved category: the new records its feed brings now, and an
    unbounded demand for a category with an open recovery search, or whose feed has a coverage gap (its recovery
    search is scheduled by this run)."""
    waiting_by_category: dict[str, int | None] = {
        arxiv_category: 0 for arxiv_category in approved_arxiv_categories(context)
    }
    for polled_feed in polled_feeds:
        waiting_by_category[polled_feed.arxiv_category] = (
            None
            if polled_feed.has_coverage_gap(context.started_at)
            else count_new_records(context, polled_feed.feed_records)
        )
    for recovery_checkpoint in load_source_checkpoints_with_prefix(
        context.connection, ARXIV_RECOVERY_CHECKPOINT_PREFIX
    ):
        arxiv_category = recovery_checkpoint.query_parameters.get(PARAMETER_PUBLIC_VALUE)
        if arxiv_category in waiting_by_category and is_search_resumable(recovery_checkpoint):
            waiting_by_category[arxiv_category] = None
    context.candidate_budget.plan_category_feed_shares(waiting_by_category)


def count_new_records(context: CollectionContext, collected_records: Iterable[CollectedPaperRecord]) -> int:
    """Records that would add a paper to the catalog (read-only). A paper cross-listed in two fetched feeds counts in
    both; the second feed then admits fewer than its share, and the rest stays in the daily budget."""
    new_record_count = 0
    for collected_record in collected_records:
        try:
            new_record_count += would_create_new_paper(context.connection, prepare_catalog_record(collected_record))
        except SourceInputError:
            continue  # malformed: skipped and counted at ingestion
    return new_record_count


def store_category_feed(context: CollectionContext, polled_feed: PolledCategoryFeed) -> str:
    if polled_feed.blocked_report_line is not None:
        return polled_feed.blocked_report_line
    arxiv_category, feed_checkpoint = polled_feed.arxiv_category, polled_feed.feed_checkpoint
    previous_complete_poll, parsed_feed = polled_feed.previous_complete_poll, polled_feed.parsed_feed
    feed_route = arxiv_feed_route(arxiv_category)
    gap_is_too_long = polled_feed.has_coverage_gap(context.started_at)
    feed_response_headers = polled_feed.feed_response.response_headers if polled_feed.feed_response else {}
    response_etag = feed_response_headers.get(FEED_ETAG_RESPONSE_HEADER) or polled_feed.stored_etag
    admission_limit = context.candidate_budget.remaining_for_route(feed_route)
    # Which limit binds: this category's fair share (a chosen limit), or the whole day's budget (a coverage gap).
    share_is_the_binding_limit = (
        context.candidate_budget.remaining_share_of_category(arxiv_category) < context.candidate_budget.remaining_today
    )
    feed_records = rank_records_by_topic_words(
        polled_feed.feed_records, topic_words_for_category(context.app_config.topics, arxiv_category)
    )
    scheduled_recovery_keys: list[str] = []

    def feed_checkpoints_after_page(page_result: PageIngestionResult) -> list[SourceCheckpoint]:
        """Inside the page transaction: the feed's own checkpoint, plus a recovery search if coverage has a gap."""
        checkpoints: list[SourceCheckpoint] = []
        cut_by_daily_budget = page_result.was_stopped_by_admission_limit and not share_is_the_binding_limit
        if gap_is_too_long or cut_by_daily_budget:
            recovery_checkpoint = _new_recovery_checkpoint_if_absent(context, arxiv_category, previous_complete_poll)
            if recovery_checkpoint is not None:
                checkpoints.append(recovery_checkpoint)
                scheduled_recovery_keys.append(recovery_checkpoint.search_key)
        if cut_by_daily_budget:
            # The recovery search now owns the uncovered window; forget the ETag so the next poll re-reads the feed.
            completed_feed = replace(
                feed_checkpoint,
                cursor=None,
                coverage_status=SearchCoverageStatus.CAPPED,
                stop_reason="daily candidate budget reached after part of the feed; the rest "
                "is left to the recovery search",
                last_completed_interval_end=context.started_at_timestamp,
            )
        else:
            share_note = (
                f"; fair share of the category feeds reached: {page_result.skipped_over_admission_limit} "
                f"new record(s) with the fewest topic words not admitted"
                if page_result.was_stopped_by_admission_limit
                else ""
            )
            completed_feed = replace(
                feed_checkpoint,
                cursor=response_etag,
                coverage_status=SearchCoverageStatus.EXHAUSTED,
                stop_reason=_describe_feed_poll(parsed_feed) + share_note,
                last_completed_interval_end=context.started_at_timestamp,
            )
        return checkpoints + [
            replace(
                completed_feed,
                candidates_admitted=feed_checkpoint.candidates_admitted + page_result.new_papers_admitted,
            )
        ]

    page_result = ingest_records_with_checkpoints(
        context.connection,
        feed_records,
        admission_limit=admission_limit,
        active_seed_paper_ids=context.active_seed_paper_ids,
        max_attempts_per_job=context.app_config.worker.max_attempts_per_job,
        checkpoints_after_page=feed_checkpoints_after_page,
    )
    if parsed_feed is not None:
        page_result.skipped_malformed_records += sum(parsed_feed.skipped_entry_counts.values())
    context.candidate_budget.record_route_admissions(feed_route, page_result.new_papers_admitted)

    label = f"arXiv feed {arxiv_category}"
    report_line = (
        f"{label}: unchanged since the last poll (HTTP 304)"
        if parsed_feed is None
        else page_result.describe(f"{label} ({_describe_feed_poll(parsed_feed)})")
    )
    if page_result.was_stopped_by_admission_limit and share_is_the_binding_limit:
        report_line = report_line.replace(
            "new record(s) left for later (candidate budget)",
            "new record(s) with the fewest topic words not admitted (fair share of the category feeds reached)",
        )
    for recovery_key in scheduled_recovery_keys:
        report_line += f"\n  coverage gap recorded; recovery search {recovery_key} scheduled"
    return report_line


def _describe_feed_poll(parsed_feed: ParsedFeed | None) -> str:
    if parsed_feed is None:
        return "unchanged (HTTP 304)"
    kind_counts = parsed_feed.announcement_kind_counts
    return (
        f"announcement {parsed_feed.announcement_date or 'none'}: {kind_counts.get('new', 0)} new, "
        f"{kind_counts.get('cross_list', 0)} cross-listed, {kind_counts.get('revision', 0)} revised"
    )


def _new_recovery_checkpoint_if_absent(
    context: CollectionContext, arxiv_category: str, previous_complete_poll: str | None
) -> SourceCheckpoint | None:
    """A dated search from the submission lag before the last complete poll (or before now) up to today."""
    window_anchor = parse_utc_timestamp(previous_complete_poll) if previous_complete_poll else context.started_at
    recovery_window = PublicationWindow((window_anchor - ARXIV_SUBMISSION_LAG).date(), context.started_at.date())
    recovery_key = (
        f"{ARXIV_RECOVERY_CHECKPOINT_PREFIX}{arxiv_category}:{recovery_window.first_date.isoformat()}.."
        f"{recovery_window.last_date.isoformat()}"
    )
    if load_source_checkpoint(context.connection, recovery_key) is not None:
        return None  # the same window is already being recovered
    category_query_id = _category_query_id(context, arxiv_category)
    if category_query_id is None:
        return None
    target = SearchTarget(
        "arxiv", category_query_id, recovery_window, arxiv_recovery_route(arxiv_category), RouteKind.CURRENT_RELEASE
    )
    return SourceCheckpoint(
        search_key=recovery_key,
        source_name="arxiv",
        query_parameters=search_parameters(target, arxiv_category),
        stop_reason="scheduled after a feed coverage gap",
    )


# --------------------------------------------------------------------------------------------------------------------
# Missed-release recovery
# --------------------------------------------------------------------------------------------------------------------


def run_open_recovery_searches(context: CollectionContext) -> list[str]:
    report_lines = []
    approved_categories = set(approved_arxiv_categories(context))
    for recovery_checkpoint in load_source_checkpoints_with_prefix(
        context.connection, ARXIV_RECOVERY_CHECKPOINT_PREFIX
    ):
        arxiv_category = recovery_checkpoint.query_parameters.get(PARAMETER_PUBLIC_VALUE)
        if not isinstance(arxiv_category, str) or arxiv_category not in approved_categories:
            continue  # its category is no longer approved
        category_query_id = _category_query_id(context, arxiv_category)
        if category_query_id is None or not is_search_resumable(recovery_checkpoint):
            continue  # finished, or its category is no longer approved
        target = SearchTarget(
            "arxiv",
            category_query_id,
            window_from_parameters(recovery_checkpoint.query_parameters),
            arxiv_recovery_route(arxiv_category),
            RouteKind.CURRENT_RELEASE,
        )
        summary = run_search_pages(
            context, recovery_checkpoint, target, maximum_pages=context.app_config.collection.pages_per_search_per_run
        )
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
            if recent_checkpoint is not None and _swept_recently(
                recent_checkpoint, context.started_at, sweep_interval_hours
            ):
                continue
            recent_checkpoint = _start_new_recent_window(
                context, query_id, public_query.public_value, public_query.topic_id, recent_checkpoint
            )
        target = SearchTarget(
            "openalex",
            query_id,
            window_from_parameters(recent_checkpoint.query_parameters),
            openalex_recent_route(query_id),
            RouteKind.CURRENT_RELEASE,
        )
        summary = run_search_pages(
            context, recent_checkpoint, target, maximum_pages=collection_settings.pages_per_search_per_run
        )
        report_lines.append(f"OpenAlex recent {summary.describe()}")
    return report_lines


def _has_unfinished_window(checkpoint: SourceCheckpoint) -> bool:
    """A window to resume first. One that served its first page and needs a key for more counts as finished."""
    return (
        window_from_parameters(checkpoint.query_parameters) is not None
        and is_search_resumable(checkpoint)
        and not stopped_for_a_missing_credential(checkpoint)
    )


def _swept_recently(checkpoint: SourceCheckpoint, now: datetime, sweep_interval_hours: int) -> bool:
    if not checkpoint.last_completed_interval_end:
        return False
    return now - parse_utc_timestamp(checkpoint.last_completed_interval_end) < timedelta(hours=sweep_interval_hours)


def _start_new_recent_window(
    context: CollectionContext,
    query_id: str,
    public_value: str,
    topic_id: str,
    previous_checkpoint: SourceCheckpoint | None,
) -> SourceCheckpoint:
    """From `openalex_recent_window_days` before the last completed sweep (or before today) up to today."""
    window_days = timedelta(days=context.app_config.collection.openalex_recent_window_days)
    anchor = (
        parse_utc_timestamp(previous_checkpoint.last_completed_interval_end)
        if previous_checkpoint and previous_checkpoint.last_completed_interval_end
        else context.started_at
    )
    recent_window = PublicationWindow((anchor - window_days).date(), context.started_at.date())
    target = SearchTarget(
        "openalex", query_id, recent_window, openalex_recent_route(query_id), RouteKind.CURRENT_RELEASE
    )
    base_checkpoint = previous_checkpoint or SourceCheckpoint(
        search_key=f"{OPENALEX_RECENT_CHECKPOINT_PREFIX}{query_id}", source_name="openalex", topic_id=topic_id
    )
    return replace(
        base_checkpoint,
        query_parameters=search_parameters(target, public_value),
        cursor=None,
        coverage_status=SearchCoverageStatus.ACTIVE,
        stop_reason=None,
    )
