"""The finite historical search of each topic ("Older papers").

On first use, or after a MATERIAL change, each topic gets one historical search plan (checkpoint
`history-plan:<topic id>:<hash>`) with a candidate budget of `historical_candidates_per_topic`. Material means the
topic's approved search terms or the recency schedule settings changed. Changing only the private question is not
material, because nothing that is sent changes.

Which queries
  * keyword topics: each public phrase, on arXiv and on OpenAlex;
  * seeds-only topics (no phrases): each arXiv category (category-and-date queries); the results are ranked against
    the seeds locally during screening.

Which publication dates: recency-weighted windows (recency_schedule.py)
The plan cuts the time from its creation date back to `historical_earliest_year` into equal-mass windows of an
exponential recency curve (`historical_half_life_years`), so candidates per year decay exponentially with age while
the oldest era keeps a guaranteed share. Old windows ask for their most-cited papers first. The windows are stored in
the plan, so its cursors stay valid. Each (query, window) is one paged search with its own checkpoint
`history:<query id>:<first day>..<last day>`: its query, cursor, stopping reason, and count.

How pages are scheduled: least-served first (weighted fair share)
Every (query, window) unit has a target share (its window's weight, split evenly between the queries that can search
it). The next page always goes to the resumable unit with the smallest ratio of candidates admitted to its share; ties
go to the newer window. A page may admit at most the unit's FAIR QUOTA: the pool still open to the units that can
continue (the plan's remaining budget plus what those units already hold), split by their shares, minus what this
unit already holds. Real pages hold 100 records while a share may be ~20, so without this quota the first pages
would spend the whole budget on the newest windows. A page cut short keeps its cursor and is admitted later.
Consequences:
  * each window's admitted count follows its share, so the exponential holds even with large pages;
  * a window that runs dry (an era with few matches, or arXiv before 1991) simply stops being chosen, and its unused
    share flows to the windows that still have results, so the budget is never stranded;
  * the state is just the admitted counts in the checkpoints, so scheduling resumes exactly after a restart.
Each run fetches at most `historical_pages_per_topic_per_run` pages per topic, and pages admit new papers only while
both the plan and the day's historical allowance (what current releases leave) have capacity. The plan is `capped`
when its budget is used and `exhausted` when no unit has another page.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.page_ingestion import PageIngestionResult
from paperwatch.collection.paged_search import (
    SearchTarget,
    is_search_resumable,
    run_search_pages,
    search_parameters,
    stopped_for_a_missing_credential,
)
from paperwatch.collection.recency_schedule import (
    HistoricalWindow,
    RecencyScheduleSettings,
    build_recency_weighted_windows,
)
from paperwatch.config import CollectionSettings
from paperwatch.db import write_transaction
from paperwatch.public_query_catalog import (
    QUERY_KIND_CATEGORY,
    QUERY_KIND_KEYWORDS,
    PublicQuery,
    build_public_query_catalog,
)
from paperwatch.research_profile import build_research_profiles
from paperwatch.source_checkpoints import (
    SearchCoverageStatus,
    SourceCheckpoint,
    load_existing_source_checkpoint,
    load_source_checkpoint,
    save_source_checkpoint,
)
from paperwatch.sources.discovery_routes import RouteKind, historical_search_route

HISTORY_PLAN_CHECKPOINT_PREFIX = "history-plan:"
HISTORY_WINDOW_CHECKPOINT_PREFIX = "history:"
PLAN_KEY_HASH_CHARACTERS = 10
PLAN_PARAMETER_WINDOWS = "windows"


@dataclass(frozen=True)
class HistoricalSearchUnit:
    """One paged search of the plan: one approved query over one window, with its target share of the budget."""

    public_query: PublicQuery
    window: HistoricalWindow
    window_position: int  # 0 = newest window (wins ties)
    target_share: float

    @property
    def checkpoint_key(self) -> str:
        return f"{HISTORY_WINDOW_CHECKPOINT_PREFIX}{self.public_query.query_id}:{self.window.label}"

    @property
    def search_target(self) -> SearchTarget:
        return SearchTarget(
            self.public_query.provider_name,
            self.public_query.query_id,
            self.window.publication_window,
            historical_search_route(self.public_query.query_id),
            RouteKind.HISTORICAL_SEARCH,
            self.window.result_order,
        )


def recency_schedule_settings(collection_settings: CollectionSettings) -> RecencyScheduleSettings:
    return RecencyScheduleSettings(
        earliest_year=collection_settings.historical_earliest_year,
        half_life_years=collection_settings.historical_half_life_years,
        window_count=collection_settings.historical_window_count,
        citation_ordering_min_age_years=collection_settings.historical_citation_ordering_min_age_years,
    )


def historical_queries_for_topic(topic_id: str, query_catalog: dict[str, PublicQuery]) -> list[PublicQuery]:
    """Keyword topics search their phrases (arXiv and OpenAlex); seeds-only topics search their categories."""
    topic_queries = [public_query for public_query in query_catalog.values() if public_query.topic_id == topic_id]
    keyword_queries = [public_query for public_query in topic_queries if public_query.query_kind == QUERY_KIND_KEYWORDS]
    return keyword_queries or [
        public_query for public_query in topic_queries if public_query.query_kind == QUERY_KIND_CATEGORY
    ]


def history_plan_key(
    topic_id: str, historical_queries: list[PublicQuery], schedule_settings: RecencyScheduleSettings
) -> str:
    """Changes when the approved queries or the schedule settings change (a material change: a new plan)."""
    plan_identity_text = json.dumps(
        {
            "query_ids": sorted(public_query.query_id for public_query in historical_queries),
            "schedule": schedule_settings.to_parameters(),
        },
        sort_keys=True,
    )
    plan_hash = hashlib.sha256(plan_identity_text.encode("utf-8")).hexdigest()[:PLAN_KEY_HASH_CHARACTERS]
    return f"{HISTORY_PLAN_CHECKPOINT_PREFIX}{topic_id}:{plan_hash}"


def build_search_units(
    historical_queries: list[PublicQuery], windows: list[HistoricalWindow]
) -> list[HistoricalSearchUnit]:
    """Every (query, window) pair a provider can answer, each with its share of the window's weight."""
    search_units = []
    for window_position, window in enumerate(windows):
        searchable_queries = [
            public_query for public_query in historical_queries if window.is_searchable_on(public_query.provider_name)
        ]
        for public_query in searchable_queries:
            search_units.append(
                HistoricalSearchUnit(public_query, window, window_position, window.weight / len(searchable_queries))
            )
    return search_units


def choose_least_served_unit(
    search_units: list[HistoricalSearchUnit], admitted_by_unit_key: dict[str, int]
) -> HistoricalSearchUnit | None:
    """The unit furthest below its share: smallest admitted / share, then the newer window, then query order."""
    if not search_units:
        return None
    return min(
        search_units,
        key=lambda unit: (admitted_by_unit_key.get(unit.checkpoint_key, 0) / unit.target_share, unit.window_position),
    )


def fair_admission_quota(
    unit: HistoricalSearchUnit,
    search_units: list[HistoricalSearchUnit],
    unit_checkpoints: dict[str, SourceCheckpoint],
    plan_remaining: int,
) -> int:
    """How many new papers this unit's next page may admit (at least 1, so the search always makes progress).

    Units that can no longer add papers (no more pages, or only a first page without a key) drop out of the pool, so
    their unused share is shared by the rest."""
    continuing_units = [
        candidate
        for candidate in search_units
        if is_search_resumable(unit_checkpoints[candidate.checkpoint_key])
        and not stopped_for_a_missing_credential(unit_checkpoints[candidate.checkpoint_key])
    ]
    if unit not in continuing_units:
        return max(1, plan_remaining)
    pool = plan_remaining + sum(
        unit_checkpoints[candidate.checkpoint_key].candidates_admitted for candidate in continuing_units
    )
    share_of_pool = unit.target_share / sum(candidate.target_share for candidate in continuing_units)
    return max(1, math.ceil(pool * share_of_pool) - unit_checkpoints[unit.checkpoint_key].candidates_admitted)


def run_historical_searches(context: CollectionContext) -> list[str]:
    report_lines = []
    query_catalog = build_public_query_catalog(context.app_config.topics)
    for profile in build_research_profiles(context.connection, context.app_config):
        historical_queries = historical_queries_for_topic(profile.topic_id, query_catalog)
        if historical_queries:
            report_lines.append(run_topic_historical_search(context, profile.topic_id, historical_queries))
    return report_lines


def historical_search_has_work(context: CollectionContext) -> bool:
    """True while some topic's historical plan is not started yet, or has budget left and is not exhausted."""
    query_catalog = build_public_query_catalog(context.app_config.topics)
    schedule_settings = recency_schedule_settings(context.app_config.collection)
    for topic in context.app_config.topics:
        historical_queries = historical_queries_for_topic(topic.topic_id, query_catalog)
        if not historical_queries:
            continue
        plan_checkpoint = load_source_checkpoint(
            context.connection, history_plan_key(topic.topic_id, historical_queries, schedule_settings)
        )
        if plan_checkpoint is None or (
            plan_checkpoint.coverage_status != SearchCoverageStatus.EXHAUSTED
            and plan_checkpoint.remaining_candidate_budget != 0
        ):
            return True
    return False


def run_topic_historical_search(
    context: CollectionContext, topic_id: str, historical_queries: list[PublicQuery]
) -> str:
    collection_settings = context.app_config.collection
    schedule_settings = recency_schedule_settings(collection_settings)
    plan_key = history_plan_key(topic_id, historical_queries, schedule_settings)
    plan_checkpoint = load_source_checkpoint(context.connection, plan_key)
    if plan_checkpoint is None:  # first use, or a material change: fix the windows now and store them
        plan_checkpoint = _new_plan_checkpoint(
            plan_key,
            topic_id,
            historical_queries,
            schedule_settings,
            build_recency_weighted_windows(context.started_at.date(), schedule_settings),
            collection_settings.historical_candidates_per_topic,
        )
        with write_transaction(context.connection):
            save_source_checkpoint(context.connection, plan_checkpoint)
    if (
        plan_checkpoint.coverage_status == SearchCoverageStatus.EXHAUSTED
        or plan_checkpoint.remaining_candidate_budget == 0
    ):
        return _describe_plan(plan_checkpoint, pages_fetched=0, newly_admitted=0)

    search_units = build_search_units(historical_queries, plan_windows(plan_checkpoint))
    unit_checkpoints = {
        unit.checkpoint_key: _load_or_new_unit_checkpoint(context, topic_id, unit) for unit in search_units
    }
    unit_keys_done_this_run: set[str] = set()  # blocked or first-page-only: not again in this run
    pages_fetched = newly_admitted = 0
    stopped_by_daily_allowance = False
    while pages_fetched < collection_settings.historical_pages_per_topic_per_run:
        if _plan_remaining(context, plan_key) == 0:
            break
        if context.candidate_budget.remaining_for(RouteKind.HISTORICAL_SEARCH) == 0:
            stopped_by_daily_allowance = True
            break
        available_units = [
            unit
            for unit in search_units
            if unit.checkpoint_key not in unit_keys_done_this_run
            and is_search_resumable(unit_checkpoints[unit.checkpoint_key])
        ]
        chosen_unit = choose_least_served_unit(
            available_units, {key: checkpoint.candidates_admitted for key, checkpoint in unit_checkpoints.items()}
        )
        if chosen_unit is None:
            break
        fair_quota = fair_admission_quota(
            chosen_unit, search_units, unit_checkpoints, _plan_remaining(context, plan_key) or 0
        )
        summary = run_search_pages(
            context,
            unit_checkpoints[chosen_unit.checkpoint_key],
            chosen_unit.search_target,
            maximum_pages=1,
            extra_admission_limit=lambda: min(fair_quota, _plan_remaining(context, plan_key) or 0),  # noqa: B023  # called synchronously, within this loop iteration
            related_checkpoints_after_page=lambda page_result: [_plan_after_page(context, plan_key, page_result)],
            extra_limit_stop_reason="this window's share of the topic's budget is used for now; it continues when "
            "the other windows have had theirs",
        )
        unit_checkpoints[chosen_unit.checkpoint_key] = summary.final_checkpoint
        if summary.pages_fetched == 0:
            unit_keys_done_this_run.add(chosen_unit.checkpoint_key)
        pages_fetched += summary.pages_fetched
        newly_admitted += summary.ingestion.new_papers_admitted

    plan_checkpoint = _settle_plan_coverage(
        context, plan_key, list(unit_checkpoints.values()), stopped_by_daily_allowance
    )
    return _describe_plan(plan_checkpoint, pages_fetched, newly_admitted)


def plan_windows(plan_checkpoint: SourceCheckpoint) -> list[HistoricalWindow]:
    return [
        HistoricalWindow.from_parameters(window_parameters)
        for window_parameters in plan_checkpoint.query_parameters[PLAN_PARAMETER_WINDOWS]
    ]


def _new_plan_checkpoint(
    plan_key: str,
    topic_id: str,
    historical_queries: list[PublicQuery],
    schedule_settings: RecencyScheduleSettings,
    windows: list[HistoricalWindow],
    candidate_budget: int,
) -> SourceCheckpoint:
    return SourceCheckpoint(
        search_key=plan_key,
        source_name="history-plan",
        topic_id=topic_id,
        query_parameters={
            "query_ids": [public_query.query_id for public_query in historical_queries],
            "schedule": schedule_settings.to_parameters(),
            PLAN_PARAMETER_WINDOWS: [window.to_parameters() for window in windows],
        },
        candidate_budget=candidate_budget,
        stop_reason="started",
    )


def _load_or_new_unit_checkpoint(
    context: CollectionContext, topic_id: str, unit: HistoricalSearchUnit
) -> SourceCheckpoint:
    existing_checkpoint = load_source_checkpoint(context.connection, unit.checkpoint_key)
    if existing_checkpoint is not None:
        return existing_checkpoint
    return SourceCheckpoint(
        search_key=unit.checkpoint_key,
        source_name=unit.public_query.provider_name,
        topic_id=topic_id,
        query_parameters=search_parameters(unit.search_target, unit.public_query.public_value),
    )


def _plan_remaining(context: CollectionContext, plan_key: str) -> int | None:
    plan_checkpoint = load_source_checkpoint(context.connection, plan_key)
    return plan_checkpoint.remaining_candidate_budget if plan_checkpoint else None


def _plan_after_page(context: CollectionContext, plan_key: str, page_result: PageIngestionResult) -> SourceCheckpoint:
    """Inside the page transaction: count the page's new papers against the topic's plan, and cap it when full."""
    plan_checkpoint = load_existing_source_checkpoint(context.connection, plan_key)
    candidates_admitted = plan_checkpoint.candidates_admitted + page_result.new_papers_admitted
    is_full = plan_checkpoint.candidate_budget is not None and candidates_admitted >= plan_checkpoint.candidate_budget
    return replace(
        plan_checkpoint,
        candidates_admitted=candidates_admitted,
        coverage_status=SearchCoverageStatus.CAPPED if is_full else SearchCoverageStatus.ACTIVE,
        stop_reason="topic's historical candidate budget reached" if is_full else "in progress",
    )


def _settle_plan_coverage(
    context: CollectionContext,
    plan_key: str,
    unit_checkpoints: list[SourceCheckpoint],
    stopped_by_daily_allowance: bool,
) -> SourceCheckpoint:
    """Record why the plan stopped this run: exhausted, waiting for the daily allowance or a key, or blocked."""
    plan_checkpoint = load_existing_source_checkpoint(context.connection, plan_key)
    if plan_checkpoint.remaining_candidate_budget == 0:
        return plan_checkpoint  # capped by its own budget inside the page transaction
    unfinished_units = [
        checkpoint for checkpoint in unit_checkpoints if checkpoint.coverage_status != SearchCoverageStatus.EXHAUSTED
    ]
    if not unfinished_units:
        new_status, stop_reason = SearchCoverageStatus.EXHAUSTED, "no query has another page in any window"
    elif stopped_by_daily_allowance:
        new_status, stop_reason = (
            SearchCoverageStatus.ACTIVE,
            ("today's historical allowance is used (current releases come first); resumes on a later run"),
        )
    elif all(stopped_for_a_missing_credential(checkpoint) for checkpoint in unfinished_units):
        new_status, stop_reason = (
            SearchCoverageStatus.ACTIVE,
            ("waiting: the remaining windows need an OpenAlex API key for further pages"),
        )
    elif any(checkpoint.coverage_status == SearchCoverageStatus.BLOCKED for checkpoint in unfinished_units):
        new_status, stop_reason = (
            SearchCoverageStatus.ACTIVE,
            ("some window searches are temporarily blocked; they resume on a later run"),
        )
    else:
        new_status, stop_reason = SearchCoverageStatus.ACTIVE, "in progress; resumes from saved cursors next run"
    plan_checkpoint = replace(
        plan_checkpoint,
        coverage_status=new_status,
        stop_reason=stop_reason,
        last_completed_interval_end=context.started_at_timestamp
        if new_status == SearchCoverageStatus.EXHAUSTED
        else None,
    )
    with write_transaction(context.connection):
        save_source_checkpoint(context.connection, plan_checkpoint)
    return plan_checkpoint


def _describe_plan(plan_checkpoint: SourceCheckpoint, pages_fetched: int, newly_admitted: int) -> str:
    return (
        f"Historical search {plan_checkpoint.search_key}: {pages_fetched} page(s) this run, {newly_admitted} new; "
        f"{plan_checkpoint.candidates_admitted} of {plan_checkpoint.candidate_budget} candidate(s) admitted in "
        f"total; {plan_checkpoint.coverage_status}"
        + (f" ({plan_checkpoint.stop_reason})" if plan_checkpoint.stop_reason else "")
    )
