"""Page through one checkpointed search: a missed-release recovery, a recent OpenAlex window, or one year band of a
topic's historical search.

The checkpoint row IS the search's memory: its query and window (query_parameters), cursor, admitted count, coverage,
and stopping reason. Per page:
  1. stop early if the search has no admission capacity left (daily budget, or its own or its plan's cap);
  2. fetch and parse one page OUTSIDE any transaction. On a provider outage, an access refusal (including a used-up
     request budget), or a malformed page, mark the search `blocked` with the reason and KEEP the cursor: the next run
     retries the same page;
  3. store the page's records and the advanced checkpoint in one transaction (page_ingestion.py). If the admission
     limit stopped the page part-way, the cursor stays on this page and the search is `capped` until capacity returns.
A search that runs out of pages is `exhausted`; its `last_completed_interval_end` is the run's start time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.page_ingestion import PageIngestionResult, ingest_records_with_checkpoints
from paperwatch.collection.topic_word_ranking import rank_records_by_topic_words, topic_words_for_category
from paperwatch.db import write_transaction
from paperwatch.errors import (
    PaperwatchError,
    ProviderUnavailable,
    SourceAccessDenied,
    SourceCredentialRequired,
    SourceInputError,
)
from paperwatch.network.provider_requests import PublicationWindow, SearchResultOrder
from paperwatch.source_checkpoints import SearchCoverageStatus, SourceCheckpoint, save_source_checkpoint
from paperwatch.sources.discovery_routes import RouteKind, arxiv_category_of_route
from paperwatch.sources.search_pages import fetch_search_page

STOP_REASON_DAILY_BUDGET = "daily candidate budget reached; resumes when capacity returns (next UTC day)"
STOP_REASON_FIRST_PAGE_ONLY = (
    "first page only: later pages need a provider API key (it resumes from this cursor once a key is configured)"
)
MAXIMUM_STORED_STOP_REASON_CHARACTERS = 300

# Query-parameter keys every paged search stores in its checkpoint (approved public values and dates only).
PARAMETER_QUERY_ID = "query_id"
PARAMETER_PROVIDER = "provider"
PARAMETER_PUBLIC_VALUE = "public_value"
PARAMETER_WINDOW_FIRST_DATE = "window_first_date"
PARAMETER_WINDOW_LAST_DATE = "window_last_date"
PARAMETER_RESULT_ORDER = "result_order"


@dataclass(frozen=True)
class SearchTarget:
    """What a paged search asks for and how its results are labelled."""

    provider_name: str
    query_id: str  # an approved catalog id (paperwatch.public_query_catalog)
    publication_window: PublicationWindow | None
    discovery_route: str
    route_kind: RouteKind
    result_order: SearchResultOrder = SearchResultOrder.PROVIDER_DEFAULT


@dataclass
class SearchRunSummary:
    search_key: str
    final_checkpoint: SourceCheckpoint  # where the search stands after this run (unchanged if no page was fetched)
    pages_fetched: int = 0
    ingestion: PageIngestionResult = field(default_factory=PageIngestionResult)

    def describe(self) -> str:
        checkpoint = self.final_checkpoint
        state = f"{checkpoint.coverage_status}" + (f" ({checkpoint.stop_reason})" if checkpoint.stop_reason else "")
        return (
            f"{self.search_key}: {self.pages_fetched} page(s), {self.ingestion.records_received} record(s), "
            f"{self.ingestion.new_papers_admitted} new, {self.ingestion.review_jobs_queued} new reading(s) "
            f"of promoted papers queued; {state}"
        )


def is_search_resumable(checkpoint: SourceCheckpoint) -> bool:
    """Active and blocked searches continue; a capped one continues only if its OWN cap is not what stopped it."""
    if checkpoint.coverage_status in (SearchCoverageStatus.ACTIVE, SearchCoverageStatus.BLOCKED):
        return True
    return checkpoint.coverage_status == SearchCoverageStatus.CAPPED and checkpoint.remaining_candidate_budget != 0


def stopped_for_a_missing_credential(checkpoint: SourceCheckpoint) -> bool:
    return checkpoint.stop_reason == STOP_REASON_FIRST_PAGE_ONLY


def search_parameters(target: SearchTarget, public_value: str) -> dict[str, str]:
    """What a checkpoint records about its search: only approved public values and the window dates."""
    parameters = {
        PARAMETER_QUERY_ID: target.query_id,
        PARAMETER_PROVIDER: target.provider_name,
        PARAMETER_PUBLIC_VALUE: public_value,
    }
    if target.publication_window is not None:
        parameters[PARAMETER_WINDOW_FIRST_DATE] = target.publication_window.first_date.isoformat()
        parameters[PARAMETER_WINDOW_LAST_DATE] = target.publication_window.last_date.isoformat()
    if target.result_order != SearchResultOrder.PROVIDER_DEFAULT:
        parameters[PARAMETER_RESULT_ORDER] = str(target.result_order)
    return parameters


def window_from_parameters(query_parameters: dict[str, Any]) -> PublicationWindow | None:
    if PARAMETER_WINDOW_FIRST_DATE not in query_parameters:
        return None
    return PublicationWindow(
        date.fromisoformat(query_parameters[PARAMETER_WINDOW_FIRST_DATE]),
        date.fromisoformat(query_parameters[PARAMETER_WINDOW_LAST_DATE]),
    )


def run_search_pages(
    context: CollectionContext,
    checkpoint: SourceCheckpoint,
    target: SearchTarget,
    *,
    maximum_pages: int,
    extra_admission_limit: Callable[[], int | None] = lambda: None,
    related_checkpoints_after_page: Callable[[PageIngestionResult], list[SourceCheckpoint]] | None = None,
    extra_limit_stop_reason: str = "candidate budget of this search reached",
) -> SearchRunSummary:
    """Fetch up to `maximum_pages` pages. `extra_admission_limit` adds a cap shared with other searches (a topic's
    historical plan and this search's fair share of it), and `extra_limit_stop_reason` explains a stop it causes;
    `related_checkpoints_after_page` returns rows to update in the same transaction (that plan)."""
    summary = SearchRunSummary(search_key=checkpoint.search_key, final_checkpoint=checkpoint)
    while summary.pages_fetched < maximum_pages and is_search_resumable(checkpoint):
        admission_limit = _current_admission_limit(context, checkpoint, target, extra_admission_limit())
        if admission_limit == 0:
            checkpoint = _save_checkpoint_alone(
                context,
                replace(
                    checkpoint,
                    coverage_status=SearchCoverageStatus.CAPPED,
                    stop_reason=STOP_REASON_DAILY_BUDGET
                    if context.candidate_budget.remaining_for_route(target.discovery_route) == 0
                    else extra_limit_stop_reason,
                ),
            )
            break
        try:
            parsed_page = fetch_search_page(
                context.gateway,
                provider_name=target.provider_name,
                query_id=target.query_id,
                cursor=checkpoint.cursor,
                publication_window=target.publication_window,
                discovery_route=target.discovery_route,
                result_order=target.result_order,
            )
        except SourceCredentialRequired:
            # Not an outage: this search covered what the provider serves without a key. Recent windows count as
            # complete (last_completed_interval_end), so the next sweep starts a new window.
            checkpoint = _save_checkpoint_alone(
                context,
                replace(
                    checkpoint,
                    coverage_status=SearchCoverageStatus.CAPPED,
                    stop_reason=STOP_REASON_FIRST_PAGE_ONLY,
                    last_completed_interval_end=context.started_at_timestamp,
                ),
            )
            break
        except (ProviderUnavailable, SourceAccessDenied, SourceInputError) as problem:
            checkpoint = mark_search_blocked(context, checkpoint, problem)
            break
        summary.pages_fetched += 1

        checkpoints_saved_with_page: list[SourceCheckpoint] = []

        def checkpoints_after_this_page(page_result: PageIngestionResult) -> list[SourceCheckpoint]:
            """Runs inside the page's transaction (called synchronously by the ingestion below)."""
            stop_reason_if_cut_short = (
                STOP_REASON_DAILY_BUDGET
                if context.candidate_budget.remaining_for_route(target.discovery_route)
                <= page_result.new_papers_admitted
                else extra_limit_stop_reason
            )
            advanced_checkpoint = _advance_checkpoint_after_page(
                context,
                checkpoint,  # noqa: B023  # called synchronously, within this loop iteration
                parsed_page.next_cursor,  # noqa: B023  # called synchronously, within this loop iteration
                parsed_page.total_results,  # noqa: B023  # called synchronously, within this loop iteration
                page_result,
                stop_reason_if_cut_short,
            )
            checkpoints_saved_with_page.append(advanced_checkpoint)  # noqa: B023  # called synchronously, within this loop iteration
            return [advanced_checkpoint] + (
                related_checkpoints_after_page(page_result) if related_checkpoints_after_page else []
            )

        page_records = parsed_page.records
        arxiv_category = arxiv_category_of_route(target.discovery_route)
        if arxiv_category is not None:  # a category route admits its best topic matches first (fair-share cap)
            page_records = rank_records_by_topic_words(
                page_records, topic_words_for_category(context.app_config.topics, arxiv_category)
            )
        page_result = ingest_records_with_checkpoints(
            context.connection,
            page_records,
            admission_limit=admission_limit,
            active_seed_paper_ids=context.active_seed_paper_ids,
            max_attempts_per_job=context.app_config.worker.max_attempts_per_job,
            checkpoints_after_page=checkpoints_after_this_page,
        )
        page_result.skipped_malformed_records += sum(parsed_page.skipped_entry_counts.values())
        context.candidate_budget.record_route_admissions(target.discovery_route, page_result.new_papers_admitted)
        summary.ingestion.add(page_result)
        checkpoint = checkpoints_saved_with_page[-1]
    summary.final_checkpoint = checkpoint
    return summary


def mark_search_blocked(
    context: CollectionContext, checkpoint: SourceCheckpoint, problem: PaperwatchError
) -> SourceCheckpoint:
    """Temporarily blocked: the cursor is kept, so the same page is retried on a later run."""
    return _save_checkpoint_alone(
        context,
        replace(
            checkpoint,
            coverage_status=SearchCoverageStatus.BLOCKED,
            stop_reason=f"[{problem.failed_component}] {problem}"[:MAXIMUM_STORED_STOP_REASON_CHARACTERS],
        ),
    )


def _advance_checkpoint_after_page(
    context: CollectionContext,
    page_checkpoint: SourceCheckpoint,
    next_cursor: str | None,
    total_results: int | None,
    page_result: PageIngestionResult,
    stop_reason_if_cut_short: str,
) -> SourceCheckpoint:
    if page_result.was_stopped_by_admission_limit:
        # Keep THIS page's cursor: the page is fetched again later and its remaining new papers admitted then.
        return replace(
            page_checkpoint,
            candidates_admitted=page_checkpoint.candidates_admitted + page_result.new_papers_admitted,
            coverage_status=SearchCoverageStatus.CAPPED,
            stop_reason=stop_reason_if_cut_short,
        )
    advanced_checkpoint = page_checkpoint.after_page(
        next_cursor=next_cursor, newly_admitted_candidates=page_result.new_papers_admitted
    )
    if advanced_checkpoint.coverage_status == SearchCoverageStatus.EXHAUSTED:
        advanced_checkpoint = replace(
            advanced_checkpoint,
            last_completed_interval_end=context.started_at_timestamp,
            stop_reason=f"provider returned no further page ({total_results} result(s) in total)",
        )
    return advanced_checkpoint


def _current_admission_limit(
    context: CollectionContext, checkpoint: SourceCheckpoint, target: SearchTarget, extra_limit: int | None
) -> int:
    limits = [context.candidate_budget.remaining_for_route(target.discovery_route)]
    if checkpoint.remaining_candidate_budget is not None:
        limits.append(checkpoint.remaining_candidate_budget)
    if extra_limit is not None:
        limits.append(extra_limit)
    return max(0, min(limits))


def _save_checkpoint_alone(context: CollectionContext, checkpoint: SourceCheckpoint) -> SourceCheckpoint:
    with write_transaction(context.connection):
        save_source_checkpoint(context.connection, checkpoint)
    return checkpoint
