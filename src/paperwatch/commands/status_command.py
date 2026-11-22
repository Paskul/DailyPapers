"""`paperwatch status`: a read-only summary of the research profile (topics and seeds), catalog, queue (count and age
per priority class), screening and today's promotions, model health, provider access, and the last report."""

from __future__ import annotations

import argparse
import sqlite3
from datetime import timedelta

from paperwatch.app_context import AppContext
from paperwatch.clock import format_utc_timestamp, utc_now
from paperwatch.collection.candidate_budget import CandidateAdmissionBudget
from paperwatch.collection.current_releases import (
    ARXIV_FEED_CHECKPOINT_PREFIX,
    ARXIV_RECOVERY_CHECKPOINT_PREFIX,
    OPENALEX_RECENT_CHECKPOINT_PREFIX,
)
from paperwatch.collection.historical_search import (
    historical_queries_for_topic,
    history_plan_key,
    plan_windows,
    recency_schedule_settings,
)
from paperwatch.collection.source_cache_limit import measure_source_cache
from paperwatch.comparisons.comparison_queue import count_comparison_jobs_created_today
from paperwatch.config import TOPICS_FILE_NAME
from paperwatch.dispatch_order import PRIORITY_CLASS_LABELS
from paperwatch.jobs import summarize_queued_jobs_by_class
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, read_service_health_state
from paperwatch.network.outbound_ledger import summarize_outbound_requests_today
from paperwatch.network.provider_gateway import PROVIDER_SERVICE_NAME_PREFIX
from paperwatch.public_query_catalog import build_public_query_catalog
from paperwatch.reading.stage_results import count_saved_stages_of_unfinished_readings
from paperwatch.references.reference_selection import count_reference_admissions_today
from paperwatch.references.research_branches import count_branches_requested_today
from paperwatch.research_profile import (
    TopicResearchProfile,
    build_research_profiles,
    explain_empty_research_profile,
    load_active_seed_library_paper_ids,
)
from paperwatch.screening.embedding_model_registry import look_up_pinned_embedding_model
from paperwatch.screening.review_promotion import count_screening_promotions_today, utc_day_start_timestamp
from paperwatch.screening.text_embedders import EMBEDDING_BACKEND_HASHING
from paperwatch.seeds.seed_store import count_active_seeds_by_source, seed_sources_of_all_topics
from paperwatch.source_checkpoints import SourceCheckpoint, load_source_checkpoint, load_source_checkpoints_with_prefix


def run_status_command(app: AppContext, arguments: argparse.Namespace) -> int:
    connection = app.connection
    print(f"paperwatch home: {app.config.home_dir}")

    _print_research_profile(app)

    paper_counts = connection.execute(
        "SELECT status, COUNT(*) AS paper_count FROM papers GROUP BY status ORDER BY status"
    ).fetchall()
    total_papers = sum(row["paper_count"] for row in paper_counts)
    print(
        f"Papers: {total_papers} total"
        + (
            " (" + ", ".join(f"{row['status']} {row['paper_count']}" for row in paper_counts) + ")"
            if paper_counts
            else ""
        )
    )

    edition_count = connection.execute("SELECT COUNT(*) FROM paper_editions").fetchone()[0]
    alias_count = connection.execute("SELECT COUNT(*) FROM paper_aliases").fetchone()[0]
    print(f"Identity: {edition_count} edition(s), {alias_count} identifier alias(es)")
    active_seed_paper_count = len(load_active_seed_library_paper_ids(connection, app.config))
    if active_seed_paper_count:
        print(
            f"  of the papers, {active_seed_paper_count} are in your active seed library (comparison anchors; "
            "never queued for review or recommended)"
        )
    unresolved_match_rows = connection.execute(
        """SELECT first_paper_id, second_paper_id, match_reason FROM paper_possible_matches
           WHERE resolution = 'unresolved' ORDER BY detected_at, first_paper_id"""
    ).fetchall()
    if unresolved_match_rows:
        print(f"Possible duplicate papers (not merged): {len(unresolved_match_rows)} unresolved")
        for match_row in unresolved_match_rows[:5]:
            print(f"  {match_row['first_paper_id']} ~ {match_row['second_paper_id']} ({match_row['match_reason']})")

    job_rows = connection.execute(
        """SELECT kind, status, COUNT(*) AS job_count, MIN(due_at) AS oldest_due FROM jobs
           GROUP BY kind, status ORDER BY kind, status"""
    ).fetchall()
    if not job_rows:
        print("Jobs: none")
    for job_row in job_rows:
        oldest_label = f", oldest due {job_row['oldest_due']}" if job_row["status"] == "queued" else ""
        print(f"Jobs: {job_row['kind']} {job_row['status']} {job_row['job_count']}{oldest_label}")
    now_timestamp = format_utc_timestamp(utc_now())
    expired_lease_count = connection.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'running' AND lease_expires_at <= ?", (now_timestamp,)
    ).fetchone()[0]
    if expired_lease_count:
        print(
            f"  {expired_lease_count} running job(s) have an expired lease (worker stopped); the next worker "
            "reclaims them"
        )
    for failed_row in connection.execute(
        """SELECT job_id, paper_id, last_error_component, last_error FROM jobs
               WHERE status = 'failed' ORDER BY updated_at DESC LIMIT 5"""
    ):
        print(
            f"  failed {failed_row['job_id']} (paper {failed_row['paper_id']}): "
            f"[{failed_row['last_error_component']}] {failed_row['last_error']}"
        )

    _print_queue_by_priority_class(app)
    _print_screening_summary(app)

    coverage_rows = connection.execute(
        "SELECT coverage, COUNT(*) AS review_count FROM reviews GROUP BY coverage"
    ).fetchall()
    print("Reviews: " + (", ".join(f"{row['coverage']} {row['review_count']}" for row in coverage_rows) or "none"))
    _print_reading_progress(app)
    _print_references_and_comparisons(app)

    model_health = read_service_health_state(connection, MODEL_SERVICE_NAME)
    if (model_paused_until := model_health.pause_end_at(utc_now())) is not None:
        print(
            f"Model server: PAUSED until {format_utc_timestamp(model_paused_until)} after "
            f"{model_health.consecutive_failures} failure(s); last error: {model_health.last_error}"
        )
    else:
        print(f"Model server: {app.config.model.backend} at {app.config.model.base_url}, not paused")
    last_successful_call = connection.execute(
        """SELECT role_name, prompt_version, model_identity, server_build, started_at FROM model_calls
           WHERE outcome = 'ok' ORDER BY started_at DESC LIMIT 1"""
    ).fetchone()
    if last_successful_call:
        print(
            f"Last successful model call: {last_successful_call['started_at']} role "
            f"{last_successful_call['role_name']} ({last_successful_call['prompt_version']}) on "
            f"{last_successful_call['model_identity']}, build {last_successful_call['server_build']}"
        )
    else:
        print("Last successful model call: never")

    _print_collection_coverage(app)
    _print_provider_access_summary(app)

    last_report = connection.execute(
        """SELECT report.report_date, report.status, report.output_path, report.synthesis_status,
                  (SELECT COUNT(*) FROM report_items AS item WHERE item.report_date = report.report_date) AS papers
           FROM reports AS report ORDER BY report.report_date DESC LIMIT 1"""
    ).fetchone()
    print(
        f"Last report: {last_report['report_date']} {last_report['status']}, {last_report['papers']} paper(s), "
        f"synthesis {last_report['synthesis_status'] or 'not made yet'} -> {last_report['output_path']}"
        if last_report
        else "Last report: none yet"
    )
    return 0


def _print_queue_by_priority_class(app: AppContext) -> None:
    """Plan part 6: count and age per worker class, so sustained new arrivals delaying older work are visible."""
    now = utc_now()
    class_summaries = summarize_queued_jobs_by_class(app.connection, now)
    if not class_summaries:
        return
    print(
        "Queue by class (claimed in this order): "
        + "; ".join(
            f"{int(summary.priority_class)} {PRIORITY_CLASS_LABELS[summary.priority_class]} {summary.queued_count}, "
            f"oldest {_describe_age(now - summary.oldest_queued_at)}"
            for summary in class_summaries
        )
    )


def _describe_age(age: timedelta) -> str:
    total_hours = age.total_seconds() / 3600
    return f"{total_hours / 24:.1f} d" if total_hours >= 48 else f"{total_hours:.1f} h"


def _print_screening_summary(app: AppContext) -> None:
    """Plan part 6: the embedding model, how many candidates are scored, and today's promotions per reason and topic."""
    connection, screening_settings = app.connection, app.config.screening
    model_description = screening_settings.embedding_backend
    if screening_settings.embedding_backend == EMBEDDING_BACKEND_HASHING:
        model_description += " (offline TEST embedder, not for real screening)"
    else:
        model_present = all(
            (app.config.embedding_model_dir / pinned_file.file_name).is_file()
            for pinned_file in look_up_pinned_embedding_model(screening_settings.embedding_model).pinned_files
        )
        model_description += (
            f" {screening_settings.embedding_model} in {app.config.embedding_model_dir} "
            f"({'installed' if model_present else 'MISSING: run deploy/install_embedding_model.sh'})"
        )
    scored_paper_count = connection.execute("SELECT COUNT(DISTINCT paper_id) FROM candidate_topic_scores").fetchone()[0]
    waiting_count = connection.execute(
        """SELECT COUNT(*) FROM papers WHERE status = 'waiting_input'
             AND paper_id NOT IN (SELECT paper_id FROM candidate_topic_scores)"""
    ).fetchone()[0]
    print(
        f"Screening: {model_description}; {scored_paper_count} paper(s) scored; {waiting_count} paper(s) without "
        "a score (new since the last collect, seeds, or waiting for the embedding model)"
    )
    today_counts = count_screening_promotions_today(connection, utc_now())
    promotion_rows = connection.execute(
        """SELECT promotion_reason, COALESCE(topic_id, '-') AS topic_id, COUNT(*) AS promotion_count
           FROM paper_promotions WHERE promoted_at >= ? GROUP BY promotion_reason, topic_id
           ORDER BY promotion_reason, topic_id""",
        (utc_day_start_timestamp(utc_now()),),
    ).fetchall()
    print(
        f"  promoted today (UTC): {today_counts['regular_count'] + today_counts['near_cutoff_count']} of "
        f"{screening_settings.daily_promotion_limit} ({today_counts['near_cutoff_count']} of "
        f"{screening_settings.near_cutoff_promotions} near the cutoff); minimum score "
        f"{screening_settings.minimum_screening_score:g}"
        + (
            "; by reason and topic: "
            + ", ".join(
                f"{row['promotion_reason']} {row['topic_id']} {row['promotion_count']}" for row in promotion_rows
            )
            if promotion_rows
            else ""
        )
    )


def _print_research_profile(app: AppContext) -> None:
    """Topics with their input mode and anything missing, then each seed source's sync state (plan part 4)."""
    if not app.config.topics:
        print(
            f"Topics: none configured. Add [[topics]] entries to {app.config.home_dir / TOPICS_FILE_NAME}: a "
            "private `question` with `public_queries` (keywords), and/or seed papers via `zotero_collections` or "
            "`csl_json_files` (seeds), plus `arxiv_categories` to search."
        )
        return
    profiles = build_research_profiles(app.connection, app.config)
    print(f"Topics: {len(profiles)} configured, all active")
    for profile in profiles:
        print(f"  {profile.topic_id} [{profile.input_mode}]: {_describe_topic_inputs(profile)}")
        for explanation in profile.explain_missing_inputs():
            print(f"    ! {explanation}")
    empty_profile_explanation = explain_empty_research_profile(profiles)
    if empty_profile_explanation:
        print(f"  {empty_profile_explanation}")

    for seed_source_reference in seed_sources_of_all_topics(app.config.topics):
        active_count, title_only_count = count_active_seeds_by_source(app.connection, seed_source_reference)
        checkpoint = load_source_checkpoint(app.connection, seed_source_reference.checkpoint_key)
        if checkpoint is None:
            sync_state = "never synced (run `paperwatch collect`)"
        else:
            last_complete = checkpoint.last_completed_interval_end or "never"
            sync_state = f"last complete sync {last_complete}"
            if checkpoint.cursor and seed_source_reference.seed_source == "zotero":
                sync_state += f", library version {checkpoint.cursor}"
            if checkpoint.coverage_status == "blocked":
                sync_state += f"; LAST ATTEMPT FAILED: {checkpoint.stop_reason} (using the cached seed set)"
        print(
            f"Seed source {seed_source_reference.display_label}: {active_count} active seed(s), "
            f"{title_only_count} title-only; {sync_state}"
        )


def _describe_topic_inputs(profile: TopicResearchProfile) -> str:
    topic = profile.topic
    seed_part = f"{len(profile.active_seeds)} active seed(s)"
    if profile.title_only_seed_count:
        seed_part += f" ({profile.title_only_seed_count} title-only)"
    return (
        f"question {'set' if topic.has_written_question else 'none'}; {len(topic.public_queries)} public "
        f"quer(y/ies); {len(topic.arxiv_categories)} arXiv categor(y/ies); {seed_part} from "
        f"{len(profile.seed_sources)} seed source(s)"
    )


def _print_provider_access_summary(app: AppContext) -> None:
    """Outbound traffic today (only the collector makes any), request budgets used, and paused providers."""
    outbound_rows = summarize_outbound_requests_today(app.connection)
    if not outbound_rows:
        print("Outbound requests today: none")
    else:
        print(
            "Outbound requests today: "
            + ", ".join(f"{row['provider_name']} {row['outcome']} {row['request_count']}" for row in outbound_rows)
        )
    # Plan part 10: the last successful fetch per source, so a silent outage is visible even on a quiet day.
    for last_fetch_row in app.connection.execute(
        """SELECT provider_name, MAX(started_at) AS last_success,
                      (SELECT MAX(failure.started_at) FROM outbound_requests AS failure
                       WHERE failure.provider_name = request.provider_name
                         AND failure.outcome NOT IN ('ok', 'redirect', 'not_modified', 'not_found')) AS last_failure
               FROM outbound_requests AS request WHERE outcome IN ('ok', 'not_modified')
               GROUP BY provider_name ORDER BY provider_name"""
    ):
        failure_text = (
            f"; last failure {last_fetch_row['last_failure']}"
            if last_fetch_row["last_failure"] and last_fetch_row["last_failure"] > last_fetch_row["last_success"]
            else ""
        )
        print(
            f"  Last successful fetch from {last_fetch_row['provider_name']}: {last_fetch_row['last_success']}"
            f"{failure_text}"
        )
    blocked_count = sum(row["request_count"] for row in outbound_rows if row["outcome"] == "blocked")
    if blocked_count:
        print(f"  {blocked_count} request(s) refused by the access rules today (see the outbound_requests table)")
    for paused_row in app.connection.execute(
        "SELECT service FROM service_state WHERE service LIKE ? || '%' ORDER BY service",
        (PROVIDER_SERVICE_NAME_PREFIX,),
    ):
        provider_health = read_service_health_state(app.connection, paused_row["service"])
        if (provider_paused_until := provider_health.pause_end_at(utc_now())) is not None:
            print(
                f"Provider {paused_row['service'].removeprefix(PROVIDER_SERVICE_NAME_PREFIX)}: PAUSED until "
                f"{format_utc_timestamp(provider_paused_until)}; last error: {provider_health.last_error}"
            )


def _print_reading_progress(app: AppContext) -> None:
    """Plan part 7: saved steps of readings not finished yet (they resume, not restart) and the reading priorities."""
    saved_stage_count = count_saved_stages_of_unfinished_readings(app.connection)
    if saved_stage_count:
        print(f"  {saved_stage_count} saved reading step(s) of unfinished readings (a retry resumes from them)")
    priority_row = app.connection.execute(
        """SELECT COUNT(*) AS assessed, SUM(topic_score >= 65) AS at_least_65 FROM (
               SELECT review_id, MAX(topic_score) AS topic_score FROM review_topic_assessments
               WHERE score_scale = 'priority_0_to_100' GROUP BY review_id)"""
    ).fetchone()
    if priority_row["assessed"]:
        print(
            f"  Reading priority: {priority_row['assessed']} review(s) scored, {priority_row['at_least_65'] or 0} "
            "at 65/100 or more for their best topic"
        )


def _print_references_and_comparisons(app: AppContext) -> None:
    """Plan part 8: reference fetches, the research role's choices and today's admissions against the limit, gap
    searches, unresolved bibliography references, and comparisons with today's jobs against their limit."""
    connection, moment = app.connection, utc_now()
    fetch_counts = {
        row["fetch_status"]: row["paper_count"]
        for row in connection.execute(
            "SELECT fetch_status, COUNT(*) AS paper_count FROM reference_fetches GROUP BY fetch_status"
        )
    }
    edge_row = connection.execute(
        "SELECT COUNT(*) AS edges, SUM(cited_paper_id IS NULL) AS unresolved FROM citation_edges"
    ).fetchone()
    unresolved_bibliography = connection.execute(
        "SELECT COUNT(*) FROM bibliography_identifiers WHERE confirmation_status IN ('mismatch', 'not_found')"
    ).fetchone()[0]
    print(
        f"References: {fetch_counts.get('fetched', 0)} paper(s) fetched, "
        f"{fetch_counts.get('no_provider_record', 0)} without a provider record, {fetch_counts.get('blocked', 0)} "
        f"blocked; {edge_row['edges']} citation edge(s) ({edge_row['unresolved'] or 0} not in the catalog); "
        f"{unresolved_bibliography} bibliography identifier(s) unresolved"
    )
    selection_counts = {
        row["admission_status"]: row["selection_count"]
        for row in connection.execute(
            "SELECT admission_status, COUNT(*) AS selection_count FROM reference_selections GROUP BY admission_status"
        )
    }
    print(
        f"  Chosen references: {selection_counts.get('admitted', 0)} admitted, "
        f"{selection_counts.get('waiting_daily_cap', 0)} waiting for the daily limit, "
        f"{selection_counts.get('skipped', 0)} skipped; {count_reference_admissions_today(connection, moment)} of "
        f"{app.config.references.daily_reference_readings} admitted today (UTC)"
    )
    branch_counts = {
        row["branch_status"]: row["branch_count"]
        for row in connection.execute(
            "SELECT branch_status, COUNT(*) AS branch_count FROM research_branches GROUP BY branch_status"
        )
    }
    if branch_counts:
        print(
            f"  Gap searches: {branch_counts.get('requested', 0)} requested, {branch_counts.get('done', 0)} done, "
            f"{branch_counts.get('blocked', 0)} blocked; {count_branches_requested_today(connection, moment)} of "
            f"{app.config.references.research_branches_per_day} requested today"
        )
    comparison_count = connection.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
    print(
        f"Comparisons: {comparison_count} saved; {count_comparison_jobs_created_today(connection, moment)} of "
        f"{app.config.comparisons.daily_comparison_jobs} comparison job(s) queued today (UTC)"
    )


def _print_collection_coverage(app: AppContext) -> None:
    """Plan part 5: what each search covered, where it stopped and why, open gaps, budgets, and source text."""
    connection, collection_settings = app.connection, app.config.collection
    candidate_budget = CandidateAdmissionBudget.load_for_day(connection, collection_settings, utc_now())
    print(f"Collection: {candidate_budget.describe()}")

    for feed_checkpoint in load_source_checkpoints_with_prefix(connection, ARXIV_FEED_CHECKPOINT_PREFIX):
        print(
            f"  arXiv feed {feed_checkpoint.search_key.removeprefix(ARXIV_FEED_CHECKPOINT_PREFIX)}: last complete "
            f"poll {feed_checkpoint.last_completed_interval_end or 'never'}; {_describe_coverage(feed_checkpoint)}"
        )
    open_recoveries = [
        checkpoint
        for checkpoint in load_source_checkpoints_with_prefix(connection, ARXIV_RECOVERY_CHECKPOINT_PREFIX)
        if checkpoint.coverage_status != "exhausted"
    ]
    if open_recoveries:
        print(f"  Open coverage gaps: {len(open_recoveries)} missed-release recovery search(es)")
        for recovery_checkpoint in open_recoveries:
            print(
                f"    {recovery_checkpoint.search_key}: {recovery_checkpoint.candidates_admitted} recovered so far; "
                f"{_describe_coverage(recovery_checkpoint)}"
            )
    for recent_checkpoint in load_source_checkpoints_with_prefix(connection, OPENALEX_RECENT_CHECKPOINT_PREFIX):
        window = (
            f"{recent_checkpoint.query_parameters.get('window_first_date')}.."
            f"{recent_checkpoint.query_parameters.get('window_last_date')}"
        )
        print(
            f'  OpenAlex recent "{recent_checkpoint.query_parameters.get("public_value")}" '
            f"[{recent_checkpoint.topic_id}]: window {window}, last complete sweep "
            f"{recent_checkpoint.last_completed_interval_end or 'never'}; {_describe_coverage(recent_checkpoint)}"
        )

    query_catalog = build_public_query_catalog(app.config.topics)
    for topic in app.config.topics:
        historical_queries = historical_queries_for_topic(topic.topic_id, query_catalog)
        if not historical_queries:
            continue
        plan_checkpoint = load_source_checkpoint(
            connection,
            history_plan_key(topic.topic_id, historical_queries, recency_schedule_settings(collection_settings)),
        )
        if plan_checkpoint is None:
            print(f"  Historical search {topic.topic_id}: not started (runs on the next `collect`)")
            continue
        band_rows = connection.execute(
            """SELECT coverage_status, COUNT(*) AS band_count FROM source_checkpoints
               WHERE topic_id = ? AND search_key LIKE 'history:%' GROUP BY coverage_status ORDER BY coverage_status""",
            (topic.topic_id,),
        ).fetchall()
        band_summary = ", ".join(f"{row['coverage_status']} {row['band_count']}" for row in band_rows) or "none yet"
        print(
            f"  Historical search {topic.topic_id}: {plan_checkpoint.candidates_admitted} of "
            f"{plan_checkpoint.candidate_budget} candidate(s); {_describe_coverage(plan_checkpoint)}; window "
            f"searches: {band_summary}"
        )
        _print_admitted_by_window(connection, topic.topic_id, plan_checkpoint)

    text_rows = connection.execute(
        """SELECT acquisition_status, extraction_status, COUNT(*) AS edition_count FROM edition_source_texts
           GROUP BY acquisition_status, extraction_status ORDER BY acquisition_status, extraction_status"""
    ).fetchall()
    day_start = format_utc_timestamp(utc_now().replace(hour=0, minute=0, second=0, microsecond=0))
    downloads_today = connection.execute(
        "SELECT COUNT(*) FROM edition_source_texts WHERE downloaded_from_network = 1 AND acquired_at >= ?", (day_start,)
    ).fetchone()[0]
    print(
        "Source text: "
        + (
            ", ".join(
                f"{row['acquisition_status']}"
                + (f"/{row['extraction_status']}" if row["acquisition_status"] == "acquired" else "")
                + f" {row['edition_count']}"
                for row in text_rows
            )
            or "none yet"
        )
        + f"; {downloads_today} of {collection_settings.daily_source_text_budget} download(s) used today; local "
        f"files from {app.config.local_sources_dir}"
    )
    print(measure_source_cache(app.config).describe())
    withdrawn_count = connection.execute("SELECT COUNT(*) FROM paper_withdrawals").fetchone()[0]
    if withdrawn_count:
        print(
            f"Withdrawn or retracted papers: {withdrawn_count} (never recommended; a previously reported one gets a "
            "correction in the next report)"
        )


def _describe_coverage(checkpoint: SourceCheckpoint) -> str:
    return f"{checkpoint.coverage_status}" + (f" ({checkpoint.stop_reason})" if checkpoint.stop_reason else "")


def _print_admitted_by_window(connection: sqlite3.Connection, topic_id: str, plan_checkpoint: SourceCheckpoint) -> None:
    """How the topic's historical candidates spread over its recency-weighted windows (newest first)."""
    admitted_by_window = []
    for window in plan_windows(plan_checkpoint):
        admitted_count = connection.execute(
            "SELECT COALESCE(SUM(candidates_admitted), 0) FROM source_checkpoints WHERE topic_id = ? AND "
            "search_key LIKE 'history:%' AND search_key LIKE '%:' || ?",
            (topic_id, window.label),
        ).fetchone()[0]
        admitted_by_window.append(f"{window.first_date.year}-{window.last_date.year} {admitted_count}")
    print("    admitted per window (newest first): " + ", ".join(admitted_by_window))
