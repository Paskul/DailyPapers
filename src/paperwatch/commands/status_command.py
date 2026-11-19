"""`paperwatch status`: a read-only summary of topics, catalog, queue, model health, and the last report."""

from __future__ import annotations

import argparse

from paperwatch.app_context import AppContext
from paperwatch.clock import format_utc_timestamp, utc_now
from paperwatch.config import TOPICS_FILE_NAME
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, read_service_health_state
from paperwatch.network.outbound_ledger import summarize_outbound_requests_today
from paperwatch.network.provider_gateway import PROVIDER_SERVICE_NAME_PREFIX


def run_status_command(app: AppContext, arguments: argparse.Namespace) -> int:
    connection = app.connection
    print(f"paperwatch home: {app.config.home_dir}")

    if app.config.topics:
        print(f"Topics: {len(app.config.topics)} configured "
              f"({', '.join(topic.topic_id for topic in app.config.topics)})")
    else:
        print(f"Topics: none configured. Add [[topics]] entries (a private research question plus public search "
              f"terms) to {app.config.home_dir / TOPICS_FILE_NAME}. Zotero seeds arrive in plan part 4.")

    paper_counts = connection.execute(
        "SELECT status, COUNT(*) AS paper_count FROM papers GROUP BY status ORDER BY status").fetchall()
    total_papers = sum(row["paper_count"] for row in paper_counts)
    print(f"Papers: {total_papers} total" + (" (" + ", ".join(
        f"{row['status']} {row['paper_count']}" for row in paper_counts) + ")" if paper_counts else ""))

    edition_count = connection.execute("SELECT COUNT(*) FROM paper_editions").fetchone()[0]
    alias_count = connection.execute("SELECT COUNT(*) FROM paper_aliases").fetchone()[0]
    print(f"Identity: {edition_count} edition(s), {alias_count} identifier alias(es)")
    unresolved_match_rows = connection.execute(
        """SELECT first_paper_id, second_paper_id, match_reason FROM paper_possible_matches
           WHERE resolution = 'unresolved' ORDER BY detected_at, first_paper_id""").fetchall()
    if unresolved_match_rows:
        print(f"Possible duplicate papers (not merged): {len(unresolved_match_rows)} unresolved")
        for match_row in unresolved_match_rows[:5]:
            print(f"  {match_row['first_paper_id']} ~ {match_row['second_paper_id']} ({match_row['match_reason']})")

    job_rows = connection.execute(
        """SELECT kind, status, COUNT(*) AS job_count, MIN(due_at) AS oldest_due FROM jobs
           GROUP BY kind, status ORDER BY kind, status""").fetchall()
    if not job_rows:
        print("Jobs: none")
    for job_row in job_rows:
        oldest_label = f", oldest due {job_row['oldest_due']}" if job_row["status"] == "queued" else ""
        print(f"Jobs: {job_row['kind']} {job_row['status']} {job_row['job_count']}{oldest_label}")
    now_timestamp = format_utc_timestamp(utc_now())
    expired_lease_count = connection.execute(
        "SELECT COUNT(*) FROM jobs WHERE status = 'running' AND lease_expires_at <= ?", (now_timestamp,)).fetchone()[0]
    if expired_lease_count:
        print(f"  {expired_lease_count} running job(s) have an expired lease (worker stopped); the next worker "
              "reclaims them")
    for failed_row in connection.execute(
            """SELECT job_id, paper_id, last_error_component, last_error FROM jobs
               WHERE status = 'failed' ORDER BY updated_at DESC LIMIT 5"""):
        print(f"  failed {failed_row['job_id']} (paper {failed_row['paper_id']}): "
              f"[{failed_row['last_error_component']}] {failed_row['last_error']}")

    coverage_rows = connection.execute(
        "SELECT coverage, COUNT(*) AS review_count FROM reviews GROUP BY coverage").fetchall()
    print("Reviews: " + (", ".join(f"{row['coverage']} {row['review_count']}" for row in coverage_rows) or "none"))

    model_health = read_service_health_state(connection, MODEL_SERVICE_NAME)
    if model_health.is_paused_at(utc_now()):
        print(f"Model server: PAUSED until {format_utc_timestamp(model_health.paused_until)} after "
              f"{model_health.consecutive_failures} failure(s); last error: {model_health.last_error}")
    else:
        print(f"Model server: {app.config.model.backend} at {app.config.model.base_url}, not paused")
    last_successful_call = connection.execute(
        """SELECT role_name, prompt_version, model_identity, server_build, started_at FROM model_calls
           WHERE outcome = 'ok' ORDER BY started_at DESC LIMIT 1""").fetchone()
    if last_successful_call:
        print(f"Last successful model call: {last_successful_call['started_at']} role "
              f"{last_successful_call['role_name']} ({last_successful_call['prompt_version']}) on "
              f"{last_successful_call['model_identity']}, build {last_successful_call['server_build']}")
    else:
        print("Last successful model call: never")

    _print_provider_access_summary(app)

    last_report = connection.execute(
        "SELECT report_date, status, output_path FROM reports ORDER BY report_date DESC LIMIT 1").fetchone()
    print(f"Last report: {last_report['report_date']} {last_report['status']} -> {last_report['output_path']}"
          if last_report else "Last report: none yet")
    return 0


def _print_provider_access_summary(app: AppContext) -> None:
    """Outbound traffic today (only the collector makes any), request budgets used, and paused providers."""
    outbound_rows = summarize_outbound_requests_today(app.connection)
    if not outbound_rows:
        print("Outbound requests today: none")
    else:
        print("Outbound requests today: " + ", ".join(
            f"{row['provider_name']} {row['outcome']} {row['request_count']}" for row in outbound_rows))
    blocked_count = sum(row["request_count"] for row in outbound_rows if row["outcome"] == "blocked")
    if blocked_count:
        print(f"  {blocked_count} request(s) refused by the access rules today (see the outbound_requests table)")
    for paused_row in app.connection.execute(
            "SELECT service FROM service_state WHERE service LIKE ? || '%' ORDER BY service",
            (PROVIDER_SERVICE_NAME_PREFIX,)):
        provider_health = read_service_health_state(app.connection, paused_row["service"])
        if provider_health.is_paused_at(utc_now()):
            print(f"Provider {paused_row['service'].removeprefix(PROVIDER_SERVICE_NAME_PREFIX)}: PAUSED until "
                  f"{format_utc_timestamp(provider_health.paused_until)}; last error: {provider_health.last_error}")
