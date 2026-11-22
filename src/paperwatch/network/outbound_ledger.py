"""What the collector sends to the internet, and how much of each provider's request budget it has used.

* `reserve_provider_request` claims one request from a provider's daily budget and its minimum-spacing slot in
  one short write transaction (shared by every process), and returns how long to wait before sending. The wait
  happens OUTSIDE the transaction.
* `record_outbound_request` writes one row per attempted hop: provider, operation, host, path, query parameters
  with credentials redacted, outcome, status, and size. This is how an operator (and the tests) can see
  exactly which public terms and ids left the machine.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass

from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.errors import SourceRequestBudgetExhausted
from paperwatch.identifiers import new_internal_id
from paperwatch.network.collector_credentials import REDACTED_VALUE

OUTBOUND_REQUEST_ID_PREFIX = "o"
MAXIMUM_LOGGED_ERROR_CHARACTERS = 500


@dataclass(frozen=True)
class OutboundRequestRecord:
    provider_name: str
    operation: str
    hop_number: int
    host: str
    path: str
    query_parameters: tuple[tuple[str, str], ...]  # as sent; redacted on write
    outcome: str  # ok | redirect | blocked | not_found | unavailable | rejected | budget_exhausted | error
    started_at: str
    duration_seconds: float
    status_code: int | None = None
    response_bytes: int | None = None
    error_message: str | None = None


def reserve_provider_request(
    connection: sqlite3.Connection,
    *,
    provider_name: str,
    daily_request_budget: int,
    minimum_seconds_between_requests: float,
) -> float:
    """Claim one request slot. Returns the seconds to wait before sending; raises when the daily budget is used up."""
    budget_date = utc_now().date().isoformat()
    now_epoch = time.time()
    with write_transaction(connection):
        budget_row = connection.execute(
            """SELECT requests_reserved, next_request_allowed_epoch FROM provider_request_budgets
               WHERE provider_name = ? AND budget_date = ?""",
            (provider_name, budget_date),
        ).fetchone()
        # Spacing carries over midnight: read the newest row of any date for the next allowed send time.
        latest_spacing_row = connection.execute(
            "SELECT MAX(next_request_allowed_epoch) AS next_allowed FROM provider_request_budgets "
            "WHERE provider_name = ?",
            (provider_name,),
        ).fetchone()
        requests_reserved = budget_row["requests_reserved"] if budget_row else 0
        if requests_reserved >= daily_request_budget:
            raise SourceRequestBudgetExhausted(
                f"{provider_name}: daily request budget of {daily_request_budget} is used up for {budget_date} (UTC)",
                operator_hint="collection from this provider resumes on the next UTC day",
            )
        send_at_epoch = max(now_epoch, latest_spacing_row["next_allowed"] or 0.0)
        connection.execute(
            """INSERT INTO provider_request_budgets (provider_name, budget_date, requests_reserved,
                   next_request_allowed_epoch) VALUES (?, ?, 1, ?)
               ON CONFLICT(provider_name, budget_date) DO UPDATE SET
                   requests_reserved = requests_reserved + 1,
                   next_request_allowed_epoch = excluded.next_request_allowed_epoch""",
            (provider_name, budget_date, send_at_epoch + minimum_seconds_between_requests),
        )
    return max(0.0, send_at_epoch - now_epoch)


def record_outbound_request(
    connection: sqlite3.Connection,
    record: OutboundRequestRecord,
    secret_parameter_names: frozenset[str],
    secret_values: frozenset[str],
) -> str:
    """Append one hop to the outbound log. Credentials are replaced by '[redacted]' in parameters and messages."""
    redacted_query = [
        [name, REDACTED_VALUE if name in secret_parameter_names else value] for name, value in record.query_parameters
    ]
    error_message = record.error_message
    if error_message:
        for secret_value in secret_values:
            error_message = error_message.replace(secret_value, REDACTED_VALUE)
        error_message = error_message[:MAXIMUM_LOGGED_ERROR_CHARACTERS]
    request_id = new_internal_id(OUTBOUND_REQUEST_ID_PREFIX)
    connection.execute(
        """INSERT INTO outbound_requests (request_id, provider_name, operation, hop_number, method, host, path,
               query_json, outcome, status_code, response_bytes, error_message, started_at, duration_seconds)
           VALUES (?, ?, ?, ?, 'GET', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            request_id,
            record.provider_name,
            record.operation,
            record.hop_number,
            record.host,
            record.path,
            json.dumps(redacted_query, ensure_ascii=False),
            record.outcome,
            record.status_code,
            record.response_bytes,
            error_message,
            record.started_at,
            round(record.duration_seconds, 3),
        ),
    )
    return request_id


def summarize_outbound_requests_today(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Per provider and outcome: how many hops were attempted today (UTC). Used by `status`."""
    today_prefix = utc_now_timestamp()[:10]
    return connection.execute(
        """SELECT provider_name, outcome, COUNT(*) AS request_count FROM outbound_requests
           WHERE started_at >= ? GROUP BY provider_name, outcome ORDER BY provider_name, outcome""",
        (today_prefix,),
    ).fetchall()
