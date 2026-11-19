"""Pause model work together when the llama.cpp server is down, and back off the health check.

State lives in the `service_state` table so every process (worker, report writer, doctor) sees the same pause.
While paused, no job is claimed and no inference request is sent; queued papers simply wait.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.config import ModelSettings
from paperwatch.errors import ModelServerUnavailable
from paperwatch.llm.types import ModelClient

MODEL_SERVICE_NAME = "model-server"


@dataclass(frozen=True)
class ServiceHealthState:
    service: str
    paused_until: datetime | None
    consecutive_failures: int
    last_error: str | None
    last_failure_at: str | None
    last_success_at: str | None

    def is_paused_at(self, moment: datetime) -> bool:
        return self.paused_until is not None and moment < self.paused_until


def read_service_health_state(connection: sqlite3.Connection, service: str) -> ServiceHealthState:
    row = connection.execute("SELECT * FROM service_state WHERE service = ?", (service,)).fetchone()
    if row is None:
        return ServiceHealthState(service, None, 0, None, None, None)
    return ServiceHealthState(
        service=service,
        paused_until=parse_utc_timestamp(row["paused_until"]) if row["paused_until"] else None,
        consecutive_failures=row["consecutive_failures"],
        last_error=row["last_error"],
        last_failure_at=row["last_failure_at"],
        last_success_at=row["last_success_at"])


def record_service_outage(connection: sqlite3.Connection, service: str, error_message: str,
                          model_settings: ModelSettings) -> datetime:
    """Count one more consecutive failure and pause the service with exponential back-off. Returns paused_until."""
    previous_state = read_service_health_state(connection, service)
    consecutive_failures = previous_state.consecutive_failures + 1
    backoff_seconds = min(model_settings.outage_backoff_max_seconds,
                          model_settings.outage_backoff_initial_seconds * 2 ** (consecutive_failures - 1))
    paused_until = utc_now() + timedelta(seconds=backoff_seconds)
    connection.execute(
        """INSERT INTO service_state (service, paused_until, consecutive_failures, last_error, last_failure_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(service) DO UPDATE SET paused_until = excluded.paused_until,
               consecutive_failures = excluded.consecutive_failures, last_error = excluded.last_error,
               last_failure_at = excluded.last_failure_at""",
        (service, format_utc_timestamp(paused_until), consecutive_failures, error_message, utc_now_timestamp()))
    return paused_until


def record_service_success(connection: sqlite3.Connection, service: str) -> None:
    connection.execute(
        """INSERT INTO service_state (service, paused_until, consecutive_failures, last_success_at)
           VALUES (?, NULL, 0, ?)
           ON CONFLICT(service) DO UPDATE SET paused_until = NULL, consecutive_failures = 0,
               last_success_at = excluded.last_success_at""",
        (service, utc_now_timestamp()))


def ensure_model_service_available(connection: sqlite3.Connection, model_client: ModelClient,
                                   model_settings: ModelSettings, *, probe_now_even_if_backing_off: bool) -> None:
    """Return if model work may proceed; otherwise raise ModelServerUnavailable describing the pause.

    Not paused            -> return immediately (the real request will detect a new outage).
    Paused, back-off open -> raise without touching the network, unless an operator asked to probe now.
    Paused, back-off over -> one cheap /health probe: success clears the pause, failure extends the back-off.
    """
    health_state = read_service_health_state(connection, MODEL_SERVICE_NAME)
    if health_state.paused_until is None:
        return
    if health_state.is_paused_at(utc_now()) and not probe_now_even_if_backing_off:
        raise ModelServerUnavailable(
            f"model jobs are paused until {format_utc_timestamp(health_state.paused_until)} after "
            f"{health_state.consecutive_failures} consecutive failure(s); last error: {health_state.last_error}")
    try:
        model_client.check_health()
    except ModelServerUnavailable as health_error:
        record_service_outage(connection, MODEL_SERVICE_NAME, str(health_error), model_settings)
        raise
    record_service_success(connection, MODEL_SERVICE_NAME)
