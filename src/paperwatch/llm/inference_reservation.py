"""Reserve the next inference slot for a holder that must not wait behind the queue (plan part 9: the report).

The inference lock (inference_lock.py) gives one request at a time, but it is not fair: a worker that releases the
lock after one call could take it again for its next call before the report writer's retry notices. So the report
records a reservation row at its cutoff, and every other model call checks for an unexpired reservation BEFORE it
takes the lock, and waits while one exists. A running call is never interrupted; the worker yields at its next call
boundary. A reservation expires by itself (`reserved_until`), so a crashed report process cannot stall the worker.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from paperwatch.clock import format_utc_timestamp, parse_utc_timestamp, utc_now

RESERVATION_POLL_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class InferenceReservation:
    reservation_name: str
    purpose: str
    reserved_until: datetime


def reserve_inference_slot(
    connection: sqlite3.Connection, reservation_name: str, *, purpose: str, reserved_until: datetime
) -> None:
    """Create or extend a reservation (autocommit; call outside any open transaction)."""
    connection.execute(
        """INSERT INTO inference_reservations (reservation_name, purpose, reserved_at, reserved_until)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(reservation_name) DO UPDATE SET purpose = excluded.purpose,
               reserved_at = excluded.reserved_at, reserved_until = excluded.reserved_until""",
        (reservation_name, purpose, format_utc_timestamp(utc_now()), format_utc_timestamp(reserved_until)),
    )


def release_inference_reservation(connection: sqlite3.Connection, reservation_name: str) -> None:
    connection.execute("DELETE FROM inference_reservations WHERE reservation_name = ?", (reservation_name,))


def find_active_inference_reservation(connection: sqlite3.Connection, moment: datetime) -> InferenceReservation | None:
    """The unexpired reservation that ends last, or None."""
    reservation_row = connection.execute(
        """SELECT * FROM inference_reservations WHERE reserved_until > ?
           ORDER BY reserved_until DESC LIMIT 1""",
        (format_utc_timestamp(moment),),
    ).fetchone()
    if reservation_row is None:
        return None
    return InferenceReservation(
        reservation_row["reservation_name"],
        reservation_row["purpose"],
        parse_utc_timestamp(reservation_row["reserved_until"]),
    )


def wait_while_inference_is_reserved(
    connection: sqlite3.Connection, *, sleep_function: Callable[[float], None] = time.sleep
) -> float:
    """Block until no unexpired reservation exists. Returns the seconds waited (0.0 when nothing was reserved).

    Bounded by the reservations themselves: each one expires at its `reserved_until`."""
    started_monotonic = time.monotonic()
    while (reservation := find_active_inference_reservation(connection, utc_now())) is not None:
        seconds_left = (reservation.reserved_until - utc_now()).total_seconds()
        sleep_function(max(0.05, min(RESERVATION_POLL_INTERVAL_SECONDS, seconds_left)))
    return time.monotonic() - started_monotonic
