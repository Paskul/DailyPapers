"""`paperwatch capacity [--hours 24]`: measured model time per role and per review, and the daily capacity estimate.
Read-only; makes no model call and no network request."""

from __future__ import annotations

import argparse

from paperwatch.app_context import AppContext
from paperwatch.clock import utc_now
from paperwatch.measurement.capacity import describe_capacity, measure_capacity


def run_capacity_command(app: AppContext, arguments: argparse.Namespace) -> int:
    measurement = measure_capacity(
        app.connection, app.config, window_end=utc_now(), window_hours=float(arguments.window_hours)
    )
    print("\n".join(describe_capacity(measurement)))
    return 0
