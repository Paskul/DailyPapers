"""Command-line options that are defined next to their commands rather than in paperwatch.cli.

The CLI builds its whole argument parser on every start, including for `paperwatch --help`, so this module imports
nothing heavy: the command modules themselves (and numpy, jsonschema, httpx behind them) are imported only when their
command runs (paperwatch.cli.load_command_runner).
"""

from __future__ import annotations

import argparse

from paperwatch.errors import ModelOutputInvalid
from paperwatch.monitor.monitor_defaults import DEFAULT_HISTORY_HOURS, DEFAULT_LISTEN_ADDRESS, DEFAULT_PORT

# `access-check --component`: the service account the check runs as.
COMPONENT_COLLECTOR = "collector"
COMPONENT_WORKER = "worker"

# `retry-failed` requeues failures of the model's answers unless told otherwise.
DEFAULT_RETRY_COMPONENT = ModelOutputInvalid.failed_component  # "model-output"


def add_retry_failed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--component",
        dest="components",
        action="append",
        metavar="COMPONENT",
        help=f"requeue failed jobs whose last failure came from this component (repeatable; "
        f"default: {DEFAULT_RETRY_COMPONENT})",
    )
    parser.add_argument(
        "--job",
        dest="job_ids",
        action="append",
        metavar="JOB_ID",
        help="requeue this failed job, whatever its component (repeatable)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list the jobs, change nothing")


def add_monitor_arguments(monitor_parser: argparse.ArgumentParser) -> None:
    monitor_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    monitor_parser.add_argument(
        "--listen-address",
        default=DEFAULT_LISTEN_ADDRESS,
        help="127.0.0.1 (default), ::1, or this machine's Tailscale address",
    )
    monitor_parser.add_argument(
        "--allow-host",
        dest="extra_host_names",
        action="append",
        default=[],
        metavar="NAME",
        help="another host name the page may be opened under, e.g. the machine's Tailscale MagicDNS name (repeatable)",
    )
    monitor_parser.add_argument(
        "--history-hours",
        type=float,
        default=DEFAULT_HISTORY_HOURS,
        help=f"default history window (default {DEFAULT_HISTORY_HOURS:g}; the page can change it)",
    )
    monitor_parser.add_argument(
        "--json", dest="print_json", action="store_true", help="print one snapshot as JSON and exit (no server)"
    )
