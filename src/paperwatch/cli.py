"""Command-line entry point: paperwatch [--home DIR] {init, doctor, collect, work, report, status}.

Every failure is printed as `paperwatch: FAILED [<component>] <message>` with a component-specific exit code
(see paperwatch.errors), so the operator and systemd logs show which part broke.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from paperwatch import __version__
from paperwatch.app_context import AppContext
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.doctor_command import run_doctor_command
from paperwatch.commands.init_command import run_init_command
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.status_command import run_status_command
from paperwatch.commands.work_command import run_work_command
from paperwatch.errors import PaperwatchError, describe_failure_for_operator

HOME_ENVIRONMENT_VARIABLE = "PAPERWATCH_HOME"

COMMAND_RUNNERS: dict[str, Callable[[AppContext, argparse.Namespace], int]] = {
    "doctor": run_doctor_command,
    "collect": run_collect_command,
    "work": run_work_command,
    "report": run_report_command,
    "status": run_status_command,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperwatch", description="Local daily paper monitor.")
    parser.add_argument("--version", action="version", version=f"paperwatch {__version__}")
    parser.add_argument("--home", type=Path, default=Path(os.environ.get(HOME_ENVIRONMENT_VARIABLE, ".")),
                        help=f"directory holding config.toml, topics.toml, state/ and output/ "
                             f"(default: ${HOME_ENVIRONMENT_VARIABLE} or the current directory)")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress details to stderr")
    subcommands = parser.add_subparsers(dest="command_name", required=True, metavar="COMMAND")

    subcommands.add_parser("init", help="create config files, directories and the database (idempotent)")
    subcommands.add_parser("doctor", help="check config, database and the local model server")
    collect_parser = subcommands.add_parser("collect", help="import papers from sources and queue reviews")
    collect_parser.add_argument("--fixture", dest="fixture_paths", type=Path, action="append",
                                metavar="FILE", help="import recorded papers from a JSON file (repeatable)")
    work_parser = subcommands.add_parser("work", help="run queued jobs (loops until stopped)")
    work_parser.add_argument("--once", action="store_true", help="run at most one job, then exit")
    report_parser = subcommands.add_parser("report", help="publish the daily Markdown report")
    report_parser.add_argument("--date", dest="report_date", metavar="YYYY-MM-DD",
                               help="reporting date (default: today in the report timezone)")
    subcommands.add_parser("status", help="show catalog, queue, model and report state")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if arguments.command_name == "init":
            return run_init_command(arguments.home, arguments)
        app = AppContext.load(arguments.home)
        try:
            return COMMAND_RUNNERS[arguments.command_name](app, arguments)
        finally:
            app.close()
    except PaperwatchError as failure:
        print(describe_failure_for_operator(failure), file=sys.stderr)
        return failure.process_exit_code
    except KeyboardInterrupt:
        return 130
