"""Command-line entry point: paperwatch [--home DIR] {init, doctor, collect, work, reread, report, status,
access-check}.

Every failure is printed as `paperwatch: FAILED [<component>] <message>` with a component-specific exit code
(see paperwatch.errors), so the operator and systemd logs show which part broke.

Before a command runs, its process arms one egress policy (paperwatch.network.egress_guard): only `collect` may
reach paper providers; every other command may reach only the local model server; `init` may reach nothing.
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
from paperwatch.commands.access_check_command import COMPONENT_COLLECTOR, COMPONENT_WORKER, run_access_check_command
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.doctor_command import run_doctor_command
from paperwatch.commands.init_command import run_init_command
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.reread_command import run_reread_command
from paperwatch.commands.status_command import run_status_command
from paperwatch.commands.work_command import run_work_command
from paperwatch.config import AppConfig
from paperwatch.errors import PaperwatchError, describe_failure_for_operator
from paperwatch.network.egress_guard import (arm_egress_policy, build_collector_policy, build_local_model_policy,
                                             build_no_network_policy)
from paperwatch.network.provider_rules import load_provider_access_rules

HOME_ENVIRONMENT_VARIABLE = "PAPERWATCH_HOME"

COMMAND_RUNNERS: dict[str, Callable[[AppContext, argparse.Namespace], int]] = {
    "doctor": run_doctor_command,
    "collect": run_collect_command,
    "work": run_work_command,
    "reread": run_reread_command,
    "report": run_report_command,
    "status": run_status_command,
    "access-check": run_access_check_command,
}
COLLECTOR_COMMAND_NAME = "collect"
UNGUARDED_COMMAND_NAMES = frozenset({"access-check"})  # must try forbidden connections to prove the OS blocks them


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
    collect_parser = subcommands.add_parser("collect", help="collect current and older papers, acquire source text, "
                                                            "and queue reviews")
    collect_parser.add_argument("--fixture", dest="fixture_paths", type=Path, action="append",
                                metavar="FILE", help="import recorded papers from a JSON file instead of the live "
                                                     "arXiv / OpenAlex searches (repeatable)")
    collect_parser.add_argument("--skip-searches", action="store_true",
                                help="sync seeds and use local source files only: no arXiv / OpenAlex searches or "
                                     "full-text downloads")
    work_parser = subcommands.add_parser("work", help="run queued jobs (loops until stopped)")
    work_parser.add_argument("--once", action="store_true", help="run at most one job, then exit")
    reread_parser = subcommands.add_parser("reread", help="queue a fresh reading of one paper (operator request)")
    reread_parser.add_argument("paper_id", metavar="PAPER_ID", help="internal paper id, e.g. p_3f9a1c0b7d2e4a51")
    reread_parser.add_argument("--reason", help="why the paper should be read again (stored with the job)")
    report_parser = subcommands.add_parser("report", help="publish the daily Markdown report")
    report_parser.add_argument("--date", dest="report_date", metavar="YYYY-MM-DD",
                               help="reporting date (default: today in the report timezone)")
    subcommands.add_parser("status", help="show catalog, queue, model and report state")
    access_check_parser = subcommands.add_parser(
        "access-check", help="verify accounts, permissions, egress rules and the extractor sandbox (plan part 3)")
    access_check_parser.add_argument("--component", required=True, choices=(COMPONENT_COLLECTOR, COMPONENT_WORKER),
                                     help="which service account this is run as")
    access_check_parser.add_argument("--skip-network-probes", action="store_true",
                                     help="do not attempt the test connections")
    access_check_parser.add_argument("--public-probe-address", dest="public_probe_addresses", action="append",
                                     metavar="IP", help="public IP of the probe host, resolved by the caller; the "
                                     "worker must not be able to connect to it directly (repeatable)")
    return parser


def arm_egress_policy_for_command(command_name: str, app_config: AppConfig) -> None:
    """Defence in depth behind nftables: limit what this process may resolve, connect to, and launch."""
    if command_name in UNGUARDED_COMMAND_NAMES:
        return
    if command_name == COLLECTOR_COMMAND_NAME:
        arm_egress_policy(build_collector_policy(load_provider_access_rules().all_permitted_hosts(),
                                                 app_config.extractor.container_runtime))
    else:
        arm_egress_policy(build_local_model_policy(app_config.model.base_url, component_label=command_name))


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if arguments.command_name == "init":
            arm_egress_policy(build_no_network_policy("init"))
            return run_init_command(arguments.home, arguments)
        app = AppContext.load(arguments.home)
        try:
            arm_egress_policy_for_command(arguments.command_name, app.config)
            return COMMAND_RUNNERS[arguments.command_name](app, arguments)
        finally:
            app.close()
    except PaperwatchError as failure:
        print(describe_failure_for_operator(failure), file=sys.stderr)
        return failure.process_exit_code
    except KeyboardInterrupt:
        return 130
