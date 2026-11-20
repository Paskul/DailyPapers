"""`paperwatch monitor`: a live web page of what paperwatch is doing, has done, and plans to do.

    paperwatch --home /srv/paperwatch monitor                 # http://127.0.0.1:8765
    paperwatch --home /srv/paperwatch monitor --json          # one snapshot on stdout, then exit

Run it as the operator (a member of the shared state group): it only reads the database (opened read-only), the
installed timer files, and `systemctl show`. It works with an older database too (the deployed pilot), showing the
current step inferred instead of recorded. To view it from another machine, tunnel the port over SSH
(`ssh -L 8765:127.0.0.1:8765 <host>`, then open http://localhost:8765) or serve it on this machine's Tailscale
address (`--listen-address $(tailscale ip -4)`); it refuses every other address.
"""

from __future__ import annotations

import argparse
import json
import sys

from paperwatch.config import AppConfig
from paperwatch.monitor.monitor_server import (DEFAULT_LISTEN_ADDRESS, DEFAULT_PORT, SnapshotProvider,
                                               create_monitor_server)
from paperwatch.monitor.snapshot import DEFAULT_HISTORY_HOURS


def add_monitor_arguments(monitor_parser: argparse.ArgumentParser) -> None:
    monitor_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"default {DEFAULT_PORT}")
    monitor_parser.add_argument("--listen-address", default=DEFAULT_LISTEN_ADDRESS,
                                help="127.0.0.1 (default), ::1, or this machine's Tailscale address")
    monitor_parser.add_argument("--allow-host", dest="extra_host_names", action="append", default=[],
                                metavar="NAME", help="another host name the page may be opened under, e.g. the "
                                                     "machine's Tailscale MagicDNS name (repeatable)")
    monitor_parser.add_argument("--history-hours", type=float, default=DEFAULT_HISTORY_HOURS,
                                help=f"default history window (default {DEFAULT_HISTORY_HOURS:g}; the page can "
                                     "change it)")
    monitor_parser.add_argument("--json", dest="print_json", action="store_true",
                                help="print one snapshot as JSON and exit (no server)")


def run_monitor_command(app_config: AppConfig, arguments: argparse.Namespace) -> int:
    snapshot_provider = SnapshotProvider(app_config, default_history_hours=arguments.history_hours)
    if arguments.print_json:
        json.dump(snapshot_provider.build(arguments.history_hours), sys.stdout, indent=1, ensure_ascii=False,
                  default=str)
        print()
        return 0
    server = create_monitor_server(app_config, listen_address=arguments.listen_address, port=arguments.port,
                                   extra_host_names=tuple(arguments.extra_host_names),
                                   snapshot_provider=snapshot_provider)
    host_text = f"[{server.server_name}]" if ":" in server.server_name else server.server_name
    print(f"paperwatch monitor: http://{host_text}:{server.server_port}/  (read-only; Ctrl+C stops it)",
          flush=True)
    if server.server_name.startswith("127.") or server.server_name == "::1":
        print(f"  from another machine: ssh -L {server.server_port}:127.0.0.1:{server.server_port} <this host>, "
              f"then open http://localhost:{server.server_port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
