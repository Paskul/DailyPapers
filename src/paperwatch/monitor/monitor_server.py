"""A small read-only web server for the monitor page (stdlib http.server, no dependencies, no external assets).

Routes:  /                 the page (static/index.html)
         /monitor.js, /monitor.css
         /api/snapshot     the JSON snapshot (?hours=N for the history window, 6..168)
         /healthz          "ok"
Safeguards:
  * It listens only on a loopback address (default 127.0.0.1) or on this machine's Tailscale address, never on a
    LAN or public interface. From another machine, use an SSH tunnel (ssh -L 8765:127.0.0.1:8765 host) or Tailscale.
  * The Host header must name this machine (localhost, its loopback/Tailscale address, its host name, or a name the
    operator allowed), which defeats DNS-rebinding pages that would otherwise read the API from a browser.
  * GET and HEAD only; nothing on the page or the API can change anything. The database is opened read-only per
    request. Paper titles are untrusted text and are only ever inserted into the page as text.
  * A strict Content-Security-Policy: scripts, styles, and requests only from this server, no inline code, no frames.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable
from datetime import datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any

from paperwatch.clock import utc_now
from paperwatch.config import AppConfig
from paperwatch.errors import ConfigurationError, PaperwatchError
from paperwatch.monitor.database_access import open_read_only_database
from paperwatch.monitor.schedule_state import ScheduleState, read_schedule_state
from paperwatch.monitor.snapshot import build_monitor_snapshot

LOGGER = logging.getLogger(__name__)

DEFAULT_PORT = 8765
DEFAULT_LISTEN_ADDRESS = "127.0.0.1"
SNAPSHOT_CACHE_SECONDS = 3.0
MINIMUM_HISTORY_HOURS, MAXIMUM_HISTORY_HOURS = 6.0, 168.0
TAILSCALE_NETWORKS = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
STATIC_FILES = {  # route -> (file in paperwatch/monitor/static, content type)
    "/": ("index.html", "text/html; charset=utf-8"),
    "/monitor.js": ("monitor.js", "text/javascript; charset=utf-8"),
    "/monitor.css": ("monitor.css", "text/css; charset=utf-8"),
}
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cache-Control": "no-store",
}


def validate_listen_address(listen_address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Loopback, or a Tailscale address (100.64.0.0/10, fd7a:115c:a1e0::/48). Everything else is refused."""
    try:
        address = ipaddress.ip_address(listen_address)
    except ValueError:
        raise ConfigurationError(
            f"monitor listen address '{listen_address}' is not an IP address",
            operator_hint="use 127.0.0.1 (default) or this machine's Tailscale address (`tailscale ip -4`)",
        ) from None
    if address.is_loopback or any(address in network for network in TAILSCALE_NETWORKS):
        return address
    raise ConfigurationError(
        f"the monitor serves only on loopback or a Tailscale address, not {listen_address}",
        operator_hint="to view it from another machine, use an SSH tunnel "
        "(ssh -L 8765:127.0.0.1:8765 <this host>) or this machine's Tailscale "
        "address",
    )


def build_allowed_host_names(listen_address: str, extra_host_names: tuple[str, ...] = ()) -> frozenset[str]:
    host_name = socket.gethostname().lower()
    return frozenset(
        {
            "localhost",
            "127.0.0.1",
            "::1",
            listen_address.lower(),
            host_name,
            host_name.split(".")[0],
            *(name.lower().rstrip(".") for name in extra_host_names),
        }
    )


class SnapshotProvider:
    """Builds snapshots on demand, at most one every SNAPSHOT_CACHE_SECONDS per history window."""

    def __init__(
        self,
        app_config: AppConfig,
        *,
        default_history_hours: float,
        clock: Callable[[], datetime] = utc_now,
        read_schedule: Callable[..., ScheduleState] = read_schedule_state,
    ) -> None:
        self.app_config = app_config
        self.default_history_hours = default_history_hours
        self.clock = clock
        self.read_schedule = read_schedule
        self._lock = threading.Lock()
        self._cache: dict[float, tuple[float, bytes]] = {}

    def snapshot_json(self, history_hours: float | None = None) -> bytes:
        hours = self.default_history_hours if history_hours is None else history_hours
        hours = min(MAXIMUM_HISTORY_HOURS, max(MINIMUM_HISTORY_HOURS, hours))
        with self._lock:
            cached = self._cache.get(hours)
            if cached is not None and time.monotonic() - cached[0] < SNAPSHOT_CACHE_SECONDS:
                return cached[1]
            body = json.dumps(self.build(hours), ensure_ascii=False, default=str).encode("utf-8")
            self._cache[hours] = (time.monotonic(), body)
            return body

    def build(self, history_hours: float) -> dict[str, Any]:
        now = self.clock()
        schedule = self.read_schedule(
            window_start=now - timedelta(hours=history_hours), window_end=now + timedelta(hours=30)
        )
        connection = open_read_only_database(self.app_config.database_path)
        try:
            return build_monitor_snapshot(
                connection, self.app_config, now=now, history_hours=history_hours, schedule=schedule
            )
        finally:
            connection.close()


def build_request_handler(
    snapshot_provider: SnapshotProvider, allowed_host_names: frozenset[str]
) -> type[BaseHTTPRequestHandler]:
    class MonitorRequestHandler(BaseHTTPRequestHandler):
        server_version = "paperwatch-monitor"
        sys_version = ""

        def do_GET(self) -> None:
            self._respond(include_body=True)

        def do_HEAD(self) -> None:
            self._respond(include_body=False)

        def _respond(self, *, include_body: bool) -> None:
            if not self._host_is_allowed():
                self._send(HTTPStatus.MISDIRECTED_REQUEST, b"unknown host\n", "text/plain; charset=utf-8", include_body)
                return
            parsed_url = urllib.parse.urlsplit(self.path)
            if parsed_url.path in STATIC_FILES:
                file_name, content_type = STATIC_FILES[parsed_url.path]
                body = resources.files("paperwatch.monitor").joinpath("static", file_name).read_bytes()
                self._send(HTTPStatus.OK, body, content_type, include_body)
            elif parsed_url.path == "/api/snapshot":
                self._send_snapshot(parsed_url.query, include_body)
            elif parsed_url.path == "/healthz":
                self._send(HTTPStatus.OK, b"ok\n", "text/plain; charset=utf-8", include_body)
            else:
                self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain; charset=utf-8", include_body)

        def _send_snapshot(self, query: str, include_body: bool) -> None:
            hours_values = urllib.parse.parse_qs(query).get("hours", [])
            try:
                history_hours = float(hours_values[0]) if hours_values else None
            except ValueError:
                history_hours = None
            try:
                body = snapshot_provider.snapshot_json(history_hours)
                status = HTTPStatus.OK
            except PaperwatchError as failure:
                body = json.dumps({"error": str(failure), "hint": failure.operator_hint}).encode("utf-8")
                status = HTTPStatus.SERVICE_UNAVAILABLE
            except Exception as unexpected_error:  # keep serving; the page shows the error banner
                LOGGER.exception("monitor snapshot failed")
                body = json.dumps(
                    {"error": f"snapshot failed: {type(unexpected_error).__name__}: {unexpected_error}"}
                ).encode("utf-8")
                status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._send(status, body, "application/json; charset=utf-8", include_body)

        def _host_is_allowed(self) -> bool:
            host_header = (self.headers.get("Host") or "").strip().lower()
            if host_header.startswith("["):  # [::1]:8765
                host_name = host_header[1 : host_header.find("]")] if "]" in host_header else host_header
            else:
                host_name = host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header
            return host_name.rstrip(".") in allowed_host_names

        def _send(self, status: HTTPStatus, body: bytes, content_type: str, include_body: bool) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for header_name, header_value in SECURITY_HEADERS.items():
                self.send_header(header_name, header_value)
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def do_POST(self) -> None:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, b"read-only\n", "text/plain; charset=utf-8", True)

        do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_POST

        def log_message(self, format_text: str, *arguments: Any) -> None:
            LOGGER.debug("monitor %s: " + format_text, self.client_address[0], *arguments)

    return MonitorRequestHandler


def create_monitor_server(
    app_config: AppConfig,
    *,
    listen_address: str = DEFAULT_LISTEN_ADDRESS,
    port: int = DEFAULT_PORT,
    history_hours: float = 36.0,
    extra_host_names: tuple[str, ...] = (),
    snapshot_provider: SnapshotProvider | None = None,
) -> ThreadingHTTPServer:
    address = validate_listen_address(listen_address)
    provider = snapshot_provider or SnapshotProvider(app_config, default_history_hours=history_hours)
    handler_class = build_request_handler(provider, build_allowed_host_names(str(address), extra_host_names))

    class MonitorHttpServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        daemon_threads = True

        def server_bind(self) -> None:  # skip http.server's reverse DNS lookup of the listen address
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.server_address)
            self.server_name, self.server_port = str(address), self.socket.getsockname()[1]

    return MonitorHttpServer((str(address), port), handler_class)
