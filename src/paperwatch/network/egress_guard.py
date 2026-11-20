"""Process-wide egress guard: a second line of defence behind the OS rules (nftables, service accounts).

Plan part 3 enforces network limits OUTSIDE Python (deploy/nftables). This module adds the same limits INSIDE each
paperwatch process, so a bug or a misleading paper cannot make, for example, the worker open a socket even on a
machine where the nftables rules are not installed yet. It uses a PEP 578 audit hook, which sees every Python-level
DNS lookup, socket connection, datagram send, and program launch before it happens, and raises EgressBlocked.

Each CLI command arms exactly one policy for the life of its process:
  * collector  -> resolve only hosts in the read-only provider rules; connect only to PUBLIC addresses on port 443;
                  launch only the extractor's container runtime.
  * local model -> (work, report, reread, status, doctor) connect only to the configured model server port on a
                  loopback address; resolve nothing else; launch no programs.
  * no network -> (init) no sockets, no programs.
`paperwatch access-check` deliberately arms nothing: it must try forbidden connections to prove the OS blocks them.

The address rule here is also used by the secure fetcher to check the actual connection target on every hop.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import sys
import threading
import urllib.parse
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)
HTTPS_PORT = 443
LOOPBACK_HOST_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})

# Audit events that start another program. "subprocess.Popen" carries the executable and is checked against the
# policy; the others are always refused while a policy is armed.
PROGRAM_LAUNCH_EVENTS_ALWAYS_REFUSED = frozenset({"os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                                                  "os.forkpty", "pty.spawn"})


class EgressBlocked(PermissionError):
    """Raised by the audit hook. A PermissionError, so network libraries report it as a failed connection."""


@dataclass(frozen=True)
class EgressPolicy:
    component_label: str  # e.g. "collector", shown in every refusal
    resolvable_host_names: frozenset[str] = frozenset()  # DNS names this process may look up
    allow_public_internet_https: bool = False  # collector: any PUBLIC address, port 443 only
    allowed_loopback_ports: frozenset[int] = frozenset()  # local model server port(s) on 127.0.0.0/8 or ::1
    allowed_non_loopback_endpoints: frozenset[tuple[str, int]] = frozenset()  # operator-approved LAN model server
    allowed_executables: frozenset[str] = frozenset()  # base names or absolute paths

    def refusal_reason_for_host_lookup(self, host: str | bytes | None) -> str | None:
        host_name = _decode_host(host)
        if host_name is None or _parse_ip_address(host_name) is not None:
            return None  # an IP literal needs no DNS; the connection check decides about it
        normalized_host_name = host_name.lower().rstrip(".")
        if normalized_host_name in self.resolvable_host_names:
            return None
        if normalized_host_name in LOOPBACK_HOST_NAMES and self.allowed_loopback_ports:
            return None
        return f"{self.component_label} may not resolve host '{normalized_host_name}'"

    def refusal_reason_for_connection(self, address: object) -> str | None:
        if not isinstance(address, tuple) or len(address) < 2:
            return f"{self.component_label} may not use non-IP sockets ({address!r})"
        host_text, port = _decode_host(address[0]), address[1]
        ip_address = _parse_ip_address(host_text)
        if ip_address is None:
            return f"{self.component_label} may not connect to unresolved address {host_text!r}"
        if ip_address.is_loopback and port in self.allowed_loopback_ports:
            return None
        if (str(ip_address), port) in self.allowed_non_loopback_endpoints:
            return None
        if self.allow_public_internet_https and port == HTTPS_PORT and is_public_internet_address(ip_address):
            return None
        return f"{self.component_label} may not connect to {ip_address} port {port}"

    def refusal_reason_for_program(self, executable: object) -> str | None:
        executable_text = os.fsdecode(executable) if isinstance(executable, (str, bytes, os.PathLike)) else ""
        if executable_text and (executable_text in self.allowed_executables or
                                os.path.basename(executable_text) in self.allowed_executables):
            return None
        return f"{self.component_label} may not start program {executable_text or executable!r}"


def is_public_internet_address(ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address | str) -> bool:
    """True only for globally routable unicast addresses: never private, loopback, link-local, reserved, shared
    (CGNAT), multicast, or unspecified. IPv4-mapped IPv6 addresses are judged by their IPv4 address."""
    if isinstance(ip_address, str):
        parsed_address = _parse_ip_address(ip_address)
        if parsed_address is None:
            return False
        ip_address = parsed_address
    if isinstance(ip_address, ipaddress.IPv6Address) and ip_address.ipv4_mapped is not None:
        ip_address = ip_address.ipv4_mapped
    return (ip_address.is_global and not ip_address.is_private and not ip_address.is_loopback
            and not ip_address.is_link_local and not ip_address.is_reserved and not ip_address.is_multicast
            and not ip_address.is_unspecified)


# --------------------------------------------------------------------------------------------------------------------
# Policy builders, one per component
# --------------------------------------------------------------------------------------------------------------------


def build_no_network_policy(component_label: str = "init") -> EgressPolicy:
    return EgressPolicy(component_label=component_label)


def build_local_model_policy(model_base_url: str, component_label: str) -> EgressPolicy:
    """Only the configured model server. Config validation already requires a loopback host unless the operator
    explicitly allowed a model server on their own network; then exactly that host and port are allowed."""
    parsed_url = urllib.parse.urlparse(model_base_url)
    model_host = (parsed_url.hostname or "").lower()
    model_port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    model_ip_address = _parse_ip_address(model_host)
    if model_host in LOOPBACK_HOST_NAMES or (model_ip_address is not None and model_ip_address.is_loopback):
        return EgressPolicy(component_label=component_label, allowed_loopback_ports=frozenset({model_port}))
    resolved_endpoints = frozenset((resolved_ip, model_port) for resolved_ip in _resolve_now(model_host, model_port))
    return EgressPolicy(component_label=component_label, resolvable_host_names=frozenset({model_host}),
                        allowed_non_loopback_endpoints=resolved_endpoints)


def build_monitor_policy(systemctl_path: str = "/usr/bin/systemctl") -> EgressPolicy:
    """`monitor`: no outbound connection at all (it only accepts the page's requests), and exactly one program, the
    read-only `systemctl show` query for the services' states."""
    return EgressPolicy(component_label="monitor", allowed_executables=frozenset({systemctl_path}))


def build_collector_policy(permitted_provider_hosts: frozenset[str], container_runtime: str) -> EgressPolicy:
    return EgressPolicy(component_label="collector", resolvable_host_names=permitted_provider_hosts,
                        allow_public_internet_https=True, allowed_executables=frozenset({container_runtime}))


# --------------------------------------------------------------------------------------------------------------------
# Arming the policy
# --------------------------------------------------------------------------------------------------------------------

_policy_lock = threading.Lock()
_active_policy: EgressPolicy | None = None
_audit_hook_installed = False


def arm_egress_policy(policy: EgressPolicy) -> None:
    """Arm `policy` for the rest of this process. Audit hooks cannot be removed, so this is one-way by design."""
    global _active_policy, _audit_hook_installed
    with _policy_lock:
        if not _audit_hook_installed:
            sys.addaudithook(_egress_audit_hook)
            _audit_hook_installed = True
        _active_policy = policy


def active_egress_policy() -> EgressPolicy | None:
    return _active_policy


@contextmanager
def temporarily_armed_egress_policy(policy: EgressPolicy) -> Iterator[EgressPolicy]:
    """TESTS ONLY: arm a policy for a block of code inside a long-lived test process, then restore the previous one."""
    global _active_policy
    previous_policy = _active_policy
    arm_egress_policy(policy)
    try:
        yield policy
    finally:
        with _policy_lock:
            _active_policy = previous_policy


def _egress_audit_hook(event: str, arguments: tuple) -> None:
    policy = _active_policy
    if policy is None:
        return
    refusal_reason: str | None = None
    if event == "socket.getaddrinfo":
        refusal_reason = policy.refusal_reason_for_host_lookup(arguments[0])
    elif event in ("socket.connect", "socket.sendto", "socket.sendmsg"):
        target_address = arguments[1] if len(arguments) > 1 else None
        if event != "socket.connect" and target_address is None:
            return  # a send on an already-connected socket: its connect was checked
        refusal_reason = policy.refusal_reason_for_connection(target_address)
    elif event == "subprocess.Popen":
        refusal_reason = policy.refusal_reason_for_program(arguments[0])
    elif event in PROGRAM_LAUNCH_EVENTS_ALWAYS_REFUSED:
        refusal_reason = f"{policy.component_label} may not start programs ({event})"
    if refusal_reason is not None:
        LOGGER.warning("egress guard refused: %s", refusal_reason)
        raise EgressBlocked(f"egress guard: {refusal_reason}")


# --------------------------------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------------------------------


def _decode_host(host: object) -> str | None:
    if isinstance(host, bytes):
        return host.decode("ascii", errors="replace")
    return host if isinstance(host, str) else None


def _parse_ip_address(host_text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not host_text:
        return None
    try:
        return ipaddress.ip_address(host_text.split("%", 1)[0])  # drop an IPv6 zone index such as fe80::1%eth0
    except ValueError:
        return None


def _resolve_now(host_name: str, port: int) -> set[str]:
    try:
        return {address_info[4][0] for address_info in socket.getaddrinfo(host_name, port, type=socket.SOCK_STREAM)}
    except OSError:
        return set()
