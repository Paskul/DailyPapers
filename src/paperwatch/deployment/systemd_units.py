"""systemd unit pieces shared by the one-day pilot and the permanent daily schedule."""

from __future__ import annotations

from pathlib import Path


def worker_sandbox_lines(*, worker_account: str, shared_state_group: str, state_dir: Path, output_dir: Path) -> str:
    """The [Service] lines that sandbox `work` and `report` (the processes that talk to the model): the worker
    account, a read-only system, private /tmp and /dev, write access only to state/ and output/, no new privileges,
    and IP traffic to localhost only (a third network layer after nftables and the in-process egress guard)."""
    return f"""\
User={worker_account}
Group={worker_account}
SupplementaryGroups={shared_state_group}
UMask=0007
WorkingDirectory=/
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths={state_dir} {output_dir}
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
ProtectClock=yes
ProtectHostname=yes
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
CapabilityBoundingSet=
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
# Third network layer (after nftables and the in-process egress guard): IP traffic to localhost only.
IPAddressDeny=any
IPAddressAllow=localhost
"""
