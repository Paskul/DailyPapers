"""The monitor's defaults, apart from the server: the CLI's argument parser shows them in `--help`, and building the
parser must not import the server and its snapshot code (that takes a large share of a second)."""

from __future__ import annotations

DEFAULT_PORT = 8765
DEFAULT_LISTEN_ADDRESS = "127.0.0.1"
DEFAULT_HISTORY_HOURS = 24.0  # how far back the page looks unless it asks for another window
