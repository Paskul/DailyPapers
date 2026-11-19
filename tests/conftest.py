"""Shared helpers: run the real CLI in a subprocess against a temporary home directory."""

from __future__ import annotations

import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RECORDED_PAPER_FIXTURE = FIXTURES_DIR / "recorded_paper.json"

EXAMPLE_TOPIC_TOML = """
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
public_queries = ["multi-robot runtime assurance"]
arxiv_categories = ["cs.RO"]
"""


@dataclass(frozen=True)
class CliResult:
    exit_code: int
    stdout: str
    stderr: str


class PaperwatchHome:
    """A temporary paperwatch home directory plus helpers to drive and inspect it."""

    def __init__(self, home_dir: Path) -> None:
        self.home_dir = home_dir

    def run_cli(self, *cli_arguments: str) -> CliResult:
        completed = subprocess.run([sys.executable, "-m", "paperwatch", "--home", str(self.home_dir), *cli_arguments],
                                   capture_output=True, text=True, timeout=120)
        return CliResult(completed.returncode, completed.stdout, completed.stderr)

    def replace_in_config(self, old_text: str, new_text: str) -> None:
        config_path = self.home_dir / "config.toml"
        config_text = config_path.read_text()
        assert old_text in config_text, f"{old_text!r} not found in config.toml"
        config_path.write_text(config_text.replace(old_text, new_text))

    def use_fake_model(self) -> None:
        self.replace_in_config('backend = "llama_cpp"', 'backend = "fake"')

    def use_unreachable_model_endpoint(self) -> None:
        self.replace_in_config('"http://127.0.0.1:8080/v1"', f'"http://127.0.0.1:{find_unused_local_port()}/v1"')

    def add_example_topic(self) -> None:
        with (self.home_dir / "topics.toml").open("a") as topics_file:
            topics_file.write(EXAMPLE_TOPIC_TOML)

    def query(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        connection = sqlite3.connect(self.home_dir / "state" / "paperwatch.sqlite3")
        connection.row_factory = sqlite3.Row
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()


def find_unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


@pytest.fixture
def initialized_home(tmp_path: Path) -> PaperwatchHome:
    paperwatch_home = PaperwatchHome(tmp_path / "paperwatch-home")
    init_result = paperwatch_home.run_cli("init")
    assert init_result.exit_code == 0, init_result.stderr
    return paperwatch_home
