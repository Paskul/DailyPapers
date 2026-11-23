"""Shared helpers: run the real CLI in a subprocess against a temporary home directory."""

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RECORDED_PAPER_FIXTURE = FIXTURES_DIR / "recorded_paper.json"
ARXIV_ROUTE_FIXTURE = FIXTURES_DIR / "two_routes" / "arxiv_listing.json"
OPENALEX_ROUTE_FIXTURE = FIXTURES_DIR / "two_routes" / "openalex_record.json"
PROMPT_INJECTION_FIXTURE = FIXTURES_DIR / "prompt_injection_papers.json"

EXAMPLE_TOPIC_TOML = """
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
public_queries = ["multi-robot runtime assurance"]
arxiv_categories = ["cs.RO"]
"""


ONNX_EMBEDDING_BACKEND_LINE = 'embedding_backend = "onnx"'
HASHING_EMBEDDING_BACKEND_LINE = 'embedding_backend = "hashing"'


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
        completed = subprocess.run(
            self.cli_command(*cli_arguments), capture_output=True, text=True, timeout=120, check=False
        )
        return CliResult(completed.returncode, completed.stdout, completed.stderr)

    def cli_command(self, *cli_arguments: str) -> list[str]:
        return [sys.executable, "-m", "paperwatch", "--home", str(self.home_dir), *cli_arguments]

    def start_cli(self, *cli_arguments: str, environment_overrides: dict[str, str] | None = None) -> subprocess.Popen:
        """Start the CLI without waiting, e.g. a worker the test will kill or race."""
        return subprocess.Popen(
            self.cli_command(*cli_arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **(environment_overrides or {})},
        )

    def use_short_leases(self, lease_seconds: int = 2, renewal_interval_seconds: int = 1) -> None:
        self.replace_in_config("lease_seconds = 300", f"lease_seconds = {lease_seconds}")
        self.replace_in_config(
            "lease_renewal_interval_seconds = 60", f"lease_renewal_interval_seconds = {renewal_interval_seconds}"
        )

    def replace_in_config(self, old_text: str, new_text: str) -> None:
        config_path = self.home_dir / "config.toml"
        config_text = config_path.read_text()
        assert old_text in config_text, f"{old_text!r} not found in config.toml"
        config_path.write_text(config_text.replace(old_text, new_text))

    def use_fake_model(self) -> None:
        self.replace_in_config('backend = "llama_cpp"', 'backend = "fake"')
        self.use_hashing_test_embedder()

    def use_hashing_test_embedder(self) -> None:
        """Screening with the offline word-hashing embedder instead of the pinned ONNX model (not installed in CI).
        Idempotent, so it can follow use_fake_model."""
        if HASHING_EMBEDDING_BACKEND_LINE not in (self.home_dir / "config.toml").read_text():
            self.replace_in_config(ONNX_EMBEDDING_BACKEND_LINE, HASHING_EMBEDDING_BACKEND_LINE)

    def use_unreachable_model_endpoint(self) -> None:
        self.replace_in_config('"http://127.0.0.1:8080/v1"', f'"http://127.0.0.1:{find_unused_local_port()}/v1"')

    def make_todays_report_publishable_early(self) -> str:
        """`report` publishes a date only after its cutoff, or early (`--early`) for today's date. A subprocess
        test cannot move the clock, so pick a report timezone in which today's 23:59 cutoff is still at least an
        hour ahead, and a floor of 0 (the fake model's ratings stay below 65). Returns today's date there."""
        utc_moment = datetime.now(UTC)
        timezone_name = "UTC" if utc_moment.hour < 22 else "Etc/GMT+12"  # UTC-12: about 10:00-11:59 then
        self.replace_in_config('timezone = "America/New_York"', f'timezone = "{timezone_name}"')
        self.replace_in_config('publish_time = "07:00"', 'publish_time = "23:59"')
        self.replace_in_config("minimum_reading_priority = 65", "minimum_reading_priority = 0")
        return utc_moment.astimezone(ZoneInfo(timezone_name)).date().isoformat()

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


def build_in_process_app(
    home_dir: Path,
    *,
    topics_toml: str = EXAMPLE_TOPIC_TOML,
    config_replacements: dict[str, str] | None = None,
    model_client=None,
    use_configured_model_backend: bool = False,
    use_onnx_embedder: bool = False,
):
    """Init a home in this process, write topics and config edits, and load an AppContext.

    The model is a FakeModelClient unless `model_client` is given, or `use_configured_model_backend` asks for the
    client config.toml describes (e.g. the real llama.cpp client pointed at a FakeLlamaServer). Screening uses the
    offline hashing embedder unless `use_onnx_embedder` keeps the template's pinned ONNX model."""
    from paperwatch.app_context import AppContext
    from paperwatch.commands.init_command import run_init_command
    from paperwatch.llm.fake_client import FakeModelClient

    run_init_command(home_dir, None)
    (home_dir / "topics.toml").write_text(topics_toml)
    config_path = home_dir / "config.toml"
    config_text = config_path.read_text()
    if not use_onnx_embedder:
        config_text = config_text.replace(ONNX_EMBEDDING_BACKEND_LINE, HASHING_EMBEDDING_BACKEND_LINE)
    for old_text, new_text in (config_replacements or {}).items():
        assert old_text in config_text, f"{old_text!r} not found in config.toml"
        config_text = config_text.replace(old_text, new_text)
    config_path.write_text(config_text)
    if use_configured_model_backend:
        return AppContext.load(home_dir)
    return AppContext.load(home_dir, model_client=model_client if model_client is not None else FakeModelClient())


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


@pytest.fixture(autouse=True)
def no_real_waiting_for_paused_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source-text stage sleeps through a short provider pause (collection.source_text_pause_wait_seconds). Tests
    never sleep: a test that wants the wait to succeed replaces this with a function that advances its clock."""
    from paperwatch.collection import source_text

    monkeypatch.setattr(source_text, "wait_for_provider_pause", lambda seconds: None)
