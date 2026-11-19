"""Plan part 3: the per-process egress guard, the sandboxed extractor (including a separate parser timeout), the
worker's dispatch-time recheck, stricter topic validation, and `paperwatch access-check`.

The real-Podman test runs only when podman is installed and PAPERWATCH_TEST_EXTRACTOR_IMAGE names a built image.
Everything else uses a stub container runtime, so timeouts and limits are tested on any machine.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import stat
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from paperwatch.commands.access_check_command import build_minimal_pdf_bytes
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.config import ExtractorSettings
from paperwatch.errors import ExtractionError, ExtractionTimedOut, SourceAccessDenied
from paperwatch.network.egress_guard import (EgressBlocked, build_collector_policy, build_local_model_policy,
                                             build_no_network_policy, is_public_internet_address,
                                             temporarily_armed_egress_policy)
from paperwatch.sandboxed_extractor import build_extractor_container_command, extract_text_in_sandbox
from paperwatch.source_cache import store_source_bytes

from conftest import RECORDED_PAPER_FIXTURE, build_in_process_app, find_unused_local_port

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
EXTRACTOR_ENTRYPOINT = REPOSITORY_DIR / "deploy" / "extractor" / "extract_text.py"
PINNED_TEST_IMAGE = "0" * 64
TEST_EDITION_ID = "e_1234567890abcdef"


# --------------------------------------------------------------------------------------------------------------------
# Egress guard
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("address, expected", [
    ("93.184.216.34", True), ("2606:2800:220:1:248:1893:25c8:1946", True), ("10.0.0.1", False),
    ("127.0.0.1", False), ("169.254.169.254", False), ("100.64.1.1", False), ("192.0.2.10", False),
    ("::1", False), ("fe80::1", False), ("fd12::1", False), ("::ffff:10.0.0.1", False), ("0.0.0.0", False),
    ("224.0.0.1", False), ("not-an-address", False)])
def test_only_globally_routable_unicast_addresses_count_as_public(address, expected):
    assert is_public_internet_address(address) is expected


def test_worker_policy_allows_only_the_local_model_port():
    with socket.socket() as model_listener:
        model_listener.bind(("127.0.0.1", 0))
        model_listener.listen()
        model_port = model_listener.getsockname()[1]
        with temporarily_armed_egress_policy(build_local_model_policy(f"http://127.0.0.1:{model_port}/v1", "work")):
            socket.create_connection(("127.0.0.1", model_port), timeout=2).close()  # allowed
            # The hook raises BEFORE the system call, so no packet is ever sent for these.
            for forbidden_target in (("93.184.216.34", 443), ("127.0.0.1", find_unused_local_port()),
                                     ("169.254.169.254", 80)):
                with pytest.raises(EgressBlocked):
                    socket.create_connection(forbidden_target, timeout=2)
            with pytest.raises(EgressBlocked, match="may not resolve host 'example.org'"):
                socket.getaddrinfo("example.org", 443)
            with pytest.raises(EgressBlocked, match="may not start program"):
                subprocess.run(["true"], check=False)
            with pytest.raises(EgressBlocked):
                os.system("true")
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as datagram_socket:
                with pytest.raises(EgressBlocked):
                    datagram_socket.sendto(b"x", ("8.8.8.8", 53))


def test_collector_policy_allows_public_https_to_listed_hosts_and_its_container_runtime_only():
    collector_policy = build_collector_policy(frozenset({"export.arxiv.org"}), "podman")
    assert collector_policy.refusal_reason_for_host_lookup("export.arxiv.org") is None
    assert collector_policy.refusal_reason_for_host_lookup("evil.example") is not None
    assert collector_policy.refusal_reason_for_connection(("93.184.216.34", 443)) is None
    assert collector_policy.refusal_reason_for_connection(("93.184.216.34", 80)) is not None
    assert collector_policy.refusal_reason_for_program("/usr/bin/podman") is None
    assert collector_policy.refusal_reason_for_program("/bin/sh") is not None
    with temporarily_armed_egress_policy(collector_policy):
        for private_target in (("127.0.0.1", 8080), ("10.0.0.1", 443), ("169.254.169.254", 443)):
            with pytest.raises(EgressBlocked):
                socket.create_connection(private_target, timeout=2)
        with pytest.raises(EgressBlocked):
            socket.create_connection(("::1", 443), timeout=2)


def test_no_network_policy_refuses_everything():
    with temporarily_armed_egress_policy(build_no_network_policy()):
        with pytest.raises(EgressBlocked):
            socket.create_connection(("127.0.0.1", find_unused_local_port()), timeout=2)


def test_cli_arms_the_policy_of_each_command(initialized_home):
    """Run the real CLI wiring in a fresh process and try a forbidden connection from inside it."""
    probe_script = textwrap.dedent(f"""
        import socket, sys
        from paperwatch.cli import arm_egress_policy_for_command
        from paperwatch.config import load_app_config
        from paperwatch.network.egress_guard import EgressBlocked
        arm_egress_policy_for_command(sys.argv[1], load_app_config(__import__("pathlib").Path({str(initialized_home.home_dir)!r})))
        try:
            socket.create_connection(("192.0.2.1", 443), timeout=2)
        except EgressBlocked as refusal:
            print("BLOCKED", refusal)
    """)
    for command_name in ("work", "report", "collect"):
        completed = subprocess.run([sys.executable, "-c", probe_script, command_name], capture_output=True,
                                   text=True, timeout=60)
        assert "BLOCKED" in completed.stdout, completed.stderr


# --------------------------------------------------------------------------------------------------------------------
# Sandboxed extractor
# --------------------------------------------------------------------------------------------------------------------


def write_stub_container_runtime(directory: Path, run_behaviour: str) -> Path:
    """A fake `podman`: 'rm' succeeds; 'run' does `run_behaviour` (a shell snippet) and records its arguments."""
    stub_path = directory / "fake-podman"
    stub_path.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        if [ "$1" = "rm" ]; then echo "$@" >> "{directory}/removed"; exit 0; fi
        printf '%s\\n' "$@" > "{directory}/run-arguments"
        {run_behaviour}
    """))
    stub_path.chmod(0o755)
    return stub_path


def cached_pdf(cache_dir: Path) -> Path:
    return store_source_bytes(cache_dir, TEST_EDITION_ID, build_minimal_pdf_bytes("hello"), "application/pdf").file_path


def stub_extractor_settings(stub_path: Path, **overrides) -> ExtractorSettings:
    return replace(ExtractorSettings(container_runtime=str(stub_path), image=PINNED_TEST_IMAGE), **overrides)


def test_extractor_command_has_no_network_a_read_only_root_and_one_read_only_input(tmp_path):
    input_path = cached_pdf(tmp_path / "cache")
    command = build_extractor_container_command(ExtractorSettings(image=PINNED_TEST_IMAGE), input_path, "c1")
    for required_option in ("--network=none", "--read-only", "--pull=never", "--cap-drop=all",
                            "--security-opt=no-new-privileges", "--rm"):
        assert required_option in command
    mounts = [option for option in command if option.startswith(("--mount", "--volume", "-v"))]
    assert mounts == [f"--mount=type=bind,source={input_path},target=/input/source,readonly"]
    assert command[-1] == PINNED_TEST_IMAGE  # fixed entrypoint: nothing after the image
    assert not any(option.startswith(("--env", "-e", "--secret", "--device")) for option in command)


def test_parser_timeout_stops_the_container_and_the_next_file_still_extracts(tmp_path):
    input_path = cached_pdf(tmp_path / "cache")
    hanging_runtime = write_stub_container_runtime(tmp_path, "exec sleep 30")
    with pytest.raises(ExtractionTimedOut):
        extract_text_in_sandbox(tmp_path / "cache", input_path,
                                stub_extractor_settings(hanging_runtime, timeout_seconds=1.0))
    assert "rm --force paperwatch-extract-" in (tmp_path / "removed").read_text()

    working_runtime = write_stub_container_runtime(tmp_path, r"printf 'page one\fpage two'")
    extracted = extract_text_in_sandbox(tmp_path / "cache", input_path, stub_extractor_settings(working_runtime))
    assert extracted.text == "page one\fpage two" and extracted.extractor_image == PINNED_TEST_IMAGE


@pytest.mark.parametrize("run_behaviour, settings_overrides, expected_message", [
    ("head -c 3000000 /dev/zero", {"max_output_megabytes": 1}, "output exceeds"),
    ("echo 'broken pdf' >&2; exit 3", {}, "exited with status 3: broken pdf"),
])
def test_extractor_failures_are_reported_and_bounded(tmp_path, run_behaviour, settings_overrides, expected_message):
    input_path = cached_pdf(tmp_path / "cache")
    runtime = write_stub_container_runtime(tmp_path, run_behaviour)
    with pytest.raises(ExtractionError, match=expected_message):
        extract_text_in_sandbox(tmp_path / "cache", input_path, stub_extractor_settings(runtime, **settings_overrides))


def test_extractor_accepts_only_cached_source_files_and_a_configured_image(tmp_path):
    runtime = write_stub_container_runtime(tmp_path, "exit 0")
    outside_file = tmp_path / "notes.pdf"
    outside_file.write_bytes(b"%PDF-1.4")
    with pytest.raises(SourceAccessDenied):
        extract_text_in_sandbox(tmp_path / "cache", outside_file, stub_extractor_settings(runtime))
    symlinked_file = tmp_path / "cache" / "sources" / TEST_EDITION_ID / ("a" * 32 + ".pdf")
    symlinked_file.parent.mkdir(parents=True)
    symlinked_file.symlink_to(outside_file)
    with pytest.raises(SourceAccessDenied):
        extract_text_in_sandbox(tmp_path / "cache", symlinked_file, stub_extractor_settings(runtime))
    with pytest.raises(ExtractionError, match="not configured"):
        extract_text_in_sandbox(tmp_path / "cache", cached_pdf(tmp_path / "cache"),
                                ExtractorSettings(container_runtime=str(runtime), image=""))


def load_extractor_entrypoint(input_path: Path):
    module_spec = importlib.util.spec_from_file_location("paperwatch_extractor_entrypoint", EXTRACTOR_ENTRYPOINT)
    entrypoint_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(entrypoint_module)
    entrypoint_module.INPUT_PATH = str(input_path)
    return entrypoint_module


@pytest.mark.skipif(not Path("/usr/bin/pdftotext").exists(), reason="poppler pdftotext is not installed on the host")
def test_image_entrypoint_extracts_pdf_text_with_the_pinned_parser(tmp_path):
    input_path = tmp_path / "source"
    input_path.write_bytes(build_minimal_pdf_bytes("Exit corridor monitor"))
    runner = (f"import importlib.util, sys; spec = importlib.util.spec_from_file_location('e', {str(EXTRACTOR_ENTRYPOINT)!r}); "
              f"module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
              f"module.INPUT_PATH = {str(input_path)!r}; sys.exit(module.main())")
    completed = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "Exit corridor monitor" in completed.stdout


def test_image_entrypoint_extracts_xml_blocks_and_refuses_entity_declarations(tmp_path, capsys):
    input_path = tmp_path / "source"
    input_path.write_text("<article><front><article-title>A Title</article-title></front><body><sec>"
                          "<title>Methods</title><p>We check <italic>plans</italic>.</p></sec></body></article>")
    assert load_extractor_entrypoint(input_path).main() == 0
    assert capsys.readouterr().out == "## A Title\n\n## Methods\n\nWe check plans.\n"

    input_path.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;&a;</x>')
    assert load_extractor_entrypoint(input_path).main() == 2
    input_path.write_bytes(b"\x00\x01binary")
    assert load_extractor_entrypoint(input_path).main() == 2


@pytest.mark.skipif(shutil.which("podman") is None or not os.environ.get("PAPERWATCH_TEST_EXTRACTOR_IMAGE"),
                    reason="needs podman and PAPERWATCH_TEST_EXTRACTOR_IMAGE (from deploy/build_extractor_image.sh)")
def test_real_podman_sandbox_extracts_text(tmp_path):
    settings = ExtractorSettings(image=os.environ["PAPERWATCH_TEST_EXTRACTOR_IMAGE"], timeout_seconds=60)
    input_path = store_source_bytes(tmp_path / "cache", TEST_EDITION_ID,
                                    build_minimal_pdf_bytes("sandboxed hello"), "application/pdf").file_path
    assert "sandboxed hello" in extract_text_in_sandbox(tmp_path / "cache", input_path, settings).text


# --------------------------------------------------------------------------------------------------------------------
# Worker dispatch recheck, config validation, access-check
# --------------------------------------------------------------------------------------------------------------------


def test_worker_rechecks_a_queued_edition_against_the_catalog_before_reading(tmp_path):
    from paperwatch.catalog import record_collected_paper
    from paperwatch.commands.collect_command import _queue_review_if_promoted
    from paperwatch.sources.fixture_source import RecordedFixtureSource

    app = build_in_process_app(tmp_path / "home")
    paper_id = record_collected_paper(app.connection, RecordedFixtureSource(RECORDED_PAPER_FIXTURE)
                                      .fetch_records()[0]).paper_id
    _queue_review_if_promoted(app, paper_id)
    app.connection.execute("UPDATE jobs SET input_json = json_set(input_json, '$.edition_id', 'e_ffffffffffffffff')")
    cycle_report = run_single_work_cycle(app, probe_paused_model_now=False)
    assert cycle_report.outcome == WorkCycleOutcome.JOB_RETRY_SCHEDULED
    assert "not in the catalog" in str(cycle_report.failure)
    assert app.model_client.received_requests == []  # nothing was read
    app.close()


@pytest.mark.parametrize("topic_line, expected_message", [
    ('public_queries = ["two\\nlines"]', "printable characters on one line"),
    ('arxiv_categories = ["cs.RO; rm -rf /"]', "not an arXiv category"),
    ('zotero_collections = ["My Collection"]', "not a Zotero collection key"),
])
def test_public_topic_terms_are_validated_before_they_can_leave_the_machine(initialized_home, topic_line,
                                                                            expected_message):
    (initialized_home.home_dir / "topics.toml").write_text(
        f'[[topics]]\nid = "t"\nquestion = "Q?"\n{topic_line}\n')
    result = initialized_home.run_cli("status")
    assert result.exit_code == 2 and expected_message in result.stderr


def test_unpinned_extractor_image_is_refused(initialized_home):
    initialized_home.replace_in_config('image = ""', 'image = "localhost/paperwatch-extractor:latest"')
    result = initialized_home.run_cli("status")
    assert result.exit_code == 2 and "is not pinned" in result.stderr


def test_access_check_reports_each_area_from_the_running_account(initialized_home):
    credentials_dir = initialized_home.home_dir / "credentials"
    credentials_dir.mkdir()
    (credentials_dir / "collector.toml").write_text('[openalex]\napi_key = "k"\n')
    (credentials_dir / "collector.toml").chmod(0o644)
    refused = initialized_home.run_cli("access-check", "--component", "worker", "--public-probe-address",
                                       "10.0.0.1", "--skip-network-probes")
    assert refused.exit_code == 2 and "not a public IP address" in refused.stderr
    result = initialized_home.run_cli("access-check", "--component", "collector", "--skip-network-probes")
    assert result.exit_code != 0  # this test account is not the collector service account
    for expected_fragment in ("service account: running as", "provider rules valid: rules parsed",
                              "credentials private: credentials file", "accessible to other accounts",
                              "output directory", "SKIP  network", "extractor"):
        assert expected_fragment in result.stdout, result.stdout
    assert stat.S_IMODE((credentials_dir / "collector.toml").stat().st_mode) == 0o644  # reported, not changed
