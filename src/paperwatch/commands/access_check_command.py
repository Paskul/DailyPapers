"""`paperwatch access-check --component collector|worker`: verify the plan-part-3 access limits from the account
that actually runs that component (deploy/check_access_as_service_accounts.sh runs it for both).

Unlike every other command, this one arms NO in-process egress guard: it deliberately tries connections the
component must not make, to prove the OS (nftables, file permissions) blocks them.

  account      the running account is the configured one and is in the shared state group
  files        provider rules not writable; credentials private to the collector (unreadable by the worker);
               output writable only by the worker; state/ and its WAL files usable by both; cache written by the
               collector only
  network      probes: the worker reaches only the local model (blocked both by name and by literal IPv4/IPv6
               address); the collector reaches public HTTPS but not the model
  model-server built-in tools / agent / MCP disabled, no runtime downloads, which account runs it
  extractor    (collector) podman present, image pinned and present, and a smoke test extracts a generated PDF
"""

from __future__ import annotations

import argparse
import errno
import grp
import os
import pwd
import shutil
import socket
import subprocess
import urllib.parse
from pathlib import Path

from paperwatch.app_context import AppContext
from paperwatch.commands.check_results import FAIL, PASS, SKIP, WARN, CheckResult, print_check_results
from paperwatch.errors import ConfigurationError, PaperwatchError
from paperwatch.network.collector_credentials import describe_credentials_file_problem
from paperwatch.network.egress_guard import is_public_internet_address
from paperwatch.network.provider_rules import load_provider_access_rules, packaged_provider_rules_path
from paperwatch.sandboxed_extractor import extract_text_in_sandbox
from paperwatch.source_cache import store_source_bytes

COMPONENT_COLLECTOR = "collector"
COMPONENT_WORKER = "worker"
NETWORK_PROBE_TIMEOUT_SECONDS = 4.0
PUBLIC_HTTPS_PROBE_HOST = "export.arxiv.org"  # a host the collector is allowed to contact anyway
EXTRACTOR_SMOKE_TEST_EDITION_ID = "e_0000000000000000"
EXTRACTOR_SMOKE_TEST_MARKER = "paperwatch extractor smoke test"
# llama-server flags that give the model tools or let the server download at run time.
FORBIDDEN_MODEL_SERVER_FLAGS = ("--tools", "--agent", "-ag", "--mcp-servers-config", "--mcp-servers-json",
                                "--ui-mcp-proxy", "--webui-mcp-proxy", "--allow-unauthenticated-tools",
                                "--hf-repo", "-hf", "-hfr", "--model-url", "-mu", "--connect")
FORBIDDEN_MODEL_SERVER_ENVIRONMENT_NAMES = ("LLAMA_ARG_TOOLS", "LLAMA_ARG_AGENT", "LLAMA_ARG_MCP_SERVERS_CONFIG",
                                            "LLAMA_ARG_MCP_SERVERS_JSON", "LLAMA_ARG_UI_MCP_PROXY")
PROVIDER_KEY_ENVIRONMENT_HINTS = ("ZOTERO", "OPENALEX", "API_KEY")


def run_access_check_command(app: AppContext, arguments: argparse.Namespace) -> int:
    component = arguments.component
    literal_probe_addresses = _validate_probe_addresses(getattr(arguments, "public_probe_addresses", None) or [])
    check_results = [*_check_running_account(app, component), *_check_files_and_directories(app, component)]
    if arguments.skip_network_probes:
        check_results.append(CheckResult("network", "egress probes", SKIP, "skipped (--skip-network-probes)"))
    else:
        check_results += _probe_network_egress(app, component, literal_probe_addresses)
    check_results += _check_model_server_isolation(app)
    if component == COMPONENT_COLLECTOR:
        check_results += _check_extractor_sandbox(app)
    return print_check_results(check_results, f"access-check ({component})")


# --------------------------------------------------------------------------------------------------------------------
# Account
# --------------------------------------------------------------------------------------------------------------------


def _check_running_account(app: AppContext, component: str) -> list[CheckResult]:
    access_settings = app.config.access
    expected_account = access_settings.collector_account if component == COMPONENT_COLLECTOR else \
        access_settings.worker_account
    running_account = pwd.getpwuid(os.geteuid()).pw_name
    results = [CheckResult(
        "account", "service account", PASS if running_account == expected_account else FAIL,
        f"running as '{running_account}', expected '{expected_account}'",
        hint=None if running_account == expected_account else
        f"run deploy/setup_service_accounts.sh, then: sudo -u {expected_account} paperwatch --home "
        f"{app.config.home_dir} access-check --component {component}")]
    try:
        shared_group_id = grp.getgrnam(access_settings.shared_state_group).gr_gid
        in_shared_group = shared_group_id in os.getgroups() or shared_group_id == os.getegid()
        results.append(CheckResult("account", "shared state group", PASS if in_shared_group else FAIL,
                                   f"'{access_settings.shared_state_group}' "
                                   f"{'includes' if in_shared_group else 'does not include'} this account"))
    except KeyError:
        results.append(CheckResult("account", "shared state group", FAIL,
                                   f"group '{access_settings.shared_state_group}' does not exist",
                                   hint="run deploy/setup_service_accounts.sh"))
    return results


# --------------------------------------------------------------------------------------------------------------------
# Files and directories
# --------------------------------------------------------------------------------------------------------------------


def _check_files_and_directories(app: AppContext, component: str) -> list[CheckResult]:
    app_config = app.config
    is_collector = component == COMPONENT_COLLECTOR
    rules_path = packaged_provider_rules_path()
    rules_writable = os.access(rules_path, os.W_OK) or os.access(rules_path.parent, os.W_OK)
    results = [CheckResult("files", "provider rules read-only", FAIL if rules_writable else PASS,
                           f"{rules_path} {'is WRITABLE by' if rules_writable else 'is read-only for'} this account",
                           hint="install paperwatch root-owned (deploy/install_application.sh)" if rules_writable
                           else None)]
    try:
        load_provider_access_rules()
        results.append(CheckResult("files", "provider rules valid", PASS, "rules parsed and validated"))
    except PaperwatchError as rules_problem:
        results.append(CheckResult("files", "provider rules valid", FAIL, str(rules_problem), rules_problem))

    credentials_path = app_config.collector_credentials_path
    if is_collector:
        problem = describe_credentials_file_problem(credentials_path) if credentials_path.exists() else None
        if not credentials_path.exists():
            results.append(CheckResult("files", "credentials private", WARN,
                                       f"{credentials_path} does not exist (no Zotero / OpenAlex keys yet)"))
        else:
            results.append(CheckResult("files", "credentials private", FAIL if problem else PASS,
                                       problem or f"{credentials_path} belongs to this account, mode 0600"))
    else:
        credentials_readable = _can_read(credentials_path)
        results.append(CheckResult("files", "credentials hidden", FAIL if credentials_readable else PASS,
                                   f"{credentials_path} is {'READABLE' if credentials_readable else 'not readable'} "
                                   "by the worker account"))

    output_writable = os.access(app_config.output_dir, os.W_OK)
    output_should_be_writable = not is_collector
    results.append(CheckResult("files", "output directory", PASS if output_writable == output_should_be_writable
                               else FAIL, f"{app_config.output_dir} is {'writable' if output_writable else 'not writable'}"
                               f" (expected {'writable' if output_should_be_writable else 'not writable'})"))

    state_paths = [app_config.state_dir, app_config.database_path,
                   *(Path(f"{app_config.database_path}{suffix}") for suffix in ("-wal", "-shm"))]
    unusable_paths = [str(path) for path in state_paths if path.exists() and not os.access(path, os.R_OK | os.W_OK)]
    results.append(CheckResult("files", "database and WAL files", FAIL if unusable_paths else PASS,
                               f"not readable+writable: {', '.join(unusable_paths)}" if unusable_paths else
                               "state directory, database, and WAL files are readable and writable"))
    cache_writable = os.access(app_config.source_cache_dir, os.W_OK)
    results.append(CheckResult("files", "source cache", PASS if cache_writable == is_collector else WARN,
                               f"{app_config.source_cache_dir} is {'writable' if cache_writable else 'read-only'} "
                               f"(expected {'writable' if is_collector else 'read-only'})"))
    return results


def _can_read(file_path: Path) -> bool:
    try:
        with file_path.open("rb"):
            return True
    except OSError:
        return False


# --------------------------------------------------------------------------------------------------------------------
# Network egress probes (real connection attempts; a SYN to a provider host the collector uses anyway)
# --------------------------------------------------------------------------------------------------------------------


def _probe_network_egress(app: AppContext, component: str,
                          literal_probe_addresses: list[str]) -> list[CheckResult]:
    """Real connection attempts. `literal_probe_addresses` are public IPs of the probe host, resolved by the caller
    (deploy/check_access_as_service_accounts.sh resolves them as root): a worker that cannot resolve names must
    still be unable to connect to an address it is handed directly."""
    parsed_model_url = urllib.parse.urlparse(app.config.model.base_url)
    model_host, model_port = parsed_model_url.hostname or "127.0.0.1", parsed_model_url.port or 80
    model_reachable, model_detail = _attempt_tcp_connection(model_host, model_port)
    internet_reachable, internet_detail = _attempt_tcp_connection(PUBLIC_HTTPS_PROBE_HOST, 443)
    if component == COMPONENT_WORKER:
        egress_hint = "load deploy/nftables/paperwatch-egress.nft (sudo deploy/apply_egress_rules.sh)"
        results = [
            CheckResult("network", "local model reachable", PASS if model_reachable else WARN,
                        f"{model_host}:{model_port}: {model_detail}",
                        hint=None if model_reachable else "start llama-server, or allow it in the nftables rules"),
            CheckResult("network", "internet blocked (by name)", FAIL if internet_reachable else PASS,
                        f"{PUBLIC_HTTPS_PROBE_HOST}:443: {internet_detail}",
                        hint=egress_hint if internet_reachable else None),
        ]
        if not literal_probe_addresses:
            results.append(CheckResult("network", "internet blocked (by address)", WARN,
                                       "not tested: no --public-probe-address given, so only name resolution was "
                                       "shown to be blocked",
                                       hint="run deploy/check_access_as_service_accounts.sh, which passes them"))
        for probe_address in literal_probe_addresses:
            address_reachable, address_detail = _attempt_tcp_connection(probe_address, 443)
            results.append(CheckResult("network", "internet blocked (by address)",
                                       FAIL if address_reachable else PASS, f"{probe_address}:443: {address_detail}",
                                       hint=egress_hint if address_reachable else None))
        return results
    results = [
        CheckResult("network", "public HTTPS reachable", PASS if internet_reachable else FAIL,
                    f"{PUBLIC_HTTPS_PROBE_HOST}:443: {internet_detail}"),
        CheckResult("network", "local model blocked", WARN if model_reachable else PASS,
                    f"{model_host}:{model_port}: {model_detail}",
                    hint="the collector never needs the model; load deploy/nftables/paperwatch-egress.nft"
                    if model_reachable else None),
    ]
    return results


def _validate_probe_addresses(raw_addresses: list[str]) -> list[str]:
    """Only public IP literals are probed: this option can never point the check at a private or local service."""
    validated_addresses = []
    for raw_address in raw_addresses:
        if not is_public_internet_address(raw_address):
            raise ConfigurationError(f"--public-probe-address {raw_address!r} is not a public IP address")
        validated_addresses.append(raw_address)
    return validated_addresses


def _attempt_tcp_connection(host: str, port: int) -> tuple[bool, str]:
    """(reachable, description). Refused counts as reachable: the packet got through to the host."""
    try:
        with socket.create_connection((host, port), timeout=NETWORK_PROBE_TIMEOUT_SECONDS):
            return True, "connected"
    except socket.gaierror as resolution_error:
        return False, f"name resolution blocked or failed ({resolution_error.strerror})"
    except TimeoutError:
        return False, "timed out (dropped)"
    except OSError as connection_error:
        if connection_error.errno == errno.ECONNREFUSED:
            return True, "connection refused by the host (not blocked)"
        if connection_error.errno in (errno.EPERM, errno.EACCES):
            return False, "blocked locally (operation not permitted)"
        return False, f"failed: {connection_error.strerror or connection_error}"


# --------------------------------------------------------------------------------------------------------------------
# Model server isolation
# --------------------------------------------------------------------------------------------------------------------


def _check_model_server_isolation(app: AppContext) -> list[CheckResult]:
    server_processes = _find_llama_server_processes()
    if not server_processes:
        return [CheckResult("model-server", "process", WARN, "no llama-server process found on this machine")]
    results: list[CheckResult] = []
    expected_account = app.config.access.model_server_account
    for process_id, owner_name, arguments, environment in server_processes:
        forbidden_flags = sorted({argument.split("=", 1)[0] for argument in arguments
                                  if argument.split("=", 1)[0] in FORBIDDEN_MODEL_SERVER_FLAGS})
        results.append(CheckResult("model-server", f"pid {process_id} flags", FAIL if forbidden_flags else PASS,
                                   f"forbidden flags: {', '.join(forbidden_flags)}" if forbidden_flags else
                                   "no tools, agent, MCP, or download flags"))
        if environment is None:
            results.append(CheckResult("model-server", f"pid {process_id} environment", SKIP,
                                       "not readable from this account (expected when isolated)"))
        else:
            leaked_names = sorted(name for name in environment if name in FORBIDDEN_MODEL_SERVER_ENVIRONMENT_NAMES
                                  or any(hint in name.upper() for hint in PROVIDER_KEY_ENVIRONMENT_HINTS))
            results.append(CheckResult("model-server", f"pid {process_id} environment",
                                       FAIL if leaked_names else PASS,
                                       f"sets {', '.join(leaked_names)}" if leaked_names else
                                       "no tool settings or provider keys in its environment"))
        if "--offline" not in arguments:
            results.append(CheckResult("model-server", f"pid {process_id} offline", WARN,
                                       "started without --offline (runtime downloads not disabled by flag)"))
        if expected_account:
            results.append(CheckResult("model-server", f"pid {process_id} account",
                                       PASS if owner_name == expected_account else FAIL,
                                       f"runs as '{owner_name}', expected '{expected_account}'"))
        else:
            results.append(CheckResult("model-server", f"pid {process_id} account", WARN,
                                       f"runs as '{owner_name}': not isolated in its own account yet "
                                       "(access.model_server_account is empty by owner decision)"))
    return results


def _find_llama_server_processes() -> list[tuple[int, str, list[str], dict[str, str] | None]]:
    """(pid, owner, argv, environment or None) for every running llama-server, read from /proc."""
    found_processes = []
    for process_directory in Path("/proc").glob("[0-9]*"):
        try:
            arguments = [argument.decode("utf-8", "replace") for argument in
                         (process_directory / "cmdline").read_bytes().split(b"\0") if argument]
        except OSError:
            continue
        if not arguments or Path(arguments[0]).name != "llama-server":
            continue
        try:
            owner_name = pwd.getpwuid(process_directory.stat().st_uid).pw_name
        except (KeyError, OSError):
            owner_name = "unknown"
        try:
            environment = dict(entry.decode("utf-8", "replace").partition("=")[::2] for entry in
                               (process_directory / "environ").read_bytes().split(b"\0") if b"=" in entry)
        except OSError:
            environment = None
        found_processes.append((int(process_directory.name), owner_name, arguments, environment))
    return found_processes


# --------------------------------------------------------------------------------------------------------------------
# Extractor sandbox (collector only)
# --------------------------------------------------------------------------------------------------------------------


def _check_extractor_sandbox(app: AppContext) -> list[CheckResult]:
    extractor_settings = app.config.extractor
    runtime_path = shutil.which(extractor_settings.container_runtime)
    if runtime_path is None:
        return [CheckResult("extractor", "container runtime", FAIL,
                            f"'{extractor_settings.container_runtime}' is not installed for this account",
                            hint="sudo apt install podman")]
    results = [CheckResult("extractor", "container runtime", PASS, runtime_path)]
    if not extractor_settings.image:
        results.append(CheckResult("extractor", "pinned image", FAIL, "extractor.image is not set",
                                   hint="run deploy/build_extractor_image.sh and put the printed id in config.toml"))
        return results
    try:
        image_present = subprocess.run([extractor_settings.container_runtime, "image", "exists",
                                        extractor_settings.image], stdin=subprocess.DEVNULL, capture_output=True,
                                       timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as runtime_problem:
        return results + [CheckResult("extractor", "pinned image", FAIL, f"cannot query the image store: "
                                                                         f"{runtime_problem}")]
    results.append(CheckResult("extractor", "pinned image", PASS if image_present else FAIL,
                               f"{extractor_settings.image} {'is present' if image_present else 'is NOT present'} "
                               "in this account's image store (never pulled at run time)"))
    if not image_present:
        return results
    smoke_test_file = store_source_bytes(app.config.source_cache_dir, EXTRACTOR_SMOKE_TEST_EDITION_ID,
                                         build_minimal_pdf_bytes(EXTRACTOR_SMOKE_TEST_MARKER), "application/pdf")
    try:
        extracted_text = extract_text_in_sandbox(app.config.source_cache_dir, smoke_test_file.file_path,
                                                 extractor_settings)
        marker_found = EXTRACTOR_SMOKE_TEST_MARKER in extracted_text.text
        results.append(CheckResult("extractor", "sandboxed extraction", PASS if marker_found else FAIL,
                                   f"generated PDF extracted in {extracted_text.duration_seconds:.1f} s with "
                                   "--network=none, read-only root, one read-only input"
                                   if marker_found else "extractor ran but returned unexpected text"))
    except PaperwatchError as extraction_failure:
        results.append(CheckResult("extractor", "sandboxed extraction", FAIL, str(extraction_failure),
                                   extraction_failure))
    finally:
        smoke_test_file.file_path.unlink(missing_ok=True)
        _remove_empty_directory(smoke_test_file.file_path.parent)
    return results


def _remove_empty_directory(directory_path: Path) -> None:
    try:
        directory_path.rmdir()
    except OSError:
        pass


def build_minimal_pdf_bytes(page_text: str) -> bytes:
    """A valid one-page PDF showing `page_text` (ASCII), built by hand so checks need no PDF library."""
    escaped_text = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    pdf_objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n" + content_stream +
        b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf_bytes = bytearray(b"%PDF-1.4\n")
    object_offsets = []
    for object_number, object_body in enumerate(pdf_objects, start=1):
        object_offsets.append(len(pdf_bytes))
        pdf_bytes += f"{object_number} 0 obj\n".encode("ascii") + object_body + b"\nendobj\n"
    cross_reference_offset = len(pdf_bytes)
    pdf_bytes += f"xref\n0 {len(pdf_objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for object_offset in object_offsets:
        pdf_bytes += f"{object_offset:010d} 00000 n \n".encode("ascii")
    pdf_bytes += (f"trailer\n<< /Size {len(pdf_objects) + 1} /Root 1 0 R >>\nstartxref\n"
                  f"{cross_reference_offset}\n%%EOF\n").encode("ascii")
    return bytes(pdf_bytes)
