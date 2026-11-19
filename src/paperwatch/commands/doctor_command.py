"""`paperwatch doctor`: check every component the unattended service depends on, and name what is broken.

Model checks (in order; later ones are skipped once the server is unreachable):
  1. /health answers,
  2. the configured alias is served, and the server build and model file are recorded,
  3. the per-slot context window covers model.required_context_tokens,
  4. configured reasoning levels are supported by the chat template,
  5. a short schema-constrained /v1/chat/completions request succeeds, with reasoning returned separately
     from the final answer.
Plan part 3: the server must expose no built-in tools, agent mode, or MCP proxy (checked over HTTP here; the
process flags and account are checked by `paperwatch access-check`).
"""

from __future__ import annotations

import argparse
import os

from paperwatch.app_context import AppContext
from paperwatch.commands.check_results import FAIL, PASS, SKIP, WARN, CheckResult, print_check_results
from paperwatch.config import REASONING_EFFORT_SERVER_DEFAULT
from paperwatch.db import CURRENT_SCHEMA_VERSION, read_schema_version
from paperwatch.errors import PaperwatchError
from paperwatch.llm.types import ModelServerDescription
from paperwatch.roles.base import run_role_once
from paperwatch.roles.healthcheck import HEALTHCHECK_ROLE


def run_doctor_command(app: AppContext, arguments: argparse.Namespace) -> int:
    check_results = [*_check_configuration_and_paths(app), *_check_database(app), *_check_model_server(app)]
    return print_check_results(check_results, "doctor")


def _check_configuration_and_paths(app: AppContext) -> list[CheckResult]:
    app_config = app.config
    results = [CheckResult("config", "load", PASS, f"{app_config.home_dir / 'config.toml'} is valid")]
    results.append(CheckResult("config", "topics", PASS if app_config.topics else WARN,
                               f"{len(app_config.topics)} topic(s) configured" if app_config.topics else
                               "no topics configured; add [[topics]] to topics.toml"))
    if app_config.output_dir.resolve() in app_config.state_dir.resolve().parents:
        results.append(CheckResult("paths", "state location", WARN,
                                   "state_dir is inside output_dir; keep the database out of synced notes"))
    for directory_label, directory_path in (("state_dir", app_config.state_dir),
                                            ("output_dir", app_config.output_dir)):
        is_writable = directory_path.is_dir() and os.access(directory_path, os.W_OK)
        results.append(CheckResult("paths", directory_label, PASS if is_writable else FAIL,
                                   f"{directory_path} {'is writable' if is_writable else 'is missing or not writable'}"))
    return results


def _check_database(app: AppContext) -> list[CheckResult]:
    journal_mode = app.connection.execute("PRAGMA journal_mode").fetchone()[0]
    schema_version = read_schema_version(app.connection)
    return [
        CheckResult("database", "schema", PASS if schema_version == CURRENT_SCHEMA_VERSION else FAIL,
                    f"version {schema_version} (expected {CURRENT_SCHEMA_VERSION})"),
        CheckResult("database", "journal mode", PASS if journal_mode == "wal" else WARN, journal_mode),
    ]


def _check_model_server(app: AppContext) -> list[CheckResult]:
    component = "model-server"
    model_settings = app.config.model
    results: list[CheckResult] = []

    try:
        app.model_client.check_health()
    except PaperwatchError as health_failure:
        results.append(CheckResult(component, "health", FAIL, str(health_failure), health_failure))
        skipped_detail = "skipped because the server is unreachable"
        for skipped_check_name in ("model alias", "tools and MCP", "context budget", "structured request"):
            results.append(CheckResult(component, skipped_check_name, SKIP, skipped_detail))
        return results
    results.append(CheckResult(component, "health", PASS,
                               f"{model_settings.backend} backend at {model_settings.base_url}"))

    try:
        server_description = app.model_client.describe_server()
    except PaperwatchError as describe_failure:
        results.append(CheckResult(component, "model alias", FAIL, str(describe_failure), describe_failure))
        return results
    alias_is_served = model_settings.model_alias in server_description.served_model_names
    results.append(CheckResult(
        component, "model alias", PASS if alias_is_served else FAIL,
        f"'{model_settings.model_alias}' {'served' if alias_is_served else 'NOT served'} "
        f"(served: {', '.join(server_description.served_model_names) or 'none'}); model file "
        f"{server_description.model_file_name}; build {server_description.server_build}"))

    results.append(_check_model_server_exposes_no_tools(server_description))

    context_tokens = server_description.context_tokens_per_slot
    if context_tokens is None:
        results.append(CheckResult(component, "context budget", WARN, "server did not report n_ctx"))
    else:
        context_is_sufficient = context_tokens >= model_settings.required_context_tokens
        results.append(CheckResult(
            component, "context budget", PASS if context_is_sufficient else FAIL,
            f"{context_tokens} tokens per slot across {server_description.parallel_slots} slot(s); "
            f"required {model_settings.required_context_tokens}"))

    supported_levels = server_description.supported_reasoning_effort_levels
    unsupported_role_levels = [f"{role_name}={limits.reasoning_effort}"
                               for role_name, limits in app.config.role_limits.items()
                               if limits.reasoning_effort != REASONING_EFFORT_SERVER_DEFAULT
                               and supported_levels and limits.reasoning_effort not in supported_levels]
    results.append(CheckResult(
        component, "reasoning levels", WARN if unsupported_role_levels else PASS,
        f"unsupported by the chat template ({', '.join(supported_levels)}): {', '.join(unsupported_role_levels)}"
        if unsupported_role_levels else f"server supports {', '.join(supported_levels) or 'unspecified levels'}"))

    try:
        role_result = run_role_once(role_definition=HEALTHCHECK_ROLE, role_input={}, app_config=app.config,
                                    model_client=app.model_client, connection=app.connection)
    except PaperwatchError as request_failure:
        results.append(CheckResult(request_failure.failed_component, "structured request", FAIL,
                                   str(request_failure), request_failure))
        return results
    model_response = role_result.model_response
    results.append(CheckResult(
        component, "structured request", PASS,
        f"schema-constrained answer validated in {model_response.duration_seconds:.1f} s "
        f"({model_response.completion_tokens} completion tokens) from {model_response.model_identity}"))
    results.append(CheckResult(
        component, "reasoning output", PASS if model_response.reasoning_text else WARN,
        f"reasoning returned separately ({len(model_response.reasoning_text)} characters) and excluded from "
        "the parsed answer" if model_response.reasoning_text else
        "no reasoning text returned; is the server running with --reasoning on?"))
    return results


def _check_model_server_exposes_no_tools(server_description: ModelServerDescription) -> CheckResult:
    """The model must have zero tools: no llama-server built-in tools (--tools/--agent) and no MCP proxy."""
    enabled_features = [label for label, is_enabled in (("built-in tools", server_description.server_tools_enabled),
                                                         ("MCP proxy", server_description.mcp_proxy_enabled))
                        if is_enabled]
    if enabled_features:
        return CheckResult("model-server", "tools and MCP", FAIL, f"server exposes {', '.join(enabled_features)}",
                           hint="restart llama-server without --tools, --agent, --mcp-servers-*, or "
                                "--ui-mcp-proxy; paperwatch requires a model with zero tools")
    unknown_features = [label for label, is_enabled in (("built-in tools", server_description.server_tools_enabled),
                                                        ("MCP proxy", server_description.mcp_proxy_enabled))
                        if is_enabled is None]
    if unknown_features:
        return CheckResult("model-server", "tools and MCP", WARN,
                           f"server did not report: {', '.join(unknown_features)}")
    return CheckResult("model-server", "tools and MCP", PASS, "no built-in tools, agent mode, or MCP proxy")
