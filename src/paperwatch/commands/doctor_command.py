"""`paperwatch doctor`: check every component the unattended service depends on, and name what is broken.

Model checks (in order; later ones are skipped once the server is unreachable):
  1. /health answers,
  2. the configured alias is served, and the server build and model file are recorded,
  3. the per-slot context window covers model.required_context_tokens,
  4. configured reasoning levels are supported by the chat template,
  5. a short schema-constrained /v1/chat/completions request succeeds, with reasoning returned separately
     from the final answer.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from paperwatch.app_context import AppContext
from paperwatch.config import REASONING_EFFORT_SERVER_DEFAULT
from paperwatch.db import CURRENT_SCHEMA_VERSION, read_schema_version
from paperwatch.errors import PaperwatchError
from paperwatch.roles.base import run_role_once
from paperwatch.roles.healthcheck import HEALTHCHECK_ROLE

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass(frozen=True)
class DoctorCheckResult:
    component: str
    check_name: str
    verdict: str
    detail: str
    failure: PaperwatchError | None = None


def run_doctor_command(app: AppContext, arguments: argparse.Namespace) -> int:
    check_results = [*_check_configuration_and_paths(app), *_check_database(app), *_check_model_server(app)]

    component_column_width = max(len(result.component) for result in check_results)
    for result in check_results:
        print(f"{result.verdict:4}  {result.component:<{component_column_width}}  {result.check_name}: "
              f"{result.detail}")
        if result.failure is not None and result.failure.operator_hint:
            print(f"{'':4}  {'':<{component_column_width}}  hint: {result.failure.operator_hint}")

    failed_results = [result for result in check_results if result.verdict == FAIL]
    if not failed_results:
        print("\ndoctor: all required checks passed")
        return 0
    print(f"\ndoctor: {len(failed_results)} check(s) FAILED in: "
          f"{', '.join(sorted({result.component for result in failed_results}))}")
    first_failure = next((result.failure for result in failed_results if result.failure is not None), None)
    return first_failure.process_exit_code if first_failure else 1


def _check_configuration_and_paths(app: AppContext) -> list[DoctorCheckResult]:
    app_config = app.config
    results = [DoctorCheckResult("config", "load", PASS, f"{app_config.home_dir / 'config.toml'} is valid")]
    results.append(DoctorCheckResult("config", "topics", PASS if app_config.topics else WARN,
                                     f"{len(app_config.topics)} topic(s) configured" if app_config.topics else
                                     "no topics configured; add [[topics]] to topics.toml"))
    if app_config.output_dir.resolve() in app_config.state_dir.resolve().parents:
        results.append(DoctorCheckResult("paths", "state location", WARN,
                                         "state_dir is inside output_dir; keep the database out of synced notes"))
    for directory_label, directory_path in (("state_dir", app_config.state_dir),
                                            ("output_dir", app_config.output_dir)):
        is_writable = directory_path.is_dir() and os.access(directory_path, os.W_OK)
        results.append(DoctorCheckResult("paths", directory_label, PASS if is_writable else FAIL,
                                         f"{directory_path} {'is writable' if is_writable else 'is missing or not writable'}"))
    return results


def _check_database(app: AppContext) -> list[DoctorCheckResult]:
    journal_mode = app.connection.execute("PRAGMA journal_mode").fetchone()[0]
    schema_version = read_schema_version(app.connection)
    return [
        DoctorCheckResult("database", "schema", PASS if schema_version == CURRENT_SCHEMA_VERSION else FAIL,
                          f"version {schema_version} (expected {CURRENT_SCHEMA_VERSION})"),
        DoctorCheckResult("database", "journal mode", PASS if journal_mode == "wal" else WARN, journal_mode),
    ]


def _check_model_server(app: AppContext) -> list[DoctorCheckResult]:
    component = "model-server"
    model_settings = app.config.model
    results: list[DoctorCheckResult] = []

    try:
        app.model_client.check_health()
    except PaperwatchError as health_failure:
        results.append(DoctorCheckResult(component, "health", FAIL, str(health_failure), health_failure))
        skipped_detail = "skipped because the server is unreachable"
        for skipped_check_name in ("model alias", "context budget", "structured request"):
            results.append(DoctorCheckResult(component, skipped_check_name, SKIP, skipped_detail))
        return results
    results.append(DoctorCheckResult(component, "health", PASS,
                                     f"{model_settings.backend} backend at {model_settings.base_url}"))

    try:
        server_description = app.model_client.describe_server()
    except PaperwatchError as describe_failure:
        results.append(DoctorCheckResult(component, "model alias", FAIL, str(describe_failure), describe_failure))
        return results
    alias_is_served = model_settings.model_alias in server_description.served_model_names
    results.append(DoctorCheckResult(
        component, "model alias", PASS if alias_is_served else FAIL,
        f"'{model_settings.model_alias}' {'served' if alias_is_served else 'NOT served'} "
        f"(served: {', '.join(server_description.served_model_names) or 'none'}); model file "
        f"{server_description.model_file_name}; build {server_description.server_build}"))

    context_tokens = server_description.context_tokens_per_slot
    if context_tokens is None:
        results.append(DoctorCheckResult(component, "context budget", WARN, "server did not report n_ctx"))
    else:
        context_is_sufficient = context_tokens >= model_settings.required_context_tokens
        results.append(DoctorCheckResult(
            component, "context budget", PASS if context_is_sufficient else FAIL,
            f"{context_tokens} tokens per slot across {server_description.parallel_slots} slot(s); "
            f"required {model_settings.required_context_tokens}"))

    supported_levels = server_description.supported_reasoning_effort_levels
    unsupported_role_levels = [f"{role_name}={limits.reasoning_effort}"
                               for role_name, limits in app.config.role_limits.items()
                               if limits.reasoning_effort != REASONING_EFFORT_SERVER_DEFAULT
                               and supported_levels and limits.reasoning_effort not in supported_levels]
    results.append(DoctorCheckResult(
        component, "reasoning levels", WARN if unsupported_role_levels else PASS,
        f"unsupported by the chat template ({', '.join(supported_levels)}): {', '.join(unsupported_role_levels)}"
        if unsupported_role_levels else f"server supports {', '.join(supported_levels) or 'unspecified levels'}"))

    try:
        role_result = run_role_once(role_definition=HEALTHCHECK_ROLE, role_input={}, app_config=app.config,
                                    model_client=app.model_client, connection=app.connection)
    except PaperwatchError as request_failure:
        results.append(DoctorCheckResult(request_failure.failed_component, "structured request", FAIL,
                                         str(request_failure), request_failure))
        return results
    model_response = role_result.model_response
    results.append(DoctorCheckResult(
        component, "structured request", PASS,
        f"schema-constrained answer validated in {model_response.duration_seconds:.1f} s "
        f"({model_response.completion_tokens} completion tokens) from {model_response.model_identity}"))
    results.append(DoctorCheckResult(
        component, "reasoning output", PASS if model_response.reasoning_text else WARN,
        f"reasoning returned separately ({len(model_response.reasoning_text)} characters) and excluded from "
        "the parsed answer" if model_response.reasoning_text else
        "no reasoning text returned; is the server running with --reasoning on?"))
    return results
