"""What a model role is, and the single code path that runs one.

A role is plain data: a name, a prompt version, a system prompt, an output JSON Schema, a function that renders the
user message from the role input, and a host-side semantic check (e.g. "only topic ids from the input"). The model
receives plain data and has zero tools.

`run_role_once` is the only place that sends inference requests. It
  1. takes the shared inference lock (one request at a time across roles and processes),
  2. sends one fresh-context request with the role's hard time and output limits,
  3. parses the final answer (never the reasoning) as JSON and validates schema + semantics,
  4. logs the call (role, prompt version, model identity, server build, outcome) to `model_calls`.
Persisting the validated output is the caller's job, inside its own lease-checked transaction.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from paperwatch.clock import utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.errors import ModelOutputInvalid, PaperwatchError
from paperwatch.identifiers import MODEL_CALL_ID_PREFIX, new_internal_id
from paperwatch.llm.inference_lock import hold_inference_lock
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, record_service_outage, record_service_success
from paperwatch.llm.types import ModelClient, StructuredChatRequest, StructuredChatResponse

MAX_REPORTED_SCHEMA_ERRORS = 3

RoleInput = Mapping[str, Any]


def _no_semantic_problems(validated_output: dict[str, Any], role_input: RoleInput) -> list[str]:
    return []


@dataclass(frozen=True)
class RoleDefinition:
    role_name: str
    prompt_version: str  # bump whenever the prompt or schema changes; stored with every output
    system_prompt: str
    output_schema: dict[str, Any]
    render_user_message: Callable[[RoleInput], str]
    find_semantic_problems: Callable[[dict[str, Any], RoleInput], list[str]] = field(default=_no_semantic_problems)

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(self.output_schema)  # fail at import time on a malformed schema


@dataclass(frozen=True)
class RoleRunResult:
    validated_output: dict[str, Any]
    model_response: StructuredChatResponse
    role_name: str
    prompt_version: str


def run_role_once(*, role_definition: RoleDefinition, role_input: RoleInput, app_config: AppConfig,
                  model_client: ModelClient, connection: sqlite3.Connection,
                  job_id: str | None = None) -> RoleRunResult:
    role_limits = app_config.limits_for_role(role_definition.role_name)
    chat_request = StructuredChatRequest(
        role_name=role_definition.role_name,
        system_prompt=role_definition.system_prompt,
        user_message=role_definition.render_user_message(role_input),
        output_schema_name=f"{role_definition.role_name}_output",
        output_schema=role_definition.output_schema,
        max_output_tokens=role_limits.max_output_tokens,
        timeout_seconds=role_limits.timeout_seconds,
        reasoning_effort=role_limits.reasoning_effort,
    )

    call_started_at = utc_now_timestamp()
    call_started_monotonic = time.monotonic()
    model_response: StructuredChatResponse | None = None
    try:
        with hold_inference_lock(app_config.inference_lock_path, app_config.model.inference_lock_wait_seconds):
            model_response = model_client.request_structured_completion(chat_request)
        validated_output = parse_and_validate_role_output(role_definition, model_response.final_answer_text,
                                                          role_input)
    except PaperwatchError as call_error:
        _log_model_call(connection, role_definition, job_id, call_started_at,
                        time.monotonic() - call_started_monotonic, model_response,
                        outcome=call_error.model_call_outcome, error_message=str(call_error))
        if call_error.is_service_outage:
            record_service_outage(connection, MODEL_SERVICE_NAME, str(call_error), app_config.model)
        raise

    _log_model_call(connection, role_definition, job_id, call_started_at, time.monotonic() - call_started_monotonic,
                    model_response, outcome="ok", error_message=None)
    record_service_success(connection, MODEL_SERVICE_NAME)
    return RoleRunResult(validated_output=validated_output, model_response=model_response,
                         role_name=role_definition.role_name, prompt_version=role_definition.prompt_version)


def parse_and_validate_role_output(role_definition: RoleDefinition, final_answer_text: str,
                                   role_input: RoleInput) -> dict[str, Any]:
    role_name = role_definition.role_name
    answer_text = _strip_markdown_code_fence(final_answer_text)
    if not answer_text:
        raise ModelOutputInvalid(f"role '{role_name}' returned an empty final answer")
    try:
        parsed_answer = json.loads(answer_text)
    except json.JSONDecodeError as decode_error:
        raise ModelOutputInvalid(f"role '{role_name}' answer is not valid JSON: {decode_error}") from decode_error

    schema_validator = Draft202012Validator(role_definition.output_schema)
    schema_violations = sorted(schema_validator.iter_errors(parsed_answer), key=lambda violation: list(violation.path))
    if schema_violations:
        readable_violations = "; ".join(
            f"{'/'.join(map(str, violation.path)) or '<root>'}: {violation.message}"
            for violation in schema_violations[:MAX_REPORTED_SCHEMA_ERRORS])
        raise ModelOutputInvalid(f"role '{role_name}' answer violates its schema: {readable_violations}")

    semantic_problems = role_definition.find_semantic_problems(parsed_answer, role_input)
    if semantic_problems:
        raise ModelOutputInvalid(f"role '{role_name}' answer failed host checks: {'; '.join(semantic_problems)}")
    return parsed_answer


def _strip_markdown_code_fence(answer_text: str) -> str:
    stripped_text = answer_text.strip()
    if stripped_text.startswith("```") and stripped_text.endswith("```"):
        stripped_text = stripped_text.removeprefix("```json").removeprefix("```").removesuffix("```")
    return stripped_text.strip()


def _log_model_call(connection: sqlite3.Connection, role_definition: RoleDefinition, job_id: str | None,
                    started_at: str, duration_seconds: float, model_response: StructuredChatResponse | None, *,
                    outcome: str, error_message: str | None) -> None:
    connection.execute(
        """INSERT INTO model_calls (call_id, role_name, prompt_version, job_id, model_identity, server_build,
               started_at, duration_seconds, outcome, prompt_tokens, completion_tokens, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_internal_id(MODEL_CALL_ID_PREFIX), role_definition.role_name, role_definition.prompt_version, job_id,
         model_response.model_identity if model_response else None,
         model_response.server_build if model_response else None,
         started_at, round(duration_seconds, 3), outcome,
         model_response.prompt_tokens if model_response else None,
         model_response.completion_tokens if model_response else None,
         error_message))
