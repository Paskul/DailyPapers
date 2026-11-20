"""What a model role is, and the single code path that runs one.

A role is plain data: a name, a prompt version, a system prompt, an output JSON Schema, a function that renders the
user message from the role input, and a host-side semantic check (e.g. "only topic ids from the input"). The model
receives plain data and has zero tools.

`run_role_once` is the only place that sends inference requests. It
  1. refuses a prompt that would not leave the role's output limit (reasoning + final answer) free inside
     model.required_context_tokens (plan part 7: "reserve context space for reasoning and the final answer"),
  2. waits while another holder (the daily report, plan part 9) has reserved the next inference slot, then takes
     the shared inference lock (one request at a time across roles and processes),
  3. sends one fresh-context request with the role's hard time and output limits,
  4. parses the final answer (never the reasoning) as JSON and validates schema + semantics,
  5. logs the call (role, prompt version, model identity, server build, outcome) to `model_calls`.
`run_role_with_one_repair` adds the plan's single bounded repair attempt: an invalid answer is sent back once, in a
new fresh context, with the problems listed; a second invalid answer ends the job (ModelOutputRejectedAfterRepair).
Persisting the validated output is the caller's job, inside its own lease-checked transaction.
A caller with an overall deadline (the report writer) passes `deadline_monotonic`: the lock wait and the call's time
limit shrink to the time left, and no call starts after the deadline (ModelCallDeadlineExceeded).
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
from paperwatch.config import AppConfig, RoleLimits
from paperwatch.errors import (ModelCallDeadlineExceeded, ModelOutputInvalid, ModelOutputRejectedAfterRepair,
                               ModelPromptExceedsContextBudget, PaperwatchError)
from paperwatch.identifiers import MODEL_CALL_ID_PREFIX, new_internal_id
from paperwatch.llm.inference_lock import hold_inference_lock
from paperwatch.llm.inference_reservation import wait_while_inference_is_reserved
from paperwatch.llm.service_guard import MODEL_SERVICE_NAME, record_service_outage, record_service_success
from paperwatch.llm.types import ModelClient, StructuredChatRequest, StructuredChatResponse

MAX_REPORTED_SCHEMA_ERRORS = 3
MAX_REPORTED_SEMANTIC_PROBLEMS = 8
# Conservative characters-per-token estimate for English research text with JSON around it (Qwen tokenizers average
# about 4). Used only to refuse prompts that clearly cannot fit; the server enforces the real limit.
CHARACTERS_PER_TOKEN_ESTIMATE = 3.5
CHAT_TEMPLATE_OVERHEAD_TOKENS = 200
MAXIMUM_REJECTED_ANSWER_CHARACTERS_IN_REPAIR = 12_000
REPAIR_PROMPT_VERSION_SUFFIX = "+repair"

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


def estimate_prompt_tokens(*texts: str) -> int:
    return CHAT_TEMPLATE_OVERHEAD_TOKENS + int(sum(len(text) for text in texts) / CHARACTERS_PER_TOKEN_ESTIMATE) + 1


def run_role_with_one_repair(*, role_definition: RoleDefinition, role_input: RoleInput, app_config: AppConfig,
                             model_client: ModelClient, connection: sqlite3.Connection,
                             job_id: str | None = None, deadline_monotonic: float | None = None,
                             holds_inference_reservation: bool = False) -> RoleRunResult:
    """Run the role; if its answer is invalid, ask once more with the problems listed. Outages, timeouts and lock
    waits propagate unchanged (they are retried by the worker). Two invalid answers end the job with both reasons."""
    call_options = dict(role_definition=role_definition, role_input=role_input, app_config=app_config,
                        model_client=model_client, connection=connection, job_id=job_id,
                        deadline_monotonic=deadline_monotonic,
                        holds_inference_reservation=holds_inference_reservation)
    try:
        return run_role_once(**call_options)
    except ModelOutputInvalid as first_rejection:
        try:
            return run_role_once(**call_options, rejected_first_answer=first_rejection)
        except ModelOutputInvalid as second_rejection:
            raise ModelOutputRejectedAfterRepair(
                f"role '{role_definition.role_name}' answer still invalid after one repair attempt: "
                f"{second_rejection} (first answer: {first_rejection})",
                rejected_answer_text=second_rejection.rejected_answer_text,
                problems=second_rejection.problems) from second_rejection


def run_role_once(*, role_definition: RoleDefinition, role_input: RoleInput, app_config: AppConfig,
                  model_client: ModelClient, connection: sqlite3.Connection,
                  job_id: str | None = None, rejected_first_answer: ModelOutputInvalid | None = None,
                  deadline_monotonic: float | None = None,
                  holds_inference_reservation: bool = False) -> RoleRunResult:
    """One request. With `rejected_first_answer` it is the repair request: the same input plus the rejected answer
    and its problems, in a new fresh context. `holds_inference_reservation` is True only for the holder of the
    reservation (the report writer); every other caller yields to a reservation before taking the lock."""
    role_limits = app_config.limits_for_role(role_definition.role_name)
    user_message = role_definition.render_user_message(role_input)
    logged_prompt_version = role_definition.prompt_version
    if rejected_first_answer is not None:
        user_message += _render_repair_request(rejected_first_answer)
        logged_prompt_version += REPAIR_PROMPT_VERSION_SUFFIX
    estimated_prompt_tokens = estimate_prompt_tokens(role_definition.system_prompt, user_message)
    if estimated_prompt_tokens + role_limits.max_output_tokens > app_config.model.required_context_tokens:
        raise ModelPromptExceedsContextBudget(
            f"role '{role_definition.role_name}': about {estimated_prompt_tokens} prompt tokens plus "
            f"{role_limits.max_output_tokens} reserved for reasoning and the answer exceed "
            f"model.required_context_tokens = {app_config.model.required_context_tokens}",
            operator_hint="lower reading.section_max_characters, or raise model.required_context_tokens if the "
                          "server's per-slot context allows it")
    lock_wait_seconds = float(app_config.model.inference_lock_wait_seconds)
    call_timeout_seconds = float(role_limits.timeout_seconds)

    call_started_at = utc_now_timestamp()
    call_started_monotonic = time.monotonic()
    model_response: StructuredChatResponse | None = None
    try:
        if not holds_inference_reservation:
            wait_while_inference_is_reserved(connection)  # plan part 9: the report gets the next slot
        if deadline_monotonic is not None:
            seconds_left = deadline_monotonic - time.monotonic()
            if seconds_left <= 0:
                raise ModelCallDeadlineExceeded(f"role '{role_definition.role_name}': the caller's deadline passed "
                                                "before the call could start")
            lock_wait_seconds = min(lock_wait_seconds, seconds_left)
            call_timeout_seconds = min(call_timeout_seconds, seconds_left)
        with hold_inference_lock(app_config.inference_lock_path, lock_wait_seconds):
            if deadline_monotonic is not None:  # the wait for the lock used part of the time left
                call_timeout_seconds = min(call_timeout_seconds, deadline_monotonic - time.monotonic())
                if call_timeout_seconds <= 0:
                    raise ModelCallDeadlineExceeded(f"role '{role_definition.role_name}': the caller's deadline "
                                                    "passed while waiting for the inference slot")
            model_response = model_client.request_structured_completion(
                _build_chat_request(role_definition, role_input, user_message, role_limits, call_timeout_seconds))
        validated_output = parse_and_validate_role_output(role_definition, model_response.final_answer_text,
                                                          role_input)
    except PaperwatchError as call_error:
        _log_model_call(connection, role_definition.role_name, logged_prompt_version, job_id, call_started_at,
                        time.monotonic() - call_started_monotonic, model_response,
                        outcome=call_error.model_call_outcome, error_message=str(call_error))
        if call_error.is_service_outage:
            record_service_outage(connection, MODEL_SERVICE_NAME, str(call_error), app_config.model)
        raise

    _log_model_call(connection, role_definition.role_name, logged_prompt_version, job_id, call_started_at,
                    time.monotonic() - call_started_monotonic, model_response, outcome="ok", error_message=None)
    record_service_success(connection, MODEL_SERVICE_NAME)
    return RoleRunResult(validated_output=validated_output, model_response=model_response,
                         role_name=role_definition.role_name, prompt_version=role_definition.prompt_version)


def _build_chat_request(role_definition: RoleDefinition, role_input: RoleInput, user_message: str,
                        role_limits: RoleLimits, call_timeout_seconds: float) -> StructuredChatRequest:
    return StructuredChatRequest(
        role_name=role_definition.role_name,
        system_prompt=role_definition.system_prompt,
        user_message=user_message,
        output_schema_name=f"{role_definition.role_name}_output",
        output_schema=role_definition.output_schema,
        max_output_tokens=role_limits.max_output_tokens,
        timeout_seconds=call_timeout_seconds,
        reasoning_effort=role_limits.reasoning_effort,
        role_input=role_input,
    )


def parse_and_validate_role_output(role_definition: RoleDefinition, final_answer_text: str,
                                   role_input: RoleInput) -> dict[str, Any]:
    role_name = role_definition.role_name
    answer_text = _strip_markdown_code_fence(final_answer_text)
    if not answer_text:
        raise ModelOutputInvalid(f"role '{role_name}' returned an empty final answer", rejected_answer_text="")
    try:
        parsed_answer = json.loads(answer_text)
    except json.JSONDecodeError as decode_error:
        raise ModelOutputInvalid(f"role '{role_name}' answer is not valid JSON: {decode_error}",
                                 rejected_answer_text=answer_text,
                                 problems=(f"not valid JSON: {decode_error}",)) from decode_error

    schema_validator = Draft202012Validator(role_definition.output_schema)
    schema_violations = sorted(schema_validator.iter_errors(parsed_answer), key=lambda violation: list(violation.path))
    if schema_violations:
        readable_violations = "; ".join(
            f"{'/'.join(map(str, violation.path)) or '<root>'}: {violation.message}"
            for violation in schema_violations[:MAX_REPORTED_SCHEMA_ERRORS])
        raise ModelOutputInvalid(f"role '{role_name}' answer violates its schema: {readable_violations}",
                                 rejected_answer_text=answer_text, problems=(readable_violations,))

    semantic_problems = role_definition.find_semantic_problems(parsed_answer, role_input)
    if semantic_problems:
        reported_problems = semantic_problems[:MAX_REPORTED_SEMANTIC_PROBLEMS]
        raise ModelOutputInvalid(f"role '{role_name}' answer failed host checks: {'; '.join(reported_problems)}",
                                 rejected_answer_text=answer_text, problems=tuple(reported_problems))
    return parsed_answer


def _render_repair_request(rejected_answer: ModelOutputInvalid) -> str:
    """Appended to the original user message for the one repair attempt."""
    previous_answer = (rejected_answer.rejected_answer_text or "")[:MAXIMUM_REJECTED_ANSWER_CHARACTERS_IN_REPAIR]
    problem_lines = "\n".join(f"- {problem}" for problem in rejected_answer.problems)
    return (f"\nREPAIR REQUEST: your previous answer to this input was rejected by automatic checks.\n"
            f"PROBLEMS:\n{problem_lines}\n"
            f"PREVIOUS ANSWER (rejected):\n{previous_answer or '(empty or cut off)'}\n"
            "Return a corrected, complete JSON object that fixes every problem. Where the input cannot support a "
            "claim, remove it or mark it as insufficient evidence instead of guessing.\n")


def _strip_markdown_code_fence(answer_text: str) -> str:
    stripped_text = answer_text.strip()
    if stripped_text.startswith("```") and stripped_text.endswith("```"):
        stripped_text = stripped_text.removeprefix("```json").removeprefix("```").removesuffix("```")
    return stripped_text.strip()


def _log_model_call(connection: sqlite3.Connection, role_name: str, prompt_version: str, job_id: str | None,
                    started_at: str, duration_seconds: float, model_response: StructuredChatResponse | None, *,
                    outcome: str, error_message: str | None) -> None:
    connection.execute(
        """INSERT INTO model_calls (call_id, role_name, prompt_version, job_id, model_identity, server_build,
               started_at, duration_seconds, outcome, prompt_tokens, completion_tokens, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_internal_id(MODEL_CALL_ID_PREFIX), role_name, prompt_version, job_id,
         model_response.model_identity if model_response else None,
         model_response.server_build if model_response else None,
         started_at, round(duration_seconds, 3), outcome,
         model_response.prompt_tokens if model_response else None,
         model_response.completion_tokens if model_response else None,
         error_message))
