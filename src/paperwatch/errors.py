"""Error types. Every error names the component that failed so the CLI can tell the operator where to look.

Two class attributes drive recovery behaviour in the worker:
  * `is_service_outage`: True means the whole service is down (e.g. the llama.cpp server). The worker pauses
    every job that needs that service, releases the current job without charging it an attempt, and backs off.
  * `model_call_outcome`: the label written to the `model_calls` log when the error happened during a model call.
"""

from __future__ import annotations


class PaperwatchError(Exception):
    failed_component: str = "paperwatch"
    process_exit_code: int = 1
    is_service_outage: bool = False
    model_call_outcome: str = "error"

    def __init__(self, message: str, *, operator_hint: str | None = None) -> None:
        super().__init__(message)
        self.operator_hint = operator_hint


class ConfigurationError(PaperwatchError):
    failed_component = "config"
    process_exit_code = 2


class ModelServerUnavailable(PaperwatchError):
    """The llama.cpp server cannot be reached, is still loading, or returned a server-side error."""

    failed_component = "model-server"
    process_exit_code = 3
    is_service_outage = True
    model_call_outcome = "unavailable"


class DatabaseError(PaperwatchError):
    failed_component = "database"
    process_exit_code = 4


class ModelOutputInvalid(PaperwatchError):
    """The model answered, but the answer is not valid JSON, violates the role schema, or fails host checks."""

    failed_component = "model-output"
    process_exit_code = 5
    model_call_outcome = "invalid_output"


class ModelOutputTruncated(ModelOutputInvalid):
    """The role hit its hard output-token limit before finishing (reasoning tokens count toward it)."""

    model_call_outcome = "truncated"


class ModelCallTimedOut(PaperwatchError):
    """The role hit its hard time limit. Charged to the job (a long paper), not treated as a server outage."""

    failed_component = "model-server"
    process_exit_code = 5
    model_call_outcome = "timeout"


class ModelRequestRejected(PaperwatchError):
    """The server refused this specific request (HTTP 4xx), e.g. the prompt exceeds the context window."""

    failed_component = "model-server"
    process_exit_code = 5
    model_call_outcome = "rejected"


class InferenceLockTimeout(PaperwatchError):
    """Another process held the shared one-request-at-a-time inference lock for too long."""

    failed_component = "inference-lock"
    process_exit_code = 6
    model_call_outcome = "lock_timeout"


class SourceInputError(PaperwatchError):
    failed_component = "source"
    process_exit_code = 7


class OutputWriteError(PaperwatchError):
    failed_component = "output"
    process_exit_code = 8


class UnexpectedWorkerError(PaperwatchError):
    """A bug or unknown job kind inside the worker. Charged to the job so it cannot loop forever."""

    failed_component = "worker"
    process_exit_code = 9


def describe_failure_for_operator(failure: PaperwatchError, *, message_override: str | None = None) -> str:
    """One consistent line (plus an optional hint line) naming the failed component."""
    description = f"paperwatch: FAILED [{failure.failed_component}] {message_override or failure}"
    if failure.operator_hint:
        description += f"\n  hint: {failure.operator_hint}"
    return description
