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


class SourceRecordNotFound(SourceInputError):
    """The provider answered 404/410 for this specific record. Paper-specific, so it is charged to the job."""


class SourceResponseRejected(SourceInputError):
    """A provider response broke a bound: too large, too slow, wrong content type, or bad compression."""


class ProviderUnavailable(PaperwatchError):
    """A whole provider is down, rate-limiting us, or refusing our credentials. Collection from it pauses."""

    failed_component = "source"
    process_exit_code = 7
    is_service_outage = True

    def __init__(self, message: str, *, provider_name: str, retry_after_seconds: float | None = None,
                 operator_hint: str | None = None) -> None:
        super().__init__(message, operator_hint=operator_hint)
        self.provider_name = provider_name
        self.retry_after_seconds = retry_after_seconds  # the provider's own Retry-After, honoured as a minimum pause


class SourceAccessDenied(PaperwatchError):
    """A request was refused by paperwatch's own access rules (plan part 3), before or during the network call.

    Examples: a host, path, or parameter outside the read-only provider rules; a redirect to an unlisted host; a
    connection target in a private, loopback, link-local, or reserved range; an id that is not a known source record.
    These are never retried automatically: the same request would be refused again.
    """

    failed_component = "source-access"
    process_exit_code = 10


class SourceCredentialRequired(SourceAccessDenied):
    """The provider serves this request only with a credential the operator has not configured (e.g. OpenAlex
    pages after the first). Refused locally, before anything is sent; not an outage, so nothing is paused."""


class SourceRequestBudgetExhausted(SourceAccessDenied):
    """The provider's daily request budget is used up. Collection from it resumes the next UTC day."""


class ExtractionError(PaperwatchError):
    """The sandboxed PDF/XML extractor failed, returned too much output, or is not set up."""

    failed_component = "extractor"
    process_exit_code = 11


class ExtractionTimedOut(ExtractionError):
    """The extractor exceeded its time limit; the container was killed. Charged to that one source file."""


class ScreeningUnavailable(PaperwatchError):
    """The pinned sentence-embedding model is missing, does not match its pinned hashes, or cannot run. Candidates
    wait unscored (nothing is promoted on partial signals) until the model is installed."""

    failed_component = "screening"
    process_exit_code = 12


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
