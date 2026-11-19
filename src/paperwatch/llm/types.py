"""The contract between paperwatch and any model backend (real llama.cpp or the fake used in tests).

A role never talks HTTP. It builds a StructuredChatRequest; a ModelClient turns it into exactly one fresh
two-message conversation (system + user) and returns the final answer separated from the reasoning text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StructuredChatRequest:
    role_name: str
    system_prompt: str
    user_message: str
    output_schema_name: str
    output_schema: dict[str, Any]  # JSON Schema; the server constrains decoding to it
    max_output_tokens: int  # hard limit, reasoning included
    timeout_seconds: float  # hard limit for the whole request
    reasoning_effort: str  # "server-default" or a server-supported level


@dataclass(frozen=True)
class StructuredChatResponse:
    final_answer_text: str  # what the model answered, with any reasoning removed
    reasoning_text: str  # kept for debugging only; never parsed as the answer
    finish_reason: str
    model_identity: str  # e.g. "qwen-local (Qwen3.8-27B-Q8_0.gguf)"
    server_build: str  # e.g. "b12373-ded522f83"
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_seconds: float


@dataclass(frozen=True)
class ModelServerDescription:
    served_model_names: tuple[str, ...]  # ids and aliases from /v1/models
    server_build: str
    model_file_name: str
    context_tokens_per_slot: int | None
    parallel_slots: int | None
    supported_reasoning_effort_levels: tuple[str, ...]
    # Plan part 3: llama-server must run without agent mode, built-in tools, MCP servers, or the MCP proxy.
    mcp_proxy_enabled: bool | None = None  # /props cors_proxy_enabled; None if the server does not report it
    server_tools_enabled: bool | None = None  # True if /tools lists built-in tools


class ModelClient(Protocol):
    def check_health(self) -> None:
        """Return if the server is ready; raise ModelServerUnavailable otherwise. Cheap, no inference."""

    def describe_server(self) -> ModelServerDescription:
        """Model names, build, and context budget. Raises ModelServerUnavailable."""

    def request_structured_completion(self, request: StructuredChatRequest) -> StructuredChatResponse:
        """One inference request. Raises ModelServerUnavailable, ModelCallTimedOut, ModelRequestRejected,
        or ModelOutputTruncated. JSON parsing and schema validation happen in the role layer."""
