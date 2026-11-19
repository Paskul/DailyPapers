"""Deterministic stand-in for the llama.cpp server, used by tests and the offline vertical slice.

By default it answers every role with the smallest JSON instance that satisfies the role's output schema, so new
roles work with the fake without extra code. Tests can pass canned final answers per role (including invalid ones)
to exercise validation and failure paths through the real role pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Any

from paperwatch.llm.types import ModelServerDescription, StructuredChatRequest, StructuredChatResponse

FAKE_MODEL_IDENTITY = "fake-model (no weights)"
FAKE_SERVER_BUILD = "fake-build"
FAKE_CONTEXT_TOKENS_PER_SLOT = 262144


class FakeModelClient:
    def __init__(self, canned_final_answers_by_role: dict[str, str] | None = None, *,
                 response_delay_seconds: float = 0.0) -> None:
        self._canned_final_answers_by_role = dict(canned_final_answers_by_role or {})
        self._response_delay_seconds = response_delay_seconds  # simulates a slow model in crash/lease tests
        self.received_requests: list[StructuredChatRequest] = []  # lets tests count inference calls

    def check_health(self) -> None:
        return None

    def describe_server(self) -> ModelServerDescription:
        return ModelServerDescription(
            served_model_names=("qwen-local",), server_build=FAKE_SERVER_BUILD, model_file_name="fake.gguf",
            context_tokens_per_slot=FAKE_CONTEXT_TOKENS_PER_SLOT, parallel_slots=1,
            supported_reasoning_effort_levels=("low", "medium", "xhigh"), mcp_proxy_enabled=False,
            server_tools_enabled=False)

    def request_structured_completion(self, request: StructuredChatRequest) -> StructuredChatResponse:
        self.received_requests.append(request)
        if self._response_delay_seconds > 0:
            time.sleep(self._response_delay_seconds)
        if request.role_name in self._canned_final_answers_by_role:
            final_answer_text = self._canned_final_answers_by_role[request.role_name]
        else:
            final_answer_text = json.dumps(build_minimal_instance_for_schema(request.output_schema))
        return StructuredChatResponse(
            final_answer_text=final_answer_text, reasoning_text="(fake reasoning)", finish_reason="stop",
            model_identity=FAKE_MODEL_IDENTITY, server_build=FAKE_SERVER_BUILD,
            prompt_tokens=0, completion_tokens=0, duration_seconds=0.0)


def build_minimal_instance_for_schema(schema: dict[str, Any]) -> Any:
    """Smallest value satisfying the subset of JSON Schema the role schemas use."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return schema["enum"][0]
    schema_type = schema.get("type")
    if schema_type == "object":
        return {property_name: build_minimal_instance_for_schema(schema["properties"][property_name])
                for property_name in schema.get("required", [])}
    if schema_type == "array":
        return [build_minimal_instance_for_schema(schema.get("items", {}))
                for _ in range(schema.get("minItems", 0))]
    if schema_type == "string":
        minimum_length = schema.get("minLength", 0)
        return ("fake text " * (1 + minimum_length // 10))[:max(minimum_length, 9)]
    if schema_type == "integer":
        return schema.get("minimum", 0)
    if schema_type == "number":
        return float(schema.get("minimum", 0))
    if schema_type == "boolean":
        return False
    return None
