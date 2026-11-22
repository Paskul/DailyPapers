"""ModelClient for the existing local llama.cpp server (OpenAI-compatible /v1 API plus /health and /props).

Only loopback addresses are allowed by config validation. There is deliberately no fallback to any other model:
if this server is unavailable, the caller pauses model work and keeps the input queued.
"""

from __future__ import annotations

import time
from pathlib import PurePath
from typing import Any

import httpx

from paperwatch.config import REASONING_EFFORT_SERVER_DEFAULT, ModelSettings
from paperwatch.errors import (
    ModelCallTimedOut,
    ModelOutputInvalid,
    ModelOutputTruncated,
    ModelRequestRejected,
    ModelServerUnavailable,
)
from paperwatch.llm.reasoning import separate_reasoning_from_final_answer
from paperwatch.llm.types import ModelServerDescription, StructuredChatRequest, StructuredChatResponse

METADATA_REQUEST_TIMEOUT_SECONDS = 10.0
# Request fields that would give the model tools or let it call functions. paperwatch never sends them: the model
# receives plain data and has zero tools (plan part 3). build_chat_completion_payload asserts their absence.
TOOL_REQUEST_FIELDS = frozenset({"tools", "tool_choice", "functions", "function_call", "parallel_tool_calls"})
MAX_ERROR_BODY_CHARACTERS = 300
UNKNOWN_VALUE = "unknown"


class LlamaCppModelClient:
    def __init__(self, model_settings: ModelSettings) -> None:
        self._model_settings = model_settings
        self._openai_api_base_url = model_settings.base_url.rstrip("/")
        # /health and /props live at the server root, outside the /v1 prefix.
        self._server_root_url = self._openai_api_base_url.removesuffix("/v1")
        self._cached_model_file_name: str | None = None

    # ---- ModelClient protocol ----------------------------------------------------------------------------------------

    def check_health(self) -> None:
        health_response = self._send(
            "GET", f"{self._server_root_url}/health", read_timeout_seconds=METADATA_REQUEST_TIMEOUT_SECONDS
        )
        if health_response.status_code != 200:
            raise ModelServerUnavailable(
                f"llama.cpp /health returned HTTP {health_response.status_code} "
                f"(the model may still be loading): {_shorten(health_response.text)}"
            )

    def describe_server(self) -> ModelServerDescription:
        models_payload = self._get_json(f"{self._openai_api_base_url}/models")
        served_model_names: list[str] = []
        for model_entry in models_payload.get("data", []):
            served_model_names.append(model_entry.get("id", ""))
            served_model_names.extend(model_entry.get("aliases", []) or [])

        properties_payload = self._get_json(f"{self._server_root_url}/props")
        # /tools lists built-in server tools when llama-server runs with --tools or --agent; otherwise it refuses.
        tools_response = self._send(
            "GET", f"{self._server_root_url}/tools", read_timeout_seconds=METADATA_REQUEST_TIMEOUT_SECONDS
        )
        generation_defaults = properties_payload.get("default_generation_settings", {}) or {}
        template_capabilities = properties_payload.get("chat_template_caps", {}) or {}
        model_file_name = PurePath(properties_payload.get("model_path", UNKNOWN_VALUE)).name
        self._cached_model_file_name = model_file_name
        return ModelServerDescription(
            served_model_names=tuple(dict.fromkeys(name for name in served_model_names if name)),  # dedupe, keep order
            server_build=properties_payload.get("build_info", UNKNOWN_VALUE),
            model_file_name=model_file_name,
            context_tokens_per_slot=generation_defaults.get("n_ctx"),
            parallel_slots=properties_payload.get("total_slots"),
            supported_reasoning_effort_levels=tuple(template_capabilities.get("reasoning_effort_levels", []) or []),
            mcp_proxy_enabled=properties_payload.get("cors_proxy_enabled"),
            server_tools_enabled=_server_tools_are_listed(tools_response),
        )

    def request_structured_completion(self, request: StructuredChatRequest) -> StructuredChatResponse:
        request_payload = build_chat_completion_payload(request, self._model_settings.model_alias)
        request_started = time.monotonic()
        completion_response = self._send(
            "POST",
            f"{self._openai_api_base_url}/chat/completions",
            read_timeout_seconds=request.timeout_seconds,
            json_body=request_payload,
            role_name_for_timeout_message=request.role_name,
        )
        duration_seconds = time.monotonic() - request_started

        if 400 <= completion_response.status_code < 500:
            raise ModelRequestRejected(
                f"llama.cpp rejected the {request.role_name} request with HTTP "
                f"{completion_response.status_code}: {_shorten(completion_response.text)}"
            )
        if completion_response.status_code != 200:
            raise ModelServerUnavailable(
                f"llama.cpp returned HTTP {completion_response.status_code} for the "
                f"{request.role_name} request: {_shorten(completion_response.text)}"
            )
        try:
            completion_payload = completion_response.json()
            first_choice = completion_payload["choices"][0]
            message = first_choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as parse_error:
            raise ModelOutputInvalid(
                f"llama.cpp response for {request.role_name} has an unexpected shape: {parse_error}"
            ) from parse_error

        final_answer_text, reasoning_text = separate_reasoning_from_final_answer(
            message.get("content"), message.get("reasoning_content")
        )
        finish_reason = first_choice.get("finish_reason") or UNKNOWN_VALUE
        if finish_reason == "length":
            raise ModelOutputTruncated(
                f"role '{request.role_name}' hit its output limit of {request.max_output_tokens} tokens "
                "(reasoning included) before finishing its answer",
                operator_hint=f"raise roles.{request.role_name}.max_output_tokens or lower its reasoning_effort",
            )

        usage = completion_payload.get("usage", {}) or {}
        return StructuredChatResponse(
            final_answer_text=final_answer_text,
            reasoning_text=reasoning_text,
            finish_reason=finish_reason,
            model_identity=self._model_identity(completion_payload.get("model")),
            server_build=completion_payload.get("system_fingerprint") or UNKNOWN_VALUE,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            duration_seconds=duration_seconds,
        )

    # ---- helpers -----------------------------------------------------------------------------------------------------

    def _model_identity(self, reported_model_name: str | None) -> str:
        """Alias plus the GGUF file name, so a swapped model file under the same alias is still recorded."""
        if self._cached_model_file_name is None:
            try:
                self.describe_server()
            except ModelServerUnavailable:
                self._cached_model_file_name = UNKNOWN_VALUE
        model_name = reported_model_name or self._model_settings.model_alias
        return f"{model_name} ({self._cached_model_file_name})"

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self._send("GET", url, read_timeout_seconds=METADATA_REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            raise ModelServerUnavailable(f"GET {url} returned HTTP {response.status_code}: {_shorten(response.text)}")
        try:
            return response.json()
        except ValueError as decode_error:
            raise ModelServerUnavailable(f"GET {url} did not return JSON") from decode_error

    def _send(
        self,
        method: str,
        url: str,
        *,
        read_timeout_seconds: float,
        json_body: dict | None = None,
        role_name_for_timeout_message: str | None = None,
    ) -> httpx.Response:
        timeouts = httpx.Timeout(
            connect=self._model_settings.connect_timeout_seconds,
            read=read_timeout_seconds,
            write=30.0,
            pool=self._model_settings.connect_timeout_seconds,
        )
        # trust_env=False: never route local model traffic through proxy settings from the environment.
        with httpx.Client(timeout=timeouts, follow_redirects=False, trust_env=False) as http_client:
            try:
                return http_client.request(method, url, json=json_body)
            except httpx.ConnectError as connect_error:
                raise ModelServerUnavailable(
                    f"cannot connect to the llama.cpp server at {self._server_root_url}: {connect_error}",
                    operator_hint="start the llama.cpp server (or check model.base_url); queued papers are kept",
                ) from connect_error
            except httpx.ConnectTimeout as connect_timeout:
                raise ModelServerUnavailable(f"timed out connecting to {self._server_root_url}") from connect_timeout
            except httpx.ReadTimeout as read_timeout:
                if role_name_for_timeout_message is None:
                    raise ModelServerUnavailable(f"{method} {url} timed out") from read_timeout
                raise ModelCallTimedOut(
                    f"role '{role_name_for_timeout_message}' exceeded its {read_timeout_seconds:.0f} s time limit",
                    operator_hint=f"raise roles.{role_name_for_timeout_message}.timeout_seconds if this repeats",
                ) from read_timeout
            except httpx.HTTPError as other_http_error:
                raise ModelServerUnavailable(f"{method} {url} failed: {other_http_error!r}") from other_http_error


def build_chat_completion_payload(request: StructuredChatRequest, model_alias: str) -> dict[str, Any]:
    """The exact JSON body sent to /v1/chat/completions: two fresh messages, a schema, limits. Never any tools."""
    request_payload: dict[str, Any] = {
        "model": model_alias,
        "messages": [  # a fresh context for every job: no conversation history is ever carried over
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_message},
        ],
        "max_tokens": request.max_output_tokens,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": request.output_schema_name, "strict": True, "schema": request.output_schema},
        },
    }
    if request.reasoning_effort != REASONING_EFFORT_SERVER_DEFAULT:
        request_payload["reasoning_effort"] = request.reasoning_effort
    if TOOL_REQUEST_FIELDS & request_payload.keys():  # explicit check: survives `python -O`
        raise RuntimeError("model requests must never carry tools")
    return request_payload


def _server_tools_are_listed(tools_response: httpx.Response) -> bool | None:
    """True if /tools lists at least one tool, False if refused/absent/empty, None if the answer is unreadable."""
    if tools_response.status_code in (401, 403, 404, 405, 501):
        return False
    if tools_response.status_code != 200:
        return None
    try:
        tools_payload = tools_response.json()
    except ValueError:
        return None
    listed_tools = (
        tools_payload.get("data", tools_payload.get("tools", [])) if isinstance(tools_payload, dict) else tools_payload
    )
    return bool(listed_tools)


def _shorten(text: str) -> str:
    collapsed_text = " ".join(text.split())
    return collapsed_text[:MAX_ERROR_BODY_CHARACTERS] + ("…" if len(collapsed_text) > MAX_ERROR_BODY_CHARACTERS else "")
