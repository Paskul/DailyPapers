"""Local model access: the client contract, the llama.cpp client, a fake for tests, and outage handling."""

from __future__ import annotations

from paperwatch.config import ModelSettings
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.llama_cpp_client import LlamaCppModelClient
from paperwatch.llm.types import ModelClient


def build_model_client(model_settings: ModelSettings) -> ModelClient:
    if model_settings.backend == "fake":
        # Imported here to avoid a cycle: the healthcheck role imports from paperwatch.llm.
        from paperwatch.roles.healthcheck import HEALTHCHECK_ROLE, HEALTHCHECK_EXPECTED_FINAL_ANSWER
        return FakeModelClient({HEALTHCHECK_ROLE.role_name: HEALTHCHECK_EXPECTED_FINAL_ANSWER})
    return LlamaCppModelClient(model_settings)
