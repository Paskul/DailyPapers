"""Local model access: the client contract, the llama.cpp client, a fake for tests, and outage handling."""

from __future__ import annotations

import os

from paperwatch.config import ModelSettings
from paperwatch.llm.fake_client import FakeModelClient
from paperwatch.llm.llama_cpp_client import LlamaCppModelClient
from paperwatch.llm.types import ModelClient

# Testing aid for the fake backend only: makes every fake answer take this long, so tests can kill or race workers
# in the middle of a "model call". Ignored by the real llama.cpp backend.
FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE = "PAPERWATCH_FAKE_MODEL_DELAY_SECONDS"


def build_model_client(model_settings: ModelSettings) -> ModelClient:
    if model_settings.backend == "fake":
        # Imported here to avoid a cycle: the healthcheck role imports from paperwatch.llm.
        from paperwatch.roles.healthcheck import HEALTHCHECK_EXPECTED_FINAL_ANSWER, HEALTHCHECK_ROLE

        return FakeModelClient(
            {HEALTHCHECK_ROLE.role_name: HEALTHCHECK_EXPECTED_FINAL_ANSWER},
            response_delay_seconds=float(os.environ.get(FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE, "0")),
        )
    return LlamaCppModelClient(model_settings)
