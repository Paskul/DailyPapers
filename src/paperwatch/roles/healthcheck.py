"""Tiny role used by `paperwatch doctor` to prove schema-constrained decoding works with reasoning enabled.

It asks a question with one checkable answer, so a server that ignores the schema or garbles output is caught.
"""

from __future__ import annotations

import json
from typing import Any

from paperwatch.roles.base import RoleDefinition, RoleInput

HEALTHCHECK_FIRST_ADDEND = 17
HEALTHCHECK_SECOND_ADDEND = 25
HEALTHCHECK_EXPECTED_SUM = HEALTHCHECK_FIRST_ADDEND + HEALTHCHECK_SECOND_ADDEND
# Used by the fake model backend so `doctor` also passes offline.
HEALTHCHECK_EXPECTED_FINAL_ANSWER = json.dumps({"status": "ready", "sum": HEALTHCHECK_EXPECTED_SUM})

HEALTHCHECK_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ready"]},
        "sum": {"type": "integer"},
    },
    "required": ["status", "sum"],
    "additionalProperties": False,
}


def _render_healthcheck_user_message(role_input: RoleInput) -> str:
    return (f"Compute {HEALTHCHECK_FIRST_ADDEND} + {HEALTHCHECK_SECOND_ADDEND}. "
            "Answer with a JSON object: status set to \"ready\" and sum set to the result.")


def _find_healthcheck_problems(validated_output: dict[str, Any], role_input: RoleInput) -> list[str]:
    if validated_output["sum"] != HEALTHCHECK_EXPECTED_SUM:
        return [f"expected sum {HEALTHCHECK_EXPECTED_SUM}, got {validated_output['sum']}"]
    return []


HEALTHCHECK_ROLE = RoleDefinition(
    role_name="healthcheck",
    prompt_version="healthcheck-v1",
    system_prompt="You are a connectivity check for a local service. Reply only with the requested JSON object.",
    output_schema=HEALTHCHECK_OUTPUT_SCHEMA,
    render_user_message=_render_healthcheck_user_message,
    find_semantic_problems=_find_healthcheck_problems,
)
