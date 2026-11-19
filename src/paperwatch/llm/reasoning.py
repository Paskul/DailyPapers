"""Separate the final answer from reasoning output.

llama.cpp started with `--reasoning on` returns reasoning in `message.reasoning_content` and the answer in
`message.content`. If the server is ever started with `--reasoning-format none`, the reasoning arrives inline as
<think>...</think> inside `content` instead. Both shapes are handled here so the answer parser only sees the answer.
"""

from __future__ import annotations

import re

COMPLETE_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def separate_reasoning_from_final_answer(message_content: str | None,
                                         message_reasoning_content: str | None) -> tuple[str, str]:
    """Return (final_answer_text, reasoning_text)."""
    remaining_content = message_content or ""
    reasoning_parts = [message_reasoning_content] if message_reasoning_content else []

    reasoning_parts.extend(COMPLETE_THINK_BLOCK_PATTERN.findall(remaining_content))
    remaining_content = COMPLETE_THINK_BLOCK_PATTERN.sub("", remaining_content)

    # Some chat templates put the opening tag in the prompt, so only a closing tag appears in the output.
    if THINK_CLOSE_TAG in remaining_content:
        leading_reasoning, _, remaining_content = remaining_content.rpartition(THINK_CLOSE_TAG)
        reasoning_parts.append(leading_reasoning)
    # An unterminated opening tag means the output stopped while still reasoning: there is no final answer.
    if THINK_OPEN_TAG in remaining_content:
        answer_before_reasoning, _, unfinished_reasoning = remaining_content.partition(THINK_OPEN_TAG)
        reasoning_parts.append(unfinished_reasoning)
        remaining_content = answer_before_reasoning

    reasoning_text = "\n".join(part.strip() for part in reasoning_parts if part and part.strip())
    return remaining_content.strip(), reasoning_text
