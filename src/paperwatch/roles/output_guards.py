"""Host-side checks against invented actions and references in model answers.

The model has zero tools, so an "action" (upload a file, open a link, fetch something) can only show up as data in
its answer. The host never executes anything from an answer, and:
  * unknown fields such as {"actions": [...]} are rejected by every role's schema (additionalProperties: false);
  * ids outside the input are rejected by each role's own check (topic, passage, and candidate ids);
  * web addresses that do not occur in the role input are rejected here as invented references.
A web address that DOES occur in the (untrusted) input may be quoted back; the renderer defangs it so it is never
a clickable link (rendering/markdown_safety.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

WEB_ADDRESS_PATTERN = re.compile(r"(?i)\b(?:https?|ftp|file)://[^\s<>\"'`)\]]+|\bwww\.[^\s<>\"'`)\]]+")
TRAILING_PUNCTUATION = ".,;:!?"


def iterate_text_values(value: Any) -> Iterator[str]:
    """Every string inside a JSON-like value (dict keys excluded)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for nested_value in value.values():
            yield from iterate_text_values(nested_value)
    elif isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from iterate_text_values(nested_value)


def find_web_addresses(text: str) -> list[str]:
    return [match.rstrip(TRAILING_PUNCTUATION) for match in WEB_ADDRESS_PATTERN.findall(text)]


def find_invented_web_addresses(validated_output: Mapping[str, Any], role_input: Mapping[str, Any]) -> list[str]:
    """Problems for each web address in the answer that does not occur verbatim anywhere in the role input."""
    input_text = "\n".join(iterate_text_values(role_input)).lower()
    invented_addresses: list[str] = []
    for output_text in iterate_text_values(validated_output):
        for web_address in find_web_addresses(output_text):
            if web_address.lower() not in input_text and web_address not in invented_addresses:
                invented_addresses.append(web_address)
    return [f"invented reference '{web_address}' does not occur in the input" for web_address in invented_addresses]
