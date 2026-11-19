"""Small, dependency-free text normalizers shared by the catalog, review inputs, and schema migrations.

Kept separate so that low-level modules (migrations, identity) can use them without importing higher layers.
"""

from __future__ import annotations

import re
import unicodedata

NON_ALPHANUMERIC_RUN_PATTERN = re.compile(r"[^0-9a-z]+")


def collapse_whitespace(text: str) -> str:
    """Unicode NFC, then every whitespace run becomes one space; leading/trailing whitespace is removed."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def build_title_match_key(title: str) -> str:
    """Accent-free, case-folded, punctuation-free title. Equal keys mark a POSSIBLE duplicate, never a merge."""
    decomposed_title = unicodedata.normalize("NFKD", title)
    accent_free_title = "".join(character for character in decomposed_title if not unicodedata.combining(character))
    return NON_ALPHANUMERIC_RUN_PATTERN.sub(" ", accent_free_title.casefold()).strip()


def extract_author_family_names(author_names: list[str] | tuple[str, ...]) -> frozenset[str]:
    """Best-effort family names: 'Lovelace, Ada' -> 'lovelace'; 'Ada Lovelace' -> 'lovelace'. For overlap checks."""
    family_names: set[str] = set()
    for author_name in author_names:
        family_part = author_name.split(",")[0] if "," in author_name else (author_name.split() or [""])[-1]
        family_key = build_title_match_key(family_part)
        if family_key:
            family_names.add(family_key)
    return frozenset(family_names)
