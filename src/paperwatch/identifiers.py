"""Internal identifiers. These, never titles or provider strings, are used in filenames and note paths."""

from __future__ import annotations

import secrets

PAPER_ID_PREFIX = "p"
JOB_ID_PREFIX = "j"
REVIEW_ID_PREFIX = "r"
MODEL_CALL_ID_PREFIX = "c"
EDITION_ID_PREFIX = "e"
READING_GENERATION_ID_PREFIX = "g"  # explicit operator rereads
COMPARISON_ID_PREFIX = "k"


def new_internal_id(prefix: str) -> str:
    """Return e.g. 'p_3f9a1c0b7d2e4a51'. Only [a-z0-9_] characters, so it is always safe as a path component."""
    return f"{prefix}_{secrets.token_hex(8)}"
