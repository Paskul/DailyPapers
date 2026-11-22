"""Flag records that LOOK like the same work but share no identifier. Flag only; never merge.

Plan part 2: "Treat similar titles and author names as possible matches. Do not silently merge papers on that
evidence alone." A flagged pair is stored once (sorted ids) in paper_possible_matches with its evidence, for the
operator (or a later trusted identifier link) to resolve. Call these helpers inside the catalog's write transaction.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from paperwatch.clock import utc_now_timestamp
from paperwatch.text_normalization import extract_author_family_names

MATCH_REASON_SAME_TITLE = "same_title"
MATCH_REASON_SAME_TITLE_SHARED_AUTHOR = "same_title_shared_author"
MATCH_REASON_CONFLICTING_IDENTIFIER_LINK = "conflicting_identifier_link"
MINIMUM_WORDS_IN_TITLE_MATCH_KEY = 4  # "Introduction" or "Editorial" must not pair unrelated papers


def flag_papers_with_same_title(
    connection: sqlite3.Connection, *, new_paper_id: str, title_match_key: str, author_names: tuple[str, ...]
) -> list[str]:
    """Record a possible match for every other paper whose normalized title equals this one. Returns their ids."""
    if len(title_match_key.split()) < MINIMUM_WORDS_IN_TITLE_MATCH_KEY:
        return []
    new_paper_family_names = extract_author_family_names(author_names)
    flagged_paper_ids: list[str] = []
    for other_paper_row in connection.execute(
        "SELECT paper_id, authors_json FROM papers WHERE title_match_key = ? AND paper_id != ? ORDER BY paper_id",
        (title_match_key, new_paper_id),
    ).fetchall():
        shared_family_names = sorted(
            new_paper_family_names & extract_author_family_names(json.loads(other_paper_row["authors_json"]))
        )
        match_reason = MATCH_REASON_SAME_TITLE_SHARED_AUTHOR if shared_family_names else MATCH_REASON_SAME_TITLE
        record_possible_match(
            connection,
            new_paper_id,
            other_paper_row["paper_id"],
            match_reason,
            {"title_match_key": title_match_key, "shared_family_names": shared_family_names},
        )
        flagged_paper_ids.append(other_paper_row["paper_id"])
    return flagged_paper_ids


def record_possible_match(
    connection: sqlite3.Connection, paper_id_a: str, paper_id_b: str, match_reason: str, evidence: dict[str, Any]
) -> None:
    """Insert the unordered pair once; a later sighting of the same pair keeps the first evidence."""
    first_paper_id, second_paper_id = sorted((paper_id_a, paper_id_b))
    if first_paper_id == second_paper_id:
        return
    connection.execute(
        """INSERT OR IGNORE INTO paper_possible_matches (first_paper_id, second_paper_id, match_reason, evidence_json,
               detected_at) VALUES (?, ?, ?, ?, ?)""",
        (
            first_paper_id,
            second_paper_id,
            match_reason,
            json.dumps(evidence, ensure_ascii=False, sort_keys=True),
            utc_now_timestamp(),
        ),
    )
