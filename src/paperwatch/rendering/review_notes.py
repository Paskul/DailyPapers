"""Load a saved review with everything its note needs, and write the note file (one place for worker and report).

The worker writes the note right after the review commits; the report writes any note that is missing (a crash
between commit and file write, or a deleted file), always from the saved review, so the bytes are the same.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from paperwatch.catalog import load_paper_aliases
from paperwatch.rendering.file_output import make_output_file_group_readable, write_output_file_atomically
from paperwatch.rendering.notes import SelectedReview, render_paper_note
from paperwatch.review_store import load_latest_topic_assessments, load_review_evidence_passages
from paperwatch.seeds.seed_items import build_seed_id


def load_selected_review(connection: sqlite3.Connection, review_id: str, *, rank: int = 0,
                         assessed_no_later_than: str | None = None) -> SelectedReview:
    """`assessed_no_later_than` (a report's cutoff) ignores topic re-assessments made after it (plan part 9)."""
    review_row = connection.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
    topic_assessments = load_latest_topic_assessments(connection, review_id,
                                                      assessed_no_later_than=assessed_no_later_than)
    matched_seed_ids = {seed_id for assessment in topic_assessments
                        for seed_id in assessment.assessment_record.get("matched_seed_ids", [])}
    return SelectedReview(
        rank=rank,
        paper_row=connection.execute("SELECT * FROM papers WHERE paper_id = ?", (review_row["paper_id"],)).fetchone(),
        review_row=review_row,
        review_content=json.loads(review_row["review_json"]),
        paper_aliases=load_paper_aliases(connection, review_row["paper_id"]),
        topic_assessments=topic_assessments,
        evidence_passages=load_review_evidence_passages(connection, review_id),
        seed_titles_by_id=load_seed_titles(connection, matched_seed_ids))


def load_seed_titles(connection: sqlite3.Connection, seed_ids: set[str]) -> dict[str, str]:
    seed_titles: dict[str, str] = {}
    for seed_id in seed_ids:
        seed_source, _separator, seed_item_key = seed_id.partition(":")
        seed_row = connection.execute("SELECT title FROM seed_items WHERE seed_source = ? AND seed_item_key = ?",
                                      (seed_source, seed_item_key)).fetchone()
        if seed_row is not None and build_seed_id(seed_source, seed_item_key) == seed_id:
            seed_titles[seed_id] = seed_row["title"]
    return seed_titles


def paper_note_path(paper_notes_dir: Path, selected_review: SelectedReview) -> Path:
    """output/papers/<paper id>/<review id>.md: internal ids only, never paper text."""
    return paper_notes_dir / selected_review.paper_row["paper_id"] / f"{selected_review.review_row['review_id']}.md"


def write_paper_note_if_missing(paper_notes_dir: Path, selected_review: SelectedReview) -> bool:
    """Notes are immutable per review revision: write or restore, never rewrite. Returns True if written."""
    note_path = paper_note_path(paper_notes_dir, selected_review)
    if note_path.exists():
        make_output_file_group_readable(note_path)
        return False
    write_output_file_atomically(note_path, render_paper_note(selected_review).encode("utf-8"))
    return True
