"""The permanent note of one comparison: output/comparisons/<comparison id>.md (plan part 8; owner decision
2026-11-20: comparisons get their own note, and part 9 links to it).

Rendered only from the saved comparison and trusted metadata. It links both paper notes relatively
(../papers/<paper id>/<review id>.md, Obsidian-safe) and their sources (built from identifiers). Every model or paper
text is escaped. Like reading notes, a comparison note is written once and never rewritten; `report` restores a
missing one from the database, producing the same bytes.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from paperwatch.catalog import load_paper_aliases
from paperwatch.rendering.file_output import make_output_file_group_readable, write_output_file_atomically
from paperwatch.rendering.markdown_safety import escape_untrusted_inline_text
from paperwatch.rendering.notes import ATTRIBUTION_NOTE, render_code_span, render_source_links
from paperwatch.review_store import load_review_evidence_passages

COMPARABILITY_LABELS = {
    "comparable": "comparable",
    "partially_comparable": "partially comparable",
    "incompatible": "incompatible",
}


def comparison_note_path(comparison_notes_dir: Path, comparison_id: str) -> Path:
    return comparison_notes_dir / f"{comparison_id}.md"


def write_comparison_note_if_missing(
    connection: sqlite3.Connection, comparison_notes_dir: Path, comparison_id: str
) -> bool:
    note_path = comparison_note_path(comparison_notes_dir, comparison_id)
    if note_path.exists():
        make_output_file_group_readable(note_path)
        return False
    write_output_file_atomically(note_path, render_comparison_note(connection, comparison_id).encode("utf-8"))
    return True


def restore_missing_comparison_notes(connection: sqlite3.Connection, comparison_notes_dir: Path) -> int:
    restored_count = 0
    for comparison_row in connection.execute("SELECT comparison_id FROM comparisons ORDER BY comparison_id"):
        restored_count += write_comparison_note_if_missing(
            connection, comparison_notes_dir, comparison_row["comparison_id"]
        )
    return restored_count


def render_comparison_note(connection: sqlite3.Connection, comparison_id: str) -> str:
    comparison_row = connection.execute(
        "SELECT * FROM comparisons WHERE comparison_id = ?", (comparison_id,)
    ).fetchone()
    content = json.loads(comparison_row["comparison_json"])
    side_by_review_id = {
        review_id: _load_side(connection, review_id, label)
        for review_id, label in ((comparison_row["lower_review_id"], "A"), (comparison_row["higher_review_id"], "B"))
    }
    note_lines = [
        f"# Comparison {render_code_span(comparison_id)}",
        "",
        f"- Question: topic {render_code_span(comparison_row['topic_id'])}",
        f"- Conditions: **{COMPARABILITY_LABELS.get(content['conditions_comparability'], 'unknown')}**"
        + (
            f" · shared evidence dimensions: {', '.join(content.get('shared_dimensions', []))}"
            if content.get("shared_dimensions")
            else ""
        ),
        f"- Compared {comparison_row['completed_at']} · model "
        f"{escape_untrusted_inline_text(comparison_row['model_identity'])} · prompt "
        f"{comparison_row['prompt_version']}",
        "",
        "## Papers",
        "",
    ]
    for side in side_by_review_id.values():
        note_lines.append(
            f"- **{side['label']}** {escape_untrusted_inline_text(side['title'])}"
            f" ({escape_untrusted_inline_text(side['year']) or 'year unknown'}, {side['coverage']}) · "
            f"[reading note](<../papers/{side['paper_id']}/{side['review_id']}.md>) · "
            f"source: {side['sources']}"
        )
    if content["insufficient_evidence"]:
        note_lines += ["", "> The comparison found the two reviews insufficient to support a conclusion."]
    note_lines += [
        "",
        "## What the evidence permits the reader to conclude",
        "",
        _render_supported(content["reader_conclusion"], side_by_review_id),
    ]
    note_lines += ["", "## Differences in conditions", ""]
    note_lines += [
        f"- ({escape_untrusted_inline_text(difference['dimension'])}) "
        + _render_supported(difference, side_by_review_id)
        for difference in content["condition_differences"]
    ] or ["- None stated."]
    for heading, field_name in (("Agreements", "agreements"), ("Differences", "differences")):
        note_lines += ["", f"## {heading}", ""]
        note_lines += [f"- {_render_supported(item, side_by_review_id)}" for item in content[field_name]] or [
            "- None stated."
        ]
    if content["result_ranking"]:
        ranking = content["result_ranking"][0]
        preferred_side = side_by_review_id.get(ranking["preferred_review_id"])
        note_lines += [
            "",
            "## Ranking of the results",
            "",
            f"- Preferred: **{preferred_side['label'] if preferred_side else '?'}** for "
            f"{escape_untrusted_inline_text(ranking['criterion'])}. "
            f"{escape_untrusted_inline_text(ranking['justification'])}"
            + _render_evidence(ranking["evidence"], side_by_review_id),
        ]
        if ranking["conditions_caveat"].strip():
            note_lines.append(
                "- Caveat about the conditions: " + escape_untrusted_inline_text(ranking["conditions_caveat"])
            )
    note_lines += ["", "## Evidence passages", "", f"> {ATTRIBUTION_NOTE}", ""]
    for side in side_by_review_id.values():
        for passage in side["passages"].values():
            note_lines.append(
                f"- {side['label']} `{passage.passage_id}` "
                f"({escape_untrusted_inline_text(passage.location_label)}): "
                f"{escape_untrusted_inline_text(passage.passage_text[:700])}"
            )
    return "\n".join(note_lines).rstrip() + "\n"


def _load_side(connection: sqlite3.Connection, review_id: str, label: str) -> dict[str, Any]:
    review_row = connection.execute(
        """SELECT review.review_id, review.paper_id, review.coverage, paper.title, paper.first_published_date
           FROM reviews AS review JOIN papers AS paper ON paper.paper_id = review.paper_id
           WHERE review.review_id = ?""",
        (review_id,),
    ).fetchone()
    return {
        "label": label,
        "review_id": review_id,
        "paper_id": review_row["paper_id"],
        "title": review_row["title"],
        "year": (review_row["first_published_date"] or "")[:4],
        "coverage": review_row["coverage"],
        "sources": render_source_links(load_paper_aliases(connection, review_row["paper_id"])),
        "passages": load_review_evidence_passages(connection, review_id),
    }


def _render_supported(item: dict[str, Any], side_by_review_id: dict[str, dict[str, Any]]) -> str:
    return (escape_untrusted_inline_text(item["statement"]) or "Not stated.") + _render_evidence(
        item["evidence"], side_by_review_id
    )


def _render_evidence(evidence: list[dict[str, Any]], side_by_review_id: dict[str, dict[str, Any]]) -> str:
    rendered = []
    for item in evidence:
        side = side_by_review_id.get(item["review_id"])
        if side is None:
            continue
        passage = side["passages"].get(item["passage_id"])
        location = f" ({escape_untrusted_inline_text(passage.location_label)})" if passage else ""
        rendered.append(
            f"{side['label']} `{item['passage_id']}`{location}: “{escape_untrusted_inline_text(item['excerpt'])}”"
        )
    return (" — " + "; ".join(rendered)) if rendered else ""
