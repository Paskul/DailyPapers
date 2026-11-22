"""The saved comparisons a report shows (plan part 9: "Include completed comparisons that help interpret the selected
work"). Loaded by id from the frozen snapshot; the writer sees them as data, the renderer links their notes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ComparisonSummary:
    comparison_id: str
    topic_id: str
    review_ids: tuple[str, str]
    paper_titles: tuple[str, str]  # untrusted text; escaped when rendered
    conditions_comparability: str  # comparable | partially_comparable | incompatible
    reader_conclusion: str  # untrusted model text; escaped when rendered
    insufficient_evidence: bool

    @property
    def note_relative_path_from_daily_dir(self) -> str:
        return f"../comparisons/{self.comparison_id}.md"

    def to_writer_input(self) -> dict:
        return {
            "comparison_id": self.comparison_id,
            "topic_id": self.topic_id,
            "review_ids": list(self.review_ids),
            "conditions_comparability": self.conditions_comparability,
            "reader_conclusion": self.reader_conclusion,
            "insufficient_evidence": self.insufficient_evidence,
        }


def load_comparison_summaries(
    connection: sqlite3.Connection, comparison_ids: tuple[str, ...]
) -> list[ComparisonSummary]:
    summaries = []
    for comparison_id in comparison_ids:
        comparison_row = connection.execute(
            """SELECT comparison.*, lower_paper.title AS lower_title, higher_paper.title AS higher_title
               FROM comparisons AS comparison
               JOIN reviews AS lower_review ON lower_review.review_id = comparison.lower_review_id
               JOIN papers AS lower_paper ON lower_paper.paper_id = lower_review.paper_id
               JOIN reviews AS higher_review ON higher_review.review_id = comparison.higher_review_id
               JOIN papers AS higher_paper ON higher_paper.paper_id = higher_review.paper_id
               WHERE comparison.comparison_id = ?""",
            (comparison_id,),
        ).fetchone()
        if comparison_row is None:
            continue
        content = json.loads(comparison_row["comparison_json"])
        summaries.append(
            ComparisonSummary(
                comparison_id=comparison_id,
                topic_id=comparison_row["topic_id"],
                review_ids=(comparison_row["lower_review_id"], comparison_row["higher_review_id"]),
                paper_titles=(comparison_row["lower_title"], comparison_row["higher_title"]),
                conditions_comparability=content.get("conditions_comparability", "unknown"),
                reader_conclusion=content.get("reader_conclusion", {}).get("statement", ""),
                insufficient_evidence=bool(content.get("insufficient_evidence")),
            )
        )
    return summaries
