"""The cited works offered to the research role for one reviewed parent paper (plan part 8).

A candidate is a work the parent cites (a citation edge resolved to a catalog paper) that is NOT
    the parent itself, an active seed of the user's library, withdrawn, already reviewed, already queued, or already
    promoted for a reading (this skips previously reviewed inputs and ends cycles: a paper's ancestors are reviewed).
Candidates are offered best screening score first (the candidate's best raw topic score), at most
references.candidates_offered_per_selection, each with its citation contexts in the parent's text when the parent's
full text is cached. Candidate ids are internal paper ids, so a choice can be checked against this exact list.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from paperwatch.config import AppConfig
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND
from paperwatch.references.citation_contexts import CitationContext, CitedWorkDescription, find_citation_contexts
from paperwatch.references.paper_text import bibliography_passages, body_passages, load_extracted_text_passages
from paperwatch.readings import select_edition_to_read
from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME
from paperwatch.text_normalization import extract_author_family_names

MAXIMUM_CANDIDATE_ABSTRACT_CHARACTERS = 600
MAXIMUM_AUTHORS_SHOWN = 3


@dataclass(frozen=True)
class ReferenceCandidate:
    paper_id: str
    title: str
    authors: tuple[str, ...]
    publication_year: str | None
    abstract: str | None
    best_screening_score: float | None  # raw 0-1 screening score of its best topic; None if not screened yet
    citation_contexts: tuple[CitationContext, ...] = field(default_factory=tuple)

    def as_model_input(self) -> dict[str, Any]:
        return {
            "candidate_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.publication_year,
            "abstract": (self.abstract or "(no abstract available)")[:MAXIMUM_CANDIDATE_ABSTRACT_CHARACTERS],
            "citation_contexts": [context.as_model_input() for context in self.citation_contexts],
        }


def list_reference_candidates(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    parent_paper_id: str,
    *,
    active_seed_paper_ids: frozenset[str],
    limit: int,
    with_citation_contexts: bool = True,
) -> list[ReferenceCandidate]:
    candidate_rows = connection.execute(
        f"""SELECT edge.cited_paper_id AS paper_id,
                   (SELECT MAX(score.unnormalized_score) FROM candidate_topic_scores AS score
                    WHERE score.paper_id = edge.cited_paper_id) AS best_screening_score
            FROM citation_edges AS edge
            WHERE edge.citing_paper_id = :parent AND edge.cited_paper_id IS NOT NULL
              AND edge.cited_paper_id != :parent
              AND edge.cited_paper_id NOT IN (SELECT paper_id FROM paper_withdrawals)
              AND edge.cited_paper_id NOT IN (SELECT paper_id FROM reviews)
              AND edge.cited_paper_id NOT IN (SELECT paper_id FROM paper_promotions)
              AND edge.cited_paper_id NOT IN (SELECT paper_id FROM jobs
                                              WHERE kind = '{REVIEW_PAPER_JOB_KIND}' AND paper_id IS NOT NULL)
            GROUP BY edge.cited_paper_id
            ORDER BY best_screening_score IS NULL, best_screening_score DESC, edge.cited_paper_id""",
        {"parent": parent_paper_id},
    ).fetchall()
    candidate_rows = [row for row in candidate_rows if row["paper_id"] not in active_seed_paper_ids][:limit]
    if not candidate_rows:
        return []
    parent_passages = (
        load_extracted_text_passages(connection, app_config, parent_paper_id) if with_citation_contexts else None
    )
    candidates = []
    for candidate_row in candidate_rows:
        edition_row = select_edition_to_read(connection, candidate_row["paper_id"])
        if edition_row is None:
            continue
        authors = tuple(json.loads(edition_row["authors_json"] or "[]"))
        publication_year = (edition_row["published_date"] or "")[:4] or None
        contexts: tuple[CitationContext, ...] = ()
        if parent_passages:
            cited_work = CitedWorkDescription(
                title=edition_row["title"],
                publication_year=publication_year,
                first_author_family_name=_first_author_family_name(authors),
                identifier_values=_citable_identifier_values(connection, candidate_row["paper_id"]),
            )
            contexts = tuple(
                find_citation_contexts(
                    bibliography_passages(parent_passages), body_passages(parent_passages), cited_work
                )
            )
        candidates.append(
            ReferenceCandidate(
                paper_id=candidate_row["paper_id"],
                title=edition_row["title"],
                authors=authors[:MAXIMUM_AUTHORS_SHOWN],
                publication_year=publication_year,
                abstract=edition_row["abstract"],
                best_screening_score=candidate_row["best_screening_score"],
                citation_contexts=contexts,
            )
        )
    return candidates


def _first_author_family_name(authors: tuple[str, ...]) -> str | None:
    if not authors:
        return None
    family_names = extract_author_family_names([authors[0]])
    return next(iter(family_names), None)


def _citable_identifier_values(connection: sqlite3.Connection, paper_id: str) -> tuple[str, ...]:
    """DOIs and arXiv ids as they could be printed in a bibliography."""
    return tuple(
        alias_row["provider_paper_id"]
        for alias_row in connection.execute(
            "SELECT provider_paper_id FROM paper_aliases WHERE paper_id = ? AND provider IN (?, ?)",
            (paper_id, DOI_SCHEME, ARXIV_SCHEME),
        )
    )
