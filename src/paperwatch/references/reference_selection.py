"""From fetched references to class-3 readings (plan part 8).

    collector  queue_reference_selection_jobs: one `select_references` job per reviewed parent whose references were
               fetched and which has at least one candidate. Also admit_waiting_reference_selections, so choices the
               daily limit held back are admitted on a later day.
    worker     prepare_reference_selection builds the research role's input (parent review, its topics, candidates
               with citation contexts, unused branches); write_reference_selection_outcome stores the validated
               decision, admits the choices, and records the chosen gap branch, all in the job's finish transaction.
Admission rules for each choice (enforced here, never by the model):
  * hop depth = parent's depth + 1; above references.max_citation_hops it is skipped;
  * already reviewed, already queued or promoted, withdrawn, or an active seed: skipped (ends cycles, no rereads);
  * at most references.daily_reference_readings admissions per UTC day, counted from the table, so the cap holds
    across restarts; a choice over the cap waits ('waiting_daily_cap') and is admitted on a later day, most important
    first;
  * an admitted choice records a 'reference' promotion and queues the reading in class 3, ordered by importance then
    topic score, held for references.reference_reading_hold_minutes so `collect` can fetch its full text first.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from paperwatch.catalog import is_paper_withdrawn
from paperwatch.clock import format_utc_timestamp, utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.dispatch_order import dispatch_order_for_important_reference, dispatch_order_for_reference_selection
from paperwatch.jobs import REVIEW_PAPER_JOB_KIND, enqueue_job_if_absent
from paperwatch.references.citation_hops import citation_hop_depth_of_paper, describe_followable_review
from paperwatch.references.reference_candidates import ReferenceCandidate, list_reference_candidates
from paperwatch.references.research_branches import ResearchBranch, offer_unused_branches, record_branch_request
from paperwatch.reading.reading_pipeline import summarize_review_for_assessment
from paperwatch.review_store import load_latest_topic_assessments
from paperwatch.roles.base import RoleRunResult
from paperwatch.roles.research import RESEARCH_ROLE_NAME
from paperwatch.screening.review_promotion import record_reference_promotion_and_queue_reading

SELECT_REFERENCES_JOB_KIND = "select_references"
ADMISSION_ADMITTED = "admitted"
ADMISSION_WAITING_DAILY_CAP = "waiting_daily_cap"
ADMISSION_SKIPPED = "skipped"
MAXIMUM_STORED_EXPLANATION_CHARACTERS = 1200


def build_select_references_job_key(parent_review_id: str) -> str:
    return f"{SELECT_REFERENCES_JOB_KIND}:{parent_review_id}"


@dataclass(frozen=True)
class ReferenceSelectionJobInput:
    parent_review_id: str
    parent_paper_id: str

    def to_payload(self) -> dict[str, str]:
        return {"parent_review_id": self.parent_review_id, "parent_paper_id": self.parent_paper_id}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ReferenceSelectionJobInput:
        return cls(parent_review_id=payload["parent_review_id"], parent_paper_id=payload["parent_paper_id"])


# --------------------------------------------------------------------------------------------------------------------
# Collector: queue the research role's jobs
# --------------------------------------------------------------------------------------------------------------------


def queue_reference_selection_jobs(
    connection: sqlite3.Connection, app_config: AppConfig, *, active_seed_paper_ids: frozenset[str]
) -> int:
    """One job per parent review whose references were fetched, still qualifies, has a candidate, and was not
    decided yet. Call inside a write transaction. Returns how many jobs were added."""
    if not app_config.references.follow_references:
        return 0
    queued_count = 0
    for fetch_row in connection.execute(
        """SELECT fetched.citing_paper_id, fetched.parent_review_id FROM reference_fetches AS fetched
               WHERE fetched.fetch_status = 'fetched'
                 AND fetched.parent_review_id NOT IN (SELECT parent_review_id FROM reference_selection_runs)
               ORDER BY fetched.fetched_at, fetched.citing_paper_id"""
    ).fetchall():
        parent = describe_followable_review(
            connection, app_config, fetch_row["citing_paper_id"], fetch_row["parent_review_id"]
        )
        if parent is None or not list_reference_candidates(
            connection,
            app_config,
            parent.paper_id,
            active_seed_paper_ids=active_seed_paper_ids,
            limit=1,
            with_citation_contexts=False,
        ):
            continue
        queued_count += enqueue_job_if_absent(
            connection,
            job_key=build_select_references_job_key(parent.review_id),
            kind=SELECT_REFERENCES_JOB_KIND,
            paper_id=None,
            input_version=parent.review_id,
            input_payload=ReferenceSelectionJobInput(parent.review_id, parent.paper_id).to_payload(),
            dispatch_order=dispatch_order_for_reference_selection(parent.best_reading_priority / 100.0),
            max_attempts=app_config.worker.max_attempts_per_job,
        )
    return queued_count


# --------------------------------------------------------------------------------------------------------------------
# Worker: the research role's input
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedReferenceSelection:
    job_input: ReferenceSelectionJobInput
    parent_hop_depth: int
    candidates: list[ReferenceCandidate]
    branches: list[ResearchBranch]
    role_input: dict[str, Any]


def prepare_reference_selection(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    job_input: ReferenceSelectionJobInput,
    *,
    active_seed_paper_ids: frozenset[str],
    moment: datetime,
) -> PreparedReferenceSelection:
    review_row = connection.execute(
        "SELECT * FROM reviews WHERE review_id = ? AND paper_id = ?",
        (job_input.parent_review_id, job_input.parent_paper_id),
    ).fetchone()
    if review_row is None:
        raise LookupError(f"review {job_input.parent_review_id} of paper {job_input.parent_paper_id} does not exist")
    paper_row = connection.execute(
        "SELECT title, first_published_date FROM papers WHERE paper_id = ?", (job_input.parent_paper_id,)
    ).fetchone()
    addressed_topic_ids = [
        assessment.topic_id
        for assessment in load_latest_topic_assessments(connection, job_input.parent_review_id)
        if assessment.question_fit >= app_config.references.minimum_parent_question_fit
    ]
    topic_by_id = {topic.topic_id: topic for topic in app_config.topics}
    addressed_topic_ids = [topic_id for topic_id in addressed_topic_ids if topic_id in topic_by_id]
    candidates = list_reference_candidates(
        connection,
        app_config,
        job_input.parent_paper_id,
        active_seed_paper_ids=active_seed_paper_ids,
        limit=app_config.references.candidates_offered_per_selection,
    )
    branches = offer_unused_branches(connection, app_config, addressed_topic_ids, moment)
    review_content = json.loads(review_row["review_json"])
    role_input = {
        "parent": {
            "review_id": review_row["review_id"],
            "title": paper_row["title"],
            "year": (paper_row["first_published_date"] or "")[:4] or None,
            "coverage": review_row["coverage"],
            **summarize_review_for_assessment(review_content),
        },
        "topics": [
            {"topic_id": topic_id, "question": topic_by_id[topic_id].question_for_local_model}
            for topic_id in addressed_topic_ids
        ],
        "candidates": [candidate.as_model_input() for candidate in candidates],
        "branches": [branch.as_model_input() for branch in branches],
        "maximum_selections": app_config.references.references_per_paper,
    }
    return PreparedReferenceSelection(
        job_input, citation_hop_depth_of_paper(connection, job_input.parent_paper_id), candidates, branches, role_input
    )


# --------------------------------------------------------------------------------------------------------------------
# Worker: store the decision and admit the choices (inside the job's finish transaction)
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class SelectionWriteSummary:
    admitted: int = 0
    waiting: int = 0
    skipped: int = 0
    branch_recorded: bool = False

    def describe(self) -> str:
        return (
            f"{self.admitted} reference reading(s) admitted, {self.waiting} waiting for the daily limit, "
            f"{self.skipped} skipped" + ("; one gap search requested" if self.branch_recorded else "")
        )


def write_reference_selection_outcome(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    prepared: PreparedReferenceSelection,
    role_result: RoleRunResult | None,
    *,
    job_id: str,
    active_seed_paper_ids: frozenset[str],
    moment: datetime,
) -> SelectionWriteSummary:
    """`role_result` is None when there was nothing to choose from (no model call was made)."""
    job_input = prepared.job_input
    research_output = role_result.validated_output if role_result else {}
    connection.execute(
        """INSERT OR IGNORE INTO reference_selection_runs (parent_review_id, parent_paper_id, job_id,
               offered_candidate_count, offered_branch_count, research_output_json, role_name, prompt_version,
               model_identity, server_build, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_input.parent_review_id,
            job_input.parent_paper_id,
            job_id,
            len(prepared.candidates),
            len(prepared.branches),
            json.dumps(research_output, ensure_ascii=False),
            RESEARCH_ROLE_NAME if role_result else None,
            role_result.prompt_version if role_result else None,
            role_result.model_response.model_identity if role_result else None,
            role_result.model_response.server_build if role_result else None,
            utc_now_timestamp(),
        ),
    )
    summary = SelectionWriteSummary()
    child_hop_depth = prepared.parent_hop_depth + 1
    for selection in only_offered_choices(research_output, prepared)[0][: app_config.references.references_per_paper]:
        connection.execute(
            """INSERT OR IGNORE INTO reference_selections (parent_review_id, cited_paper_id, parent_paper_id,
                   citation_hop_depth, role_in_parent, importance, explanation, admission_status, selected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_input.parent_review_id,
                selection["candidate_id"],
                job_input.parent_paper_id,
                child_hop_depth,
                selection["role_in_parent"],
                selection["importance"],
                selection["explanation"][:MAXIMUM_STORED_EXPLANATION_CHARACTERS],
                ADMISSION_WAITING_DAILY_CAP,
                format_utc_timestamp(moment),
            ),
        )
    admission_counts = admit_waiting_reference_selections(
        connection,
        app_config,
        active_seed_paper_ids=active_seed_paper_ids,
        moment=moment,
        only_parent_review_id=job_input.parent_review_id,
    )
    summary.admitted, summary.waiting, summary.skipped = admission_counts
    branch_by_id = {branch.branch_id: branch for branch in prepared.branches}
    for gap in only_offered_choices(research_output, prepared)[1][:1]:
        summary.branch_recorded = record_branch_request(
            connection,
            branch_by_id[gap["branch_id"]],
            parent_review_id=job_input.parent_review_id,
            gap_statement=f"{gap['gap']} — {gap['explanation']}",
        )
    return summary


def only_offered_choices(
    research_output: dict[str, Any], prepared: PreparedReferenceSelection
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(selections, gaps) whose ids the host itself offered in this job, each id at most once.

    The role's own check (roles/research.py) already rejects any other id before this point. This repeats it where
    the choice is persisted (plan part 10, defence in depth): even if validation were skipped or changed, a model
    answer can only ever name a catalog paper the parent cites, or an unused branch of an approved search plan."""
    offered_candidate_ids = {candidate.paper_id for candidate in prepared.candidates}
    offered_branch_ids = {branch.branch_id for branch in prepared.branches}
    chosen_candidate_ids: set[str] = set()
    selections = []
    for selection in research_output.get("selected_references", []) or []:
        candidate_id = selection.get("candidate_id") if isinstance(selection, dict) else None
        if candidate_id in offered_candidate_ids and candidate_id not in chosen_candidate_ids:
            chosen_candidate_ids.add(candidate_id)
            selections.append(selection)
    gaps = [
        gap
        for gap in research_output.get("search_gap", []) or []
        if isinstance(gap, dict) and gap.get("branch_id") in offered_branch_ids
    ]
    return selections, gaps


def count_reference_admissions_today(connection: sqlite3.Connection, moment: datetime) -> int:
    day_start = format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))
    return connection.execute(
        "SELECT COUNT(*) FROM reference_selections WHERE admission_status = 'admitted' AND admitted_at >= ?",
        (day_start,),
    ).fetchone()[0]


def admit_waiting_reference_selections(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    *,
    active_seed_paper_ids: frozenset[str],
    moment: datetime,
    only_parent_review_id: str | None = None,
) -> tuple[int, int, int]:
    """Admit waiting choices within today's limit, most important first. Call inside a write transaction.
    Returns (admitted, still waiting, skipped) among the rows considered."""
    reference_settings = app_config.references
    waiting_rows = connection.execute(
        f"""SELECT * FROM reference_selections WHERE admission_status = '{ADMISSION_WAITING_DAILY_CAP}'
              AND (:parent IS NULL OR parent_review_id = :parent)
            ORDER BY importance DESC, selected_at, parent_review_id, cited_paper_id""",
        {"parent": only_parent_review_id},
    ).fetchall()
    admitted_today = count_reference_admissions_today(connection, moment)
    admitted = waiting = skipped = 0
    for selection_row in waiting_rows:
        skip_reason = _reason_to_skip(connection, app_config, selection_row, active_seed_paper_ids)
        if skip_reason is not None:
            _settle_selection(connection, selection_row, ADMISSION_SKIPPED, skip_reason, None)
            skipped += 1
            continue
        if admitted_today >= reference_settings.daily_reference_readings:
            waiting += 1
            continue
        if _queue_reference_reading(connection, app_config, selection_row, moment):
            _settle_selection(connection, selection_row, ADMISSION_ADMITTED, None, format_utc_timestamp(moment))
            admitted_today += 1
            admitted += 1
        else:
            _settle_selection(connection, selection_row, ADMISSION_SKIPPED, "no edition to read", None)
            skipped += 1
    return admitted, waiting, skipped


def _reason_to_skip(
    connection: sqlite3.Connection,
    app_config: AppConfig,
    selection_row: sqlite3.Row,
    active_seed_paper_ids: frozenset[str],
) -> str | None:
    cited_paper_id = selection_row["cited_paper_id"]
    if selection_row["citation_hop_depth"] > app_config.references.max_citation_hops:
        return "beyond the citation hop limit"
    if cited_paper_id in active_seed_paper_ids:
        return "already in your seed library"
    if is_paper_withdrawn(connection, cited_paper_id):
        return "withdrawn or retracted"
    if connection.execute("SELECT 1 FROM reviews WHERE paper_id = ? LIMIT 1", (cited_paper_id,)).fetchone():
        return "already reviewed"
    if (
        connection.execute(
            "SELECT 1 FROM jobs WHERE kind = ? AND paper_id = ? AND status IN ('queued', 'running') LIMIT 1",
            (REVIEW_PAPER_JOB_KIND, cited_paper_id),
        ).fetchone()
        or connection.execute("SELECT 1 FROM paper_promotions WHERE paper_id = ?", (cited_paper_id,)).fetchone()
    ):
        return "already queued or chosen for a reading"
    return None


def _queue_reference_reading(
    connection: sqlite3.Connection, app_config: AppConfig, selection_row: sqlite3.Row, moment: datetime
) -> bool:
    best_score_row = connection.execute(
        """SELECT topic_id, unnormalized_score FROM candidate_topic_scores WHERE paper_id = ?
           ORDER BY unnormalized_score DESC, topic_id LIMIT 1""",
        (selection_row["cited_paper_id"],),
    ).fetchone()
    topic_score = best_score_row["unnormalized_score"] if best_score_row else None
    hold_minutes = app_config.references.reference_reading_hold_minutes
    return record_reference_promotion_and_queue_reading(
        connection,
        selection_row["cited_paper_id"],
        best_topic_id=best_score_row["topic_id"] if best_score_row else None,
        topic_score=topic_score,
        dispatch_order=dispatch_order_for_important_reference(float(selection_row["importance"]), topic_score or 0.0),
        max_attempts=app_config.worker.max_attempts_per_job,
        hold_until=moment + timedelta(minutes=hold_minutes) if hold_minutes > 0 else None,
        moment=moment,
    )


def _settle_selection(
    connection: sqlite3.Connection,
    selection_row: sqlite3.Row,
    admission_status: str,
    skip_reason: str | None,
    admitted_at: str | None,
) -> None:
    connection.execute(
        """UPDATE reference_selections SET admission_status = ?, skip_reason = ?, admitted_at = ?
           WHERE parent_review_id = ? AND cited_paper_id = ?""",
        (
            admission_status,
            skip_reason,
            admitted_at,
            selection_row["parent_review_id"],
            selection_row["cited_paper_id"],
        ),
    )
