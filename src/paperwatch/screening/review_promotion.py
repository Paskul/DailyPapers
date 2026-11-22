"""Promote screened candidates to a reading, within the daily limit, in their priority class (plan part 6).

Replaces the part-5 placeholder that queued every collected paper. Now:
  * page ingestion only catalogs records; a NEW paper waits for screening (status `waiting_input`);
  * after collection, screening scores every candidate. `promote_screened_candidates` ranks each raw signal within
    its topic's pool of still-unpromoted candidates (signal_normalization.py), so signals and topics share one scale,
    chooses today's papers from those scores (promotion_selection.py), and queues each reading as class 1 (inside
    its seven-day fresh window) or class 2, ordered by that score (paperwatch.dispatch_order). The promotion row
    keeps the normalized score, the raw score, and the pool size. The rest become `screened_out`: they stay
    searchable in SQLite, are rescored when the profile changes, and can be promoted on a later day;
  * an already promoted paper whose text changes (a new arXiv version, an abstract appearing, or full text
    extracted by the source-text stage) gets its new reading queued without spending another slot: it was already
    chosen once;
  * `collect` holds the readings it promotes until its source-text stage has run (paperwatch.readings), so the
    worker reads the full text instead of the abstract when the PDF is fetched in the same run.
The daily limit counts promotions per UTC day (like the candidate budget), and the selection plus its queueing commit
in one write transaction, so two collectors running at once cannot exceed it. Active seeds and withdrawn works are
never promoted. Source text is then acquired for the promoted papers by the same `collect` run.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

from paperwatch.catalog import is_paper_withdrawn, load_paper_row, set_paper_status
from paperwatch.clock import format_utc_timestamp, utc_now
from paperwatch.config import ScreeningSettings
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import (
    JobDispatchOrder,
    dispatch_order_for_important_reference,
    dispatch_order_for_promoted_paper,
)
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed
from paperwatch.screening.candidate_screening import CandidateScreeningOutcome
from paperwatch.screening.promotion_selection import (
    PromotionCandidate,
    PromotionDecision,
    PromotionReason,
    compute_slots_available,
    select_papers_for_promotion,
)
from paperwatch.screening.screening_score import SignalWeights, combine_screening_signals
from paperwatch.screening.signal_normalization import compute_pool_normalized_scores

PROMOTION_REASON_DIRECT = "direct"  # an operator tool or a test queued the paper without screening
# Part 8: the research role chose the paper from a reviewed paper's references. Not counted against the daily
# promotion limit (it has its own: references.daily_reference_readings); its readings stay in class 3.
PROMOTION_REASON_REFERENCE = "reference"
SCREENING_PROMOTION_REASONS = tuple(reason.value for reason in PromotionReason)


@dataclass(frozen=True)
class PromotionScoreRecord:
    """The scores a promotion row keeps: normalized (what decided), raw (for calibration), and the pool size."""

    screening_score: float | None = None
    unnormalized_score: float | None = None
    normalization_pool_size: int | None = None


DIRECT_PROMOTION_SCORES = PromotionScoreRecord()


@dataclass
class PromotionRunOutcome:
    promoted: list[PromotionDecision] = field(default_factory=list)
    newly_screened_out: int = 0
    promoted_today_before_this_run: int = 0
    daily_promotion_limit: int = 0

    def describe(self) -> str:
        reason_counts = {
            reason: sum(decision.promotion_reason == reason for decision in self.promoted) for reason in PromotionReason
        }
        promoted_today = self.promoted_today_before_this_run + len(self.promoted)
        return (
            f"Promotion: {len(self.promoted)} paper(s) queued for review now "
            f"({reason_counts[PromotionReason.TOPIC_SHARE]} by topic share, "
            f"{reason_counts[PromotionReason.SPARE_SLOT]} by spare slot, "
            f"{reason_counts[PromotionReason.NEAR_CUTOFF]} near the cutoff); {promoted_today} of "
            f"{self.daily_promotion_limit} promoted today (UTC); {self.newly_screened_out} newly screened out"
        )


def utc_day_start_timestamp(moment: datetime) -> str:
    return format_utc_timestamp(moment.replace(hour=0, minute=0, second=0, microsecond=0))


def count_screening_promotions_today(connection: sqlite3.Connection, moment: datetime) -> sqlite3.Row:
    placeholders = ", ".join("?" * len(SCREENING_PROMOTION_REASONS))
    return connection.execute(
        f"""SELECT COALESCE(SUM(promotion_reason != 'near_cutoff'), 0) AS regular_count,
                   COALESCE(SUM(promotion_reason = 'near_cutoff'), 0) AS near_cutoff_count
            FROM paper_promotions WHERE promoted_at >= ? AND promotion_reason IN ({placeholders})""",  # noqa: S608  # only ? placeholders are interpolated
        (utc_day_start_timestamp(moment), *SCREENING_PROMOTION_REASONS),
    ).fetchone()


def promote_screened_candidates(
    connection: sqlite3.Connection,
    screening_settings: ScreeningSettings,
    screening_outcome: CandidateScreeningOutcome,
    active_topic_ids: tuple[str, ...],
    *,
    max_attempts_per_job: int,
    moment: datetime | None = None,
    hold_new_readings_until: datetime | None = None,
) -> PromotionRunOutcome:
    moment = moment or utc_now()
    outcome = PromotionRunOutcome(daily_promotion_limit=screening_settings.daily_promotion_limit)
    with write_transaction(connection):
        # Re-check inside the transaction: a concurrent collector may have promoted some of these meanwhile.
        already_taken_paper_ids = {
            row["paper_id"]
            for row in connection.execute(
                "SELECT paper_id FROM paper_promotions UNION SELECT paper_id FROM jobs WHERE paper_id IS NOT NULL"
            )
        }
        pool_signals = {
            paper_id: {
                topic_id: signals for topic_id, signals in signals_by_topic.items() if topic_id in active_topic_ids
            }
            for paper_id, signals_by_topic in screening_outcome.current_signals_by_paper.items()
            if paper_id not in already_taken_paper_ids
        }
        normalized_scores = compute_pool_normalized_scores(
            pool_signals, SignalWeights.from_settings(screening_settings)
        )
        pool_size_by_topic_id = {
            topic_id: sum(topic_id in signals_by_topic for signals_by_topic in pool_signals.values())
            for topic_id in active_topic_ids
        }
        promotion_candidates = [
            PromotionCandidate(paper_id, normalized_scores.get(paper_id, {})) for paper_id in sorted(pool_signals)
        ]

        today_counts = count_screening_promotions_today(connection, moment)
        outcome.promoted_today_before_this_run = today_counts["regular_count"] + today_counts["near_cutoff_count"]
        topic_share_counts = {
            row["topic_id"]: row["share_count"]
            for row in connection.execute(
                """SELECT topic_id, COUNT(*) AS share_count FROM paper_promotions
               WHERE promoted_at >= ? AND promotion_reason = 'topic_share' GROUP BY topic_id""",
                (utc_day_start_timestamp(moment),),
            )
        }
        slots = compute_slots_available(
            active_topic_ids=active_topic_ids,
            daily_promotion_limit=screening_settings.daily_promotion_limit,
            near_cutoff_promotions=screening_settings.near_cutoff_promotions,
            regular_promotions_today=today_counts["regular_count"],
            near_cutoff_promotions_today=today_counts["near_cutoff_count"],
            topic_share_promotions_today_by_topic_id=topic_share_counts,
        )
        decisions = select_papers_for_promotion(promotion_candidates, slots, screening_settings.minimum_screening_score)

        for decision in decisions:
            candidate = screening_outcome.candidates[decision.paper_id]
            dispatch_order = dispatch_order_for_promoted_paper(
                candidate.first_published_date, decision.screening_score, moment
            )
            best_topic_signals = pool_signals[decision.paper_id][decision.best_topic_id]
            score_record = PromotionScoreRecord(
                screening_score=decision.screening_score,
                unnormalized_score=combine_screening_signals(
                    best_topic_signals, SignalWeights.from_settings(screening_settings)
                ),
                normalization_pool_size=pool_size_by_topic_id[decision.best_topic_id],
            )
            _record_promotion(
                connection,
                decision.paper_id,
                decision.promotion_reason.value,
                decision.selecting_topic_id,
                decision.best_topic_id,
                score_record,
                dispatch_order,
                moment,
            )
            _queue_reading_and_mark_paper(
                connection, decision.paper_id, dispatch_order, max_attempts_per_job, hold_until=hold_new_readings_until
            )
            outcome.promoted.append(decision)

        promoted_paper_ids = {decision.paper_id for decision in decisions}
        for candidate in promotion_candidates:
            if candidate.paper_id not in promoted_paper_ids:
                outcome.newly_screened_out += connection.execute(
                    "UPDATE papers SET status = 'screened_out', updated_at = ? "
                    "WHERE paper_id = ? AND status = 'waiting_input'",
                    (format_utc_timestamp(moment), candidate.paper_id),
                ).rowcount
    return outcome


def queue_new_reading_for_promoted_paper_in_transaction(
    connection: sqlite3.Connection, paper_id: str, *, active_seed_paper_ids: frozenset[str], max_attempts: int
) -> bool:
    """During ingestion: if the paper was promoted before and its preferred text is new, queue the new reading in
    the paper's current class. New papers return False here and wait for screening. Returns True if a job was added."""
    promotion_row = connection.execute(
        "SELECT promotion_reason, screening_score FROM paper_promotions WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if promotion_row is None or paper_id in active_seed_paper_ids or is_paper_withdrawn(connection, paper_id):
        return False
    if promotion_row["promotion_reason"] == PROMOTION_REASON_REFERENCE:
        dispatch_order = dispatch_order_of_reference_reading(connection, paper_id, promotion_row["screening_score"])
    else:
        paper_row = load_paper_row(connection, paper_id)
        dispatch_order = dispatch_order_for_promoted_paper(
            paper_row["first_published_date"], promotion_row["screening_score"] or 0.0, utc_now()
        )
    return _queue_reading_and_mark_paper(connection, paper_id, dispatch_order, max_attempts)


def dispatch_order_of_reference_reading(
    connection: sqlite3.Connection, paper_id: str, topic_score: float | None
) -> JobDispatchOrder:
    """Class 3, by the highest importance any admitting parent gave the paper, then its topic score (plan part 6)."""
    importance_row = connection.execute(
        """SELECT MAX(importance) AS importance FROM reference_selections
           WHERE cited_paper_id = ? AND admission_status = 'admitted'""",
        (paper_id,),
    ).fetchone()
    return dispatch_order_for_important_reference(float(importance_row["importance"] or 1), topic_score or 0.0)


def record_reference_promotion_and_queue_reading(
    connection: sqlite3.Connection,
    paper_id: str,
    *,
    best_topic_id: str | None,
    topic_score: float | None,
    dispatch_order: JobDispatchOrder,
    max_attempts: int,
    hold_until: datetime | None,
    moment: datetime,
) -> bool:
    """Part 8: record a 'reference' promotion and queue the reading in class 3, held until `collect` has tried to
    fetch the full text (owner decision 2026-11-20). Call inside a write transaction. Returns True if queued."""
    _record_promotion(
        connection,
        paper_id,
        PROMOTION_REASON_REFERENCE,
        best_topic_id,
        best_topic_id,
        PromotionScoreRecord(screening_score=topic_score, unnormalized_score=topic_score),
        dispatch_order,
        moment,
    )
    return _queue_reading_and_mark_paper(connection, paper_id, dispatch_order, max_attempts, hold_until=hold_until)


def promote_paper_directly_and_queue_review(
    connection: sqlite3.Connection,
    paper_id: str,
    *,
    max_attempts: int,
    active_seed_paper_ids: frozenset[str] = frozenset(),
) -> bool:
    """Queue a reading of one paper WITHOUT screening (operator tools and tests), in its own transaction. It is
    recorded as a 'direct' promotion, which does not count against the daily limit. Active seeds and withdrawn works
    are still refused. Returns True if a job was added."""
    with write_transaction(connection):
        if paper_id in active_seed_paper_ids or is_paper_withdrawn(connection, paper_id):
            return False
        paper_row = load_paper_row(connection, paper_id)
        moment = utc_now()
        dispatch_order = dispatch_order_for_promoted_paper(paper_row["first_published_date"], 0.0, moment)
        _record_promotion(
            connection, paper_id, PROMOTION_REASON_DIRECT, None, None, DIRECT_PROMOTION_SCORES, dispatch_order, moment
        )
        return _queue_reading_and_mark_paper(connection, paper_id, dispatch_order, max_attempts)


def _record_promotion(
    connection: sqlite3.Connection,
    paper_id: str,
    promotion_reason: str,
    selecting_topic_id: str | None,
    best_topic_id: str | None,
    score_record: PromotionScoreRecord,
    dispatch_order: JobDispatchOrder,
    moment: datetime,
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO paper_promotions (paper_id, promotion_reason, topic_id, best_topic_id,
               screening_score, unnormalized_score, normalization_pool_size, priority_class, promoted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            paper_id,
            promotion_reason,
            selecting_topic_id,
            best_topic_id,
            score_record.screening_score,
            score_record.unnormalized_score,
            score_record.normalization_pool_size,
            int(dispatch_order.priority_class),
            format_utc_timestamp(moment),
        ),
    )


def _queue_reading_and_mark_paper(
    connection: sqlite3.Connection,
    paper_id: str,
    dispatch_order: JobDispatchOrder,
    max_attempts: int,
    hold_until: datetime | None = None,
) -> bool:
    enqueue_outcome, _reading_input = enqueue_reading_if_needed(
        connection, paper_id, dispatch_order=dispatch_order, max_attempts=max_attempts, hold_until=hold_until
    )
    if enqueue_outcome == ReadingEnqueueOutcome.QUEUED:
        set_paper_status(connection, paper_id, "queued")
        return True
    return False
