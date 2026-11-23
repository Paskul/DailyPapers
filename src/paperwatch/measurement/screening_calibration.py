"""Calibrate `screening.minimum_unnormalized_score` from the topic assessor's verdicts on papers already reviewed.

Every reviewed paper that screening scored has two numbers per topic: the RAW screening score it was promoted on (the
weighted mean of its raw signals, candidate_topic_scores.unnormalized_score) and the reading priority (0-100) the
topic assessor gave it after reading it. A floor F would have kept every paper below F from being read. For a range
of floors the table shows how many reviews each would have avoided, how many of those the assessor rated low, and
how many papers that reached the report (report.minimum_reading_priority) it would have lost.

The recommendation is the highest floor, in steps of FLOOR_STEP, that stays at least SAFETY_MARGIN below every topic's
lowest raw score among papers that reached the report, given at least MINIMUM_REVIEWED_PAPERS reviewed papers and
MINIMUM_REPORTED_PAPERS that reached the report for that topic. Third pilot (2026-11-23, one topic, 119 reviewed
papers): 0.55 would have avoided 19 reviews, 15 of them rated below 40, and lost none of the 33 that reached 65.

The floor depends on the embedding model and on how each topic is worded: recalibrate after changing either. Near-
cutoff promotions (screening.near_cutoff_promotions) keep reading a few papers below the floor, so this table can be
redone with fresh evidence.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from paperwatch.config import AppConfig

FLOOR_STEP = 0.01
SAFETY_MARGIN = 0.005
LOW_READING_PRIORITY = 40  # "rated low" in the table
MINIMUM_REVIEWED_PAPERS = 30
MINIMUM_REPORTED_PAPERS = 5
TABLE_FLOOR_COUNT = 10  # floors shown below and around the recommendation


@dataclass(frozen=True)
class ReviewedPaperScores:
    topic_id: str
    paper_id: str
    unnormalized_score: float
    reading_priority: int  # the topic assessor's latest verdict, 0-100


@dataclass(frozen=True)
class FloorEffect:
    floor: float
    reviews_avoided: int
    avoided_rated_low: int
    reported_papers_lost: int


@dataclass(frozen=True)
class TopicCalibration:
    topic_id: str
    reviewed_papers: int
    reported_papers: int
    lowest_reported_unnormalized_score: float | None
    recommended_floor: float | None  # None: not enough evidence for this topic
    floor_effects: tuple[FloorEffect, ...]


def load_reviewed_paper_scores(connection: sqlite3.Connection) -> list[ReviewedPaperScores]:
    """One row per (topic, reviewed paper with a raw screening score): the raw score and the latest verdict."""
    return [
        ReviewedPaperScores(
            topic_id=row["topic_id"],
            paper_id=row["paper_id"],
            unnormalized_score=float(row["unnormalized_score"]),
            reading_priority=int(row["topic_score"]),
        )
        for row in connection.execute(
            """SELECT score.topic_id, score.paper_id, score.unnormalized_score, assessment.topic_score
               FROM candidate_topic_scores AS score
               JOIN reviews AS review ON review.paper_id = score.paper_id
               JOIN review_topic_assessments AS assessment
                 ON assessment.review_id = review.review_id AND assessment.topic_id = score.topic_id
               WHERE score.unnormalized_score IS NOT NULL
                 AND review.review_id = (SELECT newest.review_id FROM reviews AS newest
                                         WHERE newest.paper_id = score.paper_id
                                         ORDER BY newest.completed_at DESC, newest.review_id DESC LIMIT 1)
                 AND assessment.assessed_at = (SELECT MAX(latest.assessed_at) FROM review_topic_assessments AS latest
                                               WHERE latest.review_id = review.review_id
                                                 AND latest.topic_id = score.topic_id)
               ORDER BY score.topic_id, score.paper_id"""
        )
    ]


def calibrate_screening_floor(
    reviewed_papers: list[ReviewedPaperScores], minimum_reading_priority: int
) -> list[TopicCalibration]:
    calibrations = []
    for topic_id in sorted({paper.topic_id for paper in reviewed_papers}):
        topic_papers = [paper for paper in reviewed_papers if paper.topic_id == topic_id]
        reported_scores = [
            paper.unnormalized_score for paper in topic_papers if paper.reading_priority >= minimum_reading_priority
        ]
        lowest_reported = min(reported_scores, default=None)
        recommended_floor = None
        if (
            lowest_reported is not None
            and len(topic_papers) >= MINIMUM_REVIEWED_PAPERS
            and len(reported_scores) >= MINIMUM_REPORTED_PAPERS
        ):
            recommended_floor = round(math.floor((lowest_reported - SAFETY_MARGIN) / FLOOR_STEP) * FLOOR_STEP, 2)
        table_top = recommended_floor if recommended_floor is not None else (lowest_reported or 0.5)
        floors = [
            round(table_top + FLOOR_STEP * (offset - TABLE_FLOOR_COUNT + 3), 2) for offset in range(TABLE_FLOOR_COUNT)
        ]
        calibrations.append(
            TopicCalibration(
                topic_id=topic_id,
                reviewed_papers=len(topic_papers),
                reported_papers=len(reported_scores),
                lowest_reported_unnormalized_score=lowest_reported,
                recommended_floor=recommended_floor,
                floor_effects=tuple(
                    _effect_of_floor(floor, topic_papers, minimum_reading_priority) for floor in floors if floor > 0
                ),
            )
        )
    return calibrations


def _effect_of_floor(
    floor: float, topic_papers: list[ReviewedPaperScores], minimum_reading_priority: int
) -> FloorEffect:
    avoided = [paper for paper in topic_papers if paper.unnormalized_score < floor]
    return FloorEffect(
        floor=floor,
        reviews_avoided=len(avoided),
        avoided_rated_low=sum(paper.reading_priority < LOW_READING_PRIORITY for paper in avoided),
        reported_papers_lost=sum(paper.reading_priority >= minimum_reading_priority for paper in avoided),
    )


def describe_screening_calibration(app_config: AppConfig, calibrations: list[TopicCalibration]) -> list[str]:
    minimum_reading_priority = app_config.report.minimum_reading_priority
    current_floor = app_config.screening.minimum_unnormalized_score
    lines = [
        (
            f"Screening calibration: the raw-score floor (screening.minimum_unnormalized_score, now "
            f"{current_floor:g}), judged by the topic assessor's verdicts; the report shows papers rated "
            f"{minimum_reading_priority}+"
        )
    ]
    if not calibrations:
        return [*lines, "No reviewed paper has a screening score yet: run `collect` and `work` first."]
    for calibration in calibrations:
        lowest_text = (
            f"{calibration.lowest_reported_unnormalized_score:.3f}"
            if calibration.lowest_reported_unnormalized_score is not None
            else "none yet"
        )
        lines += [
            "",
            (
                f"Topic {calibration.topic_id}: {calibration.reviewed_papers} reviewed paper(s), "
                f"{calibration.reported_papers} rated {minimum_reading_priority}+; their lowest raw score: "
                f"{lowest_text}"
            ),
            "  floor  reviews avoided  of them rated below 40  report papers lost",
        ]
        lines += [
            f"  {effect.floor:.2f}  {effect.reviews_avoided:15d}  {effect.avoided_rated_low:22d}  "
            f"{effect.reported_papers_lost:18d}"
            for effect in calibration.floor_effects
        ]
    recommendations = [calibration.recommended_floor for calibration in calibrations]
    if any(recommendation is None for recommendation in recommendations):
        lines += [
            "",
            (
                f"Not enough evidence for a recommendation: each topic needs {MINIMUM_REVIEWED_PAPERS} reviewed "
                f"papers, {MINIMUM_REPORTED_PAPERS} of them rated {minimum_reading_priority}+."
            ),
        ]
    else:
        recommended_floor = min(recommendation for recommendation in recommendations if recommendation is not None)
        lines += [
            "",
            (
                f"Recommended: [screening] minimum_unnormalized_score = {recommended_floor:.2f} (the lowest "
                f"topic's value; it would have lost none of the papers rated {minimum_reading_priority}+)"
            ),
        ]
    return lines
