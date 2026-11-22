"""The reading-priority rubric. The model proposes an integer rating 0-4 per dimension and explains
it; this module validates nothing about the model and computes the score itself.

| Dimension          | Weight | 0 / 2 / 4                                                                          |
| question_fit       |     40 | unrelated / addresses part of it / studies the question directly                   |
| method_fit         |     25 | incompatible / useful with changes / directly applicable under stated assumptions  |
| evidence_strength  |     20 | unsupported / partial support / direct proof or evaluation under stated conditions |
| relative_value     |     15 | repeats a known point / adds a useful difference / changes a concrete design       |
|                    |        | choice or supplies a missing comparison                                            |

score = round_half_up(100 * sum(weight * rating / 4) / sum(active weights)); e.g. 4, 3, 2, 3 gives 80/100.
  * relative_value is omitted ("comparison unavailable") when no comparable saved paper (seed or earlier review on
    the same question) was available to the assessment; its weight then leaves the denominator.
  * evidence_strength is capped at 1 for an abstract-only reading: the abstract alone cannot show direct evidence.
The score is a READING PRIORITY for the user's question, not a judgement of the paper's novelty in its field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RUBRIC_VERSION = "rubric-v1"
SCORE_SCALE_PRIORITY_0_TO_100 = "priority_0_to_100"
MAXIMUM_RATING = 4
ABSTRACT_ONLY_EVIDENCE_STRENGTH_CAP = 1


@dataclass(frozen=True)
class RubricDimension:
    dimension_name: str
    weight: int
    label: str
    meaning_of_0_2_4: str


QUESTION_FIT = RubricDimension(
    "question_fit",
    40,
    "Fit to the research question",
    "0 unrelated / 2 addresses part of it / 4 studies the question directly",
)
METHOD_FIT = RubricDimension(
    "method_fit",
    25,
    "Fit of method or assumptions",
    "0 incompatible / 2 useful with changes / 4 directly applicable under stated assumptions",
)
EVIDENCE_STRENGTH = RubricDimension(
    "evidence_strength",
    20,
    "Strength of evidence available to this review",
    "0 unsupported / 2 partial support / 4 direct proof or evaluation under stated conditions",
)
RELATIVE_VALUE = RubricDimension(
    "relative_value",
    15,
    "Value relative to nearby saved papers",
    "0 repeats a known point / 2 adds a useful difference / 4 changes a concrete design "
    "choice or supplies a missing comparison",
)
RUBRIC_DIMENSIONS: tuple[RubricDimension, ...] = (QUESTION_FIT, METHOD_FIT, EVIDENCE_STRENGTH, RELATIVE_VALUE)
RUBRIC_DIMENSION_BY_NAME = {dimension.dimension_name: dimension for dimension in RUBRIC_DIMENSIONS}


@dataclass(frozen=True)
class ScoredTopicAssessment:
    topic_id: str
    reading_priority_score: int  # 0-100
    applied_ratings: dict[str, int]  # after caps; only the active dimensions
    proposed_ratings: dict[str, int]  # as the model proposed them
    comparison_available: bool
    evidence_strength_capped: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "rubric_version": RUBRIC_VERSION,
            "reading_priority_score": self.reading_priority_score,
            "applied_ratings": self.applied_ratings,
            "proposed_ratings": self.proposed_ratings,
            "comparison_available": self.comparison_available,
            "evidence_strength_capped_for_abstract_only": self.evidence_strength_capped,
        }


def score_topic_assessment(
    topic_id: str, proposed_ratings: dict[str, int], *, comparison_available: bool, is_abstract_only: bool
) -> ScoredTopicAssessment:
    """Apply the caps and compute the 0-100 score from the model's proposed ratings."""
    applied_ratings: dict[str, int] = {}
    for dimension in RUBRIC_DIMENSIONS:
        if dimension is RELATIVE_VALUE and not comparison_available:
            continue
        applied_ratings[dimension.dimension_name] = min(
            max(proposed_ratings[dimension.dimension_name], 0), MAXIMUM_RATING
        )
    evidence_strength_capped = False
    if is_abstract_only and applied_ratings[EVIDENCE_STRENGTH.dimension_name] > ABSTRACT_ONLY_EVIDENCE_STRENGTH_CAP:
        applied_ratings[EVIDENCE_STRENGTH.dimension_name] = ABSTRACT_ONLY_EVIDENCE_STRENGTH_CAP
        evidence_strength_capped = True
    return ScoredTopicAssessment(
        topic_id,
        compute_reading_priority_score(applied_ratings),
        applied_ratings,
        dict(proposed_ratings),
        comparison_available,
        evidence_strength_capped,
    )


def compute_reading_priority_score(applied_ratings: dict[str, int]) -> int:
    """round_half_up(100 * sum(weight * rating / 4) / sum(weights)), in integer arithmetic so no float edge case
    (79.4999...) can flip a score. Python's round() would round halves to even, which the plan does not intend."""
    numerator = 100 * sum(RUBRIC_DIMENSION_BY_NAME[name].weight * rating for name, rating in applied_ratings.items())
    denominator = MAXIMUM_RATING * sum(RUBRIC_DIMENSION_BY_NAME[name].weight for name in applied_ratings)
    return (2 * numerator + denominator) // (2 * denominator)


LEGACY_RELEVANCE_SCALE = "rating_0_to_4"  # the provisional step-1 reviewer stored one 0-4 relevance rating


def question_fit_from_stored_assessment(score_scale: str, topic_score: float, assessment_record: dict[str, Any]) -> int:
    """The applied question_fit rating (0-4) of a saved topic assessment. A legacy 0-4 relevance rating counts as
    question fit. Used to decide which reviews address a question (comparison anchors, references, comparisons)."""
    if score_scale == LEGACY_RELEVANCE_SCALE:
        return int(topic_score)
    applied_ratings = assessment_record.get("scoring", {}).get("applied_ratings", {})
    return int(applied_ratings.get(QUESTION_FIT.dimension_name, 0))
