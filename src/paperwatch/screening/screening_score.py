"""The screening formula (plan part 6): pure functions, no database, no model.

For one candidate and one topic, three signals on a 0-1 scale:
    question similarity  normalize(cosine(paper, topic question))
    seed similarity      mean of normalize(cosine(paper, seed)) over the `closest_seed_count` most similar seeds
                         ("the three closest AVAILABLE seeds": fewer when the topic has fewer)
    phrase match         matched approved phrases / approved phrases
with the documented normalization normalize(cosine) = max(0, cosine). A signal the topic cannot provide (no question
text, no active seeds, no phrases) is omitted, and the score divides by the sum of the ACTIVE weights:
    score = sum(weight_i * signal_i) / sum(weight_i over available signals)
A topic with no available signal at all cannot score the paper (None).

The same combination runs twice: on the RAW signals (stored as `unnormalized_score`, for inspection and calibration),
and at promotion on each signal's percentile within the topic's pool (signal_normalization.py). Promotion uses the
second, because raw cosines and the 0/1 phrase signal do not share a scale.

The screening score selects papers for reading only. The report ranks by the evidence-based score of part 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paperwatch.config import ScreeningSettings


def normalize_cosine_similarity(cosine_similarity: float) -> float:
    """The documented normalization: negative cosines (opposed directions) count as no similarity."""
    return max(0.0, min(1.0, float(cosine_similarity)))


@dataclass(frozen=True)
class SignalWeights:
    question_similarity: float
    seed_similarity: float
    phrase_match: float

    @classmethod
    def from_settings(cls, screening_settings: ScreeningSettings) -> SignalWeights:
        return cls(
            screening_settings.question_similarity_weight,
            screening_settings.seed_similarity_weight,
            screening_settings.phrase_match_weight,
        )


@dataclass(frozen=True)
class ScreeningSignals:
    """The component values of one topic score; None = unavailable for this topic (omitted, not zero)."""

    question_similarity: float | None
    seed_similarity: float | None
    phrase_match_fraction: float | None
    closest_seed_ids: tuple[str, ...] = ()
    matched_phrases: tuple[str, ...] = ()


def combine_screening_signals(signals: ScreeningSignals, weights: SignalWeights) -> float | None:
    """Weighted mean of the available signals, renormalized by their weights; None if no signal is available."""
    weighted_signals = [
        (weights.question_similarity, signals.question_similarity),
        (weights.seed_similarity, signals.seed_similarity),
        (weights.phrase_match, signals.phrase_match_fraction),
    ]
    active_pairs = [(weight, signal) for weight, signal in weighted_signals if signal is not None and weight > 0]
    active_weight_sum = sum(weight for weight, _signal in active_pairs)
    if active_weight_sum <= 0:
        return None
    return sum(weight * signal for weight, signal in active_pairs) / active_weight_sum


def closest_seed_similarity(
    cosines_to_seeds: np.ndarray, seed_ids: tuple[str, ...], closest_seed_count: int
) -> tuple[float | None, tuple[str, ...]]:
    """Mean normalized similarity of the closest seeds and their ids (most similar first); (None, ()) without seeds.

    Ties are broken by seed id, so the same inputs always name the same seeds.
    """
    if len(seed_ids) == 0:
        return None, ()
    normalized = [normalize_cosine_similarity(cosine) for cosine in cosines_to_seeds]
    ranked = sorted(zip(normalized, seed_ids), key=lambda pair: (-pair[0], pair[1]))[:closest_seed_count]
    return sum(similarity for similarity, _seed_id in ranked) / len(ranked), tuple(seed_id for _s, seed_id in ranked)
