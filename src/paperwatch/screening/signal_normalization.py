"""Put every screening signal on the same scale before combining them: its percentile within the topic's pool.

Why: the raw signals do not share a scale. Embedding cosines bunch up in a narrow band that depends on the model and
on how a topic is worded (bge-base gave 0.53-0.75 for one live topic and 0.43-0.47 for another), while the phrase
signal is mostly exactly 0 or 1. A fixed weighted sum of raw values therefore lets the phrase signal swing a score
more than its weight says, and makes one topic's scores look better than another's for reasons unrelated to
relevance. That matters wherever topics are compared: the best-topic choice, the spare slots, the near-cutoff picks.

How: for each topic and each available signal, every candidate in the CURRENT POOL (the unpromoted candidates of this
run) gets the mid-rank percentile of its raw value among the pool's values for that topic:

    percentile = (number of candidates with a lower value + 0.5 × number with an equal value) / pool size

so the median candidate gets about 0.5, ties share one value (every candidate without a phrase match gets the same
low-ish percentile), and a lone candidate gets 0.5. The percentiles are then combined with the configured weights by
the same formula as before (screening_score.combine_screening_signals): unavailable signals are still omitted.

Consequences, by design:
  * within one signal the ranking is unchanged (a seeds-only topic ranks exactly as before);
  * a topic's scores mean "position among today's candidates for this topic", so topics are comparable;
  * the normalized score depends on the pool, so it is computed at promotion time and never cached. The RAW signals
    stay stored and cached (candidate_topic_scores), and the promotion row keeps both scores plus the pool size;
  * `minimum_screening_score` therefore reads as a relative position (0.8 = top fifth of the pool). An absolute
    floor, if part 10 wants one, belongs on the raw signals.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import replace

from paperwatch.screening.screening_score import ScreeningSignals, SignalWeights, combine_screening_signals

SIGNAL_FIELD_NAMES = ("question_similarity", "seed_similarity", "phrase_match_fraction")


def midrank_percentiles(raw_values: list[float]) -> list[float]:
    """The mid-rank percentile of each value among all of them, in input order (see the module docstring)."""
    sorted_values = sorted(raw_values)
    pool_size = len(sorted_values)
    return [
        (
            bisect_left(sorted_values, value)
            + 0.5 * (bisect_right(sorted_values, value) - bisect_left(sorted_values, value))
        )
        / pool_size
        for value in raw_values
    ]


def normalize_topic_signals_within_pool(
    signals_by_paper_id: dict[str, ScreeningSignals],
) -> dict[str, ScreeningSignals]:
    """One topic: replace each available raw signal by its percentile among the papers that have that signal."""
    normalized = dict(signals_by_paper_id)
    for field_name in SIGNAL_FIELD_NAMES:
        papers_with_signal = [
            paper_id for paper_id, signals in signals_by_paper_id.items() if getattr(signals, field_name) is not None
        ]
        if not papers_with_signal:
            continue
        percentiles = midrank_percentiles(
            [getattr(signals_by_paper_id[paper_id], field_name) for paper_id in papers_with_signal]
        )
        for paper_id, percentile in zip(papers_with_signal, percentiles):
            normalized[paper_id] = replace(normalized[paper_id], **{field_name: percentile})
    return normalized


def compute_pool_normalized_scores(
    signals_by_paper_and_topic: dict[str, dict[str, ScreeningSignals]], weights: SignalWeights
) -> dict[str, dict[str, float]]:
    """{paper id: {topic id: normalized screening score}} for every (paper, topic) pair the pool holds."""
    signals_by_topic: dict[str, dict[str, ScreeningSignals]] = {}
    for paper_id, signals_by_topic_id in signals_by_paper_and_topic.items():
        for topic_id, signals in signals_by_topic_id.items():
            signals_by_topic.setdefault(topic_id, {})[paper_id] = signals
    normalized_scores: dict[str, dict[str, float]] = {}
    for topic_id, topic_signals in signals_by_topic.items():
        for paper_id, normalized_signals in normalize_topic_signals_within_pool(topic_signals).items():
            combined_score = combine_screening_signals(normalized_signals, weights)
            if combined_score is not None:
                normalized_scores.setdefault(paper_id, {})[topic_id] = combined_score
    return normalized_scores
