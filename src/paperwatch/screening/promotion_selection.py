"""Which screened candidates get today's review slots: a pure function of scores and remaining slots (plan part 6).

Every topic is active and none is primary (owner decision), so slots are divided across topics BEFORE score decides:
  1. Topic shares. The day's regular slots (daily limit minus the near-cutoff picks) are split evenly across the
     active topics; each topic's share for this run is its daily share minus what it already took today. Topics pick
     in turns (round robin), each taking its highest-scoring remaining candidate whose score FOR THAT TOPIC reaches
     the minimum, so a paper wanted by two topics is taken by the first to reach it, not counted twice.
  2. Spare slots. Regular slots a share could not use (too few qualifying candidates, or the remainder of the even
     split) are filled by the best remaining candidates by their best-topic score, again above the minimum.
  3. Near the cutoff. Up to `near_cutoff_slots` more go to the best candidates that were NOT selected, whatever the
     minimum: they sit just below the cutoff and show what the filter misses (part 10 calibrates with them).
Ties are broken by paper id, so the same inputs always give the same selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PromotionReason(StrEnum):
    TOPIC_SHARE = "topic_share"
    SPARE_SLOT = "spare_slot"
    NEAR_CUTOFF = "near_cutoff"


@dataclass(frozen=True)
class PromotionCandidate:
    paper_id: str
    score_by_topic_id: dict[str, float]

    @property
    def best_topic_id(self) -> str:
        return min(self.score_by_topic_id, key=lambda topic_id: (-self.score_by_topic_id[topic_id], topic_id))

    @property
    def best_score(self) -> float:
        return self.score_by_topic_id[self.best_topic_id]


@dataclass(frozen=True)
class PromotionDecision:
    paper_id: str
    promotion_reason: PromotionReason
    selecting_topic_id: str  # the topic whose share or score selected the paper
    best_topic_id: str
    screening_score: float  # best-topic score


@dataclass(frozen=True)
class SlotsAvailable:
    """What is left of today's slots (the caller derives it from today's promotions)."""

    remaining_share_by_topic_id: dict[str, int]  # per active topic, may be 0
    remaining_regular_slots: int  # shares + spare, for this run
    remaining_near_cutoff_slots: int


def compute_slots_available(*, active_topic_ids: tuple[str, ...], daily_promotion_limit: int,
                            near_cutoff_promotions: int, regular_promotions_today: int,
                            near_cutoff_promotions_today: int,
                            topic_share_promotions_today_by_topic_id: dict[str, int]) -> SlotsAvailable:
    daily_regular_slots = daily_promotion_limit - near_cutoff_promotions
    remaining_regular_slots = max(0, daily_regular_slots - regular_promotions_today)
    daily_share_per_topic = daily_regular_slots // len(active_topic_ids) if active_topic_ids else 0
    remaining_share_by_topic_id = {
        topic_id: max(0, daily_share_per_topic - topic_share_promotions_today_by_topic_id.get(topic_id, 0))
        for topic_id in active_topic_ids}
    return SlotsAvailable(remaining_share_by_topic_id=remaining_share_by_topic_id,
                          remaining_regular_slots=remaining_regular_slots,
                          remaining_near_cutoff_slots=max(0, near_cutoff_promotions - near_cutoff_promotions_today))


def select_papers_for_promotion(candidates: list[PromotionCandidate], slots: SlotsAvailable,
                                minimum_screening_score: float) -> list[PromotionDecision]:
    scored_candidates = [candidate for candidate in candidates if candidate.score_by_topic_id]
    decisions: list[PromotionDecision] = []
    selected_paper_ids: set[str] = set()

    def select(candidate: PromotionCandidate, reason: PromotionReason, selecting_topic_id: str) -> None:
        decisions.append(PromotionDecision(candidate.paper_id, reason, selecting_topic_id, candidate.best_topic_id,
                                           candidate.best_score))
        selected_paper_ids.add(candidate.paper_id)

    # 1. Topic shares, in turns.
    ranked_by_topic = {
        topic_id: sorted((candidate for candidate in scored_candidates
                          if candidate.score_by_topic_id.get(topic_id, -1.0) >= minimum_screening_score),
                         key=lambda candidate, topic_id=topic_id: (-candidate.score_by_topic_id[topic_id],
                                                                   candidate.paper_id))
        for topic_id in slots.remaining_share_by_topic_id}
    share_left = dict(slots.remaining_share_by_topic_id)
    regular_slots_left = slots.remaining_regular_slots
    next_position = {topic_id: 0 for topic_id in ranked_by_topic}
    while regular_slots_left > 0:
        picked_in_this_turn = False
        for topic_id, topic_ranking in ranked_by_topic.items():
            if share_left[topic_id] == 0 or regular_slots_left == 0:
                continue
            while next_position[topic_id] < len(topic_ranking) and \
                    topic_ranking[next_position[topic_id]].paper_id in selected_paper_ids:
                next_position[topic_id] += 1
            if next_position[topic_id] == len(topic_ranking):
                share_left[topic_id] = 0  # this topic has no qualifying candidate left: its share becomes spare
                continue
            select(topic_ranking[next_position[topic_id]], PromotionReason.TOPIC_SHARE, topic_id)
            share_left[topic_id] -= 1
            regular_slots_left -= 1
            picked_in_this_turn = True
        if not picked_in_this_turn:
            break

    # 2. Spare slots by best-topic score.
    ranked_by_best_score = sorted(scored_candidates, key=lambda candidate: (-candidate.best_score, candidate.paper_id))
    for candidate in ranked_by_best_score:
        if regular_slots_left == 0:
            break
        if candidate.paper_id not in selected_paper_ids and candidate.best_score >= minimum_screening_score:
            select(candidate, PromotionReason.SPARE_SLOT, candidate.best_topic_id)
            regular_slots_left -= 1

    # 3. The best of the rest, just below the cutoff.
    near_cutoff_slots_left = slots.remaining_near_cutoff_slots
    for candidate in ranked_by_best_score:
        if near_cutoff_slots_left == 0:
            break
        if candidate.paper_id not in selected_paper_ids:
            select(candidate, PromotionReason.NEAR_CUTOFF, candidate.best_topic_id)
            near_cutoff_slots_left -= 1
    return decisions
