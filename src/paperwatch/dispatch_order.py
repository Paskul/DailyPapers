"""Where a job sits in the worker's queue: its priority class and its order within that class (plan part 6).

| Class | Job                                              | Order within the class (higher first, then queue age) |
| ----- | ------------------------------------------------ | ------------------------------------------------------ |
| 0     | Operator reread (not in the plan's table)        | queue age                                              |
| 1     | Highly relevant new release (7-day fresh window) | screening score                                        |
| 2     | Older paper from the initial topic search        | topic (screening) score                                |
| 3     | Important reference from a completed review      | importance to the parent's argument, then topic score  |
| 4     | Comparison on the same question                  | shared evidence dimensions                             |

A job stores its class plus two descending order values (`within_class_order_primary`, `..._secondary`) in indexed
columns, so one SQL ORDER BY gives the claim order. A new release also stores when its fresh window ends: after that
moment the claim query treats it as class 2 (older literature) with the same score, so a release that waited a week
no longer jumps ahead of older work.

A paper that qualifies for several classes is assigned to the highest applicable one (lowest number):
paperwatch.jobs.upgrade_queued_job_dispatch_order_if_higher moves a queued job up, never down.
Part 8 queues class 3 and 4 work: a research-role selection over a reviewed paper's references (class 3, ahead of
the reference readings it may create, because it is one short call that decides them), the chosen reference
readings (class 3), and comparisons of two reviews (class 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum

from paperwatch.clock import format_utc_timestamp
from paperwatch.publication_dates import (
    NEW_RELEASE_FRESH_WINDOW_DAYS,
    ReleaseClass,
    classify_release,
    earliest_calendar_day,
    normalize_partial_date,
)


class JobPriorityClass(IntEnum):
    """Worker priority classes from plan part 6 (lower number = claimed first)."""

    OPERATOR_REQUEST = 0  # explicit operator reread; not in the plan's table, so it goes ahead of every class
    NEW_RELEASE = 1
    HISTORICAL_SEARCH = 2  # older paper from the topic searches (also a new release whose fresh window ended)
    IMPORTANT_REFERENCE = 3
    COMPARISON = 4


PRIORITY_CLASS_LABELS = {
    JobPriorityClass.OPERATOR_REQUEST: "operator reread",
    JobPriorityClass.NEW_RELEASE: "new release",
    JobPriorityClass.HISTORICAL_SEARCH: "older paper",
    JobPriorityClass.IMPORTANT_REFERENCE: "important reference",
    JobPriorityClass.COMPARISON: "comparison",
}


@dataclass(frozen=True)
class JobDispatchOrder:
    """Claim order of one job. Lower class first; within a class the higher order values first; then queue age."""

    priority_class: JobPriorityClass
    within_class_order_primary: float = 0.0
    within_class_order_secondary: float = 0.0
    new_release_fresh_until: str | None = None  # UTC timestamp; class 1 only. After it the job counts as class 2.

    def is_higher_than(self, other: JobDispatchOrder) -> bool:
        """True if this order would be claimed before `other` (used to move a queued job up, never down)."""
        return (int(self.priority_class), -self.within_class_order_primary, -self.within_class_order_secondary) < (
            int(other.priority_class),
            -other.within_class_order_primary,
            -other.within_class_order_secondary,
        )


def dispatch_order_for_operator_request() -> JobDispatchOrder:
    return JobDispatchOrder(JobPriorityClass.OPERATOR_REQUEST)


def compute_new_release_fresh_until(first_published_date: str) -> datetime:
    """The end of the seven-day fresh window counted from the original release date (the earliest day a coarse date
    can mean, as in publication_dates.classify_release): through the whole seventh day after release."""
    release_day = earliest_calendar_day(first_published_date)
    return datetime(release_day.year, release_day.month, release_day.day, tzinfo=UTC) + timedelta(
        days=NEW_RELEASE_FRESH_WINDOW_DAYS + 1
    )


def dispatch_order_for_promoted_paper(
    first_published_date: str | None, screening_score: float, moment: datetime
) -> JobDispatchOrder:
    """Class 1 if the paper is still inside its fresh window at `moment`, else class 2; ordered by screening score.

    The release class uses the ORIGINAL release date, never the discovery time: a 2018 paper found today is class 2.
    """
    release_date = normalize_partial_date(first_published_date)  # also turns a full timestamp into its day
    if release_date is not None and classify_release(release_date, moment) == ReleaseClass.NEW_RELEASE:
        return JobDispatchOrder(
            JobPriorityClass.NEW_RELEASE,
            within_class_order_primary=screening_score,
            new_release_fresh_until=format_utc_timestamp(compute_new_release_fresh_until(release_date)),
        )
    return dispatch_order_for_older_paper(screening_score)


def dispatch_order_for_older_paper(topic_score: float) -> JobDispatchOrder:
    return JobDispatchOrder(JobPriorityClass.HISTORICAL_SEARCH, within_class_order_primary=topic_score)


def dispatch_order_for_important_reference(
    importance_to_parent_argument: float, topic_score: float
) -> JobDispatchOrder:
    """Part 8: a reference chosen from a completed review. `importance_to_parent_argument` is the research role's
    validated rating (higher = the parent's argument depends on it more)."""
    return JobDispatchOrder(
        JobPriorityClass.IMPORTANT_REFERENCE,
        within_class_order_primary=importance_to_parent_argument,
        within_class_order_secondary=topic_score,
    )


# The research role's selection is ordered before every reference reading of class 3: importance ratings run 1-4.
REFERENCE_SELECTION_ORDER_ABOVE_ALL_IMPORTANCE_RATINGS = 5.0


def dispatch_order_for_reference_selection(parent_topic_score: float) -> JobDispatchOrder:
    """Part 8: the research role chooses which references of a completed review to read. Parents with a higher
    reading priority (0-1) are decided first."""
    return JobDispatchOrder(
        JobPriorityClass.IMPORTANT_REFERENCE,
        within_class_order_primary=REFERENCE_SELECTION_ORDER_ABOVE_ALL_IMPORTANCE_RATINGS,
        within_class_order_secondary=parent_topic_score,
    )


def dispatch_order_for_comparison(shared_evidence_dimension_count: int) -> JobDispatchOrder:
    """Part 8: a comparison of two completed reviews; pairs sharing more evidence dimensions go first."""
    return JobDispatchOrder(
        JobPriorityClass.COMPARISON, within_class_order_primary=float(shared_evidence_dimension_count)
    )
