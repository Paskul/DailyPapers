"""Host-side salvage of an evidence answer that fails only on some of its items ("fix, then drop").

Owner decision, 2026-11-21 (after the first pilot): in the first pilot, 38 of 62 section-reader answers were rejected,
and almost every problem sat in ONE list item. Most items cited a number or a quotation that IS in the paper, but
from a neighbouring passage (a results table split from the sentence that discusses it). Rejecting the whole answer,
and usually its repair as well, threw away about 8 hours of model time. So, before an answer is rejected:

  1. Fix the citation: a finding whose excerpt does not occur in its cited passage is moved to the passage that
     holds it, but only when exactly ONE passage of the input contains the excerpt and backs every number of the
     finding. Nothing else is ever rewritten.
  2. Drop the item: every item of a droppable list (findings, results, assumptions, limitations, inferences,
     flagged passages) that still fails its checks is removed. Its problems are kept for the log.
  3. Re-check: the corrected answer must pass the role's complete host checks again; only then can it be used.

The salvage refuses (returns None) when an answer has a problem outside the droppable lists (e.g. the review's
method statement, a number in the one-sentence summary), or when it contains a web address that is not in the input:
an invented reference may be the trace of an instruction hidden in a paper, so that answer is rejected as a whole.
Whether a salvage is used without a repair call is decided by roles/base.py: when at most half of the items were
dropped; after the repair call, whatever the share.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from paperwatch.reading.evidence_checks import (PassageTextById, check_numbers_occur_in_passages,
                                                normalize_for_matching)
from paperwatch.roles.evidence_fields import ItemCheck, check_finding
from paperwatch.roles.output_guards import find_invented_web_addresses

RoleInput = Mapping[str, Any]


@dataclass(frozen=True)
class DroppableListField:
    """A list field of an answer whose failing items may be dropped one by one."""

    field_name: str
    item_check: ItemCheck
    holds_findings: bool = False  # items are {claim, conditions, passage_id, excerpt}: their citation may be moved


@dataclass(frozen=True)
class AnswerSalvage:
    salvaged_output: dict[str, Any]
    relocated_citations: tuple[str, ...]  # "findings[3]: P57 -> P58"
    dropped_items: tuple[str, ...]  # one line per dropped item, with its first problem
    droppable_item_count: int  # items in the droppable lists of the ORIGINAL answer

    @property
    def dropped_at_most_half(self) -> bool:
        return 2 * len(self.dropped_items) <= self.droppable_item_count

    def describe(self) -> str:
        parts = []
        if self.relocated_citations:
            parts.append(f"moved {len(self.relocated_citations)} citation(s) to the passage that holds the quote "
                         f"({'; '.join(self.relocated_citations)})")
        if self.dropped_items:
            parts.append(f"dropped {len(self.dropped_items)} of {self.droppable_item_count} item(s) that failed the "
                         f"evidence checks ({'; '.join(self.dropped_items)})")
        return "host corrections: " + ("; ".join(parts) if parts else "none")


def salvage_evidence_answer(answer: Mapping[str, Any], role_input: RoleInput, passage_text_by_id: PassageTextById,
                            droppable_fields: tuple[DroppableListField, ...]) -> AnswerSalvage | None:
    """Fix citations, then drop failing items. None when the answer cannot be salvaged (see the module doc). The
    caller still re-runs the role's full checks on `salvaged_output` before using it."""
    if find_invented_web_addresses(answer, role_input):
        return None
    salvaged_output = copy.deepcopy(dict(answer))
    relocated_citations: list[str] = []
    dropped_items: list[str] = []
    droppable_item_count = 0
    for droppable_field in droppable_fields:
        kept_items = []
        for item_index, item in enumerate(salvaged_output.get(droppable_field.field_name, []), start=1):
            droppable_item_count += 1
            where = f"{droppable_field.field_name}[{item_index}]"
            if droppable_field.holds_findings:
                relocated_passage_id = find_passage_that_backs_finding(item, passage_text_by_id)
                if relocated_passage_id is not None and relocated_passage_id != item["passage_id"]:
                    relocated_citations.append(f"{where}: {item['passage_id']} -> {relocated_passage_id}")
                    item = {**item, "passage_id": relocated_passage_id}
            item_problems = droppable_field.item_check(item, passage_text_by_id, where)
            if item_problems:
                dropped_items.append(item_problems[0])
            else:
                kept_items.append(item)
        if droppable_field.field_name in salvaged_output:
            salvaged_output[droppable_field.field_name] = kept_items
    return AnswerSalvage(salvaged_output, tuple(relocated_citations), tuple(dropped_items), droppable_item_count)


def find_passage_that_backs_finding(finding: Mapping[str, Any], passage_text_by_id: PassageTextById) -> str | None:
    """The cited passage if it already passes; otherwise the ONE passage that contains the excerpt and backs every
    number of the claim and conditions. None when no passage, or more than one, qualifies (no guessing)."""
    if not check_finding(finding, passage_text_by_id, "finding"):
        return finding["passage_id"]
    normalized_excerpt = normalize_for_matching(str(finding["excerpt"]).strip(" \"'"))
    if not normalized_excerpt:
        return None
    stated_text = f"{finding['claim']} {finding['conditions']}"
    backing_passage_ids = [
        passage_id for passage_id, passage_text in passage_text_by_id.items()
        if normalized_excerpt in normalize_for_matching(passage_text)
        and not check_numbers_occur_in_passages(stated_text, [passage_id], passage_text_by_id, "finding")]
    if len(backing_passage_ids) != 1:
        return None
    candidate_finding = {**finding, "passage_id": backing_passage_ids[0]}
    return backing_passage_ids[0] if not check_finding(candidate_finding, passage_text_by_id, "finding") else None
