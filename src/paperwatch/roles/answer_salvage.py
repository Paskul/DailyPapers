"""Host-side salvage of an evidence answer that fails only on some of its items ("fix, then drop").

Owner decision, 2026-11-21 (after the first pilot): in the first pilot, 38 of 62 section-reader answers were rejected,
and almost every problem sat in ONE list item. Most items cited a number or a quotation that IS in the paper, but
from a neighbouring passage (a results table split from the sentence that discusses it). Rejecting the whole answer,
and usually its repair as well, threw away about 8 hours of model time. So, before an answer is rejected:

  1. Fix the citation, never the text. Three corrections exist, each logged:
     a. split joined passage ids: a list entry such as "P18P19", "'P19', 'P20'" or "P35-P37" becomes the ids it
        names, but only when every one of them is a passage of the input (a range spans at most
        MAXIMUM_PASSAGES_IN_A_RANGE ids). Second pilot, 2026-11-22: 13 of 47 rejected reviews had such an entry,
        usually in the method statement, whose cap of 6 passages the model tried to exceed;
     b. move a finding: a finding whose excerpt does not occur in its cited passage is moved to the passage that
        holds it, but only when exactly ONE passage of the input contains the excerpt and backs every number;
     c. add the passage that holds a number: a statement (no excerpt) whose number occurs in none of its cited
        passages gets the ONE passage of the input that states that number, when there is exactly one and the
        statement's cap allows it. Second pilot: 43 of 47 rejected reviews stated a number from an uncited passage,
        often one the section notes' uncited summary had mentioned.
  2. Drop the item: every item of a droppable list (findings, results, assumptions, limitations, inferences,
     flagged passages) that still fails its checks is removed. Its problems are kept for the log.
  3. Re-check: the corrected answer must pass the role's complete host checks again; only then can it be used.

Single statements (the review's research question, method, evaluation setting) get the citation corrections but are
never dropped: when one still fails, the salvage's re-check fails, and the answer needs the repair call.
The salvage refuses (returns None) when the answer contains a web address that is not in the input: an invented
reference may be the trace of an instruction hidden in a paper, so that answer is rejected as a whole.
Whether a salvage is used without a repair call is decided by roles/base.py: when at most half of the items were
dropped; after the repair call, whatever the share.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from paperwatch.reading.evidence_checks import (
    PassageTextById,
    check_numbers_occur_in_passages,
    find_numbers_backed_by_passage,
    find_numbers_needing_evidence,
    normalize_for_matching,
)
from paperwatch.roles.evidence_fields import MAXIMUM_PASSAGES_PER_STATEMENT, ItemCheck, check_finding
from paperwatch.roles.output_guards import find_invented_web_addresses

RoleInput = Mapping[str, Any]

MAXIMUM_PASSAGES_IN_A_RANGE = 10  # "P35-P37" is split; "P1-P300" is not a citation
PASSAGE_ID_LIST_SEPARATOR_PATTERN = re.compile(r"[\s,;'\"\[\]()]+|\band\b")
PASSAGE_ID_RANGE_PATTERN = re.compile(r"^([APT])(\d+)[-–—]([APT])?(\d+)$")
JOINED_PASSAGE_IDS_PATTERN = re.compile(r"^(?:[APT]\d+)+$")
SINGLE_PASSAGE_ID_PATTERN = re.compile(r"[APT]\d+")
TEXT_FIELD_OF_A_STATEMENT = ("statement", "inference")  # the field whose numbers a statement-like item must back


@dataclass(frozen=True)
class DroppableListField:
    """A list field of an answer whose failing items may be dropped one by one."""

    field_name: str
    item_check: ItemCheck
    holds_findings: bool = False  # items are {claim, conditions, passage_id, excerpt}: their citation may be moved
    maximum_cited_passages: int = MAXIMUM_PASSAGES_PER_STATEMENT  # the schema's cap on supporting_passage_ids


@dataclass(frozen=True)
class SingleStatementField:
    """A required {statement, evidence_status, supporting_passage_ids} field: citations may be fixed, never dropped."""

    field_name: str
    maximum_cited_passages: int


@dataclass(frozen=True)
class AnswerSalvage:
    salvaged_output: dict[str, Any]
    relocated_citations: tuple[str, ...]  # "findings[3]: P57 -> P58"
    dropped_items: tuple[str, ...]  # one line per dropped item, with its first problem
    droppable_item_count: int  # items in the droppable lists of the ORIGINAL answer
    split_passage_ids: tuple[str, ...] = ()  # "method: 'P18P19' -> P18, P19"
    added_citations: tuple[str, ...] = ()  # "method: + P41 (states 16)"

    @property
    def dropped_at_most_half(self) -> bool:
        return 2 * len(self.dropped_items) <= self.droppable_item_count

    def describe(self) -> str:
        parts = []
        if self.split_passage_ids:
            parts.append(
                f"split {len(self.split_passage_ids)} joined passage id(s) ({'; '.join(self.split_passage_ids)})"
            )
        if self.relocated_citations:
            parts.append(
                f"moved {len(self.relocated_citations)} citation(s) to the passage that holds the quote "
                f"({'; '.join(self.relocated_citations)})"
            )
        if self.added_citations:
            parts.append(
                f"added {len(self.added_citations)} citation(s) of the one passage that states a number "
                f"({'; '.join(self.added_citations)})"
            )
        if self.dropped_items:
            parts.append(
                f"dropped {len(self.dropped_items)} of {self.droppable_item_count} item(s) that failed the "
                f"evidence checks ({'; '.join(self.dropped_items)})"
            )
        return "host corrections: " + ("; ".join(parts) if parts else "none")


@dataclass
class _CorrectionLog:
    split_passage_ids: list[str] = field(default_factory=list)
    relocated_citations: list[str] = field(default_factory=list)
    added_citations: list[str] = field(default_factory=list)
    dropped_items: list[str] = field(default_factory=list)


def salvage_evidence_answer(
    answer: Mapping[str, Any],
    role_input: RoleInput,
    passage_text_by_id: PassageTextById,
    droppable_fields: tuple[DroppableListField, ...],
    single_statement_fields: tuple[SingleStatementField, ...] = (),
) -> AnswerSalvage | None:
    """Fix citations, then drop failing items. None when the answer cannot be salvaged (see the module doc). The
    caller still re-runs the role's full checks on `salvaged_output` before using it."""
    if find_invented_web_addresses(answer, role_input):
        return None
    salvaged_output = copy.deepcopy(dict(answer))
    numbers_by_passage_id = {
        passage_id: find_numbers_backed_by_passage(passage_text)
        for passage_id, passage_text in passage_text_by_id.items()
    }
    corrections = _CorrectionLog()

    for single_statement in single_statement_fields:
        statement = salvaged_output.get(single_statement.field_name)
        if isinstance(statement, dict):
            salvaged_output[single_statement.field_name] = _fix_statement_citations(
                statement,
                single_statement.field_name,
                single_statement.maximum_cited_passages,
                passage_text_by_id,
                numbers_by_passage_id,
                corrections,
            )

    droppable_item_count = 0
    for droppable_field in droppable_fields:
        kept_items = []
        for item_index, original_item in enumerate(salvaged_output.get(droppable_field.field_name, []), start=1):
            droppable_item_count += 1
            where = f"{droppable_field.field_name}[{item_index}]"
            item = original_item
            if droppable_field.holds_findings:
                relocated_passage_id = find_passage_that_backs_finding(item, passage_text_by_id)
                if relocated_passage_id is not None and relocated_passage_id != item["passage_id"]:
                    corrections.relocated_citations.append(f"{where}: {item['passage_id']} -> {relocated_passage_id}")
                    item = {**item, "passage_id": relocated_passage_id}
            elif "supporting_passage_ids" in item:
                item = _fix_statement_citations(
                    item,
                    where,
                    droppable_field.maximum_cited_passages,
                    passage_text_by_id,
                    numbers_by_passage_id,
                    corrections,
                )
            item_problems = droppable_field.item_check(item, passage_text_by_id, where)
            if item_problems:
                corrections.dropped_items.append(item_problems[0])
            else:
                kept_items.append(item)
        if droppable_field.field_name in salvaged_output:
            salvaged_output[droppable_field.field_name] = kept_items
    return AnswerSalvage(
        salvaged_output,
        tuple(corrections.relocated_citations),
        tuple(corrections.dropped_items),
        droppable_item_count,
        tuple(corrections.split_passage_ids),
        tuple(corrections.added_citations),
    )


# --------------------------------------------------------------------------------------------------------------------
# Citation fixes
# --------------------------------------------------------------------------------------------------------------------


def _fix_statement_citations(
    statement: dict[str, Any],
    where: str,
    maximum_cited_passages: int,
    passage_text_by_id: PassageTextById,
    numbers_by_passage_id: Mapping[str, set[str]],
    corrections: _CorrectionLog,
) -> dict[str, Any]:
    """Split joined ids, then add the one passage that states each uncited number. Returns a corrected copy."""
    cited_passage_ids: list[str] = []
    for cited_entry in statement["supporting_passage_ids"]:
        split_ids = split_joined_passage_ids(cited_entry, passage_text_by_id)
        if split_ids is None:
            cited_passage_ids.append(cited_entry)  # unknown and not splittable: the check reports it
            continue
        if split_ids != [cited_entry]:
            corrections.split_passage_ids.append(f"{where}: '{cited_entry}' -> {', '.join(split_ids)}")
        cited_passage_ids += split_ids
    cited_passage_ids = list(dict.fromkeys(cited_passage_ids))
    if len(cited_passage_ids) > maximum_cited_passages:
        return {**statement, "supporting_passage_ids": cited_passage_ids}  # over the cap: the schema check rejects it

    statement_text = next(
        (str(statement[text_field]) for text_field in TEXT_FIELD_OF_A_STATEMENT if text_field in statement), ""
    )
    added_passage_ids = find_passages_that_state_uncited_numbers(
        statement_text, cited_passage_ids, passage_text_by_id, numbers_by_passage_id
    )
    if added_passage_ids and len(cited_passage_ids) + len(added_passage_ids) <= maximum_cited_passages:
        for passage_id, numbers in added_passage_ids.items():
            corrections.added_citations.append(f"{where}: + {passage_id} (states {', '.join(numbers)})")
        cited_passage_ids += list(added_passage_ids)
    return {**statement, "supporting_passage_ids": cited_passage_ids}


def split_joined_passage_ids(cited_entry: str, passage_text_by_id: PassageTextById) -> list[str] | None:
    """The input passage ids one list entry names: itself if it is one; the parts of "P18P19", "'P19', 'P20'",
    "P16, P17" or a short range "P35-P37". None when any part is not a passage of the input (no guessing)."""
    if cited_entry in passage_text_by_id:
        return [cited_entry]
    named_passage_ids: list[str] = []
    for token in PASSAGE_ID_LIST_SEPARATOR_PATTERN.split(cited_entry):
        if not token:
            continue
        if token in passage_text_by_id:
            named_passage_ids.append(token)
            continue
        range_ids = _expand_passage_id_range(token)
        if range_ids is not None:
            parts = range_ids
        elif JOINED_PASSAGE_IDS_PATTERN.fullmatch(token):
            parts = SINGLE_PASSAGE_ID_PATTERN.findall(token)
        else:
            return None
        if not parts or any(part not in passage_text_by_id for part in parts):
            return None
        named_passage_ids += parts
    return list(dict.fromkeys(named_passage_ids)) or None


def _expand_passage_id_range(token: str) -> list[str] | None:
    range_match = PASSAGE_ID_RANGE_PATTERN.fullmatch(token)
    if range_match is None:
        return None
    prefix, first_text, end_prefix, last_text = range_match.groups()
    first_number, last_number = int(first_text), int(last_text)
    if (end_prefix not in (None, prefix)) or not 0 < last_number - first_number < MAXIMUM_PASSAGES_IN_A_RANGE:
        return None
    return [f"{prefix}{number}" for number in range(first_number, last_number + 1)]


def find_passages_that_state_uncited_numbers(
    statement_text: str,
    cited_passage_ids: list[str],
    passage_text_by_id: PassageTextById,
    numbers_by_passage_id: Mapping[str, set[str]],
) -> dict[str, list[str]]:
    """{passage id to add: the numbers it states} so that every number of the statement is backed, or {} when that is
    not possible: a cited id is unknown, or some uncited number is stated by no passage or by more than one."""
    if any(passage_id not in passage_text_by_id for passage_id in cited_passage_ids):
        return {}
    if not check_numbers_occur_in_passages(statement_text, cited_passage_ids, passage_text_by_id, "statement"):
        return {}
    cited_numbers = set().union(*(numbers_by_passage_id[passage_id] for passage_id in cited_passage_ids))
    added_passage_ids: dict[str, list[str]] = {}
    for number in find_numbers_needing_evidence(statement_text):
        if number in cited_numbers or number.lstrip("+-") in cited_numbers:
            continue
        stating_passage_ids = [
            passage_id
            for passage_id, passage_numbers in numbers_by_passage_id.items()
            if number in passage_numbers or number.lstrip("+-") in passage_numbers
        ]
        if len(stating_passage_ids) != 1:
            return {}
        added_passage_ids.setdefault(stating_passage_ids[0], []).append(number)
    return added_passage_ids


def find_passage_that_backs_finding(finding: Mapping[str, Any], passage_text_by_id: PassageTextById) -> str | None:
    """The cited passage if it already passes; otherwise the ONE passage that contains the excerpt and backs every
    number of the claim and conditions. None when no passage, or more than one, qualifies (no guessing)."""
    if not check_finding(finding, passage_text_by_id, "finding"):
        return str(finding["passage_id"])
    normalized_excerpt = normalize_for_matching(str(finding["excerpt"]).strip(" \"'"))
    if not normalized_excerpt:
        return None
    stated_text = f"{finding['claim']} {finding['conditions']}"
    backing_passage_ids = [
        passage_id
        for passage_id, passage_text in passage_text_by_id.items()
        if normalized_excerpt in normalize_for_matching(passage_text)
        and not check_numbers_occur_in_passages(stated_text, [passage_id], passage_text_by_id, "finding")
    ]
    if len(backing_passage_ids) != 1:
        return None
    candidate_finding = {**finding, "passage_id": backing_passage_ids[0]}
    return backing_passage_ids[0] if not check_finding(candidate_finding, passage_text_by_id, "finding") else None
