"""Host-side salvage of an evidence answer that fails only on some of its items ("fix, then drop").

Owner decision, 2026-11-21 (after the first pilot): in the first pilot, 38 of 62 section-reader answers were rejected,
and almost every problem sat in ONE list item. Most items cited a number or a quotation that IS in the paper, but
from a neighbouring passage (a results table split from the sentence that discusses it). Rejecting the whole answer,
and usually its repair as well, threw away about 8 hours of model time. So, before an answer is rejected:

  1. Fix the citation, never the text. The corrections, each logged:
     a. split joined passage ids: a list entry such as "P18P19", "'P19', 'P20'" or "P35-P37" becomes the ids it
        names, but only when every one of them is a passage of the input (a range spans at most
        MAXIMUM_PASSAGES_IN_A_RANGE ids). Second pilot, 2026-11-22: 13 of 47 rejected reviews had such an entry,
        usually in the method statement, whose cap of 6 passages the model tried to exceed. A finding's single
        passage_id written as "P41, P42" keeps the part that holds its quotation;
     b. move a finding: a finding whose excerpt does not occur in its cited passage is moved to the passage that
        holds it, when exactly ONE passage of the input contains the excerpt (or, among several, exactly one also
        states the claim's numbers); a quotation is never moved by any other tie-break;
     c. add the passage that states a number: a statement whose number occurs in none of its cited passages gets
        the passage that states it (within the statement's cap), and a finding whose quotation is right but whose
        numbers are in another passage gets that passage in its number_passage_ids. Third pilot, 2026-11-23: 117
        results and 45 statements stated a number from a passage they did not cite, and ~90% of those numbers do
        occur in the paper (a table beside the sentence, or a passage the section notes cited). Which passage is
        chosen without guessing (choose_passages_for_numbers):
          - the ONE passage of the input that states every missing number;
          - among several, the rules below narrow the candidates in turn until one is left (a rule that would
            leave none is skipped): the passage the answer itself named for this item; the passages whose words
            around the number match the statement's (at least CLEAR_CONTEXT_WORDS shared words: "108 total
            trials"); the passages the section notes cited for those numbers (where the reviewer read them); the
            passages next to a cited passage (within NEIGHBOURING_PASSAGE_DISTANCE: a table beside its
            discussion); and finally the passage that shares the most words around the number;
            when several passages are left that all describe the number with a word the statement also uses
            (one sentence repeated in the abstract and the introduction), the first of them. When several
            passages state a small whole number ("60"), only those the answer named, the notes cited, or whose
            words around it agree with the statement can be chosen at all;
          - otherwise one passage per number by the same rules;
        when the rules leave more than one passage (or none), nothing is added and the item fails its check.
  2. Drop the item: every item of a droppable list (findings, results, assumptions, limitations, inferences,
     flagged passages) that still fails its checks is removed. Its problems are kept for the log.
  3. Re-check: the corrected answer must pass the role's complete host checks again; only then can it be used.

Single statements (the review's research question, method, evaluation setting) get the citation corrections but are
never dropped: when one still fails, the salvage's re-check fails, and the answer needs the repair call.
The salvage refuses (returns None) when the answer contains a web address that is not in the input: an invented
reference may be the trace of an instruction hidden in a paper, so that answer is rejected as a whole.
Whether a salvage is used without a repair call is decided by roles/base.py: when at most half of the items were
dropped (and no required part was given up); after the repair call, whatever the share.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from paperwatch.reading.evidence_checks import (
    PassageTextById,
    find_numbers_backed_by_passage,
    find_numbers_needing_evidence,
    find_words_around_number,
    normalize_for_matching,
)
from paperwatch.roles.evidence_fields import (
    MAXIMUM_NUMBER_PASSAGES_PER_FINDING,
    MAXIMUM_PASSAGES_PER_STATEMENT,
    ItemCheck,
)
from paperwatch.roles.output_guards import find_invented_web_addresses

RoleInput = Mapping[str, Any]

MAXIMUM_PASSAGES_IN_A_RANGE = 10  # "P35-P37" is split; "P1-P300" is not a citation
PASSAGE_ID_LIST_SEPARATOR_PATTERN = re.compile(r"[\s,;'\"\[\]()]+|\band\b")
PASSAGE_ID_RANGE_PATTERN = re.compile(r"^([APT])(\d+)[-–—]([APT])?(\d+)$")
JOINED_PASSAGE_IDS_PATTERN = re.compile(r"^(?:[APT]\d+)+$")
SINGLE_PASSAGE_ID_PATTERN = re.compile(r"[APT]\d+")
# A passage id, optionally keyed by its review as the writer cites it ("r_ab12#P7").
PASSAGE_ID_PARTS_PATTERN = re.compile(r"^(?:(?P<review>[^#]+)#)?(?P<kind>[APT])(?P<number>\d+)$")
TEXT_FIELD_OF_A_STATEMENT = ("statement", "inference")  # the field whose numbers a statement-like item must back
# A results table or figure caption usually sits within two passages of the sentence that discusses it.
NEIGHBOURING_PASSAGE_DISTANCE = 2
CONTEXT_WINDOW_WORDS = 4  # words on each side of a number that say what it counts ("18 total flights")
CLEAR_CONTEXT_WORDS = 2  # shared context words that single out a passage before the weaker rules are tried
DISTINCTIVE_NUMBER_DIGITS = 3  # "349", "2,000", "0.81": unlikely to mean two different things in one paper


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
    # A part the answer cannot do without was given up (the writer's overview or read_first): such a salvage is used
    # only when the repair call cannot do better (roles/base.py), never instead of it.
    gave_up_a_required_part: bool = False

    @property
    def dropped_at_most_half(self) -> bool:
        return not self.gave_up_a_required_part and 2 * len(self.dropped_items) <= self.droppable_item_count

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
                f"added {len(self.added_citations)} citation(s) of the passage that states a number "
                f"({'; '.join(self.added_citations)})"
            )
        if self.dropped_items:
            parts.append(
                f"dropped {len(self.dropped_items)} of {self.droppable_item_count} item(s) that failed the "
                f"evidence checks ({'; '.join(self.dropped_items)})"
            )
        return "host corrections: " + ("; ".join(parts) if parts else "none")


@dataclass
class CorrectionLog:
    split_passage_ids: list[str] = field(default_factory=list)
    relocated_citations: list[str] = field(default_factory=list)
    added_citations: list[str] = field(default_factory=list)
    dropped_items: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------------------------
# Where the input states each number
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberSources:
    """Every number each passage states (digits, number words, percent rescalings), and, for the reviewer reading
    section notes, the passages the notes cited for each number: where the reviewer actually read it."""

    passage_text_by_id: PassageTextById
    numbers_by_passage_id: Mapping[str, set[str]]
    noted_passage_ids_by_number: Mapping[str, frozenset[str]]

    @classmethod
    def from_input(cls, passage_text_by_id: PassageTextById, role_input: RoleInput) -> NumberSources:
        numbers_by_passage_id = {
            passage_id: find_numbers_backed_by_passage(passage_text)
            for passage_id, passage_text in passage_text_by_id.items()
        }
        noted_passage_ids: dict[str, set[str]] = {}
        for cited_passage_ids, stated_text in _cited_items_of_section_notes(role_input):
            for number in find_numbers_needing_evidence(stated_text):
                for passage_id in cited_passage_ids:
                    if _passage_states(numbers_by_passage_id.get(passage_id), number):
                        noted_passage_ids.setdefault(number, set()).add(passage_id)
        return cls(
            passage_text_by_id,
            numbers_by_passage_id,
            {number: frozenset(passage_ids) for number, passage_ids in noted_passage_ids.items()},
        )

    def states(self, passage_id: str, number: str) -> bool:
        return _passage_states(self.numbers_by_passage_id.get(passage_id), number)

    def shared_context_words(self, passage_id: str, stated_text: str, numbers: Sequence[str]) -> int:
        """How many words around `numbers` the passage shares with the statement that states them."""
        passage_text = self.passage_text_by_id.get(passage_id, "")
        return sum(
            len(
                find_words_around_number(stated_text, number, CONTEXT_WINDOW_WORDS)
                & find_words_around_number(passage_text, number, CONTEXT_WINDOW_WORDS)
            )
            for number in numbers
        )

    def words_shared_by_all(self, passage_ids: Sequence[str], stated_text: str, numbers: Sequence[str]) -> set[str]:
        """The words around `numbers` that the statement and EVERY one of the passages use: all of them say the
        same thing ("38 included studies", "38 studies were identified")."""
        shared_words: set[str] | None = None
        for number in numbers:
            statement_words = find_words_around_number(stated_text, number, CONTEXT_WINDOW_WORDS)
            for passage_id in passage_ids:
                passage_words = find_words_around_number(
                    self.passage_text_by_id.get(passage_id, ""), number, CONTEXT_WINDOW_WORDS
                )
                shared_words = (
                    (statement_words & passage_words) if shared_words is None else (shared_words & passage_words)
                )
        return shared_words or set()

    def missing_numbers(self, stated_text: str, cited_passage_ids: Sequence[str]) -> list[str]:
        """The numbers of `stated_text` that no cited passage states."""
        return [
            number
            for number in find_numbers_needing_evidence(stated_text)
            if not any(self.states(passage_id, number) for passage_id in cited_passage_ids)
        ]


def _passage_states(passage_numbers: set[str] | None, number: str) -> bool:
    return passage_numbers is not None and (number in passage_numbers or number.lstrip("+-") in passage_numbers)


def _cited_items_of_section_notes(role_input: RoleInput) -> Iterator[tuple[list[str], str]]:
    """(cited passage ids, stated text) of every finding and inference in the section notes the reviewer read."""
    for note_entry in role_input.get("section_notes", []) or []:
        notes = note_entry.get("notes", {}) if isinstance(note_entry, Mapping) else {}
        for value in notes.values() if isinstance(notes, Mapping) else ():
            for item in value if isinstance(value, list) else ():
                if not isinstance(item, Mapping):
                    continue
                if "claim" in item and "passage_id" in item:
                    yield (
                        [str(item["passage_id"]), *map(str, item.get("number_passage_ids", []))],
                        f"{item['claim']} {item.get('conditions', '')}",
                    )
                elif "inference" in item:
                    yield (
                        [str(passage_id) for passage_id in item.get("supporting_passage_ids", [])],
                        str(item["inference"]),
                    )


def choose_passages_for_numbers(
    missing_numbers: Sequence[str],
    stated_text: str,
    anchor_passage_ids: Sequence[str],
    sources: NumberSources,
    maximum_added: int,
    named_passage_ids: Sequence[str] = (),
) -> list[str] | None:
    """The passages to cite so that every missing number is stated by a cited passage; [] when nothing is missing;
    None when that cannot be done without guessing or within `maximum_added` (see the module doc, rule 1c).
    `stated_text` is the statement or claim that states the numbers; `anchor_passage_ids` are the item's cited
    passages (for the neighbour rule); `named_passage_ids` are passages the answer itself named for this item, e.g.
    the other half of a joined id."""
    if not missing_numbers:
        return []
    if maximum_added <= 0:
        return None
    passages_stating_all = [
        passage_id
        for passage_id in sources.numbers_by_passage_id
        if all(sources.states(passage_id, number) for number in missing_numbers)
    ]
    one_passage = _single_out(
        passages_stating_all, missing_numbers, stated_text, anchor_passage_ids, sources, named_passage_ids
    )
    if one_passage is not None:
        return [one_passage]
    candidates_by_number = {
        number: [passage_id for passage_id in sources.numbers_by_passage_id if sources.states(passage_id, number)]
        for number in missing_numbers
    }
    chosen_passage_ids: list[str] = []
    # Numbers with the fewest candidates first: a passage they force may also state the more common numbers. A small
    # whole number ("50") counts as stated by an already chosen passage only where the words around it agree: a
    # table that says "K is 50" does not back "a 50% mixture".
    for number in sorted(missing_numbers, key=lambda number: len(candidates_by_number[number])):
        if any(
            sources.states(passage_id, number)
            and (_is_distinctive_number(number) or sources.shared_context_words(passage_id, stated_text, [number]))
            for passage_id in chosen_passage_ids
        ):
            continue
        passage_id = _single_out(
            candidates_by_number[number], [number], stated_text, anchor_passage_ids, sources, named_passage_ids
        )
        if passage_id is None:
            return None
        chosen_passage_ids.append(passage_id)
    if len(chosen_passage_ids) > maximum_added:
        return None
    return sorted(chosen_passage_ids, key=list(sources.passage_text_by_id).index)  # in document order


def _is_distinctive_number(number: str) -> bool:
    """A decimal or a number of three or more digits rarely means two different things in one paper."""
    digits = number.lstrip("+-")
    return "." in digits or len(digits) >= DISTINCTIVE_NUMBER_DIGITS


def _single_out(
    candidate_passage_ids: list[str],
    numbers: Sequence[str],
    stated_text: str,
    anchor_passage_ids: Sequence[str],
    sources: NumberSources,
    named_passage_ids: Sequence[str],
) -> str | None:
    """The one candidate, or the one left after narrowing by the rules in order (a rule that would leave none is
    skipped); None while more than one is left (see the module doc, rule 1c)."""
    if len(candidate_passage_ids) <= 1:
        return candidate_passage_ids[0] if candidate_passage_ids else None

    def shared_words(passage_id: str) -> int:
        return sources.shared_context_words(passage_id, stated_text, numbers)

    remaining_passage_ids = list(candidate_passage_ids)
    if not any(_is_distinctive_number(number) for number in numbers):
        # Several passages state a small whole number ("60"): only those the answer named, the section notes cited
        # for it, or whose words around it agree with the statement can be meant.
        remaining_passage_ids = [
            passage_id
            for passage_id in remaining_passage_ids
            if passage_id in named_passage_ids
            or all(passage_id in sources.noted_passage_ids_by_number.get(number, frozenset()) for number in numbers)
            or shared_words(passage_id) > 0
        ]
        if len(remaining_passage_ids) <= 1:
            return remaining_passage_ids[0] if remaining_passage_ids else None

    def most_shared_context(minimum_shared_words: int) -> Callable[[list[str]], list[str]]:
        def narrow(passage_ids: list[str]) -> list[str]:
            shared_by_passage_id = {passage_id: shared_words(passage_id) for passage_id in passage_ids}
            most_shared = max(shared_by_passage_id.values())
            if most_shared < minimum_shared_words:
                return []
            return [passage_id for passage_id in passage_ids if shared_by_passage_id[passage_id] == most_shared]

        return narrow

    narrowing_rules: tuple[Callable[[list[str]], list[str]], ...] = (
        lambda passage_ids: [passage_id for passage_id in passage_ids if passage_id in named_passage_ids],
        most_shared_context(CLEAR_CONTEXT_WORDS),
        lambda passage_ids: [
            passage_id
            for passage_id in passage_ids
            if all(passage_id in sources.noted_passage_ids_by_number.get(number, frozenset()) for number in numbers)
        ],
        lambda passage_ids: [
            passage_id for passage_id in passage_ids if _is_next_to_any(passage_id, anchor_passage_ids)
        ],
        most_shared_context(1),
    )
    for narrow in narrowing_rules:
        narrowed_passage_ids = narrow(remaining_passage_ids)
        if narrowed_passage_ids:
            remaining_passage_ids = narrowed_passage_ids
        if len(remaining_passage_ids) == 1:
            return remaining_passage_ids[0]
    # The same statement repeated (an abstract and an introduction that both say "108 total trials"): when every
    # remaining passage describes the numbers with a word the statement also uses, its first occurrence is cited.
    if sources.words_shared_by_all(remaining_passage_ids, stated_text, numbers):
        document_order = {passage_id: position for position, passage_id in enumerate(sources.passage_text_by_id)}
        return min(remaining_passage_ids, key=lambda passage_id: document_order.get(passage_id, len(document_order)))
    return None


def _is_next_to_any(passage_id: str, anchor_passage_ids: Sequence[str]) -> bool:
    """True if `passage_id` lies within NEIGHBOURING_PASSAGE_DISTANCE of an anchor of the same kind (P, A or T) in
    the same review."""
    parts = PASSAGE_ID_PARTS_PATTERN.fullmatch(passage_id)
    if parts is None:
        return False
    for anchor_passage_id in anchor_passage_ids:
        anchor_parts = PASSAGE_ID_PARTS_PATTERN.fullmatch(anchor_passage_id)
        if (
            anchor_parts is not None
            and anchor_parts["review"] == parts["review"]
            and anchor_parts["kind"] == parts["kind"]
            and 0 < abs(int(anchor_parts["number"]) - int(parts["number"])) <= NEIGHBOURING_PASSAGE_DISTANCE
        ):
            return True
    return False


# --------------------------------------------------------------------------------------------------------------------
# Salvage
# --------------------------------------------------------------------------------------------------------------------


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
    sources = NumberSources.from_input(passage_text_by_id, role_input)
    corrections = CorrectionLog()

    for single_statement in single_statement_fields:
        statement = salvaged_output.get(single_statement.field_name)
        if isinstance(statement, dict):
            salvaged_output[single_statement.field_name] = fix_statement_citations(
                statement,
                single_statement.field_name,
                single_statement.maximum_cited_passages,
                passage_text_by_id,
                sources,
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
                item = fix_finding_citations(item, where, passage_text_by_id, sources, corrections)
            elif "supporting_passage_ids" in item:
                item = fix_statement_citations(
                    item, where, droppable_field.maximum_cited_passages, passage_text_by_id, sources, corrections
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


def fix_statement_citations(
    statement: dict[str, Any],
    where: str,
    maximum_cited_passages: int,
    passage_text_by_id: PassageTextById,
    sources: NumberSources,
    corrections: CorrectionLog,
) -> dict[str, Any]:
    """Split joined ids, then add the passage(s) that state each uncited number. Returns a corrected copy."""
    cited_passage_ids = _split_cited_entries(
        statement["supporting_passage_ids"], where, passage_text_by_id, corrections
    )
    if len(cited_passage_ids) > maximum_cited_passages:
        return {**statement, "supporting_passage_ids": cited_passage_ids}  # over the cap: the schema check rejects it
    if any(passage_id not in passage_text_by_id for passage_id in cited_passage_ids):
        return {**statement, "supporting_passage_ids": cited_passage_ids}  # an unknown id: the check reports it

    statement_text = next(
        (str(statement[text_field]) for text_field in TEXT_FIELD_OF_A_STATEMENT if text_field in statement), ""
    )
    missing_numbers = sources.missing_numbers(statement_text, cited_passage_ids)
    added_passage_ids = choose_passages_for_numbers(
        missing_numbers, statement_text, cited_passage_ids, sources, maximum_cited_passages - len(cited_passage_ids)
    )
    if added_passage_ids:
        corrections.added_citations.append(
            f"{where}: + {', '.join(added_passage_ids)} (states {', '.join(missing_numbers)})"
        )
        cited_passage_ids += added_passage_ids
    return {**statement, "supporting_passage_ids": cited_passage_ids}


def fix_finding_citations(
    finding: Mapping[str, Any],
    where: str,
    passage_text_by_id: PassageTextById,
    sources: NumberSources,
    corrections: CorrectionLog,
) -> dict[str, Any]:
    """Split a joined passage id, move the quotation to the passage that holds it, then name the passage(s) that
    state the claim's numbers. Returns a corrected copy; the claim, conditions and excerpt are never changed."""
    corrected = dict(finding)
    named_passage_ids: list[str] = []  # passages the answer named for this finding but that do not hold the quote
    cited_passage_id = str(corrected["passage_id"])
    if cited_passage_id not in passage_text_by_id:
        split_ids = split_joined_passage_ids(cited_passage_id, passage_text_by_id) or []
        quote_holders = _passages_holding_excerpt(corrected["excerpt"], split_ids, passage_text_by_id)
        if len(quote_holders) == 1:
            corrections.split_passage_ids.append(f"{where}: '{cited_passage_id}' -> {quote_holders[0]}")
            named_passage_ids = [passage_id for passage_id in split_ids if passage_id != quote_holders[0]]
            corrected["passage_id"] = quote_holders[0]
    if "number_passage_ids" in corrected:
        corrected["number_passage_ids"] = _split_cited_entries(
            corrected["number_passage_ids"], f"{where}.number_passage_ids", passage_text_by_id, corrections
        )

    if not _passages_holding_excerpt(corrected["excerpt"], [corrected["passage_id"]], passage_text_by_id):
        holding_passage_id = _the_passage_that_holds_the_quote(corrected, passage_text_by_id, sources)
        if holding_passage_id is not None:
            corrections.relocated_citations.append(f"{where}: {corrected['passage_id']} -> {holding_passage_id}")
            named_passage_ids.append(str(corrected["passage_id"]))
            corrected["passage_id"] = holding_passage_id

    number_passage_ids = list(corrected.get("number_passage_ids", []))
    cited_passage_ids = [corrected["passage_id"], *number_passage_ids]
    if any(passage_id not in passage_text_by_id for passage_id in cited_passage_ids):
        return corrected  # an unknown id: the check reports it
    stated_text = f"{corrected['claim']} {corrected['conditions']}"
    missing_numbers = sources.missing_numbers(stated_text, cited_passage_ids)
    added_passage_ids = choose_passages_for_numbers(
        missing_numbers,
        stated_text,
        [corrected["passage_id"]],
        sources,
        MAXIMUM_NUMBER_PASSAGES_PER_FINDING - len(number_passage_ids),
        named_passage_ids=named_passage_ids,
    )
    if added_passage_ids:
        corrections.added_citations.append(
            f"{where}: + {', '.join(added_passage_ids)} for its numbers (states {', '.join(missing_numbers)})"
        )
        corrected["number_passage_ids"] = number_passage_ids + added_passage_ids
    return corrected


def _the_passage_that_holds_the_quote(
    finding: Mapping[str, Any], passage_text_by_id: PassageTextById, sources: NumberSources
) -> str | None:
    """The ONE passage of the input that contains the finding's excerpt; among several, the one that also states
    every number of the claim (owner decision, 2026-11-21: a quotation found in several passages is never moved by
    a guess). None when that leaves no single passage."""
    quote_holders = _passages_holding_excerpt(finding["excerpt"], list(passage_text_by_id), passage_text_by_id)
    if len(quote_holders) <= 1:
        return quote_holders[0] if quote_holders else None
    stated_text = f"{finding['claim']} {finding['conditions']}"
    holders_stating_the_numbers = [
        passage_id for passage_id in quote_holders if not sources.missing_numbers(stated_text, [passage_id])
    ]
    return holders_stating_the_numbers[0] if len(holders_stating_the_numbers) == 1 else None


def _passages_holding_excerpt(
    excerpt: Any, passage_ids: Sequence[str], passage_text_by_id: PassageTextById
) -> list[str]:
    normalized_excerpt = normalize_for_matching(str(excerpt).strip(" \"'"))
    if not normalized_excerpt:
        return []
    return [
        passage_id
        for passage_id in passage_ids
        if passage_id in passage_text_by_id
        and normalized_excerpt in normalize_for_matching(passage_text_by_id[passage_id])
    ]


def _split_cited_entries(
    cited_entries: Sequence[Any], where: str, passage_text_by_id: PassageTextById, corrections: CorrectionLog
) -> list[str]:
    """The cited list with every joined entry split into the ids it names (an unknown entry is kept for the check
    to report), without repeats."""
    cited_passage_ids: list[str] = []
    for cited_entry in map(str, cited_entries):
        split_ids = split_joined_passage_ids(cited_entry, passage_text_by_id)
        if split_ids is None:
            cited_passage_ids.append(cited_entry)
            continue
        if split_ids != [cited_entry]:
            corrections.split_passage_ids.append(f"{where}: '{cited_entry}' -> {', '.join(split_ids)}")
        cited_passage_ids += split_ids
    return list(dict.fromkeys(cited_passage_ids))


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
