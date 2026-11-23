"""Host corrections for evidence items that cite SAVED reviews: {review_id, passage_id, excerpt}.

The writer (the daily report) and the comparison role write statements about several saved reviews and support them
with such items; a passage is keyed by its review ("r_ab12#P7"), because every review has a P1. Before an answer is
rejected, each claim's evidence is corrected the same way ("fix, then drop", roles/answer_salvage.py):
  * an item whose excerpt is not in its passage moves to the ONE passage of the same review that holds it, else it is
    dropped (owner decision, 2026-11-21: a quotation is never moved by a guess);
  * a number the claim states but none of its passages does gets a host quotation, word for word, of the passage
    that states it, chosen among the passages of the reviews the claim may cite, with the same rules as for the
    reviewer (answer_salvage.choose_passages_for_numbers), within the evidence limit;
  * `claim_problem` then says whether the corrected claim passes (length, evidence left, numbers backed).
Every change is logged in the form AnswerSalvage.describe reports.
"""

from __future__ import annotations

import re
from typing import Any

from paperwatch.reading.evidence_checks import (
    MAXIMUM_EXCERPT_WORDS,
    MINIMUM_EXCERPT_WORDS,
    NUMBER_PATTERN,
    check_excerpt_matches_passage,
    check_numbers_occur_in_passages,
    find_numbers_backed_by_passage,
    normalize_number_text,
)
from paperwatch.roles.answer_salvage import NumberSources, choose_passages_for_numbers

HOST_QUOTE_WORDS = 30  # the length of a quotation the host takes to show where a number is stated
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?;])\s+|\n+")


def evidence_passage_key(review_id: str, passage_id: str) -> str:
    """Passage ids repeat across reviews (every review has a P1), so evidence is keyed by review and passage."""
    return f"{review_id}#{passage_id}"


class KeyedEvidenceCorrections:
    """Corrects the evidence of one answer's claims in place and logs each change."""

    def __init__(
        self, passage_text_by_key: dict[str, str], review_ids: list[str], maximum_evidence_per_statement: int
    ) -> None:
        self.passage_text_by_key = passage_text_by_key
        self.passage_text_by_review_id: dict[str, dict[str, str]] = {review_id: {} for review_id in review_ids}
        for passage_key, passage_text in passage_text_by_key.items():
            review_id, _, passage_id = passage_key.partition("#")
            if review_id in self.passage_text_by_review_id:
                self.passage_text_by_review_id[review_id][passage_id] = passage_text
        self.maximum_evidence_per_statement = maximum_evidence_per_statement
        self.relocated_citations: list[str] = []
        self.added_citations: list[str] = []
        self.dropped_items: list[str] = []
        self.droppable_item_count = 0
        self.gave_up_a_required_part = False

    def fix_evidence(
        self, evidence: list[dict[str, Any]], where: str, only_review_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The items that quote their passage, after moving a misplaced quotation to the one passage that holds it."""
        kept_evidence = []
        for evidence_index, item in enumerate(evidence, start=1):
            self.droppable_item_count += 1
            item_where = f"{where}.evidence[{evidence_index}]"
            review_passages = self.passage_text_by_review_id.get(item["review_id"])
            if review_passages is None or (only_review_id is not None and item["review_id"] != only_review_id):
                self.dropped_items.append(f"{item_where}: review '{item['review_id']}' cannot be cited here")
                continue
            cited_text = review_passages.get(item["passage_id"])
            if cited_text is not None and not check_excerpt_matches_passage(
                item["excerpt"], item["passage_id"], cited_text, item_where
            ):
                kept_evidence.append(item)
                continue
            holding_passage_id = the_one_passage_that_holds_the_excerpt(item["excerpt"], review_passages)
            if holding_passage_id is not None:
                self.relocated_citations.append(f"{item_where}: {item['passage_id']} -> {holding_passage_id}")
                kept_evidence.append({**item, "passage_id": holding_passage_id})
                continue
            self.dropped_items.append(f"{item_where}: excerpt does not occur in passage {item['passage_id']}")
        return kept_evidence

    def add_number_evidence(
        self, claim_text: str, evidence: list[dict[str, Any]], where: str, citable_review_ids: set[str]
    ) -> list[dict[str, Any]]:
        """The claim's evidence plus a host quotation of the passage that states each number the evidence does not
        state, chosen among the passages of `citable_review_ids` as for the reviewer. Unchanged when no number is
        missing, or when no passage can be chosen without guessing or within the evidence limit."""
        cited_keys = [evidence_passage_key(item["review_id"], item["passage_id"]) for item in evidence]
        candidate_texts = {
            passage_key: passage_text
            for passage_key, passage_text in self.passage_text_by_key.items()
            if passage_key.partition("#")[0] in citable_review_ids or passage_key in cited_keys
        }
        number_sources = NumberSources.from_input(candidate_texts, {})
        missing_numbers = number_sources.missing_numbers(claim_text, cited_keys)
        added_keys = choose_passages_for_numbers(
            missing_numbers, claim_text, cited_keys, number_sources, self.maximum_evidence_per_statement - len(evidence)
        )
        if not added_keys:
            return evidence
        added_evidence = []
        for passage_key in added_keys:
            stated_numbers = [number for number in missing_numbers if number_sources.states(passage_key, number)]
            quotation = quote_stating_numbers(self.passage_text_by_key[passage_key], stated_numbers)
            if quotation is None:
                return evidence
            review_id, _, passage_id = passage_key.partition("#")
            added_evidence.append({"review_id": review_id, "passage_id": passage_id, "excerpt": quotation})
        self.added_citations.append(f"{where}: + {', '.join(added_keys)} (states {', '.join(missing_numbers)})")
        return evidence + added_evidence

    def claim_problem(
        self,
        claim_text: str,
        evidence: list[dict[str, Any]],
        where: str,
        maximum_characters: int,
        *,
        evidence_is_required: bool = True,
    ) -> str:
        """The first reason a corrected claim still fails ('' when it passes)."""
        if len(claim_text) > maximum_characters:
            return f"{where}: longer than {maximum_characters} characters"
        if not evidence and evidence_is_required:
            return f"{where}: no evidence left"
        cited_keys = [evidence_passage_key(item["review_id"], item["passage_id"]) for item in evidence]
        number_problems = check_numbers_occur_in_passages(claim_text, cited_keys, self.passage_text_by_key, where)
        return number_problems[0] if number_problems else ""


def quote_stating_numbers(passage_text: str, numbers: list[str]) -> str | None:
    """A short quotation, word for word, of the first sentence of the passage that states one of `numbers` (the
    HOST_QUOTE_WORDS words around the number when the sentence is longer); None if no sentence gives a quotation
    that passes the excerpt rules (e.g. every candidate contains '...')."""
    wanted_values = {number.lstrip("+-") for number in numbers}
    for sentence in SENTENCE_BOUNDARY_PATTERN.split(passage_text):
        stated_values = {value.lstrip("+-") for value in find_numbers_backed_by_passage(sentence)}
        if not wanted_values & stated_values:
            continue
        words = sentence.split()
        if len(words) > min(HOST_QUOTE_WORDS, MAXIMUM_EXCERPT_WORDS):
            number_word_index = next(
                (
                    word_index
                    for word_index, word in enumerate(words)
                    if any(
                        normalize_number_text(number_match.group()).lstrip("+-") in wanted_values
                        for number_match in NUMBER_PATTERN.finditer(word)
                    )
                ),
                0,
            )
            first_word = max(0, min(number_word_index - HOST_QUOTE_WORDS // 2, len(words) - HOST_QUOTE_WORDS))
            words = words[first_word : first_word + HOST_QUOTE_WORDS]
        quotation = " ".join(words)
        if len(words) >= MINIMUM_EXCERPT_WORDS and not check_excerpt_matches_passage(
            quotation, "passage", passage_text, "quotation"
        ):
            return quotation
    return None


def the_one_passage_that_holds_the_excerpt(excerpt: str, passage_text_by_id: dict[str, str]) -> str | None:
    holding_passage_ids = [
        passage_id
        for passage_id, passage_text in passage_text_by_id.items()
        if not check_excerpt_matches_passage(excerpt, passage_id, passage_text, "evidence")
    ]
    return holding_passage_ids[0] if len(holding_passage_ids) == 1 else None
