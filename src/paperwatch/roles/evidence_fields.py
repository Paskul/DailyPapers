"""Schema fragments and host checks shared by the evidence-reading roles (section reader and reviewer, plan part 7).

Field shapes (every object rejects unknown fields):
    finding     {claim, conditions, passage_id, excerpt}    an AUTHORS' claim, quoted from one passage
    statement   {statement, evidence_status, supporting_passage_ids}
                evidence_status "insufficient_evidence" lets the reviewer say the text does not support an answer
    limitation  {statement, stated_by: authors|reviewer, supporting_passage_ids}
    inference   {inference, supporting_passage_ids}         the MODEL's reasoning, kept apart from authors' claims
    ambiguous   {passage_id, issue}                         a passage that can be read more than one way
    parser      {passage_id, problem}                       text the extractor garbled (tables, equations)
Enum values are ordered conservative-first ("insufficient_evidence" before "supported"), so the smallest valid
answer claims nothing. String lengths are checked here, not in the schema: long maxLength bounds make llama.cpp's
grammar large and slow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from paperwatch.reading.evidence_checks import (PassageTextById, check_cited_statement, check_quoted_finding,
                                                find_unknown_passage_ids)

EVIDENCE_STATUS_INSUFFICIENT = "insufficient_evidence"
EVIDENCE_STATUS_SUPPORTED = "supported"
MAXIMUM_STATEMENT_CHARACTERS = 1200
MAXIMUM_PASSAGES_PER_STATEMENT = 6

PASSAGE_ID_LIST_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string"},
                                          "maxItems": MAXIMUM_PASSAGES_PER_STATEMENT}
FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claim": {"type": "string"}, "conditions": {"type": "string"},
                   "passage_id": {"type": "string"}, "excerpt": {"type": "string"}},
    "required": ["claim", "conditions", "passage_id", "excerpt"],
    "additionalProperties": False,
}
STATEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"statement": {"type": "string"},
                   "evidence_status": {"type": "string",
                                       "enum": [EVIDENCE_STATUS_INSUFFICIENT, EVIDENCE_STATUS_SUPPORTED]},
                   "supporting_passage_ids": PASSAGE_ID_LIST_SCHEMA},
    "required": ["statement", "evidence_status", "supporting_passage_ids"],
    "additionalProperties": False,
}
LIMITATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"statement": {"type": "string"}, "stated_by": {"type": "string", "enum": ["reviewer", "authors"]},
                   "supporting_passage_ids": PASSAGE_ID_LIST_SCHEMA},
    "required": ["statement", "stated_by", "supporting_passage_ids"],
    "additionalProperties": False,
}
INFERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"inference": {"type": "string"}, "supporting_passage_ids": PASSAGE_ID_LIST_SCHEMA},
    "required": ["inference", "supporting_passage_ids"],
    "additionalProperties": False,
}
AMBIGUOUS_PASSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"passage_id": {"type": "string"}, "issue": {"type": "string"}},
    "required": ["passage_id", "issue"],
    "additionalProperties": False,
}
PARSER_PROBLEM_KINDS = ("table_unreadable", "equation_unreadable", "text_garbled")
PARSER_PROBLEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"passage_id": {"type": "string"},
                   "problem": {"type": "string", "enum": list(PARSER_PROBLEM_KINDS)}},
    "required": ["passage_id", "problem"],
    "additionalProperties": False,
}


def array_of(item_schema: dict[str, Any], maximum_items: int) -> dict[str, Any]:
    return {"type": "array", "items": item_schema, "maxItems": maximum_items}


EVIDENCE_RULES_FOR_PROMPTS = """\
Evidence rules:
- Cite passages only by the passage_id values given in the input. Never invent an id.
- A finding is the AUTHORS' claim. Its excerpt is a short contiguous quotation (3 to 60 words, no "...") copied \
from the one passage it cites. Every number in its claim and conditions must appear in that passage.
- Put the conditions a result holds under (setting, dataset, assumptions, sample size) in "conditions".
- A statement with evidence_status "supported" cites the passages that support it; numbers in it must appear \
in those passages. If the text does not answer the question, write "insufficient evidence" and use \
evidence_status "insufficient_evidence".
- Your own reasoning goes ONLY into reviewer inferences or limitations with stated_by "reviewer"; never present \
it as the authors' claim.
- Flag passages that can be read more than one way in ambiguous_passages, and tables or equations the text \
extraction garbled in parser_problems."""


def check_finding_list(findings: list[dict[str, Any]], passage_text_by_id: PassageTextById,
                       field_name: str) -> list[str]:
    problems: list[str] = []
    for finding_index, finding in enumerate(findings, start=1):
        where = f"{field_name}[{finding_index}]"
        problems += check_quoted_finding(f"{finding['claim']} {finding['conditions']}", finding["passage_id"],
                                         finding["excerpt"], passage_text_by_id, where)
        problems += _check_length(finding["claim"], where) + _check_length(finding["conditions"], where)
    return problems


def check_statement(statement: Mapping[str, Any], passage_text_by_id: PassageTextById, where: str) -> list[str]:
    problems = _check_length(statement["statement"], where)
    cited_passage_ids = list(statement["supporting_passage_ids"])
    if statement["evidence_status"] == EVIDENCE_STATUS_SUPPORTED and not cited_passage_ids:
        problems.append(f"{where}: a supported statement must cite at least one passage")
    return problems + check_cited_statement(statement["statement"], cited_passage_ids, passage_text_by_id, where)


def check_limitation_list(limitations: list[dict[str, Any]], passage_text_by_id: PassageTextById) -> list[str]:
    problems: list[str] = []
    for limitation_index, limitation in enumerate(limitations, start=1):
        where = f"limitations[{limitation_index}]"
        problems += _check_length(limitation["statement"], where)
        if limitation["stated_by"] == "authors" and not limitation["supporting_passage_ids"]:
            problems.append(f"{where}: a limitation stated by the authors must cite the passage that states it")
        problems += check_cited_statement(limitation["statement"], list(limitation["supporting_passage_ids"]),
                                          passage_text_by_id, where)
    return problems


def check_inference_list(inferences: list[dict[str, Any]], passage_text_by_id: PassageTextById) -> list[str]:
    problems: list[str] = []
    for inference_index, inference in enumerate(inferences, start=1):
        where = f"reviewer_inferences[{inference_index}]"
        problems += _check_length(inference["inference"], where)
        problems += check_cited_statement(inference["inference"], list(inference["supporting_passage_ids"]),
                                          passage_text_by_id, where)
    return problems


def check_flagged_passages(answer: Mapping[str, Any], passage_text_by_id: PassageTextById) -> list[str]:
    """ambiguous_passages and parser_problems may only name passages from the input."""
    return (find_unknown_passage_ids((item["passage_id"] for item in answer.get("ambiguous_passages", [])),
                                     passage_text_by_id, "ambiguous_passages")
            + find_unknown_passage_ids((item["passage_id"] for item in answer.get("parser_problems", [])),
                                       passage_text_by_id, "parser_problems"))


def _check_length(text: str, where: str) -> list[str]:
    if len(text) > MAXIMUM_STATEMENT_CHARACTERS:
        return [f"{where}: {len(text)} characters; keep each text field under {MAXIMUM_STATEMENT_CHARACTERS}"]
    return []
