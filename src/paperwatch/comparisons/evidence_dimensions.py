"""Which evidence dimensions two reviews share, decided by plain word overlap (no model call, no embedding).

Plan part 8: queue a comparison when two reviews "have comparable assumptions, guarantees, or evaluation settings".
For each dimension the content words of both reviews are compared:
    assumptions          the supported assumption statements
    guarantees           the authors' result claims (what is shown or guaranteed)
    evaluation_setting   the supported evaluation-setting statement plus the conditions of each result
A dimension is shared when the two word sets overlap in at least MINIMUM_SHARED_WORDS words and in at least
MINIMUM_OVERLAP_COEFFICIENT of the smaller set. Statements marked insufficient evidence do not count, so an abstract
that says nothing about the setting cannot look comparable. The count orders comparison jobs (class 4) and the
comparison role is told which dimensions matched; the role itself judges whether the conditions are truly comparable.
"""

from __future__ import annotations

import re
from typing import Any

DIMENSION_ASSUMPTIONS = "assumptions"
DIMENSION_GUARANTEES = "guarantees"
DIMENSION_EVALUATION_SETTING = "evaluation_setting"
EVIDENCE_DIMENSIONS = (DIMENSION_ASSUMPTIONS, DIMENSION_GUARANTEES, DIMENSION_EVALUATION_SETTING)
MINIMUM_SHARED_WORDS = 2
MINIMUM_OVERLAP_COEFFICIENT = 0.3
CONTENT_WORD_PATTERN = re.compile(r"[a-z][a-z-]{3,}")
STOPWORDS = frozenset(
    """
about above after again against also among and another because been before being below between both cannot could
does doing down during each either every from further have having here into itself just more most much must only
other over same should since some such than that their them then there these they this those through under until
upon very were what when where whether which while with within without would your paper authors results result
show shows shown study studies propose proposed approach method methods using used based insufficient evidence
""".split()
)
SUPPORTED_EVIDENCE_STATUS = "supported"


def content_words(text: str) -> set[str]:
    """Lower-case content words (4+ letters, no stopwords, a plural "s" dropped) for word-overlap tests."""
    words = set()
    for word in CONTENT_WORD_PATTERN.findall(text.lower()):
        if word in STOPWORDS:
            continue
        words.add(word[:-1] if word.endswith("s") and not word.endswith("ss") else word)
    return words


def _supported_statement(statement: Any) -> str:
    if isinstance(statement, dict) and statement.get("evidence_status") == SUPPORTED_EVIDENCE_STATUS:
        return str(statement.get("statement", ""))
    return ""


def evidence_dimension_words(review_content: dict[str, Any]) -> dict[str, set[str]]:
    """Content words per dimension of one validated part-7 review (a legacy review yields empty sets)."""
    results = [finding for finding in review_content.get("results", []) if isinstance(finding, dict)]
    return {
        DIMENSION_ASSUMPTIONS: content_words(
            " ".join(_supported_statement(statement) for statement in review_content.get("assumptions", []))
        ),
        DIMENSION_GUARANTEES: content_words(" ".join(str(finding.get("claim", "")) for finding in results)),
        DIMENSION_EVALUATION_SETTING: content_words(
            _supported_statement(review_content.get("evaluation_setting"))
            + " "
            + " ".join(str(finding.get("conditions", "")) for finding in results)
        ),
    }


def shared_evidence_dimensions(review_content_a: dict[str, Any], review_content_b: dict[str, Any]) -> list[str]:
    words_a, words_b = evidence_dimension_words(review_content_a), evidence_dimension_words(review_content_b)
    shared = []
    for dimension in EVIDENCE_DIMENSIONS:
        common_words = words_a[dimension] & words_b[dimension]
        smaller_size = min(len(words_a[dimension]), len(words_b[dimension]))
        if (
            smaller_size
            and len(common_words) >= MINIMUM_SHARED_WORDS
            and len(common_words) / smaller_size >= MINIMUM_OVERLAP_COEFFICIENT
        ):
            shared.append(dimension)
    return shared
