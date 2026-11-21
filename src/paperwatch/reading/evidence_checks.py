"""Host-side evidence checks for the reviewer roles (plan part 7). Run before any review is committed.

Every rule returns readable problems (an empty list means the answer passes):
  * a cited passage id must be one of the passages the model was given (ids outside the packet are rejected);
  * a finding's excerpt must be a short, contiguous quotation of its cited passage. Matching ignores case,
    whitespace, hyphenation at line ends, ligatures, and quote or dash styles, but nothing else;
  * every number in a claim, its conditions, or a statement must occur in a passage it cites (so an invented metric
    such as "97.3% accuracy" is rejected unless the cited evidence says 97.3). Numbers that only label things
    ("Table 2", "Section 3", "Eq. 4") are exempt. Two spellings of the SAME value count as occurring: a number word
    in the passage ("three" backs 3) and a fraction stated as a percentage or back ("0.924" backs 92.4).
Exact matching checks attribution: it shows the words are in the paper. It cannot show the interpretation is sound;
the note says so, and ambiguous passages are flagged by the reviewer itself.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation


MINIMUM_EXCERPT_WORDS = 3
MAXIMUM_EXCERPT_WORDS = 60
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
LABEL_BEFORE_NUMBER_PATTERN = re.compile(
    r"(?i)\b(?:table|tab\.|figure|fig\.|section|sec\.|§|eq\.|equation|eqn\.|theorem|lemma|corollary|definition|"
    r"proposition|appendix|algorithm|alg\.|line|step|chapter|page|p\.|pp\.|ref\.|reference|v)\s*\(?$")
PASSAGE_ID_TOKEN_PATTERN = re.compile(r"\b[APT]\d{1,5}\b")
# Number words a passage may use for a value the model writes in digits ("three baselines" backs "3 baselines").
NUMBER_WORD_VALUES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                      "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                      "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
                      "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                      "ninety": 90, "hundred": 100, "thousand": 1000, "single": 1, "twice": 2, "half": 0.5,
                      "double": 2, "triple": 3}
NUMBER_WORD_PATTERN = re.compile(r"(?i)\b(" + "|".join(NUMBER_WORD_VALUES) + r")\b")
PERCENT_SCALE = Decimal(100)
CHARACTER_EQUIVALENTS = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-",
                                       "—": "-", "−": "-", "‐": "-", "‑": "-", "·": ".", " ": " "})


def normalize_for_matching(text: str) -> str:
    """Comparison form: NFKC, unified quotes and dashes, hyphenated line breaks joined, lowercase, one space."""
    normalized_text = unicodedata.normalize("NFKC", text).translate(CHARACTER_EQUIVALENTS).replace("­", "")
    normalized_text = re.sub(r"(\w)-\s+(\w)", r"\1\2", normalized_text)  # "moni- tor" -> "monitor"
    return " ".join(normalized_text.casefold().split())


PassageTextById = Mapping[str, str]  # passage id -> passage text, exactly as the model was given it


def find_unknown_passage_ids(cited_passage_ids: Iterable[str], passage_text_by_id: PassageTextById,
                             where: str) -> list[str]:
    return [f"{where}: passage id '{passage_id}' is not in the input packet"
            for passage_id in dict.fromkeys(cited_passage_ids) if passage_id not in passage_text_by_id]


def check_excerpt_matches_passage(excerpt: str, passage_id: str, passage_text: str, where: str) -> list[str]:
    excerpt_word_count = len(excerpt.split())
    if excerpt_word_count < MINIMUM_EXCERPT_WORDS:
        return [f"{where}: excerpt must quote at least {MINIMUM_EXCERPT_WORDS} words of passage {passage_id}"]
    if excerpt_word_count > MAXIMUM_EXCERPT_WORDS:
        return [f"{where}: excerpt is {excerpt_word_count} words; quote at most {MAXIMUM_EXCERPT_WORDS}"]
    if "..." in excerpt or "…" in excerpt:
        return [f"{where}: excerpt must be one contiguous quotation without '...'"]
    if normalize_for_matching(excerpt.strip(" \"'")) not in normalize_for_matching(passage_text):
        return [f"{where}: excerpt does not occur in passage {passage_id}"]
    return []


def find_numbers_needing_evidence(text: str) -> list[str]:
    """Numbers stated in `text` that must be supported, normalized ('1,000' -> '1000', '−3' -> '-3')."""
    text_without_passage_ids = PASSAGE_ID_TOKEN_PATTERN.sub(" ", text)
    numbers: list[str] = []
    for number_match in NUMBER_PATTERN.finditer(text_without_passage_ids):
        if LABEL_BEFORE_NUMBER_PATTERN.search(text_without_passage_ids[:number_match.start()]):
            continue
        numbers.append(_normalize_number(number_match.group()))
    return list(dict.fromkeys(numbers))


def check_numbers_occur_in_passages(stated_text: str, cited_passage_ids: list[str],
                                    passage_text_by_id: PassageTextById, where: str) -> list[str]:
    """Each number in `stated_text` must occur in at least one cited passage (ids are assumed known)."""
    stated_numbers = find_numbers_needing_evidence(stated_text)
    if not stated_numbers:
        return []
    if not cited_passage_ids:
        return [f"{where}: states {', '.join(stated_numbers)} but cites no passage"]
    evidence_numbers: set[str] = set()
    for passage_id in cited_passage_ids:
        evidence_numbers |= find_numbers_backed_by_passage(passage_text_by_id[passage_id])
    unsupported_numbers = [number for number in stated_numbers if number not in evidence_numbers
                           and number.lstrip("+-") not in evidence_numbers]
    if unsupported_numbers:
        return [f"{where}: the number(s) {', '.join(unsupported_numbers)} do not occur in the cited passage(s) "
                f"{', '.join(cited_passage_ids)}"]
    return []


def find_numbers_backed_by_passage(passage_text: str) -> set[str]:
    """Every normalized value the passage states: its digits (with and without a sign), its number words, and each
    value rescaled between a fraction and a percentage (0.924 <-> 92.4), which is the same quantity."""
    stated_values = {_normalize_number(number) for number in NUMBER_PATTERN.findall(passage_text)}
    stated_values |= {_normalize_number(str(NUMBER_WORD_VALUES[word.lower()]))
                      for word in NUMBER_WORD_PATTERN.findall(passage_text)}
    backed_values = set(stated_values)
    for value_text in stated_values:
        try:
            value = Decimal(value_text)
        except InvalidOperation:
            continue
        backed_values.add(_normalize_number(format(value * PERCENT_SCALE, "f")))
        backed_values.add(_normalize_number(format(value / PERCENT_SCALE, "f")))
    return backed_values | {value.lstrip("+-") for value in backed_values}


def check_cited_statement(statement_text: str, cited_passage_ids: list[str], passage_text_by_id: PassageTextById,
                          where: str) -> list[str]:
    """Known ids, and numbers backed by the cited passages (for statements without an excerpt)."""
    problems = find_unknown_passage_ids(cited_passage_ids, passage_text_by_id, where)
    if problems:
        return problems
    return check_numbers_occur_in_passages(statement_text, cited_passage_ids, passage_text_by_id, where)


def check_quoted_finding(claim_text: str, passage_id: str, excerpt: str, passage_text_by_id: PassageTextById,
                         where: str) -> list[str]:
    """A finding {claim, passage_id, excerpt}: known id, matching excerpt, and numbers backed by that passage."""
    if passage_id not in passage_text_by_id:
        return [f"{where}: passage id '{passage_id}' is not in the input packet"]
    return (check_excerpt_matches_passage(excerpt, passage_id, passage_text_by_id[passage_id], where)
            + check_numbers_occur_in_passages(claim_text, [passage_id], passage_text_by_id, where))


def _normalize_number(number_text: str) -> str:
    normalized_number = number_text.replace("−", "-").replace(",", "")
    if "." in normalized_number:
        normalized_number = normalized_number.rstrip("0").rstrip(".") or "0"
    return normalized_number
