"""The lexical screening signal: which approved public phrases occur verbatim in a paper's title and abstract.

A broad semantic match can bury an exact term ("Dafny", "signal temporal logic"), so the phrase signal checks for the
exact words in order. Matching is deliberately simple and documented:
  * text is lower-cased and split into runs of letters and digits, so case, punctuation, and hyphens do not matter
    ("Multi-Robot" matches "multi robot");
  * a word longer than three letters loses one trailing "s" (not "ss"), so "monitors" matches "monitor";
  * a phrase matches when its words occur consecutively in the text.
The signal is the fraction of the topic's phrases that match (plan part 6). A topic without phrases has no phrase
signal; the scoring formula then leaves it out instead of counting it as zero.
"""

from __future__ import annotations

import re

WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
MINIMUM_WORD_LENGTH_FOR_PLURAL_STRIPPING = 4


def split_into_match_words(text: str) -> tuple[str, ...]:
    return tuple(_strip_simple_plural(word) for word in WORD_PATTERN.findall(text.casefold()))


def _strip_simple_plural(word: str) -> str:
    if len(word) >= MINIMUM_WORD_LENGTH_FOR_PLURAL_STRIPPING and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def phrase_occurs_in_words(phrase_words: tuple[str, ...], text_words: tuple[str, ...]) -> bool:
    if not phrase_words or len(phrase_words) > len(text_words):
        return False
    phrase_length = len(phrase_words)
    return any(
        text_words[start : start + phrase_length] == phrase_words
        for start in range(len(text_words) - phrase_length + 1)
        if text_words[start] == phrase_words[0]
    )


def find_matched_phrases(approved_phrases: tuple[str, ...], text_words: tuple[str, ...]) -> tuple[str, ...]:
    """The approved phrases (as written in topics.toml) that occur in the text, in configuration order."""
    return tuple(
        phrase for phrase in approved_phrases if phrase_occurs_in_words(split_into_match_words(phrase), text_words)
    )
