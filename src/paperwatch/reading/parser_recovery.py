"""Did the text extractor recover tables and equations? A heuristic, recorded with every full-text review.

pdftotext keeps the words of a table but loses its grid, and it flattens equations into a line of symbols. The plan
asks the review to record whether the parser could recover them, so the reader knows how far a numeric claim from a
table can be trusted. This module only COUNTS signs in the raw extracted text:
  * table captions ("Table 3: ..."), and table-like lines (three or more numbers on a short line);
  * equation-like lines (mathematical symbols, or a trailing equation number such as "(4)").
The reviewer roles may add passages they found unreadable (`parser_problems`); both are shown in the note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TABLE_CAPTION_PATTERN = re.compile(r"^\s*(?:Table|TABLE)\s+(?:[0-9]{1,3}|[IVX]{1,5})\s*[.:|]")
NUMBER_TOKEN_PATTERN = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)?%?(?![\w])")
MATHEMATICAL_SYMBOLS = frozenset("=≤≥≠≈∑∏∫∀∃∈∉⊂⊆∪∩→⇒⇔∧∨¬∞∂∇λμσθαβγδεπφψωΔΣΠΩ")
TRAILING_EQUATION_NUMBER_PATTERN = re.compile(r"\(\d{1,3}[a-z]?\)\s*$")
MAXIMUM_WORDS_OF_TABLE_LINE = 16
MINIMUM_NUMBERS_OF_TABLE_LINE = 3
MINIMUM_SYMBOLS_OF_EQUATION_LINE = 2
TABLE_LINES_PER_CAPTION_FOR_RECOVERED_ROWS = 2


@dataclass(frozen=True)
class ParserRecoveryAssessment:
    table_captions_found: int
    table_like_lines_found: int
    equation_like_lines_found: int

    @property
    def tables_summary(self) -> str:
        if self.table_captions_found == 0:
            return ("no table captions found" if self.table_like_lines_found == 0 else
                    f"no captions, but {self.table_like_lines_found} table-like line(s); columns may be merged")
        if self.table_like_lines_found >= TABLE_LINES_PER_CAPTION_FOR_RECOVERED_ROWS * self.table_captions_found:
            return (f"{self.table_captions_found} table(s); rows recovered as text, column alignment lost "
                    "(check numbers against the source)")
        return f"{self.table_captions_found} table caption(s), but table bodies were mostly not recovered"

    @property
    def equations_summary(self) -> str:
        if self.equation_like_lines_found == 0:
            return "no equation-like lines found"
        return (f"{self.equation_like_lines_found} equation-like line(s); layout, sub/superscripts and some "
                "symbols may be lost")

    def to_record(self) -> dict[str, object]:
        return {"table_captions_found": self.table_captions_found,
                "table_like_lines_found": self.table_like_lines_found,
                "equation_like_lines_found": self.equation_like_lines_found,
                "tables": self.tables_summary, "equations": self.equations_summary}


def assess_parser_recovery(extracted_text: str) -> ParserRecoveryAssessment:
    table_captions = table_like_lines = equation_like_lines = 0
    for raw_line in extracted_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if TABLE_CAPTION_PATTERN.match(line):
            table_captions += 1
            continue
        if len(line.split()) <= MAXIMUM_WORDS_OF_TABLE_LINE and \
                len(NUMBER_TOKEN_PATTERN.findall(line)) >= MINIMUM_NUMBERS_OF_TABLE_LINE:
            table_like_lines += 1
            continue
        symbol_count = sum(character in MATHEMATICAL_SYMBOLS for character in line)
        if symbol_count >= MINIMUM_SYMBOLS_OF_EQUATION_LINE or \
                (symbol_count >= 1 and TRAILING_EQUATION_NUMBER_PATTERN.search(line)):
            equation_like_lines += 1
    return ParserRecoveryAssessment(table_captions, table_like_lines, equation_like_lines)
