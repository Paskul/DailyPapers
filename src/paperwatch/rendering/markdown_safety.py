"""Make untrusted text (titles, abstracts, model output) inert inside Markdown.

Neutralises HTML tags, links/images, Obsidian embeds ([[...]] and ![[...]]), emphasis, headings, tables, and code
spans, and collapses newlines for inline use. External URLs are never taken from paper text: they are built from
trusted provider identifiers only.
"""

from __future__ import annotations

import re
import urllib.parse

MARKDOWN_SPECIAL_CHARACTERS_PATTERN = re.compile(r"([\\`*_{}\[\]()#+!|~>])")
ARXIV_IDENTIFIER_PATTERN = re.compile(r"^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})$")
DOI_IDENTIFIER_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")
OPENALEX_WORK_PATTERN = re.compile(r"^W\d+$")


def escape_untrusted_inline_text(untrusted_text: str | None) -> str:
    if not untrusted_text:
        return ""
    single_line_text = " ".join(untrusted_text.split())
    html_neutralised_text = single_line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return MARKDOWN_SPECIAL_CHARACTERS_PATTERN.sub(r"\\\1", html_neutralised_text)


def build_trusted_source_url(provider: str, provider_paper_id: str) -> str | None:
    """Construct an external link from a validated identifier, or return None if the identifier is not trusted."""
    if provider == "arxiv" and ARXIV_IDENTIFIER_PATTERN.match(provider_paper_id):
        return f"https://arxiv.org/abs/{provider_paper_id}"
    if provider == "doi" and DOI_IDENTIFIER_PATTERN.match(provider_paper_id):
        return f"https://doi.org/{urllib.parse.quote(provider_paper_id, safe='/')}"
    if provider == "openalex" and OPENALEX_WORK_PATTERN.match(provider_paper_id):
        return f"https://openalex.org/{provider_paper_id}"
    return None
