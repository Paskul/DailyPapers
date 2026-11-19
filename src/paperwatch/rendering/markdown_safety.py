"""Make untrusted text (titles, abstracts, model output) inert inside Markdown.

Neutralises HTML tags, links/images, Obsidian embeds ([[...]] and ![[...]]), emphasis, headings, tables, and code
spans, and collapses newlines for inline use. Bare web addresses are defanged ("https[:]//", "www[.]") so Markdown
and Obsidian never autolink them. External URLs are never taken from paper or model text: they are built from
trusted provider identifiers only.
"""

from __future__ import annotations

import re
import urllib.parse

from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME, OPENALEX_SCHEME, is_canonical_identifier

MARKDOWN_SPECIAL_CHARACTERS_PATTERN = re.compile(r"([\\`*_{}\[\]()#+!|~>])")
URL_SCHEME_SEPARATOR_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.-]{1,15}):(//)")  # https://, ftp://, file://
WWW_PREFIX_PATTERN = re.compile(r"(?i)\b(www)\.")
TRUSTED_LINK_SCHEMES = (ARXIV_SCHEME, DOI_SCHEME, OPENALEX_SCHEME)


def escape_untrusted_inline_text(untrusted_text: str | None) -> str:
    if not untrusted_text:
        return ""
    single_line_text = " ".join(untrusted_text.split())
    single_line_text = defang_web_addresses(single_line_text)
    html_neutralised_text = single_line_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return MARKDOWN_SPECIAL_CHARACTERS_PATTERN.sub(r"\\\1", html_neutralised_text)


def defang_web_addresses(untrusted_text: str) -> str:
    """'https://x.example' -> 'https[:]//x.example', 'www.x.example' -> 'www[.]x.example': readable, not clickable."""
    return WWW_PREFIX_PATTERN.sub(r"\1[.]", URL_SCHEME_SEPARATOR_PATTERN.sub(r"\1[:]\2", untrusted_text))


def build_trusted_source_url(provider: str, provider_paper_id: str) -> str | None:
    """Construct an external link from a canonical identifier, or return None if it is not one we trust.

    Aliases are stored normalized (paperwatch.scholarly_identifiers), and only canonical forms get a link.
    """
    if provider not in TRUSTED_LINK_SCHEMES or not is_canonical_identifier(provider, provider_paper_id):
        return None
    if provider == ARXIV_SCHEME:
        return f"https://arxiv.org/abs/{provider_paper_id}"
    if provider == DOI_SCHEME:
        return f"https://doi.org/{urllib.parse.quote(provider_paper_id, safe='/')}"
    return f"https://openalex.org/{provider_paper_id}"
