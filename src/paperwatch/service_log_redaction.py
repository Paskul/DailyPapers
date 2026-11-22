"""Keep keys, signed URLs, private queries, and raw paper or model text out of service logs (plan part 10).

Under systemd, everything a paperwatch process prints or logs ends up in the journal, which other administrators,
backups, and crash reports may see. The full details of every failure stay in the database (jobs.last_error,
model_calls.error_message, outbound_requests), which only the paperwatch accounts and the operator can read. So the
journal gets a redacted, bounded line:
  * credential-like values in URLs and headers become "[redacted]" (api_key=, key=, token=, signature=, X-Amz-*,
    Authorization / Zotero-API-Key headers);
  * long quoted fragments are replaced by "[text omitted]": validation messages quote the model's answer or the
    paper's passages, which are untrusted and may contain private topic wording;
  * each message is capped at MAXIMUM_SERVICE_LOG_LINE_CHARACTERS.
`install_service_log_redaction` also keeps the HTTP libraries below INFO level, because httpx logs every request
URL at INFO, and an OpenAlex key travels as a query parameter.
"""

from __future__ import annotations

import logging
import re

REDACTED_VALUE_TEXT = "[redacted]"
OMITTED_TEXT_MARKER = "[text omitted]"
MAXIMUM_SERVICE_LOG_LINE_CHARACTERS = 600
# A quoted fragment longer than this is assumed to be paper or model text, not an identifier or a short reason.
MAXIMUM_KEPT_QUOTED_CHARACTERS = 60
# Libraries that would log full request URLs (with query-parameter credentials) at INFO or DEBUG.
CHATTY_HTTP_LOGGER_NAMES = ("httpx", "httpcore", "urllib3")

SECRET_QUERY_PARAMETER_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|key|access[_-]?token|token|signature|sig|x-amz-[a-z-]+|x-goog-[a-z-]+)=([^&\s'\"<>]+)"
)
SECRET_HEADER_PATTERN = re.compile(
    r"(?i)\b(authorization|zotero-api-key|x-api-key)(['\"]?\s*[:=]\s*['\"]?)"
    r"(bearer\s+)?([^\s'\",}]+)"
)
QUOTED_FRAGMENT_PATTERN = re.compile(
    r"'([^'\n]{%d,})'|\"([^\"\n]{%d,})\"" % (MAXIMUM_KEPT_QUOTED_CHARACTERS + 1, MAXIMUM_KEPT_QUOTED_CHARACTERS + 1)
)


def redact_service_log_text(text: str) -> str:
    """The form of `text` that may be written to a service log."""
    redacted_text = SECRET_QUERY_PARAMETER_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED_VALUE_TEXT}", text)
    redacted_text = SECRET_HEADER_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{match.group(3) or ''}{REDACTED_VALUE_TEXT}", redacted_text
    )
    redacted_text = QUOTED_FRAGMENT_PATTERN.sub(OMITTED_TEXT_MARKER, redacted_text)
    if len(redacted_text) > MAXIMUM_SERVICE_LOG_LINE_CHARACTERS:
        redacted_text = redacted_text[:MAXIMUM_SERVICE_LOG_LINE_CHARACTERS] + " … (full text in the database)"
    return redacted_text


class ServiceLogRedactionFilter(logging.Filter):
    """Rewrites every log record's final message through redact_service_log_text."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered_message = record.getMessage()
        except Exception:  # a malformed record: keep only its template
            rendered_message = str(record.msg)
        record.msg = redact_service_log_text(rendered_message)
        record.args = None
        if record.exc_info:  # tracebacks can quote paper text in exception messages; keep the type only
            exception_type = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            record.msg += f" ({exception_type}; traceback omitted from the service log)"
            record.exc_info = None
            record.exc_text = None
        return True


def install_service_log_redaction() -> None:
    """Call once after logging.basicConfig: redact every handler of the root logger, silence HTTP request logs."""
    redaction_filter = ServiceLogRedactionFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(redaction_filter)
    for logger_name in CHATTY_HTTP_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
