"""`paperwatch reread PAPER_ID [--reason TEXT]`: an explicit operator request for a fresh reading of one paper.

The reread gets its own generation id, so it is a new logical job and produces a separate review; the earlier
review stays untouched. This is the only way to read unchanged text again (plan part 2).
"""

from __future__ import annotations

import argparse
import re

from paperwatch.app_context import AppContext
from paperwatch.catalog import set_paper_status
from paperwatch.db import write_transaction
from paperwatch.dispatch_order import dispatch_order_for_operator_request
from paperwatch.errors import ConfigurationError
from paperwatch.readings import ReadingEnqueueOutcome, enqueue_reading_if_needed, new_reading_generation_id

INTERNAL_PAPER_ID_PATTERN = re.compile(r"^p_[0-9a-f]{16}$")
MAXIMUM_REASON_LENGTH = 500


def run_reread_command(app: AppContext, arguments: argparse.Namespace) -> int:
    paper_id = arguments.paper_id.strip()
    if not INTERNAL_PAPER_ID_PATTERN.match(paper_id):
        raise ConfigurationError(
            f"'{paper_id}' is not an internal paper id (p_ followed by 16 hex digits)",
            operator_hint="`paperwatch status` and the notes in output/papers/ show paper ids",
        )
    if app.connection.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone() is None:
        raise ConfigurationError(f"paper {paper_id} is not in the catalog")
    operator_reason = (arguments.reason or "operator request").strip()[:MAXIMUM_REASON_LENGTH]

    reading_generation = new_reading_generation_id()
    with write_transaction(app.connection):
        enqueue_outcome, reading_input = enqueue_reading_if_needed(
            app.connection,
            paper_id,
            dispatch_order=dispatch_order_for_operator_request(),
            max_attempts=app.config.worker.max_attempts_per_job,
            reading_generation=reading_generation,
            operator_reason=operator_reason,
        )
        if enqueue_outcome == ReadingEnqueueOutcome.QUEUED:
            set_paper_status(app.connection, paper_id, "queued")

    if enqueue_outcome != ReadingEnqueueOutcome.QUEUED or reading_input is None:
        print(f"no reread queued for {paper_id}: {enqueue_outcome}")
        return 1
    print(
        f"queued reread of {paper_id} as generation {reading_generation} "
        f"(edition {reading_input.edition_id}, {reading_input.text_kind}); run `paperwatch work` to process it"
    )
    return 0
