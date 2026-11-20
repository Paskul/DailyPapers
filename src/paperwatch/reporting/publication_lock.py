"""One publication process at a time (plan part 9: "Enforce one date key and one publication process per date").

The date key is the reports table's primary key. The process limit is an exclusive, non-blocking flock on
state/report-publication.lock: a second `report` (two timer launches, or an operator run during the timer's) refuses
at once with ReportPublicationInProgress instead of racing. The kernel releases the lock if the holder dies.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from paperwatch.errors import ReportPublicationInProgress


@contextmanager
def hold_report_publication_lock(lock_file_path: Path) -> Iterator[None]:
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_file_path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ReportPublicationInProgress(
                f"another `paperwatch report` holds {lock_file_path}; it is publishing now",
                operator_hint="wait for it to finish; a rerun afterwards changes nothing") from None
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
