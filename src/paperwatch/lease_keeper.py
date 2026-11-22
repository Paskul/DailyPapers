"""Keep a claimed job's lease alive while its slow work (waiting for the inference lock, a long model call) runs.

Leases are deliberately short (worker.lease_seconds), so a crashed worker's job becomes reclaimable quickly. A live
worker therefore renews its lease every worker.lease_renewal_interval_seconds from a background thread.

The thread uses its OWN SQLite connection (connections must not be shared across threads) and only ever runs one
short autocommit UPDATE that matches the lease token, so it can never extend a lease that another worker reclaimed.
If renewal finds the lease gone, `lease_was_lost` turns True; the final lease-checked commit then discards the
result anyway, so a stale worker never overwrites a newer one.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from types import TracebackType

from paperwatch.db import connect_database
from paperwatch.errors import PaperwatchError
from paperwatch.jobs import ClaimedJob, renew_job_lease

LOGGER = logging.getLogger(__name__)


class JobLeaseKeeper:
    """Context manager: `with JobLeaseKeeper(...): run_the_slow_part()` renews the lease until the block exits."""

    def __init__(
        self,
        database_path: Path,
        claimed_job: ClaimedJob,
        *,
        lease_duration_seconds: float,
        renewal_interval_seconds: float,
    ) -> None:
        self._database_path = database_path
        self._claimed_job = claimed_job
        self._lease_duration_seconds = lease_duration_seconds
        self._renewal_interval_seconds = renewal_interval_seconds
        self._stop_requested = threading.Event()
        self._renewal_thread = threading.Thread(
            target=self._renew_until_stopped, daemon=True, name=f"lease-keeper-{claimed_job.job_id}"
        )
        self.lease_was_lost = False
        self.successful_renewal_count = 0

    def __enter__(self) -> JobLeaseKeeper:
        self._renewal_thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop_requested.set()
        self._renewal_thread.join()

    def _renew_until_stopped(self) -> None:
        renewal_connection: sqlite3.Connection | None = None
        try:
            while not self._stop_requested.wait(self._renewal_interval_seconds):
                try:
                    renewal_connection = renewal_connection or connect_database(self._database_path)
                    if not renew_job_lease(
                        renewal_connection,
                        self._claimed_job.job_id,
                        self._claimed_job.lease_token,
                        self._lease_duration_seconds,
                    ):
                        self.lease_was_lost = True
                        LOGGER.warning(
                            "lease on %s was reclaimed by another worker; its result will be discarded",
                            self._claimed_job.job_id,
                        )
                        return
                    self.successful_renewal_count += 1
                except (sqlite3.Error, PaperwatchError) as renewal_error:  # e.g. busy database: try next interval
                    LOGGER.warning("could not renew lease on %s: %s", self._claimed_job.job_id, renewal_error)
        finally:
            if renewal_connection is not None:
                renewal_connection.close()
