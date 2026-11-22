"""One inference request at a time across every role and every paperwatch process.

An exclusive flock on state/inference.lock. The kernel releases it automatically if the holder dies, so a crashed
worker can never wedge the queue.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from paperwatch.errors import InferenceLockTimeout

LOCK_RETRY_INTERVAL_SECONDS = 0.5


@contextmanager
def hold_inference_lock(lock_file_path: Path, maximum_wait_seconds: float) -> Iterator[None]:
    with lock_file_path.open("a") as lock_file:
        wait_deadline = time.monotonic() + maximum_wait_seconds
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= wait_deadline:
                    raise InferenceLockTimeout(
                        f"waited {maximum_wait_seconds:.0f} s for the inference lock {lock_file_path}; another "
                        "paperwatch process is still using the model"
                    ) from None
                time.sleep(LOCK_RETRY_INTERVAL_SECONDS)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
