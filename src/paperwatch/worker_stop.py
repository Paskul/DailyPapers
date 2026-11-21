"""Stopping the worker promptly, even in the middle of a long model call (fix after the first pilot, 2026-11-21).

In the first pilot the worker only stopped "after the current job". A long-paper job runs for hours, so at the report
cutoff systemd's TimeoutStopSec ran out, the worker was killed with SIGKILL, and its job stayed `running` until the
lease expired, which then cost the job an attempt.

Now SIGTERM / SIGINT call `request_worker_stop`. It always records the request. If the main thread is waiting for the
model at that moment (the reservation wait, the inference lock, or the HTTP request itself, all marked with
`interruptible_model_wait`), it raises WorkerStopRequested right there. That aborts the request, and llama-server
cancels the generation when the connection closes. Outside such a wait (a short database write, a file write) it only
records the request, and the next model wait raises it before it starts. The worker then puts the job back in the
queue WITHOUT charging an attempt. Validated stage results are already saved, so the next run continues where this one
stopped.

WorkerStopRequested derives from BaseException on purpose: generic `except Exception` handlers (the worker's "a bug
charges an attempt" path) must not swallow it.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class WorkerStopRequested(BaseException):
    """The operator or systemd asked the worker to stop; raised only inside an interruptible model wait."""


_stop_requested = threading.Event()
_interruptible_waits_in_main_thread = 0


def request_worker_stop() -> None:
    """Called from the worker's signal handler (always in the main thread)."""
    _stop_requested.set()
    if _interruptible_waits_in_main_thread > 0:
        raise WorkerStopRequested("stop requested while waiting for the model; the job goes back to the queue")


def stop_was_requested() -> bool:
    return _stop_requested.is_set()


def wait_for_stop_request(timeout_seconds: float) -> bool:
    """Sleep up to `timeout_seconds`, waking at once on a stop request. True if a stop was requested."""
    return _stop_requested.wait(timeout_seconds)


def reset_worker_stop_request() -> None:
    """For a new worker loop in the same process (tests)."""
    global _interruptible_waits_in_main_thread
    _stop_requested.clear()
    _interruptible_waits_in_main_thread = 0


@contextmanager
def interruptible_model_wait() -> Iterator[None]:
    """Mark a wait for the model that a stop request may abort. Raises at once if a stop was already requested.
    Only the main thread counts: signal handlers run there, so only there can the raise land inside this block."""
    global _interruptible_waits_in_main_thread
    in_main_thread = threading.current_thread() is threading.main_thread()
    if _stop_requested.is_set() and in_main_thread:
        raise WorkerStopRequested("stop requested before the model call started; the job goes back to the queue")
    if in_main_thread:
        _interruptible_waits_in_main_thread += 1
    try:
        yield
    finally:
        if in_main_thread:
            _interruptible_waits_in_main_thread -= 1
