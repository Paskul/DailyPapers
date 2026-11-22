"""Atomic file writes: temp file on the same filesystem, fsync, then rename. Readers never see half a file.

Reports and notes are written with OUTPUT_FILE_MODE (0640): the worker account owns them and the shared state group
(which the operator belongs to) can read them. Before 2026-11-21 they kept mkstemp's private 0600, so the operator
could not open the first pilot's report without sudo. `make_output_file_group_readable` fixes such an older file
whenever paperwatch touches it again (it only chmods files the current account owns).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from paperwatch.errors import OutputWriteError

OUTPUT_FILE_MODE = 0o640  # reports and notes: owner read/write, the shared group (the operator) reads


def write_bytes_atomically(destination_path: Path, file_bytes: bytes, *, file_mode: int | None = None) -> None:
    """`file_mode` (e.g. 0o640) is applied before the rename; by default the file keeps mkstemp's private 0600."""
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_path = tempfile.mkstemp(
            dir=destination_path.parent, prefix=f".{destination_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(temporary_fd, "wb") as temporary_file:
                temporary_file.write(file_bytes)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if file_mode is not None:
                os.chmod(temporary_path, file_mode)
            os.replace(temporary_path, destination_path)
        except BaseException:
            Path(temporary_path).unlink(missing_ok=True)
            raise
    except OSError as write_error:
        raise OutputWriteError(f"cannot write {destination_path}: {write_error}") from write_error


def write_output_file_atomically(destination_path: Path, file_bytes: bytes) -> None:
    """A report or note: atomic, and readable by the shared group (OUTPUT_FILE_MODE)."""
    write_bytes_atomically(destination_path, file_bytes, file_mode=OUTPUT_FILE_MODE)


def make_output_file_group_readable(file_path: Path) -> None:
    """Give an existing report or note OUTPUT_FILE_MODE if this account owns it. Best effort: never fails."""
    try:
        file_status = file_path.stat()
        if file_status.st_uid == os.geteuid() and (file_status.st_mode & 0o777) != OUTPUT_FILE_MODE:
            os.chmod(file_path, OUTPUT_FILE_MODE)
    except OSError:
        pass
