"""Atomic file writes: temp file on the same filesystem, fsync, then rename. Readers never see half a file."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from paperwatch.errors import OutputWriteError


def write_bytes_atomically(destination_path: Path, file_bytes: bytes, *, file_mode: int | None = None) -> None:
    """`file_mode` (e.g. 0o640) is applied before the rename; by default the file keeps mkstemp's private 0600."""
    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_path = tempfile.mkstemp(dir=destination_path.parent,
                                                        prefix=f".{destination_path.name}.", suffix=".tmp")
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
