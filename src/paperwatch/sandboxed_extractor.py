"""Run the pinned PDF/XML text extractor in rootless Podman and return its text (plan part 3).

The container gets exactly what the plan allows and nothing more:
  * the digest-pinned image only (`--pull=never`: never downloaded at run time; build it once with
    deploy/build_extractor_image.sh),
  * no network (`--network=none`), a read-only root filesystem, no capabilities, no privilege escalation,
  * ONE read-only input: a cached source file (named by internal ids, see paperwatch.source_cache),
  * bounded scratch space (tmpfs), memory, CPU, process count, and wall-clock time,
  * no environment from the host, no home directory, no database, no model socket, no keys.
The image's fixed entrypoint writes plain text to stdout (PDF pages separated by form feeds). Output is bounded too.
A timeout kills the container and is charged to that one source file; the next file extracts normally.
"""

from __future__ import annotations

import math
import os
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from paperwatch.config import ExtractorSettings
from paperwatch.errors import ExtractionError, ExtractionTimedOut
from paperwatch.source_cache import require_cached_source_file

CONTAINER_INPUT_PATH = "/input/source"
CONTAINER_NAME_PREFIX = "paperwatch-extract-"
CONTAINER_PROCESS_LIMIT = 64
MAXIMUM_STDERR_BYTES = 16_384
CONTAINER_REMOVAL_TIMEOUT_SECONDS = 30
BYTES_PER_MEGABYTE = 1024 * 1024
# The only host variables the container runtime itself needs to find its rootless storage. Never passed into the
# container (podman passes no host environment unless asked), and never containing provider keys.
RUNTIME_ENVIRONMENT_VARIABLES = ("PATH", "HOME", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "LANG")


@dataclass(frozen=True)
class ExtractedText:
    text: str
    extractor_image: str  # the digest-pinned image that produced it (recorded with passages in part 7)
    duration_seconds: float


def build_extractor_container_command(
    extractor_settings: ExtractorSettings, input_file_path: Path, container_name: str
) -> list[str]:
    """The complete, fixed `podman run` command. Nothing in it comes from paper text."""
    if "," in str(input_file_path) or "=" in str(input_file_path):
        raise ExtractionError(f"cache path {input_file_path} contains ',' or '=', which a bind mount cannot express")
    memory_limit = f"{extractor_settings.memory_megabytes}m"
    return [
        extractor_settings.container_runtime,
        "run",
        "--rm",
        f"--name={container_name}",
        "--pull=never",  # preinstalled image only; no downloads at run time
        "--network=none",  # no network at all
        "--read-only",  # read-only root filesystem
        "--cap-drop=all",
        "--security-opt=no-new-privileges",
        "--no-hosts",
        "--hostname=extractor",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        f"--pids-limit={CONTAINER_PROCESS_LIMIT}",
        f"--cpus={extractor_settings.cpu_limit}",
        f"--tmpfs=/tmp:rw,size={extractor_settings.temporary_space_megabytes}m,mode=1777,nosuid,nodev,noexec",
        f"--timeout={math.ceil(extractor_settings.timeout_seconds)}",  # runtime-side kill as a second guard
        f"--mount=type=bind,source={input_file_path},target={CONTAINER_INPUT_PATH},readonly",
        extractor_settings.image,  # the image's fixed entrypoint reads /input/source
    ]


def extract_text_in_sandbox(
    source_cache_dir: Path, source_file_path: Path, extractor_settings: ExtractorSettings
) -> ExtractedText:
    if not extractor_settings.image:
        raise ExtractionError(
            "the text extractor image is not configured",
            operator_hint="run deploy/build_extractor_image.sh and set extractor.image in "
            "config.toml to the image id it prints",
        )
    input_file_path = require_cached_source_file(source_cache_dir, source_file_path)
    input_bytes = input_file_path.stat().st_size
    if input_bytes > extractor_settings.max_input_megabytes * BYTES_PER_MEGABYTE:
        raise ExtractionError(f"source file of {input_bytes} bytes exceeds extractor.max_input_megabytes")

    container_name = CONTAINER_NAME_PREFIX + secrets.token_hex(6)
    command = build_extractor_container_command(extractor_settings, input_file_path, container_name)
    started_monotonic = time.monotonic()
    try:
        extractor_process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_runtime_environment(),
            close_fds=True,
        )
    except OSError as launch_error:
        raise ExtractionError(
            f"cannot start the container runtime '{extractor_settings.container_runtime}': {launch_error}",
            operator_hint="install podman (rootless) for this account",
        ) from None

    stdout_reader = BoundedStreamReader(
        extractor_process.stdout,
        extractor_settings.max_output_megabytes * BYTES_PER_MEGABYTE,
        on_overflow=extractor_process.kill,
    )
    stderr_reader = BoundedStreamReader(extractor_process.stderr, MAXIMUM_STDERR_BYTES, on_overflow=None)
    try:
        extractor_process.wait(timeout=extractor_settings.timeout_seconds)
    except subprocess.TimeoutExpired:
        _stop_container(extractor_process, extractor_settings.container_runtime, container_name)
        raise ExtractionTimedOut(
            f"extractor exceeded its {extractor_settings.timeout_seconds:g} s limit; the container was stopped"
        ) from None
    finally:
        stdout_reader.join()
        stderr_reader.join()

    if stdout_reader.overflowed:
        _stop_container(extractor_process, extractor_settings.container_runtime, container_name)
        raise ExtractionError(
            f"extractor output exceeds extractor.max_output_megabytes ({extractor_settings.max_output_megabytes} MB)"
        )
    if extractor_process.returncode != 0:
        error_text = stderr_reader.collected_bytes.decode("utf-8", errors="replace").strip()
        raise ExtractionError(
            f"extractor exited with status {extractor_process.returncode}: "
            f"{' '.join(error_text.split())[:300] or 'no error output'}"
        )
    return ExtractedText(
        text=stdout_reader.collected_bytes.decode("utf-8", errors="replace"),
        extractor_image=extractor_settings.image,
        duration_seconds=time.monotonic() - started_monotonic,
    )


class BoundedStreamReader:
    """Drain a pipe on a background thread, keeping at most `maximum_bytes`; call `on_overflow` once if exceeded."""

    def __init__(self, stream: IO[bytes] | None, maximum_bytes: int, on_overflow) -> None:
        self._stream = stream
        self._maximum_bytes = maximum_bytes
        self._on_overflow = on_overflow
        self._collected = bytearray()
        self.overflowed = False
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    @property
    def collected_bytes(self) -> bytes:
        return bytes(self._collected)

    def join(self) -> None:
        self._thread.join(timeout=CONTAINER_REMOVAL_TIMEOUT_SECONDS)

    def _drain(self) -> None:
        if self._stream is None:
            return
        read_available_bytes = getattr(self._stream, "read1", self._stream.read)  # read1: return what is ready
        while chunk := read_available_bytes(65_536):
            if len(self._collected) + len(chunk) > self._maximum_bytes:
                self.overflowed = True
                if self._on_overflow is not None:
                    self._on_overflow()
                break
            self._collected.extend(chunk)
        self._stream.close()


def _stop_container(extractor_process: subprocess.Popen, container_runtime: str, container_name: str) -> None:
    """Kill the runtime client and make sure the container itself is gone (it could outlive its client)."""
    extractor_process.kill()
    extractor_process.wait()
    try:
        subprocess.run(
            [container_runtime, "rm", "--force", container_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_runtime_environment(),
            timeout=CONTAINER_REMOVAL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass  # --rm and --timeout inside the runtime remain as the fallback


def _runtime_environment() -> dict[str, str]:
    return {name: os.environ[name] for name in RUNTIME_ENVIRONMENT_VARIABLES if name in os.environ}
