"""Where downloaded source bytes live: state/cache/sources/<edition_id>/<sha256 prefix>.<pdf|xml>, and the text the
sandboxed extractor produced from them: state/cache/sources/<edition_id>/<source sha256 prefix>.txt (plan part 5).

File and directory names are built only from internal ids and content hashes, never from titles, provider ids, or
URLs (plan part 3: untrusted text stays out of filenames). The extractor accepts only paths that pass
`require_cached_source_file`, so it can never be pointed at an arbitrary file.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from paperwatch.errors import SourceAccessDenied
from paperwatch.rendering.file_output import write_bytes_atomically

SOURCE_FILES_SUBDIRECTORY = "sources"
INTERNAL_ID_PATTERN = re.compile(r"^[a-z]_[0-9a-f]{16}$")
CACHED_SOURCE_FILE_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|xml)$")
FILE_SUFFIX_BY_CONTENT_TYPE = {"application/pdf": ".pdf", "application/xml": ".xml", "text/xml": ".xml",
                               "application/jats+xml": ".xml", "application/tei+xml": ".xml"}
SHA256_PREFIX_CHARACTERS = 32
EXTRACTED_TEXT_SUFFIX = ".txt"
SOURCE_FILE_MODE = 0o640  # the collector writes; the worker (same group) reads extracted text later (part 7)


@dataclass(frozen=True)
class CachedSourceFile:
    edition_id: str
    file_path: Path
    content_sha256: str
    content_type: str
    byte_count: int


def store_source_bytes(cache_dir: Path, edition_id: str, body_bytes: bytes, content_type: str) -> CachedSourceFile:
    """Write the bytes atomically under the edition's directory. The same bytes always land at the same path."""
    if not INTERNAL_ID_PATTERN.match(edition_id):
        raise SourceAccessDenied(f"'{edition_id}' is not an internal edition id")
    if content_type not in FILE_SUFFIX_BY_CONTENT_TYPE:
        raise SourceAccessDenied(f"content type '{content_type}' cannot be stored as a source file")
    content_sha256 = hashlib.sha256(body_bytes).hexdigest()
    file_path = cache_dir / SOURCE_FILES_SUBDIRECTORY / edition_id / \
        f"{content_sha256[:SHA256_PREFIX_CHARACTERS]}{FILE_SUFFIX_BY_CONTENT_TYPE[content_type]}"
    write_bytes_atomically(file_path, body_bytes, file_mode=SOURCE_FILE_MODE)
    return CachedSourceFile(edition_id, file_path, content_sha256, content_type, len(body_bytes))


def require_cached_source_file(cache_dir: Path, candidate_path: Path) -> Path:
    """Return the resolved path if it is a regular source file inside the cache with a cache-generated name."""
    resolved_cache_dir = (cache_dir / SOURCE_FILES_SUBDIRECTORY).resolve()
    resolved_path = candidate_path.resolve()
    if resolved_path.parent.parent != resolved_cache_dir or \
            not INTERNAL_ID_PATTERN.match(resolved_path.parent.name) or \
            not CACHED_SOURCE_FILE_NAME_PATTERN.match(resolved_path.name):
        raise SourceAccessDenied(f"{candidate_path} is not a cached source file "
                                 f"({resolved_cache_dir}/<edition id>/<hash>.pdf|xml)")
    if candidate_path.is_symlink() or not resolved_path.is_file():
        raise SourceAccessDenied(f"{candidate_path} is missing or not a regular file")
    return resolved_path


def store_extracted_text(cache_dir: Path, edition_id: str, source_sha256: str, extracted_text: str) -> Path:
    """Save the extractor's text next to its source file. The name comes from the SOURCE hash, so re-extracting the
    same file overwrites the same path. A .txt file can never be fed back to the extractor (pdf|xml only)."""
    if not INTERNAL_ID_PATTERN.match(edition_id) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise SourceAccessDenied("extracted text needs an internal edition id and a source SHA-256")
    text_path = cache_dir / SOURCE_FILES_SUBDIRECTORY / edition_id / \
        f"{source_sha256[:SHA256_PREFIX_CHARACTERS]}{EXTRACTED_TEXT_SUFFIX}"
    write_bytes_atomically(text_path, extracted_text.encode("utf-8"), file_mode=SOURCE_FILE_MODE)
    return text_path


def cached_edition_directory(cache_dir: Path, edition_id: str) -> Path:
    if not INTERNAL_ID_PATTERN.match(edition_id):
        raise SourceAccessDenied(f"'{edition_id}' is not an internal edition id")
    return cache_dir / SOURCE_FILES_SUBDIRECTORY / edition_id
