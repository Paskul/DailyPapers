"""Find a full-text file the user placed in the local sources folder (plan part 5: "a local permitted file").

The folder (`collection.local_sources_dir`, default <home>/local_sources) holds PDFs or JATS/TEI XML named by the
work's identifier, so no download is needed:
    2401.01234v2.pdf      exactly arXiv version 2         hep-th_9901001v1.pdf   old-style id ('/' becomes '_')
    2401.01234.pdf        some version of that arXiv id   10.1109_tro.2024.123.pdf   a DOI ('/' becomes '_')
Only names COMPUTED from the paper's own trusted identifiers are looked up (case-insensitive); a file name is never
parsed into an identifier, and symbolic links, oversized files, and files that are not PDF or XML are refused. The
bytes are copied into the source cache, so the extractor only ever sees a cached file named by internal ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paperwatch.scholarly_identifiers import ARXIV_SCHEME, DOI_SCHEME, OPENALEX_SCHEME

CONTENT_TYPE_BY_LOCAL_SUFFIX = {".pdf": "application/pdf", ".xml": "application/xml"}
PDF_MAGIC_BYTES = b"%PDF-"
UTF8_BYTE_ORDER_MARK = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class LocalSourceFile:
    file_path: Path
    content_type: str


class LocalSourceFolder:
    """A one-time listing of the folder for this run (an absent folder simply has no files)."""

    def __init__(self, folder_path: Path) -> None:
        self.folder_path = folder_path
        self._regular_file_by_lowercase_name: dict[str, Path] = {}
        if folder_path.is_dir():
            for child_path in folder_path.iterdir():
                if child_path.suffix.lower() in CONTENT_TYPE_BY_LOCAL_SUFFIX and not child_path.is_symlink() and \
                        child_path.is_file():
                    self._regular_file_by_lowercase_name[child_path.name.lower()] = child_path

    @property
    def is_empty(self) -> bool:
        return not self._regular_file_by_lowercase_name

    def find(self, candidate_file_stems: list[str]) -> LocalSourceFile | None:
        for file_stem in candidate_file_stems:
            for suffix, content_type in CONTENT_TYPE_BY_LOCAL_SUFFIX.items():
                found_path = self._regular_file_by_lowercase_name.get(f"{file_stem}{suffix}".lower())
                if found_path is not None:
                    return LocalSourceFile(found_path, content_type)
        return None


def file_stems_for_edition(edition_key: str) -> list[str]:
    """The exact-edition name: 'arxiv:2401.01234@v2' -> ['2401.01234v2']; a DOI edition -> its DOI name."""
    identifier_text, _, version_label = edition_key.partition("@")
    scheme, _, value = identifier_text.partition(":")
    if scheme == ARXIV_SCHEME and version_label.startswith("v"):
        return [f"{value}{version_label}".replace("/", "_")]
    if scheme == DOI_SCHEME:
        return [value.replace("/", "_")]
    if scheme == OPENALEX_SCHEME:
        return [value]
    return []


def file_stems_for_paper(paper_aliases: list[tuple[str, str]]) -> list[str]:
    """Version-free names that stand for the work as a whole: its arXiv id and DOIs."""
    file_stems = []
    for scheme, value in paper_aliases:
        if scheme in (ARXIV_SCHEME, DOI_SCHEME):
            file_stems.append(value.replace("/", "_"))
    return file_stems


def read_local_source_bytes(local_source_file: LocalSourceFile, maximum_bytes: int) -> bytes:
    """Read a bounded copy; raise ValueError if it is too large or its content does not match its suffix."""
    if local_source_file.file_path.is_symlink():
        raise ValueError("is a symbolic link")
    with local_source_file.file_path.open("rb") as local_file:
        file_bytes = local_file.read(maximum_bytes + 1)
    if len(file_bytes) > maximum_bytes:
        raise ValueError(f"is larger than {maximum_bytes} bytes")
    if local_source_file.content_type == "application/pdf" and not file_bytes.startswith(PDF_MAGIC_BYTES):
        raise ValueError("does not start like a PDF file")
    if local_source_file.content_type == "application/xml" and \
            not file_bytes.removeprefix(UTF8_BYTE_ORDER_MARK).lstrip().startswith(b"<"):
        raise ValueError("does not start like an XML file")
    return file_bytes
