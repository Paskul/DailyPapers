"""Recorded papers from a local JSON file. Used for the first vertical test and for offline replays.

File format: one object or a list of objects with these keys (only provider, provider_paper_id and title required):
    {"provider": "fixture", "provider_paper_id": "demo-0001", "title": "...", "authors": ["..."],
     "abstract": "...", "first_published_date": "2026-11-17", "latest_revision_date": null, "source_version": "v1",
     "linked_identifiers": ["doi:10.1109/example.2026.1", "arxiv:2411.01234"]}

`linked_identifiers` are "scheme:value" strings (split at the first colon) that the recorded provider stated for the
same work. The discovery route of every record is "fixture:<file name>", so two files are two discovery routes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperwatch.errors import SourceInputError
from paperwatch.sources.types import CollectedPaperRecord, RawIdentifier

REQUIRED_FIXTURE_KEYS = ("provider", "provider_paper_id", "title")
OPTIONAL_FIXTURE_KEYS = (
    "authors",
    "abstract",
    "first_published_date",
    "latest_revision_date",
    "source_version",
    "linked_identifiers",
)


class RecordedFixtureSource:
    def __init__(self, fixture_file_path: Path) -> None:
        self.fixture_file_path = fixture_file_path
        self.source_name = f"fixture:{fixture_file_path.name}"

    def fetch_records(self) -> list[CollectedPaperRecord]:
        try:
            raw_fixture = json.loads(self.fixture_file_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise SourceInputError(f"fixture file {self.fixture_file_path} does not exist") from None
        except (OSError, json.JSONDecodeError) as read_error:
            raise SourceInputError(f"cannot read fixture file {self.fixture_file_path}: {read_error}") from read_error
        raw_entries = raw_fixture if isinstance(raw_fixture, list) else [raw_fixture]
        return [self._convert_entry(entry_index, raw_entry) for entry_index, raw_entry in enumerate(raw_entries)]

    def _convert_entry(self, entry_index: int, raw_entry: Any) -> CollectedPaperRecord:
        entry_label = f"{self.fixture_file_path.name} entry #{entry_index + 1}"
        if not isinstance(raw_entry, dict):
            raise SourceInputError(f"{entry_label} is not a JSON object")
        unknown_keys = set(raw_entry) - set(REQUIRED_FIXTURE_KEYS) - set(OPTIONAL_FIXTURE_KEYS)
        if unknown_keys:
            raise SourceInputError(f"{entry_label} has unknown key(s) {sorted(unknown_keys)}")
        for required_key in REQUIRED_FIXTURE_KEYS:
            if not isinstance(raw_entry.get(required_key), str) or not raw_entry[required_key].strip():
                raise SourceInputError(f"{entry_label} needs a non-empty string '{required_key}'")
        authors = raw_entry.get("authors", [])
        if not isinstance(authors, list) or not all(isinstance(author, str) for author in authors):
            raise SourceInputError(f"{entry_label}: 'authors' must be a list of strings")
        return CollectedPaperRecord(
            provider=raw_entry["provider"].strip().lower(),
            provider_paper_id=raw_entry["provider_paper_id"].strip(),
            title=raw_entry["title"].strip(),
            authors=tuple(authors),
            abstract=(raw_entry.get("abstract") or "").strip() or None,
            first_published_date=raw_entry.get("first_published_date"),
            latest_revision_date=raw_entry.get("latest_revision_date"),
            source_version=raw_entry.get("source_version"),
            discovery_route=self.source_name,
            linked_identifiers=_parse_linked_identifiers(entry_label, raw_entry.get("linked_identifiers", [])),
        )


def _parse_linked_identifiers(entry_label: str, raw_linked_identifiers: Any) -> tuple[RawIdentifier, ...]:
    if not isinstance(raw_linked_identifiers, list):
        raise SourceInputError(f"{entry_label}: 'linked_identifiers' must be a list of \"scheme:value\" strings")
    parsed_identifiers = []
    for raw_identifier in raw_linked_identifiers:
        if not isinstance(raw_identifier, str) or ":" not in raw_identifier:
            raise SourceInputError(f'{entry_label}: linked identifier {raw_identifier!r} is not "scheme:value"')
        scheme, _, value = raw_identifier.partition(":")
        parsed_identifiers.append(RawIdentifier(scheme=scheme, value=value))
    return tuple(parsed_identifiers)
