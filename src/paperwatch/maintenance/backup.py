"""Back up a paperwatch home, and check that a backup restores.

"Back up SQLite with its online backup API and retain the config and output. Restore to a fresh directory once."

A backup is itself a complete paperwatch home, so restoring is copying it to a fresh directory and pointing --home at
it:
    <destination>/paperwatch-backup-<UTC time>/
        config.toml, topics.toml, seeds/   the settings (copied as they are)
        state/paperwatch.sqlite3           SQLite's online backup API: a consistent snapshot even while the worker
                                           and collector are writing (no need to stop them)
        state/cache/sources/<edition>/*.txt  the extracted text every full-text review cited; the raw downloads are
                                           replaceable and are left out
        output/                            notes, comparison notes, and daily reports
        BACKUP_MANIFEST.json               when, from where, schema version, row counts, integrity result
NOT included: credentials/ (provider keys are the collector's secret; keep them out of backups) and models/ (pinned
and reinstallable with deploy/install_embedding_model.sh).

The backup is written to a hidden partial directory and renamed when complete, so a crash never leaves something
that looks like a finished backup. It is private to whoever ran it (mode 0700): it holds your private topic
questions.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from paperwatch.config import CONFIG_FILE_NAME, TOPICS_FILE_NAME, AppConfig, load_app_config
from paperwatch.db import read_schema_version
from paperwatch.source_cache import EXTRACTED_TEXT_SUFFIX, SOURCE_FILES_SUBDIRECTORY

BACKUP_DIRECTORY_PREFIX = "paperwatch-backup-"
MANIFEST_FILE_NAME = "BACKUP_MANIFEST.json"
SEEDS_DIRECTORY_NAME = "seeds"
# Tables whose row counts are recorded and compared on restore: what the user would miss if it were lost.
COUNTED_TABLES = (
    "papers",
    "paper_editions",
    "reviews",
    "review_topic_assessments",
    "review_evidence_passages",
    "comparisons",
    "reports",
    "report_items",
    "jobs",
    "model_calls",
)


class BackupError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackupResult:
    backup_dir: Path
    manifest: dict[str, Any]

    def describe(self) -> str:
        counts = self.manifest["row_counts"]
        return (
            f"backup written to {self.backup_dir}: schema v{self.manifest['schema_version']}, "
            f"{counts.get('papers', 0)} papers, {counts.get('reviews', 0)} reviews, "
            f"{counts.get('reports', 0)} reports, {self.manifest['extracted_text_files']} extracted text file(s), "
            f"{self.manifest['output_files']} output file(s); integrity {self.manifest['integrity_check']}"
        )


def create_backup(app_config: AppConfig, destination_dir: Path, *, moment: datetime) -> BackupResult:
    destination_dir = destination_dir.expanduser().resolve()
    home_dir = app_config.home_dir.resolve()
    if destination_dir == home_dir or home_dir in destination_dir.parents:
        raise BackupError(f"the backup destination {destination_dir} must be outside the paperwatch home")
    backup_name = f"{BACKUP_DIRECTORY_PREFIX}{moment:%Y%m%dT%H%M%SZ}"
    final_dir = destination_dir / backup_name
    if final_dir.exists():
        raise BackupError(f"{final_dir} already exists")
    partial_dir = destination_dir / f".partial-{backup_name}"
    destination_dir.mkdir(parents=True, exist_ok=True)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(mode=0o700)
    try:
        _copy_settings(app_config, partial_dir)
        backup_database_path = partial_dir / "state" / app_config.database_path.name
        backup_database_path.parent.mkdir(mode=0o700)
        _snapshot_database(app_config.database_path, backup_database_path)
        extracted_text_files = _copy_extracted_texts(
            app_config.source_cache_dir, partial_dir / "state" / app_config.source_cache_dir.name
        )
        output_files = _copy_output(app_config.output_dir, partial_dir / "output")
        manifest = {
            "created_at": f"{moment:%Y-%m-%dT%H:%M:%SZ}",
            "source_home": str(home_dir),
            "source_paths": {"state_dir": str(app_config.state_dir), "output_dir": str(app_config.output_dir)},
            "extracted_text_files": extracted_text_files,
            "output_files": output_files,
            **_describe_database(backup_database_path),
        }
        if manifest["integrity_check"] != "ok":
            raise BackupError(f"the database snapshot failed its integrity check: {manifest['integrity_check']}")
        (partial_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
        os.chmod(partial_dir / MANIFEST_FILE_NAME, 0o600)
        partial_dir.rename(final_dir)
    except BaseException:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    return BackupResult(final_dir, manifest)


@dataclass(frozen=True)
class RestoreCheck:
    problems: list[str]
    restored_home: Path

    @property
    def passed(self) -> bool:
        return not self.problems


def check_restored_home(restored_home: Path) -> RestoreCheck:
    """Load a copied backup as a paperwatch home and compare it with its manifest (read-only)."""
    problems: list[str] = []
    manifest_path = restored_home / MANIFEST_FILE_NAME
    if not manifest_path.is_file():
        return RestoreCheck([f"{manifest_path} is missing: not a paperwatch backup"], restored_home)
    manifest = json.loads(manifest_path.read_text())
    try:
        app_config = load_app_config(restored_home)
    except Exception as load_error:  # noqa: BLE001  # any config problem means the restore is not usable as is
        return RestoreCheck([f"the restored config does not load: {load_error}"], restored_home)
    for path_name, restored_path in (("state_dir", app_config.state_dir), ("output_dir", app_config.output_dir)):
        if restored_home.resolve() not in restored_path.resolve().parents:
            problems.append(
                f"{path_name} points outside the restored home ({restored_path}); edit [paths] in "
                f"{restored_home / CONFIG_FILE_NAME} before using it"
            )
    restored_description = _describe_database(app_config.database_path)
    for compared_key in ("schema_version", "row_counts", "integrity_check"):
        if restored_description[compared_key] != manifest[compared_key]:
            problems.append(
                f"{compared_key} differs: backup {manifest[compared_key]!r}, restored "
                f"{restored_description[compared_key]!r}"
            )
    restored_output_files = _count_files(app_config.output_dir)
    if restored_output_files != manifest["output_files"]:
        problems.append(f"output files: backup {manifest['output_files']}, restored {restored_output_files}")
    connection = _open_read_only(app_config.database_path)
    try:
        for report_row in connection.execute("SELECT report_date FROM reports WHERE status = 'published'"):
            if not (app_config.daily_reports_dir / f"{report_row['report_date']}.md").is_file():
                problems.append(f"the published report for {report_row['report_date']} is missing")
    finally:
        connection.close()
    return RestoreCheck(problems, restored_home)


# --------------------------------------------------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------------------------------------------------


def _copy_settings(app_config: AppConfig, backup_dir: Path) -> None:
    for settings_name in (CONFIG_FILE_NAME, TOPICS_FILE_NAME):
        shutil.copy2(app_config.home_dir / settings_name, backup_dir / settings_name)
    seeds_dir = app_config.home_dir / SEEDS_DIRECTORY_NAME
    if seeds_dir.is_dir():
        shutil.copytree(seeds_dir, backup_dir / SEEDS_DIRECTORY_NAME, symlinks=False)


def _snapshot_database(source_path: Path, backup_path: Path) -> None:
    """SQLite online backup API: a consistent copy while other processes keep writing."""
    source_connection = _open_read_only(source_path)
    backup_connection = sqlite3.connect(backup_path)
    try:
        source_connection.backup(backup_connection)
        backup_connection.execute("PRAGMA journal_mode = DELETE")  # one self-contained file, no -wal next to it
    finally:
        backup_connection.close()
        source_connection.close()
    os.chmod(backup_path, 0o600)


def _copy_extracted_texts(source_cache_dir: Path, backup_cache_dir: Path) -> int:
    copied_count = 0
    for text_path in sorted((source_cache_dir / SOURCE_FILES_SUBDIRECTORY).glob(f"*/*{EXTRACTED_TEXT_SUFFIX}")):
        if text_path.is_symlink() or not text_path.is_file():
            continue
        destination_path = backup_cache_dir / SOURCE_FILES_SUBDIRECTORY / text_path.parent.name / text_path.name
        destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copy2(text_path, destination_path)
        copied_count += 1
    return copied_count


def _copy_output(output_dir: Path, backup_output_dir: Path) -> int:
    if output_dir.is_dir():
        shutil.copytree(output_dir, backup_output_dir, symlinks=False)
    else:
        backup_output_dir.mkdir()
    return _count_files(backup_output_dir)


def _count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file()) if directory.is_dir() else 0


def _open_read_only(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _describe_database(database_path: Path) -> dict[str, Any]:
    connection = _open_read_only(database_path)
    try:
        existing_tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return {
            "schema_version": read_schema_version(connection),
            "integrity_check": connection.execute("PRAGMA integrity_check").fetchone()[0],
            "row_counts": {
                table_name: connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]  # noqa: S608  # table names come from COUNTED_TABLES
                for table_name in COUNTED_TABLES
                if table_name in existing_tables
            },
        }
    finally:
        connection.close()
