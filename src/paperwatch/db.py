"""SQLite connection settings, schema migrations, and the write-transaction helper.

Rules used throughout the code base:
  * Connections run in autocommit mode; multi-statement writes use `write_transaction()` (BEGIN IMMEDIATE).
  * Never make a network or model call while a write transaction is open.
  * Columns used for joins, ordering, uniqueness, and recovery are real indexed columns; variable review content is
    validated JSON in a TEXT column.

Schema changes are appended to paperwatch.db_migrations.SCHEMA_MIGRATIONS (never edit an applied migration).
PRAGMA user_version records how many migrations a database has applied.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from paperwatch.db_migrations import SCHEMA_MIGRATIONS
from paperwatch.errors import DatabaseError

BUSY_TIMEOUT_MILLISECONDS = 10_000

CURRENT_SCHEMA_VERSION = len(SCHEMA_MIGRATIONS)


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode, foreign keys, and a busy timeout. Does not create or migrate tables."""
    try:
        connection = sqlite3.connect(
            database_path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1000, isolation_level=None
        )  # autocommit; explicit BEGIN where needed
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MILLISECONDS}")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection
    except sqlite3.Error as sqlite_error:
        raise DatabaseError(f"cannot open database {database_path}: {sqlite_error}") from sqlite_error


def read_schema_version(connection: sqlite3.Connection) -> int:
    return connection.execute("PRAGMA user_version").fetchone()[0]


def apply_pending_migrations(connection: sqlite3.Connection) -> tuple[int, int]:
    """Apply every migration the database has not seen yet. Returns (version_before, version_after)."""
    version_before = read_schema_version(connection)
    if version_before > CURRENT_SCHEMA_VERSION:
        raise DatabaseError(
            f"database schema version {version_before} is newer than this code "
            f"(supports {CURRENT_SCHEMA_VERSION}); upgrade paperwatch"
        )
    for migration_index in range(version_before, CURRENT_SCHEMA_VERSION):
        migration = SCHEMA_MIGRATIONS[migration_index]
        # DDL, Python backfill and the version bump commit together. executescript() commits any open transaction
        # first, then runs the script; the script's BEGIN IMMEDIATE leaves our transaction open for the backfill.
        try:
            connection.executescript(f"BEGIN IMMEDIATE;\n{migration.schema_sql}")
            if migration.backfill_existing_rows is not None:
                migration.backfill_existing_rows(connection)
            connection.execute(f"PRAGMA user_version = {migration_index + 1}")
            connection.execute("COMMIT")
        except BaseException as migration_error:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            if isinstance(migration_error, sqlite3.Error):
                raise DatabaseError(
                    f"migration {migration_index + 1} ({migration.description}) failed: {migration_error}"
                ) from migration_error
            raise
    return version_before, CURRENT_SCHEMA_VERSION


def open_initialized_database(database_path: Path) -> sqlite3.Connection:
    """Open the database for normal commands; refuse if `init` has not created or migrated it."""
    if not database_path.is_file():
        raise DatabaseError(f"database {database_path} does not exist", operator_hint="run `paperwatch init`")
    connection = connect_database(database_path)
    schema_version = read_schema_version(connection)
    if schema_version != CURRENT_SCHEMA_VERSION:
        connection.close()
        raise DatabaseError(
            f"database schema version is {schema_version}, this code expects {CURRENT_SCHEMA_VERSION}",
            operator_hint="run `paperwatch init` to migrate",
        )
    return connection


@contextmanager
def write_transaction(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT, rolling back on any exception. Keep the body short and free of network calls."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield connection
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")
