"""SQLite connection settings, schema migrations, and the write-transaction helper.

Rules used throughout the code base:
  * Connections run in autocommit mode; multi-statement writes use `write_transaction()` (BEGIN IMMEDIATE).
  * Never make a network or model call while a write transaction is open.
  * Columns used for joins, ordering, uniqueness, and recovery are real indexed columns; variable review content is
    validated JSON in a TEXT column.

Schema changes are appended to SCHEMA_MIGRATIONS (never edit an applied migration). PRAGMA user_version records how
many migrations a database has applied.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from paperwatch.errors import DatabaseError

BUSY_TIMEOUT_MILLISECONDS = 10_000

# --------------------------------------------------------------------------------------------------------------------
# Migration 1: the step-1 contracts. Part 2 of the plan extends identity, editions and leases on top of this.
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0001_INITIAL_SCHEMA = """
CREATE TABLE papers (
    paper_id              TEXT PRIMARY KEY,                -- internal id, used for note paths
    title                 TEXT NOT NULL,
    authors_json          TEXT NOT NULL DEFAULT '[]',
    abstract              TEXT,                            -- NULL when the provider supplied none (title-only)
    first_published_date  TEXT,                            -- as reported by the provider (precision varies)
    latest_revision_date  TEXT,
    source_version        TEXT,                            -- e.g. arXiv 'v2'
    origin_route          TEXT NOT NULL,                   -- the discovery route that found it first
    discovered_at         TEXT NOT NULL,                   -- local discovery time (UTC)
    status                TEXT NOT NULL DEFAULT 'waiting_input'
        CHECK (status IN ('screened_out', 'waiting_input', 'queued', 'reviewing', 'complete', 'failed')),
    updated_at            TEXT NOT NULL
);

CREATE TABLE paper_aliases (                                -- each trusted provider id maps to exactly one paper
    provider          TEXT NOT NULL,
    provider_paper_id TEXT NOT NULL,
    paper_id          TEXT NOT NULL REFERENCES papers(paper_id),
    PRIMARY KEY (provider, provider_paper_id)
);
CREATE INDEX paper_aliases_by_paper ON paper_aliases(paper_id);

CREATE TABLE paper_discoveries (                            -- every route that found a paper, first time per route
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id),
    route         TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY (paper_id, route)
);

CREATE TABLE jobs (
    job_id               TEXT PRIMARY KEY,
    job_key              TEXT NOT NULL UNIQUE,             -- one row per logical job, e.g. kind:paper:input
    kind                 TEXT NOT NULL,
    paper_id             TEXT REFERENCES papers(paper_id),
    input_version        TEXT NOT NULL,
    input_json           TEXT NOT NULL DEFAULT '{}',
    priority             INTEGER NOT NULL,                 -- 1 = most urgent
    status               TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts             INTEGER NOT NULL DEFAULT 0,
    max_attempts         INTEGER NOT NULL,
    due_at               TEXT NOT NULL,
    lease_token          TEXT,
    lease_expires_at     TEXT,
    last_error_component TEXT,
    last_error           TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL
);
CREATE INDEX jobs_by_claim_order ON jobs(status, priority, due_at);
CREATE INDEX jobs_by_paper ON jobs(paper_id);

CREATE TABLE reviews (                                      -- immutable once written
    review_id        TEXT PRIMARY KEY,
    paper_id         TEXT NOT NULL REFERENCES papers(paper_id),
    job_id           TEXT REFERENCES jobs(job_id),
    review_input_key TEXT NOT NULL,                        -- fingerprint of exactly what the model read
    coverage         TEXT NOT NULL CHECK (coverage IN ('full_text', 'partial_text', 'abstract_only')),
    review_json      TEXT NOT NULL,                        -- validated role output
    role_name        TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    model_identity   TEXT NOT NULL,
    server_build     TEXT NOT NULL,
    completed_at     TEXT NOT NULL,
    UNIQUE (paper_id, review_input_key)
);
CREATE INDEX reviews_by_completion ON reviews(completed_at);

CREATE TABLE model_calls (                                  -- one row per inference request, success or not
    call_id           TEXT PRIMARY KEY,
    role_name         TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    job_id            TEXT,
    model_identity    TEXT,
    server_build      TEXT,
    started_at        TEXT NOT NULL,
    duration_seconds  REAL NOT NULL,
    outcome           TEXT NOT NULL,                       -- ok | unavailable | invalid_output | truncated | ...
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    error_message     TEXT
);
CREATE INDEX model_calls_by_start ON model_calls(started_at);

CREATE TABLE service_state (                                -- pause/back-off bookkeeping per external service
    service              TEXT PRIMARY KEY,
    paused_until         TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_error           TEXT,
    last_failure_at      TEXT,
    last_success_at      TEXT
);

CREATE TABLE reports (                                      -- one report per local reporting date
    report_date     TEXT PRIMARY KEY,                       -- YYYY-MM-DD in the report timezone
    cutoff_at       TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending', 'published')),
    output_path     TEXT NOT NULL,
    rendered_sha256 TEXT NOT NULL,
    rendered_bytes  BLOB NOT NULL,                          -- exact bytes, so a rerun republishes identically
    created_at      TEXT NOT NULL,
    published_at    TEXT
);

CREATE TABLE report_items (
    report_date TEXT NOT NULL REFERENCES reports(report_date),
    rank        INTEGER NOT NULL,
    paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
    review_id   TEXT NOT NULL REFERENCES reviews(review_id),
    PRIMARY KEY (report_date, rank),
    UNIQUE (report_date, paper_id)
);
CREATE INDEX report_items_by_paper ON report_items(paper_id);
"""

SCHEMA_MIGRATIONS: tuple[str, ...] = (MIGRATION_0001_INITIAL_SCHEMA,)
CURRENT_SCHEMA_VERSION = len(SCHEMA_MIGRATIONS)


def connect_database(database_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode, foreign keys, and a busy timeout. Does not create or migrate tables."""
    try:
        connection = sqlite3.connect(database_path, timeout=BUSY_TIMEOUT_MILLISECONDS / 1000,
                                     isolation_level=None)  # autocommit; explicit BEGIN where needed
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
        raise DatabaseError(f"database schema version {version_before} is newer than this code "
                            f"(supports {CURRENT_SCHEMA_VERSION}); upgrade paperwatch")
    for migration_index in range(version_before, CURRENT_SCHEMA_VERSION):
        migration_sql = SCHEMA_MIGRATIONS[migration_index]
        # executescript() commits any open transaction first; the migration and its version bump commit together.
        connection.executescript(f"BEGIN IMMEDIATE;\n{migration_sql}\nPRAGMA user_version = {migration_index + 1};"
                                 "\nCOMMIT;")
    return version_before, CURRENT_SCHEMA_VERSION


def open_initialized_database(database_path: Path) -> sqlite3.Connection:
    """Open the database for normal commands; refuse if `init` has not created or migrated it."""
    if not database_path.is_file():
        raise DatabaseError(f"database {database_path} does not exist", operator_hint="run `paperwatch init`")
    connection = connect_database(database_path)
    schema_version = read_schema_version(connection)
    if schema_version != CURRENT_SCHEMA_VERSION:
        connection.close()
        raise DatabaseError(f"database schema version is {schema_version}, this code expects "
                            f"{CURRENT_SCHEMA_VERSION}", operator_hint="run `paperwatch init` to migrate")
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
