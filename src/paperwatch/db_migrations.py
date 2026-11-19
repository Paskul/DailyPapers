"""Schema migrations, in order. Append new ones; never edit one that has shipped.

A migration is DDL plus an optional Python backfill. `paperwatch.db.apply_pending_migrations` runs the DDL, the
backfill, and the PRAGMA user_version bump in ONE transaction, so a database is never left half-migrated.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from paperwatch.errors import SourceInputError
from paperwatch.identifiers import EDITION_ID_PREFIX, new_internal_id
from paperwatch.review_inputs import (COVERAGE_ABSTRACT_ONLY, INITIAL_READING_GENERATION, build_review_input_key,
                                      fingerprint_abstract_reading_text)
from paperwatch.scholarly_identifiers import (build_edition_key, normalize_identifier, parse_version_number,
                                             resolve_edition_version_label)
from paperwatch.text_normalization import build_title_match_key


@dataclass(frozen=True)
class SchemaMigration:
    description: str
    schema_sql: str
    backfill_existing_rows: Callable[[sqlite3.Connection], None] | None = None


# --------------------------------------------------------------------------------------------------------------------
# Migration 1: the step-1 contracts (moved here unchanged from db.py).
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

# --------------------------------------------------------------------------------------------------------------------
# Migration 2 (plan part 2): paper identity, editions, reading identity, topic assessments, references and
# comparisons, seed membership, and source checkpoints.
# In paper_aliases, `provider` now means the identifier scheme (doi, arxiv, openalex, or a provider name) and
# `provider_paper_id` its NORMALIZED, version-free value.
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0002_IDENTITY_EDITIONS_AND_RECORDS = """
ALTER TABLE paper_aliases ADD COLUMN asserted_by_route TEXT;   -- discovery route whose record supplied this id
ALTER TABLE paper_aliases ADD COLUMN first_seen_at TEXT;

ALTER TABLE papers ADD COLUMN title_match_key TEXT;            -- normalized title; flags possible duplicates only
CREATE INDEX papers_by_title_match_key ON papers(title_match_key);

CREATE TABLE paper_editions (                                  -- one row per distinct published version of a work
    edition_id         TEXT PRIMARY KEY,                       -- internal id
    paper_id           TEXT NOT NULL REFERENCES papers(paper_id),
    edition_key        TEXT NOT NULL UNIQUE,                   -- e.g. 'arxiv:2401.01234@v2', 'doi:10.1109/x'
    version_label      TEXT,                                   -- 'v2'; NULL when the provider has no versions
    version_number     INTEGER,                                -- 2 for 'v2'; used to find the latest revision
    title              TEXT NOT NULL,                          -- text snapshot as this edition's provider gave it
    authors_json       TEXT NOT NULL DEFAULT '[]',
    abstract           TEXT,
    published_date     TEXT,
    first_seen_route   TEXT NOT NULL,
    first_seen_at      TEXT NOT NULL,
    abstract_added_at  TEXT                                    -- set when a title-only edition later got its abstract
);
CREATE INDEX paper_editions_by_paper ON paper_editions(paper_id, first_seen_at);

CREATE TABLE paper_possible_matches (                          -- similar records the catalog refused to merge
    first_paper_id   TEXT NOT NULL REFERENCES papers(paper_id),
    second_paper_id  TEXT NOT NULL REFERENCES papers(paper_id),
    match_reason     TEXT NOT NULL,                            -- same_title | same_title_shared_author | ...
    evidence_json    TEXT NOT NULL DEFAULT '{}',
    detected_at      TEXT NOT NULL,
    resolution       TEXT NOT NULL DEFAULT 'unresolved'
        CHECK (resolution IN ('unresolved', 'same_work', 'different_works')),
    PRIMARY KEY (first_paper_id, second_paper_id),
    CHECK (first_paper_id < second_paper_id)
);

CREATE TABLE paper_seed_memberships (                          -- filled by the Zotero / CSL JSON seed import (part 4)
    paper_id            TEXT NOT NULL REFERENCES papers(paper_id),
    seed_source         TEXT NOT NULL,                         -- 'zotero' | 'csl_json'
    seed_item_key       TEXT NOT NULL,                         -- Zotero item key or export id
    seed_collection_key TEXT NOT NULL DEFAULT '',
    first_seen_at       TEXT NOT NULL,
    removed_at          TEXT,                                  -- set when the item leaves the library/collection
    PRIMARY KEY (seed_source, seed_item_key, seed_collection_key)
);
CREATE INDEX paper_seed_memberships_by_paper ON paper_seed_memberships(paper_id);

ALTER TABLE reviews ADD COLUMN edition_id TEXT REFERENCES paper_editions(edition_id);
ALTER TABLE reviews ADD COLUMN source_text_sha256 TEXT;        -- normalized hash of the text that was read
ALTER TABLE reviews ADD COLUMN reading_generation TEXT NOT NULL DEFAULT 'initial';  -- 'g_...' for operator rereads
CREATE INDEX reviews_by_paper_and_text ON reviews(paper_id, source_text_sha256);

CREATE TABLE review_topic_assessments (                        -- separate from the reading: reusable on topic change
    review_id            TEXT NOT NULL REFERENCES reviews(review_id),
    topic_id             TEXT NOT NULL,
    topic_profile_sha256 TEXT NOT NULL,                        -- '' when migrated from schema v1 (not recorded)
    topic_score          INTEGER NOT NULL,
    score_scale          TEXT NOT NULL,                        -- 'rating_0_to_4' now; part 7: 'priority_0_to_100'
    assessment_json      TEXT NOT NULL,                        -- ratings, rationale, matched seed ids
    role_name            TEXT NOT NULL,
    prompt_version       TEXT NOT NULL,
    assessed_at          TEXT NOT NULL,
    PRIMARY KEY (review_id, topic_id, topic_profile_sha256)
);
CREATE INDEX review_topic_assessments_by_topic ON review_topic_assessments(topic_id, topic_score);

CREATE TABLE citation_edges (                                  -- directed: citing paper -> cited work
    citing_paper_id  TEXT NOT NULL REFERENCES papers(paper_id),
    cited_scheme     TEXT NOT NULL,                            -- normalized identifier of the cited work
    cited_value      TEXT NOT NULL,
    cited_paper_id   TEXT REFERENCES papers(paper_id),         -- NULL while the cited work is not in the catalog
    provenance       TEXT NOT NULL,                            -- e.g. 'openalex:referenced_works'
    citation_context TEXT,
    recorded_at      TEXT NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_scheme, cited_value, provenance)
);
CREATE INDEX citation_edges_by_cited_paper ON citation_edges(cited_paper_id);
CREATE INDEX citation_edges_by_cited_identifier ON citation_edges(cited_scheme, cited_value);

CREATE TABLE comparisons (                                     -- one per question and unordered pair of reviews
    comparison_id         TEXT PRIMARY KEY,
    comparison_key        TEXT NOT NULL UNIQUE,                -- topic + question hash + sorted pair of review ids
    topic_id              TEXT NOT NULL,
    question_sha256       TEXT NOT NULL,
    lower_review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    higher_review_id      TEXT NOT NULL REFERENCES reviews(review_id),
    comparison_json       TEXT NOT NULL,
    role_name             TEXT NOT NULL,
    prompt_version        TEXT NOT NULL,
    model_identity        TEXT NOT NULL,
    server_build          TEXT NOT NULL,
    completed_at          TEXT NOT NULL,
    CHECK (lower_review_id < higher_review_id)
);
CREATE INDEX comparisons_by_lower_review ON comparisons(lower_review_id);
CREATE INDEX comparisons_by_higher_review ON comparisons(higher_review_id);

CREATE TABLE source_checkpoints (                              -- one cursor and budget per search
    search_key                  TEXT PRIMARY KEY,              -- e.g. 'arxiv-feed:cs.RO'
    source_name                 TEXT NOT NULL,                 -- 'arxiv' | 'openalex' | 'zotero' | ...
    topic_id                    TEXT,
    query_json                  TEXT NOT NULL DEFAULT '{}',    -- approved public parameters only
    cursor                      TEXT,
    coverage_status             TEXT NOT NULL DEFAULT 'active'
        CHECK (coverage_status IN ('active', 'capped', 'exhausted', 'blocked')),
    stop_reason                 TEXT,
    candidates_admitted         INTEGER NOT NULL DEFAULT 0,
    candidate_budget            INTEGER,
    last_completed_interval_end TEXT,
    updated_at                  TEXT NOT NULL
);
"""


def _backfill_schema_v1_rows_for_identity_and_editions(connection: sqlite3.Connection) -> None:
    """Give every v1 paper one edition, re-key its reviews and jobs by text hash, and copy topic assessments out.

    Uses the same key functions as the live code, so re-collecting a migrated paper finds its edition and review
    instead of queueing a second reading.
    """
    paper_rows = connection.execute("SELECT * FROM papers ORDER BY discovered_at, paper_id").fetchall()
    for paper_row in paper_rows:
        paper_id = paper_row["paper_id"]
        connection.execute("UPDATE papers SET title_match_key = ? WHERE paper_id = ?",
                           (build_title_match_key(paper_row["title"]), paper_id))
        connection.execute("UPDATE paper_aliases SET asserted_by_route = ?, first_seen_at = ? WHERE paper_id = ?",
                           (paper_row["origin_route"], paper_row["discovered_at"], paper_id))

        first_alias_row = connection.execute(
            "SELECT provider, provider_paper_id FROM paper_aliases WHERE paper_id = ? ORDER BY provider LIMIT 1",
            (paper_id,)).fetchone()
        version_label = None
        try:
            primary_identifier = normalize_identifier(first_alias_row["provider"], first_alias_row["provider_paper_id"])
            version_label = resolve_edition_version_label(primary_identifier, paper_row["source_version"])
            edition_key = build_edition_key(primary_identifier, version_label)
        except (SourceInputError, TypeError):  # no alias, or a v1 alias that is not a valid identifier
            edition_key = f"legacy:{paper_id}"
        edition_id = new_internal_id(EDITION_ID_PREFIX)
        connection.execute(
            """INSERT INTO paper_editions (edition_id, paper_id, edition_key, version_label, version_number, title,
                   authors_json, abstract, published_date, first_seen_route, first_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (edition_id, paper_id, edition_key, version_label, parse_version_number(version_label),
             paper_row["title"], paper_row["authors_json"], paper_row["abstract"], paper_row["first_published_date"],
             paper_row["origin_route"], paper_row["discovered_at"]))

        source_text_sha256 = fingerprint_abstract_reading_text(paper_row["title"], paper_row["abstract"])
        review_input_key = build_review_input_key(COVERAGE_ABSTRACT_ONLY, source_text_sha256,
                                                  INITIAL_READING_GENERATION)
        for review_row in connection.execute("SELECT * FROM reviews WHERE paper_id = ?", (paper_id,)).fetchall():
            connection.execute(
                """UPDATE reviews SET edition_id = ?, source_text_sha256 = ?, review_input_key = ?
                   WHERE review_id = ?""", (edition_id, source_text_sha256, review_input_key, review_row["review_id"]))
            for topic_assessment in json.loads(review_row["review_json"]).get("topic_relevance", []):
                connection.execute(
                    """INSERT OR IGNORE INTO review_topic_assessments (review_id, topic_id, topic_profile_sha256,
                           topic_score, score_scale, assessment_json, role_name, prompt_version, assessed_at)
                       VALUES (?, ?, '', ?, 'rating_0_to_4', ?, ?, ?, ?)""",
                    (review_row["review_id"], topic_assessment["topic_id"], topic_assessment["relevance_rating"],
                     json.dumps(topic_assessment, ensure_ascii=False), review_row["role_name"],
                     review_row["prompt_version"], review_row["completed_at"]))
        connection.execute(
            """UPDATE jobs SET input_version = ?, job_key = 'review_paper:' || paper_id || ':' || ?, input_json = ?
               WHERE paper_id = ? AND kind = 'review_paper'""",
            (review_input_key, review_input_key,
             json.dumps({"coverage": COVERAGE_ABSTRACT_ONLY, "edition_id": edition_id,
                         "reading_generation": INITIAL_READING_GENERATION}), paper_id))


# --------------------------------------------------------------------------------------------------------------------
# Migration 3 (plan part 3): what the collector sent to the internet, and how much of each provider's budget it used.
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0003_OUTBOUND_REQUEST_LEDGER = """
CREATE TABLE outbound_requests (                               -- one row per HTTPS hop the collector attempted
    request_id        TEXT PRIMARY KEY,
    provider_name     TEXT NOT NULL,                           -- 'arxiv' | 'openalex' | 'zotero'
    operation         TEXT NOT NULL,                           -- 'search' | 'get_work' | 'get_text' | ...
    hop_number        INTEGER NOT NULL DEFAULT 0,              -- 0 = first request, 1+ = an allowed redirect
    method            TEXT NOT NULL,
    host              TEXT NOT NULL,
    path              TEXT NOT NULL,
    query_json        TEXT NOT NULL DEFAULT '[]',              -- [[name, value], ...]; credentials are redacted
    outcome           TEXT NOT NULL,                           -- ok | redirect | blocked | not_found | unavailable | ...
    status_code       INTEGER,
    response_bytes    INTEGER,
    error_message     TEXT,
    started_at        TEXT NOT NULL,
    duration_seconds  REAL NOT NULL
);
CREATE INDEX outbound_requests_by_provider_and_start ON outbound_requests(provider_name, started_at);

CREATE TABLE provider_request_budgets (                        -- daily request budget and spacing per provider
    provider_name                 TEXT NOT NULL,
    budget_date                   TEXT NOT NULL,               -- UTC date, YYYY-MM-DD
    requests_reserved             INTEGER NOT NULL DEFAULT 0,
    next_request_allowed_epoch    REAL NOT NULL DEFAULT 0,     -- enforces minimum spacing across processes
    PRIMARY KEY (provider_name, budget_date)
);
"""


SCHEMA_MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration("initial schema (plan part 1)", MIGRATION_0001_INITIAL_SCHEMA),
    SchemaMigration("paper identity, editions, reading identity and part-2 records",
                    MIGRATION_0002_IDENTITY_EDITIONS_AND_RECORDS,
                    _backfill_schema_v1_rows_for_identity_and_editions),
    SchemaMigration("outbound request log and provider request budgets (plan part 3)",
                    MIGRATION_0003_OUTBOUND_REQUEST_LEDGER),
)
