"""Schema migrations, in order. Append new ones; never edit one that has shipped.

A migration is DDL plus an optional Python backfill. `paperwatch.db.apply_pending_migrations` runs the DDL, the
backfill, and the PRAGMA user_version bump in ONE transaction, so a database is never left half-migrated.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from paperwatch.errors import SourceInputError
from paperwatch.identifiers import EDITION_ID_PREFIX, new_internal_id
from paperwatch.review_inputs import (COVERAGE_ABSTRACT_ONLY, INITIAL_READING_GENERATION, build_review_input_key,
                                      fingerprint_abstract_reading_text, fingerprint_extracted_reading_text)
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


# --------------------------------------------------------------------------------------------------------------------
# Migration 4 (plan part 4): the seed cache. Each seed's last successfully synced metadata, independent of the catalog
# paper it maps to (the catalog never rewrites a title; a seed follows the library). Collection membership stays in
# paper_seed_memberships (migration 2); a sync cursor per seed source lives in source_checkpoints.
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0004_SEED_CACHE = """
CREATE TABLE seed_items (                                      -- last successfully synced metadata of each seed
    seed_source         TEXT NOT NULL CHECK (seed_source IN ('zotero', 'csl_json')),
    seed_item_key       TEXT NOT NULL,                         -- Zotero item key, or '<csl file key>#<CSL id>'
    paper_id            TEXT NOT NULL REFERENCES papers(paper_id),
    item_type           TEXT NOT NULL,
    item_version        INTEGER,                               -- Zotero item version; NULL for CSL JSON
    title               TEXT NOT NULL,
    authors_json        TEXT NOT NULL DEFAULT '[]',
    abstract            TEXT,                                  -- NULL: title-only seed (a recorded limitation)
    publication_year    INTEGER,
    identifiers_json    TEXT NOT NULL DEFAULT '[]',            -- ["doi:...", "arxiv:..."] as the library states them
    matching_text_kind  TEXT NOT NULL CHECK (matching_text_kind IN ('title_and_abstract', 'title_only')),
    metadata_sha256     TEXT NOT NULL,                         -- changes the topic profile hash when it changes
    first_seen_at       TEXT NOT NULL,
    last_synced_at      TEXT NOT NULL,
    PRIMARY KEY (seed_source, seed_item_key)
);
CREATE INDEX seed_items_by_paper ON seed_items(paper_id);
CREATE INDEX paper_seed_memberships_by_collection
    ON paper_seed_memberships(seed_source, seed_collection_key, removed_at);
"""


# --------------------------------------------------------------------------------------------------------------------
# Migration 5 (plan part 5): what each route reported, withdrawals and retractions, and acquired source text.
# Search cursors, coverage, and gaps keep using source_checkpoints (migration 2): a missed-release recovery search is
# its own checkpoint row, so an open gap is simply a recovery checkpoint that is not yet exhausted.
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0005_COLLECTION_ROUTES_WITHDRAWALS_AND_SOURCE_TEXT = """
ALTER TABLE paper_discoveries ADD COLUMN announcement_kind TEXT;          -- arXiv: new | cross_list | revision
ALTER TABLE paper_discoveries ADD COLUMN reported_publication_date TEXT;  -- as reported; its length = precision
ALTER TABLE paper_editions ADD COLUMN announcement_kind TEXT;             -- how this edition was first announced
CREATE INDEX papers_by_discovery_time ON papers(discovered_at);           -- daily budget of new metadata candidates

CREATE TABLE paper_withdrawals (                               -- a provider reported the work withdrawn or retracted
    paper_id                 TEXT PRIMARY KEY REFERENCES papers(paper_id),
    notice_kind              TEXT NOT NULL CHECK (notice_kind IN ('withdrawn', 'retracted')),
    reported_by_route        TEXT NOT NULL,
    evidence                 TEXT,                             -- short provider text that carried the notice
    detected_at              TEXT NOT NULL,
    correction_report_date   TEXT                              -- report that announced the correction (part 9)
);

CREATE TABLE edition_source_texts (                            -- full text acquired for one edition (plan part 5)
    edition_id               TEXT PRIMARY KEY REFERENCES paper_editions(edition_id),
    paper_id                 TEXT NOT NULL REFERENCES papers(paper_id),
    acquisition_status       TEXT NOT NULL CHECK (acquisition_status IN ('acquired', 'unavailable', 'failed')),
    text_route               TEXT,                             -- local_file | arxiv_pdf | openalex_content
    source_file_name         TEXT,                             -- inside state/cache/sources/<edition_id>/
    source_sha256            TEXT,
    content_type             TEXT,
    source_bytes             INTEGER,
    extraction_status        TEXT NOT NULL DEFAULT 'pending'
        CHECK (extraction_status IN ('pending', 'extracted', 'failed', 'not_configured')),
    extracted_text_file_name TEXT,                             -- inside the same cache directory
    extracted_text_sha256    TEXT,
    extractor_image          TEXT,
    attempts                 INTEGER NOT NULL DEFAULT 0,
    last_error               TEXT,
    acquired_at              TEXT,                             -- set once, when the bytes were stored
    downloaded_from_network  INTEGER NOT NULL DEFAULT 0,       -- 1 if it used the daily source-text budget
    updated_at               TEXT NOT NULL
);
CREATE INDEX edition_source_texts_by_paper ON edition_source_texts(paper_id);
CREATE INDEX edition_source_texts_by_acquisition ON edition_source_texts(acquired_at);
"""


# --------------------------------------------------------------------------------------------------------------------
# Migration 6 (plan part 6): cheap CPU screening and ordered dispatch.
#   text_embeddings         cache of sentence embeddings, keyed by the pinned model's identity and the text's hash
#   candidate_topic_scores  the screening score of one candidate for one topic, with its component signals; rows of
#                           unpromoted candidates are rescored whenever the topic profile or the paper's text changes
#   paper_promotions        which papers screening chose for a reading, why, and when (the daily promotion limit)
#   jobs                    two descending within-class order values and the end of a new release's fresh window
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0006_SCREENING_AND_DISPATCH_ORDER = """
ALTER TABLE jobs ADD COLUMN within_class_order_primary REAL NOT NULL DEFAULT 0;    -- higher is claimed first
ALTER TABLE jobs ADD COLUMN within_class_order_secondary REAL NOT NULL DEFAULT 0;  -- tie-breaker, higher first
ALTER TABLE jobs ADD COLUMN new_release_fresh_until TEXT;  -- class 1 only: after this it is claimed as class 2
CREATE INDEX jobs_by_status_and_priority ON jobs(status, priority, within_class_order_primary DESC, created_at);

CREATE TABLE text_embeddings (                                 -- one vector per (pinned model, exact model input)
    embedding_model_identity TEXT NOT NULL,                    -- e.g. 'onnx:bge-base-en-v1.5@<revision>:<file hash>'
    text_sha256              TEXT NOT NULL,                    -- of the exact model input, query instruction included
    dimensions               INTEGER NOT NULL,
    vector_float32           BLOB NOT NULL,                    -- little-endian float32, L2-normalized
    created_at               TEXT NOT NULL,
    PRIMARY KEY (embedding_model_identity, text_sha256)
);

CREATE TABLE candidate_topic_scores (                          -- screening score of one candidate for one topic
    paper_id                TEXT NOT NULL REFERENCES papers(paper_id),
    topic_id                TEXT NOT NULL,
    edition_id              TEXT NOT NULL REFERENCES paper_editions(edition_id),  -- the text that was scored
    unnormalized_score      REAL NOT NULL,                     -- raw weighted mean; promotion ranks each signal
                                                               -- within the topic's pool first (not stored)
    question_similarity     REAL,                              -- NULL when the signal is unavailable for the topic
    seed_similarity         REAL,                              -- mean of the closest seeds (up to 3)
    phrase_match_fraction   REAL,                              -- fraction of the topic's approved phrases found
    closest_seed_ids_json   TEXT NOT NULL DEFAULT '[]',
    matched_phrases_json    TEXT NOT NULL DEFAULT '[]',
    title_only              INTEGER NOT NULL DEFAULT 0,        -- 1: the candidate had no abstract (a limitation)
    screening_input_sha256  TEXT NOT NULL,                     -- text + topic profile + scoring settings + model
    scored_at               TEXT NOT NULL,
    PRIMARY KEY (paper_id, topic_id)
);
CREATE INDEX candidate_topic_scores_by_topic ON candidate_topic_scores(topic_id, unnormalized_score DESC);

CREATE TABLE paper_promotions (                                -- this paper was chosen for a reading
    paper_id          TEXT PRIMARY KEY REFERENCES papers(paper_id),
    promotion_reason  TEXT NOT NULL                            -- 'direct': an operator tool or test, not screening;
        CHECK (promotion_reason IN ('topic_share', 'spare_slot', 'near_cutoff', 'direct')),  -- not in the limit
    topic_id          TEXT,                                    -- the topic whose share or score selected it
    best_topic_id     TEXT,
    screening_score   REAL,                                    -- pool-normalized best-topic score (NULL if direct)
    unnormalized_score       REAL,                             -- the same topic's raw weighted mean
    normalization_pool_size  INTEGER,                          -- candidates the percentiles were taken over
    priority_class    INTEGER NOT NULL,                        -- class the reading was queued in
    promoted_at       TEXT NOT NULL
);
CREATE INDEX paper_promotions_by_time ON paper_promotions(promoted_at);
"""


# --------------------------------------------------------------------------------------------------------------------
# Migration 7 (plan part 7): evidence-backed reading.
#   review_stage_results      validated result of each step of a reading (one per reading section, then the
#                             synthesis), so a timeout resumes instead of restarting a long paper
#   review_evidence_passages  the passages a review cites, copied out of the source cache, so the evidence survives
#                             when replaceable downloads are pruned (plan part 10)
#   reviews.reading_record_json        sections read and not read, why the coverage is partial, parser recovery
#   edition_source_texts.normalized_text_sha256 / extracted_text_characters: the full-text reading identity and
#                             whether the text is long enough to read (backfilled from the cache when readable)
# --------------------------------------------------------------------------------------------------------------------
MIGRATION_0007_EVIDENCE_BACKED_READING = """
CREATE TABLE review_stage_results (
    paper_id             TEXT NOT NULL REFERENCES papers(paper_id),
    review_input_key     TEXT NOT NULL,
    reading_plan_sha256  TEXT NOT NULL,                        -- section boundaries + limits; never mixed
    stage_key            TEXT NOT NULL,                        -- 'section:003' | 'synthesis'
    prompt_version       TEXT NOT NULL,
    role_name            TEXT NOT NULL,
    result_json          TEXT NOT NULL,                        -- validated role output
    model_identity       TEXT NOT NULL,
    server_build         TEXT NOT NULL,
    job_id               TEXT,
    completed_at         TEXT NOT NULL,
    PRIMARY KEY (paper_id, review_input_key, reading_plan_sha256, stage_key, prompt_version)
);

CREATE TABLE review_evidence_passages (
    review_id     TEXT NOT NULL REFERENCES reviews(review_id),
    passage_id    TEXT NOT NULL,                               -- A1.. abstract, T1 title, P1.. extracted text
    page_number   INTEGER,
    section_title TEXT NOT NULL,
    passage_text  TEXT NOT NULL,                               -- untrusted; escaped when rendered
    PRIMARY KEY (review_id, passage_id)
);

ALTER TABLE reviews ADD COLUMN reading_record_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE edition_source_texts ADD COLUMN normalized_text_sha256 TEXT;
ALTER TABLE edition_source_texts ADD COLUMN extracted_text_characters INTEGER;
"""
CACHED_TEXT_FILE_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.txt$")
INTERNAL_EDITION_ID_PATTERN = re.compile(r"^e_[0-9a-f]{16}$")


def _backfill_full_text_reading_identity(connection: sqlite3.Connection) -> None:
    """Fingerprint texts extracted before schema v7. The cache sits next to the database (state/cache/sources/);
    an unreadable file stays NULL, so that paper is read from its abstract until the text is extracted again."""
    database_file = Path(connection.execute("PRAGMA database_list").fetchone()["file"] or ".")
    source_files_dir = database_file.parent / "cache" / "sources"
    for text_row in connection.execute(
            """SELECT edition_id, extracted_text_file_name FROM edition_source_texts
               WHERE extraction_status = 'extracted' AND extracted_text_file_name IS NOT NULL""").fetchall():
        if not INTERNAL_EDITION_ID_PATTERN.match(text_row["edition_id"]) or \
                not CACHED_TEXT_FILE_NAME_PATTERN.match(text_row["extracted_text_file_name"]):
            continue
        try:
            extracted_text = (source_files_dir / text_row["edition_id"] / text_row["extracted_text_file_name"]
                              ).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        connection.execute(
            """UPDATE edition_source_texts SET normalized_text_sha256 = ?, extracted_text_characters = ?
               WHERE edition_id = ?""",
            (fingerprint_extracted_reading_text(extracted_text), len(extracted_text.strip()),
             text_row["edition_id"]))


SCHEMA_MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration("initial schema (plan part 1)", MIGRATION_0001_INITIAL_SCHEMA),
    SchemaMigration("paper identity, editions, reading identity and part-2 records",
                    MIGRATION_0002_IDENTITY_EDITIONS_AND_RECORDS,
                    _backfill_schema_v1_rows_for_identity_and_editions),
    SchemaMigration("outbound request log and provider request budgets (plan part 3)",
                    MIGRATION_0003_OUTBOUND_REQUEST_LEDGER),
    SchemaMigration("seed cache for Zotero and CSL JSON seeds (plan part 4)", MIGRATION_0004_SEED_CACHE),
    SchemaMigration("route reports, withdrawals, and acquired source text (plan part 5)",
                    MIGRATION_0005_COLLECTION_ROUTES_WITHDRAWALS_AND_SOURCE_TEXT),
    SchemaMigration("screening scores, embedding cache, promotions, and dispatch order (plan part 6)",
                    MIGRATION_0006_SCREENING_AND_DISPATCH_ORDER),
    SchemaMigration("evidence-backed reading: stage results, cited passages, reading records (plan part 7)",
                    MIGRATION_0007_EVIDENCE_BACKED_READING, _backfill_full_text_reading_identity),
)
