"""Collector stage: fetch the outgoing references of reviewed papers (plan part 8). Runs inside `collect` only.

For each reviewed paper worth following (citation_hops.find_reviewed_papers_to_follow), at most
references.reference_fetches_per_run per run:
  1. Find its OpenAlex work: its own OpenAlex alias, else a lookup by its DOI, else by arXiv's DataCite DOI of its
     arXiv id. The returned record must link back to this paper through an identifier it already has; only then is
     the OpenAlex id added as an alias. The record is NOT stored as a new edition (that would change which abstract
     the paper is read from and could trigger a second reading).
  2. Store one citation edge per id in its `referenced_works` (provenance OPENALEX_REFERENCES_PROVENANCE), then look
     the cited works up in batches of 100 (get_works: every id must be such a stored edge) and catalog them under the
     route reference:<citing paper id>. They are then screened like every other candidate.
  3. "When the graph is incomplete": DOIs and arXiv ids in the paper's own bibliography that no edge covers yet are
     stored as pending bibliography identifiers and looked up (at most references.bibliography_lookups_per_paper).
     A provider record whose title matches the bibliography entry becomes a cited work and an edge (provenance
     BIBLIOGRAPHY_CONFIRMED_PROVENANCE); any other outcome stays stored locally as an unresolved reference.
  4. The fetch is recorded in reference_fetches, so a paper's references are fetched once.
A provider outage or a used-up request budget stops the stage for this run; the paper is retried next run (up to
MAXIMUM_REFERENCE_FETCH_ATTEMPTS times).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from paperwatch.catalog import find_paper_id_by_identifier, load_paper_aliases, prepare_catalog_record
from paperwatch.clock import utc_now_timestamp
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.config import MAXIMUM_REFERENCES_PER_LOOKUP_REQUEST
from paperwatch.db import write_transaction
from paperwatch.errors import (
    PaperwatchError,
    ProviderUnavailable,
    SourceAccessDenied,
    SourceInputError,
    SourceRecordNotFound,
    SourceRequestBudgetExhausted,
)
from paperwatch.references.bibliography_identifiers import (
    find_bibliography_identifiers,
    provider_title_matches_entry,
)
from paperwatch.references.citation_hops import ReviewedPaperToFollow, find_reviewed_papers_to_follow
from paperwatch.references.paper_text import bibliography_passages, load_extracted_text_passages
from paperwatch.research_graph import record_citation_edge, resolve_citation_edges
from paperwatch.scholarly_identifiers import (
    ARXIV_DATACITE_DOI_PREFIX,
    ARXIV_SCHEME,
    DOI_SCHEME,
    OPENALEX_SCHEME,
    NormalizedIdentifier,
)
from paperwatch.sources.arxiv_atom import parse_api_search_page
from paperwatch.sources.discovery_routes import openalex_lookup_route, reference_route
from paperwatch.sources.openalex_works import ParsedSingleWork, parse_single_work, parse_works_page
from paperwatch.sources.types import CollectedPaperRecord

OPENALEX_REFERENCES_PROVENANCE = "openalex:referenced_works"
BIBLIOGRAPHY_CONFIRMED_PROVENANCE = "bibliography:confirmed_by_provider"
FETCH_STATUS_FETCHED = "fetched"
FETCH_STATUS_NO_PROVIDER_RECORD = "no_provider_record"
FETCH_STATUS_BLOCKED = "blocked"
MAXIMUM_STORED_ERROR_CHARACTERS = 300


@dataclass
class ReferenceFetchRunCounts:
    papers_fetched: int = 0
    papers_without_provider_record: int = 0
    papers_blocked: int = 0
    referenced_ids: int = 0
    newly_catalogued_works: int = 0
    bibliography_confirmed: int = 0
    bibliography_unresolved: int = 0
    stopped_reason: str | None = None

    def describe(self, considered_count: int) -> str:
        description = (
            f"References: {considered_count} reviewed paper(s) to follow; {self.papers_fetched} fetched "
            f"({self.referenced_ids} cited id(s), {self.newly_catalogued_works} work(s) new to the "
            f"catalog), {self.papers_without_provider_record} without a provider record, "
            f"{self.papers_blocked} blocked; bibliography: {self.bibliography_confirmed} confirmed, "
            f"{self.bibliography_unresolved} unresolved"
        )
        return description + (f"; stopped: {self.stopped_reason}" if self.stopped_reason else "")


@dataclass(frozen=True)
class OneFetchOutcome:
    openalex_work_id: str | None
    referenced_id_count: int
    newly_catalogued_works: int
    bibliography_found: int
    bibliography_confirmed: int
    bibliography_unresolved: int


def fetch_references_of_reviewed_papers(context: CollectionContext) -> list[str]:
    reference_settings = context.app_config.references
    if not reference_settings.follow_references:
        return ["References: not followed (references.follow_references = false)"]
    parents = find_reviewed_papers_to_follow(
        context.connection, context.app_config, reference_settings.reference_fetches_per_run
    )
    run_counts = ReferenceFetchRunCounts()
    for parent in parents:
        try:
            outcome = fetch_references_of_one_paper(context, parent)
        except (ProviderUnavailable, SourceRequestBudgetExhausted) as provider_wide_problem:
            _record_fetch(context.connection, parent, FETCH_STATUS_BLOCKED, error=provider_wide_problem)
            run_counts.papers_blocked += 1
            run_counts.stopped_reason = f"[{provider_wide_problem.failed_component}] {provider_wide_problem}"
            break
        except (SourceAccessDenied, SourceInputError) as paper_problem:
            _record_fetch(context.connection, parent, FETCH_STATUS_BLOCKED, error=paper_problem)
            run_counts.papers_blocked += 1
            continue
        final_status = (
            FETCH_STATUS_FETCHED
            if outcome.openalex_work_id or outcome.bibliography_found
            else FETCH_STATUS_NO_PROVIDER_RECORD
        )
        _record_fetch(context.connection, parent, final_status, outcome=outcome)
        if final_status == FETCH_STATUS_FETCHED:
            run_counts.papers_fetched += 1
        else:
            run_counts.papers_without_provider_record += 1
        run_counts.referenced_ids += outcome.referenced_id_count
        run_counts.newly_catalogued_works += outcome.newly_catalogued_works
        run_counts.bibliography_confirmed += outcome.bibliography_confirmed
        run_counts.bibliography_unresolved += outcome.bibliography_unresolved
    return [run_counts.describe(len(parents))]


def fetch_references_of_one_paper(context: CollectionContext, parent: ReviewedPaperToFollow) -> OneFetchOutcome:
    openalex_work = look_up_openalex_work_of_paper(context, parent.paper_id)
    referenced_work_ids: list[str] = []
    newly_catalogued_works = 0
    if openalex_work is not None:
        referenced_work_ids = list(
            openalex_work.referenced_work_ids[: context.app_config.references.max_references_fetched_per_paper]
        )
        with write_transaction(context.connection):
            for work_id in referenced_work_ids:
                record_citation_edge(
                    context.connection,
                    citing_paper_id=parent.paper_id,
                    cited_identifier=NormalizedIdentifier(OPENALEX_SCHEME, work_id),
                    provenance=OPENALEX_REFERENCES_PROVENANCE,
                )
        for batch_start in range(0, len(referenced_work_ids), MAXIMUM_REFERENCES_PER_LOOKUP_REQUEST):
            batch = referenced_work_ids[batch_start : batch_start + MAXIMUM_REFERENCES_PER_LOOKUP_REQUEST]
            provider_response = context.gateway.get_works([f"{OPENALEX_SCHEME}:{work_id}" for work_id in batch])
            parsed_page = parse_works_page(provider_response.body_bytes, reference_route(parent.paper_id))
            newly_catalogued_works += _catalog_cited_records(context, parsed_page.records)
    bibliography_counts = confirm_bibliography_identifiers(context, parent.paper_id)
    with write_transaction(context.connection):
        resolve_citation_edges(context.connection)
    return OneFetchOutcome(
        openalex_work_id=openalex_work.record.provider_paper_id if openalex_work and openalex_work.record else None,
        referenced_id_count=len(referenced_work_ids),
        newly_catalogued_works=newly_catalogued_works,
        bibliography_found=bibliography_counts[0],
        bibliography_confirmed=bibliography_counts[1],
        bibliography_unresolved=bibliography_counts[2],
    )


# --------------------------------------------------------------------------------------------------------------------
# Step 1: the paper's own OpenAlex work
# --------------------------------------------------------------------------------------------------------------------


def openalex_lookup_identifiers(connection: sqlite3.Connection, paper_id: str) -> list[str]:
    """'scheme:value' ids to try, best first: the OpenAlex id, the DOIs, then arXiv's DataCite DOI of the arXiv id."""
    aliases = load_paper_aliases(connection, paper_id)
    lookup_ids = [f"{OPENALEX_SCHEME}:{value}" for scheme, value in aliases if scheme == OPENALEX_SCHEME]
    lookup_ids += [f"{DOI_SCHEME}:{value}" for scheme, value in aliases if scheme == DOI_SCHEME]
    lookup_ids += [
        f"{DOI_SCHEME}:{ARXIV_DATACITE_DOI_PREFIX}{value}" for scheme, value in aliases if scheme == ARXIV_SCHEME
    ]
    return list(dict.fromkeys(lookup_ids))


def look_up_openalex_work_of_paper(context: CollectionContext, paper_id: str) -> ParsedSingleWork | None:
    for lookup_id in openalex_lookup_identifiers(context.connection, paper_id):
        try:
            provider_response = context.gateway.get_work(lookup_id)
        except SourceRecordNotFound:
            continue
        parsed_work = parse_single_work(provider_response.body_bytes, openalex_lookup_route(paper_id))
        if parsed_work.record is not None and _record_belongs_to_paper(
            context.connection, parsed_work.record, paper_id
        ):
            _record_openalex_alias(context.connection, paper_id, parsed_work.record)
            return parsed_work
    return None


def _record_belongs_to_paper(connection: sqlite3.Connection, record: CollectedPaperRecord, paper_id: str) -> bool:
    """The OpenAlex record links to this paper through an identifier the paper already has, and its own OpenAlex id
    belongs to no other paper. Look-alike titles never count."""
    try:
        prepared_record = prepare_catalog_record(record)
    except SourceInputError:
        return False
    owners = {
        identifier: find_paper_id_by_identifier(connection, identifier)
        for identifier in prepared_record.work_identifiers
    }
    if owners.get(prepared_record.primary_identifier) not in (None, paper_id):
        return False
    return (
        any(
            owner == paper_id
            for identifier, owner in owners.items()
            if identifier != prepared_record.primary_identifier
        )
        or owners.get(prepared_record.primary_identifier) == paper_id
    )


def _record_openalex_alias(connection: sqlite3.Connection, paper_id: str, record: CollectedPaperRecord) -> None:
    openalex_identifier = prepare_catalog_record(record).primary_identifier
    with write_transaction(connection):
        connection.execute(
            """INSERT OR IGNORE INTO paper_aliases (provider, provider_paper_id, paper_id, asserted_by_route,
                   first_seen_at) VALUES (?, ?, ?, ?, ?)""",
            (
                openalex_identifier.scheme,
                openalex_identifier.value,
                paper_id,
                record.discovery_route,
                utc_now_timestamp(),
            ),
        )


def _catalog_cited_records(context: CollectionContext, records: list[CollectedPaperRecord]) -> int:
    if not records:
        return 0
    page_result = ingest_records_with_checkpoints(
        context.connection,
        records,
        admission_limit=None,
        active_seed_paper_ids=context.active_seed_paper_ids,
        max_attempts_per_job=context.app_config.worker.max_attempts_per_job,
    )
    return page_result.new_papers_admitted


# --------------------------------------------------------------------------------------------------------------------
# Step 3: identifiers in the paper's own bibliography
# --------------------------------------------------------------------------------------------------------------------


def confirm_bibliography_identifiers(context: CollectionContext, citing_paper_id: str) -> tuple[int, int, int]:
    """Store new bibliography identifiers, then look up the pending ones. Returns (found, confirmed, unresolved)."""
    connection = context.connection
    passages = load_extracted_text_passages(connection, context.app_config, citing_paper_id)
    found_identifiers = find_bibliography_identifiers(bibliography_passages(passages)) if passages else []
    uncovered_identifiers = [
        found
        for found in found_identifiers
        if not _is_covered_by_existing_edge(connection, citing_paper_id, found.identifier)
    ]
    with write_transaction(connection):
        for found in uncovered_identifiers:
            connection.execute(
                """INSERT OR IGNORE INTO bibliography_identifiers (citing_paper_id, identifier_scheme,
                       identifier_value, passage_id, entry_excerpt, confirmation_status, recorded_at)
                   VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    citing_paper_id,
                    found.identifier.scheme,
                    found.identifier.value,
                    found.passage_id,
                    found.stored_excerpt,
                    utc_now_timestamp(),
                ),
            )
    pending_rows = connection.execute(
        """SELECT * FROM bibliography_identifiers WHERE citing_paper_id = ? AND confirmation_status = 'pending'
           ORDER BY recorded_at, identifier_scheme, identifier_value LIMIT ?""",
        (citing_paper_id, context.app_config.references.bibliography_lookups_per_paper),
    ).fetchall()
    confirmed_count = unresolved_count = 0
    for pending_row in pending_rows:
        new_status = _confirm_one_bibliography_identifier(context, citing_paper_id, pending_row)
        confirmed_count += new_status == "confirmed"
        unresolved_count += new_status != "confirmed"
    return len(found_identifiers), confirmed_count, unresolved_count


def _is_covered_by_existing_edge(
    connection: sqlite3.Connection, citing_paper_id: str, identifier: NormalizedIdentifier
) -> bool:
    """The identifier is the citing paper itself, or already a cited work of an edge from it."""
    owner_paper_id = find_paper_id_by_identifier(connection, identifier)
    if owner_paper_id == citing_paper_id:
        return True
    return (
        connection.execute(
            """SELECT 1 FROM citation_edges WHERE citing_paper_id = ?
             AND ((cited_scheme = ? AND cited_value = ?) OR (cited_paper_id IS NOT NULL AND cited_paper_id = ?))
           LIMIT 1""",
            (citing_paper_id, identifier.scheme, identifier.value, owner_paper_id),
        ).fetchone()
        is not None
    )


def _confirm_one_bibliography_identifier(
    context: CollectionContext, citing_paper_id: str, pending_row: sqlite3.Row
) -> str:
    """Look the identifier up and compare the provider's title with the bibliography entry. Provider-wide problems
    propagate (the row stays pending and is retried on a later run)."""
    identifier = NormalizedIdentifier(pending_row["identifier_scheme"], pending_row["identifier_value"])
    try:
        provider_response = context.gateway.get_work(identifier.scheme_and_value)
        provider_record = _parse_single_record(identifier, provider_response.body_bytes, citing_paper_id)
    except SourceRecordNotFound:
        provider_record = None
    except SourceRequestBudgetExhausted:
        raise
    except (SourceAccessDenied, SourceInputError):  # refused at dispatch, or a malformed provider record
        _settle_bibliography_identifier(context.connection, pending_row, "refused", None)
        return "refused"
    if provider_record is None:
        _settle_bibliography_identifier(context.connection, pending_row, "not_found", None)
        return "not_found"
    if not provider_title_matches_entry(provider_record.title, pending_row["entry_excerpt"]):
        _settle_bibliography_identifier(context.connection, pending_row, "mismatch", None)
        return "mismatch"
    _catalog_cited_records(context, [provider_record])
    with write_transaction(context.connection):
        record_citation_edge(
            context.connection,
            citing_paper_id=citing_paper_id,
            cited_identifier=identifier,
            provenance=BIBLIOGRAPHY_CONFIRMED_PROVENANCE,
        )
    _settle_bibliography_identifier(
        context.connection, pending_row, "confirmed", find_paper_id_by_identifier(context.connection, identifier)
    )
    return "confirmed"


def _parse_single_record(
    identifier: NormalizedIdentifier, body_bytes: bytes, citing_paper_id: str
) -> CollectedPaperRecord | None:
    route = reference_route(citing_paper_id)
    if identifier.scheme == ARXIV_SCHEME:
        parsed_page = parse_api_search_page(body_bytes, route, start_offset=0)
        return parsed_page.records[0] if parsed_page.records else None
    return parse_single_work(body_bytes, route).record


def _settle_bibliography_identifier(
    connection: sqlite3.Connection, pending_row: sqlite3.Row, new_status: str, confirmed_paper_id: str | None
) -> None:
    with write_transaction(connection):
        connection.execute(
            """UPDATE bibliography_identifiers SET confirmation_status = ?, confirmed_paper_id = ?, checked_at = ?
               WHERE citing_paper_id = ? AND identifier_scheme = ? AND identifier_value = ?""",
            (
                new_status,
                confirmed_paper_id,
                utc_now_timestamp(),
                pending_row["citing_paper_id"],
                pending_row["identifier_scheme"],
                pending_row["identifier_value"],
            ),
        )


# --------------------------------------------------------------------------------------------------------------------
# Step 4: the fetch record
# --------------------------------------------------------------------------------------------------------------------


def _record_fetch(
    connection: sqlite3.Connection,
    parent: ReviewedPaperToFollow,
    fetch_status: str,
    *,
    outcome: OneFetchOutcome | None = None,
    error: PaperwatchError | None = None,
) -> None:
    now_timestamp = utc_now_timestamp()
    catalogued_count = connection.execute(
        "SELECT COUNT(DISTINCT cited_paper_id) FROM citation_edges WHERE citing_paper_id = ? "
        "AND cited_paper_id IS NOT NULL",
        (parent.paper_id,),
    ).fetchone()[0]
    with write_transaction(connection):
        connection.execute(
            """INSERT INTO reference_fetches (citing_paper_id, parent_review_id, citation_hop_depth, fetch_status,
                   openalex_work_id, referenced_work_count, catalogued_reference_count, bibliography_identifier_count,
                   attempts, last_error, fetched_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
               ON CONFLICT (citing_paper_id) DO UPDATE SET
                   parent_review_id = excluded.parent_review_id, fetch_status = excluded.fetch_status,
                   openalex_work_id = COALESCE(excluded.openalex_work_id, openalex_work_id),
                   referenced_work_count = excluded.referenced_work_count,
                   catalogued_reference_count = excluded.catalogued_reference_count,
                   bibliography_identifier_count = excluded.bibliography_identifier_count,
                   attempts = attempts + 1, last_error = excluded.last_error,
                   fetched_at = COALESCE(excluded.fetched_at, fetched_at), updated_at = excluded.updated_at""",
            (
                parent.paper_id,
                parent.review_id,
                parent.citation_hop_depth,
                fetch_status,
                outcome.openalex_work_id if outcome else None,
                outcome.referenced_id_count if outcome else 0,
                catalogued_count,
                outcome.bibliography_found if outcome else 0,
                f"[{error.failed_component}] {error}"[:MAXIMUM_STORED_ERROR_CHARACTERS] if error else None,
                now_timestamp if fetch_status != FETCH_STATUS_BLOCKED else None,
                now_timestamp,
            ),
        )
