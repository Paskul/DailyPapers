"""The paper catalog: one canonical paper per work, its trusted identifiers, its editions, and how it was found.

Identity rules (plan part 2):
  * Every identifier is normalized first (paperwatch.scholarly_identifiers). An alias (scheme, value) maps to
    exactly one paper (PRIMARY KEY), so the same work found by two routes lands on one paper.
  * A record joins an existing paper ONLY through an identifier: its own id, or a link its provider states
    explicitly (CollectedPaperRecord.linked_identifiers, e.g. an OpenAlex work listing its arXiv id). Similar titles
    and author names never merge papers; they only add a row to paper_possible_matches.
  * If one record's identifiers point at two different existing papers, nothing is merged automatically: the record
    joins the paper its own id belongs to (else the oldest one), and the conflict is flagged as a possible match.
  * Each distinct published version (arXiv v1, v2, a journal version) is an edition holding its own text snapshot.
    Edition rows are never rewritten, except that a title-only edition may receive its abstract later.
  * A second sighting never rewrites the paper's title (a changed title must not trigger another reading); it only
    adds aliases, editions and discovery routes, and advances the latest-revision date and version.
First publication, latest revision, and local discovery time are stored independently.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from paperwatch.clock import utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.identifiers import EDITION_ID_PREFIX, PAPER_ID_PREFIX, new_internal_id
from paperwatch.possible_matches import (MATCH_REASON_CONFLICTING_IDENTIFIER_LINK, flag_papers_with_same_title,
                                         record_possible_match)
from paperwatch.scholarly_identifiers import (NormalizedIdentifier, build_edition_key, identifiers_implied_by,
                                              normalize_identifier, parse_version_number,
                                              resolve_edition_version_label)
from paperwatch.sources.types import CollectedPaperRecord
from paperwatch.text_normalization import build_title_match_key


@dataclass(frozen=True)
class CatalogUpsertOutcome:
    paper_id: str
    edition_id: str
    was_newly_created: bool  # a new canonical paper
    was_new_edition: bool  # a version of this work we had not seen
    edition_received_abstract: bool  # a title-only edition now has its abstract (newly available text)
    possible_match_paper_ids: tuple[str, ...]  # flagged, NOT merged


def record_collected_paper(connection: sqlite3.Connection, collected_record: CollectedPaperRecord,
                           ) -> CatalogUpsertOutcome:
    """Resolve the record to one canonical paper and record its identifiers, edition and discovery route.

    Normalization happens before the transaction, so a malformed identifier fails without touching the database.
    """
    primary_identifier = normalize_identifier(collected_record.provider, collected_record.provider_paper_id)
    version_label = resolve_edition_version_label(primary_identifier, collected_record.source_version)
    edition_key = build_edition_key(primary_identifier, version_label)
    work_identifiers = _collect_work_identifiers(primary_identifier, collected_record)
    now_timestamp = utc_now_timestamp()

    with write_transaction(connection):
        paper_id_by_identifier = _look_up_papers_by_identifier(connection, work_identifiers)
        matched_paper_ids = list(dict.fromkeys(paper_id_by_identifier.values()))  # unique, in identifier order
        possible_match_paper_ids: list[str] = []

        if not matched_paper_ids:
            paper_id = _insert_new_paper(connection, collected_record, version_label, now_timestamp)
            was_newly_created = True
            possible_match_paper_ids += flag_papers_with_same_title(
                connection, new_paper_id=paper_id, title_match_key=build_title_match_key(collected_record.title),
                author_names=collected_record.authors)
        else:
            paper_id = _choose_paper_for_record(connection, primary_identifier, paper_id_by_identifier,
                                                matched_paper_ids)
            was_newly_created = False
            for conflicting_paper_id in matched_paper_ids:
                if conflicting_paper_id == paper_id:
                    continue
                record_possible_match(connection, paper_id, conflicting_paper_id,
                                      MATCH_REASON_CONFLICTING_IDENTIFIER_LINK,
                                      {"record": edition_key, "route": collected_record.discovery_route,
                                       "linking_identifiers": sorted(
                                           identifier.scheme_and_value for identifier, linked_paper_id
                                           in paper_id_by_identifier.items() if linked_paper_id in (
                                               paper_id, conflicting_paper_id))})
                possible_match_paper_ids.append(conflicting_paper_id)

        for work_identifier in work_identifiers:
            if work_identifier not in paper_id_by_identifier:  # never steal an alias that belongs to another paper
                connection.execute(
                    """INSERT OR IGNORE INTO paper_aliases (provider, provider_paper_id, paper_id, asserted_by_route,
                           first_seen_at) VALUES (?, ?, ?, ?, ?)""",
                    (work_identifier.scheme, work_identifier.value, paper_id, collected_record.discovery_route,
                     now_timestamp))

        edition_id, was_new_edition, edition_received_abstract = _record_edition(
            connection, paper_id, edition_key, version_label, collected_record, now_timestamp)
        if not was_newly_created:
            _advance_paper_from_new_sighting(connection, paper_id, collected_record, version_label,
                                             edition_received_abstract or was_new_edition, now_timestamp)
        connection.execute("INSERT OR IGNORE INTO paper_discoveries (paper_id, route, discovered_at) VALUES (?, ?, ?)",
                           (paper_id, collected_record.discovery_route, now_timestamp))

    return CatalogUpsertOutcome(paper_id=paper_id, edition_id=edition_id, was_newly_created=was_newly_created,
                                was_new_edition=was_new_edition, edition_received_abstract=edition_received_abstract,
                                possible_match_paper_ids=tuple(possible_match_paper_ids))


# --------------------------------------------------------------------------------------------------------------------
# Read helpers
# --------------------------------------------------------------------------------------------------------------------


def set_paper_status(connection: sqlite3.Connection, paper_id: str, new_status: str) -> None:
    connection.execute("UPDATE papers SET status = ?, updated_at = ? WHERE paper_id = ?",
                       (new_status, utc_now_timestamp(), paper_id))


def load_paper_row(connection: sqlite3.Connection, paper_id: str) -> sqlite3.Row:
    paper_row = connection.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if paper_row is None:
        raise LookupError(f"paper {paper_id} not found")
    return paper_row


def load_edition_row(connection: sqlite3.Connection, edition_id: str) -> sqlite3.Row:
    edition_row = connection.execute("SELECT * FROM paper_editions WHERE edition_id = ?", (edition_id,)).fetchone()
    if edition_row is None:
        raise LookupError(f"edition {edition_id} not found")
    return edition_row


def load_paper_aliases(connection: sqlite3.Connection, paper_id: str) -> list[tuple[str, str]]:
    return [(row["provider"], row["provider_paper_id"]) for row in connection.execute(
        "SELECT provider, provider_paper_id FROM paper_aliases WHERE paper_id = ? ORDER BY provider, provider_paper_id",
        (paper_id,))]


def find_paper_id_by_identifier(connection: sqlite3.Connection, identifier: NormalizedIdentifier) -> str | None:
    alias_row = connection.execute(
        "SELECT paper_id FROM paper_aliases WHERE provider = ? AND provider_paper_id = ?",
        (identifier.scheme, identifier.value)).fetchone()
    return alias_row["paper_id"] if alias_row else None


# --------------------------------------------------------------------------------------------------------------------
# Steps of record_collected_paper
# --------------------------------------------------------------------------------------------------------------------


def _collect_work_identifiers(primary_identifier: NormalizedIdentifier,
                              collected_record: CollectedPaperRecord) -> list[NormalizedIdentifier]:
    """Version-free identifiers of the WORK: the record's own id first, then provider-stated links, de-duplicated."""
    stated_identifiers = [primary_identifier] + [normalize_identifier(linked.scheme, linked.value)
                                                 for linked in collected_record.linked_identifiers]
    work_identifiers: list[NormalizedIdentifier] = []
    for stated_identifier in stated_identifiers:
        for work_identifier in (stated_identifier, *identifiers_implied_by(stated_identifier)):
            version_free_identifier = work_identifier.without_version()
            if version_free_identifier not in work_identifiers:
                work_identifiers.append(version_free_identifier)
    return work_identifiers


def _look_up_papers_by_identifier(connection: sqlite3.Connection, work_identifiers: list[NormalizedIdentifier],
                                  ) -> dict[NormalizedIdentifier, str]:
    paper_id_by_identifier: dict[NormalizedIdentifier, str] = {}
    for work_identifier in work_identifiers:
        existing_paper_id = find_paper_id_by_identifier(connection, work_identifier)
        if existing_paper_id is not None:
            paper_id_by_identifier[work_identifier] = existing_paper_id
    return paper_id_by_identifier


def _choose_paper_for_record(connection: sqlite3.Connection, primary_identifier: NormalizedIdentifier,
                             paper_id_by_identifier: dict[NormalizedIdentifier, str],
                             matched_paper_ids: list[str]) -> str:
    """One match: that paper. Several: the paper owning the record's own id, else the earliest catalogued one
    (discovery time, then insertion order, so the choice is deterministic even within one second)."""
    if len(matched_paper_ids) == 1:
        return matched_paper_ids[0]
    primary_paper_id = paper_id_by_identifier.get(primary_identifier.without_version())
    if primary_paper_id is not None:
        return primary_paper_id
    placeholders = ",".join("?" * len(matched_paper_ids))
    return connection.execute(
        f"SELECT paper_id FROM papers WHERE paper_id IN ({placeholders}) ORDER BY discovered_at, rowid LIMIT 1",
        matched_paper_ids).fetchone()["paper_id"]


def _insert_new_paper(connection: sqlite3.Connection, collected_record: CollectedPaperRecord,
                      version_label: str | None, now_timestamp: str) -> str:
    new_paper_id = new_internal_id(PAPER_ID_PREFIX)
    connection.execute(
        """INSERT INTO papers (paper_id, title, authors_json, abstract, first_published_date, latest_revision_date,
               source_version, origin_route, discovered_at, status, updated_at, title_match_key)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'waiting_input', ?, ?)""",
        (new_paper_id, collected_record.title, json.dumps(list(collected_record.authors), ensure_ascii=False),
         collected_record.abstract, collected_record.first_published_date, collected_record.latest_revision_date,
         version_label, collected_record.discovery_route, now_timestamp, now_timestamp,
         build_title_match_key(collected_record.title)))
    return new_paper_id


def _record_edition(connection: sqlite3.Connection, paper_id: str, edition_key: str, version_label: str | None,
                    collected_record: CollectedPaperRecord, now_timestamp: str) -> tuple[str, bool, bool]:
    """Returns (edition_id, was_new_edition, edition_received_abstract)."""
    existing_edition_row = connection.execute(
        "SELECT edition_id, abstract FROM paper_editions WHERE edition_key = ?", (edition_key,)).fetchone()
    if existing_edition_row is not None:
        if existing_edition_row["abstract"] is None and collected_record.abstract:
            connection.execute("UPDATE paper_editions SET abstract = ?, abstract_added_at = ? WHERE edition_id = ?",
                               (collected_record.abstract, now_timestamp, existing_edition_row["edition_id"]))
            return existing_edition_row["edition_id"], False, True
        return existing_edition_row["edition_id"], False, False

    new_edition_id = new_internal_id(EDITION_ID_PREFIX)
    connection.execute(
        """INSERT INTO paper_editions (edition_id, paper_id, edition_key, version_label, version_number, title,
               authors_json, abstract, published_date, first_seen_route, first_seen_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (new_edition_id, paper_id, edition_key, version_label, parse_version_number(version_label),
         collected_record.title, json.dumps(list(collected_record.authors), ensure_ascii=False),
         collected_record.abstract, collected_record.latest_revision_date or collected_record.first_published_date,
         collected_record.discovery_route, now_timestamp))
    return new_edition_id, True, False


def _advance_paper_from_new_sighting(connection: sqlite3.Connection, paper_id: str,
                                     collected_record: CollectedPaperRecord, version_label: str | None,
                                     sighting_added_text: bool, now_timestamp: str) -> None:
    """Fill gaps and move the latest revision forward. The title and origin route are never rewritten."""
    paper_row = load_paper_row(connection, paper_id)
    column_updates: dict[str, str] = {}
    if paper_row["abstract"] is None and collected_record.abstract and sighting_added_text:
        column_updates["abstract"] = collected_record.abstract  # display copy; readings use edition snapshots
    if paper_row["first_published_date"] is None and collected_record.first_published_date:
        column_updates["first_published_date"] = collected_record.first_published_date
    if collected_record.latest_revision_date and (paper_row["latest_revision_date"] is None or
                                                  collected_record.latest_revision_date
                                                  > paper_row["latest_revision_date"]):
        column_updates["latest_revision_date"] = collected_record.latest_revision_date
    new_version_number = parse_version_number(version_label)
    if new_version_number is not None and new_version_number > (parse_version_number(paper_row["source_version"])
                                                                or 0):
        column_updates["source_version"] = version_label
    if column_updates:
        assignments = ", ".join(f"{column_name} = ?" for column_name in column_updates)
        connection.execute(f"UPDATE papers SET {assignments}, updated_at = ? WHERE paper_id = ?",
                           (*column_updates.values(), now_timestamp, paper_id))
