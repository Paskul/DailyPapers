"""Plan part 4: the research profile from Zotero, CSL JSON, topics, or both.

Check from the plan: run all three input modes (keywords alone, seeds alone, both), including a missing Zotero key and
a title-only seed. Inspect the recorded outgoing query parameters: only approved public terms, categories, and
permitted ids may leave the machine.

No real network: Zotero is a FakeZoteroLibrary behind httpx.MockTransport, and DNS is a fake resolver.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from conftest import FIXTURES_DIR, build_in_process_app

from paperwatch.catalog import record_collected_paper
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.config import TopicDefinition
from paperwatch.errors import SourceAccessDenied
from paperwatch.network.collector_credentials import CollectorCredentials
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.network.provider_rules import load_provider_access_rules, packaged_provider_rules_path
from paperwatch.public_query_catalog import build_public_query_catalog
from paperwatch.research_profile import (
    TopicInputMode,
    build_research_profiles,
    compute_topic_profile_hashes,
    load_active_seed_library_paper_ids,
)
from paperwatch.seeds.csl_json_items import parse_csl_json_export
from paperwatch.seeds.seed_items import build_csl_json_file_key
from paperwatch.seeds.seed_store import SeedSourceReference, load_active_seeds
from paperwatch.seeds.seed_sync import (
    SeedSyncResult,
    import_csl_json_seed_file,
    synchronize_all_seed_sources,
    synchronize_zotero_collection,
)
from paperwatch.seeds.zotero_items import parse_zotero_items_page
from paperwatch.sources.types import CollectedPaperRecord, RawIdentifier

PUBLIC_ADDRESS = "93.184.216.34"
ZOTERO_COLLECTION_KEY = "ABCD2345"
ZOTERO_LIBRARY_ID = "4242"
ZOTERO_API_KEY = "SECRET-ZOTERO-KEY-987"
ZOTERO_CREDENTIALS = CollectorCredentials(
    values_by_provider={"zotero": {"api_key": ZOTERO_API_KEY, "library_id": ZOTERO_LIBRARY_ID}}
)
CSL_SEED_FIXTURE = FIXTURES_DIR / "seeds" / "team_safety_seeds.csl.json"

KEYWORDS_QUESTION = "PRIVATE-QUESTION-KW: Which exit-clearance monitors work for drone swarms?"
BOTH_QUESTION = "PRIVATE-QUESTION-BOTH: Can a runtime monitor reject a joint robot plan before it blocks an exit?"
THREE_MODE_TOPICS = f"""
[[topics]]
id = "keywords-only"
question = "{KEYWORDS_QUESTION}"
public_queries = ["swarm exit clearance"]
arxiv_categories = ["cs.MA"]

[[topics]]
id = "seeds-only"
arxiv_categories = ["cs.RO"]
csl_json_files = ["seeds/team_safety_seeds.csl.json"]

[[topics]]
id = "keywords-and-seeds"
question = "{BOTH_QUESTION}"
public_queries = ["multi-robot runtime assurance"]
arxiv_categories = ["cs.LO"]
zotero_collections = ["{ZOTERO_COLLECTION_KEY}"]
"""
# Private text that must never appear in any outbound request: questions, notes, seed titles, collection names.
PRIVATE_MARKERS = (
    "PRIVATE-QUESTION",
    "PRIVATE-NOTE",
    "PRIVATE-CSL-NOTE",
    "Private Collection Name",
    "synthetic seed",
    "Zotero seed",
    "SYNTHETIC SEED",
)


# --------------------------------------------------------------------------------------------------------------------
# A fake Zotero Web API library
# --------------------------------------------------------------------------------------------------------------------


def zotero_item(
    item_key: str,
    title: str | None,
    *,
    item_type: str = "journalArticle",
    abstract: str = "",
    version: int = 1,
    **extra_data,
) -> dict:
    data = {
        "key": item_key,
        "version": version,
        "itemType": item_type,
        "abstractNote": abstract,
        "creators": [
            {"creatorType": "author", "firstName": "Ada", "lastName": "Zoterian"},
            {"creatorType": "editor", "name": "Ed Itor"},
        ],
        "date": "2020-03-04",
        "collections": [ZOTERO_COLLECTION_KEY],
        **extra_data,
    }
    if title is not None:
        data["title"] = title
    return {
        "key": item_key,
        "version": version,
        "library": {"type": "user", "id": int(ZOTERO_LIBRARY_ID), "name": "Private Collection Name"},
        "meta": {"parsedDate": "2020-03-04"},
        "data": data,
    }


def standard_collection_items() -> list[dict]:
    return [
        zotero_item(
            "ITEMDOI1",
            "Runtime Assurance for Robot Teams (Zotero seed)",
            abstract="Zotero seed abstract.",
            DOI="https://doi.org/10.5555/ZOTERO.SEED.1",
        ),
        zotero_item(
            "ITEMARX2",
            "Monitoring Joint Plans (Zotero seed)",
            item_type="preprint",
            abstract="Zotero seed preprint abstract.",
            archiveID="arXiv:2304.04321",
            extra="Citation Key: monitor2023\nPRIVATE-NOTE: personal remark kept in Extra",
        ),
        zotero_item("ITEMTTL3", "Title Only Zotero seed About Exits"),  # no abstract: matched by title only
        zotero_item("NOTEITM4", None, item_type="note", note="<p>PRIVATE-NOTE: never a seed</p>"),
        zotero_item("ATTACHM5", "paper.pdf", item_type="attachment", contentType="application/pdf"),
        zotero_item("TRASHED6", "Trashed Zotero seed", deleted=1),
    ]


class FakeZoteroLibrary:
    """Serves /users/<id>/collections/<key>/items/top with paging, library versions, and 304 Not Modified."""

    def __init__(self, items: list[dict], library_version: int = 100) -> None:
        self.items = items
        self.library_version = library_version
        self.version_bumps_after_each_page = 0  # simulate edits while a sync is paging
        self.sent_requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self._respond)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.sent_requests.append(request)
        version_headers = {"Last-Modified-Version": str(self.library_version)}
        known_version = request.headers.get("If-Modified-Since-Version")
        if known_version is not None and int(known_version) >= self.library_version:
            return httpx.Response(304, headers=version_headers, stream=httpx.ByteStream(b""))
        start, limit = int(request.url.params["start"]), int(request.url.params["limit"])
        page_items = self.items[start : start + limit]
        response = httpx.Response(
            200,
            headers={"content-type": "application/json", **version_headers, "Total-Results": str(len(self.items))},
            stream=httpx.ByteStream(json.dumps(page_items).encode()),
        )
        self.library_version += self.version_bumps_after_each_page
        return response


def zotero_rules_with_page_size(page_size: int) -> str:
    rules_text = packaged_provider_rules_path().read_text()
    old_text = "minimum_seconds_between_requests = 1.0\nsearch_page_size = 100"
    assert old_text in rules_text
    return rules_text.replace(old_text, f"minimum_seconds_between_requests = 0.0\nsearch_page_size = {page_size}")


def make_gateway(app, transport: httpx.BaseTransport, *, credentials=ZOTERO_CREDENTIALS, page_size: int = 2):
    return ProviderGateway(
        app.connection,
        app.config,
        credentials=credentials,
        http_transport=transport,
        provider_rules=load_provider_access_rules(zotero_rules_with_page_size(page_size)),
        address_resolver=lambda host, port: [PUBLIC_ADDRESS],
        sleep_function=lambda _: None,
    )


def outbound_rows(app):
    return app.connection.execute("SELECT * FROM outbound_requests ORDER BY started_at, rowid").fetchall()


def build_three_mode_app(tmp_path: Path, **app_options):
    home_dir = tmp_path / "home"
    test_app = build_in_process_app(home_dir, topics_toml=THREE_MODE_TOPICS, **app_options)
    (home_dir / "seeds").mkdir()
    shutil.copy(CSL_SEED_FIXTURE, home_dir / "seeds" / "team_safety_seeds.csl.json")
    return test_app


@pytest.fixture
def app(tmp_path):
    test_app = build_three_mode_app(tmp_path)
    yield test_app
    test_app.close()


ZOTERO_SOURCE = SeedSourceReference.for_zotero_collection(ZOTERO_COLLECTION_KEY)
CSL_SOURCE = SeedSourceReference.for_csl_json_file("seeds/team_safety_seeds.csl.json")


def profiles_by_topic(app):
    return {profile.topic_id: profile for profile in build_research_profiles(app.connection, app.config)}


# --------------------------------------------------------------------------------------------------------------------
# The plan's check: three input modes, missing key, title-only seed, and what leaves the machine
# --------------------------------------------------------------------------------------------------------------------


def test_three_input_modes_send_only_approved_terms_and_permitted_ids(app, capsys):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    gateway = make_gateway(app, fake_library.transport)
    assert (
        run_collect_command(
            app, argparse.Namespace(fixture_paths=None, skip_searches=True), build_gateway=lambda: gateway
        )
        == 0
    )

    profiles = profiles_by_topic(app)
    assert profiles["keywords-only"].input_mode == TopicInputMode.KEYWORDS
    assert profiles["seeds-only"].input_mode == TopicInputMode.SEEDS
    assert profiles["keywords-and-seeds"].input_mode == TopicInputMode.KEYWORDS_AND_SEEDS
    assert len(profiles["seeds-only"].active_seeds) == 3 and profiles["seeds-only"].title_only_seed_count == 1
    assert len(profiles["keywords-and-seeds"].active_seeds) == 3
    assert profiles["keywords-and-seeds"].title_only_seed_count == 1
    printed = capsys.readouterr().out
    assert "Topic seeds-only: seeds; 3 active seed(s)" in printed
    assert "Topic keywords-only: keywords; 0 active seed(s)" in printed

    # Now run every search the profiles approve, as part 5 will, and inspect everything that left the machine.
    search_transport_requests: list[httpx.Request] = []

    def answer_search(request: httpx.Request) -> httpx.Response:
        search_transport_requests.append(request)
        content_type = "application/json" if request.url.host == "api.openalex.org" else "application/atom+xml"
        return httpx.Response(200, headers={"content-type": content_type}, stream=httpx.ByteStream(b"{}"))

    search_gateway = make_gateway(app, httpx.MockTransport(answer_search))
    for profile in profiles.values():
        for query_id in profile.approved_public_query_ids:
            search_gateway.search(query_id)

    approved_values = {
        public_query.public_value for public_query in build_public_query_catalog(app.config.topics).values()
    }
    assert approved_values == {"swarm exit clearance", "cs.MA", "cs.RO", "multi-robot runtime assurance", "cs.LO"}
    seeds_only_queries = {
        build_public_query_catalog(app.config.topics)[query_id].public_value
        for query_id in profiles["seeds-only"].approved_public_query_ids
    }
    assert seeds_only_queries == {"cs.RO"}  # seeds alone: only the public category is searched

    for search_request in search_transport_requests:
        free_text_value = search_request.url.params.get("search_query") or search_request.url.params.get("search")
        assert any(approved_value in free_text_value for approved_value in approved_values), free_text_value
    for zotero_request in fake_library.sent_requests:
        assert zotero_request.url.path == f"/users/{ZOTERO_LIBRARY_ID}/collections/{ZOTERO_COLLECTION_KEY}/items/top"
        assert set(zotero_request.url.params) == {
            "format",
            "include",
            "itemType",
            "sort",
            "direction",
            "start",
            "limit",
        }

    everything_sent = " ".join(
        str(request.url) + json.dumps(dict(request.headers))
        for request in search_transport_requests + fake_library.sent_requests
    )
    everything_logged = json.dumps([dict(row) for row in outbound_rows(app)])
    for private_marker in PRIVATE_MARKERS:
        assert private_marker not in everything_sent, private_marker
        assert private_marker not in everything_logged, private_marker
    assert ZOTERO_API_KEY not in everything_logged  # the key is sent to Zotero only, and redacted in the log
    assert {row["provider_name"] for row in outbound_rows(app)} == {"zotero", "arxiv", "openalex"}


@pytest.mark.parametrize(
    "credentials, expected_reason",
    [
        (CollectorCredentials(), "library_id is missing"),
        (CollectorCredentials(values_by_provider={"zotero": {"library_id": ZOTERO_LIBRARY_ID}}), "needs an API key"),
    ],
)
def test_missing_zotero_key_sends_nothing_and_keeps_every_topic_working(app, capsys, credentials, expected_reason):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    gateway = make_gateway(app, fake_library.transport, credentials=credentials)
    assert (
        run_collect_command(
            app, argparse.Namespace(fixture_paths=None, skip_searches=True), build_gateway=lambda: gateway
        )
        == 0
    )

    printed = capsys.readouterr().out
    assert f"Zotero collection {ZOTERO_COLLECTION_KEY}: NOT synced" in printed and expected_reason in printed
    assert fake_library.sent_requests == [] and outbound_rows(app) == []
    profiles = profiles_by_topic(app)
    assert profiles["keywords-and-seeds"].input_mode == TopicInputMode.KEYWORDS  # falls back to its keywords
    assert profiles["seeds-only"].input_mode == TopicInputMode.SEEDS  # the CSL JSON source is unaffected
    checkpoint_row = app.connection.execute(
        "SELECT coverage_status, stop_reason FROM source_checkpoints WHERE search_key = ?",
        (ZOTERO_SOURCE.checkpoint_key,),
    ).fetchone()
    assert checkpoint_row["coverage_status"] == "blocked" and expected_reason in checkpoint_row["stop_reason"]


def test_missing_key_keeps_the_last_successful_seed_set(app):
    synchronize_zotero_collection(
        app.connection, make_gateway(app, FakeZoteroLibrary(standard_collection_items()).transport), ZOTERO_SOURCE
    )
    outcome = synchronize_zotero_collection(
        app.connection,
        make_gateway(app, FakeZoteroLibrary([]).transport, credentials=CollectorCredentials()),
        ZOTERO_SOURCE,
    )
    assert outcome.result == SeedSyncResult.UNAVAILABLE and outcome.active_seed_count == 3
    assert len(load_active_seeds(app.connection, [ZOTERO_SOURCE])) == 3


# --------------------------------------------------------------------------------------------------------------------
# Zotero live sync: notes and attachments, title-only seeds, library versions, removals, mid-sync changes
# --------------------------------------------------------------------------------------------------------------------


def test_zotero_sync_reads_bibliographic_items_only_and_records_title_only_seeds(app):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    outcome = synchronize_zotero_collection(app.connection, make_gateway(app, fake_library.transport), ZOTERO_SOURCE)
    assert outcome.result == SeedSyncResult.SYNCED and outcome.detail == "library version 100"
    assert (outcome.active_seed_count, outcome.title_only_seed_count) == (3, 1)
    assert outcome.skipped_item_counts is not None
    assert sum(outcome.skipped_item_counts.values()) == 3  # the note, the attachment, the trashed item
    assert len(fake_library.sent_requests) == 3  # 6 items in pages of 2

    seeds = {seed.seed_id: seed for seed in load_active_seeds(app.connection, [ZOTERO_SOURCE])}
    assert set(seeds) == {"zotero:ITEMDOI1", "zotero:ITEMARX2", "zotero:ITEMTTL3"}
    assert seeds["zotero:ITEMDOI1"].stated_identifiers == ("doi:10.5555/zotero.seed.1",)
    assert seeds["zotero:ITEMARX2"].stated_identifiers == ("arxiv:2304.04321",)
    assert seeds["zotero:ITEMTTL3"].matching_text_kind == "title_only" and seeds["zotero:ITEMTTL3"].abstract is None
    assert seeds["zotero:ITEMDOI1"].authors == ("Ada Zoterian",)  # editors are not authors
    stored_text = json.dumps([dict(row) for row in app.connection.execute("SELECT * FROM seed_items")])
    assert "PRIVATE-NOTE" not in stored_text and "Private Collection Name" not in stored_text

    # Seeds are catalogued: the arXiv seed and the same preprint found by a public route are one paper.
    arxiv_outcome = record_collected_paper(
        app.connection,
        CollectedPaperRecord(
            provider="arxiv",
            provider_paper_id="2304.04321v2",
            title="Monitoring Joint Plans",
            authors=(),
            abstract="Public abstract.",
            first_published_date="2023-04-08",
            latest_revision_date=None,
            source_version=None,
            discovery_route="test:arxiv",
        ),
    )
    assert arxiv_outcome.paper_id == seeds["zotero:ITEMARX2"].paper_id


def test_unchanged_library_costs_one_conditional_request(app):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    gateway = make_gateway(app, fake_library.transport)
    synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)
    requests_after_first_sync = len(fake_library.sent_requests)

    outcome = synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)
    assert outcome.result == SeedSyncResult.UNCHANGED and outcome.active_seed_count == 3
    conditional_request = fake_library.sent_requests[requests_after_first_sync]
    assert len(fake_library.sent_requests) == requests_after_first_sync + 1
    assert conditional_request.headers["If-Modified-Since-Version"] == "100"
    assert outbound_rows(app)[-1]["outcome"] == "not_modified"


def test_deleted_and_moved_items_leave_the_seed_set_and_return_when_restored(app):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    gateway = make_gateway(app, fake_library.transport)
    synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)

    # ITEMDOI1 is moved to another collection, ITEMTTL3 deleted, and a new item is added.
    fake_library.items = [
        item for item in standard_collection_items() if item["key"] not in ("ITEMDOI1", "ITEMTTL3")
    ] + [zotero_item("ITEMNEW7", "Freshly Added Zotero seed", abstract="New abstract.", version=101)]
    fake_library.library_version = 101
    outcome = synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)
    assert outcome.changes is not None
    assert (outcome.changes.added, outcome.changes.removed, outcome.active_seed_count) == (1, 2, 2)
    assert {seed.seed_id for seed in load_active_seeds(app.connection, [ZOTERO_SOURCE])} == {
        "zotero:ITEMARX2",
        "zotero:ITEMNEW7",
    }
    removed_rows = app.connection.execute(
        "SELECT seed_item_key FROM paper_seed_memberships WHERE removed_at IS NOT NULL ORDER BY seed_item_key"
    ).fetchall()
    assert [row["seed_item_key"] for row in removed_rows] == ["ITEMDOI1", "ITEMTTL3"]

    fake_library.items = standard_collection_items()  # moved back
    fake_library.library_version = 102
    outcome = synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)
    assert outcome.changes is not None
    assert outcome.changes.added == 2 and outcome.changes.removed == 1 and outcome.active_seed_count == 3


def test_library_changing_during_the_sync_commits_nothing(app):
    fake_library = FakeZoteroLibrary(standard_collection_items())
    gateway = make_gateway(app, fake_library.transport)
    synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)

    fake_library.items = standard_collection_items()[:1]  # a big change is in progress...
    fake_library.library_version = 200
    fake_library.version_bumps_after_each_page = 1  # ...and keeps changing while we page
    fake_library.items = fake_library.items + standard_collection_items()[1:3]
    outcome = synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE)

    assert outcome.result == SeedSyncResult.UNAVAILABLE and "library changed during the sync" in outcome.detail
    assert outcome.active_seed_count == 3  # the previous complete snapshot is still the seed set
    checkpoint_row = app.connection.execute(
        "SELECT cursor FROM source_checkpoints WHERE search_key = ?", (ZOTERO_SOURCE.checkpoint_key,)
    ).fetchone()
    assert checkpoint_row["cursor"] == "100"  # the checkpoint advances only after a complete sync

    fake_library.version_bumps_after_each_page = 0  # editing finished: the next run completes
    assert synchronize_zotero_collection(app.connection, gateway, ZOTERO_SOURCE).result == SeedSyncResult.SYNCED


# --------------------------------------------------------------------------------------------------------------------
# CSL JSON: an equal seed source, offline
# --------------------------------------------------------------------------------------------------------------------


def test_csl_json_import_matches_zotero_semantics(app):
    outcome = import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    assert outcome.result == SeedSyncResult.SYNCED
    assert (outcome.active_seed_count, outcome.title_only_seed_count) == (3, 1)
    assert outcome.skipped_item_counts is not None
    assert dict(outcome.skipped_item_counts) == {"no title": 1}
    file_key = build_csl_json_file_key("seeds/team_safety_seeds.csl.json")
    seeds = {seed.seed_id: seed for seed in load_active_seeds(app.connection, [CSL_SOURCE])}
    assert set(seeds) == {
        f"csl_json:{file_key}#http://zotero.org/users/local/abc/items/CSL00001",
        f"csl_json:{file_key}#csl-seed-2",
        f"csl_json:{file_key}#csl-seed-3",
    }
    assert seeds[f"csl_json:{file_key}#csl-seed-2"].stated_identifiers == ("arxiv:2105.01234",)
    assert seeds[f"csl_json:{file_key}#csl-seed-3"].publication_year == 2019
    stored_text = json.dumps([dict(row) for row in app.connection.execute("SELECT * FROM seed_items")])
    assert "PRIVATE-CSL-NOTE" not in stored_text

    assert import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE).result == SeedSyncResult.UNCHANGED

    seed_file = app.config.home_dir / "seeds" / "team_safety_seeds.csl.json"
    seed_file.unlink()
    missing_outcome = import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    assert missing_outcome.result == SeedSyncResult.UNAVAILABLE and "does not exist" in missing_outcome.detail
    assert missing_outcome.active_seed_count == 3  # the last import stays in use

    trimmed_export = json.loads(CSL_SEED_FIXTURE.read_text())[1:]
    seed_file.write_text(json.dumps(trimmed_export))
    trimmed_outcome = import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    assert trimmed_outcome.changes is not None
    assert (trimmed_outcome.changes.removed, trimmed_outcome.active_seed_count) == (1, 2)


def test_invalid_csl_json_file_is_reported_and_keeps_the_cache(app):
    import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    (app.config.home_dir / "seeds" / "team_safety_seeds.csl.json").write_text('{"not": "an array"}')
    outcome = import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    assert outcome.result == SeedSyncResult.UNAVAILABLE and "JSON array" in outcome.detail
    assert outcome.active_seed_count == 3


# --------------------------------------------------------------------------------------------------------------------
# Seed identifiers stay private unless the operator opts in
# --------------------------------------------------------------------------------------------------------------------


def recording_empty_json_responder(sent_requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """A mock transport handler that records each request and answers an empty JSON object."""

    def respond(request: httpx.Request) -> httpx.Response:
        sent_requests.append(request)
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(b"{}"))

    return respond


def test_looking_up_a_seed_doi_needs_the_explicit_setting(app):
    import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    sent_requests: list[httpx.Request] = []
    gateway = make_gateway(
        app,
        httpx.MockTransport(recording_empty_json_responder(sent_requests)),
    )
    with pytest.raises(SourceAccessDenied, match="known only from the private seed library"):
        gateway.get_work("doi:10.5555/synthetic.seed.0001")
    with pytest.raises(SourceAccessDenied, match="known only from the private seed library"):
        gateway.get_work("arxiv:2105.01234")
    assert sent_requests == [] and outbound_rows(app) == []

    # Once a public provider route also found the work, looking it up discloses nothing new.
    record_collected_paper(
        app.connection,
        CollectedPaperRecord(
            provider="openalex",
            provider_paper_id="W4400000099",
            title="Runtime Verification of Multi-Robot Plans",
            authors=(),
            abstract=None,
            first_published_date="2021",
            latest_revision_date=None,
            source_version=None,
            discovery_route="test:openalex",
            linked_identifiers=(RawIdentifier("doi", "10.5555/synthetic.seed.0001"),),
        ),
    )
    gateway.get_work("doi:10.5555/synthetic.seed.0001")
    assert len(sent_requests) == 1


def test_seed_lookup_is_allowed_when_the_operator_opts_in(tmp_path):
    opted_in_app = build_three_mode_app(
        tmp_path,
        config_replacements={"allow_lookup_by_seed_identifier = false": "allow_lookup_by_seed_identifier = true"},
    )
    try:
        import_csl_json_seed_file(opted_in_app.connection, opted_in_app.config, CSL_SOURCE)
        sent_requests: list[httpx.Request] = []
        gateway = make_gateway(
            opted_in_app,
            httpx.MockTransport(recording_empty_json_responder(sent_requests)),
        )
        gateway.get_work("doi:10.5555/synthetic.seed.0001")
        assert sent_requests[0].url.path == "/works/doi:10.5555/synthetic.seed.0001"
    finally:
        opted_in_app.close()


# --------------------------------------------------------------------------------------------------------------------
# Seeds and the rest of the pipeline
# --------------------------------------------------------------------------------------------------------------------


def test_papers_in_the_seed_library_are_not_queued_for_review(app, tmp_path):
    import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    fixture_path = tmp_path / "public_papers.json"
    fixture_path.write_text(
        json.dumps(
            [
                {
                    "provider": "arxiv",
                    "provider_paper_id": "2105.01234v1",
                    "title": "Plan Monitoring for Robot Teams",
                    "abstract": "Public abstract of a paper the user already holds.",
                },
                {
                    "provider": "arxiv",
                    "provider_paper_id": "2411.09999v1",
                    "title": "A Brand New Paper",
                    "abstract": "Public abstract of a new paper.",
                },
            ]
        )
    )
    run_collect_command(
        app,
        argparse.Namespace(fixture_paths=[fixture_path]),
        build_gateway=lambda: make_gateway(app, FakeZoteroLibrary([]).transport),
    )
    queued_titles = {
        row["title"] for row in app.connection.execute("SELECT papers.title FROM jobs JOIN papers USING (paper_id)")
    }
    assert queued_titles == {"A Brand New Paper"}
    assert len(load_active_seed_library_paper_ids(app.connection, app.config)) == 3


def test_profile_hash_covers_seeds_but_is_unchanged_for_topics_without_seeds(app):
    keyword_topic = app.config.topics[0]
    assert (
        keyword_topic.compute_profile_sha256()
        == TopicDefinition(topic_id="keywords-only", question=KEYWORDS_QUESTION).compute_profile_sha256()
    )

    hashes_before_seeds = compute_topic_profile_hashes(app.connection, app.config)
    import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    hashes_with_seeds = compute_topic_profile_hashes(app.connection, app.config)
    assert hashes_with_seeds["keywords-only"] == hashes_before_seeds["keywords-only"]
    assert hashes_with_seeds["seeds-only"] != hashes_before_seeds["seeds-only"]

    seed_file = app.config.home_dir / "seeds" / "team_safety_seeds.csl.json"
    edited_export = json.loads(seed_file.read_text())
    edited_export[2]["abstract"] = "Now the title-only seed has an abstract."
    seed_file.write_text(json.dumps(edited_export))
    import_csl_json_seed_file(app.connection, app.config, CSL_SOURCE)
    assert compute_topic_profile_hashes(app.connection, app.config)["seeds-only"] != hashes_with_seeds["seeds-only"]


def test_a_collection_shared_by_two_topics_is_synced_once(tmp_path):
    shared_app = build_in_process_app(
        tmp_path / "home",
        topics_toml=f"""
[[topics]]
id = "first"
question = "First private question?"
zotero_collections = ["{ZOTERO_COLLECTION_KEY}"]

[[topics]]
id = "second"
arxiv_categories = ["cs.RO"]
zotero_collections = ["{ZOTERO_COLLECTION_KEY}"]
""",
    )
    try:
        fake_library = FakeZoteroLibrary(standard_collection_items())
        gateway = make_gateway(shared_app, fake_library.transport, page_size=100)
        outcomes = synchronize_all_seed_sources(shared_app.connection, shared_app.config, build_gateway=lambda: gateway)
        assert len(outcomes) == 1 and len(fake_library.sent_requests) == 1
        assert all(
            len(profile.active_seeds) == 3
            for profile in build_research_profiles(shared_app.connection, shared_app.config)
        )
    finally:
        shared_app.close()


# --------------------------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------------------------


def test_zotero_parser_reads_identifiers_from_structured_fields_only():
    page = parse_zotero_items_page(
        json.dumps(
            [
                zotero_item(
                    "REPORT01",
                    "A Technical Report",
                    item_type="report",
                    extra="DOI: 10.1234/Report.Two.\nSome note mentioning 10.9999/not.a.stated.doi",
                ),
                zotero_item("CASENAME", None, item_type="case", caseName="A Court Case"),
                {"key": "bad key", "data": {"itemType": "book", "title": "Malformed key"}},
                "not an object",
            ]
        ).encode()
    )
    assert page.raw_item_count == 4 and page.skipped_item_counts["malformed item"] == 2
    report_seed, case_seed = page.seed_items
    assert report_seed.doi == "10.1234/report.two" and report_seed.publication_year == 2020
    assert case_seed.title == "A Court Case"


def test_csl_parser_handles_date_forms_and_duplicate_ids():
    parsed_export = parse_csl_json_export(
        json.dumps(
            [
                {"id": 7, "title": "Integer id", "issued": {"date-parts": [["2018"]]}},
                {"id": 7, "title": "Duplicate id"},
                {
                    "id": "x",
                    "title": "Literal author",
                    "author": [{"literal": "A Consortium"}],
                    "URL": "https://arxiv.org/abs/2301.00001v3",
                },
            ]
        ).encode(),
        "abcdef1234",
    )
    assert [seed.seed_item_key for seed in parsed_export.seed_items] == ["abcdef1234#7", "abcdef1234#x"]
    assert parsed_export.seed_items[0].publication_year == 2018
    assert parsed_export.seed_items[1].authors == ("A Consortium",)
    assert parsed_export.seed_items[1].arxiv_id == "2301.00001v3"
    assert parsed_export.skipped_item_counts["duplicate id"] == 1


# --------------------------------------------------------------------------------------------------------------------
# status explains what to add (CLI)
# --------------------------------------------------------------------------------------------------------------------


def test_status_explains_what_to_add_when_both_inputs_are_empty(initialized_home):
    status_result = initialized_home.run_cli("status")
    assert status_result.exit_code == 0, status_result.stderr
    assert "Topics: none configured" in status_result.stdout and "csl_json_files" in status_result.stdout

    (initialized_home.home_dir / "topics.toml").write_text(f"""
[[topics]]
id = "library-only"
zotero_collections = ["{ZOTERO_COLLECTION_KEY}"]
""")
    # No credentials file: Zotero is skipped, and nothing is sent.
    collect_result = initialized_home.run_cli("collect", "--skip-searches")
    assert collect_result.exit_code == 0, collect_result.stderr
    assert "NOT synced" in collect_result.stdout
    status_result = initialized_home.run_cli("status")
    assert "library-only [no input]" in status_result.stdout
    assert "its seed sources hold no seeds" in status_result.stdout
    assert "add `arxiv_categories`" in status_result.stdout
    assert "Nothing defines relevance yet" in status_result.stdout
    assert f"Seed source Zotero collection {ZOTERO_COLLECTION_KEY}: 0 active seed(s)" in status_result.stdout
    assert "LAST ATTEMPT FAILED" in status_result.stdout
    assert initialized_home.query("SELECT COUNT(*) AS n FROM outbound_requests")[0]["n"] == 0


def test_status_shows_each_topic_mode_after_collecting_csl_seeds(initialized_home):
    seeds_dir = initialized_home.home_dir / "seeds"
    seeds_dir.mkdir()
    shutil.copy(CSL_SEED_FIXTURE, seeds_dir / "team_safety_seeds.csl.json")
    (initialized_home.home_dir / "topics.toml").write_text(THREE_MODE_TOPICS)
    collect_result = initialized_home.run_cli("collect", "--skip-searches")
    assert collect_result.exit_code == 0, collect_result.stderr
    assert "CSL JSON seeds/team_safety_seeds.csl.json: synced" in collect_result.stdout

    status_stdout = initialized_home.run_cli("status").stdout
    assert "keywords-only [keywords]" in status_stdout
    assert "seeds-only [seeds]: question none" in status_stdout
    assert "keywords-and-seeds [keywords]" in status_stdout  # Zotero not synced (no key): keywords carry it
    assert "1 seed(s) have no abstract and are matched by title only" in status_stdout
    assert "are in your active seed library" in status_stdout
