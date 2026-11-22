"""The collector's fixed operations, the read-only provider rules, and the secure HTTPS hop.

No real network: every test injects httpx.MockTransport and a fake DNS resolver. The checks cover the approved query
catalog, request rebuilding from validated ids only, blocked and allowed redirects (credentials dropped on a host
change), private-address refusal, response bounds, budgets and spacing, provider pauses, and the redacted log.
"""

from __future__ import annotations

import gzip
import json
import os
import stat
from datetime import date

import httpx
import pytest
from conftest import build_in_process_app

from paperwatch.catalog import record_collected_paper
from paperwatch.errors import (
    ConfigurationError,
    ProviderUnavailable,
    SourceAccessDenied,
    SourceRequestBudgetExhausted,
    SourceResponseRejected,
)
from paperwatch.network.collector_credentials import CollectorCredentials, load_collector_credentials
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.network.provider_requests import ProviderRequest, PublicationWindow
from paperwatch.network.provider_rules import load_provider_access_rules, packaged_provider_rules_path
from paperwatch.network.secure_fetcher import HttpsHop, SecureHttpsFetcher
from paperwatch.public_query_catalog import build_public_query_catalog
from paperwatch.sources.types import CollectedPaperRecord, RawIdentifier

PRIVATE_QUESTION = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
TOPICS_WITH_ALL_INPUTS = f"""
[[topics]]
id = "team-safety"
question = "{PRIVATE_QUESTION}"
public_queries = ["multi-robot runtime assurance"]
arxiv_categories = ["cs.RO"]
zotero_collections = ["ABCD2345"]
"""
PUBLIC_ADDRESS = "93.184.216.34"
SECRET_OPENALEX_KEY = "SECRET-OPENALEX-KEY-123"
ATOM_FEED = b"<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"
ARXIV_RECORD = CollectedPaperRecord(
    provider="arxiv",
    provider_paper_id="2411.05555v1",
    title="Checking Joint Robot Plans (synthetic)",
    authors=("C. Synthetic",),
    abstract="SYNTHETIC.",
    first_published_date="2026-11-01",
    latest_revision_date=None,
    source_version=None,
    discovery_route="test:arxiv",
    linked_identifiers=(RawIdentifier("openalex", "W4400000001"),),
)


def streamed_response(status_code: int, *, headers: dict | None = None, content: bytes = b"") -> httpx.Response:
    """A response whose body is streamed like a real network response (iter_raw works exactly once)."""
    return httpx.Response(status_code, headers=headers, stream=httpx.ByteStream(content))


class RecordingTransport(httpx.MockTransport):
    """MockTransport that remembers every request it was asked to send."""

    def __init__(self, respond) -> None:
        self.sent_requests: list[httpx.Request] = []

        def record_and_respond(request: httpx.Request) -> httpx.Response:
            self.sent_requests.append(request)
            return respond(request)

        super().__init__(record_and_respond)


def atom_response(request: httpx.Request) -> httpx.Response:
    return streamed_response(200, headers={"content-type": "application/atom+xml; charset=UTF-8"}, content=ATOM_FEED)


@pytest.fixture
def app(tmp_path):
    test_app = build_in_process_app(tmp_path / "home", topics_toml=TOPICS_WITH_ALL_INPUTS)
    yield test_app
    test_app.close()


NO_COLLECTOR_CREDENTIALS = CollectorCredentials()


def make_gateway(
    app,
    respond=atom_response,
    *,
    credentials=NO_COLLECTOR_CREDENTIALS,
    rules_text=None,
    resolved_addresses=(PUBLIC_ADDRESS,),
):
    transport = RecordingTransport(respond)
    sleeps: list[float] = []
    gateway = ProviderGateway(
        app.connection,
        app.config,
        credentials=credentials,
        http_transport=transport,
        provider_rules=load_provider_access_rules(rules_text) if rules_text else None,
        address_resolver=lambda host, port: list(resolved_addresses),
        sleep_function=sleeps.append,
    )
    return gateway, transport, sleeps


def outbound_rows(app):
    return app.connection.execute("SELECT * FROM outbound_requests ORDER BY started_at, rowid").fetchall()


def rules_with(old_text: str, new_text: str) -> str:
    rules_text = packaged_provider_rules_path().read_text()
    assert old_text in rules_text
    return rules_text.replace(old_text, new_text)


# --------------------------------------------------------------------------------------------------------------------
# search(topic_query_id, cursor): only approved public terms leave the machine
# --------------------------------------------------------------------------------------------------------------------


def test_search_sends_only_the_approved_public_phrase_and_logs_it(app):
    gateway, transport, _ = make_gateway(app)
    keyword_query_id = next(
        query_id
        for query_id in build_public_query_catalog(app.config.topics)
        if query_id.startswith("team-safety:arxiv:keywords:")
    )
    gateway.search(
        keyword_query_id, cursor="100", publication_window=PublicationWindow(date(2026, 11, 1), date(2026, 11, 7))
    )

    sent_url = transport.sent_requests[0].url
    assert (sent_url.scheme, sent_url.host, sent_url.path) == ("https", "export.arxiv.org", "/api/query")
    assert (
        sent_url.params["search_query"]
        == '(all:"multi-robot runtime assurance") AND submittedDate:[202611010000 TO 202611072359]'
    )
    assert sent_url.params["start"] == "100"
    logged_row = outbound_rows(app)[0]
    assert (logged_row["provider_name"], logged_row["operation"], logged_row["outcome"]) == ("arxiv", "search", "ok")
    everything_sent = str(sent_url) + json.dumps([dict(row) for row in outbound_rows(app)])
    assert "runtime monitor reject" not in everything_sent  # the private question never leaves the machine


def test_search_refuses_anything_but_an_approved_catalog_id(app):
    gateway, transport, _ = make_gateway(app)
    for unapproved_query in (
        "team-safety:arxiv:keywords:ffffffffff",
        PRIVATE_QUESTION,
        "team-safety:arxiv:category:cs.AI",
        "https://evil.example/?q=1",
    ):
        with pytest.raises(SourceAccessDenied, match="not an approved public query id"):
            gateway.search(unapproved_query)
    with pytest.raises(SourceAccessDenied, match="cursor"):
        gateway.search("team-safety:arxiv:category:cs.RO", cursor="1; DROP TABLE")
    assert transport.sent_requests == []


def test_query_catalog_has_stable_ids_per_provider_and_no_private_text(app):
    catalog = build_public_query_catalog(app.config.topics)
    id_shapes = sorted(query_id.rsplit(":", 1)[0] for query_id in catalog)
    assert id_shapes == ["team-safety:arxiv:category", "team-safety:arxiv:keywords", "team-safety:openalex:keywords"]
    assert catalog == build_public_query_catalog(app.config.topics)  # deterministic ids across runs
    assert {entry.public_value for entry in catalog.values()} == {"multi-robot runtime assurance", "cs.RO"}


# --------------------------------------------------------------------------------------------------------------------
# get_work(provider_id) and get_text(known_edition_id): ids are rechecked against source records at dispatch
# --------------------------------------------------------------------------------------------------------------------


def test_get_work_only_for_identifiers_known_from_source_records(app):
    gateway, transport, _ = make_gateway(app)
    with pytest.raises(SourceAccessDenied, match="not a known source record"):
        gateway.get_work("arxiv:2411.09999")
    with pytest.raises(SourceAccessDenied, match="invalid provider id"):
        gateway.get_work("arxiv:../../etc/passwd")
    assert transport.sent_requests == []

    record_collected_paper(app.connection, ARXIV_RECORD)
    gateway.get_work("arxiv:2411.05555v1")
    assert transport.sent_requests[0].url.params["id_list"] == "2411.05555v1"


def test_get_text_fetches_the_exact_versioned_edition_into_the_cache(app):
    pdf_bytes = b"%PDF-1.4 synthetic"
    gateway, transport, _ = make_gateway(
        app, lambda request: streamed_response(200, headers={"content-type": "application/pdf"}, content=pdf_bytes)
    )
    edition_id = record_collected_paper(app.connection, ARXIV_RECORD).edition_id

    cached_file = gateway.get_text(edition_id)
    assert str(transport.sent_requests[0].url) == "https://export.arxiv.org/pdf/2411.05555v1"
    assert cached_file.file_path.read_bytes() == pdf_bytes
    assert cached_file.file_path.parent == app.config.source_cache_dir / "sources" / edition_id
    assert stat.S_IMODE(cached_file.file_path.stat().st_mode) == 0o640
    for refused_edition_id in ("../../etc/passwd", "e_ffffffffffffffff"):
        with pytest.raises(SourceAccessDenied):
            gateway.get_text(refused_edition_id)


def test_get_text_refuses_editions_without_a_permitted_text_route(app):
    gateway, transport, _ = make_gateway(app)
    fixture_edition_id = record_collected_paper(
        app.connection,
        CollectedPaperRecord(
            provider="fixture",
            provider_paper_id="demo-1",
            title="Fixture",
            authors=(),
            abstract=None,
            first_published_date=None,
            latest_revision_date=None,
            source_version=None,
            discovery_route="test",
        ),
    ).edition_id
    with pytest.raises(SourceAccessDenied, match="no permitted text route"):
        gateway.get_text(fixture_edition_id)
    assert transport.sent_requests == []


# --------------------------------------------------------------------------------------------------------------------
# Redirects, addresses, and response bounds
# --------------------------------------------------------------------------------------------------------------------


def test_redirect_to_an_unlisted_host_is_blocked_and_the_next_request_still_works(app):
    record_collected_paper(app.connection, ARXIV_RECORD)

    def redirect_then_answer(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("id_list") == "2411.05555":
            return streamed_response(302, headers={"location": "https://evil.example/collect?stolen=1"})
        return atom_response(request)

    gateway, transport, _ = make_gateway(app, redirect_then_answer)
    with pytest.raises(SourceAccessDenied, match="redirect from export.arxiv.org to 'evil.example' refused"):
        gateway.get_work("arxiv:2411.05555")
    assert [request.url.host for request in transport.sent_requests] == ["export.arxiv.org"]
    assert [row["outcome"] for row in outbound_rows(app)] == ["redirect", "blocked"]

    assert gateway.get_work("arxiv:2411.05555v1").body_bytes == ATOM_FEED  # the next valid request completes


def test_allowed_redirect_drops_credentials_on_a_host_change_and_the_log_is_redacted(app):
    rules_text = rules_with(
        "path_pattern = '/works/W[0-9]+\\.pdf'",
        "path_pattern = '/works/W[0-9]+\\.pdf'\nredirect_hosts = [\"cdn.example.org\"]",
    )
    edition_id = record_collected_paper(
        app.connection,
        CollectedPaperRecord(
            provider="openalex",
            provider_paper_id="W4400000002",
            title="An OpenAlex work",
            authors=(),
            abstract="Text.",
            first_published_date=None,
            latest_revision_date=None,
            source_version=None,
            discovery_route="test",
        ),
    ).edition_id

    def redirect_to_cdn(request: httpx.Request) -> httpx.Response:
        if request.url.host == "content.openalex.org":
            return streamed_response(302, headers={"location": "https://cdn.example.org/blob/77?signature=abc"})
        return streamed_response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.4 x")

    credentials = CollectorCredentials(values_by_provider={"openalex": {"api_key": SECRET_OPENALEX_KEY}})
    gateway, transport, _ = make_gateway(app, redirect_to_cdn, credentials=credentials, rules_text=rules_text)
    gateway.get_text(edition_id)

    first_hop, second_hop = transport.sent_requests
    assert first_hop.url.params["api_key"] == SECRET_OPENALEX_KEY
    assert second_hop.url.host == "cdn.example.org"
    assert SECRET_OPENALEX_KEY not in str(second_hop.url) + str(second_hop.headers)
    logged_text = json.dumps([dict(row) for row in outbound_rows(app)])
    assert SECRET_OPENALEX_KEY not in logged_text and "[redacted]" in logged_text


@pytest.mark.parametrize(
    "location",
    [
        "http://export.arxiv.org/api/query?id_list=1",  # downgrade to plain HTTP
        "https://user:pw@export.arxiv.org/api/query",  # embedded credentials
        "https://export.arxiv.org:8443/api/query",  # unusual port
        "https://export.arxiv.org/admin?x=1",
    ],
)  # path outside the rule
def test_unsafe_redirect_targets_are_refused(app, location):
    record_collected_paper(app.connection, ARXIV_RECORD)
    gateway, _, _ = make_gateway(app, lambda request: streamed_response(302, headers={"location": location}))
    with pytest.raises(SourceAccessDenied):
        gateway.get_work("arxiv:2411.05555")


@pytest.mark.parametrize(
    "resolved_address",
    ["10.0.0.5", "127.0.0.1", "169.254.169.254", "::1", "fd00::1", "::ffff:192.168.1.1", "100.64.0.1"],
)
def test_hosts_resolving_to_non_public_addresses_are_refused_before_sending(app, resolved_address):
    gateway, transport, _ = make_gateway(app, resolved_addresses=(PUBLIC_ADDRESS, resolved_address))
    with pytest.raises(SourceAccessDenied, match="non-public address"):
        gateway.search("team-safety:arxiv:category:cs.RO")
    assert transport.sent_requests == []
    assert outbound_rows(app)[-1]["outcome"] == "blocked"


def test_connected_address_is_checked_after_connecting():
    class FakeNetworkStream:
        def get_extra_info(self, name):
            return ("10.1.2.3", 443) if name == "server_addr" else None

    fetcher = SecureHttpsFetcher(user_agent="test", address_resolver=lambda host, port: [PUBLIC_ADDRESS])
    response = httpx.Response(200, extensions={"network_stream": FakeNetworkStream()})
    with pytest.raises(SourceAccessDenied, match="connected to non-public address 10.1.2.3"):
        fetcher._check_connected_address(HttpsHop("arxiv", "export.arxiv.org", "/api/query"), response)


@pytest.mark.parametrize(
    "respond, expected_message",
    [
        (
            lambda request: streamed_response(200, headers={"content-type": "text/html"}, content=b"<html>"),
            "content type 'text/html'",
        ),
        (
            lambda request: streamed_response(
                200, headers={"content-type": "application/atom+xml", "content-length": "999999999"}, content=b"x"
            ),
            "exceeds",
        ),
        (
            lambda request: streamed_response(
                200,
                headers={"content-type": "application/atom+xml", "content-encoding": "gzip"},
                content=gzip.compress(b"\0" * 12_000_000),
            ),
            "decoded response exceeds",
        ),
        (
            lambda request: streamed_response(
                200, headers={"content-type": "application/atom+xml", "content-encoding": "br"}, content=b"x"
            ),
            "unsupported content encoding",
        ),
    ],
)
def test_response_bounds_are_enforced(app, respond, expected_message):
    record_collected_paper(app.connection, ARXIV_RECORD)
    gateway, _, _ = make_gateway(app, respond)
    with pytest.raises(SourceResponseRejected, match=expected_message):
        gateway.get_work("arxiv:2411.05555")


def test_gzip_responses_within_bounds_are_decoded(app):
    record_collected_paper(app.connection, ARXIV_RECORD)
    gateway, _, _ = make_gateway(
        app,
        lambda request: streamed_response(
            200,
            headers={"content-type": "application/atom+xml", "content-encoding": "gzip"},
            content=gzip.compress(ATOM_FEED),
        ),
    )
    assert gateway.get_work("arxiv:2411.05555").body_bytes == ATOM_FEED


# --------------------------------------------------------------------------------------------------------------------
# Budgets, spacing, provider pauses, and request shapes outside the rules
# --------------------------------------------------------------------------------------------------------------------


def test_daily_budget_and_minimum_spacing_are_enforced_across_requests(app):
    gateway, transport, sleeps = make_gateway(
        app, rules_text=rules_with("daily_request_budget = 2000", "daily_request_budget = 2")
    )
    gateway.search("team-safety:arxiv:category:cs.RO")
    gateway.search("team-safety:arxiv:category:cs.RO", cursor="100")
    assert sleeps and 2.0 < sleeps[-1] <= 3.0  # arXiv: at least three seconds between requests
    with pytest.raises(SourceRequestBudgetExhausted):
        gateway.search("team-safety:arxiv:category:cs.RO", cursor="200")
    assert len(transport.sent_requests) == 2
    assert outbound_rows(app)[-1]["outcome"] == "budget_exhausted"


def test_rate_limit_pauses_the_provider_and_honours_retry_after(app):
    gateway, transport, _ = make_gateway(app, lambda request: streamed_response(429, headers={"retry-after": "7200"}))
    with pytest.raises(ProviderUnavailable) as outage:
        gateway.search("team-safety:arxiv:category:cs.RO")
    assert outage.value.retry_after_seconds == 7200 and outage.value.is_service_outage
    with pytest.raises(ProviderUnavailable, match="arxiv is paused until"):
        gateway.search("team-safety:arxiv:category:cs.RO")
    assert len(transport.sent_requests) == 1
    paused_until = app.connection.execute(
        "SELECT paused_until FROM service_state WHERE service = 'provider:arxiv'"
    ).fetchone()["paused_until"]
    assert paused_until > "2026-11-19T01:00:00Z"


def test_requests_outside_the_rules_are_refused_even_if_a_builder_produced_them(app):
    gateway, transport, _ = make_gateway(app)
    for unsafe_request in (
        ProviderRequest("arxiv", "search", "/api/query", (("callback", "x"),)),
        ProviderRequest("arxiv", "search", "/admin"),
        ProviderRequest("arxiv", "upload", "/api/query"),
        ProviderRequest("example", "search", "/"),
    ):
        with pytest.raises(SourceAccessDenied):
            gateway._execute(unsafe_request)
    assert transport.sent_requests == []


def test_zotero_seed_items_need_an_approved_collection_and_the_collector_key(app):
    gateway, transport, _ = make_gateway(app)
    with pytest.raises(SourceAccessDenied, match="not listed in topics.toml"):
        gateway.list_seed_items("ZZZZ9999")
    with pytest.raises(SourceAccessDenied, match="library_id is missing"):
        gateway.list_seed_items("ABCD2345")

    credentials = CollectorCredentials(values_by_provider={"zotero": {"api_key": "ZKEY", "library_id": "42"}})
    gateway, transport, _ = make_gateway(
        app,
        lambda request: streamed_response(200, headers={"content-type": "application/json"}, content=b"[]"),
        credentials=credentials,
    )
    gateway.list_seed_items("ABCD2345", start=25)
    sent_request = transport.sent_requests[0]
    assert sent_request.url.path == "/users/42/collections/ABCD2345/items/top"
    assert sent_request.headers["zotero-api-key"] == "ZKEY" and sent_request.headers["zotero-api-version"] == "3"
    assert "ZKEY" not in json.dumps([dict(row) for row in outbound_rows(app)])


# --------------------------------------------------------------------------------------------------------------------
# Read-only rules and the collector-only credentials file
# --------------------------------------------------------------------------------------------------------------------


def test_packaged_rules_allow_only_https_hosts_by_name():
    provider_rules = load_provider_access_rules()
    assert provider_rules.all_permitted_hosts() == {
        "export.arxiv.org",
        "rss.arxiv.org",
        "api.openalex.org",
        "content.openalex.org",
        "api.zotero.org",
    }
    with pytest.raises(ConfigurationError, match="not an IP address|DNS host name"):
        load_provider_access_rules(rules_with('host = "rss.arxiv.org"', 'host = "127.0.0.1"'))
    with pytest.raises(ConfigurationError, match="unknown key"):
        load_provider_access_rules(rules_with("[defaults]", "[defaults]\nallow_http = true"))


def test_credentials_file_must_be_private_and_never_shows_its_secrets(tmp_path):
    credentials_path = tmp_path / "collector.toml"
    credentials_path.write_text(f'[openalex]\napi_key = "{SECRET_OPENALEX_KEY}"\n')
    os.chmod(credentials_path, 0o644)
    with pytest.raises(ConfigurationError, match="accessible to other accounts"):
        load_collector_credentials(credentials_path)
    os.chmod(credentials_path, 0o600)
    credentials = load_collector_credentials(credentials_path)
    assert credentials.get("openalex", "api_key") == SECRET_OPENALEX_KEY
    assert SECRET_OPENALEX_KEY not in repr(credentials)
    assert load_collector_credentials(tmp_path / "missing.toml").values_by_provider == {}
    credentials_path.write_text('[github]\ntoken = "x"\n')
    with pytest.raises(ConfigurationError, match="unknown provider"):
        load_collector_credentials(credentials_path)
