"""Plan part 5: collect current papers and run a finite historical search.

Check from the plan: replay a feed, a mid-page failure, a week of downtime, a late-indexed work, a revision, and a
historical search that reaches its cap. Check the recovered ids, preserved cursors, and date labels.
Also covered: the daily candidate budget with capacity reserved for current releases, source text from a local file
and from a capped download with sandboxed extraction, withdrawals (excluded, then corrected in the next report), and
that only approved public terms leave the machine.

No real network: arXiv and OpenAlex are tests/fake_paper_providers.py behind httpx.MockTransport, DNS is faked, and
the clock is controlled.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from datetime import UTC, date, datetime, timedelta

import pytest
from conftest import RECORDED_PAPER_FIXTURE, build_in_process_app
from fake_paper_providers import (
    PUBLIC_ADDRESS,
    FakeArxivPaper,
    FakeOpenAlexWork,
    FakePaperProviders,
    FeedItem,
    build_api_page_xml,
    openalex_work_json,
    search_windows_sent,
)

from paperwatch import clock
from paperwatch.catalog import load_paper_row
from paperwatch.collection.collection_context import CollectionContext
from paperwatch.collection.historical_search import (
    historical_queries_for_topic,
    history_plan_key,
    plan_windows,
    recency_schedule_settings,
)
from paperwatch.collection.page_ingestion import ingest_records_with_checkpoints
from paperwatch.collection.recency_schedule import (
    RecencyScheduleSettings,
    build_recency_weighted_windows,
    exponential_boundary_ages,
)
from paperwatch.commands.collect_command import run_collect_command
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.work_command import run_single_work_cycle
from paperwatch.jobs import JobPriorityClass
from paperwatch.network.collector_credentials import CollectorCredentials
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.network.provider_rules import load_provider_access_rules, packaged_provider_rules_path
from paperwatch.public_query_catalog import build_public_query_catalog
from paperwatch.publication_dates import ReleaseClass, classify_release, describe_date_precision
from paperwatch.source_checkpoints import (
    load_existing_source_checkpoint,
    load_source_checkpoint,
    load_source_checkpoints_with_prefix,
)
from paperwatch.sources.arxiv_atom import arxiv_id_submission_month, parse_api_search_page
from paperwatch.sources.openalex_works import parse_works_page
from paperwatch.sources.types import CollectedPaperRecord

PHRASE = "runtime assurance"
PRIVATE_QUESTION = "PRIVATE-QUESTION: Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
TOPICS = f"""
[[topics]]
id = "team-safety"
question = "{PRIVATE_QUESTION}"
public_queries = ["{PHRASE}"]
arxiv_categories = ["cs.RO"]
"""
SEEDS_ONLY_STYLE_TOPICS = """
[[topics]]
id = "category-watch"
arxiv_categories = ["cs.RO"]
"""
START_OF_TEST_WEEK = datetime(2026, 11, 16, 11, 0, tzinfo=UTC)  # a Monday


# --------------------------------------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------------------------------------


class ControlledClock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, **time_delta) -> None:
        self.moment += timedelta(**time_delta)


@pytest.fixture
def test_clock(monkeypatch) -> ControlledClock:
    """Replace utc_now everywhere it was imported, so stored timestamps, windows, pauses, and budgets agree."""
    controlled_clock = ControlledClock(START_OF_TEST_WEEK)
    original_utc_now = clock.utc_now
    for module in list(sys.modules.values()):
        if (
            getattr(module, "__name__", "").startswith("paperwatch")
            and getattr(module, "utc_now", None) is original_utc_now
        ):
            monkeypatch.setattr(module, "utc_now", controlled_clock)
    return controlled_clock


@pytest.fixture
def providers(test_clock) -> FakePaperProviders:
    return FakePaperProviders(current_time=test_clock)


def open_app(tmp_path, topics_toml: str = TOPICS, **config_values):
    replacements = {f"{key} = {default}": f"{key} = {value}" for key, (default, value) in config_values.items()}
    return build_in_process_app(tmp_path / "home", topics_toml=topics_toml, config_replacements=replacements)


@pytest.fixture
def app(tmp_path, test_clock):
    test_app = open_app(tmp_path)
    yield test_app
    test_app.close()


def make_gateway(
    app, providers: FakePaperProviders, page_size: int = 2, credentials: CollectorCredentials | None = None
) -> ProviderGateway:
    rules_text = (
        packaged_provider_rules_path().read_text().replace("search_page_size = 100", f"search_page_size = {page_size}")
    )
    return ProviderGateway(
        app.connection,
        app.config,
        credentials=credentials or CollectorCredentials(),
        http_transport=providers.transport,
        provider_rules=load_provider_access_rules(rules_text),
        address_resolver=lambda host, port: [PUBLIC_ADDRESS],
        sleep_function=lambda seconds: None,
    )


def collect(app, providers: FakePaperProviders, *, page_size: int = 2, credentials=None, **arguments) -> None:
    gateway = make_gateway(app, providers, page_size, credentials)
    namespace = argparse.Namespace(
        fixture_paths=arguments.get("fixture_paths"), skip_searches=arguments.get("skip_searches", False)
    )
    assert run_collect_command(app, namespace, build_gateway=lambda: gateway) == 0


def paper_by_arxiv_id(app, arxiv_id: str):
    alias_row = app.connection.execute(
        "SELECT paper_id FROM paper_aliases WHERE provider = 'arxiv' AND provider_paper_id = ?", (arxiv_id,)
    ).fetchone()
    return load_paper_row(app.connection, alias_row["paper_id"]) if alias_row else None


def discovery_row(app, paper_id: str, route: str):
    return app.connection.execute(
        "SELECT * FROM paper_discoveries WHERE paper_id = ? AND route = ?", (paper_id, route)
    ).fetchone()


def synthetic_paper(arxiv_id: str, submitted_on: date, *, phrase: bool = False, **fields) -> FakeArxivPaper:
    topic_words = f" It studies {PHRASE} for robot teams." if phrase else ""
    return FakeArxivPaper(
        arxiv_id=arxiv_id,
        title=f"Synthetic paper {arxiv_id}",
        abstract=f"SYNTHETIC abstract of {arxiv_id}.{topic_words}",
        submitted_on=submitted_on,
        **fields,
    )


# --------------------------------------------------------------------------------------------------------------------
# Replay a feed; a revision; date labels
# --------------------------------------------------------------------------------------------------------------------


def test_feed_replay_labels_new_cross_and_revised_items_and_an_unchanged_feed_costs_one_304(app, providers):
    brand_new = synthetic_paper("2611.00001", date(2026, 11, 13))
    cross_listed = synthetic_paper("2611.00002", date(2026, 11, 13), categories=("cs.SE", "cs.RO"))
    revised_old_paper = synthetic_paper("1806.01234", date(2018, 6, 15), version_number=3)
    providers.feeds["cs.RO"] = (
        date(2026, 11, 16),
        [FeedItem(brand_new, "new"), FeedItem(cross_listed, "cross"), FeedItem(revised_old_paper, "replace")],
    )
    collect(app, providers)

    feed_route = "arxiv-feed:cs.RO"
    new_row, cross_row, revised_row = (
        paper_by_arxiv_id(app, paper.arxiv_id) for paper in (brand_new, cross_listed, revised_old_paper)
    )
    assert [
        discovery_row(app, row["paper_id"], feed_route)["announcement_kind"]
        for row in (new_row, cross_row, revised_row)
    ] == ["new", "cross_list", "revision"]
    # Date labels: precision is the date's form. A 2018 paper revised today is older literature, not a new release.
    assert (new_row["first_published_date"], describe_date_precision(new_row["first_published_date"])) == (
        "2026-11-16",
        "day",
    )
    assert (revised_row["first_published_date"], describe_date_precision(revised_row["first_published_date"])) == (
        "2018-06",
        "month",
    )
    assert revised_row["latest_revision_date"] == "2026-11-16" and revised_row["source_version"] == "v3"
    assert revised_row["discovered_at"].startswith("2026-11-16")
    priority_by_paper = {
        row["paper_id"]: row["priority"]
        for row in app.connection.execute("SELECT paper_id, priority FROM jobs WHERE kind = 'review_paper'")
    }
    assert priority_by_paper[new_row["paper_id"]] == JobPriorityClass.NEW_RELEASE
    assert priority_by_paper[revised_row["paper_id"]] == JobPriorityClass.HISTORICAL_SEARCH

    feed_requests_before = len(providers.requests_to("rss.arxiv.org"))
    advance_shared_clock(providers, hours=6)
    collect(app, providers)  # the feed has not changed: one conditional request, answered 304, nothing re-stored
    assert len(providers.requests_to("rss.arxiv.org")) == feed_requests_before + 1
    assert providers.requests_to("rss.arxiv.org")[-1].headers.get("If-None-Match")
    last_feed_outcome = app.connection.execute(
        "SELECT outcome FROM outbound_requests WHERE operation = 'feed' ORDER BY started_at DESC, rowid DESC LIMIT 1"
    ).fetchone()["outcome"]
    assert last_feed_outcome == "not_modified"
    assert app.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3


def test_a_revision_of_a_known_paper_adds_an_edition_and_keeps_the_first_publication(app, providers):
    first_version = synthetic_paper("2611.00001", date(2026, 11, 13))
    providers.feeds["cs.RO"] = (date(2026, 11, 16), [FeedItem(first_version, "new")])
    collect(app, providers)
    review_jobs_before = app.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    advance_shared_clock(providers, days=1)
    second_version = synthetic_paper("2611.00001", date(2026, 11, 13), version_number=2)  # same abstract text
    providers.feeds["cs.RO"] = (date(2026, 11, 17), [FeedItem(second_version, "replace")])
    collect(app, providers)

    paper_row = paper_by_arxiv_id(app, "2611.00001")
    editions = app.connection.execute(
        "SELECT edition_key, announcement_kind FROM paper_editions WHERE paper_id = ? "
        "ORDER BY first_seen_at, edition_key",
        (paper_row["paper_id"],),
    ).fetchall()
    assert [(row["edition_key"], row["announcement_kind"]) for row in editions] == [
        ("arxiv:2611.00001@v1", "new"),
        ("arxiv:2611.00001@v2", "revision"),
    ]
    assert paper_row["first_published_date"] == "2026-11-16"  # first publication is never moved by a revision
    assert paper_row["latest_revision_date"] == "2026-11-17" and paper_row["source_version"] == "v2"
    assert app.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    # The revision's text is identical, so no second reading is queued (plan part 2).
    assert app.connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == review_jobs_before


def refuse_provider_gateway() -> ProviderGateway:
    """For a context that must never reach a provider."""
    raise AssertionError("this test must not reach the paper providers")


def advance_shared_clock(providers: FakePaperProviders, **time_delta) -> None:
    """Move the shared test clock (the providers hold the same clock object)."""
    shared_clock = providers.current_time
    assert isinstance(shared_clock, ControlledClock)
    shared_clock.advance(**time_delta)


# --------------------------------------------------------------------------------------------------------------------
# A week of downtime; a mid-page failure
# --------------------------------------------------------------------------------------------------------------------


def test_a_week_of_downtime_is_recovered_by_a_dated_api_search(app, providers, test_clock):
    test_clock.moment = datetime(2026, 11, 9, 11, 0, tzinfo=UTC)
    before_downtime = synthetic_paper("2611.00100", date(2026, 11, 6))
    providers.arxiv_papers.append(before_downtime)
    providers.feeds["cs.RO"] = (date(2026, 11, 9), [FeedItem(before_downtime, "new")])
    collect(app, providers)
    assert (
        load_existing_source_checkpoint(app.connection, "arxiv-feed:cs.RO").last_completed_interval_end
        == "2026-11-09T11:00:00Z"
    )

    missed_papers = [synthetic_paper(f"2611.0020{day}", date(2026, 11, 9 + day)) for day in range(5)]
    latest_paper = synthetic_paper("2611.00300", date(2026, 11, 14))
    providers.arxiv_papers += missed_papers + [latest_paper]
    providers.feeds["cs.RO"] = (date(2026, 11, 16), [FeedItem(latest_paper, "new")])  # only the latest announcement
    test_clock.moment = datetime(2026, 11, 16, 11, 0, tzinfo=UTC)  # a week later
    collect(app, providers)

    [recovery_checkpoint] = load_source_checkpoints_with_prefix(app.connection, "arxiv-recovery:")
    assert recovery_checkpoint.search_key == "arxiv-recovery:cs.RO:2026-11-05..2026-11-16"
    assert recovery_checkpoint.coverage_status == "exhausted"
    for missed_paper in missed_papers:
        paper_row = paper_by_arxiv_id(app, missed_paper.arxiv_id)
        assert paper_row is not None, f"{missed_paper.arxiv_id} was not recovered"
        assert paper_row["origin_route"] == "arxiv-recovery:cs.RO"
        assert paper_row["first_published_date"] == missed_paper.submitted_on.isoformat()  # API: submission day
    recovery_requests = [
        request
        for request in providers.requests_to("export.arxiv.org", "/api/query")
        if request.url.params["search_query"].startswith("(cat:cs.RO)")
    ]
    assert search_windows_sent(recovery_requests)[0] == "20261105..20261116"


@pytest.fixture
def app_with_more_history_pages(tmp_path, test_clock):
    test_app = open_app(tmp_path, historical_pages_per_topic_per_run=("6", "30"))
    yield test_app
    test_app.close()


def test_a_mid_page_failure_keeps_the_cursor_and_the_next_run_resumes_at_that_page(
    app_with_more_history_pages, providers, test_clock
):
    app = app_with_more_history_pages
    providers.arxiv_papers += [
        synthetic_paper(f"2611.{index:05d}", date(2026, 11, 1) + timedelta(days=index), phrase=True)
        for index in range(1, 7)
    ]
    providers.failures_remaining[("export.arxiv.org", "2")] = 1  # the second page fails once with HTTP 503
    collect(app, providers)

    band_key = _first_arxiv_band_key(app)
    blocked_checkpoint = load_existing_source_checkpoint(app.connection, band_key)
    assert blocked_checkpoint.coverage_status == "blocked" and blocked_checkpoint.cursor == "2"
    assert "503" in (blocked_checkpoint.stop_reason or "")
    assert len(_arxiv_ids_from_route(app, "history:")) == 2  # page one was stored with its cursor

    test_clock.advance(hours=1)  # past the provider's outage pause
    collect(app, providers)
    starts_for_band = [
        request.url.params["start"]
        for request in providers.requests_to("export.arxiv.org", "/api/query")
        if "runtime assurance" in request.url.params["search_query"]
        and f"[{_newest_window(app).first_date:%Y%m%d}" in request.url.params["search_query"]
    ]
    assert starts_for_band[:3] == ["0", "2", "2"]  # the failed page was retried, not skipped and not restarted
    assert _arxiv_ids_from_route(app, "history:") == {f"2611.{index:05d}" for index in range(1, 7)}
    assert load_existing_source_checkpoint(app.connection, band_key).coverage_status == "exhausted"


def _first_arxiv_band_key(app) -> str:
    query_catalog = build_public_query_catalog(app.config.topics)
    arxiv_query_id = next(
        query_id
        for query_id, public_query in query_catalog.items()
        if public_query.provider_name == "arxiv" and public_query.query_kind == "keywords"
    )
    return f"history:{arxiv_query_id}:{_newest_window(app).label}"


def _topic_plan(app, topic_id: str = "team-safety"):
    return load_source_checkpoint(
        app.connection,
        history_plan_key(
            topic_id,
            historical_queries_for_topic(topic_id, build_public_query_catalog(app.config.topics)),
            recency_schedule_settings(app.config.collection),
        ),
    )


def _newest_window(app):
    return plan_windows(_topic_plan(app))[0]


def _arxiv_ids_from_route(app, route_prefix: str) -> set[str]:
    return {
        row["provider_paper_id"]
        for row in app.connection.execute(
            """SELECT alias.provider_paper_id FROM paper_aliases AS alias JOIN papers ON papers.paper_id = alias.paper_id
           WHERE alias.provider = 'arxiv' AND papers.origin_route LIKE ? || '%'""",
            (route_prefix,),
        )
    }


# --------------------------------------------------------------------------------------------------------------------
# A late-indexed work
# --------------------------------------------------------------------------------------------------------------------


def test_a_late_indexed_openalex_work_is_found_by_the_overlapping_recent_window(app, providers, test_clock):
    late_work = FakeOpenAlexWork(
        "W4400000777",
        f"Synthetic journal study of {PHRASE}",
        "SYNTHETIC abstract text.",
        publication_date=date(2026, 11, 10),
        visible_from=datetime(2026, 11, 17, 0, 0, tzinfo=UTC),
        doi="10.5555/late.1",
    )
    providers.openalex_works.append(late_work)
    collect(app, providers)  # 11-16: published 11-10 but not indexed yet
    assert _openalex_paper(app, "W4400000777") is None

    test_clock.advance(hours=6)  # inside the 12-hour sweep interval: no new OpenAlex window
    openalex_requests_before = len(providers.requests_to("api.openalex.org"))
    collect(app, providers)
    recent_requests = [
        request
        for request in providers.requests_to("api.openalex.org")[openalex_requests_before:]
        if "to_publication_date:2026-11-16" in request.url.params.get("filter", "")
    ]
    assert recent_requests == []

    test_clock.advance(days=1)  # 11-17 17:00: the window re-checks back to 14 days before the last sweep
    collect(app, providers)
    paper_row = _openalex_paper(app, "W4400000777")
    assert paper_row is not None and paper_row["origin_route"].startswith("openalex-recent:team-safety:openalex:")
    assert paper_row["first_published_date"] == "2026-11-10"
    assert (
        classify_release(paper_row["first_published_date"], datetime(2026, 11, 17, tzinfo=UTC))
        == ReleaseClass.NEW_RELEASE
    )
    [recent_checkpoint] = load_source_checkpoints_with_prefix(app.connection, "openalex-recent:")
    assert recent_checkpoint.query_parameters["window_first_date"] == "2026-11-02"  # 11-16 minus 14 days


def _openalex_paper(app, work_id: str):
    alias_row = app.connection.execute(
        "SELECT paper_id FROM paper_aliases WHERE provider = 'openalex' AND provider_paper_id = ?", (work_id,)
    ).fetchone()
    return load_paper_row(app.connection, alias_row["paper_id"]) if alias_row else None


# --------------------------------------------------------------------------------------------------------------------
# A historical search that reaches its cap; the least-served window goes next
# --------------------------------------------------------------------------------------------------------------------


def test_a_historical_search_reaches_its_cap_and_keeps_its_cursors(tmp_path, providers):
    app = open_app(
        tmp_path, historical_candidates_per_topic=("300", "5"), historical_pages_per_topic_per_run=("6", "20")
    )
    try:
        for band_start_year in (2025, 2020, 2015):
            providers.arxiv_papers += [
                synthetic_paper(
                    f"{band_start_year % 100:02d}0{month}.{index:05d}", date(band_start_year, month, 1), phrase=True
                )
                for month in (2, 3)
                for index in (1, 2)
            ]
        collect(app, providers)

        plan_checkpoint = _topic_plan(app)
        assert plan_checkpoint.coverage_status == "capped" and plan_checkpoint.candidates_admitted == 5
        assert len(_arxiv_ids_from_route(app, "history:")) == 5  # exactly the cap, even mid-page
        # Least-served first: the newest window is asked first, and every window is visited once before any window
        # is asked again (a page cut short by its fair quota is re-fetched later for its remaining new papers).
        arxiv_windows = search_windows_sent(providers.requests_to("export.arxiv.org", "/api/query"))
        assert arxiv_windows[0].startswith(f"{_newest_window(app).first_date:%Y%m%d}")
        assert len(set(arxiv_windows[:8])) == 8
        unfinished_bands = [
            checkpoint
            for checkpoint in load_source_checkpoints_with_prefix(app.connection, "history:")
            if checkpoint.coverage_status != "exhausted"
        ]
        assert any(checkpoint.cursor for checkpoint in unfinished_bands)  # overflow cursors are persisted

        requests_before = len(providers.sent_requests)
        collect(app, providers)  # the plan is capped: no historical page is fetched again
        later_history_requests = [
            request
            for request in providers.sent_requests[requests_before:]
            if "runtime assurance" in str(request.url) or "runtime+assurance" in str(request.url)
        ]
        assert all(
            "to_publication_date:2026-11-16" in request.url.params.get("filter", "")
            for request in later_history_requests
        )  # only the recent OpenAlex window, if any
    finally:
        app.close()


def test_a_seeds_only_topic_uses_category_and_year_queries_for_history(tmp_path, providers):
    app = open_app(tmp_path, topics_toml=SEEDS_ONLY_STYLE_TOPICS)
    try:
        providers.arxiv_papers.append(synthetic_paper("1703.00001", date(2017, 3, 1)))
        collect(app, providers)
        history_queries = [
            request.url.params["search_query"] for request in providers.requests_to("export.arxiv.org", "/api/query")
        ]
        assert history_queries and all(query.startswith("(cat:cs.RO) AND submittedDate:[") for query in history_queries)
        assert paper_by_arxiv_id(app, "1703.00001")["origin_route"].startswith("history:category-watch:arxiv:category")
        assert providers.requests_to("api.openalex.org") == []  # a category is not an OpenAlex search term
    finally:
        app.close()


# --------------------------------------------------------------------------------------------------------------------
# Daily budget: current releases first
# --------------------------------------------------------------------------------------------------------------------


def test_the_daily_budget_reserves_capacity_for_current_releases_and_records_the_overflow(tmp_path, providers):
    app = open_app(
        tmp_path,
        daily_new_candidate_budget=("1000", "4"),
        current_release_reserved_candidates=("200", "3"),
        category_feed_candidate_limit=("400", "3"),
        historical_reserved_candidates=("100", "0"),
    )
    try:
        feed_papers = [synthetic_paper(f"2611.0100{index}", date(2026, 11, 13)) for index in range(2)]
        providers.feeds["cs.RO"] = (date(2026, 11, 16), [FeedItem(paper, "new") for paper in feed_papers])
        providers.arxiv_papers += [
            synthetic_paper(f"2503.0000{index}", date(2025, 3, index + 1), phrase=True) for index in range(1, 5)
        ]
        collect(app, providers)
        assert len(_arxiv_ids_from_route(app, "arxiv-feed:")) == 2
        # 4 per day, 3 reserved for current releases, 2 used by them: history may add 4 - 2 - (3 - 2) = 1.
        assert len(_arxiv_ids_from_route(app, "history:")) == 1

        bigger_feed = [synthetic_paper(f"2611.0200{index}", date(2026, 11, 14)) for index in range(3)]
        providers.feeds["cs.RO"] = (date(2026, 11, 16), [FeedItem(paper, "new") for paper in bigger_feed])
        collect(app, providers)  # the day's last slot goes to the feed; the rest is left to a recovery search
        assert len(_arxiv_ids_from_route(app, "arxiv-feed:")) == 3
        feed_checkpoint = load_existing_source_checkpoint(app.connection, "arxiv-feed:cs.RO")
        assert feed_checkpoint.coverage_status == "capped" and feed_checkpoint.cursor is None
        [recovery_checkpoint] = load_source_checkpoints_with_prefix(app.connection, "arxiv-recovery:")
        assert recovery_checkpoint.coverage_status in ("active", "capped")
    finally:
        app.close()


def test_category_feeds_share_their_limit_fairly_and_a_share_cut_schedules_no_recovery(tmp_path, providers):
    """Owner decision 2026-11-21: a broad feed (cs.AI) can no longer take the whole budget; within its share it
    admits the papers that share the most words with the topic, and the rest is a chosen limit, not a gap."""
    two_category_topics = TOPICS.replace('arxiv_categories = ["cs.RO"]', 'arxiv_categories = ["cs.AI", "cs.RO"]')
    app = open_app(tmp_path, topics_toml=two_category_topics, category_feed_candidate_limit=("400", "4"))
    try:
        broad_feed = [
            synthetic_paper(f"2611.0400{index}", date(2026, 11, 13), phrase=index in (4, 5)) for index in range(6)
        ]
        narrow_feed = [synthetic_paper("2611.04100", date(2026, 11, 13))]
        providers.feeds["cs.AI"] = (date(2026, 11, 16), [FeedItem(paper, "new") for paper in broad_feed])
        providers.feeds["cs.RO"] = (date(2026, 11, 16), [FeedItem(paper, "new") for paper in narrow_feed])
        collect(app, providers)

        # water_fill(4, {cs.AI: 6, cs.RO: 1}): cs.RO needs only 1, so cs.AI gets the other 3.
        admitted_broad = _arxiv_ids_from_route(app, "arxiv-feed:cs.AI")
        assert len(admitted_broad) == 3 and {"2611.04004", "2611.04005"} <= set(admitted_broad)
        assert _arxiv_ids_from_route(app, "arxiv-feed:cs.RO") == {"2611.04100"}
        assert load_source_checkpoints_with_prefix(app.connection, "arxiv-recovery:") == []
        broad_checkpoint = load_existing_source_checkpoint(app.connection, "arxiv-feed:cs.AI")
        assert broad_checkpoint.coverage_status == "exhausted" and broad_checkpoint.cursor  # ETag kept
        assert "fair share of the category feeds reached: 3 new record(s)" in (broad_checkpoint.stop_reason or "")
    finally:
        app.close()


# --------------------------------------------------------------------------------------------------------------------
# Source text
# --------------------------------------------------------------------------------------------------------------------


def test_source_text_uses_a_local_file_first_then_a_capped_download_then_extracts(tmp_path, providers):
    stub_directory = tmp_path / "stub"
    stub_directory.mkdir()
    stub_runtime = stub_directory / "fake-podman"
    stub_runtime.write_text("#!/bin/sh\nprintf 'extracted page one\\fpage two'\n")
    stub_runtime.chmod(0o755)
    app = open_app(
        tmp_path,
        daily_source_text_budget=("30", "1"),
        container_runtime=('"podman"', f'"{stub_runtime}"'),
        image=('""', f'"{"ab" * 32}"'),
    )
    try:
        local_paper, downloaded_paper, waiting_paper = (
            synthetic_paper(f"2611.0300{index}", date(2026, 11, 13)) for index in range(3)
        )
        providers.feeds["cs.RO"] = (
            date(2026, 11, 16),
            [FeedItem(paper, "new") for paper in (local_paper, downloaded_paper, waiting_paper)],
        )
        (app.config.local_sources_dir / "2611.03000v1.pdf").write_bytes(b"%PDF-1.4 local synthetic copy")
        collect(app, providers)

        text_rows = {
            row["text_route"] or row["acquisition_status"]: row
            for row in app.connection.execute("SELECT * FROM edition_source_texts")
        }
        assert set(text_rows) == {"local_file", "arxiv_pdf"}
        assert text_rows["local_file"]["downloaded_from_network"] == 0
        assert text_rows["arxiv_pdf"]["downloaded_from_network"] == 1
        pdf_requests = providers.requests_to("export.arxiv.org", "/pdf/")
        assert len(pdf_requests) == 1 and pdf_requests[0].url.path in ("/pdf/2611.03001v1", "/pdf/2611.03002v1")
        for text_row in text_rows.values():
            assert text_row["extraction_status"] == "extracted"
            text_path = (
                app.config.source_cache_dir / "sources" / text_row["edition_id"] / text_row["extracted_text_file_name"]
            )
            assert text_path.read_text() == "extracted page one\fpage two"
    finally:
        app.close()


def test_source_text_without_an_extractor_image_waits_and_skip_searches_downloads_nothing(app, providers):
    providers.feeds["cs.RO"] = (
        date(2026, 11, 16),
        [FeedItem(synthetic_paper("2611.04000", date(2026, 11, 13)), "new")],
    )
    collect(app, providers)
    [text_row] = app.connection.execute("SELECT * FROM edition_source_texts").fetchall()
    assert text_row["acquisition_status"] == "acquired" and text_row["extraction_status"] == "not_configured"

    requests_before = len(providers.sent_requests)
    collect(app, providers, skip_searches=True)
    assert len(providers.sent_requests) == requests_before


# --------------------------------------------------------------------------------------------------------------------
# Withdrawals and retractions
# --------------------------------------------------------------------------------------------------------------------


def test_parsers_report_arxiv_withdrawals_and_openalex_retractions():
    withdrawn_paper = synthetic_paper(
        "2611.05000",
        date(2026, 11, 2),
        version_number=2,
        comment="This paper has been withdrawn by the authors due to an error.",
    )
    parsed_page = parse_api_search_page(build_api_page_xml([withdrawn_paper], 1, 0), "test:arxiv", start_offset=0)
    assert parsed_page.records[0].withdrawal_notice == "withdrawn" and parsed_page.next_cursor is None
    import json

    retracted_work = FakeOpenAlexWork(
        "W4400000888",
        "Synthetic retracted study",
        "SYNTHETIC.",
        date(2024, 5, 1),
        is_retracted=True,
        arxiv_id="2405.00001",
    )
    parsed_works = parse_works_page(
        json.dumps({"meta": {"next_cursor": None}, "results": [openalex_work_json(retracted_work)]}).encode(),
        "test:openalex",
    )
    record = parsed_works.records[0]
    assert record.withdrawal_notice == "retracted"
    assert {(link.scheme, link.value) for link in record.linked_identifiers} >= {
        ("arxiv", "https://arxiv.org/abs/2405.00001")
    }
    assert (
        arxiv_id_submission_month("hep-th/9901001") == "1999-01" and arxiv_id_submission_month("0704.0001") == "2007-04"
    )


def test_a_withdrawn_paper_is_never_recommended_and_a_reported_one_gets_a_correction(tmp_path, test_clock):
    # The fake model's ratings stay below the default floor of 65 (plan part 9), so this test reports at floor 0.
    app = open_app(tmp_path, minimum_reading_priority=(65, 0))
    collect_fixture_and_review(app)
    test_clock.advance(hours=2)  # past the 07:00 New York cutoff of 2026-11-16
    assert run_report_command(app, argparse.Namespace(report_date="2026-11-16")) == 0
    first_report_path = app.config.daily_reports_dir / "2026-11-16.md"
    first_report_bytes = first_report_path.read_bytes()
    assert b"### 1." in first_report_bytes

    fixture_paper_row = app.connection.execute("SELECT * FROM papers").fetchone()
    ingest_records_with_checkpoints(
        app.connection,
        [
            CollectedPaperRecord(
                provider="fixture",
                provider_paper_id="synthetic-0001",
                title=fixture_paper_row["title"],
                authors=(),
                abstract=None,
                first_published_date=None,
                latest_revision_date=None,
                source_version="v1",
                discovery_route="test:provider-notice",
                withdrawal_notice="withdrawn",
                withdrawal_evidence="This paper has been withdrawn",
            )
        ],
        admission_limit=None,
        active_seed_paper_ids=frozenset(),
        max_attempts_per_job=3,
    )

    test_clock.advance(days=1)
    assert run_report_command(app, argparse.Namespace(report_date="2026-11-17")) == 0
    second_report = (app.config.daily_reports_dir / "2026-11-17.md").read_text()
    assert "## Corrections" in second_report and "recommended on 2026-11-16" in second_report
    assert "withdrawn" in second_report
    assert first_report_path.read_bytes() == first_report_bytes  # the historical file is never altered

    test_clock.advance(days=1)
    assert run_report_command(app, argparse.Namespace(report_date="2026-11-18")) == 0
    assert "## Corrections" not in (app.config.daily_reports_dir / "2026-11-18.md").read_text()  # only once
    app.close()


def test_a_paper_first_seen_with_a_withdrawal_notice_is_not_queued(app):
    page_result = ingest_records_with_checkpoints(
        app.connection,
        [
            CollectedPaperRecord(
                provider="arxiv",
                provider_paper_id="2611.06000v2",
                title="Synthetic withdrawn paper",
                authors=(),
                abstract="SYNTHETIC.",
                first_published_date="2026-11-01",
                latest_revision_date="2026-11-15",
                source_version=None,
                discovery_route="arxiv-feed:cs.RO",
                announcement_kind="revision",
                withdrawal_notice="withdrawn",
            )
        ],
        admission_limit=None,
        active_seed_paper_ids=frozenset(),
        max_attempts_per_job=3,
    )
    assert page_result.new_papers_admitted == 1 and page_result.review_jobs_queued == 0
    assert app.connection.execute("SELECT notice_kind FROM paper_withdrawals").fetchone()["notice_kind"] == "withdrawn"


def collect_fixture_and_review(app) -> None:
    assert (
        run_collect_command(
            app,
            argparse.Namespace(fixture_paths=[RECORDED_PAPER_FIXTURE], skip_searches=False),
            build_gateway=lambda: pytest.fail("a fixture replay must not build a gateway"),
        )
        == 0
    )
    run_single_work_cycle(app, probe_paused_model_now=True)
    assert app.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1


# --------------------------------------------------------------------------------------------------------------------
# What leaves the machine; restart safety of the whole run
# --------------------------------------------------------------------------------------------------------------------


def test_only_approved_public_terms_leave_the_machine_on_every_part5_route(app, providers):
    providers.feeds["cs.RO"] = (
        date(2026, 11, 16),
        [FeedItem(synthetic_paper("2611.07000", date(2026, 11, 13)), "new")],
    )
    collect(app, providers)
    assert {request.url.host for request in providers.sent_requests} >= {
        "rss.arxiv.org",
        "export.arxiv.org",
        "api.openalex.org",
    }
    for request in providers.sent_requests:
        sent_text = str(request.url) + " ".join(f"{name}: {value}" for name, value in request.headers.items())
        assert "PRIVATE-QUESTION" not in sent_text and "exit" not in request.url.params.get("search", "")
        free_text = (
            request.url.params.get("search_query")
            or request.url.params.get("search")
            or request.url.params.get("filter", "").partition("title_and_abstract.search:")[2]
            or None
        )
        if free_text is not None:
            assert PHRASE in free_text or "cat:cs.RO" in free_text, free_text
    logged_queries = " ".join(
        row["query_json"] for row in app.connection.execute("SELECT query_json FROM outbound_requests")
    )
    assert "PRIVATE-QUESTION" not in logged_queries


def test_collect_twice_in_a_row_admits_nothing_new_and_status_shows_coverage(app, providers, capsys):
    from paperwatch.commands.status_command import run_status_command

    providers.feeds["cs.RO"] = (
        date(2026, 11, 16),
        [FeedItem(synthetic_paper("2611.08000", date(2026, 11, 13)), "new")],
    )
    providers.arxiv_papers.append(synthetic_paper("2502.08001", date(2025, 2, 1), phrase=True))
    collect(app, providers)
    papers_after_first_run = app.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    collect(app, providers)
    assert app.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == papers_after_first_run
    capsys.readouterr()
    assert run_status_command(app, argparse.Namespace()) == 0
    status_output = capsys.readouterr().out
    assert "arXiv feed cs.RO: last complete poll 2026-11-16T11:00:00Z" in status_output
    assert "Historical search team-safety:" in status_output and "window searches:" in status_output
    assert 'OpenAlex recent "runtime assurance" [team-safety]' in status_output
    assert "Source text:" in status_output and "Collection: 2 of 1000 new candidate(s)" in status_output


def test_collection_context_counts_todays_candidates_from_the_catalog(app, providers):
    providers.feeds["cs.RO"] = (
        date(2026, 11, 16),
        [FeedItem(synthetic_paper("2611.09000", date(2026, 11, 13)), "new")],
    )
    collect(app, providers, fixture_paths=None)
    context = CollectionContext.start(app.connection, app.config, build_gateway=refuse_provider_gateway)
    assert context.candidate_budget.admitted_today_by_current_releases == 1


def test_without_an_openalex_key_only_first_pages_are_requested_and_nothing_is_paused(app, providers, test_clock):
    providers.openalex_works += [
        FakeOpenAlexWork(
            f"W44000010{index}",
            f"Synthetic {PHRASE} study {index}",
            "SYNTHETIC.",
            publication_date=date(2026, 11, 10 + index),
        )
        for index in range(3)
    ]  # page size 2: a second page exists
    collect(app, providers)
    collect(app, providers)
    openalex_cursors = [request.url.params["cursor"] for request in providers.requests_to("api.openalex.org")]
    assert openalex_cursors and set(openalex_cursors) == {"*"}  # OpenAlex answers 401 to cursor pages without a key
    [recent_checkpoint] = load_source_checkpoints_with_prefix(app.connection, "openalex-recent:")
    assert recent_checkpoint.coverage_status == "capped" and (recent_checkpoint.stop_reason or "").startswith(
        "first page only"
    )
    assert recent_checkpoint.last_completed_interval_end is not None  # the window counts as swept
    assert (
        app.connection.execute(
            "SELECT COUNT(*) FROM service_state WHERE service = 'provider:openalex' AND paused_until IS NOT NULL"
        ).fetchone()[0]
        == 0
    )

    keyed_gateway = ProviderGateway(
        app.connection,
        app.config,
        credentials=CollectorCredentials(values_by_provider={"openalex": {"api_key": "SYNTHETIC-OPENALEX-KEY"}}),
        http_transport=providers.transport,
        provider_rules=load_provider_access_rules(
            packaged_provider_rules_path().read_text().replace("search_page_size = 100", "search_page_size = 2")
        ),
        address_resolver=lambda host, port: [PUBLIC_ADDRESS],
        sleep_function=lambda seconds: None,
    )
    test_clock.advance(days=1)
    assert (
        run_collect_command(
            app, argparse.Namespace(fixture_paths=None, skip_searches=False), build_gateway=lambda: keyed_gateway
        )
        == 0
    )
    assert any(request.url.params["cursor"] == "page2" for request in providers.requests_to("api.openalex.org"))
    assert {_openalex_paper(app, f"W44000010{index}") is not None for index in range(3)} == {True}


# --------------------------------------------------------------------------------------------------------------------
# Recency-weighted historical windows (exponential toward the present, floor 1940)
# --------------------------------------------------------------------------------------------------------------------

OPENALEX_TEST_KEY = CollectorCredentials(values_by_provider={"openalex": {"api_key": "SYNTHETIC-OPENALEX-KEY"}})


def test_recency_windows_are_contiguous_equal_mass_slices_of_an_exponential():
    plan_date, settings = date(2026, 11, 16), RecencyScheduleSettings()
    windows = build_recency_weighted_windows(plan_date, settings)
    assert len(windows) == 8 and windows[0].last_date == plan_date and windows[-1].first_date == date(1940, 1, 1)
    for newer, older in itertools.pairwise(windows):
        assert older.last_date + timedelta(days=1) == newer.first_date  # no gap, no overlap
    widths = [(window.last_date - window.first_date).days + 1 for window in windows]
    assert widths == sorted(widths) and widths[-1] > 50 * widths[0]  # narrow recent windows, one wide old one
    assert sum(window.weight for window in windows) == pytest.approx(1.0)

    # Each window holds 1/8 of the area under 2^(-age/6) between the plan date and 1940 (checked in closed form).
    span_years = (plan_date - date(1940, 1, 1)).days / 365.2425
    boundary_ages = exponential_boundary_ages(span_years, 6.0, 8)

    def area_up_to(age_years: float) -> float:
        return 1.0 - 2.0 ** (-age_years / 6.0)

    for newer_age, older_age in itertools.pairwise(boundary_ages):
        assert (area_up_to(older_age) - area_up_to(newer_age)) / area_up_to(span_years) == pytest.approx(1 / 8)
    assert boundary_ages[4] == pytest.approx(6.0, abs=0.05)  # half the weight falls within one half-life
    # Old windows take their most-cited papers first; recent ones use relevance.
    assert [str(window.result_order) for window in windows] == ["relevance"] * 4 + ["most_cited"] * 4


def uniform_literature_1940_to_2026(works_per_year: int = 40, first_year: int = 1940) -> list[FakeOpenAlexWork]:
    """A synthetic literature with the SAME number of matching works in every year, so any recency skew in what is
    admitted comes from the schedule, not from the data. Citation counts vary so 'most cited' is well defined."""
    works = []
    for year in range(first_year, 2027):
        for index in range(works_per_year):
            work_number = (year - 1900) * 100 + index
            works.append(
                FakeOpenAlexWork(
                    f"W5{work_number:07d}",
                    f"Synthetic {PHRASE} work {year}-{index}",
                    "SYNTHETIC.",
                    publication_date=date(year, 1 + index % 12, 1 + index % 28),
                    cited_by_count=(work_number * 37) % 997,
                )
            )
    return works


def admitted_publication_dates_by_window(app) -> list[list[str]]:
    plan = _topic_plan(app)
    history_rows = app.connection.execute(
        "SELECT first_published_date FROM papers WHERE origin_route LIKE 'history:%'"
    ).fetchall()
    return [
        [
            row["first_published_date"]
            for row in history_rows
            if window.first_date.isoformat() <= row["first_published_date"] <= window.last_date.isoformat()
        ]
        for window in plan_windows(plan)
    ]


def test_historical_candidates_decay_exponentially_with_age_and_old_picks_are_the_most_cited(tmp_path, providers):
    app = open_app(
        tmp_path, historical_candidates_per_topic=("300", "160"), historical_pages_per_topic_per_run=("6", "200")
    )
    try:
        providers.openalex_works += uniform_literature_1940_to_2026()
        # Pages of 50 are larger than a window's share (20): the fair quota must stop the newest windows from
        # taking the whole budget (a real page holds 100 records).
        collect(app, providers, page_size=50, credentials=OPENALEX_TEST_KEY)

        windows = plan_windows(_topic_plan(app))
        dates_by_window = admitted_publication_dates_by_window(app)
        assert sum(len(window_dates) for window_dates in dates_by_window) == 160
        assert all(19 <= len(window_dates) <= 21 for window_dates in dates_by_window), [
            len(window_dates) for window_dates in dates_by_window
        ]  # equal shares of 160
        candidates_per_year = [
            len(window_dates) / (((window.last_date - window.first_date).days + 1) / 365.2425)
            for window, window_dates in zip(windows, dates_by_window)
        ]
        assert candidates_per_year == sorted(candidates_per_year, reverse=True)  # density falls with age
        assert candidates_per_year[0] > 40 * candidates_per_year[-1]

        all_years = sorted(int(published[:4]) for window_dates in dates_by_window for published in window_dates)
        assert all_years[len(all_years) // 2] >= 2018  # half of the candidates are from the last ~8 years
        assert sum(year < 2000 for year in all_years) >= 10  # yet the old literature is still represented
        # The oldest window's share went to the most-cited works of 1940-2008, not to arbitrary ones.
        oldest_window = windows[-1]
        expected_most_cited = sorted(
            (
                work
                for work in providers.openalex_works
                if oldest_window.first_date <= work.publication_date <= oldest_window.last_date
            ),
            key=lambda work: (-work.cited_by_count, work.work_id),
        )[: len(dates_by_window[-1])]
        assert {work.publication_date.isoformat() for work in expected_most_cited} == set(dates_by_window[-1])
        old_window_requests = [
            request
            for request in providers.requests_to("api.openalex.org")
            if f"from_publication_date:{oldest_window.first_date}" in request.url.params["filter"]
        ]
        assert old_window_requests and all(
            request.url.params.get("sort") == "cited_by_count:desc"
            and "search" not in request.url.params
            and f'title_and_abstract.search:"{PHRASE}"' in request.url.params["filter"]
            for request in old_window_requests
        )  # the exact phrase in title/abstract, most cited first
        newest_window_requests = [
            request
            for request in providers.requests_to("api.openalex.org")
            if f"from_publication_date:{windows[0].first_date}" in request.url.params["filter"]
        ]
        assert newest_window_requests and all("sort" not in request.url.params for request in newest_window_requests)
        # arXiv holds nothing before August 1991, so it is never asked about such windows (none end that early here).
        assert all(window.last_date.year >= 1991 for window in windows)
    finally:
        app.close()


def test_the_share_of_an_empty_era_flows_to_the_windows_that_still_have_papers(tmp_path, providers):
    app = open_app(
        tmp_path, historical_candidates_per_topic=("300", "120"), historical_pages_per_topic_per_run=("6", "200")
    )
    try:
        providers.openalex_works += uniform_literature_1940_to_2026(first_year=2016)  # nothing before 2016
        collect(app, providers, page_size=50, credentials=OPENALEX_TEST_KEY)

        dates_by_window = admitted_publication_dates_by_window(app)
        assert sum(len(window_dates) for window_dates in dates_by_window) == 120  # the budget is fully used
        assert dates_by_window[-1] == [] and dates_by_window[-2] == []  # 1940-2008 and 2008-2014: no papers
        assert min(len(window_dates) for window_dates in dates_by_window[:5]) >= 120 // 8  # each got at least its share
        assert _topic_plan(app).coverage_status == "capped"
    finally:
        app.close()


def test_changing_the_recency_schedule_is_a_material_change_with_a_fresh_plan(tmp_path, providers):
    app = open_app(tmp_path)
    try:
        collect(app, providers)
        first_plan_key = _topic_plan(app).search_key
    finally:
        app.close()
    config_path = tmp_path / "home" / "config.toml"
    config_path.write_text(
        config_path.read_text().replace("historical_half_life_years = 6.0", "historical_half_life_years = 10.0")
    )
    from paperwatch.app_context import AppContext
    from paperwatch.llm.fake_client import FakeModelClient

    reopened_app = AppContext.load(tmp_path / "home", model_client=FakeModelClient())
    try:
        collect(reopened_app, providers)
        new_plan = _topic_plan(reopened_app)
        assert new_plan.search_key != first_plan_key and new_plan.query_parameters["schedule"]["half_life_years"] == 10
    finally:
        reopened_app.close()
