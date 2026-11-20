"""Names of discovery routes: how a paper was found. Every catalog sighting carries one (paper_discoveries.route).

A route is "<kind prefix>:<detail>", built only from approved public values (a category, a catalog query id) or
internal keys, never from paper text:
    arxiv-feed:cs.RO                                  arXiv's daily announcement feed of one category   (current)
    arxiv-recovery:cs.RO                              dated arXiv API search after a feed gap            (current)
    openalex-recent:<query id>                        OpenAlex search over a recent publication window   (current)
    history:<query id>                                the finite historical search of a topic            (historical)
    zotero-seed:<collection> / csl-seed:<file key>    the private seed library (plan part 4)             (seed)
    fixture:<file name>                               recorded fixtures for tests and replays            (fixture)
    research-branch:<query id>                        a gap search the research role chose (part 8)      (historical)
    reference:<citing paper id>                       a work cited by a reviewed paper (part 8)           (reference)
    openalex-lookup:<paper id>                        a reviewed paper's own OpenAlex record (part 8)     (reference)
The paper's ORIGINAL route (papers.origin_route) is never rewritten, even when a later route raises its priority.
Only current and historical routes count against the daily budget of new metadata candidates. Reference routes are
bounded instead by the reference settings (papers fetched per run, ids per paper).
"""

from __future__ import annotations

from enum import StrEnum

from paperwatch.seeds.seed_items import SEED_DISCOVERY_ROUTE_PREFIXES

ARXIV_FEED_ROUTE_PREFIX = "arxiv-feed:"
ARXIV_RECOVERY_ROUTE_PREFIX = "arxiv-recovery:"
OPENALEX_RECENT_ROUTE_PREFIX = "openalex-recent:"
HISTORICAL_SEARCH_ROUTE_PREFIX = "history:"
FIXTURE_ROUTE_PREFIX = "fixture:"
RESEARCH_BRANCH_ROUTE_PREFIX = "research-branch:"
REFERENCE_ROUTE_PREFIX = "reference:"
OPENALEX_LOOKUP_ROUTE_PREFIX = "openalex-lookup:"

CURRENT_RELEASE_ROUTE_PREFIXES = (ARXIV_FEED_ROUTE_PREFIX, ARXIV_RECOVERY_ROUTE_PREFIX, OPENALEX_RECENT_ROUTE_PREFIX)
HISTORICAL_ROUTE_PREFIXES = (HISTORICAL_SEARCH_ROUTE_PREFIX, RESEARCH_BRANCH_ROUTE_PREFIX)
REFERENCE_ROUTE_PREFIXES = (REFERENCE_ROUTE_PREFIX, OPENALEX_LOOKUP_ROUTE_PREFIX)
PROVIDER_SEARCH_ROUTE_PREFIXES = CURRENT_RELEASE_ROUTE_PREFIXES + HISTORICAL_ROUTE_PREFIXES


class RouteKind(StrEnum):
    CURRENT_RELEASE = "current_release"
    HISTORICAL_SEARCH = "historical_search"
    SEED_LIBRARY = "seed_library"
    RECORDED_FIXTURE = "recorded_fixture"
    REFERENCE = "reference"
    OTHER = "other"  # e.g. a test route


def classify_discovery_route(route: str) -> RouteKind:
    if route.startswith(CURRENT_RELEASE_ROUTE_PREFIXES):
        return RouteKind.CURRENT_RELEASE
    if route.startswith(HISTORICAL_ROUTE_PREFIXES):
        return RouteKind.HISTORICAL_SEARCH
    if route.startswith(SEED_DISCOVERY_ROUTE_PREFIXES):
        return RouteKind.SEED_LIBRARY
    if route.startswith(FIXTURE_ROUTE_PREFIX):
        return RouteKind.RECORDED_FIXTURE
    if route.startswith(REFERENCE_ROUTE_PREFIXES):
        return RouteKind.REFERENCE
    return RouteKind.OTHER


def arxiv_feed_route(arxiv_category: str) -> str:
    return f"{ARXIV_FEED_ROUTE_PREFIX}{arxiv_category}"


def arxiv_recovery_route(arxiv_category: str) -> str:
    return f"{ARXIV_RECOVERY_ROUTE_PREFIX}{arxiv_category}"


def openalex_recent_route(query_id: str) -> str:
    return f"{OPENALEX_RECENT_ROUTE_PREFIX}{query_id}"


def historical_search_route(query_id: str) -> str:
    return f"{HISTORICAL_SEARCH_ROUTE_PREFIX}{query_id}"


def research_branch_route(query_id: str) -> str:
    return f"{RESEARCH_BRANCH_ROUTE_PREFIX}{query_id}"


def reference_route(citing_paper_id: str) -> str:
    return f"{REFERENCE_ROUTE_PREFIX}{citing_paper_id}"


def openalex_lookup_route(paper_id: str) -> str:
    return f"{OPENALEX_LOOKUP_ROUTE_PREFIX}{paper_id}"
