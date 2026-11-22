"""Fake arXiv (announcement feeds, legacy API, PDFs) and OpenAlex (works search) behind one httpx.MockTransport.

The fakes follow the real formats checked against live responses on 2026-11-19: the Atom announcement feed at
rss.arxiv.org/atom/<category> (with ETag), export.arxiv.org/api/query with opensearch paging and submittedDate
windows, and api.openalex.org/works with cursor paging and publication-date filters. For plan part 8 it also serves
single works (/works/W123, /works/doi:...) with their `referenced_works`, and id batches (filter=openalex:W1|W2).
All paper text is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime
from xml.sax.saxutils import escape

import httpx

PUBLIC_ADDRESS = "93.184.216.34"
ARXIV_API_QUERY_PATTERN = re.compile(
    r"^\((?:cat:(?P<category>[^)\s]+)|all:\"(?P<phrase>[^\"]+)\")\)(?: AND submittedDate:\[(?P<first>\d{8})\d{4} TO "
    r"(?P<last>\d{8})\d{4}\])?$"
)


@dataclass
class FakeArxivPaper:
    arxiv_id: str  # base id, e.g. "2611.00001"
    title: str
    abstract: str
    submitted_on: date  # v1 submission
    categories: tuple[str, ...] = ("cs.RO",)
    version_number: int = 1
    updated_on: date | None = None  # this version's date
    comment: str = ""
    doi: str = ""
    authors: tuple[str, ...] = ("A. Synthetic", "B. Example")

    @property
    def versioned_id(self) -> str:
        return f"{self.arxiv_id}v{self.version_number}"


@dataclass
class FeedItem:
    paper: FakeArxivPaper
    announce_type: str  # new | cross | replace | replace-cross


@dataclass
class FakeOpenAlexWork:
    work_id: str  # "W123"
    title: str
    abstract: str
    publication_date: date
    visible_from: datetime | None = None  # late indexing: invisible before this moment
    doi: str = ""
    arxiv_id: str = ""  # an arXiv location of the same work
    is_retracted: bool = False
    cited_by_count: int = 0
    referenced_work_ids: tuple[str, ...] = ()  # the works this work cites ("W2", ...)


@dataclass
class FakePaperProviders:
    current_time: callable = None  # returns the test's "now" (for OpenAlex late indexing)
    arxiv_papers: list[FakeArxivPaper] = field(default_factory=list)
    feeds: dict[str, tuple[date, list[FeedItem]]] = field(default_factory=dict)  # category -> (announced, items)
    openalex_works: list[FakeOpenAlexWork] = field(default_factory=list)
    failures_remaining: dict[tuple[str, str], int] = field(default_factory=dict)  # (host, start or cursor) -> count
    sent_requests: list[httpx.Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.transport = httpx.MockTransport(self._respond)

    # ---- routing --------------------------------------------------------------------------------------------------
    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.sent_requests.append(request)
        paging_value = request.url.params.get("start") or request.url.params.get("cursor") or ""
        if self.failures_remaining.get((request.url.host, paging_value), 0) > 0:
            self.failures_remaining[(request.url.host, paging_value)] -= 1
            return _streamed(503, {"content-type": "text/plain"}, b"temporarily unavailable")
        if request.url.host == "rss.arxiv.org":
            return self._feed(request)
        if request.url.host == "export.arxiv.org" and request.url.path == "/api/query":
            return self._arxiv_api(request)
        if request.url.host == "export.arxiv.org" and request.url.path.startswith("/pdf/"):
            return _streamed(
                200, {"content-type": "application/pdf"}, b"%PDF-1.4 synthetic " + request.url.path.encode()
            )
        if request.url.host == "api.openalex.org":
            return self._openalex(request)
        return _streamed(404, {"content-type": "text/plain"}, b"not found")

    # ---- arXiv feed -----------------------------------------------------------------------------------------------
    def _feed(self, request: httpx.Request) -> httpx.Response:
        category = request.url.path.removeprefix("/atom/")
        announced_on, items = self.feeds.get(category, (None, []))
        feed_bytes = build_feed_xml(category, announced_on, items)
        entity_tag = '"' + hashlib.sha256(feed_bytes).hexdigest()[:32] + '"'
        if request.headers.get("If-None-Match") == entity_tag:
            return _streamed(304, {"ETag": entity_tag}, b"")
        return _streamed(200, {"content-type": "application/atom+xml", "ETag": entity_tag}, feed_bytes)

    # ---- arXiv legacy API -----------------------------------------------------------------------------------------
    def _arxiv_api(self, request: httpx.Request) -> httpx.Response:
        query_match = ARXIV_API_QUERY_PATTERN.match(request.url.params["search_query"])
        assert query_match, request.url.params["search_query"]
        matching_papers = [paper for paper in self.arxiv_papers if _arxiv_paper_matches(paper, query_match)]
        matching_papers.sort(key=lambda paper: (paper.submitted_on, paper.arxiv_id), reverse=True)
        start, page_size = int(request.url.params["start"]), int(request.url.params["max_results"])
        return _streamed(
            200,
            {"content-type": "application/atom+xml; charset=utf-8"},
            build_api_page_xml(matching_papers[start : start + page_size], len(matching_papers), start),
        )

    # ---- OpenAlex -------------------------------------------------------------------------------------------------
    def _openalex(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/works/"):
            return self._openalex_single_work(request.url.path.removeprefix("/works/"))
        filters = dict(part.split(":", 1) for part in request.url.params.get("filter", "").split(",") if part)
        if "openalex" in filters:  # a batch of known ids (plan part 8)
            wanted_ids = filters["openalex"].split("|")
            document = {
                "meta": {"count": 0, "next_cursor": None, "per_page": len(wanted_ids)},
                "results": [openalex_work_json(work) for work in self.openalex_works if work.work_id in wanted_ids],
            }
            document["meta"]["count"] = len(document["results"])
            return _streamed(200, {"content-type": "application/json"}, json.dumps(document).encode())
        phrase = (request.url.params.get("search") or filters.get("title_and_abstract.search", "").strip('"')).lower()
        first_date = (
            date.fromisoformat(filters["from_publication_date"]) if "from_publication_date" in filters else date.min
        )
        last_date = date.fromisoformat(filters["to_publication_date"]) if "to_publication_date" in filters else date.max
        now = self.current_time() if self.current_time else None
        visible_works = [
            work
            for work in self.openalex_works
            if (work.visible_from is None or now is None or now >= work.visible_from)
            and first_date <= work.publication_date <= last_date
            and phrase in (work.title + " " + work.abstract).lower()
        ]
        if request.url.params.get("sort") == "cited_by_count:desc":
            visible_works.sort(key=lambda work: (-work.cited_by_count, work.work_id))
        page_size = int(request.url.params["per-page"])
        cursor = request.url.params["cursor"]
        start = 0 if cursor == "*" else int(cursor.removeprefix("page"))
        page_works = visible_works[start : start + page_size]
        next_cursor = f"page{start + page_size}" if start + page_size < len(visible_works) else None
        document = {
            "meta": {"count": len(visible_works), "next_cursor": next_cursor, "per_page": page_size},
            "results": [openalex_work_json(work) for work in page_works],
        }
        return _streamed(200, {"content-type": "application/json"}, json.dumps(document).encode())

    def _openalex_single_work(self, work_reference: str) -> httpx.Response:
        for work in self.openalex_works:
            if (
                work_reference == work.work_id
                or (work.doi and work_reference.lower() == f"doi:{work.doi}".lower())
                or (work.arxiv_id and work_reference.lower() == f"doi:10.48550/arxiv.{work.arxiv_id}".lower())
            ):
                return _streamed(
                    200, {"content-type": "application/json"}, json.dumps(openalex_work_json(work)).encode()
                )
        return _streamed(404, {"content-type": "application/json"}, b'{"error": "not found"}')

    # ---- inspection -----------------------------------------------------------------------------------------------
    def requests_to(self, host: str, path_prefix: str = "") -> list[httpx.Request]:
        return [
            request
            for request in self.sent_requests
            if request.url.host == host and request.url.path.startswith(path_prefix)
        ]


def _arxiv_paper_matches(paper: FakeArxivPaper, query_match: re.Match) -> bool:
    if query_match["category"] and query_match["category"] not in paper.categories:
        return False
    if query_match["phrase"] and query_match["phrase"].lower() not in (paper.title + " " + paper.abstract).lower():
        return False
    if query_match["first"]:
        first_date = date.fromisoformat(query_match["first"])  # YYYYMMDD
        last_date = date.fromisoformat(query_match["last"])
        return first_date <= paper.submitted_on <= last_date
    return True


# --------------------------------------------------------------------------------------------------------------------
# Response bodies in the real formats
# --------------------------------------------------------------------------------------------------------------------


def build_feed_xml(category: str, announced_on: date | None, items: list[FeedItem]) -> bytes:
    entries = []
    for item in items:
        paper = item.paper
        entries.append(f"""
  <entry>
    <id>oai:arXiv.org:{paper.versioned_id}</id>
    <title>{escape(paper.title)}</title>
    <updated>{announced_on.isoformat()}T05:01:26.819743+00:00</updated>
    <link href="https://arxiv.org/abs/{paper.arxiv_id}"/>
    <summary>arXiv:{paper.versioned_id} Announce Type: {item.announce_type}
Abstract: {escape(paper.abstract)}</summary>
    {"".join(f'<category term="{escape(c)}"/>' for c in paper.categories)}
    <published>{announced_on.isoformat()}T00:00:00-05:00</published>
    <arxiv:announce_type>{item.announce_type}</arxiv:announce_type>
    <dc:rights>http://creativecommons.org/licenses/by/4.0/</dc:rights>
    {f"<arxiv:DOI>{escape(paper.doi)}</arxiv:DOI>" if paper.doi else ""}
    <dc:creator>{escape(", ".join(paper.authors))}</dc:creator>
  </entry>""")
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns="http://www.w3.org/2005/Atom" xml:lang="en-us">
  <id>http://rss.arxiv.org/atom/{category}</id>
  <title>{category} updates on arXiv.org</title>
  <updated>{(announced_on or date(2026, 1, 1)).isoformat()}T05:01:26.819627+00:00</updated>{"".join(entries)}
</feed>""".encode()


def build_api_page_xml(papers: list[FakeArxivPaper], total_results: int, start: int) -> bytes:
    entries = []
    for paper in papers:
        updated_on = paper.updated_on or paper.submitted_on
        entries.append(f"""
  <entry>
    <id>http://arxiv.org/abs/{paper.versioned_id}</id>
    <title>{escape(paper.title)}</title>
    <updated>{updated_on.isoformat()}T18:59:38Z</updated>
    <summary>{escape(paper.abstract)}</summary>
    {"".join(f'<category term="{escape(c)}" scheme="http://arxiv.org/schemas/atom"/>' for c in paper.categories)}
    <published>{paper.submitted_on.isoformat()}T18:59:38Z</published>
    {f"<arxiv:comment>{escape(paper.comment)}</arxiv:comment>" if paper.comment else ""}
    {f"<arxiv:doi>{escape(paper.doi)}</arxiv:doi>" if paper.doi else ""}
    {"".join(f"<author><name>{escape(name)}</name></author>" for name in paper.authors)}
  </entry>""")
    return f"""<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <id>https://arxiv.org/api/synthetic</id>
  <title>arXiv Query</title>
  <opensearch:itemsPerPage>{len(papers)}</opensearch:itemsPerPage>
  <opensearch:totalResults>{total_results}</opensearch:totalResults>
  <opensearch:startIndex>{start}</opensearch:startIndex>{"".join(entries)}
</feed>""".encode()


def openalex_work_json(work: FakeOpenAlexWork) -> dict:
    abstract_words = work.abstract.split()
    inverted_index: dict[str, list[int]] = {}
    for position, word in enumerate(abstract_words):
        inverted_index.setdefault(word, []).append(position)
    locations = []
    if work.doi:
        locations.append(
            {
                "id": f"doi:{work.doi}",
                "doi": f"https://doi.org/{work.doi}",
                "landing_page_url": f"https://doi.org/{work.doi}",
                "pdf_url": None,
                "source": {"display_name": "Synthetic Journal"},
                "version": "publishedVersion",
            }
        )
    if work.arxiv_id:
        locations.append(
            {
                "landing_page_url": f"https://arxiv.org/abs/{work.arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{work.arxiv_id}",
                "source": {"display_name": "arXiv (Cornell University)"},
                "version": "submittedVersion",
            }
        )
        locations.append(
            {
                "id": f"doi:10.48550/arxiv.{work.arxiv_id}",
                "landing_page_url": f"https://doi.org/10.48550/arxiv.{work.arxiv_id}",
            }
        )
    return {
        "id": f"https://openalex.org/{work.work_id}",
        "doi": f"https://doi.org/{work.doi}" if work.doi else None,
        "display_name": work.title,
        "publication_date": work.publication_date.isoformat(),
        "publication_year": work.publication_date.year,
        "type": "article",
        "is_retracted": work.is_retracted,
        "cited_by_count": work.cited_by_count,
        "authorships": [{"author": {"display_name": "C. Synthetic"}}],
        "abstract_inverted_index": inverted_index,
        "ids": {
            "openalex": f"https://openalex.org/{work.work_id}",
            **({"doi": f"https://doi.org/{work.doi}"} if work.doi else {}),
        },
        "locations": locations,
        "referenced_works": [f"https://openalex.org/{work_id}" for work_id in work.referenced_work_ids],
    }


def _streamed(status_code: int, headers: dict, content: bytes) -> httpx.Response:
    return httpx.Response(status_code, headers=headers, stream=httpx.ByteStream(content))


def search_windows_sent(requests: list[httpx.Request]) -> list[str]:
    """The submittedDate window (or OpenAlex filter) of each search request, in order."""
    windows = []
    for request in requests:
        query_text = request.url.params.get("search_query") or request.url.params.get("filter") or ""
        window_match = re.search(
            r"submittedDate:\[(\d{8})\d{4} TO (\d{8})\d{4}\]|from_publication_date:([\d-]+),"
            r"to_publication_date:([\d-]+)",
            urllib.parse.unquote(query_text),
        )
        windows.append("..".join(group for group in window_match.groups() if group) if window_match else "")
    return windows
