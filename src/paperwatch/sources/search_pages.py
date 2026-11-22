"""Fetch and parse one page of an approved search, whatever the provider (plan part 5).

The caller names an approved catalog query id, a cursor, an optional publication window, and the discovery route to
stamp on the records; the gateway builds and checks the request, and the provider's parser turns the bytes into
records. Nothing here writes to the database (the outbound ledger aside), so a failed page changes no checkpoint.
"""

from __future__ import annotations

from paperwatch.errors import SourceAccessDenied
from paperwatch.network.provider_gateway import ProviderGateway
from paperwatch.network.provider_requests import PublicationWindow, SearchResultOrder
from paperwatch.sources.arxiv_atom import parse_api_search_page
from paperwatch.sources.openalex_works import parse_works_page
from paperwatch.sources.types import ParsedSearchPage


def fetch_search_page(
    gateway: ProviderGateway,
    *,
    provider_name: str,
    query_id: str,
    cursor: str | None,
    publication_window: PublicationWindow | None,
    discovery_route: str,
    result_order: SearchResultOrder = SearchResultOrder.PROVIDER_DEFAULT,
) -> ParsedSearchPage:
    provider_response = gateway.search(
        query_id, cursor, publication_window=publication_window, result_order=result_order
    )
    if provider_name == "arxiv":
        return parse_api_search_page(
            provider_response.body_bytes, discovery_route, start_offset=int(cursor) if cursor else 0
        )
    if provider_name == "openalex":
        return parse_works_page(provider_response.body_bytes, discovery_route)
    raise SourceAccessDenied(f"no search page parser exists for provider '{provider_name}'")
