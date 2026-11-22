"""ProviderGateway: the collector's only door to the internet.

It offers a handful of FIXED operations; callers never supply a URL, host, or free-text query:

    search(topic_query_id, cursor)          an approved catalog query (paperwatch.public_query_catalog)
    poll_category_feed(arxiv_category)      arXiv's announcement feed of a category some topic approved
    get_work(provider_id)                   metadata of a work already known from a source record
    get_works(openalex_provider_ids)        metadata of up to 100 cited works, each a stored citation edge
    get_text(known_edition_id)              the text of one catalogued edition, saved to the source cache
    list_seed_items(collection_key, start)  items of an approved Zotero collection

For every operation, code:
  1. re-checks the id against source records at dispatch (known alias / citation edge / catalog entry / approved
     collection), so a stale or invented id from a queue never reaches a provider. An identifier known ONLY from the
     private seed library is refused unless collector.allow_lookup_by_seed_identifier is set;
  2. constructs the request (provider_requests.py) and checks it against the read-only provider rules;
  3. refuses while the provider is paused after an outage, and reserves one request from its budget and spacing;
  4. attaches that provider's credential only for its own host, sends one bounded hop (secure_fetcher.py), and
     follows a redirect only to an allowed host, dropping credentials on any host change;
  5. logs every hop, with credentials redacted, to `outbound_requests`.
Network calls happen outside write transactions; each budget reservation and log row is its own short write.
Response parsing belongs to the source modules: Zotero JSON in paperwatch.seeds, Atom and OpenAlex JSON in
paperwatch.sources. A conditional request (Zotero's If-Modified-Since-Version, the feed's If-None-Match) may
come back as `was_not_modified`.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from paperwatch.catalog import find_paper_id_by_identifier, load_edition_row
from paperwatch.clock import format_utc_timestamp, utc_now, utc_now_timestamp
from paperwatch.config import AppConfig
from paperwatch.errors import (
    ProviderUnavailable,
    SourceAccessDenied,
    SourceCredentialRequired,
    SourceInputError,
    SourceRecordNotFound,
    SourceRequestBudgetExhausted,
    SourceResponseRejected,
)
from paperwatch.llm.service_guard import read_service_health_state, record_service_outage, record_service_success
from paperwatch.network.collector_credentials import CollectorCredentials, load_collector_credentials
from paperwatch.network.outbound_ledger import OutboundRequestRecord, record_outbound_request, reserve_provider_request
from paperwatch.network.provider_requests import (
    ProviderRequest,
    PublicationWindow,
    SearchResultOrder,
    build_category_feed_request,
    build_get_text_request,
    build_get_work_request,
    build_get_works_request,
    build_search_request,
    build_seed_items_request,
)
from paperwatch.network.provider_rules import (
    CREDENTIAL_KIND_HEADER,
    CREDENTIAL_KIND_QUERY_PARAMETER,
    OperationRule,
    ProviderAccessRules,
    ProviderRule,
    load_provider_access_rules,
)
from paperwatch.network.secure_fetcher import (
    AddressResolver,
    HopResult,
    HttpsHop,
    ResponseBounds,
    SecureHttpsFetcher,
    resolve_allowed_redirect,
    resolve_public_addresses_only,
)
from paperwatch.public_query_catalog import QUERY_KIND_CATEGORY, build_public_query_catalog, find_public_query
from paperwatch.scholarly_identifiers import (
    ARXIV_SCHEME,
    NormalizedIdentifier,
    identifiers_implied_by,
    normalize_identifier,
    parse_version_number,
)
from paperwatch.seeds.seed_items import SEED_DISCOVERY_ROUTE_PREFIXES
from paperwatch.source_cache import INTERNAL_ID_PATTERN, CachedSourceFile, store_source_bytes

PROVIDER_SERVICE_NAME_PREFIX = "provider:"
CREDENTIAL_SECRET_FIELD = "api_key"  # noqa: S105  # a field or option name, not a secret
STATUS_CODES_MEANING_RECORD_NOT_FOUND = frozenset({404, 410})
STATUS_CODES_MEANING_PROVIDER_UNAVAILABLE = frozenset({401, 403, 429})  # plus every 5xx


@dataclass(frozen=True)
class ProviderResponse:
    provider_name: str
    operation: str
    content_type: str
    body_bytes: bytes
    final_host: str
    response_headers: dict[str, str] = field(default_factory=dict)  # only those the operation rule exposes
    was_not_modified: bool = False  # a conditional request found nothing changed; body_bytes is empty


def provider_service_name(provider_name: str) -> str:
    return f"{PROVIDER_SERVICE_NAME_PREFIX}{provider_name}"


class ProviderGateway:
    def __init__(
        self,
        connection: sqlite3.Connection,
        app_config: AppConfig,
        *,
        credentials: CollectorCredentials,
        provider_rules: ProviderAccessRules | None = None,
        http_transport: httpx.BaseTransport | None = None,
        address_resolver: AddressResolver = resolve_public_addresses_only,
        sleep_function: Callable[[float], None] = time.sleep,
    ) -> None:
        self._connection = connection
        self._app_config = app_config
        self._credentials = credentials
        self._rules = provider_rules or load_provider_access_rules()
        self._query_catalog = build_public_query_catalog(app_config.topics)
        self._fetcher = SecureHttpsFetcher(
            user_agent=app_config.collector.user_agent, transport=http_transport, address_resolver=address_resolver
        )
        self._sleep = sleep_function

    @classmethod
    def for_collector(cls, connection: sqlite3.Connection, app_config: AppConfig) -> ProviderGateway:
        """Production constructor: the packaged rules and the collector-only credentials file."""
        return cls(
            connection, app_config, credentials=load_collector_credentials(app_config.collector_credentials_path)
        )

    # ---- fixed operations -----------------------------------------------------------------------------------------

    def search(
        self,
        query_id: str,
        cursor: str | None = None,
        *,
        publication_window: PublicationWindow | None = None,
        result_order: SearchResultOrder = SearchResultOrder.PROVIDER_DEFAULT,
    ) -> ProviderResponse:
        public_query = find_public_query(self._query_catalog, query_id)
        page_size = self._rules.provider_rule(public_query.provider_name).search_page_size
        search_rule = self._rules.operation_rule(public_query.provider_name, "search")
        is_first_page = cursor in (None, "", "*", "0")
        if (
            search_rule.cursor_paging_requires_credential
            and not is_first_page
            and self._credentials.get(public_query.provider_name, CREDENTIAL_SECRET_FIELD) is None
        ):
            raise SourceCredentialRequired(
                f"{search_rule.describe()}: pages after the first need an API key, and none is configured",
                operator_hint=f"add a free key as [{public_query.provider_name}] api_key in "
                f"{self._app_config.collector_credentials_path}",
            )
        return self._execute(build_search_request(public_query, cursor, page_size, publication_window, result_order))

    def poll_category_feed(self, arxiv_category: str, *, if_none_match_etag: str | None = None) -> ProviderResponse:
        """The current announcement feed of one arXiv category, which must be an approved category of some topic."""
        approved_categories = {
            public_query.public_value
            for public_query in self._query_catalog.values()
            if public_query.query_kind == QUERY_KIND_CATEGORY
        }
        if arxiv_category not in approved_categories:
            raise SourceAccessDenied(f"arXiv category '{arxiv_category}' is not an approved category in topics.toml")
        return self._execute(build_category_feed_request(arxiv_category, if_none_match_etag))

    def get_work(self, provider_id: str) -> ProviderResponse:
        """`provider_id` is 'scheme:value', e.g. 'arxiv:2401.01234v2', 'openalex:W123', 'doi:10.1109/x'."""
        identifier = _parse_scheme_and_value(provider_id)
        if not self._is_known_source_identifier(identifier):
            raise SourceAccessDenied(
                f"'{identifier.scheme_and_value}' is not a known source record (neither a "
                "catalog alias nor a provider-supplied citation); refusing to look it up"
            )
        return self._execute(build_get_work_request(identifier))

    def get_works(self, openalex_provider_ids: list[str]) -> ProviderResponse:
        """Metadata of up to 100 works by OpenAlex id ('openalex:W123'). Every id must already be a known source
        record (in practice: a citation edge a provider supplied for a reviewed paper); otherwise nothing is sent."""
        identifiers = [_parse_scheme_and_value(provider_id) for provider_id in openalex_provider_ids]
        unknown_identifiers = [
            identifier.scheme_and_value
            for identifier in identifiers
            if not self._is_known_source_identifier(identifier)
        ]
        if unknown_identifiers:
            raise SourceAccessDenied(
                f"{len(unknown_identifiers)} id(s) are not known source records (first: "
                f"'{unknown_identifiers[0]}'); refusing the whole lookup"
            )
        return self._execute(build_get_works_request(identifiers))

    def get_text(self, known_edition_id: str) -> CachedSourceFile:
        if not isinstance(known_edition_id, str) or not INTERNAL_ID_PATTERN.match(known_edition_id):
            raise SourceAccessDenied(f"'{known_edition_id}' is not an internal edition id")
        try:
            edition_row = load_edition_row(self._connection, known_edition_id)
        except LookupError:
            raise SourceAccessDenied(f"edition {known_edition_id} is not in the catalog") from None
        edition_identifier = self._identifier_of_catalogued_edition(edition_row)
        provider_response = self._execute(build_get_text_request(edition_identifier))
        return store_source_bytes(
            self._app_config.source_cache_dir,
            known_edition_id,
            provider_response.body_bytes,
            provider_response.content_type,
        )

    def list_seed_items(
        self, collection_key: str, start: int = 0, *, if_modified_since_library_version: int | None = None
    ) -> ProviderResponse:
        approved_collection_keys = {
            collection_key for topic in self._app_config.topics for collection_key in topic.zotero_collections
        }
        if collection_key not in approved_collection_keys:
            raise SourceAccessDenied(f"Zotero collection '{collection_key}' is not listed in topics.toml")
        library_id = self._credentials.get("zotero", "library_id")
        if library_id is None:
            raise SourceAccessDenied(
                "Zotero library_id is missing from the collector credentials file",
                operator_hint=f"add [zotero] library_id to {self._app_config.collector_credentials_path}",
            )
        return self._execute(
            build_seed_items_request(
                library_type=self._credentials.get("zotero", "library_type") or "user",
                library_id=library_id,
                collection_key=collection_key,
                start=start,
                page_size=self._rules.provider_rule("zotero").search_page_size,
                if_modified_since_library_version=if_modified_since_library_version,
            )
        )

    # ---- dispatch-time checks against source records ----------------------------------------------------------

    def _is_known_source_identifier(self, identifier: NormalizedIdentifier) -> bool:
        """Known = a provider supplied it (a citation edge or a paper found by a public route), it was found in the
        bibliography of a paper the collector downloaded and awaits confirmation by exactly this lookup,
        or it is a seed's identifier and the operator explicitly allowed seed lookups (looking it up discloses
        interest in it). An identifier that by definition names a known work (arXiv's own DataCite DOI of a
        catalogued arXiv paper) counts as that work."""
        version_free_identifier = identifier.without_version()
        if (
            self._connection.execute(
                "SELECT 1 FROM citation_edges WHERE cited_scheme = ? AND cited_value = ? LIMIT 1",
                (version_free_identifier.scheme, version_free_identifier.value),
            ).fetchone()
            is not None
        ):
            return True
        if (
            self._connection.execute(
                """SELECT 1 FROM bibliography_identifiers WHERE identifier_scheme = ? AND identifier_value = ?
                     AND confirmation_status = 'pending' LIMIT 1""",
                (version_free_identifier.scheme, version_free_identifier.value),
            ).fetchone()
            is not None
        ):
            return True
        paper_id = find_paper_id_by_identifier(self._connection, version_free_identifier)
        for implied_identifier in identifiers_implied_by(version_free_identifier):
            paper_id = paper_id or find_paper_id_by_identifier(self._connection, implied_identifier.without_version())
        if paper_id is None:
            return False
        if self._app_config.collector.allow_lookup_by_seed_identifier or _paper_was_found_by_a_public_route(
            self._connection, paper_id
        ):
            return True
        raise SourceAccessDenied(
            f"'{identifier.scheme_and_value}' is known only from the private seed library; looking it up would "
            "disclose interest in that paper",
            operator_hint="set collector.allow_lookup_by_seed_identifier = true in config.toml to permit this",
        )

    def _identifier_of_catalogued_edition(self, edition_row: sqlite3.Row) -> NormalizedIdentifier:
        """The edition's own identifier (with version), re-checked against the paper's aliases."""
        identifier_text, _, version_label = edition_row["edition_key"].partition("@")
        edition_identifier = _parse_scheme_and_value(identifier_text)
        if find_paper_id_by_identifier(self._connection, edition_identifier) != edition_row["paper_id"]:
            raise SourceAccessDenied(
                f"edition {edition_row['edition_id']} no longer matches a source record of its "
                "paper; refusing to fetch its text"
            )
        if edition_identifier.scheme == ARXIV_SCHEME:
            return NormalizedIdentifier(ARXIV_SCHEME, edition_identifier.value, parse_version_number(version_label))
        return edition_identifier

    # ---- the one shared request path ------------------------------------------------------------------------------

    def _execute(self, provider_request: ProviderRequest) -> ProviderResponse:
        provider_rule = self._rules.provider_rule(provider_request.provider_name)
        operation_rule = self._rules.operation_rule(provider_request.provider_name, provider_request.operation)
        _require_request_within_rule(provider_request, operation_rule)
        self._require_provider_not_paused(provider_rule.provider_name)
        credential_headers, credential_parameters = self._credential_for(provider_rule, operation_rule)
        bounds = ResponseBounds(
            accepted_content_types=operation_rule.content_types,
            max_response_bytes=operation_rule.max_response_bytes,
            max_decompressed_bytes=operation_rule.max_decompressed_bytes,
            connect_timeout_seconds=self._rules.connect_timeout_seconds,
            total_timeout_seconds=operation_rule.total_timeout_seconds,
        )
        current_hop = HttpsHop(
            provider_rule.provider_name,
            operation_rule.host,
            provider_request.path,
            provider_request.query_parameters + credential_parameters,
            provider_rule.fixed_headers + provider_request.request_headers + credential_headers,
        )

        for hop_number in range(self._rules.max_redirect_hops + 1):
            hop_result = self._send_one_hop(provider_rule, operation_rule, current_hop, hop_number, bounds)
            was_not_modified = hop_result.is_not_modified and bool(provider_request.request_headers)
            if 200 <= hop_result.status_code < 300 or was_not_modified:
                record_service_success(self._connection, provider_service_name(provider_rule.provider_name))
                return ProviderResponse(
                    provider_rule.provider_name,
                    operation_rule.operation,
                    hop_result.content_type,
                    hop_result.body_bytes,
                    current_hop.host,
                    _exposed_response_headers(hop_result, operation_rule),
                    was_not_modified,
                )
            if hop_result.is_redirect:
                current_hop = self._next_hop_after_redirect(
                    provider_rule, operation_rule, current_hop, hop_result, hop_number
                )
                continue
            self._raise_for_unsuccessful_status(provider_rule, operation_rule, hop_result)
        raise SourceAccessDenied(
            f"{operation_rule.describe()}: more than {self._rules.max_redirect_hops} redirect(s); refused"
        )

    def _send_one_hop(
        self,
        provider_rule: ProviderRule,
        operation_rule: OperationRule,
        hop: HttpsHop,
        hop_number: int,
        bounds: ResponseBounds,
    ) -> HopResult:
        started_at, started_monotonic = utc_now_timestamp(), time.monotonic()

        def log_hop(
            outcome: str,
            *,
            status_code: int | None = None,
            response_bytes: int | None = None,
            error_message: str | None = None,
        ) -> None:
            self._log(
                OutboundRequestRecord(
                    provider_name=provider_rule.provider_name,
                    operation=operation_rule.operation,
                    hop_number=hop_number,
                    host=hop.host,
                    path=hop.path,
                    query_parameters=hop.query_parameters,
                    outcome=outcome,
                    started_at=started_at,
                    duration_seconds=time.monotonic() - started_monotonic,
                    status_code=status_code,
                    response_bytes=response_bytes,
                    error_message=error_message,
                ),
                provider_rule,
            )

        try:
            wait_seconds = reserve_provider_request(
                self._connection,
                provider_name=provider_rule.provider_name,
                daily_request_budget=provider_rule.effective_daily_request_budget(
                    self._credentials.get(provider_rule.provider_name, CREDENTIAL_SECRET_FIELD) is not None
                ),
                minimum_seconds_between_requests=max(
                    provider_rule.minimum_seconds_between_requests,
                    operation_rule.minimum_seconds_between_requests or 0.0,
                ),
            )
        except SourceRequestBudgetExhausted as budget_error:
            log_hop("budget_exhausted", error_message=str(budget_error))
            raise
        if wait_seconds > 0:
            self._sleep(wait_seconds)  # minimum spacing between requests to this provider, across processes
        started_at, started_monotonic = utc_now_timestamp(), time.monotonic()
        try:
            hop_result = self._fetcher.fetch_hop(hop, bounds)
        except SourceAccessDenied as refusal:
            log_hop("blocked", error_message=str(refusal))
            raise
        except SourceResponseRejected as rejection:
            log_hop("rejected", error_message=str(rejection))
            raise
        except ProviderUnavailable as outage:
            log_hop("unavailable", error_message=str(outage))
            self._pause_provider(outage)
            raise
        outcome = (
            "ok"
            if 200 <= hop_result.status_code < 300
            else "redirect"
            if hop_result.is_redirect
            else "not_modified"
            if hop_result.is_not_modified
            else "not_found"
            if hop_result.status_code in STATUS_CODES_MEANING_RECORD_NOT_FOUND
            else "error"
        )
        log_hop(outcome, status_code=hop_result.status_code, response_bytes=len(hop_result.body_bytes))
        return hop_result

    def _next_hop_after_redirect(
        self,
        provider_rule: ProviderRule,
        operation_rule: OperationRule,
        current_hop: HttpsHop,
        hop_result: HopResult,
        hop_number: int,
    ) -> HttpsHop:
        try:
            target_host, target_path, target_query = resolve_allowed_redirect(
                current_hop, hop_result.location_header, operation_rule.redirect_hosts | {operation_rule.host}
            )
        except SourceAccessDenied as refusal:
            self._log(
                OutboundRequestRecord(
                    provider_name=provider_rule.provider_name,
                    operation=operation_rule.operation,
                    hop_number=hop_number + 1,
                    host="(refused redirect)",
                    path="",
                    query_parameters=(),
                    outcome="blocked",
                    started_at=utc_now_timestamp(),
                    duration_seconds=0.0,
                    error_message=str(refusal),
                ),
                provider_rule,
            )
            raise
        if target_host == operation_rule.host:
            # Same host: the new path and parameters must still fit the rule, and the credential stays.
            _require_request_within_rule(
                ProviderRequest(
                    provider_rule.provider_name,
                    operation_rule.operation,
                    target_path,
                    _without_credential(target_query, provider_rule),
                ),
                operation_rule,
            )
            carried_credential_parameters = tuple(
                (name, value)
                for name, value in current_hop.query_parameters
                if provider_rule.credential_kind == CREDENTIAL_KIND_QUERY_PARAMETER
                and name == provider_rule.credential_name
            )
            return HttpsHop(
                current_hop.provider_name,
                target_host,
                target_path,
                _without_credential(target_query, provider_rule) + carried_credential_parameters,
                current_hop.headers,
            )
        # Host change: drop every provider credential (header and query parameter); keep only fixed headers.
        return HttpsHop(
            current_hop.provider_name,
            target_host,
            target_path,
            _without_credential(target_query, provider_rule),
            provider_rule.fixed_headers,
        )

    def _raise_for_unsuccessful_status(
        self, provider_rule: ProviderRule, operation_rule: OperationRule, hop_result: HopResult
    ) -> None:
        description = f"{operation_rule.describe()} answered HTTP {hop_result.status_code}"
        if hop_result.status_code in STATUS_CODES_MEANING_RECORD_NOT_FOUND:
            raise SourceRecordNotFound(description)
        if hop_result.status_code in STATUS_CODES_MEANING_PROVIDER_UNAVAILABLE or hop_result.status_code >= 500:
            outage = ProviderUnavailable(
                description,
                provider_name=provider_rule.provider_name,
                retry_after_seconds=hop_result.retry_after_seconds,
                operator_hint="check the API key in the collector credentials file"
                if hop_result.status_code in (401, 403)
                else None,
            )
            self._pause_provider(outage)
            raise outage
        raise SourceInputError(f"{description}; request rejected")

    # ---- credentials, pauses, logging -------------------------------------------------------------------------

    def _credential_for(
        self, provider_rule: ProviderRule, operation_rule: OperationRule
    ) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        """(headers, query parameters) carrying the provider's credential, for the operation's own host only."""
        secret = self._credentials.get(provider_rule.provider_name, CREDENTIAL_SECRET_FIELD)
        if secret is None:
            if operation_rule.credential_required:
                raise SourceAccessDenied(
                    f"{operation_rule.describe()} needs an API key, and none is configured",
                    operator_hint=f"add [{provider_rule.provider_name}] api_key to "
                    f"{self._app_config.collector_credentials_path} (collector account, mode 0600)",
                )
            return (), ()
        if provider_rule.credential_kind == CREDENTIAL_KIND_HEADER:
            return ((provider_rule.credential_name, secret),), ()
        if provider_rule.credential_kind == CREDENTIAL_KIND_QUERY_PARAMETER:
            return (), ((provider_rule.credential_name, secret),)
        return (), ()

    def _require_provider_not_paused(self, provider_name: str) -> None:
        health_state = read_service_health_state(self._connection, provider_service_name(provider_name))
        if (paused_until := health_state.pause_end_at(utc_now())) is not None:
            raise ProviderUnavailable(
                f"{provider_name} is paused until {format_utc_timestamp(paused_until)} after "
                f"{health_state.consecutive_failures} failure(s); last error: {health_state.last_error}",
                provider_name=provider_name,
            )

    def _pause_provider(self, outage: ProviderUnavailable) -> None:
        record_service_outage(
            self._connection,
            provider_service_name(outage.provider_name),
            self._redact(str(outage)),
            self._rules,
            minimum_pause_seconds=outage.retry_after_seconds or 0.0,
        )

    def _log(self, record: OutboundRequestRecord, provider_rule: ProviderRule) -> None:
        secret_parameter_names = (
            frozenset({provider_rule.credential_name})
            if provider_rule.credential_kind == CREDENTIAL_KIND_QUERY_PARAMETER
            else frozenset()
        )
        record_outbound_request(self._connection, record, secret_parameter_names, self._credentials.all_secret_values())

    def _redact(self, text: str) -> str:
        for secret_value in self._credentials.all_secret_values():
            text = text.replace(secret_value, "[redacted]")
        return text


# --------------------------------------------------------------------------------------------------------------------
# Module helpers
# --------------------------------------------------------------------------------------------------------------------


def _parse_scheme_and_value(provider_id: str) -> NormalizedIdentifier:
    if not isinstance(provider_id, str) or ":" not in provider_id:
        raise SourceAccessDenied(f"provider id {provider_id!r} must be 'scheme:value'")
    scheme, _, value = provider_id.partition(":")
    try:
        return normalize_identifier(scheme, value)
    except SourceInputError as invalid_identifier:
        raise SourceAccessDenied(f"invalid provider id: {invalid_identifier}") from None


def _require_request_within_rule(provider_request: ProviderRequest, operation_rule: OperationRule) -> None:
    if not operation_rule.path_pattern.fullmatch(provider_request.path):
        raise SourceAccessDenied(
            f"{operation_rule.describe()}: path '{provider_request.path}' is not allowed by the provider rules"
        )
    parameter_names = {name for name, _ in provider_request.query_parameters}
    disallowed_names = parameter_names - operation_rule.query_parameter_names
    if disallowed_names:
        raise SourceAccessDenied(
            f"{operation_rule.describe()}: query parameter(s) {sorted(disallowed_names)} are "
            "not allowed by the provider rules"
        )
    for name, value in provider_request.query_parameters:
        if any(not character.isprintable() for character in value):
            raise SourceAccessDenied(f"{operation_rule.describe()}: parameter '{name}' contains control characters")
    for header_name, header_value in provider_request.request_headers:
        if header_name.lower() not in operation_rule.request_header_names:
            raise SourceAccessDenied(
                f"{operation_rule.describe()}: request header '{header_name}' is not allowed by the provider rules"
            )
        if any(not character.isprintable() for character in header_value):
            raise SourceAccessDenied(f"{operation_rule.describe()}: header '{header_name}' contains control characters")


def _exposed_response_headers(hop_result: HopResult, operation_rule: OperationRule) -> dict[str, str]:
    """Only the response headers the operation rule names (lowercase keys), e.g. Zotero's Last-Modified-Version."""
    return {name: value for name, value in hop_result.response_headers if name in operation_rule.response_header_names}


def _paper_was_found_by_a_public_route(connection: sqlite3.Connection, paper_id: str) -> bool:
    """True if any discovery route of the paper is a public provider route rather than the private seed library."""
    for discovery_row in connection.execute("SELECT route FROM paper_discoveries WHERE paper_id = ?", (paper_id,)):
        if not discovery_row["route"].startswith(SEED_DISCOVERY_ROUTE_PREFIXES):
            return True
    return False


def _without_credential(
    query_parameters: tuple[tuple[str, str], ...], provider_rule: ProviderRule
) -> tuple[tuple[str, str], ...]:
    if provider_rule.credential_kind != CREDENTIAL_KIND_QUERY_PARAMETER:
        return query_parameters
    return tuple((name, value) for name, value in query_parameters if name != provider_rule.credential_name)
