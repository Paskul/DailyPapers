"""One HTTPS GET hop under the collector's access rules. Pure network code: no database, no redirect following.

What one hop guarantees (plan part 3):
  * strict TLS: system CA store, hostname verification, TLS 1.2 or newer; certificate failures are refusals;
  * no proxy from the environment, no cookies carried between hops, no automatic redirects (the caller decides
    whether a redirect target is allowed, see `resolve_allowed_redirect`);
  * the target host must resolve ONLY to public addresses, and the address actually connected to is checked
    again after connecting (the egress guard's audit hook also checks every connect in collector processes);
  * bounded response: Content-Length and streamed wire bytes, bytes after gzip/deflate decoding (decompression
    bombs stop at the limit), and total wall-clock time per hop; unexpected content types are refused.
"""

from __future__ import annotations

import email.utils
import ssl
import time
import urllib.parse
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from paperwatch.errors import ProviderUnavailable, SourceAccessDenied, SourceResponseRejected
from paperwatch.network.egress_guard import HTTPS_PORT, EgressBlocked, is_public_internet_address

ALLOWED_CONTENT_ENCODINGS = ("", "identity", "gzip", "x-gzip", "deflate")
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
MAXIMUM_RETRY_AFTER_SECONDS = 7 * 24 * 3600

AddressResolver = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class HttpsHop:
    """A fully constructed request. Built by code from validated ids (provider_requests.py), never from paper or
    model text. `headers` already include a credential if this hop may carry one."""

    provider_name: str
    host: str
    path: str  # percent-encoded
    query_parameters: tuple[tuple[str, str], ...] = ()
    headers: tuple[tuple[str, str], ...] = ()

    @property
    def url(self) -> str:
        query_string = urllib.parse.urlencode(self.query_parameters, quote_via=urllib.parse.quote)
        return f"https://{self.host}{self.path}" + (f"?{query_string}" if query_string else "")


@dataclass(frozen=True)
class ResponseBounds:
    accepted_content_types: tuple[str, ...]
    max_response_bytes: int
    max_decompressed_bytes: int
    connect_timeout_seconds: float
    total_timeout_seconds: float


@dataclass(frozen=True)
class HopResult:
    status_code: int
    content_type: str  # media type without parameters, lowercase ("" if absent)
    location_header: str | None  # set for redirects
    retry_after_seconds: float | None  # from Retry-After, for 429/503
    body_bytes: bytes  # decoded body of a 2xx response; b"" otherwise
    wire_bytes: int
    connected_address: str | None

    @property
    def is_redirect(self) -> bool:
        return self.status_code in REDIRECT_STATUS_CODES


def resolve_public_addresses_only(host: str, port: int) -> list[str]:
    """Default resolver: every address the host resolves to (the fetcher then requires all of them to be public)."""
    import socket  # local import: tests inject a resolver and never touch DNS
    try:
        return sorted({address_info[4][0] for address_info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)})
    except OSError:
        return []


def build_strict_tls_context() -> ssl.SSLContext:
    tls_context = ssl.create_default_context()  # CERT_REQUIRED + check_hostname, system trust store
    tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
    return tls_context


class SecureHttpsFetcher:
    def __init__(self, *, user_agent: str, transport: httpx.BaseTransport | None = None,
                 address_resolver: AddressResolver = resolve_public_addresses_only) -> None:
        self._user_agent = user_agent
        self._transport = transport  # tests inject httpx.MockTransport; production uses httpx's own
        self._address_resolver = address_resolver
        # With a real transport the connected peer address is always available and MUST be checked.
        self._require_connected_address_check = transport is None

    def fetch_hop(self, hop: HttpsHop, bounds: ResponseBounds) -> HopResult:
        self._require_only_public_addresses(hop)
        request_headers = {"User-Agent": self._user_agent, "Accept": ", ".join(bounds.accepted_content_types),
                           "Accept-Encoding": "gzip", **dict(hop.headers)}
        timeouts = httpx.Timeout(connect=bounds.connect_timeout_seconds, read=bounds.total_timeout_seconds,
                                 write=bounds.connect_timeout_seconds, pool=bounds.connect_timeout_seconds)
        hop_deadline = time.monotonic() + bounds.total_timeout_seconds
        client_options = dict(timeout=timeouts, follow_redirects=False, trust_env=False)
        if self._transport is not None:
            client_options["transport"] = self._transport
        else:
            client_options["verify"] = build_strict_tls_context()
        try:
            with httpx.Client(**client_options) as http_client:  # fresh client per hop: no cookies carried over
                with http_client.stream("GET", hop.url, headers=request_headers) as response:
                    connected_address = self._check_connected_address(hop, response)
                    return self._read_response(hop, bounds, response, connected_address, hop_deadline)
        except (SourceAccessDenied, SourceResponseRejected, ProviderUnavailable):
            raise
        except httpx.TimeoutException as timeout_error:
            raise ProviderUnavailable(f"{hop.provider_name}: {hop.host} timed out ({type(timeout_error).__name__})",
                                      provider_name=hop.provider_name) from timeout_error
        except httpx.HTTPError as http_error:
            if _find_in_cause_chain(http_error, EgressBlocked) is not None:
                raise SourceAccessDenied(f"{hop.provider_name}: connection to {hop.host} refused by the egress "
                                         f"guard: {_find_in_cause_chain(http_error, EgressBlocked)}") from http_error
            if _find_in_cause_chain(http_error, ssl.SSLError) is not None:
                raise SourceAccessDenied(f"{hop.provider_name}: TLS verification with {hop.host} failed: "
                                         f"{_find_in_cause_chain(http_error, ssl.SSLError)}") from http_error
            raise ProviderUnavailable(f"{hop.provider_name}: request to {hop.host} failed: {http_error!r}",
                                      provider_name=hop.provider_name) from http_error

    # ---- steps --------------------------------------------------------------------------------------------------

    def _require_only_public_addresses(self, hop: HttpsHop) -> None:
        resolved_addresses = self._address_resolver(hop.host, HTTPS_PORT)
        if not resolved_addresses:
            raise ProviderUnavailable(f"{hop.provider_name}: cannot resolve {hop.host}",
                                      provider_name=hop.provider_name)
        non_public_addresses = [address for address in resolved_addresses if not is_public_internet_address(address)]
        if non_public_addresses:
            raise SourceAccessDenied(f"{hop.provider_name}: {hop.host} resolves to non-public address(es) "
                                     f"{', '.join(non_public_addresses)}; request refused")

    def _check_connected_address(self, hop: HttpsHop, response: httpx.Response) -> str | None:
        network_stream = response.extensions.get("network_stream")
        server_address = network_stream.get_extra_info("server_addr") if network_stream is not None else None
        if server_address is None:
            if self._require_connected_address_check:
                raise SourceAccessDenied(f"{hop.provider_name}: cannot verify the address connected to for "
                                         f"{hop.host}; request refused")
            return None
        connected_address = str(server_address[0])
        if not is_public_internet_address(connected_address):
            raise SourceAccessDenied(f"{hop.provider_name}: {hop.host} connected to non-public address "
                                     f"{connected_address}; response discarded")
        return connected_address

    def _read_response(self, hop: HttpsHop, bounds: ResponseBounds, response: httpx.Response,
                       connected_address: str | None, hop_deadline: float) -> HopResult:
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        result_without_body = dict(status_code=response.status_code, content_type=content_type,
                                   location_header=response.headers.get("location"),
                                   retry_after_seconds=parse_retry_after_seconds(response.headers.get("retry-after")),
                                   connected_address=connected_address)
        if not 200 <= response.status_code < 300:
            return HopResult(body_bytes=b"", wire_bytes=0, **result_without_body)  # body never read

        if content_type not in bounds.accepted_content_types:
            raise SourceResponseRejected(f"{hop.provider_name}: {hop.host} answered with content type "
                                         f"'{content_type or 'none'}', expected one of {bounds.accepted_content_types}")
        declared_length = response.headers.get("content-length")
        if declared_length and declared_length.isdigit() and int(declared_length) > bounds.max_response_bytes:
            raise SourceResponseRejected(f"{hop.provider_name}: response of {declared_length} bytes exceeds the "
                                         f"{bounds.max_response_bytes}-byte limit")
        content_encoding = response.headers.get("content-encoding", "").strip().lower()
        if content_encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise SourceResponseRejected(f"{hop.provider_name}: unsupported content encoding '{content_encoding}'")

        body_decoder = BoundedBodyDecoder(content_encoding, bounds.max_decompressed_bytes, hop.provider_name)
        wire_bytes = 0
        for raw_chunk in response.iter_raw():
            wire_bytes += len(raw_chunk)
            if wire_bytes > bounds.max_response_bytes:
                raise SourceResponseRejected(f"{hop.provider_name}: response exceeds the "
                                             f"{bounds.max_response_bytes}-byte limit on the wire")
            if time.monotonic() > hop_deadline:
                raise SourceResponseRejected(f"{hop.provider_name}: response took longer than "
                                             f"{bounds.total_timeout_seconds:.0f} s")
            body_decoder.feed(raw_chunk)
        return HopResult(body_bytes=body_decoder.finish(), wire_bytes=wire_bytes, **result_without_body)


class BoundedBodyDecoder:
    """gzip/deflate decoding that never produces more than `max_decoded_bytes` (stops decompression bombs early)."""

    def __init__(self, content_encoding: str, max_decoded_bytes: int, provider_name: str) -> None:
        self._max_decoded_bytes = max_decoded_bytes
        self._provider_name = provider_name
        self._decoded = bytearray()
        # wbits 32 + MAX_WBITS accepts both gzip and zlib-wrapped deflate.
        self._decompressor = None if content_encoding in ("", "identity") else \
            zlib.decompressobj(32 + zlib.MAX_WBITS)

    def feed(self, raw_chunk: bytes) -> None:
        if self._decompressor is None:
            self._append(raw_chunk)
            return
        pending_input = raw_chunk
        while pending_input:
            try:
                decoded_piece = self._decompressor.decompress(pending_input, self._remaining_capacity() + 1)
            except zlib.error as decode_error:
                raise SourceResponseRejected(f"{self._provider_name}: corrupt compressed response: "
                                             f"{decode_error}") from None
            self._append(decoded_piece)
            pending_input = self._decompressor.unconsumed_tail

    def finish(self) -> bytes:
        if self._decompressor is not None:
            self._append(self._decompressor.flush())
            if not self._decompressor.eof:
                raise SourceResponseRejected(f"{self._provider_name}: compressed response is truncated")
        return bytes(self._decoded)

    def _remaining_capacity(self) -> int:
        return self._max_decoded_bytes - len(self._decoded)

    def _append(self, decoded_piece: bytes) -> None:
        if len(decoded_piece) > self._remaining_capacity():
            raise SourceResponseRejected(f"{self._provider_name}: decoded response exceeds the "
                                         f"{self._max_decoded_bytes}-byte limit")
        self._decoded.extend(decoded_piece)


def resolve_allowed_redirect(current_hop: HttpsHop, location_header: str | None,
                             allowed_redirect_hosts: frozenset[str]) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    """Return (host, path, query) of an allowed redirect target, or raise SourceAccessDenied.

    Allowed only if the target is HTTPS, on the default port, without user info, and its host is listed in the
    operation's `redirect_hosts`. Relative locations resolve against the current hop (same host).
    """
    if not location_header:
        raise SourceAccessDenied(f"{current_hop.provider_name}: redirect without a Location header refused")
    target_url = urllib.parse.urlsplit(urllib.parse.urljoin(current_hop.url, location_header.strip()))
    target_host = (target_url.hostname or "").lower()
    if target_url.scheme != "https" or target_url.username or target_url.password or \
            target_url.port not in (None, HTTPS_PORT):
        raise SourceAccessDenied(f"{current_hop.provider_name}: redirect to a non-HTTPS or unusual URL refused "
                                 f"(scheme '{target_url.scheme}', host '{target_host}')")
    if target_host not in allowed_redirect_hosts:
        raise SourceAccessDenied(f"{current_hop.provider_name}: redirect from {current_hop.host} to '{target_host}' "
                                 "refused; that host is not an allowed redirect destination for this operation")
    return target_host, target_url.path or "/", tuple(urllib.parse.parse_qsl(target_url.query,
                                                                             keep_blank_values=True))


def parse_retry_after_seconds(retry_after_header: str | None) -> float | None:
    """Retry-After as seconds (either delta-seconds or an HTTP date), capped at one week."""
    if not retry_after_header:
        return None
    header_value = retry_after_header.strip()
    if header_value.isdigit():
        return float(min(int(header_value), MAXIMUM_RETRY_AFTER_SECONDS))
    try:
        retry_moment = email.utils.parsedate_to_datetime(header_value)
    except (TypeError, ValueError):
        return None
    if retry_moment.tzinfo is None:
        retry_moment = retry_moment.replace(tzinfo=timezone.utc)
    return float(min(max(0.0, (retry_moment - datetime.now(timezone.utc)).total_seconds()),
                     MAXIMUM_RETRY_AFTER_SECONDS))


def _find_in_cause_chain(error: BaseException, wanted_type: type[BaseException]) -> BaseException | None:
    seen_errors: set[int] = set()
    current_error: BaseException | None = error
    while current_error is not None and id(current_error) not in seen_errors:
        if isinstance(current_error, wanted_type):
            return current_error
        seen_errors.add(id(current_error))
        current_error = current_error.__cause__ or current_error.__context__
    return None
