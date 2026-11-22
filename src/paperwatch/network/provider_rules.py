"""Load and validate the read-only provider access rules (provider_rules.toml, shipped inside the package).

The rules say which HTTPS requests the collector may make: per provider and fixed operation, one host, one path
shape, the allowed query parameter names, allowed redirect hosts, response bounds, a daily request budget, minimum
spacing, and where a credential goes. Anything not described is refused (SourceAccessDenied).
"""

from __future__ import annotations

import ipaddress
import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from paperwatch.errors import ConfigurationError, SourceAccessDenied

PROVIDER_RULES_RESOURCE_NAME = "provider_rules.toml"
HOSTNAME_PATTERN = re.compile(r"^(?=.{1,253}$)[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")

CREDENTIAL_KIND_NONE = "none"
CREDENTIAL_KIND_HEADER = "header"
CREDENTIAL_KIND_QUERY_PARAMETER = "query_parameter"
SUPPORTED_CREDENTIAL_KINDS = (CREDENTIAL_KIND_NONE, CREDENTIAL_KIND_HEADER, CREDENTIAL_KIND_QUERY_PARAMETER)


@dataclass(frozen=True)
class OperationRule:
    """What one fixed operation (e.g. arxiv/get_text) may request."""

    provider_name: str
    operation: str
    host: str
    path_pattern: re.Pattern[str]  # must FULLY match the percent-encoded request path
    query_parameter_names: frozenset[str]
    content_types: tuple[str, ...]  # accepted media types of a successful response (parameters ignored)
    max_response_bytes: int  # bytes on the wire (compressed)
    max_decompressed_bytes: int  # bytes after gzip/deflate decoding
    total_timeout_seconds: float  # wall-clock limit for one hop, headers and body
    redirect_hosts: frozenset[str]  # empty: every redirect is refused
    credential_required: bool
    request_header_names: frozenset[str] = frozenset()  # lowercase; extra headers a request may carry
    response_header_names: tuple[str, ...] = ()  # lowercase; response headers handed back to the caller
    cursor_paging_requires_credential: bool = False  # pages after the first need the provider's credential
    # A longer spacing after a request of THIS operation than the provider's (e.g. OpenAlex's PDF host answered 429
    # to ~10 downloads at the provider's 0.2 s spacing in the first pilot); None = the provider's spacing.
    minimum_seconds_between_requests: float | None = None

    def describe(self) -> str:
        return f"{self.provider_name}/{self.operation}"


@dataclass(frozen=True)
class ProviderRule:
    provider_name: str
    credential_kind: str  # none | header | query_parameter
    credential_name: str  # header name or query parameter name ("" when credential_kind is none)
    fixed_headers: tuple[tuple[str, str], ...]
    daily_request_budget: int
    minimum_seconds_between_requests: float
    search_page_size: int
    operations: dict[str, OperationRule]
    # A smaller budget for days without the provider's credential (e.g. OpenAlex meters unkeyed use per IP address).
    daily_request_budget_without_credential: int | None = None

    def effective_daily_request_budget(self, has_credential: bool) -> int:
        if has_credential or self.daily_request_budget_without_credential is None:
            return self.daily_request_budget
        return min(self.daily_request_budget, self.daily_request_budget_without_credential)


@dataclass(frozen=True)
class ProviderAccessRules:
    max_redirect_hops: int
    connect_timeout_seconds: float
    outage_backoff_initial_seconds: int
    outage_backoff_max_seconds: int
    providers: dict[str, ProviderRule]
    source_description: str  # where the rules came from, for operator messages

    def provider_rule(self, provider_name: str) -> ProviderRule:
        if provider_name not in self.providers:
            raise SourceAccessDenied(f"provider '{provider_name}' is not in the read-only provider rules")
        return self.providers[provider_name]

    def operation_rule(self, provider_name: str, operation: str) -> OperationRule:
        provider_rule = self.provider_rule(provider_name)
        if operation not in provider_rule.operations:
            raise SourceAccessDenied(f"operation '{operation}' is not allowed for provider '{provider_name}'")
        return provider_rule.operations[operation]

    def all_permitted_hosts(self) -> frozenset[str]:
        """Every host the collector may resolve or connect to: operation hosts plus allowed redirect hosts."""
        permitted_hosts: set[str] = set()
        for provider_rule in self.providers.values():
            for operation_rule in provider_rule.operations.values():
                permitted_hosts.add(operation_rule.host)
                permitted_hosts.update(operation_rule.redirect_hosts)
        return frozenset(permitted_hosts)


def packaged_provider_rules_path() -> Path:
    """Filesystem path of the shipped rules file (used by access-check to verify it is not writable)."""
    return Path(str(resources.files("paperwatch.network").joinpath(PROVIDER_RULES_RESOURCE_NAME)))


def load_provider_access_rules(rules_toml_text: str | None = None) -> ProviderAccessRules:
    """Parse the packaged rules, or `rules_toml_text` (tests only), into validated frozen objects."""
    source_description = "test rules" if rules_toml_text is not None else str(packaged_provider_rules_path())
    if rules_toml_text is None:
        rules_toml_text = (
            resources.files("paperwatch.network").joinpath(PROVIDER_RULES_RESOURCE_NAME).read_text(encoding="utf-8")
        )
    try:
        raw_rules = tomllib.loads(rules_toml_text)
    except tomllib.TOMLDecodeError as decode_error:
        raise ConfigurationError(f"provider rules {source_description} are not valid TOML: {decode_error}") from None

    _reject_unknown_keys(raw_rules, {"defaults", "providers"}, "provider rules")
    raw_defaults = _require_table(raw_rules.get("defaults", {}), "[defaults]")
    _reject_unknown_keys(
        raw_defaults,
        {
            "max_redirect_hops",
            "connect_timeout_seconds",
            "outage_backoff_initial_seconds",
            "outage_backoff_max_seconds",
        },
        "[defaults]",
    )
    providers = {
        provider_name: _build_provider_rule(provider_name, raw_provider)
        for provider_name, raw_provider in _require_table(raw_rules.get("providers", {}), "[providers]").items()
    }
    return ProviderAccessRules(
        max_redirect_hops=_require_number(raw_defaults.get("max_redirect_hops", 2), "max_redirect_hops", int),
        connect_timeout_seconds=_require_number(
            raw_defaults.get("connect_timeout_seconds", 10.0), "connect_timeout_seconds", float
        ),
        outage_backoff_initial_seconds=_require_number(
            raw_defaults.get("outage_backoff_initial_seconds", 300), "outage_backoff_initial_seconds", int
        ),
        outage_backoff_max_seconds=_require_number(
            raw_defaults.get("outage_backoff_max_seconds", 21600), "outage_backoff_max_seconds", int
        ),
        providers=providers,
        source_description=source_description,
    )


# --------------------------------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------------------------------

PROVIDER_KEYS = {
    "credential_kind",
    "credential_name",
    "credential_required",
    "fixed_headers",
    "daily_request_budget",
    "daily_request_budget_without_credential",
    "minimum_seconds_between_requests",
    "search_page_size",
    "operations",
}
OPERATION_KEYS = {
    "host",
    "path_pattern",
    "query_parameters",
    "content_types",
    "max_response_bytes",
    "max_decompressed_bytes",
    "total_timeout_seconds",
    "redirect_hosts",
    "credential_required",
    "request_headers",
    "response_headers",
    "cursor_paging_requires_credential",
    "minimum_seconds_between_requests",
}


def _build_provider_rule(provider_name: str, raw_provider: Any) -> ProviderRule:
    label = f"[providers.{provider_name}]"
    raw_provider = _require_table(raw_provider, label)
    _reject_unknown_keys(raw_provider, PROVIDER_KEYS, label)
    credential_kind = raw_provider.get("credential_kind", CREDENTIAL_KIND_NONE)
    if credential_kind not in SUPPORTED_CREDENTIAL_KINDS:
        raise ConfigurationError(f"{label} credential_kind must be one of {SUPPORTED_CREDENTIAL_KINDS}")
    credential_name = raw_provider.get("credential_name", "")
    if (credential_kind == CREDENTIAL_KIND_NONE) != (credential_name == ""):
        raise ConfigurationError(f"{label} needs credential_name exactly when credential_kind is not 'none'")
    provider_credential_required = bool(raw_provider.get("credential_required", False))
    fixed_headers = _require_table(raw_provider.get("fixed_headers", {}), f"{label} fixed_headers")
    operations = {
        operation: _build_operation_rule(provider_name, operation, raw_operation, provider_credential_required)
        for operation, raw_operation in _require_table(
            raw_provider.get("operations", {}), f"{label} operations"
        ).items()
    }
    if not operations:
        raise ConfigurationError(f"{label} has no operations")
    return ProviderRule(
        provider_name=provider_name,
        credential_kind=credential_kind,
        credential_name=credential_name,
        fixed_headers=tuple(sorted((str(name), str(value)) for name, value in fixed_headers.items())),
        daily_request_budget=_require_number(
            raw_provider.get("daily_request_budget"), f"{label} daily_request_budget", int
        ),
        minimum_seconds_between_requests=_require_number(
            raw_provider.get("minimum_seconds_between_requests", 0.0),
            f"{label} minimum_seconds_between_requests",
            float,
            allow_zero=True,
        ),
        search_page_size=_require_number(raw_provider.get("search_page_size", 100), f"{label} search_page_size", int),
        operations=operations,
        daily_request_budget_without_credential=None
        if "daily_request_budget_without_credential" not in raw_provider
        else _require_number(
            raw_provider["daily_request_budget_without_credential"],
            f"{label} daily_request_budget_without_credential",
            int,
        ),
    )


def _build_operation_rule(
    provider_name: str, operation: str, raw_operation: Any, provider_credential_required: bool
) -> OperationRule:
    label = f"[providers.{provider_name}.operations.{operation}]"
    raw_operation = _require_table(raw_operation, label)
    _reject_unknown_keys(raw_operation, OPERATION_KEYS, label)
    host = _validate_hostname(raw_operation.get("host"), f"{label} host")
    try:
        path_pattern = re.compile(raw_operation["path_pattern"])
    except (KeyError, TypeError, re.error) as pattern_error:
        raise ConfigurationError(
            f"{label} path_pattern is missing or not a valid regular expression: {pattern_error}"
        ) from None
    return OperationRule(
        provider_name=provider_name,
        operation=operation,
        host=host,
        path_pattern=path_pattern,
        query_parameter_names=frozenset(
            _require_string_list(raw_operation.get("query_parameters", []), f"{label} query_parameters")
        ),
        content_types=tuple(
            content_type.lower()
            for content_type in _require_string_list(raw_operation.get("content_types", []), f"{label} content_types")
        ),
        max_response_bytes=_require_number(raw_operation.get("max_response_bytes"), f"{label} max_response_bytes", int),
        max_decompressed_bytes=_require_number(
            raw_operation.get("max_decompressed_bytes"), f"{label} max_decompressed_bytes", int
        ),
        total_timeout_seconds=_require_number(
            raw_operation.get("total_timeout_seconds"), f"{label} total_timeout_seconds", float
        ),
        redirect_hosts=frozenset(
            _validate_hostname(redirect_host, f"{label} redirect_hosts")
            for redirect_host in _require_string_list(
                raw_operation.get("redirect_hosts", []), f"{label} redirect_hosts"
            )
        ),
        credential_required=bool(raw_operation.get("credential_required", provider_credential_required)),
        request_header_names=frozenset(
            header_name.lower()
            for header_name in _require_string_list(
                raw_operation.get("request_headers", []), f"{label} request_headers"
            )
        ),
        response_header_names=tuple(
            header_name.lower()
            for header_name in _require_string_list(
                raw_operation.get("response_headers", []), f"{label} response_headers"
            )
        ),
        cursor_paging_requires_credential=bool(raw_operation.get("cursor_paging_requires_credential", False)),
        minimum_seconds_between_requests=None
        if "minimum_seconds_between_requests" not in raw_operation
        else _require_number(
            raw_operation["minimum_seconds_between_requests"], f"{label} minimum_seconds_between_requests", float
        ),
    )


def _validate_hostname(raw_host: Any, label: str) -> str:
    """A DNS name only: IP literals are refused, so a rule can never point at a fixed internal address."""
    if not isinstance(raw_host, str) or not HOSTNAME_PATTERN.match(raw_host):
        raise ConfigurationError(f"{label} must be a lowercase DNS host name, got {raw_host!r}")
    try:
        ipaddress.ip_address(raw_host)
    except ValueError:
        return raw_host
    raise ConfigurationError(f"{label} must be a DNS host name, not an IP address")


def _require_table(raw_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        raise ConfigurationError(f"{label} in the provider rules must be a table")
    return raw_value


def _reject_unknown_keys(raw_table: dict[str, Any], allowed_keys: set[str], label: str) -> None:
    unknown_keys = set(raw_table) - allowed_keys
    if unknown_keys:
        raise ConfigurationError(f"unknown key(s) {sorted(unknown_keys)} in {label} of the provider rules")


def _require_string_list(raw_value: Any, label: str) -> list[str]:
    if not isinstance(raw_value, list) or not all(isinstance(item, str) for item in raw_value):
        raise ConfigurationError(f"{label} must be a list of strings")
    return raw_value


def _require_number[NumberT: (int, float)](
    raw_value: Any, label: str, number_type: type[NumberT], *, allow_zero: bool = False
) -> NumberT:
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, (int, float))
        or (number_type is int and not isinstance(raw_value, int))
    ):
        raise ConfigurationError(f"{label} must be a {number_type.__name__}, got {raw_value!r}")
    if raw_value < 0 or (raw_value == 0 and not allow_zero):
        raise ConfigurationError(f"{label} must be positive")
    return number_type(raw_value)
