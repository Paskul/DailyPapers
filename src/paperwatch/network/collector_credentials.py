"""Provider credentials: one collector-only TOML file, never config.toml, never the worker's environment.

File format (default path: <home>/credentials/collector.toml, set by collector.credentials_file):

    [zotero]
    api_key = "..."          # read-only key from zotero.org/settings/keys
    library_type = "user"    # "user" or "group"
    library_id = "1234567"   # numeric id shown on the same page

    [openalex]
    api_key = "..."          # optional for the works API, required for the content (full text) API

Rules:
  * Only the collector command loads this file; the worker, report writer, and model server never do.
  * The file must belong to the running account and grant no group or world access (mode 0600 or 0400),
    otherwise it is refused. A missing file just means "no credentials".
  * Secrets never appear in repr(), logs, or the outbound request log (they are redacted there).
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from paperwatch.errors import ConfigurationError

ALLOWED_CREDENTIAL_FIELDS: dict[str, frozenset[str]] = {
    "zotero": frozenset({"api_key", "library_type", "library_id"}),
    "openalex": frozenset({"api_key"}),
}
SECRET_FIELD_NAMES = frozenset({"api_key"})
REDACTED_VALUE = "[redacted]"
CREDENTIAL_VALUE_PATTERN = re.compile(r"^[\x21-\x7e]{1,256}$")  # printable ASCII, no spaces: safe in a header


@dataclass(frozen=True)
class CollectorCredentials:
    values_by_provider: dict[str, dict[str, str]] = field(default_factory=dict, repr=False)
    source_path: Path | None = None

    def get(self, provider_name: str, field_name: str) -> str | None:
        return self.values_by_provider.get(provider_name, {}).get(field_name)

    def has(self, provider_name: str, field_name: str) -> bool:
        return self.get(provider_name, field_name) is not None

    def all_secret_values(self) -> frozenset[str]:
        """Every secret, so log writers can scrub them from any text as a last resort."""
        return frozenset(
            value
            for provider_values in self.values_by_provider.values()
            for field_name, value in provider_values.items()
            if field_name in SECRET_FIELD_NAMES
        )

    def __repr__(self) -> str:
        configured = {provider: sorted(values) for provider, values in self.values_by_provider.items()}
        return f"CollectorCredentials(source={self.source_path}, fields={configured}, values={REDACTED_VALUE})"


NO_CREDENTIALS = CollectorCredentials()


def load_collector_credentials(credentials_path: Path) -> CollectorCredentials:
    if not credentials_path.exists():
        return CollectorCredentials(source_path=None)
    _require_private_file(credentials_path)
    try:
        with credentials_path.open("rb") as credentials_file:
            raw_credentials = tomllib.load(credentials_file)
    except tomllib.TOMLDecodeError as decode_error:
        raise ConfigurationError(f"credentials file {credentials_path} is not valid TOML: {decode_error}") from None
    except PermissionError:
        raise ConfigurationError(
            f"credentials file {credentials_path} is not readable by this account",
            operator_hint="only the collector account may read it; run `collect` as that account",
        ) from None

    values_by_provider: dict[str, dict[str, str]] = {}
    for provider_name, raw_provider_values in raw_credentials.items():
        if provider_name not in ALLOWED_CREDENTIAL_FIELDS:
            raise ConfigurationError(
                f"unknown provider [{provider_name}] in {credentials_path}; "
                f"allowed: {sorted(ALLOWED_CREDENTIAL_FIELDS)}"
            )
        if not isinstance(raw_provider_values, dict):
            raise ConfigurationError(f"[{provider_name}] in {credentials_path} must be a table")
        unknown_fields = set(raw_provider_values) - ALLOWED_CREDENTIAL_FIELDS[provider_name]
        if unknown_fields:
            raise ConfigurationError(
                f"unknown key(s) {sorted(unknown_fields)} in [{provider_name}] of {credentials_path}"
            )
        for field_name, value in raw_provider_values.items():
            if not isinstance(value, str) or not CREDENTIAL_VALUE_PATTERN.match(value):
                # Never echo the value: it may be a real key with a typo.
                raise ConfigurationError(
                    f"[{provider_name}].{field_name} in {credentials_path} must be a "
                    "non-empty string of printable characters without spaces"
                )
        values_by_provider[provider_name] = dict(raw_provider_values)
    _validate_zotero_library(values_by_provider.get("zotero", {}), credentials_path)
    return CollectorCredentials(values_by_provider=values_by_provider, source_path=credentials_path)


def describe_credentials_file_problem(credentials_path: Path) -> str | None:
    """None if the file is absent or private to its owner; otherwise what is wrong (used by access-check)."""
    try:
        _require_private_file(credentials_path)
    except ConfigurationError as problem:
        return str(problem)
    return None


def _require_private_file(credentials_path: Path) -> None:
    try:
        file_status = credentials_path.stat()
    except PermissionError:
        raise ConfigurationError(f"credentials file {credentials_path} is not accessible to this account") from None
    if not stat.S_ISREG(file_status.st_mode):
        raise ConfigurationError(f"credentials path {credentials_path} is not a regular file")
    if file_status.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigurationError(
            f"credentials file {credentials_path} is accessible to other accounts "
            f"(mode {stat.S_IMODE(file_status.st_mode):04o})",
            operator_hint=f"run: chmod 600 {credentials_path} (as the collector account)",
        )
    if file_status.st_uid != os.geteuid():
        raise ConfigurationError(
            f"credentials file {credentials_path} belongs to another account (uid "
            f"{file_status.st_uid}); it must belong to the collector account"
        )


def _validate_zotero_library(zotero_values: dict[str, str], credentials_path: Path) -> None:
    if not zotero_values:
        return
    if zotero_values.get("library_type", "user") not in ("user", "group"):
        raise ConfigurationError(f'[zotero].library_type in {credentials_path} must be "user" or "group"')
    if "library_id" in zotero_values and not zotero_values["library_id"].isdigit():
        raise ConfigurationError(f"[zotero].library_id in {credentials_path} must be the numeric library id")
