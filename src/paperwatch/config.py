"""Load and validate config.toml and topics.toml into typed, read-only settings objects.

Layout of a paperwatch home directory (relative paths in config.toml resolve against this directory):

    config.toml                       # Paths, model address, limits, schedule
    topics.toml                       # Research questions and public search terms
    state/paperwatch.sqlite3          # Catalog, jobs, reviews, report snapshots
    state/cache/                      # Source bytes and extracted passages
    output/papers/<id>/<revision>.md  # Stable, version-specific reading note
    output/daily/YYYY-MM-DD.md        # One report per reporting date

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import re
import tomllib
import typing
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paperwatch.errors import ConfigurationError

CONFIG_FILE_NAME = "config.toml"
TOPICS_FILE_NAME = "topics.toml"
DATABASE_FILE_NAME = "paperwatch.sqlite3"
INFERENCE_LOCK_FILE_NAME = "inference.lock"

SUPPORTED_MODEL_BACKENDS = ("llama_cpp", "fake")
REASONING_EFFORT_SERVER_DEFAULT = "server-default"
TOPIC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CLOCK_TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
# Public search terms are the only topic text that may leave the machine (plan part 3), so their shape is checked.
MAXIMUM_PUBLIC_QUERY_CHARACTERS = 200
ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z-]*(\.[A-Za-z][A-Za-z-]*)?$")  # cs.RO, math.OC, hep-th, q-bio.NC
ZOTERO_COLLECTION_KEY_PATTERN = re.compile(r"^[A-Z0-9]{8}$")
# The extractor image must be pinned: "name@sha256:<64 hex>" or a full 64-hex image id (content-addressed, and stable
# across `podman save | podman load` into the collector account's storage).
PINNED_IMAGE_REFERENCE_PATTERN = re.compile(r"^([a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}|[0-9a-f]{64})$")


# --------------------------------------------------------------------------------------------------------------------
# Settings sections. Dataclass defaults ARE the documented defaults; the config.toml template mirrors them.
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PathSettings:
    state_dir: str = "state"  # database + cache; keep outside any synced notes folder
    output_dir: str = "output"  # Markdown notes and daily reports; may point into a notes vault


@dataclass(frozen=True)
class ModelSettings:
    backend: str = "llama_cpp"  # "llama_cpp" for the real server, "fake" for offline tests
    base_url: str = "http://127.0.0.1:8080/v1"
    model_alias: str = "qwen-local"
    connect_timeout_seconds: float = 5.0
    required_context_tokens: int = 32768  # doctor fails if the server's per-slot context is smaller
    allow_non_loopback_host: bool = False  # the model must stay local; never a hosted fallback
    outage_backoff_initial_seconds: int = 30
    outage_backoff_max_seconds: int = 1800
    inference_lock_wait_seconds: int = 3600  # how long a role waits for the one-at-a-time inference slot


@dataclass(frozen=True)
class RoleLimits:
    """Hard per-role limits. Reasoning tokens count toward max_output_tokens on llama.cpp."""

    max_output_tokens: int
    timeout_seconds: float
    reasoning_effort: str = REASONING_EFFORT_SERVER_DEFAULT


DEFAULT_ROLE_LIMITS: dict[str, RoleLimits] = {
    "healthcheck": RoleLimits(max_output_tokens=2048, timeout_seconds=120, reasoning_effort="low"),
    "reviewer": RoleLimits(max_output_tokens=16384, timeout_seconds=1800),
}


@dataclass(frozen=True)
class WorkerSettings:
    max_attempts_per_job: int = 3  # for paper-specific transient errors; outages never consume attempts
    retry_delay_seconds: int = 600  # multiplied by the attempt number
    idle_poll_seconds: int = 60  # upper bound on sleep when no job is due (zero inference calls while idle)
    lease_seconds: int = 300  # a claimed job is reclaimable this long after its worker stops renewing
    lease_renewal_interval_seconds: int = 60  # a live worker renews its lease this often during long calls


@dataclass(frozen=True)
class ReportSettings:
    timezone: str = "America/New_York"
    publish_time: str = "07:00"
    max_papers_per_report: int = 15


@dataclass(frozen=True)
class ScheduleSettings:
    collect_times: tuple[str, ...] = ("00:30", "06:30", "12:30", "18:30")


@dataclass(frozen=True)
class CollectorSettings:
    """Settings of the only component that may contact paper providers (plan part 3).

    Host, path, budget, and response-size rules are NOT here: they live in the read-only provider rules that ship
    with the package (paperwatch/network/provider_rules.toml), so editing config.toml cannot widen network access.
    """

    credentials_file: str = "credentials/collector.toml"  # collector-only; mode 0600, never readable by the worker
    user_agent: str = "paperwatch/0.1 (local research monitor)"  # generic: no name, e-mail, or host details


@dataclass(frozen=True)
class ExtractorSettings:
    """The PDF/XML text extractor runs in rootless Podman: no network, read-only root, one read-only input."""

    container_runtime: str = "podman"  # executable name or absolute path
    image: str = ""  # pinned image id printed by deploy/build_extractor_image.sh; empty = not set up yet
    timeout_seconds: float = 120.0
    max_input_megabytes: int = 60
    max_output_megabytes: int = 20
    memory_megabytes: int = 1024
    temporary_space_megabytes: int = 64
    cpu_limit: float = 1.0


@dataclass(frozen=True)
class AccessSettings:
    """Linux accounts the deploy scripts create; `paperwatch access-check` verifies the rules from each one."""

    collector_account: str = "paperwatch-collector"
    worker_account: str = "paperwatch-worker"  # runs `work` and `report`
    shared_state_group: str = "paperwatch-state"  # group owning state/ (SQLite + WAL files)
    model_server_account: str = ""  # empty: llama-server still runs as a desktop user (owner decision, part 3)


@dataclass(frozen=True)
class TopicDefinition:
    """One research question. `question` is private and never leaves the machine; `public_queries` may."""

    topic_id: str
    question: str
    public_queries: tuple[str, ...] = ()
    arxiv_categories: tuple[str, ...] = ()
    zotero_collections: tuple[str, ...] = ()

    def compute_profile_sha256(self) -> str:
        """Fingerprint of what a topic assessment depends on: the topic id and its question (part 4 adds seeds).

        Stored with every assessment, so a changed question is visible as a stale assessment, while the reading
        itself is reused.
        """
        profile_text = json.dumps({"topic_id": self.topic_id, "question": " ".join(self.question.split())},
                                  ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(profile_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AppConfig:
    home_dir: Path
    paths: PathSettings
    model: ModelSettings
    role_limits: dict[str, RoleLimits]
    worker: WorkerSettings
    report: ReportSettings
    schedule: ScheduleSettings
    collector: CollectorSettings
    extractor: ExtractorSettings
    access: AccessSettings
    topics: tuple[TopicDefinition, ...]

    # ---- Derived paths --------------------------------------------------------------------------------------------
    @property
    def state_dir(self) -> Path:
        return _resolve_against_home(self.home_dir, self.paths.state_dir)

    @property
    def output_dir(self) -> Path:
        return _resolve_against_home(self.home_dir, self.paths.output_dir)

    @property
    def database_path(self) -> Path:
        return self.state_dir / DATABASE_FILE_NAME

    @property
    def source_cache_dir(self) -> Path:
        return self.state_dir / "cache"

    @property
    def collector_credentials_path(self) -> Path:
        return _resolve_against_home(self.home_dir, self.collector.credentials_file)

    @property
    def inference_lock_path(self) -> Path:
        return self.state_dir / INFERENCE_LOCK_FILE_NAME

    @property
    def paper_notes_dir(self) -> Path:
        return self.output_dir / "papers"

    @property
    def daily_reports_dir(self) -> Path:
        return self.output_dir / "daily"

    @property
    def report_timezone(self) -> ZoneInfo:
        return ZoneInfo(self.report.timezone)

    def limits_for_role(self, role_name: str) -> RoleLimits:
        if role_name not in self.role_limits:
            raise ConfigurationError(f"no limits configured for model role '{role_name}'",
                                     operator_hint=f"add a [roles.{role_name}] section to config.toml")
        return self.role_limits[role_name]


# --------------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------------


def load_app_config(home_dir: Path) -> AppConfig:
    home_dir = home_dir.expanduser().resolve()
    raw_config = _read_toml_file(home_dir / CONFIG_FILE_NAME)
    raw_topics = _read_toml_file(home_dir / TOPICS_FILE_NAME)

    known_sections = {"paths", "model", "roles", "worker", "report", "schedule", "collector", "extractor", "access"}
    unknown_sections = set(raw_config) - known_sections
    if unknown_sections:
        raise ConfigurationError(f"unknown section(s) {sorted(unknown_sections)} in {CONFIG_FILE_NAME}")

    app_config = AppConfig(
        home_dir=home_dir,
        paths=_build_settings_section(PathSettings, raw_config.get("paths", {}), "paths"),
        model=_build_settings_section(ModelSettings, raw_config.get("model", {}), "model"),
        role_limits=_build_role_limits(raw_config.get("roles", {})),
        worker=_build_settings_section(WorkerSettings, raw_config.get("worker", {}), "worker"),
        report=_build_settings_section(ReportSettings, raw_config.get("report", {}), "report"),
        schedule=_build_settings_section(ScheduleSettings, raw_config.get("schedule", {}), "schedule"),
        collector=_build_settings_section(CollectorSettings, raw_config.get("collector", {}), "collector"),
        extractor=_build_settings_section(ExtractorSettings, raw_config.get("extractor", {}), "extractor"),
        access=_build_settings_section(AccessSettings, raw_config.get("access", {}), "access"),
        topics=_build_topics(raw_topics),
    )
    _validate_cross_field_rules(app_config)
    return app_config


def _read_toml_file(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("rb") as toml_file:
            return tomllib.load(toml_file)
    except PermissionError:
        # Under the service-account layout (plan part 3) only members of the shared state group may read the home.
        raise ConfigurationError(f"{file_path} is not readable by this account",
                                 operator_hint="run as a paperwatch service account, or as a member of the shared "
                                               "state group (after adding yourself, log out and in, or use "
                                               "`sg paperwatch-state -c '...'`)") from None
    except FileNotFoundError:
        raise ConfigurationError(f"{file_path} does not exist",
                                 operator_hint=f"run `paperwatch --home {file_path.parent} init` first") from None
    except IsADirectoryError:
        raise ConfigurationError(f"{file_path} is a directory, not a TOML file") from None
    except tomllib.TOMLDecodeError as decode_error:
        raise ConfigurationError(f"{file_path} is not valid TOML: {decode_error}") from decode_error


def _build_role_limits(raw_roles_section: Any) -> dict[str, RoleLimits]:
    if not isinstance(raw_roles_section, dict):
        raise ConfigurationError("[roles] in config.toml must be a table of [roles.<role_name>] sections")
    merged_role_limits = dict(DEFAULT_ROLE_LIMITS)
    for role_name, raw_role_section in raw_roles_section.items():
        default_limits = DEFAULT_ROLE_LIMITS.get(role_name)
        base_values = dataclasses.asdict(default_limits) if default_limits else {}
        if not isinstance(raw_role_section, dict):
            raise ConfigurationError(f"[roles.{role_name}] in config.toml must be a table")
        merged_role_limits[role_name] = _build_settings_section(
            RoleLimits, {**base_values, **raw_role_section}, f"roles.{role_name}")
    return merged_role_limits


def _build_topics(raw_topics_file: dict[str, Any]) -> tuple[TopicDefinition, ...]:
    unknown_top_level_keys = set(raw_topics_file) - {"topics"}
    if unknown_top_level_keys:
        raise ConfigurationError(f"unknown key(s) {sorted(unknown_top_level_keys)} in {TOPICS_FILE_NAME}; "
                                 "topics are declared as [[topics]] entries")
    raw_topic_entries = raw_topics_file.get("topics", [])
    if not isinstance(raw_topic_entries, list):
        raise ConfigurationError(f"`topics` in {TOPICS_FILE_NAME} must be an array of tables ([[topics]])")

    topics: list[TopicDefinition] = []
    seen_topic_ids: set[str] = set()
    for entry_index, raw_topic_entry in enumerate(raw_topic_entries):
        if not isinstance(raw_topic_entry, dict):
            raise ConfigurationError(f"topic #{entry_index + 1} in {TOPICS_FILE_NAME} must be a table")
        # The file uses the short key `id`; the code uses the descriptive name `topic_id`.
        renamed_entry = {("topic_id" if key == "id" else key): value for key, value in raw_topic_entry.items()}
        topic = _build_settings_section(TopicDefinition, renamed_entry, f"topics #{entry_index + 1}")
        if not TOPIC_ID_PATTERN.match(topic.topic_id):
            raise ConfigurationError(f"topic id '{topic.topic_id}' must match {TOPIC_ID_PATTERN.pattern}")
        if topic.topic_id in seen_topic_ids:
            raise ConfigurationError(f"duplicate topic id '{topic.topic_id}' in {TOPICS_FILE_NAME}")
        if not topic.question.strip():
            raise ConfigurationError(f"topic '{topic.topic_id}' has an empty question")
        _validate_public_topic_terms(topic)
        seen_topic_ids.add(topic.topic_id)
        topics.append(topic)
    return tuple(topics)


def _validate_public_topic_terms(topic: TopicDefinition) -> None:
    """Check the terms the collector may send to providers. The private question is never checked here: it never
    leaves the machine, so it may contain anything."""
    for public_query in topic.public_queries:
        if not public_query.strip() or len(public_query) > MAXIMUM_PUBLIC_QUERY_CHARACTERS or \
                any(not character.isprintable() for character in public_query):
            raise ConfigurationError(
                f"topic '{topic.topic_id}': public query {public_query!r} must be 1-{MAXIMUM_PUBLIC_QUERY_CHARACTERS} "
                "printable characters on one line")
    for arxiv_category in topic.arxiv_categories:
        if not ARXIV_CATEGORY_PATTERN.match(arxiv_category):
            raise ConfigurationError(f"topic '{topic.topic_id}': '{arxiv_category}' is not an arXiv category "
                                     "such as cs.RO or hep-th")
    for collection_key in topic.zotero_collections:
        if not ZOTERO_COLLECTION_KEY_PATTERN.match(collection_key):
            raise ConfigurationError(f"topic '{topic.topic_id}': '{collection_key}' is not a Zotero collection key "
                                     "(8 characters A-Z / 0-9, e.g. from the collection URL)")


def _build_settings_section(settings_class: type, raw_section: Any, section_label: str):
    """Build a frozen settings dataclass from a TOML table, rejecting unknown keys and wrong value types."""
    if not isinstance(raw_section, dict):
        raise ConfigurationError(f"[{section_label}] must be a table")
    field_type_hints = typing.get_type_hints(settings_class)
    known_field_names = {settings_field.name for settings_field in dataclasses.fields(settings_class)}
    unknown_keys = set(raw_section) - known_field_names
    if unknown_keys:
        raise ConfigurationError(f"unknown key(s) {sorted(unknown_keys)} in [{section_label}]; "
                                 f"allowed keys: {sorted(known_field_names)}")
    coerced_values = {key: _coerce_toml_value(value, field_type_hints[key], f"{section_label}.{key}")
                      for key, value in raw_section.items()}
    try:
        return settings_class(**coerced_values)
    except TypeError as construction_error:
        raise ConfigurationError(f"[{section_label}] is missing a required key: {construction_error}") from None


def _coerce_toml_value(raw_value: Any, expected_type: Any, key_label: str) -> Any:
    if expected_type is bool:
        if isinstance(raw_value, bool):
            return raw_value
    elif expected_type is int:
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            return raw_value
    elif expected_type is float:
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            return float(raw_value)
    elif expected_type is str:
        if isinstance(raw_value, str):
            return raw_value
    elif typing.get_origin(expected_type) is tuple:  # only tuple[str, ...] is used
        if isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
            return tuple(raw_value)
    else:
        raise AssertionError(f"unsupported settings type {expected_type} for {key_label}")
    readable_type = "list of strings" if typing.get_origin(expected_type) is tuple else expected_type.__name__
    raise ConfigurationError(f"`{key_label}` must be a {readable_type}, got {raw_value!r}")


def _validate_cross_field_rules(app_config: AppConfig) -> None:
    model_settings = app_config.model
    if model_settings.backend not in SUPPORTED_MODEL_BACKENDS:
        raise ConfigurationError(f"model.backend must be one of {SUPPORTED_MODEL_BACKENDS}, "
                                 f"got '{model_settings.backend}'")
    parsed_model_url = urllib.parse.urlparse(model_settings.base_url)
    if parsed_model_url.scheme not in ("http", "https") or not parsed_model_url.hostname:
        raise ConfigurationError(f"model.base_url must be an http(s) URL, got '{model_settings.base_url}'")
    if not model_settings.allow_non_loopback_host and not _is_loopback_host(parsed_model_url.hostname):
        raise ConfigurationError(
            f"model.base_url host '{parsed_model_url.hostname}' is not a loopback address",
            operator_hint="paperwatch only talks to a local model; set model.allow_non_loopback_host = true "
                          "only for a model server you control on your own network")

    for role_name, role_limits in app_config.role_limits.items():
        if role_limits.max_output_tokens <= 0 or role_limits.timeout_seconds <= 0:
            raise ConfigurationError(f"[roles.{role_name}] limits must be positive")

    try:
        ZoneInfo(app_config.report.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigurationError(f"report.timezone '{app_config.report.timezone}' is not a known IANA timezone")
    for clock_time in (app_config.report.publish_time, *app_config.schedule.collect_times):
        if not CLOCK_TIME_PATTERN.match(clock_time):
            raise ConfigurationError(f"time '{clock_time}' must be in 24-hour HH:MM form")
    if app_config.report.max_papers_per_report <= 0:
        raise ConfigurationError("report.max_papers_per_report must be positive")
    _validate_extractor_settings(app_config.extractor)
    worker_settings = app_config.worker
    if worker_settings.max_attempts_per_job <= 0:
        raise ConfigurationError("worker.max_attempts_per_job must be positive")
    if worker_settings.lease_renewal_interval_seconds <= 0 or \
            worker_settings.lease_seconds < 2 * worker_settings.lease_renewal_interval_seconds:
        raise ConfigurationError(
            "worker.lease_seconds must be at least twice worker.lease_renewal_interval_seconds (and both positive), "
            "so one missed renewal does not let another worker reclaim a live job")


def _validate_extractor_settings(extractor_settings: ExtractorSettings) -> None:
    if extractor_settings.image and not PINNED_IMAGE_REFERENCE_PATTERN.match(extractor_settings.image):
        raise ConfigurationError(
            f"extractor.image '{extractor_settings.image}' is not pinned (a tag such as ':latest' can change)",
            operator_hint="use the 64-hex image id printed by deploy/build_extractor_image.sh, or name@sha256:<digest>")
    numeric_limits = {"timeout_seconds": extractor_settings.timeout_seconds,
                      "max_input_megabytes": extractor_settings.max_input_megabytes,
                      "max_output_megabytes": extractor_settings.max_output_megabytes,
                      "memory_megabytes": extractor_settings.memory_megabytes,
                      "temporary_space_megabytes": extractor_settings.temporary_space_megabytes,
                      "cpu_limit": extractor_settings.cpu_limit}
    for limit_name, limit_value in numeric_limits.items():
        if limit_value <= 0:
            raise ConfigurationError(f"extractor.{limit_name} must be positive")
    if not extractor_settings.container_runtime.strip():
        raise ConfigurationError("extractor.container_runtime must name an executable, e.g. \"podman\"")


def _is_loopback_host(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _resolve_against_home(home_dir: Path, configured_path: str) -> Path:
    candidate_path = Path(configured_path).expanduser()
    return candidate_path if candidate_path.is_absolute() else (home_dir / candidate_path)
