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
import ipaddress
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
    lease_margin_seconds: int = 120  # added on top of role timeout + lock wait when leasing a job


@dataclass(frozen=True)
class ReportSettings:
    timezone: str = "America/New_York"
    publish_time: str = "07:00"
    max_papers_per_report: int = 15


@dataclass(frozen=True)
class ScheduleSettings:
    collect_times: tuple[str, ...] = ("00:30", "06:30", "12:30", "18:30")


@dataclass(frozen=True)
class TopicDefinition:
    """One research question. `question` is private and never leaves the machine; `public_queries` may."""

    topic_id: str
    question: str
    public_queries: tuple[str, ...] = ()
    arxiv_categories: tuple[str, ...] = ()
    zotero_collections: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppConfig:
    home_dir: Path
    paths: PathSettings
    model: ModelSettings
    role_limits: dict[str, RoleLimits]
    worker: WorkerSettings
    report: ReportSettings
    schedule: ScheduleSettings
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

    known_sections = {"paths", "model", "roles", "worker", "report", "schedule"}
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
        topics=_build_topics(raw_topics),
    )
    _validate_cross_field_rules(app_config)
    return app_config


def _read_toml_file(file_path: Path) -> dict[str, Any]:
    if not file_path.is_file():
        raise ConfigurationError(f"{file_path} does not exist",
                                 operator_hint=f"run `paperwatch --home {file_path.parent} init` first")
    try:
        with file_path.open("rb") as toml_file:
            return tomllib.load(toml_file)
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
        seen_topic_ids.add(topic.topic_id)
        topics.append(topic)
    return tuple(topics)


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
    if app_config.worker.max_attempts_per_job <= 0:
        raise ConfigurationError("worker.max_attempts_per_job must be positive")


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
