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
# "onnx" runs the pinned sentence-embedding model in-process on the CPU; "hashing" is a deterministic word-hashing
# stand-in for offline tests (like model.backend = "fake"), never for real screening.
SUPPORTED_EMBEDDING_BACKENDS = ("onnx", "hashing")
REASONING_EFFORT_SERVER_DEFAULT = "server-default"
TOPIC_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CLOCK_TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
# Public search terms are the only topic text that may leave the machine (plan part 3), so their shape is checked.
MAXIMUM_PUBLIC_QUERY_CHARACTERS = 200
ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z-]*(\.[A-Za-z][A-Za-z-]*)?$")  # cs.RO, math.OC, hep-th, q-bio.NC
ZOTERO_COLLECTION_KEY_PATTERN = re.compile(r"^[A-Z0-9]{8}$")
MAXIMUM_SEED_FILE_PATH_CHARACTERS = 400
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
    # Plan part 7 roles. The owner chose the server's default reasoning (xhigh) for every reading call (2026-11-19).
    "section_reader": RoleLimits(max_output_tokens=16384, timeout_seconds=1800),  # one bounded reading section
    "reviewer": RoleLimits(max_output_tokens=16384, timeout_seconds=1800),  # the evidence-backed review
    "topic_assessor": RoleLimits(max_output_tokens=12288, timeout_seconds=1200),  # rubric ratings per topic
    # Plan part 8 roles, also at the server's default reasoning (the same choice as the reading roles).
    "research": RoleLimits(max_output_tokens=8192, timeout_seconds=1200),  # chooses references and gap searches
    "comparison": RoleLimits(max_output_tokens=12288, timeout_seconds=1200),  # compares two saved reviews
}
# Room the reading roles need besides the passages themselves: system prompt, orientation (title + abstract), JSON
# keys. Used by the load-time check that a reading section fits the context budget.
READING_PROMPT_OVERHEAD_TOKENS = 3000
READING_CHARACTERS_PER_TOKEN_ESTIMATE = 3.5  # same estimate as paperwatch.roles.base


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
    # Looking up a work by an identifier that is known ONLY from the private seed library (e.g. a seed's DOI) tells
    # the provider which papers the user collects. Off unless the operator opts in (plan part 4).
    allow_lookup_by_seed_identifier: bool = False


@dataclass(frozen=True)
class CollectionSettings:
    """What one `collect` run may admit and fetch (plan part 5). Starting values; measure them in part 10.

    A "new metadata candidate" is a paper the catalog did not know, found by an arXiv or OpenAlex route. Current
    releases (feeds, missed-release recovery, recent OpenAlex windows) run first; the historical search may use only
    what is left after `current_release_reserved_candidates`, so it can never starve the day's new releases.
    """

    daily_new_candidate_budget: int = 500  # per UTC day, across all topics and routes
    current_release_reserved_candidates: int = 200  # of the daily budget, kept for current releases
    historical_candidates_per_topic: int = 300  # per initial (or materially changed) historical search of a topic
    # Recency-weighted historical windows (paperwatch.collection.recency_schedule): candidates per publication year
    # decay exponentially with age, halving every `historical_half_life_years`, down to the floor year.
    historical_earliest_year: int = 1940
    historical_half_life_years: float = 6.0
    historical_window_count: int = 8  # equal-mass windows: each gets the same share of the topic's budget
    historical_citation_ordering_min_age_years: float = 5.0  # older windows take their most-cited papers first
    historical_pages_per_topic_per_run: int = 6  # bounds one run's time (arXiv needs 3 s between requests)
    pages_per_search_per_run: int = 10  # missed-release recovery and recent OpenAlex windows
    openalex_recent_window_days: int = 14  # recent window; also how far back each sweep re-checks late indexing
    openalex_recent_sweep_interval_hours: int = 12  # a recent OpenAlex sweep per query at most this often
    daily_source_text_budget: int = 30  # full-text DOWNLOADS per UTC day (local files are free)
    local_sources_dir: str = "local_sources"  # PDFs/XML you place here, named by arXiv id or DOI
    # Readings that `collect` queues wait until the same run has acquired their source text (so the worker does not
    # read an abstract seconds before the PDF arrives). If that stage never finishes, they become due after this.
    source_text_hold_minutes: int = 60


@dataclass(frozen=True)
class ReadingSettings:
    """How a promoted paper is read (plan part 7). Starting values; measure them on the workstation in part 10.

    A long text is read in bounded reading sections, one model call each, then synthesized; see
    paperwatch.reading.reading_plan for the order (methods and results first) and the full_text / partial_text rule.
    """

    section_max_characters: int = 24000  # passages per reading section (~7k tokens), leaving room for reasoning
    max_sections_per_paper: int = 10  # reading sections read per paper; more makes the review partial_text
    passage_max_characters: int = 1500  # a longer paragraph is split at sentence ends into several passages
    passage_min_characters: int = 200  # a shorter paragraph is merged with the next one on the same page
    closest_seeds_per_topic: int = 3  # seed papers attached to each topic's assessment as comparison anchors
    comparison_reviews_per_topic: int = 3  # earlier reviews on the same question attached as anchors
    anchor_abstract_characters: int = 800  # each seed anchor's abstract is cut to this length


@dataclass(frozen=True)
class ScreeningSettings:
    """Cheap CPU screening of every candidate and the daily promotion of the best ones to a reading (plan part 6).

    Screening score per topic (0-1) = weighted mean of the AVAILABLE signals (an unavailable one is omitted and the
    weights of the rest are renormalized):
        question similarity   max(0, cosine) between the paper and the topic's question (or its phrases)
        seed similarity       mean max(0, cosine) to the `closest_seed_count` most similar active seeds
        phrase match          fraction of the topic's approved public phrases found verbatim in title + abstract
    Before combining, promotion replaces each signal by its percentile within the topic's pool of unpromoted
    candidates (paperwatch.screening.signal_normalization), so signals and topics share one scale. A paper's screening
    score is its best topic's score; `minimum_screening_score` is therefore a relative position (0.8 = top fifth of the
    topic's pool). Starting values; calibrate in part 10.
    """

    embedding_backend: str = "onnx"
    embedding_model: str = "bge-base-en-v1.5"  # a name from paperwatch/screening/embedding_model_registry.py
    embedding_model_dir: str = "models/bge-base-en-v1.5"  # installed by deploy/install_embedding_model.sh
    embedding_threads: int = 16  # CPU threads per batch (measured: 425 abstracts take 30 s at 16, 60 s at 8)
    embedding_batch_size: int = 32
    question_similarity_weight: float = 0.5
    seed_similarity_weight: float = 0.3
    phrase_match_weight: float = 0.2
    closest_seed_count: int = 3
    daily_promotion_limit: int = 30  # readings promoted per UTC day, across all topics, near-cutoff picks included
    near_cutoff_promotions: int = 3  # of those, the best candidates just below the cutoff (tests what it misses)
    minimum_screening_score: float = 0.0  # pool percentile, 0-1; 0 = rank only, until part 10 calibrates it


@dataclass(frozen=True)
class ReferenceSettings:
    """Following important references of completed reviews (plan part 8). Starting values; measure in part 10.

    After a review, `collect` fetches the paper's outgoing OpenAlex references (and confirms identifiers found in its
    own bibliography with a provider record). The research role then picks up to `references_per_paper` of them that
    the paper's method, theorem, data, or baseline depends on; each pick becomes a class-3 reading. Owner decisions
    (2026-11-20): reference readings have their own daily limit on top of screening's 30; they wait for the next
    `collect` to fetch their full text; only reviews that address part of some topic's question are followed.
    """

    follow_references: bool = True
    max_citation_hops: int = 2  # a reference of a reference is hop 2; references of hop-2 papers are not followed
    references_per_paper: int = 3  # readings the research role may choose per parent paper (at most 3)
    daily_reference_readings: int = 20  # newly admitted reference readings per UTC day, on top of screening's limit
    minimum_parent_question_fit: int = 2  # follow a review's references only if it rated question_fit >= this
    reference_fetches_per_run: int = 10  # reviewed papers whose references one `collect` run fetches
    max_references_fetched_per_paper: int = 200  # OpenAlex ids looked up per paper (100 per request)
    bibliography_lookups_per_paper: int = 10  # identifiers from the paper's own bibliography confirmed per paper
    candidates_offered_per_selection: int = 30  # cited works shown to the research role, best screening score first
    reference_reading_hold_minutes: int = 360  # a chosen reference waits this long for `collect` to fetch its text
    research_branches_per_day: int = 3  # gap searches (unused approved query + year window) per UTC day
    branches_offered_per_selection: int = 12  # unused branches shown to the research role


@dataclass(frozen=True)
class ComparisonSettings:
    """Comparisons of two completed reviews that address the same question (plan part 8)."""

    daily_comparison_jobs: int = 5  # new comparison jobs per UTC day
    minimum_shared_evidence_dimensions: int = 1  # of assumptions / guarantees / evaluation setting
    sweep_reviews_completed_within_days: int = 14  # `collect` retries queueing for recent reviews the cap held back


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
    """One research interest (plan part 4). The question is the unit of relevance.

    Relevance inputs (at least one should be present; `status` explains what is missing):
      question             PRIVATE sentence for the local model; never leaves the machine. Optional in seeds-only mode.
      public_queries       phrases the collector MAY send to providers (keyword mode).
      zotero_collections   Zotero collection keys whose items are seed papers (live Web API).
      csl_json_files       CSL JSON exports whose items are seed papers (offline). Equal in standing to Zotero.
    Search inputs: `public_queries` and `arxiv_categories` are the only topic terms that may leave the machine.
    """

    topic_id: str
    question: str = ""
    public_queries: tuple[str, ...] = ()
    arxiv_categories: tuple[str, ...] = ()
    zotero_collections: tuple[str, ...] = ()
    csl_json_files: tuple[str, ...] = ()

    @property
    def has_written_question(self) -> bool:
        return bool(self.question.strip())

    @property
    def has_keyword_input(self) -> bool:
        """Keyword mode: the topic has a private question or approved public phrases to judge relevance against."""
        return self.has_written_question or bool(self.public_queries)

    @property
    def has_configured_seed_sources(self) -> bool:
        return bool(self.zotero_collections or self.csl_json_files)

    @property
    def question_for_local_model(self) -> str:
        """The question as given to the LOCAL model. A seeds-only topic has none, so describe what defines it."""
        if self.has_written_question:
            return self.question
        if self.public_queries:
            return f"(no written question; the topic is defined by these terms: {'; '.join(self.public_queries)})"
        return "(no written question; the topic is defined by the seed papers in the user's library)"

    def compute_profile_sha256(self, seed_fingerprints: tuple[str, ...] = ()) -> str:
        """Fingerprint of what a topic assessment depends on: the topic id, its question, and its active seeds.

        `seed_fingerprints` are "<seed id>:<metadata hash>" strings (paperwatch.research_profile builds them). Stored
        with every assessment, so a changed question or seed set is visible as a stale assessment while the reading
        itself is reused. A topic without seeds keeps the hash it had before seeds existed (plan part 2).
        """
        profile_fields: dict[str, Any] = {"topic_id": self.topic_id, "question": " ".join(self.question.split())}
        if seed_fingerprints:
            profile_fields["seed_fingerprints"] = sorted(seed_fingerprints)
        profile_text = json.dumps(profile_fields, ensure_ascii=False, sort_keys=True)
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
    collection: CollectionSettings
    screening: ScreeningSettings
    reading: ReadingSettings
    extractor: ExtractorSettings
    access: AccessSettings
    topics: tuple[TopicDefinition, ...]
    references: ReferenceSettings = ReferenceSettings()
    comparisons: ComparisonSettings = ComparisonSettings()

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
    def local_sources_dir(self) -> Path:
        return _resolve_against_home(self.home_dir, self.collection.local_sources_dir)

    @property
    def embedding_model_dir(self) -> Path:
        return _resolve_against_home(self.home_dir, self.screening.embedding_model_dir)

    def resolve_seed_file_path(self, configured_seed_file_path: str) -> Path:
        """A topic's `csl_json_files` entry as a path (relative entries resolve against the home directory)."""
        return _resolve_against_home(self.home_dir, configured_seed_file_path)

    @property
    def inference_lock_path(self) -> Path:
        return self.state_dir / INFERENCE_LOCK_FILE_NAME

    @property
    def paper_notes_dir(self) -> Path:
        return self.output_dir / "papers"

    @property
    def comparison_notes_dir(self) -> Path:
        return self.output_dir / "comparisons"

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

    known_sections = {"paths", "model", "roles", "worker", "report", "schedule", "collector", "collection",
                      "screening", "reading", "extractor", "access", "references", "comparisons"}
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
        collection=_build_settings_section(CollectionSettings, raw_config.get("collection", {}), "collection"),
        screening=_build_settings_section(ScreeningSettings, raw_config.get("screening", {}), "screening"),
        reading=_build_settings_section(ReadingSettings, raw_config.get("reading", {}), "reading"),
        extractor=_build_settings_section(ExtractorSettings, raw_config.get("extractor", {}), "extractor"),
        access=_build_settings_section(AccessSettings, raw_config.get("access", {}), "access"),
        topics=_build_topics(raw_topics),
        references=_build_settings_section(ReferenceSettings, raw_config.get("references", {}), "references"),
        comparisons=_build_settings_section(ComparisonSettings, raw_config.get("comparisons", {}), "comparisons"),
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
        # An empty question is allowed (seeds-only mode, plan part 4); `status` explains a topic with no input at all.
        _validate_public_topic_terms(topic)
        _validate_seed_file_paths(topic)
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


def _validate_seed_file_paths(topic: TopicDefinition) -> None:
    """CSL JSON seed files are read locally by the collector; the path itself never leaves the machine."""
    for seed_file_path in topic.csl_json_files:
        if not seed_file_path.strip() or len(seed_file_path) > MAXIMUM_SEED_FILE_PATH_CHARACTERS or \
                any(not character.isprintable() for character in seed_file_path):
            raise ConfigurationError(f"topic '{topic.topic_id}': CSL JSON file path {seed_file_path!r} must be "
                                     f"1-{MAXIMUM_SEED_FILE_PATH_CHARACTERS} printable characters")


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
    _validate_collection_settings(app_config.collection)
    _validate_screening_settings(app_config.screening)
    _validate_reading_settings(app_config)
    _validate_reference_and_comparison_settings(app_config)
    worker_settings = app_config.worker
    if worker_settings.max_attempts_per_job <= 0:
        raise ConfigurationError("worker.max_attempts_per_job must be positive")
    if worker_settings.lease_renewal_interval_seconds <= 0 or \
            worker_settings.lease_seconds < 2 * worker_settings.lease_renewal_interval_seconds:
        raise ConfigurationError(
            "worker.lease_seconds must be at least twice worker.lease_renewal_interval_seconds (and both positive), "
            "so one missed renewal does not let another worker reclaim a live job")


def _validate_collection_settings(collection_settings: CollectionSettings) -> None:
    non_negative_limits = {name: value for name, value in dataclasses.asdict(collection_settings).items()
                           if isinstance(value, int)}
    for limit_name, limit_value in non_negative_limits.items():
        if limit_value < 0:
            raise ConfigurationError(f"collection.{limit_name} must not be negative")
    if collection_settings.current_release_reserved_candidates > collection_settings.daily_new_candidate_budget:
        raise ConfigurationError("collection.current_release_reserved_candidates cannot exceed "
                                 "collection.daily_new_candidate_budget")
    if not 1800 <= collection_settings.historical_earliest_year:
        raise ConfigurationError("collection.historical_earliest_year must be 1800 or later")
    if collection_settings.historical_half_life_years <= 0 or not 1 <= collection_settings.historical_window_count \
            <= 50 or collection_settings.historical_citation_ordering_min_age_years < 0:
        raise ConfigurationError("collection.historical_half_life_years must be positive, "
                                 "collection.historical_window_count between 1 and 50, and "
                                 "collection.historical_citation_ordering_min_age_years not negative")
    if not collection_settings.local_sources_dir.strip():
        raise ConfigurationError("collection.local_sources_dir must name a directory")


def _validate_screening_settings(screening_settings: ScreeningSettings) -> None:
    if screening_settings.embedding_backend not in SUPPORTED_EMBEDDING_BACKENDS:
        raise ConfigurationError(f"screening.embedding_backend must be one of {SUPPORTED_EMBEDDING_BACKENDS}, "
                                 f"got '{screening_settings.embedding_backend}'")
    if not screening_settings.embedding_model_dir.strip():
        raise ConfigurationError("screening.embedding_model_dir must name a directory")
    signal_weights = (screening_settings.question_similarity_weight, screening_settings.seed_similarity_weight,
                      screening_settings.phrase_match_weight)
    if any(weight < 0 for weight in signal_weights) or sum(signal_weights) <= 0:
        raise ConfigurationError("the screening signal weights must not be negative, and at least one must be "
                                 "positive")
    if screening_settings.embedding_threads <= 0 or screening_settings.embedding_batch_size <= 0 or \
            screening_settings.closest_seed_count <= 0:
        raise ConfigurationError("screening.embedding_threads, embedding_batch_size, and closest_seed_count must be "
                                 "positive")
    if screening_settings.daily_promotion_limit < 0 or screening_settings.near_cutoff_promotions < 0 or \
            screening_settings.near_cutoff_promotions > screening_settings.daily_promotion_limit:
        raise ConfigurationError("screening.daily_promotion_limit must not be negative, and "
                                 "screening.near_cutoff_promotions must lie between 0 and the daily limit")
    if not 0.0 <= screening_settings.minimum_screening_score <= 1.0:
        raise ConfigurationError("screening.minimum_screening_score must lie between 0 and 1")


def _validate_reading_settings(app_config: AppConfig) -> None:
    reading_settings = app_config.reading
    for setting_name, setting_value in dataclasses.asdict(reading_settings).items():
        if setting_value <= 0:
            raise ConfigurationError(f"reading.{setting_name} must be positive")
    if not reading_settings.passage_min_characters <= reading_settings.passage_max_characters \
            <= reading_settings.section_max_characters:
        raise ConfigurationError("reading.passage_min_characters <= passage_max_characters <= "
                                 "section_max_characters must hold")
    # Plan part 7: reserve context space for reasoning and the final answer. One reading section plus the prompt
    # around it plus the section reader's whole output limit must fit the context paperwatch may rely on.
    section_prompt_tokens = READING_PROMPT_OVERHEAD_TOKENS + int(
        reading_settings.section_max_characters / READING_CHARACTERS_PER_TOKEN_ESTIMATE)
    for reading_role_name in ("section_reader", "reviewer"):
        reserved_output_tokens = app_config.role_limits[reading_role_name].max_output_tokens
        if section_prompt_tokens + reserved_output_tokens > app_config.model.required_context_tokens:
            raise ConfigurationError(
                f"a reading section of {reading_settings.section_max_characters} characters (about "
                f"{section_prompt_tokens} prompt tokens) plus [roles.{reading_role_name}] max_output_tokens = "
                f"{reserved_output_tokens} exceeds model.required_context_tokens = "
                f"{app_config.model.required_context_tokens}",
                operator_hint="lower reading.section_max_characters or the role's max_output_tokens, or raise "
                              "model.required_context_tokens if the server's per-slot context allows it")


MAXIMUM_REFERENCES_PER_PAPER = 3  # plan part 8: "choose up to three"; the research role's schema allows no more
MAXIMUM_REFERENCES_PER_LOOKUP_REQUEST = 100  # OpenAlex: up to 100 ids in one filter, 100 results per page


def _validate_reference_and_comparison_settings(app_config: AppConfig) -> None:
    for section_name, settings in (("references", app_config.references), ("comparisons", app_config.comparisons)):
        for setting_name, setting_value in dataclasses.asdict(settings).items():
            if isinstance(setting_value, int) and not isinstance(setting_value, bool) and setting_value < 0:
                raise ConfigurationError(f"{section_name}.{setting_name} must not be negative")
    reference_settings = app_config.references
    if not 1 <= reference_settings.references_per_paper <= MAXIMUM_REFERENCES_PER_PAPER:
        raise ConfigurationError(f"references.references_per_paper must lie between 1 and "
                                 f"{MAXIMUM_REFERENCES_PER_PAPER}")
    if not 0 <= reference_settings.minimum_parent_question_fit <= 4:
        raise ConfigurationError("references.minimum_parent_question_fit is a rubric rating between 0 and 4")
    if reference_settings.candidates_offered_per_selection == 0:
        raise ConfigurationError("references.candidates_offered_per_selection must be positive")
    if app_config.comparisons.minimum_shared_evidence_dimensions > 3:
        raise ConfigurationError("comparisons.minimum_shared_evidence_dimensions counts assumptions, guarantees and "
                                 "evaluation setting, so it cannot exceed 3")


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
