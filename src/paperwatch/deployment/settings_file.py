"""The ONE settings file the operator edits: settings/paperwatch.toml.

It holds three things, in plain TOML:
  [setup]        how and where to install: the home directory, the mode (hardened or local), which optional steps
                 to run, the daily schedule, and the monitor. Read only by ./setup.sh.
  config tables  every [paths], [model], [roles.*], ... table of config.toml (see the package template).
  [[topics]]     the research topics of topics.toml.
API keys stay in their own private file (settings/collector_credentials.toml, mode 0600), never in this one.

./setup.sh calls `python -m paperwatch.deployment.settings_file stage <settings file> <stage dir>`, which
  1. splits the file by its top-level tables into <stage dir>/config.toml and <stage dir>/topics.toml, keeping every
     comment, and proves the split exact (the two parts parse back to precisely the original tables);
  2. validates the result with the application's own loader (unknown keys, bad values, cross-field rules), checks
     that every CSL JSON seed file exists, and that Zotero topics have a key in the credentials file;
  3. writes <stage dir>/setup.env: the [setup] values and a few config values as shell-quoted assignments.
Nothing is installed by this module; it only reads the settings and writes into the stage directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shlex
import sys
import tomllib
import urllib.parse
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from paperwatch.config import CONFIG_FILE_NAME, TOPICS_FILE_NAME, load_app_config
from paperwatch.errors import ConfigurationError, PaperwatchError, describe_failure_for_operator
from paperwatch.monitor.monitor_server import DEFAULT_LISTEN_ADDRESS, DEFAULT_PORT, validate_listen_address
from paperwatch.network.collector_credentials import load_collector_credentials

SETTINGS_FILE_NAME = "paperwatch.toml"
SETUP_ENV_FILE_NAME = "setup.env"
SEEDS_DIR_NAME = "seeds"  # settings/seeds/<file>.json is installed as <home>/seeds/<file>.json
SETUP_SECTION, TOPICS_SECTION = "setup", "topics"
MODES = ("hardened", "local")

# A top-level table header: [name], [name.sub], or [[name]] (an array of tables), optionally with a comment.
TABLE_HEADER_PATTERN = re.compile(r"^\s*(\[\[?)\s*([A-Za-z0-9_\-]+)(?:\.[A-Za-z0-9_.\-]+)?\s*(\]\]?)\s*(#.*)?$")


@dataclass(frozen=True)
class SetupSettings:
    """The [setup] table. Defaults are the documented defaults of the example file."""
    home: str = "/srv/paperwatch"
    mode: str = "hardened"
    credentials_file: str = "collector_credentials.toml"
    install_embedding_model: bool = True
    build_extractor_image: bool = True
    daily_schedule: bool = False
    monitor_service: bool = False
    monitor_port: int = DEFAULT_PORT
    monitor_listen_address: str = DEFAULT_LISTEN_ADDRESS


@dataclass(frozen=True)
class SplitSettings:
    setup: SetupSettings
    config_text: str
    topics_text: str


# --------------------------------------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------------------------------------


def split_settings_text(settings_text: str, source_label: str = SETTINGS_FILE_NAME) -> SplitSettings:
    try:
        whole = tomllib.loads(settings_text)
    except tomllib.TOMLDecodeError as decode_error:
        raise ConfigurationError(f"{source_label} is not valid TOML: {decode_error}") from None
    misplaced_keys = [key for key, value in whole.items() if not isinstance(value, (dict, list))]
    if misplaced_keys:
        raise ConfigurationError(f"{source_label}: key(s) {misplaced_keys} are outside any [table]; every setting "
                                 "belongs to a table such as [setup], [model], or [[topics]]")

    buckets: dict[str, list[str]] = {"preamble": [], SETUP_SECTION: [], "config": [], TOPICS_SECTION: []}
    current_bucket = "preamble"
    for line in settings_text.splitlines():
        header_match = TABLE_HEADER_PATTERN.match(line)
        if header_match and (header_match.group(1) == "[[") == (header_match.group(3) == "]]"):
            table_name = header_match.group(2)
            next_bucket = table_name if table_name in (SETUP_SECTION, TOPICS_SECTION) else "config"
            if next_bucket != current_bucket:
                # Comments (and blank lines) right above a header describe that header: move them along with it.
                trailing_lines = _pop_trailing_comment_block(buckets[current_bucket])
                buckets[next_bucket].extend(trailing_lines)
                current_bucket = next_bucket
        buckets[current_bucket].append(line)

    setup_text = "\n".join(buckets[SETUP_SECTION]) + "\n"
    config_text = "\n".join(_strip_blank_edges(buckets["config"])) + "\n"
    topics_text = "\n".join(_strip_blank_edges(buckets[TOPICS_SECTION])) + "\n"
    expected_config = {key: value for key, value in whole.items() if key not in (SETUP_SECTION, TOPICS_SECTION)}
    expected_topics = {TOPICS_SECTION: whole[TOPICS_SECTION]} if TOPICS_SECTION in whole else {}
    try:
        exact = (tomllib.loads(config_text) == expected_config and tomllib.loads(topics_text) == expected_topics
                 and tomllib.loads(setup_text).get(SETUP_SECTION, {}) == whole.get(SETUP_SECTION, {}))
    except tomllib.TOMLDecodeError:
        exact = False
    if not exact:
        raise ConfigurationError(
            f"{source_label} could not be split into config.toml and topics.toml exactly",
            operator_hint="keep each table's keys directly under its [header] (no dotted keys that reach into "
                          "another table, no table headers inside multi-line strings)")
    return SplitSettings(setup=parse_setup_table(whole.get(SETUP_SECTION, {}), source_label),
                         config_text=config_text, topics_text=topics_text)


def _pop_trailing_comment_block(lines: list[str]) -> list[str]:
    moved: list[str] = []
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
        moved.insert(0, lines.pop())
    while moved and not moved[0].strip():  # leave the separating blank line where it was
        lines.append(moved.pop(0))
    return moved


def _strip_blank_edges(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def parse_setup_table(raw_setup: Any, source_label: str = SETTINGS_FILE_NAME) -> SetupSettings:
    if not isinstance(raw_setup, dict):
        raise ConfigurationError(f"[setup] in {source_label} must be a table")
    known_fields = {field.name: field for field in dataclasses.fields(SetupSettings)}
    unknown_keys = sorted(set(raw_setup) - set(known_fields))
    if unknown_keys:
        raise ConfigurationError(f"unknown key(s) {unknown_keys} in [setup] of {source_label}",
                                 operator_hint=f"known keys: {', '.join(known_fields)}")
    for key, value in raw_setup.items():
        expected_type = {"bool": bool, "int": int, "str": str}[str(known_fields[key].type)]
        if type(value) is not expected_type:
            raise ConfigurationError(f"[setup] {key} in {source_label} must be a {expected_type.__name__}, "
                                     f"not {value!r}")
    setup = SetupSettings(**raw_setup)
    if setup.mode not in MODES:
        raise ConfigurationError(f"[setup] mode must be one of {list(MODES)}, not '{setup.mode}'")
    if not Path(setup.home).is_absolute():
        raise ConfigurationError(f"[setup] home must be an absolute path, not '{setup.home}'")
    if setup.mode == "hardened" and Path(setup.home).resolve().is_relative_to(Path.home().resolve()):
        raise ConfigurationError(f"[setup] home {setup.home} is inside your home directory, which the service "
                                 "accounts cannot enter", operator_hint="use a path such as /srv/paperwatch, or "
                                                                         "mode = \"local\"")
    if not 1 <= setup.monitor_port <= 65535:
        raise ConfigurationError(f"[setup] monitor_port must be 1..65535, not {setup.monitor_port}")
    validate_listen_address(setup.monitor_listen_address)
    return setup


# --------------------------------------------------------------------------------------------------------------------
# Staging (what ./setup.sh calls)
# --------------------------------------------------------------------------------------------------------------------


def stage_settings(settings_path: Path, stage_dir: Path, *, extractor_image: str | None = None) -> list[str]:
    """Split, set the extractor image if given, validate, and write config.toml, topics.toml, and setup.env into
    `stage_dir`. Returns lines for the operator. Raises ConfigurationError on any problem."""
    settings_dir = settings_path.resolve().parent
    split = split_settings_text(settings_path.read_text(encoding="utf-8"), str(settings_path))
    config_text = split.config_text
    if extractor_image:
        config_text = set_extractor_image(config_text, extractor_image)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / CONFIG_FILE_NAME).write_text(config_text, encoding="utf-8")
    (stage_dir / TOPICS_FILE_NAME).write_text(split.topics_text, encoding="utf-8")

    app_config = load_app_config(stage_dir)  # the application's own validation
    credentials_path = _resolve_relative(settings_dir, split.setup.credentials_file)
    credentials = None
    if credentials_path is not None and credentials_path.exists():
        credentials = load_collector_credentials(credentials_path)
    problems = []
    for topic in app_config.topics:
        for seed_file in topic.csl_json_files:
            seed_path = Path(seed_file)
            if not seed_path.is_absolute() and not (settings_dir / seed_path).is_file():
                problems.append(f"topic '{topic.topic_id}': CSL JSON file {seed_file} is not in "
                                f"{settings_dir} (expected {settings_dir / seed_path})")
    if any(topic.zotero_collections for topic in app_config.topics):
        if credentials is None:
            problems.append(f"topics use zotero_collections, but there is no credentials file "
                            f"({credentials_path or 'credentials_file is empty'})")
        elif not (credentials.has("zotero", "api_key") and credentials.has("zotero", "library_id")):
            problems.append(f"topics use zotero_collections, but {credentials_path} lacks [zotero] api_key or "
                            "library_id")
    if problems:
        raise ConfigurationError("; ".join(problems))

    seeds_dir = settings_dir / SEEDS_DIR_NAME
    parsed_model_url = urllib.parse.urlparse(app_config.model.base_url)
    shell_values = {
        "PAPERWATCH_HOME": split.setup.home,
        "SETUP_MODE": split.setup.mode,
        "CREDENTIALS_FILE": str(credentials_path) if credentials is not None else "",
        "SEEDS_DIR": str(seeds_dir) if seeds_dir.is_dir() and any(seeds_dir.iterdir()) else "",
        "INSTALL_EMBEDDING_MODEL": _flag(split.setup.install_embedding_model
                                         and app_config.screening.embedding_backend == "onnx"),
        "EMBEDDING_MODEL_NAME": app_config.screening.embedding_model,
        "BUILD_EXTRACTOR_IMAGE": _flag(split.setup.build_extractor_image),
        "CONFIGURED_EXTRACTOR_IMAGE": app_config.extractor.image,
        "DAILY_SCHEDULE": _flag(split.setup.daily_schedule),
        "MONITOR_SERVICE": _flag(split.setup.monitor_service),
        "MONITOR_PORT": str(split.setup.monitor_port),
        "MONITOR_LISTEN_ADDRESS": split.setup.monitor_listen_address,
        "MODEL_PORT": str(parsed_model_url.port or (443 if parsed_model_url.scheme == "https" else 80)),
        "COLLECTOR_ACCOUNT": app_config.access.collector_account,
        "WORKER_ACCOUNT": app_config.access.worker_account,
        "SHARED_GROUP": app_config.access.shared_state_group,
        "TOPIC_IDS": " ".join(topic.topic_id for topic in app_config.topics),
    }
    (stage_dir / SETUP_ENV_FILE_NAME).write_text(
        "".join(f"{name}={shlex.quote(value)}\n" for name, value in shell_values.items()), encoding="utf-8")
    return [f"settings valid: {len(app_config.topics)} topic(s) ({shell_values['TOPIC_IDS'] or 'none'}), mode "
            f"{split.setup.mode}, home {split.setup.home}"]


def set_extractor_image(config_text: str, image_id: str) -> str:
    """Put `image = "<id>"` into [extractor] (replacing an existing value, or adding the key or the table)."""
    if not re.fullmatch(r"[0-9a-f]{64}", image_id):
        raise ConfigurationError(f"extractor image id must be 64 hex characters, not '{image_id}'")
    lines = config_text.splitlines()
    in_extractor = False
    for index, line in enumerate(lines):
        header_match = TABLE_HEADER_PATTERN.match(line)
        if header_match:
            if in_extractor:  # [extractor] had no image key: add it at the end of the table
                lines.insert(index, f'image = "{image_id}"')
                return "\n".join(lines) + "\n"
            in_extractor = line.strip().split("#")[0].strip() == "[extractor]"
            continue
        if in_extractor and re.match(r"^\s*image\s*=", line):
            comment = line[line.index("#"):] if "#" in line.split("=", 1)[1] else ""
            lines[index] = f'image = "{image_id}"' + (f"   {comment}" if comment else "")
            return "\n".join(lines) + "\n"
    if in_extractor:
        return "\n".join(lines + [f'image = "{image_id}"']) + "\n"
    return "\n".join(lines + ["", "[extractor]", f'image = "{image_id}"']) + "\n"


def _resolve_relative(base_dir: Path, configured_path: str) -> Path | None:
    if not configured_path:
        return None
    path = Path(configured_path).expanduser()
    return path if path.is_absolute() else base_dir / path


def _flag(value: bool) -> str:
    return "1" if value else "0"


# --------------------------------------------------------------------------------------------------------------------
# The example file, and migrating the former split settings
# --------------------------------------------------------------------------------------------------------------------

SETTINGS_FILE_HEADER = """\
# paperwatch settings: the ONE file you edit. Installation ([setup]), configuration (every other table), and research
# topics ([[topics]]) live here. Apply it with:   ./setup.sh
# (re-run after every change; it is safe to repeat). ./setup.sh --check validates without changing anything.
#
# This file holds your private research questions, so settings/paperwatch.toml is gitignored. API keys do NOT go here:
# they stay in settings/collector_credentials.toml (mode 0600), installed where only the collector can read them.
"""

SETUP_TABLE_TEXT = """\
[setup]
home = "/srv/paperwatch"          # the paperwatch home: database (state/), notes and reports (output/), models/
mode = "hardened"                 # "hardened": app installed read-only in /opt/paperwatch, separate collector and
                                  #   worker accounts, firewall rules, sandboxed services (asks for sudo);
                                  # "local": everything runs as you from this repository, no sudo (home may then be
                                  #   anywhere, e.g. "/home/<you>/paperwatch")
credentials_file = "collector_credentials.toml"   # relative to this file; "" = no API keys (seeds from CSL JSON only)
install_embedding_model = true    # download the pinned screening model (about 450 MB, hash-checked) if missing
build_extractor_image = true      # build the sandboxed PDF extractor with Podman if [extractor] image is empty or
                                  #   missing; its id is filled in automatically. Without it, papers are read from
                                  #   their abstracts only.
daily_schedule = false            # true: recurring systemd timers run collect at [schedule] collect_times, keep the
                                  #   worker running, and publish the report at [report] publish_time, every day.
                                  #   false: nothing runs on its own (setup removes the timers if they exist).
monitor_service = false           # true: keep `paperwatch monitor` running as your user service (starts at boot)
monitor_port = 8765               # the monitor page: http://127.0.0.1:8765 (from another machine: an SSH tunnel,
monitor_listen_address = "127.0.0.1"  # ssh -L 8765:127.0.0.1:8765 <host>, or this machine's Tailscale address)
"""

TOPICS_HEADER_COMMENT = """\
# ===================================================================================================================
# Research topics. Add one [[topics]] block per research interest; all are active in every run.
# A topic can be defined by keywords, by seed papers, or by both:
#   id                 a short name (letters, digits, - and _)
#   question           PRIVATE. A full sentence for the local model only; never sent to any paper provider.
#                      Optional for a seeds-only topic.
#   public_queries     search terms the collector MAY send to arXiv / OpenAlex
#   arxiv_categories   public subject categories to watch
#   zotero_collections Zotero collection keys (8 characters, from zotero.org/<user>/collections/<KEY>); needs the
#                      [zotero] key in the credentials file
#   csl_json_files     CSL JSON exports (Zotero: Export Collection... -> CSL JSON) put in settings/seeds/ and listed
#                      as "seeds/<file>.json"
# ===================================================================================================================
"""

EXAMPLE_TOPICS_TEXT = """\
[[topics]]
id = "team-safety"
question = "Can a runtime monitor reject a joint robot plan before it blocks a human exit?"
public_queries = ["multi-robot runtime assurance", "multi-agent plan monitoring",
                  "runtime verification mobile robots"]
arxiv_categories = ["cs.RO", "cs.MA", "cs.LO"]
zotero_collections = []          # e.g. ["ABCD2345"]
csl_json_files = []              # e.g. ["seeds/team-safety.json"]

# Seeds only: no question; the topic is defined by the papers in its collection.
# [[topics]]
# id = "library-verification"
# arxiv_categories = ["cs.LO", "cs.PL"]
# zotero_collections = ["EFGH6789"]
"""

CONFIG_DIVIDER = """\
# ===================================================================================================================
# Configuration (installed as <home>/config.toml). Every value is the built-in default; delete a line to keep it.
# ===================================================================================================================
"""


def _template_config_body() -> str:
    """The package's config template without its two-line file preamble."""
    template_text = resources.files("paperwatch.templates").joinpath(CONFIG_FILE_NAME).read_text("utf-8")
    first_table = template_text.index("\n[") + 1
    return template_text[first_table:]


def build_example_settings_text() -> str:
    return "\n".join([SETTINGS_FILE_HEADER, SETUP_TABLE_TEXT, CONFIG_DIVIDER + _template_config_body().rstrip(),
                      "", TOPICS_HEADER_COMMENT + EXAMPLE_TOPICS_TEXT])


def build_migrated_settings_text(config_text: str, topics_text: str, *, home: str, mode: str) -> str:
    """One settings file from a former settings/config.toml + settings/topics.toml pair (contents unchanged)."""
    setup_text = SETUP_TABLE_TEXT.replace('home = "/srv/paperwatch"', f'home = "{home}"', 1)
    setup_text = setup_text.replace('mode = "hardened"', f'mode = "{mode}"', 1)
    config_body = config_text[config_text.index("\n[") + 1:] if not config_text.startswith("[") else config_text
    topic_lines = [line for line in topics_text.splitlines()]
    first_topic = next((index for index, line in enumerate(topic_lines) if line.strip().startswith("[[topics]]")),
                       len(topic_lines))
    return "\n".join([SETTINGS_FILE_HEADER, setup_text, CONFIG_DIVIDER + config_body.rstrip(), "",
                      TOPICS_HEADER_COMMENT + "\n".join(topic_lines[first_topic:]).rstrip() + "\n"])


# --------------------------------------------------------------------------------------------------------------------
# Command line (used by ./setup.sh)
# --------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split and validate settings/paperwatch.toml for ./setup.sh.")
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage", help="split + validate into a stage directory, write setup.env")
    stage_parser.add_argument("settings_path", type=Path)
    stage_parser.add_argument("stage_dir", type=Path)
    stage_parser.add_argument("--extractor-image", default=None, help="64-hex image id to put in [extractor] image")
    example_parser = commands.add_parser("write-example", help="write the example settings file")
    example_parser.add_argument("output_path", type=Path)
    migrate_parser = commands.add_parser("migrate", help="combine a former config.toml + topics.toml into one file")
    migrate_parser.add_argument("config_path", type=Path)
    migrate_parser.add_argument("topics_path", type=Path)
    migrate_parser.add_argument("output_path", type=Path)
    migrate_parser.add_argument("--home", default="/srv/paperwatch")
    migrate_parser.add_argument("--mode", default="hardened", choices=MODES)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "stage":
            for line in stage_settings(arguments.settings_path, arguments.stage_dir,
                                       extractor_image=arguments.extractor_image):
                print(line)
        elif arguments.command == "write-example":
            arguments.output_path.write_text(build_example_settings_text(), encoding="utf-8")
        else:
            if arguments.output_path.exists():
                raise ConfigurationError(f"{arguments.output_path} already exists; not overwritten")
            migrated_text = build_migrated_settings_text(
                arguments.config_path.read_text(encoding="utf-8"), arguments.topics_path.read_text(encoding="utf-8"),
                home=arguments.home, mode=arguments.mode)
            split_settings_text(migrated_text, str(arguments.output_path))  # prove it splits back exactly
            arguments.output_path.touch(mode=0o600)
            arguments.output_path.write_text(migrated_text, encoding="utf-8")
            print(f"wrote {arguments.output_path} (from {arguments.config_path} and {arguments.topics_path})")
    except PaperwatchError as failure:
        print(describe_failure_for_operator(failure), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
