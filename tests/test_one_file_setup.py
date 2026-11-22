"""The one settings file (settings/paperwatch.toml) and what ./setup.sh builds from it: the exact split into
config.toml + topics.toml, validation, the shell values for setup.sh, migration of the former two files, and the
daily schedule's systemd units."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from paperwatch.config import load_app_config
from paperwatch.deployment.daily_schedule import (
    SCOPE_SYSTEM,
    SCOPE_USER,
    daily_calendar_time,
    render_daily_units,
    render_monitor_unit,
)
from paperwatch.deployment.settings_file import (
    build_example_settings_text,
    build_migrated_settings_text,
    set_extractor_image,
    split_settings_text,
    stage_settings,
)
from paperwatch.deployment.unit_names import (
    ALL_DAILY_UNIT_NAMES,
    DAILY_COLLECT_TIMER,
    DAILY_REPORT_SERVICE,
    DAILY_REPORT_TIMER,
    DAILY_WORKER_SERVICE,
)
from paperwatch.errors import ConfigurationError

REPOSITORY_DIR = Path(__file__).parent.parent
IMAGE_ID = "b7876baf276b9e60127acd6856db59e88faabc17eeff4b54bd3e91313c545edd"


def write_settings(tmp_path: Path, text: str) -> Path:
    settings_path = tmp_path / "settings" / "paperwatch.toml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(text)
    return settings_path


def example_with(**setup_values) -> str:
    text = build_example_settings_text()
    for key, value in setup_values.items():
        rendered = f'"{value}"' if isinstance(value, str) else str(value).lower()
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith(f"{key} = "))
        lines[index] = f"{key} = {rendered}"
        text = "\n".join(lines) + "\n"
    return text


# --------------------------------------------------------------------------------------------------------------------
# Splitting and validation
# --------------------------------------------------------------------------------------------------------------------


def test_the_example_splits_exactly_and_keeps_comments():
    split = split_settings_text(build_example_settings_text())
    whole = tomllib.loads(build_example_settings_text())
    assert tomllib.loads(split.topics_text) == {"topics": whole["topics"]}
    assert tomllib.loads(split.config_text) == {
        key: value for key, value in whole.items() if key not in ("setup", "topics")
    }
    assert "[setup]" not in split.config_text and "[[topics]]" not in split.config_text
    assert "# the cutoff: reviews completed later belong to the next report" in split.config_text
    assert split.topics_text.startswith("# ====")  # the comment block above [[topics]] moved with it
    assert split.setup.mode == "hardened" and split.setup.home == "/srv/paperwatch"


def test_settings_can_be_in_any_order_and_setup_may_be_missing():
    text = """
[[topics]]
id = "a"
question = "Does it work in any order?"
public_queries = ["order independent settings"]

[report]
publish_time = "06:00"

[[topics]]
id = "b"
question = "And with two topics?"
public_queries = ["second topic query"]
"""
    split = split_settings_text(text)
    assert [topic["id"] for topic in tomllib.loads(split.topics_text)["topics"]] == ["a", "b"]
    assert tomllib.loads(split.config_text) == {"report": {"publish_time": "06:00"}}
    assert split.setup.mode == "hardened"  # defaults


@pytest.mark.parametrize(
    "bad_setup, message",
    [
        ('[setup]\nhome = "relative/path"\n', "absolute"),
        ('[setup]\nmode = "cloud"\n', "mode"),
        ('[setup]\nhome_dir = "/srv/x"\n', "unknown key"),
        ('[setup]\ndaily_schedule = "yes"\n', "must be a bool"),
        ('[setup]\nmonitor_listen_address = "0.0.0.0"\n', "loopback or a Tailscale"),
        ("[setup]\nmonitor_port = 0\n", "monitor_port"),
        ('home = "/srv/x"\n', r"outside any \[table\]"),
    ],
)
def test_bad_setup_values_are_refused(bad_setup, message):
    with pytest.raises(ConfigurationError, match=message):
        split_settings_text(bad_setup)


def test_a_hardened_home_inside_the_users_home_is_refused():
    with pytest.raises(ConfigurationError, match="service accounts cannot enter"):
        split_settings_text(f'[setup]\nhome = "{Path.home() / "paperwatch"}"\n')
    split_settings_text(f'[setup]\nhome = "{Path.home() / "paperwatch"}"\nmode = "local"\n')  # fine when local


def test_a_split_that_would_not_be_exact_is_refused():
    # A table header inside a multi-line string would send the rest of the string to the topics file.
    tricky = '[collector]\nuser_agent = """\n[[topics]]\nid = "x"\n"""\n'
    with pytest.raises(ConfigurationError, match="exactly"):
        split_settings_text(tricky)


def test_stage_validates_with_the_application_and_writes_shell_safe_values(tmp_path):
    settings_path = write_settings(tmp_path, example_with(home="/srv/it's here", credentials_file="", mode="hardened"))
    stage_dir = tmp_path / "stage"
    stage_settings(settings_path, stage_dir)
    assert load_app_config(stage_dir).topics[0].topic_id == "team-safety"
    environment = subprocess.run(
        [
            "bash",
            "-c",
            (
                f"source {shlex.quote(str(stage_dir / 'setup.env'))} && "
                'printf "%s|%s|%s|%s" "$PAPERWATCH_HOME" "$SETUP_MODE" "$MODEL_PORT" '
                '"$DAILY_SCHEDULE"'
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert environment == "/srv/it's here|hardened|8080|0"


def test_stage_refuses_invalid_config_missing_seed_files_and_zotero_without_a_key(tmp_path):
    invalid = build_example_settings_text().replace("max_papers_per_report = 15", "max_papers_per_reports = 15")
    with pytest.raises(ConfigurationError, match="max_papers_per_reports"):
        stage_settings(write_settings(tmp_path / "a", invalid), tmp_path / "a" / "stage")
    missing_seed = build_example_settings_text().replace(
        'csl_json_files = []              # e.g. ["seeds/team-safety.json"]', 'csl_json_files = ["seeds/missing.json"]'
    )
    with pytest.raises(ConfigurationError, match="missing.json"):
        stage_settings(write_settings(tmp_path / "b", missing_seed), tmp_path / "b" / "stage")
    zotero_without_key = example_with(credentials_file="").replace(
        'zotero_collections = []          # e.g. ["ABCD2345"]', 'zotero_collections = ["ABCD2345"]'
    )
    with pytest.raises(ConfigurationError, match="no credentials file"):
        stage_settings(write_settings(tmp_path / "c", zotero_without_key), tmp_path / "c" / "stage")


def test_stage_fills_in_a_built_extractor_image(tmp_path):
    settings_path = write_settings(tmp_path, example_with(credentials_file=""))
    stage_settings(settings_path, tmp_path / "stage", extractor_image=IMAGE_ID)
    assert load_app_config(tmp_path / "stage").extractor.image == IMAGE_ID
    assert f"CONFIGURED_EXTRACTOR_IMAGE={IMAGE_ID}" in (tmp_path / "stage" / "setup.env").read_text()


@pytest.mark.parametrize(
    "config_text",
    [
        '[extractor]\nimage = ""   # set me\ncpu_limit = 1.0\n',
        '[extractor]\ncpu_limit = 1.0\n\n[access]\nworker_account = "w"\n',
        '[report]\npublish_time = "07:00"\n',
    ],
)
def test_set_extractor_image_replaces_or_adds_the_key(config_text):
    updated = tomllib.loads(set_extractor_image(config_text, IMAGE_ID))
    assert updated["extractor"]["image"] == IMAGE_ID
    original = tomllib.loads(config_text)
    for table_name, table in original.items():
        for key, value in table.items():
            if (table_name, key) != ("extractor", "image"):
                assert updated[table_name][key] == value
    with pytest.raises(ConfigurationError):
        set_extractor_image(config_text, "not-an-id")


def test_former_config_and_topics_files_migrate_into_one_equivalent_file(tmp_path):
    template_dir = REPOSITORY_DIR / "src" / "paperwatch" / "templates"
    config_text = (
        (template_dir / "config.toml")
        .read_text()
        .replace('timezone = "America/New_York"', 'timezone = "Europe/Berlin"')
    )
    topics_text = (
        '# my topics\n\n[[topics]]\nid = "x"\nquestion = "A migrated question?"\npublic_queries = ["migrated query"]\n'
    )
    migrated = build_migrated_settings_text(config_text, topics_text, home="/srv/pw", mode="hardened")
    split = split_settings_text(migrated)
    assert tomllib.loads(split.config_text) == tomllib.loads(config_text)
    assert tomllib.loads(split.topics_text) == tomllib.loads(topics_text)
    assert split.setup.home == "/srv/pw"


# --------------------------------------------------------------------------------------------------------------------
# The daily schedule's units
# --------------------------------------------------------------------------------------------------------------------


@pytest.fixture
def staged_config(tmp_path):
    settings_path = write_settings(tmp_path, example_with(credentials_file=""))
    stage_settings(settings_path, tmp_path / "stage")
    return load_app_config(tmp_path / "stage")


def parse_unit(unit_text: str) -> dict[str, list[tuple[str, str]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in unit_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("["):
            current = line.strip("[]")
            sections[current] = []
        else:
            key, _, value = line.partition("=")
            sections[current].append((key, value))
    return sections


def values(unit_text: str, section: str, key: str) -> list[str]:
    return [value for found_key, value in parse_unit(unit_text).get(section, []) if found_key == key]


def test_daily_units_repeat_every_day_and_hand_the_model_to_the_report(staged_config):
    units = render_daily_units(
        staged_config, scope=SCOPE_SYSTEM, paperwatch_executable=Path("/opt/pw/paperwatch"), collector_uid=990
    )
    assert set(units) == set(ALL_DAILY_UNIT_NAMES)
    assert values(units[DAILY_COLLECT_TIMER], "Timer", "OnCalendar") == [
        f"*-*-* {clock}:00 America/New_York" for clock in staged_config.schedule.collect_times
    ]
    assert values(units[DAILY_REPORT_TIMER], "Timer", "OnCalendar") == ["*-*-* 07:00:30 America/New_York"]
    report = units[DAILY_REPORT_SERVICE]
    assert values(report, "Unit", "Conflicts") == [DAILY_WORKER_SERVICE]
    assert values(report, "Unit", "OnSuccess") == values(report, "Unit", "OnFailure") == [DAILY_WORKER_SERVICE]
    assert values(report, "Service", "ExecStart") == [f"/opt/pw/paperwatch --home {staged_config.home_dir} report"]
    worker = units[DAILY_WORKER_SERVICE]
    assert values(worker, "Service", "User") == ["paperwatch-worker"]
    assert values(worker, "Service", "IPAddressAllow") == ["localhost"]
    assert values(worker, "Install", "WantedBy") == ["multi-user.target"]
    for unit_text in units.values():  # never a boot- or activation-relative trigger
        assert "OnBootSec" not in unit_text and "OnUnitActiveSec" not in unit_text


def test_local_units_run_as_the_operator_without_accounts(staged_config):
    units = render_daily_units(staged_config, scope=SCOPE_USER, paperwatch_executable=Path("/repo/.venv/bin/pw"))
    for unit_text in units.values():
        assert "User=" not in unit_text and "paperwatch-worker" not in unit_text.replace(DAILY_WORKER_SERVICE, "")
    assert values(units[DAILY_WORKER_SERVICE], "Install", "WantedBy") == ["default.target"]


def test_daily_calendar_time_adds_the_report_delay_across_midnight():
    from datetime import timedelta

    assert daily_calendar_time("23:59", "UTC", timedelta(minutes=2)) == "*-*-* 00:01:00 UTC"


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze is not installed")
def test_rendered_units_pass_systemd_analyze(staged_config, tmp_path):
    unit_dir = tmp_path / "units"
    unit_dir.mkdir()
    units = render_daily_units(staged_config, scope=SCOPE_USER, paperwatch_executable=Path("/usr/bin/true"))
    units["paperwatch-monitor.service"] = render_monitor_unit(
        staged_config, paperwatch_executable=Path("/usr/bin/true"), port=8765, listen_address="127.0.0.1"
    )
    for name, text in units.items():
        (unit_dir / name).write_text(text)
    result = subprocess.run(
        ["systemd-analyze", "verify", "--man=no", *sorted(str(path) for path in unit_dir.iterdir())],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_setup_script_is_valid_bash_and_check_mode_changes_nothing(tmp_path):
    assert subprocess.run(["bash", "-n", str(REPOSITORY_DIR / "setup.sh")], check=False).returncode == 0
    settings_path = write_settings(
        tmp_path, example_with(home=str(tmp_path / "home"), mode="local", credentials_file="")
    )
    result = subprocess.run(
        [str(REPOSITORY_DIR / "setup.sh"), "--check", "--settings", str(settings_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Settings are valid" in result.stdout
    assert not (tmp_path / "home").exists()
