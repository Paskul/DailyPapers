"""The committed examples in settings/ must stay valid as the config format evolves."""

from __future__ import annotations

import shutil
from pathlib import Path

from paperwatch.config import load_app_config
from paperwatch.deployment.settings_file import build_example_settings_text, stage_settings
from paperwatch.network.collector_credentials import load_collector_credentials

SETTINGS_DIR = Path(__file__).parent.parent / "settings"


def test_example_settings_file_is_current_and_valid(tmp_path):
    """settings/paperwatch.example.toml is generated from the package templates and must stay in step with them."""
    example_path = SETTINGS_DIR / "paperwatch.example.toml"
    assert example_path.read_text() == build_example_settings_text(), (
        "regenerate it: uv run python -m paperwatch.deployment.settings_file write-example "
        "settings/paperwatch.example.toml")
    stage_dir = tmp_path / "stage"
    stage_settings(example_path, stage_dir)
    app_config = load_app_config(stage_dir)
    assert [topic.topic_id for topic in app_config.topics] == ["team-safety"]


def test_example_credentials_file_has_the_accepted_shape(tmp_path):
    credentials_path = tmp_path / "collector.toml"
    shutil.copy(SETTINGS_DIR / "collector_credentials.example.toml", credentials_path)
    credentials_path.chmod(0o600)
    credentials = load_collector_credentials(credentials_path)
    assert credentials.has("zotero", "api_key") and credentials.get("zotero", "library_type") == "user"
    # The example is committed to git: it must only ever hold placeholders, never a real key.
    assert credentials.get("zotero", "api_key") == "YOUR-READ-ONLY-ZOTERO-KEY"
    assert credentials.get("zotero", "library_id") == "12345678"
    assert credentials.get("openalex", "api_key") == "YOUR-OPENALEX-API-KEY"
