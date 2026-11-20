"""The committed examples in settings/ must stay valid as the config format evolves."""

from __future__ import annotations

import shutil
from pathlib import Path

from paperwatch.config import load_app_config
from paperwatch.network.collector_credentials import load_collector_credentials

SETTINGS_DIR = Path(__file__).parent.parent / "settings"


def test_example_config_and_topics_load_together(tmp_path):
    shutil.copy(SETTINGS_DIR / "config.example.toml", tmp_path / "config.toml")
    shutil.copy(SETTINGS_DIR / "topics.example.toml", tmp_path / "topics.toml")
    app_config = load_app_config(tmp_path)
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
