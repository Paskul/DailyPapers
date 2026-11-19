"""`paperwatch init`: create the home layout, write default config files (never overwriting), migrate the DB."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from paperwatch.config import CONFIG_FILE_NAME, TOPICS_FILE_NAME, load_app_config
from paperwatch.db import apply_pending_migrations, connect_database


def run_init_command(home_dir: Path, arguments: argparse.Namespace) -> int:
    home_dir = home_dir.expanduser().resolve()
    home_dir.mkdir(parents=True, exist_ok=True)

    for template_file_name in (CONFIG_FILE_NAME, TOPICS_FILE_NAME):
        destination_path = home_dir / template_file_name
        if destination_path.exists():
            print(f"kept     {destination_path} (already exists)")
            continue
        template_text = resources.files("paperwatch.templates").joinpath(template_file_name).read_text("utf-8")
        destination_path.write_text(template_text, encoding="utf-8")
        print(f"created  {destination_path}")

    app_config = load_app_config(home_dir)  # validates what is on disk, including user edits
    for directory_path in (app_config.state_dir, app_config.source_cache_dir, app_config.paper_notes_dir,
                           app_config.daily_reports_dir):
        directory_path.mkdir(parents=True, exist_ok=True)

    connection = connect_database(app_config.database_path)
    try:
        version_before, version_after = apply_pending_migrations(connection)
    finally:
        connection.close()
    if version_before == version_after:
        print(f"database {app_config.database_path} is current (schema v{version_after})")
    else:
        print(f"database {app_config.database_path} migrated v{version_before} -> v{version_after}")

    print(f"\npaperwatch home ready at {home_dir}\nNext: add research topics to {home_dir / TOPICS_FILE_NAME}, "
          "then run `paperwatch doctor`.")
    return 0
