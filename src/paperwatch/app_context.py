"""Everything a command needs, loaded once: validated config, an open database, and the model client."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperwatch.config import AppConfig, load_app_config
from paperwatch.db import open_initialized_database
from paperwatch.llm import build_model_client
from paperwatch.llm.types import ModelClient


@dataclass
class AppContext:
    config: AppConfig
    connection: sqlite3.Connection
    model_client: ModelClient

    @classmethod
    def load(cls, home_dir: Path, *, model_client: ModelClient | None = None) -> AppContext:
        app_config = load_app_config(home_dir)
        connection = open_initialized_database(app_config.database_path)
        return cls(config=app_config, connection=connection,
                   model_client=model_client or build_model_client(app_config.model))

    def close(self) -> None:
        self.connection.close()
