"""Saved results of the steps of one reading, so a timeout or crash does not restart a long paper (plan part 7).

A reading is several model calls: one per reading section ("section:003"), then the evidence synthesis
("synthesis"). Each validated result is saved as soon as it exists, in its own short transaction, keyed by
    paper + review input key + reading plan hash + stage key + prompt version.
A retry of the job (after a timeout, an outage, or a crash) loads the saved stages and continues with the first
missing one. A changed plan (other section limits) never reuses old results. A changed prompt version reuses only the
results of the versions its role lists as reusable (RoleDefinition.reusable_earlier_prompt_versions), and only after
they pass the role's CURRENT host checks again (paperwatch.reading.reading_pipeline). Saving is idempotent
(INSERT OR IGNORE), so two workers that raced on the same input cannot conflict; the job's lease check still
decides which one commits the review.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from paperwatch.clock import utc_now_timestamp
from paperwatch.db import write_transaction
from paperwatch.roles.base import RoleRunResult

SYNTHESIS_STAGE_KEY = "synthesis"


@dataclass(frozen=True)
class ReadingStageIdentity:
    paper_id: str
    review_input_key: str
    reading_plan_sha256: str


@dataclass(frozen=True)
class SavedStageResult:
    validated_output: dict[str, Any]
    model_identity: str
    server_build: str
    prompt_version: str  # the version that produced it (may be a reusable earlier one)


def load_saved_stage_result(
    connection: sqlite3.Connection, stage_identity: ReadingStageIdentity, stage_key: str, prompt_version: str
) -> SavedStageResult | None:
    return next(iter(load_saved_stage_results(connection, stage_identity, stage_key, (prompt_version,))), None)


def load_saved_stage_results(
    connection: sqlite3.Connection,
    stage_identity: ReadingStageIdentity,
    stage_key: str,
    prompt_versions: tuple[str, ...],
) -> list[SavedStageResult]:
    """Saved results of this stage made by any of `prompt_versions`, in the order the versions are given (the
    current version first)."""
    saved_results = []
    for prompt_version in prompt_versions:
        stage_row = connection.execute(
            """SELECT result_json, model_identity, server_build FROM review_stage_results
               WHERE paper_id = ? AND review_input_key = ? AND reading_plan_sha256 = ? AND stage_key = ?
                 AND prompt_version = ?""",
            (
                stage_identity.paper_id,
                stage_identity.review_input_key,
                stage_identity.reading_plan_sha256,
                stage_key,
                prompt_version,
            ),
        ).fetchone()
        if stage_row is not None:
            saved_results.append(
                SavedStageResult(
                    json.loads(stage_row["result_json"]),
                    stage_row["model_identity"],
                    stage_row["server_build"],
                    prompt_version,
                )
            )
    return saved_results


def save_stage_result(
    connection: sqlite3.Connection,
    stage_identity: ReadingStageIdentity,
    stage_key: str,
    role_result: RoleRunResult,
    job_id: str | None,
) -> None:
    with write_transaction(connection):
        connection.execute(
            """INSERT OR IGNORE INTO review_stage_results (paper_id, review_input_key, reading_plan_sha256, stage_key,
                   prompt_version, role_name, result_json, model_identity, server_build, job_id, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                stage_identity.paper_id,
                stage_identity.review_input_key,
                stage_identity.reading_plan_sha256,
                stage_key,
                role_result.prompt_version,
                role_result.role_name,
                json.dumps(role_result.validated_output, ensure_ascii=False),
                role_result.model_response.model_identity,
                role_result.model_response.server_build,
                job_id,
                utc_now_timestamp(),
            ),
        )


def count_saved_stages_of_unfinished_readings(connection: sqlite3.Connection) -> int:
    """Stage results of inputs that have no review yet (a long paper in progress or waiting for its retry)."""
    return connection.execute(
        """SELECT COUNT(*) FROM review_stage_results AS stage
           WHERE NOT EXISTS (SELECT 1 FROM reviews AS review WHERE review.paper_id = stage.paper_id
                             AND review.review_input_key = stage.review_input_key)"""
    ).fetchone()[0]
