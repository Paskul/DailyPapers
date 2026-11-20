"""The screening stage of `collect`: bring every candidate's scores up to date, then promote today's papers.

Runs after collection and before source text acquisition, so the papers it promotes get their text in the same run
and the worker never has to request a URL (plan part 6). Makes no model call and no network request.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from paperwatch.config import AppConfig
from paperwatch.errors import ScreeningUnavailable
from paperwatch.screening.candidate_screening import screen_candidates
from paperwatch.screening.review_promotion import promote_screened_candidates
from paperwatch.screening.text_embedders import TextEmbedder, configured_embedder_identity, load_text_embedder
from paperwatch.screening.topic_screening_profiles import build_topic_screening_profiles


@dataclass
class ScreeningStageResult:
    report_lines: list[str] = field(default_factory=list)
    embedder_failure: ScreeningUnavailable | None = None  # candidates wait unscored; the caller reports it


def run_screening_and_promotion(connection: sqlite3.Connection, app_config: AppConfig, *,
                                active_seed_paper_ids: frozenset[str], moment: datetime | None = None,
                                build_embedder: Callable[[], TextEmbedder] | None = None,
                                hold_new_readings_until: datetime | None = None) -> ScreeningStageResult:
    """`build_embedder` lets tests inject an embedder with known vectors; by default the configured one is loaded,
    and only if some candidate needs a (re)score. `hold_new_readings_until` keeps the new readings from being claimed
    until `collect` has acquired their source text (paperwatch.readings)."""
    stage_result = ScreeningStageResult()
    topic_profiles = build_topic_screening_profiles(connection, app_config, configured_embedder_identity(app_config))
    if not topic_profiles:
        stage_result.report_lines.append("Screening: no topic has a question, phrases, or seeds yet; nothing is "
                                         "scored or promoted (see `paperwatch status`)")
        return stage_result
    screening_outcome = screen_candidates(connection, app_config, topic_profiles,
                                          active_seed_paper_ids=active_seed_paper_ids,
                                          build_embedder=build_embedder or (lambda: load_text_embedder(app_config)))
    stage_result.report_lines.append(screening_outcome.describe(len(topic_profiles)))
    stage_result.embedder_failure = screening_outcome.embedder_failure
    promotion_outcome = promote_screened_candidates(
        connection, app_config.screening, screening_outcome,
        tuple(topic_profile.topic_id for topic_profile in topic_profiles),
        max_attempts_per_job=app_config.worker.max_attempts_per_job, moment=moment,
        hold_new_readings_until=hold_new_readings_until)
    stage_result.report_lines.append(promotion_outcome.describe())
    return stage_result
