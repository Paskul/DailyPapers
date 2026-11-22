"""Operations: the source-cache limit, status, capacity measurement, backup with a restore to a fresh
directory, the ranking-quality label sample, and the egress policy of the new commands."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import UTC, datetime

import pytest
from conftest import build_in_process_app
from test_evidence_reading import drain, install_extracted_text

from paperwatch.catalog import record_collected_paper
from paperwatch.cli import NO_NETWORK_COMMAND_NAMES, select_egress_policy_for_command
from paperwatch.clock import utc_now, utc_now_timestamp
from paperwatch.collection.source_cache_limit import measure_source_cache, prune_replaceable_downloads
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.status_command import run_status_command
from paperwatch.db import write_transaction
from paperwatch.maintenance.backup import BackupError, check_restored_home, create_backup
from paperwatch.measurement.capacity import MINUTES_PER_DAY, describe_capacity, measure_capacity
from paperwatch.measurement.label_sample import LABEL_COLUMNS, draw_label_sample, write_label_sheet
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.source_cache import SOURCE_FILES_SUBDIRECTORY
from paperwatch.sources.types import CollectedPaperRecord

KILOBYTE = 1024


def add_paper(app, number: int, *, title_word: str = "monitor") -> str:
    return record_collected_paper(
        app.connection,
        CollectedPaperRecord(
            provider="fixture",
            provider_paper_id=f"ops-{number:04d}",
            title=f"Runtime {title_word} study {number:03d}",
            authors=("A. Author",),
            abstract=f"We check joint robot plans with {title_word} variant {number}.",
            first_published_date="2026-11-15",
            latest_revision_date=None,
            source_version="v1",
            discovery_route="test:fixture",
        ),
    ).paper_id


# --------------------------------------------------------------------------------------------------------------------
# Source-cache limit
# --------------------------------------------------------------------------------------------------------------------


def add_cached_download(app, paper_id: str, *, byte_count: int, acquired_at: str) -> tuple:
    """An extracted paper: its raw PDF of `byte_count` bytes plus its extracted text, as the collector leaves them."""
    edition_id = install_extracted_text(app, paper_id, "Extracted text of the paper. " * 20)
    text_row = app.connection.execute(
        "SELECT * FROM edition_source_texts WHERE edition_id = ?", (edition_id,)
    ).fetchone()
    edition_dir = app.config.source_cache_dir / SOURCE_FILES_SUBDIRECTORY / edition_id
    pdf_path = edition_dir / text_row["source_file_name"]
    pdf_path.write_bytes(b"%PDF" + b"x" * (byte_count - 4))
    with write_transaction(app.connection):
        app.connection.execute(
            "UPDATE edition_source_texts SET acquired_at = ? WHERE edition_id = ?", (acquired_at, edition_id)
        )
    return pdf_path, edition_dir / text_row["extracted_text_file_name"]


def test_the_cache_limit_removes_oldest_extracted_downloads_but_keeps_text_and_pending_readings(tmp_path):
    app = build_in_process_app(
        tmp_path / "home",
        config_replacements={"source_cache_limit_megabytes = 4096": "source_cache_limit_megabytes = 1"},
    )
    pending_paper = add_paper(app, 1)
    assert promote_paper_directly_and_queue_review(app.connection, pending_paper, max_attempts=3)
    pending_files = add_cached_download(
        app, pending_paper, byte_count=600 * KILOBYTE, acquired_at="2026-11-01T00:00:00Z"
    )  # the oldest, but still to be read
    older_files = add_cached_download(
        app, add_paper(app, 2), byte_count=600 * KILOBYTE, acquired_at="2026-11-02T00:00:00Z"
    )
    newer_files = add_cached_download(
        app, add_paper(app, 3), byte_count=300 * KILOBYTE, acquired_at="2026-11-03T00:00:00Z"
    )
    assert measure_source_cache(app.config).raw_source_bytes == 1500 * KILOBYTE

    summary = prune_replaceable_downloads(app.connection, app.config)
    assert summary.files_removed == 1 and summary.bytes_removed == 600 * KILOBYTE
    assert not older_files[0].exists(), "the oldest extracted download without a pending reading goes first"
    assert pending_files[0].exists() and newer_files[0].exists()
    assert all(text_path.exists() for _pdf, text_path in (pending_files, older_files, newer_files)), "text is kept"
    assert summary.usage_after.raw_source_bytes <= summary.usage_after.limit_bytes
    assert prune_replaceable_downloads(app.connection, app.config).files_removed == 0, "within the limit now"
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------------------------------------------------


def test_capacity_measures_minutes_per_review_and_estimates_readings_per_day(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    for number in range(1, 4):
        assert promote_paper_directly_and_queue_review(app.connection, add_paper(app, number), max_attempts=3)
    drain(app)
    with write_transaction(app.connection):  # the fake model takes no time: give each call a measured duration
        app.connection.execute(
            "UPDATE model_calls SET duration_seconds = CASE role_name "
            "WHEN 'reviewer' THEN 600 WHEN 'topic_assessor' THEN 300 ELSE 60 END, "
            "completion_tokens = 3000"
        )
    measurement = measure_capacity(app.connection, app.config, window_end=utc_now(), window_hours=24)
    assert measurement.reviews_completed == 3 and measurement.review_jobs_queued_now == 0
    abstract_summary = measurement.minutes_per_review_by_coverage["abstract_only"]
    assert abstract_summary.count == 3 and abstract_summary.median_minutes == pytest.approx(15.0)
    research_and_comparison_minutes = sum(
        role.all_calls_total_minutes for role in measurement.roles if role.role_name in ("research", "comparison")
    )
    expected_reserved_minutes = research_and_comparison_minutes + app.config.report.writer_deadline_minutes
    assert measurement.reserved_minutes_per_day == pytest.approx(expected_reserved_minutes)  # no writer call yet
    assert measurement.estimated_reviews_per_day == pytest.approx((MINUTES_PER_DAY - expected_reserved_minutes) / 15.0)
    reviewer = next(role for role in measurement.roles if role.role_name == "reviewer")
    assert reviewer.completion_tokens_per_second == pytest.approx(5.0)
    report_lines = "\n".join(describe_capacity(measurement))
    assert f"about {measurement.estimated_reviews_per_day:.0f} reading(s) per day" in report_lines
    assert "abstract_only" in report_lines
    app.close()


def test_capacity_flags_limits_above_the_measured_capacity(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    assert promote_paper_directly_and_queue_review(app.connection, add_paper(app, 1), max_attempts=3)
    drain(app)
    with write_transaction(app.connection):
        app.connection.execute("UPDATE model_calls SET duration_seconds = 3600")  # one hour per call
    measurement = measure_capacity(app.connection, app.config, window_end=utc_now(), window_hours=24)
    assert measurement.estimated_reviews_per_day is not None
    assert measurement.estimated_reviews_per_day < measurement.configured_readings_per_day
    assert "the queue will grow" in "\n".join(describe_capacity(measurement))
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Backup and a restore to a fresh directory
# --------------------------------------------------------------------------------------------------------------------


def publishable_today_replacements() -> dict[str, str]:
    timezone_name = "UTC" if datetime.now(UTC).hour < 22 else "Etc/GMT+12"
    return {
        'timezone = "America/New_York"': f'timezone = "{timezone_name}"',
        'publish_time = "07:00"': 'publish_time = "23:59"',
        "minimum_reading_priority = 65": "minimum_reading_priority = 0",
    }


def test_a_backup_restores_to_a_fresh_directory_with_the_same_data_and_files(tmp_path, capsys):
    app = build_in_process_app(tmp_path / "home", config_replacements=publishable_today_replacements())
    reviewed_paper = add_paper(app, 1)
    assert promote_paper_directly_and_queue_review(app.connection, reviewed_paper, max_attempts=3)
    add_cached_download(app, add_paper(app, 2), byte_count=10 * KILOBYTE, acquired_at=utc_now_timestamp())
    drain(app)
    assert run_report_command(app, argparse.Namespace(report_date=None, publish_early=True)) == 0

    backup = create_backup(app.config, tmp_path / "backups", moment=utc_now())
    assert backup.manifest["integrity_check"] == "ok" and backup.manifest["row_counts"]["reviews"] == 1
    assert backup.manifest["extracted_text_files"] == 1 and backup.manifest["output_files"] >= 2  # note + report
    assert not (backup.backup_dir / "credentials").exists(), "provider keys never go into a backup"
    assert not list(backup.backup_dir.rglob("*.pdf")), "replaceable downloads are left out"
    assert (backup.backup_dir.stat().st_mode & 0o077) == 0, "private to the operator"

    fresh_home = tmp_path / "fresh" / "restored-home"
    shutil.copytree(backup.backup_dir, fresh_home)
    assert check_restored_home(fresh_home).passed
    from paperwatch.app_context import AppContext

    restored_app = AppContext.load(fresh_home)
    capsys.readouterr()
    assert run_status_command(restored_app, argparse.Namespace()) == 0
    restored_status = capsys.readouterr().out
    assert "Reviews: abstract_only 1" in restored_status and "Last report:" in restored_status
    restored_app.close()

    damaged_home = tmp_path / "fresh" / "damaged-home"
    shutil.copytree(backup.backup_dir, damaged_home)
    next((damaged_home / "output" / "daily").glob("*.md")).unlink()
    damaged_check = check_restored_home(damaged_home)
    assert not damaged_check.passed
    assert any("report" in problem for problem in damaged_check.problems)

    with pytest.raises(BackupError, match="outside the paperwatch home"):
        create_backup(app.config, app.config.home_dir / "backups", moment=utc_now())
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Label sample for the ranking-quality check
# --------------------------------------------------------------------------------------------------------------------


def screened_pool(app, candidate_count: int = 120, promoted_count: int = 12) -> list[str]:
    """Candidates with descending screening scores for the test topic; the best `promoted_count` were promoted."""
    paper_ids = [add_paper(app, number) for number in range(1, candidate_count + 1)]
    with write_transaction(app.connection):
        for rank, paper_id in enumerate(paper_ids):
            edition_id = app.connection.execute(
                "SELECT edition_id FROM paper_editions WHERE paper_id = ?", (paper_id,)
            ).fetchone()[0]
            app.connection.execute(
                """INSERT INTO candidate_topic_scores (paper_id, topic_id, edition_id, unnormalized_score,
                       screening_input_sha256, scored_at) VALUES (?, 'team-safety', ?, ?, 'x', ?)""",
                (paper_id, edition_id, 1.0 - rank / candidate_count, utc_now_timestamp()),
            )
            if rank < promoted_count:
                app.connection.execute(
                    """INSERT INTO paper_promotions (paper_id, promotion_reason, topic_id, best_topic_id,
                           screening_score, priority_class, promoted_at)
                       VALUES (?, 'topic_share', 'team-safety', 'team-safety', 0.9, 1, ?)""",
                    (paper_id, utc_now_timestamp()),
                )
    return paper_ids


def test_the_label_sample_covers_chosen_near_misses_and_the_rest_without_revealing_scores(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    ranked_paper_ids = screened_pool(app)
    sample = draw_label_sample(app.connection, app.config, sample_size=40, seed=7)
    assert len(sample) == 40 and len({candidate.paper_id for candidate in sample}) == 40
    bands = {candidate.band for candidate in sample}
    assert bands == {"chosen", "near_miss", "rest"}
    near_miss_ranks = [
        ranked_paper_ids.index(candidate.paper_id) for candidate in sample if candidate.band == "near_miss"
    ]
    assert all(12 <= rank < 12 + 36 for rank in near_miss_ranks), "just below the promoted papers"
    sampled_ranks = [ranked_paper_ids.index(candidate.paper_id) for candidate in sample]
    assert sampled_ranks != sorted(sampled_ranks), "shuffled: the order reveals no rank"
    assert [
        candidate.paper_id for candidate in draw_label_sample(app.connection, app.config, sample_size=40, seed=7)
    ] == [candidate.paper_id for candidate in sample], "reproducible for the same seed"

    sheet_path = tmp_path / "labels.csv"
    write_label_sheet(app.connection, app.config, sample, sheet_path)
    with sheet_path.open() as sheet_file:
        rows = list(csv.DictReader(sheet_file))
    assert tuple(rows[0].keys()) == LABEL_COLUMNS
    assert not any("score" in column or "rank" in column or "band" in column for column in LABEL_COLUMNS)
    assert all(row["would_read"] == "" and row["reason"] == "" for row in rows)
    assert (sheet_path.stat().st_mode & 0o077) == 0
    with pytest.raises(FileExistsError):
        write_label_sheet(app.connection, app.config, sample, sheet_path)  # labels are never overwritten
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Status and egress
# --------------------------------------------------------------------------------------------------------------------


def test_status_shows_the_source_cache_and_the_last_successful_fetch_per_source(tmp_path, capsys):
    app = build_in_process_app(tmp_path / "home")
    with write_transaction(app.connection):
        for provider_name, outcome, started_at in (
            ("arxiv", "ok", "2026-11-20T10:00:00Z"),
            ("arxiv", "unavailable", "2026-11-20T12:00:00Z"),
            ("openalex", "ok", "2026-11-20T11:00:00Z"),
        ):
            app.connection.execute(
                """INSERT INTO outbound_requests (request_id, provider_name, operation, method, host, path, outcome,
                       started_at, duration_seconds) VALUES (?, ?, 'search', 'GET', 'h', '/p', ?, ?, 0.1)""",
                (f"q_{provider_name}_{outcome}", provider_name, outcome, started_at),
            )
    capsys.readouterr()
    assert run_status_command(app, argparse.Namespace()) == 0
    status_output = capsys.readouterr().out
    assert "Last successful fetch from arxiv: 2026-11-20T10:00:00Z; last failure 2026-11-20T12:00:00Z" in status_output
    assert "Last successful fetch from openalex: 2026-11-20T11:00:00Z\n" in status_output
    assert "Source cache: 0.0 MB of downloads (limit 4096 MB" in status_output
    app.close()


def test_the_new_operator_commands_may_reach_no_network(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    assert NO_NETWORK_COMMAND_NAMES == {"retry-failed", "capacity", "backup", "label-sample"}
    for command_name in NO_NETWORK_COMMAND_NAMES:
        policy = select_egress_policy_for_command(command_name, app.config)
        assert policy is not None and not policy.allowed_loopback_ports and not policy.allow_public_internet_https
        assert not policy.resolvable_host_names and not policy.allowed_executables
    app.close()


def test_capacity_shows_where_model_time_goes_from_llama_cpp_timings(tmp_path):
    app = build_in_process_app(tmp_path / "home")
    with write_transaction(app.connection):
        for call_number, (
            prompt_tokens,
            cached_tokens,
            prompt_seconds,
            generated_tokens,
            generation_seconds,
        ) in enumerate(((12000, 2000, 50.0, 6000, 250.0), (8000, 0, 40.0, 4000, 150.0))):
            app.connection.execute(
                """INSERT INTO model_calls (call_id, role_name, prompt_version, started_at, duration_seconds, outcome,
                       prompt_tokens, completion_tokens, prompt_seconds, generation_seconds, cached_prompt_tokens,
                       draft_tokens, draft_tokens_accepted)
                   VALUES (?, 'section_reader', 'v', ?, ?, 'ok', ?, ?, ?, ?, ?, 3000, 2000)""",
                (
                    f"c_{call_number}",
                    utc_now_timestamp(),
                    prompt_seconds + generation_seconds,
                    prompt_tokens,
                    generated_tokens,
                    prompt_seconds,
                    generation_seconds,
                    cached_tokens,
                ),
            )
    measurement = measure_capacity(app.connection, app.config, window_end=utc_now(), window_hours=24)
    [section_reader] = measurement.roles
    rates = section_reader.server_rates
    assert rates.prompt_tokens_per_second == pytest.approx(18000 / 90.0)  # uncached prompt tokens per prompt second
    assert rates.generation_tokens_per_second == pytest.approx(10000 / 400.0)
    assert rates.cached_prompt_share == pytest.approx(2000 / 20000)
    assert rates.draft_acceptance == pytest.approx(4000 / 6000)
    assert (
        "llama.cpp: prompt 200 tokens/s · 10% of prompt tokens cached · generation 25.0 tokens/s · "
        "draft tokens accepted 67%" in "\n".join(describe_capacity(measurement))
    )
    app.close()
