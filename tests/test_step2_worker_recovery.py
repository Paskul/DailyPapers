"""Plan part 2 check: import one paper by two routes; kill a worker after claim and again after commit; start two
workers on the same fixture. Require one canonical paper, one accepted review for that input, and no stale worker
overwriting the result.

Real worker processes are used where the property depends on processes (SIGKILL, concurrent claims); the fake
model is slowed down with PAPERWATCH_FAKE_MODEL_DELAY_SECONDS so a "model call" is in flight when we interfere.
"""

from __future__ import annotations

import signal
import time

import pytest

from paperwatch.app_context import AppContext
from paperwatch.commands import work_command
from paperwatch.commands.work_command import WorkCycleOutcome, handle_review_paper_job, run_single_work_cycle
from paperwatch.jobs import claim_next_due_job, finish_job_with_writes
from paperwatch.llm import FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE
from paperwatch.llm.fake_client import FakeModelClient

from conftest import ARXIV_ROUTE_FIXTURE, OPENALEX_ROUTE_FIXTURE, RECORDED_PAPER_FIXTURE, PaperwatchHome

SLOW_FAKE_MODEL = {FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE: "6"}
# Stored timestamps have whole-second resolution, so a lease of N seconds lasts between N-1 and N seconds. A 3 s
# lease renewed every second keeps a safe margin for thread and connection start-up on a busy test machine.
TEST_LEASE_SECONDS = 3
SECONDS_UNTIL_UNRENEWED_LEASE_EXPIRES = TEST_LEASE_SECONDS + 1


def wait_until(condition, *, timeout_seconds: float = 20.0, description: str) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting until {description}")
        time.sleep(0.1)


def prepare_home(home: PaperwatchHome, *fixture_paths) -> None:
    home.use_fake_model()
    home.add_example_topic()
    home.use_short_leases(lease_seconds=TEST_LEASE_SECONDS, renewal_interval_seconds=1)
    for fixture_path in fixture_paths:
        collect_result = home.run_cli("collect", "--fixture", str(fixture_path))
        assert collect_result.exit_code == 0, collect_result.stderr


def job_status(home: PaperwatchHome) -> str:
    return home.query("SELECT status FROM jobs")[0]["status"]


def test_paper_imported_by_two_routes_is_one_canonical_paper_with_one_review(initialized_home):
    prepare_home(initialized_home)
    arxiv_result = initialized_home.run_cli("collect", "--fixture", str(ARXIV_ROUTE_FIXTURE))
    openalex_result = initialized_home.run_cli("collect", "--fixture", str(OPENALEX_ROUTE_FIXTURE))
    assert "1 new, 0 already known" in arxiv_result.stdout
    assert "Promotion: 1 paper(s) queued for review now" in arxiv_result.stdout
    # The linked record joins the arXiv paper (same text: no new job); the look-alike is new, flagged, and screened.
    assert "2 record(s), 1 new, 1 already known, 0 new reading(s) of promoted papers queued" in openalex_result.stdout
    assert "Promotion: 1 paper(s) queued for review now" in openalex_result.stdout
    assert "1 possible duplicate(s) flagged" in openalex_result.stdout

    linked_papers = initialized_home.query(
        "SELECT DISTINCT paper_id FROM paper_aliases WHERE provider_paper_id IN ('2411.01234', 'W4400000001')")
    assert len(linked_papers) == 1
    linked_paper_id = linked_papers[0]["paper_id"]
    assert len(initialized_home.query("SELECT * FROM paper_discoveries WHERE paper_id = ?", (linked_paper_id,))) == 2

    assert initialized_home.run_cli("work", "--once").exit_code == 0
    assert initialized_home.run_cli("work", "--once").exit_code == 0
    assert len(initialized_home.query("SELECT * FROM reviews WHERE paper_id = ?", (linked_paper_id,))) == 1
    assert "Possible duplicate papers (not merged): 1 unresolved" in initialized_home.run_cli("status").stdout

    # Report links use the trusted, normalized identifiers.
    report_date = initialized_home.make_todays_report_publishable_early()
    assert initialized_home.run_cli("report", "--date", report_date, "--early").exit_code == 0
    report_text = (initialized_home.home_dir / "output" / "daily" / f"{report_date}.md").read_text()
    assert "https://arxiv.org/abs/2411.01234" in report_text and "https://openalex.org/W4400000001" in report_text


def test_worker_killed_after_claim_is_reclaimed_and_completes_once(initialized_home):
    prepare_home(initialized_home, RECORDED_PAPER_FIXTURE)
    doomed_worker = initialized_home.start_cli("work", "--once", environment_overrides=SLOW_FAKE_MODEL)
    wait_until(lambda: job_status(initialized_home) == "running", description="the worker claimed the job")
    doomed_worker.send_signal(signal.SIGKILL)
    doomed_worker.wait(timeout=10)

    # Before the short lease expires, the job is still held; afterwards a new worker reclaims it.
    status_text = initialized_home.run_cli("status").stdout
    assert "reviewing 1" in status_text
    time.sleep(SECONDS_UNTIL_UNRENEWED_LEASE_EXPIRES)
    assert "expired lease" in initialized_home.run_cli("status").stdout
    recovery_result = initialized_home.run_cli("work", "--once")
    assert recovery_result.exit_code == 0 and "completed review_paper" in recovery_result.stdout

    [job_row] = initialized_home.query("SELECT status, attempts, lease_token FROM jobs")
    assert (job_row["status"], job_row["attempts"], job_row["lease_token"]) == ("done", 2, None)
    assert len(initialized_home.query("SELECT * FROM reviews")) == 1
    assert initialized_home.query("SELECT status FROM papers")[0]["status"] == "complete"


def test_worker_killed_right_after_commit_leaves_exactly_one_review(tmp_path, monkeypatch):
    from test_step2_identity_and_readings import BASE_RECORD, collect, load_test_app
    fake_model = FakeModelClient()
    app = load_test_app(tmp_path, fake_model)
    collect(app, BASE_RECORD)

    original_finish = work_command.finish_job_with_writes

    def finish_then_die(*arguments, **keyword_arguments):
        assert original_finish(*arguments, **keyword_arguments)
        raise KeyboardInterrupt("simulated crash right after the commit")

    monkeypatch.setattr(work_command, "finish_job_with_writes", finish_then_die)
    with pytest.raises(KeyboardInterrupt):
        run_single_work_cycle(app, probe_paused_model_now=False)
    monkeypatch.undo()
    app.close()

    restarted_app = AppContext.load(tmp_path, model_client=fake_model)
    assert run_single_work_cycle(restarted_app, probe_paused_model_now=False).outcome == WorkCycleOutcome.NO_JOB_DUE
    assert collect(restarted_app, BASE_RECORD)[1] is False
    assert restarted_app.connection.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == 1
    assert restarted_app.connection.execute("SELECT status FROM jobs").fetchone()[0] == "done"
    assert len(fake_model.received_requests) == 2  # the one reading's reviewer + topic-assessor calls, nothing more


def test_stale_worker_cannot_overwrite_the_reclaiming_workers_result(tmp_path):
    from test_step2_identity_and_readings import BASE_RECORD, collect, load_test_app
    stale_app = load_test_app(tmp_path, FakeModelClient())
    collect(stale_app, BASE_RECORD)
    stale_claim = claim_next_due_job(stale_app.connection, lease_duration_seconds=1)
    stale_result = handle_review_paper_job(stale_app, stale_claim)  # its "model call" finished, not yet committed
    time.sleep(2.1)  # the stale worker stalled past its lease without renewing

    fresh_app = AppContext.load(tmp_path, model_client=FakeModelClient())
    assert run_single_work_cycle(fresh_app, probe_paused_model_now=False).outcome == WorkCycleOutcome.JOB_COMPLETED
    [fresh_review] = fresh_app.connection.execute("SELECT review_id, completed_at FROM reviews").fetchall()

    assert finish_job_with_writes(stale_app.connection, stale_claim, stale_result.write_results) is False
    [review_after] = fresh_app.connection.execute("SELECT review_id, completed_at FROM reviews").fetchall()
    assert tuple(review_after) == tuple(fresh_review)
    assert fresh_app.connection.execute("SELECT status, lease_token FROM jobs").fetchone()[0] == "done"


def test_two_concurrent_workers_share_the_queue_without_duplicate_reviews(initialized_home):
    prepare_home(initialized_home, RECORDED_PAPER_FIXTURE, ARXIV_ROUTE_FIXTURE, OPENALEX_ROUTE_FIXTURE)
    assert len(initialized_home.query("SELECT * FROM jobs WHERE status = 'queued'")) == 3
    slow_model = {FAKE_MODEL_DELAY_ENVIRONMENT_VARIABLE: "1"}
    workers = [initialized_home.start_cli("work", environment_overrides=slow_model) for _ in range(2)]
    try:
        wait_until(lambda: len(initialized_home.query(
            "SELECT * FROM jobs WHERE status = 'done' AND kind = 'review_paper'")) == 3,
                   description="both workers drained the queue")
    finally:
        for worker in workers:
            worker.send_signal(signal.SIGTERM)
        worker_outputs = [worker.communicate(timeout=20) for worker in workers]
    assert all(worker.returncode == 0 for worker in workers), worker_outputs

    assert len(initialized_home.query("SELECT * FROM reviews")) == 3
    assert len(initialized_home.query("SELECT DISTINCT paper_id FROM reviews")) == 3
    # Reading jobs only: part 8 may also queue comparisons of the finished reviews.
    assert [row["attempts"] for row in initialized_home.query(
        "SELECT attempts FROM jobs WHERE kind = 'review_paper'")] == [1, 1, 1]
    assert all("lease" not in stderr for _stdout, stderr in worker_outputs)


def test_live_worker_renews_its_lease_so_a_second_worker_cannot_steal_the_job(initialized_home):
    prepare_home(initialized_home, RECORDED_PAPER_FIXTURE)
    slow_worker = initialized_home.start_cli("work", "--once", environment_overrides=SLOW_FAKE_MODEL)
    wait_until(lambda: job_status(initialized_home) == "running", description="the slow worker claimed the job")
    time.sleep(SECONDS_UNTIL_UNRENEWED_LEASE_EXPIRES)  # longer than the lease: only renewal keeps the job

    second_worker_result = initialized_home.run_cli("work", "--once")
    assert second_worker_result.exit_code == 0 and "no job is due" in second_worker_result.stdout

    slow_stdout, slow_stderr = slow_worker.communicate(timeout=30)
    assert slow_worker.returncode == 0 and "completed review_paper" in slow_stdout, slow_stderr
    [job_row] = initialized_home.query("SELECT status, attempts FROM jobs")
    assert (job_row["status"], job_row["attempts"]) == ("done", 1)
    assert len(initialized_home.query("SELECT * FROM reviews")) == 1
