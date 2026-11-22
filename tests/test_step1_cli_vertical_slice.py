"""Plan part 1 check: run the recorded fixture through the CLI in a temporary directory, then repeat it with the
model endpoint unavailable. The CLI must report which component failed and preserve the input."""

from __future__ import annotations

from paperwatch.roles.reviewer import REVIEWER_PROMPT_VERSION

from conftest import RECORDED_PAPER_FIXTURE, PaperwatchHome


def test_recorded_paper_flows_through_cli_with_fake_model(initialized_home: PaperwatchHome) -> None:
    initialized_home.use_fake_model()
    initialized_home.add_example_topic()

    doctor_result = initialized_home.run_cli("doctor")
    assert doctor_result.exit_code == 0, doctor_result.stdout + doctor_result.stderr

    collect_result = initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert collect_result.exit_code == 0, collect_result.stderr
    assert "1 new" in collect_result.stdout and "Promotion: 1 paper(s) queued for review now" in collect_result.stdout

    work_result = initialized_home.run_cli("work", "--once")
    assert work_result.exit_code == 0, work_result.stderr
    assert "completed review_paper" in work_result.stdout

    idle_work_result = initialized_home.run_cli("work", "--once")
    assert idle_work_result.exit_code == 0 and "no job is due" in idle_work_result.stdout

    report_date = initialized_home.make_todays_report_publishable_early()
    report_result = initialized_home.run_cli("report", "--date", report_date, "--early")
    assert report_result.exit_code == 0, report_result.stderr
    report_path = initialized_home.home_dir / "output" / "daily" / f"{report_date}.md"
    report_text = report_path.read_text()
    assert "Runtime Monitoring of Joint Multi\\-Robot Plans" not in report_text  # '-' is not escaped
    assert "Runtime Monitoring of Joint Multi-Robot Plans" in report_text
    [paper_row] = initialized_home.query("SELECT paper_id, status FROM papers")
    [review_row] = initialized_home.query("SELECT review_id, model_identity, prompt_version FROM reviews")
    assert paper_row["status"] == "complete"
    assert review_row["prompt_version"] == REVIEWER_PROMPT_VERSION
    note_path = (
        initialized_home.home_dir / "output" / "papers" / paper_row["paper_id"] / f"{review_row['review_id']}.md"
    )
    assert note_path.is_file()
    assert f"../papers/{paper_row['paper_id']}/{review_row['review_id']}.md" in report_text

    # Re-collecting the same paper must not create a second paper or a second reading.
    recollect_result = initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE))
    assert "0 new, 1 already known, 0 new reading(s) of promoted papers queued" in recollect_result.stdout
    assert "Promotion: 0 paper(s) queued for review now" in recollect_result.stdout
    assert len(initialized_home.query("SELECT * FROM jobs")) == 1

    # Re-running the report for the same date changes nothing.
    report_bytes_before = report_path.read_bytes()
    assert initialized_home.run_cli("report", "--date", report_date, "--early").exit_code == 0
    assert report_path.read_bytes() == report_bytes_before

    status_result = initialized_home.run_cli("status")
    assert status_result.exit_code == 0
    assert "complete 1" in status_result.stdout and "abstract_only 1" in status_result.stdout


def test_unavailable_model_endpoint_names_component_and_preserves_input(initialized_home: PaperwatchHome) -> None:
    initialized_home.use_unreachable_model_endpoint()
    initialized_home.use_hashing_test_embedder()
    initialized_home.add_example_topic()

    doctor_result = initialized_home.run_cli("doctor")
    assert doctor_result.exit_code == 3
    assert "FAIL  model-server  health" in doctor_result.stdout

    assert initialized_home.run_cli("collect", "--fixture", str(RECORDED_PAPER_FIXTURE)).exit_code == 0

    work_result = initialized_home.run_cli("work", "--once")
    assert work_result.exit_code == 3
    assert "FAILED [model-server]" in work_result.stderr
    assert "returned to the queue without using an attempt" in work_result.stderr
    assert "is preserved" in work_result.stderr

    [paper_row] = initialized_home.query("SELECT status, title, abstract FROM papers")
    [job_row] = initialized_home.query("SELECT status, attempts, last_error_component FROM jobs")
    assert paper_row["status"] == "queued" and paper_row["abstract"]
    assert (job_row["status"], job_row["attempts"], job_row["last_error_component"]) == ("queued", 0, "model-server")
    assert initialized_home.query("SELECT COUNT(*) FROM reviews")[0][0] == 0

    status_result = initialized_home.run_cli("status")
    assert "Model server: PAUSED" in status_result.stdout

    report_result = initialized_home.run_cli("report", "--date", "2026-11-19")
    assert report_result.exit_code == 0
    report_text = (initialized_home.home_dir / "output" / "daily" / "2026-11-19.md").read_text()
    assert "Model server: unavailable" in report_text  # an outage is distinguished from a quiet day

    # The preserved input completes once a model is available again.
    initialized_home.use_fake_model()
    recovered_work_result = initialized_home.run_cli("work", "--once")
    assert recovered_work_result.exit_code == 0, recovered_work_result.stderr
    assert initialized_home.query("SELECT status FROM papers")[0]["status"] == "complete"
    assert initialized_home.query("SELECT COUNT(*) FROM reviews")[0][0] == 1


def test_broken_config_names_config_component(initialized_home: PaperwatchHome) -> None:
    initialized_home.replace_in_config('"http://127.0.0.1:8080/v1"', '"https://api.example.com/v1"')
    status_result = initialized_home.run_cli("status")
    assert status_result.exit_code == 2
    assert "FAILED [config]" in status_result.stderr and "not a loopback address" in status_result.stderr


def test_missing_fixture_names_source_component(initialized_home: PaperwatchHome) -> None:
    collect_result = initialized_home.run_cli("collect", "--fixture", "/nonexistent/recorded.json")
    assert collect_result.exit_code == 7
    assert "FAILED [source]" in collect_result.stderr
