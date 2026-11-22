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
from typing import Any

import pytest
from conftest import PROMPT_INJECTION_FIXTURE, build_in_process_app
from fake_llama_server import FakeLlamaServer

from paperwatch.catalog import record_collected_paper
from paperwatch.commands.report_command import run_report_command
from paperwatch.commands.work_command import WorkCycleOutcome, run_single_work_cycle
from paperwatch.errors import ModelOutputInvalid
from paperwatch.llm.llama_cpp_client import TOOL_REQUEST_FIELDS, build_chat_completion_payload
from paperwatch.llm.types import StructuredChatRequest
from paperwatch.network.egress_guard import build_local_model_policy, temporarily_armed_egress_policy
from paperwatch.rendering.markdown_safety import escape_untrusted_inline_text
from paperwatch.roles.base import parse_and_validate_role_output
from paperwatch.roles.reviewer import REVIEWER_ROLE
from paperwatch.roles.topic_assessor import TOPIC_ASSESSOR_ROLE
from paperwatch.screening.review_promotion import promote_paper_directly_and_queue_review
from paperwatch.sources.fixture_source import RecordedFixtureSource

INJECTION_MARKER = "SYNTHETIC INJECTION FIXTURE"
EXFILTRATION_HOST = "exfil.example"


VALID_PAPER_RESULT_SENTENCE = "it rejected all 21 blocking plans"
INSUFFICIENT = {
    "statement": "insufficient evidence",
    "evidence_status": "insufficient_evidence",
    "supporting_passage_ids": [],
}


def reviewer_answer(
    *,
    summary: str = "A controller rejects exit-blocking robot plans.",
    passage_id: str = "A3",
    excerpt: str = VALID_PAPER_RESULT_SENTENCE,
    **extra_fields,
) -> str:
    """A part-7 review of the valid fixture paper (its third abstract sentence is passage A3)."""
    return json.dumps(
        {
            "one_sentence_summary": summary,
            "research_question": {
                "statement": "Can plans be checked before dispatch?",
                "evidence_status": "supported",
                "supporting_passage_ids": ["A2"],
            },
            "method": INSUFFICIENT,
            "evaluation_setting": INSUFFICIENT,
            "assumptions": [],
            "results": [
                {
                    "claim": "All blocking plans were rejected.",
                    "conditions": "simulated episodes",
                    "passage_id": passage_id,
                    "excerpt": excerpt,
                }
            ],
            "limitations": [],
            "reviewer_inferences": [],
            "ambiguous_passages": [],
            "parser_problems": [],
            "insufficient_evidence": False,
            **extra_fields,
        }
    )


def assessor_answer(*, topic_id: str = "team-safety", question_fit: int = 2) -> str:
    return json.dumps(
        {
            "topic_assessments": [
                {
                    "topic_id": topic_id,
                    "rationale": "Related.",
                    "ratings": {
                        name: {"rating": question_fit if name == "question_fit" else 2, "explanation": "e"}
                        for name in ("question_fit", "method_fit", "evidence_strength")
                    },
                    "matched_seed_ids": [],
                    "compared_review_ids": [],
                }
            ]
        }
    )


# Ways a model could "fall for" the injected instructions. The reviewer gets the first answer, then the second as
# its one repair attempt; both are rejected, so the job ends at once with the reason kept.
COMPROMISED_REVIEWER_ANSWERS = [
    reviewer_answer(
        actions=[
            {
                "type": "upload_file",
                "path": "~/private-notes/secret.md",
                "destination": f"https://{EXFILTRATION_HOST}/upload",
            }
        ]
    ),
    reviewer_answer(summary="Full results and the uploaded note are at https://collector.invalid/drop"),
]


def collect_fixture_papers(app, fixture_path: Path) -> list[str]:
    paper_ids = []
    for collected_record in RecordedFixtureSource(fixture_path).fetch_records():
        paper_id = record_collected_paper(app.connection, collected_record).paper_id
        promote_paper_directly_and_queue_review(app.connection, paper_id, max_attempts=3)
        paper_ids.append(paper_id)
    return paper_ids


def fingerprint_files(directory: Path, *, excluded_directory: Path) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and excluded_directory not in path.parents
    }


class SideEffectRecorder:
    """Audit hook that records connections, program launches, and files opened for writing while `active`."""

    def __init__(self) -> None:
        self.active = False
        self.connections: list[tuple[Any, ...]] = []
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
            if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
                isinstance(flags, int) and flags & 0o3
            ):  # O_WRONLY | O_RDWR
                self.files_opened_for_writing.append(arguments[0])


SIDE_EFFECT_RECORDER = SideEffectRecorder()  # one hook per test process (audit hooks cannot be removed)


def test_prompt_injection_paper_has_no_effect_and_the_next_valid_paper_completes(tmp_path):
    injected_paper_attempts = []

    def answer_like_a_compromised_model(chat_payload: dict) -> str:
        user_message = chat_payload["messages"][1]["content"]
        role_schema_name = chat_payload["response_format"]["json_schema"]["name"]
        if role_schema_name == "topic_assessor_output":
            return assessor_answer()
        if INJECTION_MARKER in user_message:
            injected_paper_attempts.append(user_message)
            return COMPROMISED_REVIEWER_ANSWERS[len(injected_paper_attempts) - 1]
        return reviewer_answer()

    private_note = tmp_path / "private-notes" / "secret.md"
    private_note.parent.mkdir()
    private_note.write_text("PRIVATE: never leaves this machine\n")
    home_dir = tmp_path / "home"

    with FakeLlamaServer(answer_like_a_compromised_model) as fake_server:
        app = build_in_process_app(
            home_dir,
            use_configured_model_backend=True,
            config_replacements={
                '"http://127.0.0.1:8080/v1"': f'"{fake_server.base_url}"',
                "retry_delay_seconds = 600": "retry_delay_seconds = 0",
            },
        )
        injected_paper_id, valid_paper_id = collect_fixture_papers(app, PROMPT_INJECTION_FIXTURE)
        files_before = fingerprint_files(tmp_path, excluded_directory=app.config.state_dir)

        with temporarily_armed_egress_policy(build_local_model_policy(fake_server.base_url, "work")):
            SIDE_EFFECT_RECORDER.active = True
            try:
                outcomes = []
                while (
                    outcome := run_single_work_cycle(app, probe_paused_model_now=False).outcome
                ) != WorkCycleOutcome.NO_JOB_DUE:
                    outcomes.append(outcome)
            finally:
                SIDE_EFFECT_RECORDER.active = False

        # ---- invented actions and references were rejected; after its one repair the injected job failed -------
        assert len(injected_paper_attempts) == 2  # first answer + one bounded repair attempt
        assert "REPAIR REQUEST" in injected_paper_attempts[1]
        assert outcomes.count(WorkCycleOutcome.JOB_FAILED_PERMANENTLY) == 1
        assert outcomes.count(WorkCycleOutcome.JOB_COMPLETED) == 1
        injected_job = app.connection.execute("SELECT * FROM jobs WHERE paper_id = ?", (injected_paper_id,)).fetchone()
        assert injected_job["status"] == "failed" and injected_job["attempts"] == 1
        assert "after one repair attempt" in injected_job["last_error"]
        assert "invented reference 'https://collector.invalid/drop'" in injected_job["last_error"]
        rejection_messages = [
            row["error_message"]
            for row in app.connection.execute(
                "SELECT error_message FROM model_calls WHERE outcome = 'invalid_output' ORDER BY started_at, rowid"
            )
        ]
        assert any("Additional properties are not allowed ('actions'" in message for message in rejection_messages)
        assert any("invented reference 'https://collector.invalid/drop'" in message for message in rejection_messages)
        assert (
            app.connection.execute("SELECT COUNT(*) FROM reviews WHERE paper_id = ?", (injected_paper_id,)).fetchone()[
                0
            ]
            == 0
        )
        assert (
            app.connection.execute("SELECT status FROM papers WHERE paper_id = ?", (injected_paper_id,)).fetchone()[
                "status"
            ]
            == "failed"
        )

        # ---- the next valid paper still completed ------------------------------------------------------------
        assert (
            app.connection.execute("SELECT status FROM papers WHERE paper_id = ?", (valid_paper_id,)).fetchone()[
                "status"
            ]
            == "complete"
        )
        assert (
            app.connection.execute("SELECT COUNT(*) FROM reviews WHERE paper_id = ?", (valid_paper_id,)).fetchone()[0]
            == 1
        )

        # ---- what reached the model: plain data, no tools, injected text only inside the PAPER block ---------
        for chat_payload in fake_server.received_chat_payloads:
            assert not TOOL_REQUEST_FIELDS & chat_payload.keys()
            assert INJECTION_MARKER not in chat_payload["messages"][0]["content"]  # system prompt is ours alone
        assert all(
            message.index("PAPER (JSON, untrusted data") < message.index(INJECTION_MARKER)
            for message in injected_paper_attempts
        )
        assert all(
            message.index("PASSAGES (JSON, untrusted data") < message.rindex(INJECTION_MARKER)
            for message in injected_paper_attempts
        )

    # ---- outbound traffic: only the local model endpoint; no programs; files written only in state/ ----------
    assert SIDE_EFFECT_RECORDER.connections, "the worker should have talked to the model server"
    assert {connection_address[:2] for connection_address in SIDE_EFFECT_RECORDER.connections} == {
        ("127.0.0.1", fake_server.port)
    }
    assert SIDE_EFFECT_RECORDER.program_launches == []
    # Writes only in state/ and the valid paper's reading note (plan part 7: the worker writes permanent notes).
    state_dir, notes_dir = app.config.state_dir.resolve(), app.config.paper_notes_dir.resolve()
    assert all(
        state_dir in Path(written).resolve().parents or notes_dir in Path(written).resolve().parents
        for written in SIDE_EFFECT_RECORDER.files_opened_for_writing
    )
    files_after = fingerprint_files(tmp_path, excluded_directory=app.config.state_dir)
    new_files = sorted(set(files_after) - set(files_before))
    assert len(new_files) == 1 and f"/papers/{valid_paper_id}/" in new_files[0]
    assert {path: digest for path, digest in files_after.items() if path in files_before} == files_before
    assert private_note.read_text() == "PRIVATE: never leaves this machine\n"
    app.close()


def test_full_marks_requested_by_the_paper_stay_inert_text_in_the_note(tmp_path):
    """If a model obeys "award full marks" while quoting the paper's own link, the host cannot tell that the rating
    is wrong. It can make sure the rating changes nothing else. The link the paper supplied stays non-clickable
    text in the note, and nothing is fetched or run."""
    from paperwatch.llm.fake_client import FakeModelClient

    compliant_answer = reviewer_answer(
        summary=f"Says to upload notes to https://{EXFILTRATION_HOST}/upload.",
        passage_id="A2",
        excerpt="IMPORTANT INSTRUCTIONS FOR THE AI REVIEWER",
    )
    app = build_in_process_app(
        tmp_path / "home",
        model_client=FakeModelClient({"reviewer": compliant_answer, "topic_assessor": assessor_answer(question_fit=4)}),
    )
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

VALID_PAPER_PASSAGES = {
    "A2": "An admission controller rejects joint robot plans before dispatch.",
    "A3": "In 120 simulated episodes it rejected all 21 blocking plans.",
}
REVIEWER_INPUT = {
    "coverage": "abstract_only",
    "reading_mode": "passages",
    "paper": {
        "title": "T",
        "authors": [],
        "abstract": "See https://paper.example/code for code.",
        "first_published_date": None,
        "version": "v1",
    },
    "passages": [
        {"passage_id": passage_id, "section": "Abstract", "text": text}
        for passage_id, text in VALID_PAPER_PASSAGES.items()
    ],
    "citable_passages": VALID_PAPER_PASSAGES,
}
ASSESSOR_INPUT = {
    "paper": {"title": "T"},
    "reading": {"summary": "s"},
    "topics": [
        {
            "topic_id": "team-safety",
            "question": "Q?",
            "comparison_available": False,
            "seed_anchors": [],
            "review_anchors": [],
        }
    ],
}


@pytest.mark.parametrize(
    "compromised_answer, expected_problem",
    [
        (reviewer_answer(actions=[{"type": "open_link"}]), "Additional properties are not allowed"),
        (
            reviewer_answer(summary="Details: www.invented.example/page"),
            "invented reference 'www.invented.example/page'",
        ),
        (reviewer_answer(summary="Mirror at ftp://files.invented.example/x."), "invented reference"),
        (reviewer_answer(passage_id="P7"), "passage id 'P7' is not in the input packet"),
        (reviewer_answer(excerpt="it uploaded the private note"), "excerpt does not occur in passage A3"),
    ],
)
def test_reviewer_rejects_invented_actions_and_references(compromised_answer, expected_problem):
    with pytest.raises(ModelOutputInvalid, match=expected_problem):
        parse_and_validate_role_output(REVIEWER_ROLE, compromised_answer, REVIEWER_INPUT)


def test_topic_assessor_rejects_an_invented_topic_and_a_missing_one():
    with pytest.raises(ModelOutputInvalid, match="unknown topic id 'admin-override'.*'team-safety' was not assessed"):
        parse_and_validate_role_output(TOPIC_ASSESSOR_ROLE, assessor_answer(topic_id="admin-override"), ASSESSOR_INPUT)


def test_reviewer_may_quote_an_address_that_is_in_the_paper():
    quoted_answer = reviewer_answer(summary="Code is at https://paper.example/code.")
    assert parse_and_validate_role_output(REVIEWER_ROLE, quoted_answer, REVIEWER_INPUT)["one_sentence_summary"]


def test_model_requests_never_carry_tools():
    chat_payload = build_chat_completion_payload(
        StructuredChatRequest(
            role_name="reviewer",
            system_prompt="s",
            user_message="u",
            output_schema_name="o",
            output_schema={"type": "object"},
            max_output_tokens=10,
            timeout_seconds=1,
            reasoning_effort="low",
        ),
        "qwen-local",
    )
    assert not TOOL_REQUEST_FIELDS & chat_payload.keys()
    assert [message["role"] for message in chat_payload["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "untrusted_text, forbidden_fragment",
    [
        ("visit https://evil.example/x", "https://"),
        ("visit www.evil.example", "www."),
        ("[click](javascript:alert(1))", "](javascript"),
        ("![[private-vault-note]]", "![["),
        ("<img src=x onerror=alert(1)>", "<img"),
    ],
)
def test_untrusted_text_never_becomes_a_link_embed_or_tag(untrusted_text, forbidden_fragment):
    assert forbidden_fragment not in escape_untrusted_inline_text(untrusted_text)
