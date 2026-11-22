"""A tiny llama.cpp look-alike on 127.0.0.1, for tests that must exercise the real HTTP model client.

It serves /health, /v1/models, /props, /tools (refused, like a server without built-in tools), and
/v1/chat/completions, whose final answer comes from a test-supplied function of the request payload.
Every chat payload is recorded so tests can inspect exactly what reached the "model".
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

FAKE_SERVER_BUILD = "fake-http-build"


class FakeLlamaServer:
    def __init__(self, choose_final_answer: Callable[[dict[str, Any]], str]) -> None:
        self._choose_final_answer = choose_final_answer
        self.received_chat_payloads: list[dict[str, Any]] = []
        self._http_server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        assert self._http_server is not None
        return self._http_server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> FakeLlamaServer:
        fake_server = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server naming)
                if self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                elif self.path == "/v1/models":
                    self._send_json(200, {"data": [{"id": "qwen-local", "aliases": []}]})
                elif self.path == "/props":
                    self._send_json(
                        200,
                        {
                            "build_info": FAKE_SERVER_BUILD,
                            "model_path": "/models/fake.gguf",
                            "default_generation_settings": {"n_ctx": 262144},
                            "total_slots": 1,
                            "chat_template_caps": {"reasoning_effort_levels": ["low", "xhigh"]},
                            "cors_proxy_enabled": False,
                        },
                    )
                elif self.path == "/tools":
                    self._send_json(403, {"error": "tools are disabled"})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                request_payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake_server.received_chat_payloads.append(request_payload)
                final_answer = fake_server._choose_final_answer(request_payload)
                self._send_json(
                    200,
                    {
                        "choices": [
                            {
                                "message": {"content": final_answer, "reasoning_content": "(fake reasoning)"},
                                "finish_reason": "stop",
                            }
                        ],
                        "model": "qwen-local",
                        "system_fingerprint": FAKE_SERVER_BUILD,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    },
                )

            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *arguments: Any) -> None:
                pass  # keep test output quiet

        self._http_server = ThreadingHTTPServer(("127.0.0.1", 0), RequestHandler)
        self._server_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._server_thread.start()
        return self

    def __exit__(self, *exception_info: object) -> None:
        assert self._http_server is not None
        self._http_server.shutdown()
        self._http_server.server_close()
