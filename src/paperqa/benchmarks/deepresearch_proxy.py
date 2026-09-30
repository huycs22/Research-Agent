"""OpenAI-compatible local proxy that applies the benchmark cost ledger."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import httpx

from .cost import (
    CostLedger,
    SpendingLimitReached,
    estimate_message_tokens,
)


class _GuardedServer(ThreadingHTTPServer):
    ledger: CostLedger
    upstream: str
    api_key: str
    stage: str


class _Handler(BaseHTTPRequestHandler):
    server: _GuardedServer

    def log_message(self, _format: str, *_args) -> None:
        return

    def _respond(self, status: int, payload: dict | bytes, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": {"message": "Not found."}})

    def do_POST(self):
        if urlsplit(self.path).path != "/v1/chat/completions":
            self._respond(404, {"error": {"message": "Only chat completions are allowed."}})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0 or length > 32 * 1024 * 1024:
                raise ValueError("Invalid request size.")
            raw = self.rfile.read(length)
            request = json.loads(raw)
            if request.get("stream"):
                raise ValueError(
                    "Streaming evaluator calls are not supported by the cost guard."
                )
            model = str(request["model"])
            maximum_output = int(
                request.get("max_completion_tokens") or request.get("max_tokens") or 128_000
            )
            input_estimate = estimate_message_tokens(request.get("messages", []))
            reservation_id = self.server.ledger.reserve(
                stage=self.server.stage,
                model=model,
                input_tokens=input_estimate,
                maximum_output_tokens=maximum_output,
            )
        except SpendingLimitReached as exc:
            self._respond(402, {"error": {"type": "spending_limit", "message": str(exc)}})
            return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._respond(400, {"error": {"type": "invalid_request", "message": str(exc)}})
            return

        target = self.server.upstream.rstrip("/") + "/chat/completions"
        try:
            # The evaluator may run behind a corporate proxy, but loopback test
            # providers and explicitly selected upstreams must not inherit shell
            # proxy variables. Judge proxying is already controlled by this
            # localhost-only boundary.
            with httpx.Client(trust_env=False, timeout=600) as client:
                response = client.post(
                    target,
                    content=raw,
                    headers={
                        "authorization": f"Bearer {self.server.api_key}",
                        "content-type": "application/json",
                    },
                )
            response_body = response.content
            if response.is_success:
                decoded = response.json()
                usage = decoded.get("usage") or {}
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                    self.server.ledger.retain_uncertain(
                        reservation_id, "Successful provider response omitted token usage."
                    )
                else:
                    self.server.ledger.reconcile(
                        reservation_id,
                        input_tokens=int(usage["prompt_tokens"]),
                        output_tokens=int(usage["completion_tokens"]),
                        cached_input_tokens=int(cached or 0),
                    )
            else:
                self.server.ledger.retain_uncertain(
                    reservation_id,
                    f"Upstream HTTP {response.status_code}; reservation retained.",
                )
            self._respond(
                response.status_code,
                response_body,
                response.headers.get("content-type", "application/json"),
            )
        except Exception as exc:
            self.server.ledger.retain_uncertain(
                reservation_id, f"Forwarding outcome uncertain: {type(exc).__name__}."
            )
            self._respond(
                502,
                {"error": {"type": "proxy_error", "message": "Judge request failed."}},
            )


@contextmanager
def metered_proxy(*, ledger: CostLedger, upstream: str, api_key: str, stage: str):
    """Yield an OpenAI-compatible base URL guarded by ``ledger``."""
    server = _GuardedServer(("127.0.0.1", 0), _Handler)
    server.ledger = ledger
    server.upstream = upstream
    server.api_key = api_key
    server.stage = stage
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
