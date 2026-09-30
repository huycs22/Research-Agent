"""Local HTTP interface for the focused research agent."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import os
import re
import threading
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from paperqa.research.config import (
    CLOUDFLARE_MODEL,
    DEFAULT_EMBEDDING,
    REASONING_EFFORTS,
    SUPPORTED_EMBEDDINGS,
    default_data_directory,
)
from paperqa.research.library import Library
from paperqa.research.providers import cloudflare_configured
from paperqa.research.router import route_prompt
from paperqa.research.runtime import local_library_answer, quick_conversation_answer
from paperqa.research.service import ResearchService
from paperqa.research.storage import RunRecord

logger = logging.getLogger(__name__)
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".html"}
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
DEMO_NAME = "PaperQA2.pdf"
DEMO_URL = "https://arxiv.org/pdf/2409.13740"
REQUEST_ID = re.compile(r"^[a-zA-Z0-9-]{12,80}$")

EMBEDDINGS = (
    {"value": "sparse", "label": "Sparse · local", "provider": "local"},
    {"value": "st-sentence-transformers/all-MiniLM-L6-v2", "label": "MiniLM L6 · local", "provider": "sentence-transformers"},
    {"value": "st-multi-qa-MiniLM-L6-cos-v1", "label": "Multi-QA MiniLM · local", "provider": "sentence-transformers"},
    {"value": "st-sentence-transformers/all-mpnet-base-v2", "label": "MPNet Base · local", "provider": "sentence-transformers"},
    {"value": DEFAULT_EMBEDDING, "label": "Hybrid MiniLM + sparse · local", "provider": "sentence-transformers"},
)


def safe_filename(value: str) -> str:
    name = Path(unquote(value)).name
    if not name or name != value or len(name) > 180 or Path(name).suffix.casefold() not in ALLOWED_SUFFIXES:
        raise ValueError("Invalid document filename.")
    return name


def redact_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for name, value in os.environ.items():
        if name.endswith(("_TOKEN", "_KEY", "_SECRET")) and len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return text[:1000]


def load_asset(name: str) -> bytes:
    if name not in {"index.html", "app.css", "app.js"}:
        raise FileNotFoundError(name)
    return resources.files("paperqa.ui").joinpath("static", name).read_bytes()


class Handler(BaseHTTPRequestHandler):
    server: Server

    def log_message(self, format: str, *args) -> None:
        logger.debug(format, *args)

    @property
    def state(self) -> ResearchService:
        return self.server.service

    def json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, detail: str, status: HTTPStatus) -> None:
        self.json({"error": detail}, status)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 100_000:
            raise ValueError("Request is too large.")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("Request must be a JSON object.")
        return value

    def asset(self, name: str) -> None:
        try:
            body = load_asset(name)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mimetypes.guess_type(name)[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            return self.asset("index.html")
        if path in {"/app.css", "/app.js"}:
            return self.asset(path[1:])
        if path == "/api/config":
            local = find_spec("sentence_transformers") is not None
            return self.json({
                "workspace_path": str(self.state.home),
                "cloudflare_configured": cloudflare_configured(),
                "default_model": CLOUDFLARE_MODEL,
                "default_reasoning_effort": os.getenv("REASONING_EFFORT", "medium"),
                "model_options": [{"value": CLOUDFLARE_MODEL, "label": "Cloudflare Workers AI · GLM-4.7-Flash", "provider": "cloudflare", "available": cloudflare_configured()}],
                "default_embedding": DEFAULT_EMBEDDING if local else "sparse",
                "embedding_options": [{**item, "available": local if item["provider"] == "sentence-transformers" else True} for item in EMBEDDINGS],
                "local_embeddings_available": local,
                "home": str(self.state.home),
                "engine": "langchain-langgraph",
            })
        if path == "/api/documents":
            return self.json({"documents": self.state.list_documents()})
        if path.startswith("/api/documents/"):
            try:
                name = safe_filename(path.removeprefix("/api/documents/"))
                body = (self.state.paper_directory / name).read_bytes()
            except (ValueError, FileNotFoundError) as exc:
                return self.error(str(exc), HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'inline; filename="{name}"')
            self.end_headers()
            return self.wfile.write(body)
        if path == "/api/conversations":
            return self.json({"conversations": self.state.list_conversations()})
        if path.startswith("/api/conversations/"):
            try:
                return self.json({"conversation": self.state.get_conversation(path.rsplit("/", 1)[-1])})
            except ValueError as exc:
                return self.error(str(exc), HTTPStatus.NOT_FOUND)
        if path == "/api/runs":
            jobs = sorted(self.state.jobs.values(), key=lambda job: job.created_at, reverse=True)
            return self.json({"runs": [job.as_dict(include_result=False) for job in jobs]})
        if path.startswith("/api/runs/"):
            try:
                return self.json(self.state.inspect(path.rsplit("/", 1)[-1]).as_dict())
            except ValueError as exc:
                return self.error(str(exc), HTTPStatus.NOT_FOUND)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        with self.state.lock:
            self._post()

    def _post(self) -> None:
        path = urlparse(self.path).path
        action = re.fullmatch(r"/api/runs/([a-f0-9]{12})/(stop|resume)", path)
        if action:
            try:
                job = self.state.inspect(action.group(1))
                if action.group(2) == "stop":
                    self.state.stop(job)
                else:
                    answer = str(self.read_json().get("answer") or "") or None
                    self.state.resume(job.id, answer)
                return self.json({"run": job.as_dict()}, HTTPStatus.ACCEPTED)
            except ValueError as exc:
                return self.error(str(exc), HTTPStatus.CONFLICT)
        if path == "/api/documents":
            if self.state.active_job():
                return self.error("Wait for or stop the active run before changing documents.", HTTPStatus.CONFLICT)
            try:
                name = safe_filename(self.headers.get("X-Filename", ""))
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_DOCUMENT_BYTES:
                    raise ValueError("Documents must be between 1 byte and 50 MB.")
                target = self.state.paper_directory / name
                temporary = target.with_suffix(target.suffix + ".upload")
                temporary.write_bytes(self.rfile.read(length))
                temporary.replace(target)
                return self.json({"document": name}, HTTPStatus.CREATED)
            except (ValueError, OSError) as exc:
                return self.error(redact_error(exc), HTTPStatus.BAD_REQUEST)
        if path == "/api/demo":
            try:
                request = urllib.request.Request(DEMO_URL, headers={"User-Agent": "Research-Agent/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    content = response.read(MAX_DOCUMENT_BYTES + 1)
                if len(content) > MAX_DOCUMENT_BYTES or not content.startswith(b"%PDF"):
                    raise ValueError("The demo download did not return a valid bounded PDF.")
                (self.state.paper_directory / DEMO_NAME).write_bytes(content)
                return self.json({"document": DEMO_NAME}, HTTPStatus.CREATED)
            except Exception as exc:
                return self.error(redact_error(exc), HTTPStatus.BAD_GATEWAY)
        if path != "/api/runs":
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            payload = self.read_json()
            question = str(payload.get("question") or "").strip()
            if not question or len(question) > 4000:
                raise ValueError("Questions must contain 1–4,000 characters.")
            request_id = payload.get("request_id")
            if request_id is not None and not REQUEST_ID.fullmatch(str(request_id)):
                raise ValueError("Invalid request ID.")
            duplicate = next((job for job in self.state.jobs.values() if request_id and job.request_id == request_id), None)
            if duplicate:
                return self.json({"run": duplicate.as_dict()}, HTTPStatus.ACCEPTED)
            if self.state.active_job():
                raise ValueError("Another research run is active.")
            embedding = str(payload.get("embedding") or DEFAULT_EMBEDDING)
            if embedding not in SUPPORTED_EMBEDDINGS:
                raise ValueError("Unsupported embedding model.")
            mode = str(payload.get("mode") or "auto")
            if mode not in {"auto", "research", "chat"}:
                raise ValueError("Choose Auto, Research, or Chat mode.")
            reasoning = str(payload.get("reasoning_effort") or os.getenv("REASONING_EFFORT", "medium"))
            if reasoning not in REASONING_EFFORTS:
                raise ValueError("Reasoning effort must be low, medium, or high.")
            conversation_id = payload.get("conversation_id")
            route = route_prompt(question, mode, self.state.last_route(conversation_id) if conversation_id else None)
            history = self.state.get_conversation(conversation_id)["messages"] if conversation_id else []
            quick = quick_conversation_answer(question, history)
            inventory = local_library_answer(question, [{"id": item["name"], "name": item["name"]} for item in self.state.list_documents()])
            if not (quick or inventory) and not cloudflare_configured():
                raise ValueError("Cloudflare is not configured in the server environment.")
            for name, default in (("evidence_k", 6), ("answer_max_sources", 5)):
                value = payload.get(name, default)
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
                    raise ValueError(f"{name} must be an integer between 1 and 20.")
            conversation_id = self.state.begin_turn(question, conversation_id)
            job = RunRecord(
                id=uuid4().hex[:12], conversation_id=conversation_id, question=question,
                model=CLOUDFLARE_MODEL, embedding=embedding, mode=mode,
                route=route.route, request_id=request_id,
            )
            job.add_event("queued", "Conversation queued" if route.route == "conversation" else "Agent queued", route.reason)
            self.state.jobs[job.id] = job
            payload.update(web_search=bool(payload.get("web_search", True)), reasoning_effort=reasoning, local_conversation_fast_path=bool(quick), max_cited_passages=8)
            self.state.launch(job, payload)
            return self.json({"run": job.as_dict()}, HTTPStatus.ACCEPTED)
        except (ValueError, json.JSONDecodeError) as exc:
            self.error(str(exc), HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        with self.state.lock:
            path = urlparse(self.path).path
            if path.startswith("/api/conversations/"):
                try:
                    identity = path.rsplit("/", 1)[-1]
                    self.state.delete_conversation(identity)
                    return self.json({"deleted": identity})
                except RuntimeError as exc:
                    return self.error(str(exc), HTTPStatus.CONFLICT)
                except ValueError as exc:
                    return self.error(str(exc), HTTPStatus.NOT_FOUND)
            if not path.startswith("/api/documents/"):
                return self.send_error(HTTPStatus.NOT_FOUND)
            if self.state.active_job():
                return self.error("Documents cannot be removed while the agent is running.", HTTPStatus.CONFLICT)
            try:
                name = safe_filename(path.removeprefix("/api/documents/"))
                target = (self.state.paper_directory / name).resolve()
                if target.parent != self.state.paper_directory:
                    raise ValueError("Invalid document path.")
                target.unlink()
                self.json({"deleted": name})
            except FileNotFoundError:
                self.error("Document not found.", HTTPStatus.NOT_FOUND)
            except (ValueError, OSError) as exc:
                self.error(str(exc), HTTPStatus.BAD_REQUEST)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ResearchService):
        self.service = service
        super().__init__(address, Handler)

    def server_close(self) -> None:
        self.service.close()
        super().server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local scientific research agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    service = ResearchService(args.data_dir or default_data_directory())
    server = Server((args.host, args.port), service)

    def prewarm() -> None:
        try:
            asyncio.run(Library(service.paper_directory, service.index_directory, DEFAULT_EMBEDDING).ensure())
        except Exception as exc:
            logger.warning("Local retrieval prewarm was skipped: %s", redact_error(exc))

    threading.Thread(target=prewarm, name="research-library-prewarm", daemon=True).start()
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Research agent UI is running at {url}")
    print(f"Documents and indexes are stored in {service.home}")
    if not args.no_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping research agent UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
