"""Shared persistent execution service for the UI and Python agent entrypoint."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langsmith import tracing_context

from .config import CLOUDFLARE_MODEL, DEFAULT_EMBEDDING, EvidencePolicy
from .library import Library
from .router import route_prompt
from .runtime import (
    build_agent,
    local_library_answer,
    quick_conversation_answer,
    result_from_state,
)
from .safety import Meter, PauseRun, redact
from .storage import RunRecord, RunStorage

CONVERSATION_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")
DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md", ".html"}


class Worker:
    """One event loop owns asynchronous clients and running jobs."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="paperqa-agent-worker", daemon=True
        )
        self.thread.start()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def close(self):
        async def drain():
            tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        self.submit(drain()).result(timeout=10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


async def delete_memory(home: Path, conversation_id: str):
    async with AsyncSqliteSaver.from_conn_string(
        str(home / "research-checkpoints.sqlite")
    ) as saver:
        await saver.setup()
        await saver.adelete_thread(conversation_id)


def complete_local_conversation(
    job,
    app,
    answer: str,
    *,
    recovered: bool = False,
    engine: str | None = None,
    verification_status: str = "no_research_claims",
    tools: list[str] | None = None,
    answer_context: list[dict] | None = None,
    completion_detail: str | None = None,
) -> None:
    """Persist a provider-free conversational answer, including legacy recovery."""
    duration = max(0.0, round(__import__("time").time() - job.created_at, 2))
    job.route = "conversation"
    if not recovered:
        job.usage = {
            "tokens": 0,
            "model_calls": 0,
            "tool_calls": 0,
            "estimated_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "seconds": duration,
        }
    job.result = {
        "session_id": job.id,
        "answer": answer,
        "formatted_answer": answer,
        "status": "success",
        "sources": [],
        "reference_count": 0,
        "retrieved_paper_count": 0,
        "embedding": job.embedding,
        "route": "conversation",
        "duration": duration,
        "cost": None,
        "token_counts": {"total": job.usage.get("tokens", 0)},
        "usage": dict(job.usage),
        "agent": {
            "engine": engine
            or ("local-conversation-recovery" if recovered else "local-conversation-fast-path"),
            "verification": {"status": verification_status},
            "plan": [],
            "phase": "finished",
            "answer_status": {
                "available": bool(answer.strip()),
                "format_required": False,
                "format_valid": True,
                "selected_label": None,
                "abstained": False,
                "fallback_abstention": False,
                "marker_count": 0,
            },
            "answer_context": list(answer_context or []),
        },
        "answer_available": bool(answer.strip()),
        "format_valid": True,
        "verification_status": verification_status,
        "execution_status": "terminal",
        "tools": list(tools or []),
    }
    job.status = "complete"
    job.error = None
    job.pause_reason = None
    job.retry_at = None
    job.add_event(
        "complete",
        "Conversation recovered" if recovered else "Conversation complete",
        (
            "Answered locally after the earlier provider connection failure"
            if recovered
            else completion_detail
            or "Answered locally; no model or research tools were needed"
        ),
        "complete",
    )
    app.save_job(job)
    app.finish_turn(job)


async def execute(
    job,
    payload: dict,
    app,
    *,
    resume: dict | None = None,
    supplied_docs=None,
    model=None,
    callbacks=None,
    library=None,
    retrieval_observer=None,
):
    if job.stop_requested:
        job.status, job.pause_reason = "stopped", "user_stop"
        app.save_job(job)
        app.finish_turn(job)
        return
    job.status, job.error = "running", None
    job.pause_reason, job.pending_question, job.retry_at = None, None, None
    app.save_job(job)
    conversation = app.get_conversation(job.conversation_id)
    history = conversation.get("messages", [])
    if history and history[-1].get("role") == "user":
        history = history[:-1]
    if payload.get("local_conversation_fast_path") and (
        quick_answer := quick_conversation_answer(job.question, history)
    ):
        complete_local_conversation(job, app, quick_answer)
        return
    meter = Meter(job.usage, lambda: app.save_job(job))
    library = library or Library(
        app.paper_directory, app.index_directory, job.embedding, supplied_docs, meter
    )
    if inventory_answer := local_library_answer(job.question, library.list()):
        inventory = library.list()
        complete_local_conversation(
            job,
            app,
            inventory_answer,
            engine="local-library-inventory",
            verification_status="tool_grounded",
            tools=["list_library"],
            answer_context=[{"kind": "library_inventory", "papers": inventory}],
            completion_detail="Answered from the current library inventory; no model call was needed",
        )
        return
    config = {
        "configurable": {"thread_id": job.conversation_id, "run_id": job.id},
        "recursion_limit": 250,
        "callbacks": list(callbacks or []),
    }
    graph = None
    try:
        with tracing_context(enabled=False):
            async with AsyncSqliteSaver.from_conn_string(
                str(app.home / "research-checkpoints.sqlite")
            ) as saver:
                # Never enable arbitrary pickle deserialization for local checkpoints.
                saver.serde = JsonPlusSerializer(
                    pickle_fallback=False, allowed_msgpack_modules=None
                )
                graph = build_agent(
                    library,
                    meter,
                    job.add_event,
                    saver,
                    job.model,
                    "chat" if job.route == "conversation" else job.mode,
                    bool(payload.get("web_search", True)),
                    int(payload.get("answer_max_sources", 5)),
                    model,
                    evidence_k=int(payload.get("evidence_k", 6)),
                    reasoning_effort=str(payload.get("reasoning_effort", "medium")),
                    retrieval_observer=retrieval_observer,
                    completion_contract=payload.get("completion_contract"),
                    evidence_policy=EvidencePolicy(
                        search_results=int(payload.get("evidence_k", 6)),
                        max_cited_passages=int(payload.get("max_cited_passages", 8)),
                        max_source_documents=int(payload.get("answer_max_sources", 5)),
                    ),
                )
                snapshot = await graph.aget_state(config)
                already_started = snapshot.values.get("run_id") == job.id
                if not already_started:
                    accessible_source_ids = (
                        sorted(await library.source_ids())
                        if job.route != "conversation" and hasattr(library, "source_ids")
                        else []
                    )
                    history = []
                    if not snapshot.values.get("messages"):
                        conversation = app.get_conversation(job.conversation_id)
                        for message in conversation["messages"][:-1]:
                            content = message.get("content", "")
                            if not content:
                                continue
                            cls = HumanMessage if message["role"] == "user" else AIMessage
                            if cls is AIMessage:
                                content = (
                                    "[Prior conversation, not verified evidence]\n" + content
                                )
                            history.append(cls(content=content, id=message["id"]))
                    values = {
                        "messages": [
                            *history,
                            HumanMessage(content=job.question, id=job.id + ":user"),
                        ],
                        "run_id": job.id,
                        "current_question": job.question,
                        "route": job.route,
                        "repair_count": 0,
                        "researched": False,
                        "verification": {},
                        "plan": [],
                        "usage": dict(job.usage),
                        "phase": "research",
                        "completion_contract": dict(payload.get("completion_contract") or {}),
                        "accessible_source_ids": accessible_source_ids,
                        "guard_reasons": [],
                        "finalization_attempts": 0,
                        "correction_attempts": 0,
                    }
                    if not snapshot.values:
                        values.update(
                            evidence={},
                            summary="",
                            summary_until=0,
                            retrieval_trace=[],
                            local_searches=[],
                            web_discoveries=[],
                            answer_context=[],
                        )
                elif snapshot.interrupts:
                    if not resume or not str(resume.get("answer", "")).strip():
                        raise PauseRun(
                            "clarification",
                            snapshot.interrupts[0].value.get(
                                "question", "Please answer the pending question."
                            ),
                        )
                    values = Command(resume=str(resume["answer"])[:4000])
                else:
                    values = None
                if not already_started or snapshot.next or snapshot.interrupts:
                    await graph.ainvoke(values, config)
                snapshot = await graph.aget_state(config)
                if snapshot.interrupts:
                    question = snapshot.interrupts[0].value.get(
                        "question", "Please clarify the research question."
                    )
                    raise PauseRun("clarification", question)
                job.result = result_from_state(
                    snapshot.values,
                    job.id,
                    job.usage,
                    job.usage.get("seconds", 0),
                    job.embedding,
                )
                job.result["tools"] = [
                    event["title"] for event in job.events if event["stage"] == "tool"
                ]
                job.result["execution_status"] = (
                    "terminal" if job.result["agent"].get("phase") == "finished" else "incomplete"
                )
                job.status = "unsure" if job.result["status"] == "unsure" else "complete"
                completion_title = (
                    "Answer saved · verification unavailable"
                    if job.result["status"] == "unverified"
                    else "Research complete"
                    if job.status == "complete"
                    else "Partial answer"
                )
                job.add_event(
                    "complete",
                    completion_title,
                    "Cited answer and sources saved; provider verification can be retried later"
                    if job.result["status"] == "unverified"
                    else "Results and sources saved",
                    "complete",
                )
    except PauseRun as exc:
        job.status, job.pause_reason = "paused", exc.reason
        job.error, job.retry_at = exc.detail, exc.retry_at
        if exc.reason == "clarification":
            job.pending_question = exc.detail
        job.add_event("paused", "Research paused", exc.detail, "paused")
    except asyncio.CancelledError:
        job.status, job.pause_reason = "stopped", "user_stop"
        job.error = "Stopped. Saved work can be resumed; already submitted requests may still use provider quota."
        job.add_event("stopped", "Research stopped", job.error, "stopped")
    except Exception as exc:
        from .safety import provider_pause

        spending_limit = type(exc).__name__ == "SpendingLimitReached"
        pause = provider_pause(exc) if not spending_limit else None
        if spending_limit:
            job.status, job.pause_reason = "paused", "spending_ceiling"
            job.retry_at = None
            job.error = str(exc)
            job.add_event("paused", "Research paused", job.error, "paused")
            return
        job.status = "paused" if pause else "failed"
        job.pause_reason = pause.reason if pause else None
        job.retry_at = pause.retry_at if pause else None
        job.error = pause.detail if pause else f"{type(exc).__name__}: {redact(exc)}"
        job.add_event(
            "error", "Research paused" if pause else "Research failed", job.error, "error"
        )
    finally:
        meter.close()
        meter.tick()
        app.save_job(job)
        app.finish_turn(job)


class ResearchService:
    """Public, durable interface shared by Python callers, the UI, and benchmarks."""

    def __init__(self, home: Path):
        self.home = home.resolve()
        self.paper_directory = self.home / "papers"
        self.index_directory = self.home / "indexes"
        self.conversation_directory = self.home / "conversations"
        for directory in (
            self.paper_directory,
            self.index_directory,
            self.conversation_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.run_storage = RunStorage(self.home)
        self.jobs: dict[str, RunRecord] = self.run_storage.load()
        self.worker: Worker | None = None
        self.futures: dict[str, Any] = {}
        self.lock = threading.RLock()
        self.storage_lock = threading.RLock()
        self._repair_conversation_records()

    def _repair_conversation_records(self) -> None:
        """Idempotently synchronize run state; never rewrite answer content."""
        for job in self.jobs.values():
            try:
                conversation = self.get_conversation(job.conversation_id)
            except ValueError:
                continue
            response = next(
                (item for item in conversation["messages"] if item.get("run_id") == job.id),
                None,
            )
            if response is None or response.get("status") != job.status:
                self.finish_turn(job)

    def save_job(self, job: RunRecord) -> None:
        self.run_storage.save(job)

    def launch(self, job: RunRecord, payload: dict, resume: dict | None = None) -> None:
        if self.worker is None:
            self.worker = Worker()
        job.payload = dict(payload)
        job.status = "queued"
        job.stop_requested = False
        self.save_job(job)
        self.futures[job.id] = self.worker.submit(
            execute(job, job.payload, self, resume=resume)
        )

    async def run(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        mode: str = "research",
        embedding: str = DEFAULT_EMBEDDING,
        web_search: bool = True,
        reasoning_effort: str = "medium",
        evidence_policy: EvidencePolicy | None = None,
        completion_contract: dict | None = None,
        library=None,
        model=None,
        retrieval_observer=None,
    ) -> RunRecord:
        """Execute one turn and return its durable run record."""
        policy = evidence_policy or EvidencePolicy()
        history = self.get_conversation(conversation_id)["messages"] if conversation_id else []
        decision = route_prompt(
            question,
            mode,
            self.last_route(conversation_id) if conversation_id else None,
        )
        quick = quick_conversation_answer(question, history)
        conversation_id = self.begin_turn(question, conversation_id)
        job = RunRecord(
            id=uuid4().hex[:12],
            conversation_id=conversation_id,
            question=question,
            model=CLOUDFLARE_MODEL,
            embedding=embedding,
            mode=mode,
            route=decision.route,
        )
        job.add_event(
            "queued",
            "Conversation queued" if decision.route == "conversation" else "Agent queued",
            decision.reason,
        )
        self.jobs[job.id] = job
        payload = {
            "web_search": web_search,
            "reasoning_effort": reasoning_effort,
            "evidence_k": policy.search_results,
            "answer_max_sources": policy.max_source_documents,
            "max_cited_passages": policy.max_cited_passages,
            "completion_contract": completion_contract or {},
            "local_conversation_fast_path": bool(quick),
        }
        job.payload = payload
        await execute(
            job,
            payload,
            self,
            library=library,
            model=model,
            retrieval_observer=retrieval_observer,
        )
        return job

    def inspect(self, run_id: str) -> RunRecord:
        try:
            return self.jobs[run_id]
        except KeyError as exc:
            raise ValueError("Run not found.") from exc

    def stop(self, job: RunRecord | str) -> None:
        job = self.inspect(job) if isinstance(job, str) else job
        job.stop_requested = True
        if job.status == "queued":
            job.status, job.pause_reason = "stopped", "user_stop"
            job.error = "Stopped before execution. Resume when ready."
            self.save_job(job)
            self.finish_turn(job)
        future = self.futures.get(job.id)
        if future and not future.done():
            future.cancel()

    def resume(self, run_id: str, answer: str | None = None) -> RunRecord:
        job = self.inspect(run_id)
        if job.status not in {"paused", "stopped", "failed"}:
            raise ValueError("This run is not resumable.")
        if job.retry_at and time.time() < job.retry_at:
            raise ValueError("The provider retry time has not arrived yet.")
        if job.pending_question and not (answer or "").strip():
            raise ValueError("Answer the pending clarification first.")
        self.launch(job, job.payload, resume={"answer": answer} if answer else {})
        return job

    def close(self) -> None:
        if self.worker:
            self.worker.close()
            self.worker = None

    def list_documents(self) -> list[dict[str, Any]]:
        return [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "modified": path.stat().st_mtime,
                "type": path.suffix.casefold().lstrip("."),
            }
            for path in sorted(self.paper_directory.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and path.suffix.casefold() in DOCUMENT_SUFFIXES
        ]

    def active_job(self) -> RunRecord | None:
        return next(
            (job for job in self.jobs.values() if job.status in {"queued", "running"}),
            None,
        )

    def _conversation_path(self, conversation_id: str) -> Path:
        if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
            raise ValueError("Invalid conversation ID.")
        return self.conversation_directory / f"{conversation_id}.json"

    def _write_conversation(self, conversation: dict[str, Any]) -> None:
        path = self._conversation_path(conversation["id"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(conversation, ensure_ascii=False, indent=2), "utf-8")
        temporary.replace(path)

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        with self.storage_lock:
            try:
                conversation = json.loads(
                    self._conversation_path(conversation_id).read_text("utf-8")
                )
            except FileNotFoundError as exc:
                raise ValueError("Conversation not found.") from exc
            if conversation.get("id") != conversation_id or not isinstance(
                conversation.get("messages"), list
            ):
                raise ValueError("Conversation data is invalid.")
            return conversation

    def begin_turn(self, question: str, conversation_id: str | None = None) -> str:
        now = time.time()
        with self.storage_lock:
            if conversation_id:
                conversation = self.get_conversation(conversation_id)
            else:
                conversation_id = uuid4().hex[:12]
                title = " ".join(question.split())
                conversation = {
                    "id": conversation_id,
                    "title": title if len(title) <= 64 else title[:61].rstrip() + "...",
                    "created_at": now,
                    "messages": [],
                }
            conversation["messages"].append({
                "id": uuid4().hex[:12],
                "role": "user",
                "content": question,
                "created_at": now,
            })
            conversation.update(status="running", updated_at=now)
            self._write_conversation(conversation)
            return conversation_id

    def finish_turn(self, job: RunRecord) -> None:
        now = time.time()
        content = (
            job.result["answer"]
            if job.result
            else job.pending_question or job.error or f"The run ended with status: {job.status}."
        )
        with self.storage_lock:
            conversation = self.get_conversation(job.conversation_id)
            existing = next(
                (item for item in conversation["messages"] if item.get("run_id") == job.id),
                None,
            )
            index = conversation["messages"].index(existing) if existing else None
            conversation["messages"] = [
                item for item in conversation["messages"] if item.get("run_id") != job.id
            ]
            conversation["messages"].append({
                "id": existing["id"] if existing else uuid4().hex[:12],
                "role": "assistant",
                "content": content,
                "created_at": existing.get("created_at", now) if existing else now,
                "status": job.status,
                "model": job.model,
                "embedding": job.embedding,
                "run_id": job.id,
                "events": list(job.events),
                "result": job.result,
                "error": job.error,
                "pause_reason": job.pause_reason,
                "pending_question": job.pending_question,
                "retry_at": job.retry_at,
                "usage": dict(job.usage),
            })
            if index is not None:
                conversation["messages"].insert(index, conversation["messages"].pop())
            conversation.update(status=job.status, updated_at=now)
            self._write_conversation(conversation)

    def list_conversations(self) -> list[dict[str, Any]]:
        result = []
        for path in self.conversation_directory.glob("*.json"):
            try:
                item = json.loads(path.read_text("utf-8"))
                messages = item.get("messages", [])
                result.append({
                    "id": item["id"],
                    "title": item.get("title") or "Untitled research",
                    "created_at": item.get("created_at", 0),
                    "updated_at": item.get("updated_at", 0),
                    "status": item.get("status", "complete"),
                    "message_count": len(messages),
                    "turn_count": sum(m.get("role") == "user" for m in messages),
                })
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(result, key=lambda item: item["updated_at"], reverse=True)

    def delete_conversation(self, conversation_id: str) -> None:
        if any(
            job.conversation_id == conversation_id and job.status in {"queued", "running"}
            for job in self.jobs.values()
        ):
            raise RuntimeError("A conversation cannot be deleted while it is running.")
        if self.worker is None:
            self.worker = Worker()
        self.worker.submit(delete_memory(self.home, conversation_id)).result(timeout=10)
        self.run_storage.delete(conversation_id)
        self.jobs = {
            key: job for key, job in self.jobs.items() if job.conversation_id != conversation_id
        }
        try:
            self._conversation_path(conversation_id).unlink()
        except FileNotFoundError as exc:
            raise ValueError("Conversation not found.") from exc

    def last_route(self, conversation_id: str) -> str | None:
        for message in reversed(self.get_conversation(conversation_id)["messages"]):
            if message.get("role") == "assistant":
                return (message.get("result") or {}).get("route", "research")
        return None
