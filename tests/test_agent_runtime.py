from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from paperqa.research.config import CLOUDFLARE_MODEL, EvidencePolicy
from paperqa.research.library import Library, _expand_passages
from paperqa.research.providers import build_chat_model, default_model, model_configured
from paperqa.research.router import route_prompt
from paperqa.research.runtime import (
    ResearchMiddleware,
    _final_evidence_context,
    cited_ids,
    quick_conversation_answer,
)
from paperqa.research.safety import Meter
from paperqa.research.service import ResearchService, execute
from paperqa.research.storage import RunRecord
from paperqa.research.validation import TOOL_MARKUP, completion_status


def tool_call(name: str, args: dict, number: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{number}"}])


class ScriptModel(BaseChatModel):
    handler: Any

    @property
    def _llm_type(self):
        return "script"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tools = kwargs.get("tools", [])
        if any(
            getattr(item, "__name__", "") == "SupportCheck"
            or (isinstance(item, dict) and item.get("function", {}).get("name") == "SupportCheck")
            for item in tools
        ):
            message = tool_call("SupportCheck", {"supported": True, "issues": []}, 99)
        else:
            message = self.handler(messages, kwargs)
        message.usage_metadata = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_job(service: ResearchService, question: str) -> RunRecord:
    conversation_id = service.begin_turn(question)
    job = RunRecord(
        id="a" * 12,
        conversation_id=conversation_id,
        question=question,
        model=CLOUDFLARE_MODEL,
        embedding="sparse",
        mode="research",
        route="research",
    )
    service.jobs[job.id] = job
    return job


def chunk(index: int) -> Document:
    source_id = f"src-{index:024x}"
    return Document(
        id=source_id,
        page_content=f"PaperQA evidence passage {index} describes retrieval and citation behavior.",
        metadata={
            "source_id": source_id,
            "document_id": "paperqa2",
            "filename": "paperqa2.pdf",
            "title": "PaperQA2",
            "citation": "PaperQA2",
            "page": "1",
            "position": index,
            "start_index": index * 100,
        },
    )


class ExpandedLibrary:
    def __init__(self):
        self.documents = [chunk(index) for index in range(11)]

    def list(self):
        return [{"id": "paperqa2", "name": "PaperQA2"}]

    async def source_ids(self):
        return {item.metadata["source_id"] for item in self.documents}

    async def corpus_identity(self):
        return "eleven-passage-fixture"

    async def search(self, query, document_ids, k):
        return self.documents[::2][:6]

    async def read(self, ids, neighbors=1):
        return _expand_passages(self.documents, ids, neighbors, preserve_pages=True)


@pytest.mark.asyncio
async def test_guard_reserves_final_and_correction_turn_for_expanded_evidence(tmp_path):
    """Six results expand to eleven; final answer still satisfies the eight-citation policy."""
    service = ResearchService(tmp_path)
    question = "Tell me about PaperQA2"
    job = make_job(service, question)
    library = ExpandedLibrary()
    research_calls = 0
    final_calls = 0

    def handler(messages, kwargs):
        nonlocal research_calls, final_calls
        system = "\n".join(str(message.content) for message in messages if message.type == "system")
        if "AVAILABLE CITABLE EVIDENCE" in system:
            final_calls += 1
            encoded = system.split("AVAILABLE CITABLE EVIDENCE:\n", 1)[1]
            evidence, _ = json.JSONDecoder().raw_decode(encoded)
            ids = [item["id"] for item in evidence]
            selected = (
                [*ids, library.documents[-1].metadata["source_id"]]
                if final_calls == 1
                else ids[:3]
            )
            return AIMessage(content="PaperQA2 uses evidence retrieval " + " ".join(f"({item})" for item in selected))
        research_calls += 1
        results = [message for message in messages if isinstance(message, ToolMessage)]
        if research_calls in {1, 5, 6}:
            return tool_call("update_research_plan", {"tasks": [{"task": "inspect evidence", "status": "active"}]}, research_calls)
        if research_calls == 2:
            return tool_call("list_library", {}, research_calls)
        if research_calls == 3:
            return tool_call("search_library", {"query": "PaperQA2 retrieval", "k": 6}, research_calls)
        ids = [item["id"] for item in json.loads(results[-1].content)]
        return tool_call("read_passages", {"source_ids": ids, "neighbors": 1}, research_calls)

    await execute(
        job,
        {
            "evidence_k": 6,
            "answer_max_sources": 5,
            "max_cited_passages": 8,
            "web_search": False,
        },
        service,
        library=library,
        model=ScriptModel(handler=handler),
    )

    assert job.status == "complete", job.error
    assert final_calls == 2
    assert 0 < len(cited_ids(job.result["answer"])) <= 8
    assert job.result["agent"]["phase"] == "finished"
    assert any(event["stage"] == "guardrail" for event in job.events)
    final_context = next(
        item for item in job.result["agent"]["answer_context"]
        if item.get("kind") == "available_citable_evidence"
    )
    assert len(json.loads(final_context["content"])) <= 8


@pytest.mark.asyncio
async def test_sparse_library_returns_overlap_and_rejects_zero_overlap(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "study.txt").write_text("The intervention reduced the value by 31 percent.", "utf-8")
    library = Library(papers, tmp_path / "indexes", "sparse")
    assert await library.search("reduced 31 percent", None, 3)
    assert await library.search("unrelated zebrafish mutation", None, 3) == []


def test_final_evidence_obeys_passage_and_document_policy():
    evidence = {
        f"src-{index:024x}": {
            "document_id": f"doc-{index}",
            "name": f"Study {index}",
            "excerpt": "Retrieval evidence with a measured value of 31 percent.",
        }
        for index in range(11)
    }
    selected = _final_evidence_context(evidence, "retrieval measured value", 8, 5)
    assert len(selected) == 5


def test_conversations_survive_restart_and_remain_isolated(tmp_path):
    first = ResearchService(tmp_path)
    alpha = first.begin_turn("My preferred format is a table.")
    beta = first.begin_turn("Use concise prose.")
    first.close()
    restarted = ResearchService(tmp_path)
    assert "table" in restarted.get_conversation(alpha)["messages"][0]["content"]
    assert "concise prose" not in restarted.get_conversation(alpha)["messages"][0]["content"]
    assert restarted.get_conversation(beta)["id"] == beta


@pytest.mark.asyncio
async def test_compaction_prompt_preserves_corrections_and_source_provenance():
    captured: dict[str, str] = {}
    source_id = "src-0123456789abcdef01234567"

    def summarize(messages, kwargs):
        captured["prompt"] = messages[-1].content
        return AIMessage(
            content=(
                f"User prefers tables. Corrected dose is 12.5 mg/kg. "
                f"Unresolved replication question cites {source_id}."
            )
        )

    usage: dict = {}
    meter = Meter(usage, lambda: None)
    middleware = ResearchMiddleware(
        meter,
        ScriptModel(handler=summarize),
        lambda *args: None,
        EvidencePolicy(),
    )
    facts = (
        f"User preference: tables. Explicit correction: dose is 12.5 mg/kg, not 15. "
        f"Unresolved question: why did replication differ? Evidence source: {source_id}. "
    )
    old_messages = [HumanMessage(content=facts + ("context " * 900)) for _ in range(5)]
    recent_messages = [HumanMessage(content=f"Recent turn {index}") for index in range(8)]
    update = await middleware.abefore_model(
        {
            "messages": [*old_messages, *recent_messages],
            "summary": "",
            "summary_until": 0,
            "current_question": "Where did the corrected dose come from?",
        },
        None,
    )
    meter.close()

    assert update["summary_until"] == 5
    assert "memory, not verified evidence" in captured["prompt"]
    assert "12.5 mg/kg, not 15" in captured["prompt"]
    assert source_id in captured["prompt"]
    assert "12.5 mg/kg" in update["summary"] and source_id in update["summary"]


def test_explicit_user_correction_supersedes_earlier_identity():
    history = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "I think your name might be Bob."},
        {"role": "user", "content": "Call me Huy."},
    ]
    assert "Huy" in quick_conversation_answer("What is my name?", history)


def test_cloudflare_is_the_only_agent_provider(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    assert default_model() == CLOUDFLARE_MODEL
    assert model_configured(CLOUDFLARE_MODEL)
    with pytest.raises(ValueError, match="Only Cloudflare"):
        build_chat_model("groq/old-model")


def test_evidence_policy_defaults_are_shared():
    assert EvidencePolicy() == EvidencePolicy(
        search_results=6,
        max_cited_passages=8,
        max_source_documents=5,
    )


def test_central_router_keeps_mixed_greeting_on_research_path():
    assert route_prompt("Hello!", "research").route == "conversation"
    assert route_prompt("Hello, compare these papers", "auto").route == "research"


@pytest.mark.asyncio
async def test_public_service_handles_simple_conversation_without_provider(tmp_path):
    service = ResearchService(tmp_path)
    job = await service.run("Hello!", mode="auto", web_search=False)
    assert job.status == "complete"
    assert job.route == "conversation"
    assert job.usage["model_calls"] == 0
    assert job.usage["tool_calls"] == 0
    service.close()


def test_completion_contract_rejects_markup_and_conflicting_markers():
    contract = {
        "labels": ["A", "B", "INSUFFICIENT"],
        "marker": "FINAL_CHOICE",
        "abstention_label": "INSUFFICIENT",
    }
    valid = completion_status("Evidence is absent.\nFINAL_CHOICE: INSUFFICIENT", contract)
    conflict = completion_status("FINAL_CHOICE: A\nFINAL_CHOICE: B", contract)
    assert valid["format_valid"] and valid["abstained"]
    assert not conflict["format_valid"] and conflict["marker_count"] == 2
    assert TOOL_MARKUP.search('<tool_call>{"name":"search_library"}</tool_call>')


@pytest.mark.asyncio
async def test_deleting_conversation_removes_runs_and_checkpoints(tmp_path):
    service = ResearchService(tmp_path)
    job = await service.run(
        "What efficacy value is supported by this empty library?",
        mode="research",
        web_search=False,
        embedding="sparse",
        model=ScriptModel(handler=lambda messages, kwargs: AIMessage(content="Insufficient evidence.")),
    )
    checkpoint = tmp_path / "research-checkpoints.sqlite"
    with sqlite3.connect(checkpoint) as connection:
        before = connection.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = ?", (job.conversation_id,)
        ).fetchone()[0]
    assert before > 0
    assert any(service.run_storage.directory.glob("*.json"))

    service.delete_conversation(job.conversation_id)

    with sqlite3.connect(checkpoint) as connection:
        after = connection.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = ?", (job.conversation_id,)
        ).fetchone()[0]
    assert after == 0
    assert not list(service.run_storage.directory.glob("*.json"))
    with pytest.raises(ValueError, match="Conversation not found"):
        service.get_conversation(job.conversation_id)
    service.close()
