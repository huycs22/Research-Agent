import argparse
import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest

from paperqa.benchmarks.cost import (
    CostLedger,
    SpendingLimitReached,
    request_cost,
)
from paperqa.benchmarks.cost_callback import BenchmarkCostCallback
from paperqa.benchmarks.deepresearch import (
    PILOT_TASK_IDS,
    ExperimentPaths,
    _run_fact,
    _run_race,
    fact_diagnostics,
    generate_async,
    load_pilot_tasks,
    preflight,
    report,
    resolve_citations,
    write_json,
    write_jsonl,
)
from paperqa.benchmarks.deepresearch_proxy import metered_proxy
from paperqa.research.library import Library


def _tasks():
    return [
        {"id": task_id, "prompt": f"Exact prompt {task_id}", "language": "en"}
        for task_id in PILOT_TASK_IDS
    ]


def _prepared(tmp_path: Path) -> ExperimentPaths:
    paths = ExperimentPaths(tmp_path)
    paths.experiment.mkdir(parents=True)
    write_json(paths.manifest, {"experiment_id": paths.experiment_id})
    write_jsonl(paths.tasks, _tasks())
    return paths


def test_loads_frozen_tasks_in_order_without_rewriting(tmp_path):
    query = tmp_path / "query.jsonl"
    rows = [
        {"id": number, "prompt": f"Prompt  {number}\nwith spacing", "language": "en"}
        for number in range(1, 101)
    ]
    write_jsonl(query, rows)
    selected = load_pilot_tasks(query)
    assert [row["id"] for row in selected] == list(PILOT_TASK_IDS)
    assert selected[0]["prompt"] == "Prompt  51\nwith spacing"
    assert all(row["language"] == "en" for row in selected)


@pytest.mark.asyncio
async def test_empty_hybrid_library_does_not_construct_embedding(tmp_path, monkeypatch):
    papers = tmp_path / "papers"
    papers.mkdir()

    def forbidden(_name):
        raise AssertionError("empty library should not load MiniLM")

    monkeypatch.setattr("paperqa.research.library.embeddings_for", forbidden)
    library = Library(
        papers,
        tmp_path / "indexes",
        "hybrid-st-sentence-transformers/all-MiniLM-L6-v2",
    )
    assert await library.search("anything", None, 6) == []
    assert library.documents == []


def test_citation_conversion_resolves_exact_url_and_flags_unknown():
    source_id = "src-" + "a" * 24
    missing_id = "src-" + "b" * 24
    article, mapping, unresolved = resolve_citations(
        f"Claim ({source_id}); gap ({missing_id}).",
        [
            {
                "id": source_id,
                "docname": "A very long but useful source title",
                "url": "https://example.org/paper",
                "excerpt": "support",
            }
        ],
    )
    assert "https://example.org/paper" in article
    assert f"unresolved citation: {missing_id}" in article
    assert mapping[0]["source_id"] == source_id
    assert unresolved == [missing_id]


def test_reports_without_citations_remain_valid():
    article, mapping, unresolved = resolve_citations("A transparent partial answer.", [])
    assert article == "A transparent partial answer."
    assert mapping == []
    assert unresolved == []


def test_citation_conversion_rejects_captcha_as_unresolved():
    source_id = "src-" + "c" * 24
    article, mapping, unresolved = resolve_citations(
        f"Claim ({source_id}).",
        [
            {
                "id": source_id,
                "url": "https://validate.perfdrive.com/challenge",
                "excerpt": "Captcha Page. Confirm you are a human.",
            }
        ],
    )
    assert mapping == []
    assert unresolved == [source_id]
    assert "unresolved citation" in article


def test_cost_ledger_reserves_reconciles_and_never_silently_changes_limit(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", 10.0)
    reservation = ledger.reserve(
        stage="race", model="gpt-5.5", input_tokens=1000, maximum_output_tokens=2000
    )
    ledger.reconcile(reservation, input_tokens=900, output_tokens=100)
    snapshot = ledger.snapshot()
    assert snapshot["reserved_usd"] == 0
    assert snapshot["spent_usd"] == pytest.approx(request_cost("gpt-5.5", 900, 100))
    with pytest.raises(ValueError, match="refusing to silently change"):
        CostLedger(tmp_path / "ledger.json", 11.0).snapshot()


def test_cost_ledger_blocks_before_request_and_accepts_openrouter_alias(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", 0.01)
    with pytest.raises(SpendingLimitReached):
        ledger.reserve(
            stage="race",
            model="openai/gpt-5.5",
            input_tokens=1000,
            maximum_output_tokens=128_000,
        )


def test_uncertain_failure_keeps_reservation(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.json", 10)
    reservation = ledger.reserve(
        stage="fact", model="gpt-5.4-mini", input_tokens=20, maximum_output_tokens=100
    )
    ledger.retain_uncertain(reservation, "network outcome unknown")
    snapshot = ledger.snapshot()
    assert snapshot["reserved_usd"] > 0
    assert snapshot["entries"][0]["status"] == "uncertain"


def test_metered_proxy_reconciles_reported_usage(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Provider(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_POST(self):
            body = json.dumps({
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    ledger = CostLedger(tmp_path / "ledger.json", 10)
    try:
        with metered_proxy(
            ledger=ledger,
            upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
            api_key="secret",
            stage="fact",
        ) as proxy:
            response = httpx.post(
                proxy + "/chat/completions",
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [{"role": "user", "content": "test"}],
                    "max_completion_tokens": 100,
                },
            )
        assert response.status_code == 200
        snapshot = ledger.snapshot()
        assert snapshot["entries"][0]["status"] == "reconciled"
        assert snapshot["reserved_usd"] == 0
    finally:
        upstream.shutdown()
        thread.join(timeout=5)
        upstream.server_close()


def test_generation_callback_reserves_and_reconciles_cloudflare(tmp_path):
    from types import SimpleNamespace
    from uuid import uuid4

    ledger = CostLedger(tmp_path / "ledger.json", 10)
    callback = BenchmarkCostCallback(
        ledger, "cloudflare/@cf/zai-org/glm-4.7-flash", maximum_output_tokens=4096
    )
    run_id = uuid4()
    callback.on_chat_model_start(
        {}, [[SimpleNamespace(type="human", content="research this")]], run_id=run_id
    )
    assert ledger.snapshot()["reserved_usd"] > 0
    response = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 20, "completion_tokens": 10}},
        generations=[],
    )
    callback.on_llm_end(response, run_id=run_id)
    snapshot = ledger.snapshot()
    assert snapshot["reserved_usd"] == 0
    assert snapshot["spent_usd"] == pytest.approx(
        request_cost("cloudflare/@cf/zai-org/glm-4.7-flash", 20, 10)
    )


def test_preflight_allows_generation_without_judge_credentials(tmp_path, monkeypatch):
    paths = _prepared(tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    okay, result = preflight(
        argparse.Namespace(
            workspace=paths.workspace,
            experiment_id=paths.experiment_id,
            env_file=tmp_path / "missing.env",
            stage="generation",
            save=False,
        )
    )
    assert okay
    assert not next(item for item in result["checks"] if "judge credential" in item["name"])[
        "required"
    ]


@pytest.mark.asyncio
async def test_generation_timeout_is_saved_without_regeneration(tmp_path, monkeypatch):
    paths = _prepared(tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setattr("paperqa.research.providers.model_configured", lambda _model: True)

    calls = 0

    async def slow_execute(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(10)

    monkeypatch.setattr("paperqa.research.service.execute", slow_execute)
    args = argparse.Namespace(
        workspace=paths.workspace,
        experiment_id=paths.experiment_id,
        env_file=tmp_path / "missing.env",
        only=[51],
        timeout=0.01,
        resume_provider=False,
    )
    result = await generate_async(args)
    assert result["attempted"] == 1
    artifact = json.loads((paths.task_results / "task-51.json").read_text("utf-8"))
    assert artifact["generation"]["timed_out"] is True
    assert artifact["generation"]["terminal"] is True
    await generate_async(args)
    assert calls == 1


@pytest.mark.asyncio
async def test_generation_uses_isolated_conversations_and_resumes_same_run(
    tmp_path, monkeypatch
):
    paths = _prepared(tmp_path)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token")
    monkeypatch.setattr("paperqa.research.providers.model_configured", lambda _model: True)

    calls = {}

    async def fake_execute(job, _payload, app, *, resume=None, **_kwargs):
        calls[job.id] = calls.get(job.id, 0) + 1
        if job.question.endswith("51") and resume is None:
            job.status = "paused"
            job.pause_reason = "rate_limit"
            job.error = "retry later"
        else:
            job.status = "complete"
            job.result = {
                "answer": "Answer without citations.",
                "sources": [],
                "agent": {"plan": [{"task": "research", "status": "complete"}]},
            }
        app.save_job(job)
        app.finish_turn(job)

    monkeypatch.setattr("paperqa.research.service.execute", fake_execute)
    common = dict(
        workspace=paths.workspace,
        experiment_id=paths.experiment_id,
        env_file=tmp_path / "missing.env",
        timeout=5,
    )
    await generate_async(argparse.Namespace(**common, only=[51], resume_provider=False))
    first = json.loads((paths.task_results / "task-51.json").read_text("utf-8"))
    assert first["generation"]["terminal"] is False
    await generate_async(argparse.Namespace(**common, only=[51], resume_provider=True))
    resumed = json.loads((paths.task_results / "task-51.json").read_text("utf-8"))
    assert resumed["generation"]["complete"] is True
    assert calls[resumed["generation"]["run_id"]] == 2

    await generate_async(argparse.Namespace(**common, only=[59], resume_provider=False))
    second = json.loads((paths.task_results / "task-59.json").read_text("utf-8"))
    assert resumed["generation"]["conversation_id"] != second["generation"]["conversation_id"]


def test_report_labels_incomplete_and_counts_unreadable_sources(tmp_path):
    paths = _prepared(tmp_path)
    paths.task_results.mkdir(parents=True)
    write_json(
        paths.task_results / "task-51.json",
        {
            "id": 51,
            "prompt": "Exact prompt 51",
            "article": "Partial answer",
            "citation_mappings": [],
            "unresolved_citations": ["src-" + "f" * 24],
            "generation": {"status": "complete", "complete": True, "wall_seconds": 1},
            "events": [],
        },
    )
    result = report(
        argparse.Namespace(
            workspace=paths.workspace,
            experiment_id=paths.experiment_id,
            cost_ceiling=10.0,
        )
    )
    assert result["counts"] == {"selected": 5, "attempted": 1, "completed": 1, "scored": 0}
    assert result["evaluation_complete"] is False
    assert "Evaluation is incomplete" in paths.report_html.read_text("utf-8")


def test_fact_diagnostics_counts_no_citations_and_unreadable_sources(tmp_path):
    validated = tmp_path / "validated.jsonl"
    write_jsonl(
        validated,
        [
            {"id": 51, "citations": [], "citations_deduped": {}},
            {
                "id": 59,
                "citations": [{"fact": "x", "url": "https://example.org"}],
                "citations_deduped": {
                    "https://example.org": {
                        "url_content": "",
                        "validate_error": "no reference",
                        "validate_res": [],
                    }
                },
            },
        ],
    )
    result = fact_diagnostics(validated)
    assert result["reports_without_citations"] == 1
    assert result["unreadable_sources"] == 1
    assert result["validation_errors"] == 1
    assert result["scorable_reports"] == 0


def test_race_adapter_uses_upstream_directory_interface(tmp_path, monkeypatch):
    paths = _prepared(tmp_path)
    paths.evaluation.mkdir(parents=True)
    write_jsonl(paths.export, [{"id": 51, "prompt": "Exact prompt 51", "article": "answer"}])
    commands = []
    monkeypatch.setattr(
        "paperqa.benchmarks.deepresearch.metered_proxy",
        lambda **_kwargs: nullcontext("http://127.0.0.1:1/v1"),
    )
    monkeypatch.setattr(
        "paperqa.benchmarks.deepresearch._run",
        lambda command, **_kwargs: commands.append(command) or "ok",
    )
    _run_race(paths, CostLedger(paths.ledger), "openai", "secret", "http://upstream/v1", False)
    command = commands[0]
    assert "--raw_data_dir" in command
    assert "--raw_data_path" not in command
    assert "--only_en" in command
    raw_copy = paths.evaluation / "race" / "raw" / f"{paths.export.stem}.jsonl"
    assert raw_copy.exists()


def test_fact_adapter_accounts_for_mocked_no_citation_response(tmp_path, monkeypatch):
    paths = _prepared(tmp_path)
    paths.evaluation.mkdir(parents=True)
    write_jsonl(paths.export, [{"id": 51, "prompt": "Exact prompt 51", "article": "answer"}])
    monkeypatch.setattr(
        "paperqa.benchmarks.deepresearch.metered_proxy",
        lambda **_kwargs: nullcontext("http://127.0.0.1:1/v1"),
    )

    def fake_run(command, **_kwargs):
        if "--output_path" in command:
            output = Path(command[command.index("--output_path") + 1])
            if "utils.extract" in command:
                rows = [{"id": 51, "citations": []}]
            else:
                source = Path(command[command.index("--raw_data_path") + 1])
                rows = [dict(row, citations_deduped={}) for row in read_jsonl(source)]
            write_jsonl(output, rows)
        return "mocked"

    from paperqa.benchmarks.deepresearch import read_jsonl

    monkeypatch.setattr("paperqa.benchmarks.deepresearch._run", fake_run)
    result = _run_fact(
        paths, CostLedger(paths.ledger), "openai", "secret", "http://upstream/v1", False
    )
    assert result["status"] == "complete"
    assert result["diagnostics"]["reports_without_citations"] == 1
    assert result["diagnostics"]["scorable_reports"] == 0
    assert "not run" in result["logs"]["stat"]


def test_report_merges_mocked_official_per_task_scores(tmp_path):
    paths = _prepared(tmp_path)
    paths.task_results.mkdir(parents=True)
    paths.evaluation.joinpath("race").mkdir(parents=True)
    paths.evaluation.joinpath("fact").mkdir(parents=True)
    race_rows = []
    fact_rows = []
    for task in _tasks():
        task_id = task["id"]
        write_json(
            paths.task_results / f"task-{task_id}.json",
            {
                "id": task_id,
                "numeric_id": task_id,
                "prompt": task["prompt"],
                "article": "Supported report",
                "citation_mappings": [],
                "unresolved_citations": [],
                "generation": {"status": "complete", "complete": True, "wall_seconds": 1},
                "events": [],
                "usage": {"model_calls": 1, "tool_calls": 1, "tokens": 10, "seconds": 1},
            },
        )
        race_rows.append({
            "id": task_id,
            "comprehensiveness": 0.5,
            "insight": 0.5,
            "instruction_following": 0.5,
            "readability": 0.5,
            "overall_score": 0.5,
        })
        fact_rows.append({
            "id": task_id,
            "extracted_citations": 1,
            "evaluated_citations": 1,
            "supported_citations": 1,
            "unknown_citations": 0,
            "validation_errors": 0,
            "support_rate": 1.0,
        })
    write_jsonl(paths.evaluation / "race" / "raw_results.jsonl", race_rows)
    (paths.evaluation / "race" / "race_result.txt").write_text(
        "Overall Score: 0.5000\n", "utf-8"
    )
    write_json(
        paths.evaluation / "fact" / "diagnostics.json",
        {
            "reports": 5,
            "scorable_reports": 5,
            "evaluated_citations": 5,
            "supported_citations": 5,
            "per_task": fact_rows,
        },
    )
    (paths.evaluation / "fact" / "fact_result.txt").write_text("valid_rate: 1.0\n", "utf-8")
    result = report(
        argparse.Namespace(
            workspace=paths.workspace,
            experiment_id=paths.experiment_id,
            cost_ceiling=10.0,
        )
    )
    assert result["evaluation_complete"] is True
    assert result["counts"]["scored"] == 5
    assert result["tasks"][0]["evaluation"]["race"]["overall_score"] == 0.5
    assert result["tasks"][0]["evaluation"]["fact"]["support_rate"] == 1.0
