"""Reproducible five-task DeepResearch Bench pilot.

Run with ``python -m paperqa.benchmarks.deepresearch --help`` or the
``pqa-deepresearch`` console script.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .cost import (
    OFFICIAL_OPENAI_PRICES,
    PRICE_SOURCES,
    CostLedger,
)
from .deepresearch_proxy import metered_proxy

UPSTREAM_URL = "https://github.com/Ayanami0730/deep_research_bench.git"
UPSTREAM_COMMIT = "852f4022d1f98fb707222e395405136e8f0e8d52"
PILOT_TASK_IDS = (51, 59, 67, 68, 71)
EXPERIMENT_ID = "paperqa-drb-en5-v1"
MODEL = "cloudflare/@cf/zai-org/glm-4.7-flash"
EMBEDDING = "hybrid-st-sentence-transformers/all-MiniLM-L6-v2"
REASONING_EFFORT = "medium"
RUN_TIMEOUT_SECONDS = 600
DEFAULT_WORKSPACE = Path(".deepresearch-bench")


@dataclass(frozen=True)
class ExperimentPaths:
    workspace: Path
    experiment_id: str = EXPERIMENT_ID

    @property
    def upstream(self) -> Path:
        return self.workspace / "upstream"

    @property
    def evaluator_venv(self) -> Path:
        return self.workspace / "evaluator-venv"

    @property
    def experiment(self) -> Path:
        return self.workspace / "experiments" / self.experiment_id

    @property
    def manifest(self) -> Path:
        return self.experiment / "manifest.json"

    @property
    def tasks(self) -> Path:
        return self.experiment / "tasks.jsonl"

    @property
    def task_results(self) -> Path:
        return self.experiment / "tasks"

    @property
    def app_state(self) -> Path:
        return self.experiment / "agent-state"

    @property
    def export(self) -> Path:
        return self.experiment / "paperqa-glm47flash.jsonl"

    @property
    def evaluation(self) -> Path:
        return self.experiment / "evaluation"

    @property
    def ledger(self) -> Path:
        return self.experiment / "cost-ledger.json"

    @property
    def report_json(self) -> Path:
        return self.experiment / "results.json"

    @property
    def report_html(self) -> Path:
        return self.experiment / "report.html"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
    )
    temporary.replace(path)


def _task_id(row: dict) -> int:
    value = row.get("id", row.get("task_id"))
    match = re.search(r"\d+", str(value))
    if not match:
        raise ValueError(f"Task has no numeric id: {value!r}")
    return int(match.group())


def _prompt(row: dict) -> str:
    for key in ("prompt", "query", "question"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"Task {_task_id(row)} has no prompt.")


def load_pilot_tasks(query_file: Path) -> list[dict]:
    by_id = {_task_id(row): row for row in read_jsonl(query_file)}
    missing = [task_id for task_id in PILOT_TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"Pinned task IDs are missing from upstream data: {missing}")
    # Keep only public generation inputs. References and criteria remain inside
    # the pinned evaluator checkout and are never placed in agent state.
    return [
        {
            "id": by_id[task_id].get("id", task_id),
            "prompt": _prompt(by_id[task_id]),
            "language": by_id[task_id].get("language", "en"),
        }
        for task_id in PILOT_TASK_IDS
    ]


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"Command failed ({process.returncode}): {' '.join(command)}\n{process.stdout[-4000:]}"
        )
    return process.stdout


def _git(repo: Path, *args: str) -> str:
    return _run(["git", "-C", str(repo), *args]).strip()


def code_fingerprint(repo: Path) -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    status = _run([
        "git",
        "-C",
        str(repo),
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]).rstrip()
    fingerprint_paths = [
        "src/paperqa/research",
        "src/paperqa/benchmarks",
        "src/paperqa/ui",
        "scripts",
        "pyproject.toml",
        "uv.lock",
    ]
    diff = _run([
        "git",
        "-C",
        str(repo),
        "diff",
        "--no-ext-diff",
        "HEAD",
        "--",
        *fingerprint_paths,
    ])
    files = []
    for root in (
        repo / "src" / "paperqa" / "research",
        repo / "src" / "paperqa" / "benchmarks",
        repo / "src" / "paperqa" / "ui",
        repo / "scripts",
    ):
        if root.exists():
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    tree = "\n".join(
        f"{path.relative_to(repo).as_posix()}:{sha256_file(path)}" for path in sorted(files)
    )
    return {
        "git_head": head,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode()),
        "diff_sha256": sha256_bytes(diff.encode()),
        "source_tree_sha256": sha256_bytes(tree.encode()),
        "changed_paths": [line[3:] for line in status.splitlines()],
    }


def evaluator_python(paths: ExperimentPaths) -> Path:
    return paths.evaluator_venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    paths.workspace.mkdir(parents=True, exist_ok=True)
    if not paths.upstream.exists():
        _run(["git", "clone", "--no-checkout", UPSTREAM_URL, str(paths.upstream)])
    _run(["git", "-C", str(paths.upstream), "fetch", "origin", UPSTREAM_COMMIT])
    _run(["git", "-C", str(paths.upstream), "checkout", "--detach", UPSTREAM_COMMIT])
    actual = _git(paths.upstream, "rev-parse", "HEAD")
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream {UPSTREAM_COMMIT}, got {actual}.")

    query_file = paths.upstream / "data" / "prompt_data" / "query.jsonl"
    tasks = load_pilot_tasks(query_file)
    paths.experiment.mkdir(parents=True, exist_ok=True)
    paths.task_results.mkdir(parents=True, exist_ok=True)
    paths.app_state.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.tasks, tasks)

    if not args.skip_evaluator_install:
        python = evaluator_python(paths)
        if not python.exists():
            _run([sys.executable, "-m", "venv", str(paths.evaluator_venv)])
        marker = paths.evaluator_venv / ".deepresearch-requirements.sha256"
        requirements = paths.upstream / "requirements.txt"
        wanted = sha256_file(requirements)
        installed = marker.read_text("utf-8").strip() if marker.exists() else ""
        if installed != wanted:
            _run([str(python), "-m", "pip", "install", "-r", str(requirements)])
            marker.write_text(wanted, "utf-8")

    repo = Path(__file__).resolve().parents[3]
    criteria = paths.upstream / "data" / "criteria_data"
    references = paths.upstream / "data" / "test_data"
    manifest = {
        "schema_version": 1,
        "experiment_id": paths.experiment_id,
        "created_at": utc_now(),
        "upstream": {"url": UPSTREAM_URL, "commit": UPSTREAM_COMMIT},
        "dataset": {
            "query_sha256": sha256_file(query_file),
            "criteria_tree_sha256": _directory_hash(criteria),
            "reference_tree_sha256": _directory_hash(references),
            "task_ids": list(PILOT_TASK_IDS),
            "pilot_tasks_sha256": sha256_file(paths.tasks),
        },
        "agent": {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "mode": "research",
            "public_search": True,
            "embedding": EMBEDDING,
            "library": "empty isolated directory",
            "answer_max_sources": 5,
            "evidence_k": 6,
            "run_timeout_seconds": RUN_TIMEOUT_SECONDS,
            "limitations": {
                "web_search": "arXiv and Crossref discovery only",
                "model_calls_before_forced_finalization": 8,
                "observed_model_calls_may_include_finalization_and_verification": True,
                "maximum_agent_tool_calls": 14,
                "maximum_web_searches": 6,
                "maximum_web_reads": 6,
                "cloudflare_max_output_tokens": 4096,
            },
        },
        "evaluation": {
            "race_model": "gpt-5.5",
            "cleaner_model": "gpt-5.6-luna",
            "fact_model": "gpt-5.4-mini",
            "cost_ceiling_usd": 10.0,
            "pricing": {name: asdict(price) for name, price in OFFICIAL_OPENAI_PRICES.items()},
            "price_sources": PRICE_SOURCES,
            "windows_launcher_sha256": sha256_file(
                repo / "scripts" / "run_deepresearch_module.py"
            ),
        },
        "code": code_fingerprint(repo),
    }
    manifest["agent"]["configuration_sha256"] = sha256_bytes(
        json.dumps(manifest["agent"], sort_keys=True, separators=(",", ":")).encode()
    )
    write_json(paths.manifest, manifest)
    return manifest


def _directory_hash(directory: Path) -> str:
    rows = []
    if directory.exists():
        rows = [
            f"{path.relative_to(directory).as_posix()}:{sha256_file(path)}"
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ]
    return sha256_bytes("\n".join(rows).encode())


def load_env_file(path: Path | None) -> None:
    candidate = path or Path.cwd() / ".env"
    if not candidate.exists():
        return
    for line in candidate.read_text("utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def preflight(args) -> tuple[bool, dict]:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    load_env_file(args.env_file)
    checks: list[dict] = []

    def check(name: str, passed: bool, required: bool, detail: str):
        checks.append({"name": name, "passed": passed, "required": required, "detail": detail})

    generation = args.stage in {"all", "generation"}
    evaluation = args.stage in {"all", "evaluation", "race", "fact"}
    fact = args.stage in {"all", "evaluation", "fact"}
    check("prepared manifest", paths.manifest.exists(), True, str(paths.manifest))
    check("pinned tasks", paths.tasks.exists(), True, str(paths.tasks))
    check(
        "Cloudflare account",
        bool(os.getenv("CLOUDFLARE_ACCOUNT_ID")),
        generation,
        "configured" if os.getenv("CLOUDFLARE_ACCOUNT_ID") else "missing",
    )
    check(
        "Cloudflare token",
        bool(os.getenv("CLOUDFLARE_API_TOKEN")),
        generation,
        "configured" if os.getenv("CLOUDFLARE_API_TOKEN") else "missing",
    )
    backend = os.getenv("LLM_BACKEND", "openai").casefold()
    judge_key = "OPENROUTER_API_KEY" if backend == "openrouter" else "OPENAI_API_KEY"
    check(
        f"judge credential ({judge_key})",
        bool(os.getenv(judge_key)),
        evaluation,
        "configured" if os.getenv(judge_key) else "missing; generation remains available",
    )
    check(
        "Jina credential",
        bool(os.getenv("JINA_API_KEY")),
        fact,
        "configured" if os.getenv("JINA_API_KEY") else "missing; FACT scraping cannot run",
    )
    python = evaluator_python(paths)
    check("evaluator environment", python.exists(), evaluation, str(python))
    check(
        "empty isolated library",
        not (paths.app_state / "papers").exists()
        or not any((paths.app_state / "papers").iterdir()),
        generation,
        str(paths.app_state / "papers"),
    )
    okay = all(item["passed"] or not item["required"] for item in checks)
    result = {"ok": okay, "stage": args.stage, "checks": checks, "checked_at": utc_now()}
    if getattr(args, "save", True):
        write_json(paths.experiment / f"preflight-{args.stage}.json", result)
    return okay, result


def _compact_source_label(source: dict) -> str:
    name = str(source.get("docname") or source.get("name") or "source").strip()
    name = re.sub(r"\s+", " ", name)
    if len(name) > 72:
        name = name[:69].rstrip() + "…"
    page = source.get("page")
    return f"{name}, p. {page}" if page and f"p. {page}" not in name else name


def resolve_citations(answer: str, sources: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Resolve stable internal IDs without inventing a URL or changing prose."""
    registry = {str(item.get("id")): item for item in sources if item.get("id")}
    mapping: list[dict] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    def replace(match: re.Match) -> str:
        source_id = match.group(1)
        source = registry.get(source_id)
        url = str((source or {}).get("url") or "").strip()
        if not source or not re.match(r"^https?://", url) or _unreadable_source(source):
            if source_id not in unresolved:
                unresolved.append(source_id)
            return f"[unresolved citation: {source_id}]"
        if source_id not in seen:
            mapping.append({
                "source_id": source_id,
                "url": url,
                "label": _compact_source_label(source),
                "excerpt": source.get("excerpt") or source.get("summary") or "",
                "provenance": source.get("provenance"),
            })
            seen.add(source_id)
        return f"[{_compact_source_label(source)}]({url})"

    converted = re.sub(r"\b(src-[a-f0-9]{24})\b", replace, answer)
    return converted, mapping, unresolved


def _unreadable_source(source: dict) -> bool:
    excerpt = str(source.get("excerpt") or source.get("summary") or "").casefold()
    url = str(source.get("url") or "").casefold()
    markers = (
        "captcha",
        "confirm you are a human",
        "request unsuccessful",
        "incapsula incident",
        "access denied",
        "page not found",
        "botmanager",
    )
    return any(marker in excerpt or marker in url for marker in markers)


def _artifact_path(paths: ExperimentPaths, task_id: int) -> Path:
    return paths.task_results / f"task-{task_id}.json"


def _selected_tasks(paths: ExperimentPaths, only: list[int] | None) -> list[dict]:
    tasks = load_pilot_tasks(paths.tasks)
    if only:
        allowed = set(only)
        tasks = [task for task in tasks if _task_id(task) in allowed]
    return tasks


async def generate_async(args) -> dict:
    from paperqa.benchmarks.cost_callback import BenchmarkCostCallback
    from paperqa.research.providers import model_configured
    from paperqa.research.service import ResearchService, execute
    from paperqa.research.storage import RunRecord

    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    load_env_file(args.env_file)
    okay, result = preflight(
        argparse.Namespace(
            workspace=paths.workspace,
            experiment_id=paths.experiment_id,
            env_file=args.env_file,
            stage="generation",
            save=True,
        )
    )
    if not okay or not model_configured(MODEL):
        raise RuntimeError(f"Generation preflight failed: {json.dumps(result)}")
    app = ResearchService(paths.app_state)
    ledger = CostLedger(paths.ledger, getattr(args, "cost_ceiling", 10.0))
    app.paper_directory.mkdir(parents=True, exist_ok=True)
    tasks = _selected_tasks(paths, args.only)
    attempted = completed = 0
    try:
        for task in tasks:
            task_id = _task_id(task)
            target = _artifact_path(paths, task_id)
            existing = json.loads(target.read_text("utf-8")) if target.exists() else None
            if existing and existing.get("generation", {}).get("terminal"):
                continue
            prompt = _prompt(task)
            cost_callback = BenchmarkCostCallback(
                ledger,
                MODEL,
                maximum_output_tokens=4096,
                stage=f"generation:task-{task_id}",
            )
            run_id = hashlib.sha256(
                f"{paths.experiment_id}:{task_id}:attempt-1".encode()
            ).hexdigest()[:12]
            job = app.jobs.get(run_id)
            if job is None:
                conversation_id = app.begin_turn(prompt)
                job = RunRecord(
                    id=run_id,
                    conversation_id=conversation_id,
                    question=prompt,
                    model=MODEL,
                    embedding=EMBEDDING,
                    mode="research",
                    route="research",
                )
                app.jobs[job.id] = job
            elif job.question != prompt:
                raise RuntimeError(
                    f"Saved run {run_id} does not preserve task {task_id}'s prompt."
                )
            if job.pending_question:
                _save_task_artifact(paths, task, job, timed_out=False)
                continue
            if job.status in {"failed", "stopped"}:
                _save_task_artifact(paths, task, job, timed_out=False)
                continue
            if job.status == "paused" and not args.resume_provider:
                _save_task_artifact(paths, task, job, timed_out=False)
                continue
            payload = {
                "web_search": True,
                "answer_max_sources": 5,
                "evidence_k": 6,
                "reasoning_effort": REASONING_EFFORT,
            }
            job.payload = payload
            job.stop_requested = False
            attempted += 1
            started = time.perf_counter()
            timed_out = False
            try:
                await asyncio.wait_for(
                    execute(
                        job,
                        payload,
                        app,
                        resume={} if job.status == "paused" else None,
                        callbacks=[cost_callback],
                    ),
                    timeout=args.timeout,
                )
            except TimeoutError:
                timed_out = True
                job.stop_requested = True
                job.status = "stopped"
                job.pause_reason = "benchmark_timeout"
                job.error = f"Benchmark outer timeout reached after {args.timeout} seconds."
                app.save_job(job)
                app.finish_turn(job)
            artifact = _save_task_artifact(
                paths,
                task,
                job,
                timed_out=timed_out,
                wall_seconds=time.perf_counter() - started,
            )
            if artifact["generation"]["complete"]:
                completed += 1
            _write_export(paths)
            if job.pause_reason in {
                "provider_quota",
                "provider_connection",
                "rate_limit",
                "spending_ceiling",
            }:
                break
    finally:
        app.close()
    _write_export(paths)
    return {"attempted": attempted, "completed": completed, "total": len(tasks)}


def _save_task_artifact(
    paths: ExperimentPaths,
    task: dict,
    job,
    *,
    timed_out: bool,
    wall_seconds: float | None = None,
) -> dict:
    raw_answer = str((job.result or {}).get("answer") or "")
    article, mappings, unresolved = resolve_citations(
        raw_answer, list((job.result or {}).get("sources") or [])
    )
    terminal = job.status in {"complete", "unsure", "failed", "stopped"} or bool(
        job.pending_question
    )
    generation = {
        "run_id": job.id,
        "conversation_id": job.conversation_id,
        "attempt": 1,
        "status": job.status,
        "complete": job.status in {"complete", "unsure"} and bool(raw_answer),
        "terminal": terminal,
        "finish_reason": job.pause_reason or job.status,
        "pending_question": job.pending_question,
        "error": job.error,
        "timed_out": timed_out,
        "wall_seconds": (
            max(
                float(wall_seconds or 0),
                float(job.usage.get("seconds", 0) or 0),
            )
        ),
        "saved_at": utc_now(),
    }
    previous_path = _artifact_path(paths, _task_id(task))
    previous = json.loads(previous_path.read_text("utf-8")) if previous_path.exists() else {}
    prior_segments = list(((previous.get("attempts") or [{}])[0]).get("segments", []))
    if previous.get("generation") and not prior_segments:
        prior_segments.append(previous["generation"])
    if not prior_segments or prior_segments[-1].get("status") != generation["status"]:
        prior_segments.append(generation)
    attempts = [
        {
            "attempt": 1,
            "run_id": job.id,
            "segments": prior_segments,
            "selected": True,
        }
    ]
    artifact = {
        "id": task.get("id", _task_id(task)),
        "numeric_id": _task_id(task),
        "prompt": _prompt(task),
        "article": article,
        "raw_answer": raw_answer,
        "citation_mappings": mappings,
        "unresolved_citations": unresolved,
        "generation": generation,
        "attempts": attempts,
        "usage": dict(job.usage),
        "events": list(job.events),
        "plan": list((job.result or {}).get("agent", {}).get("plan") or []),
        "evidence": list((job.result or {}).get("sources") or []),
        "flags": _generation_flags(job, unresolved, timed_out),
    }
    write_json(_artifact_path(paths, _task_id(task)), artifact)
    return artifact


def _generation_flags(job, unresolved: list[str], timed_out: bool) -> list[str]:
    flags = []
    titles = [str(item.get("title", "")).casefold() for item in job.events]
    if any(
        "forced" in title
        or "finalizing with available evidence" in title
        or "model-call ceiling" in str(item.get("detail", "")).casefold()
        for title, item in zip(titles, job.events, strict=False)
    ):
        flags.append("forced_finalization")
    if timed_out:
        flags.append("outer_timeout")
    if unresolved:
        flags.append("unresolved_citations")
    if job.status == "unsure":
        flags.extend(["incomplete_answer", "unsupported_or_unverified_claims"])
    if any(
        item.get("stage") == "tool_error"
        and any(code in str(item.get("detail", "")) for code in ("403", "404", "unreadable"))
        for item in job.events
    ):
        flags.append("inaccessible_sources")
    if job.pending_question:
        flags.append("clarification_unfinished")
    if job.pause_reason:
        flags.append(f"pause:{job.pause_reason}")
    if int(job.usage.get("output_tokens", 0) or 0) >= 4090:
        flags.append("possible_output_truncation")
    search_details = [
        item.get("detail", "") for item in job.events if item.get("title") == "search_web"
    ]
    if len(search_details) != len(set(search_details)):
        flags.append("repeated_search")
    if any("repeated web search blocked" in title for title in titles):
        flags.append("repeated_search")
    return list(dict.fromkeys(flags))


def _write_export(paths: ExperimentPaths) -> None:
    rows = []
    for task in read_jsonl(paths.tasks) if paths.tasks.exists() else []:
        artifact = _artifact_path(paths, _task_id(task))
        if not artifact.exists():
            continue
        value = json.loads(artifact.read_text("utf-8"))
        if value.get("generation", {}).get("complete"):
            rows.append({
                "id": task.get("id", _task_id(task)),
                "prompt": _prompt(task),
                "article": value.get("article", ""),
            })
    write_jsonl(paths.export, rows)


def generate(args) -> dict:
    return asyncio.run(generate_async(args))


def generate_batch(args) -> dict:
    """Run each task in a killable process so the ten-minute timeout is absolute."""
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    load_env_file(args.env_file)
    tasks = _selected_tasks(paths, args.only)
    attempted = completed = 0
    blocked = None
    for task in tasks:
        task_id = _task_id(task)
        target = _artifact_path(paths, task_id)
        existing = json.loads(target.read_text("utf-8")) if target.exists() else None
        if existing and existing.get("generation", {}).get("terminal"):
            completed += int(bool(existing.get("generation", {}).get("complete")))
            continue
        command = [
            sys.executable,
            "-m",
            "paperqa.benchmarks.deepresearch",
            "--workspace",
            str(paths.workspace),
            "--experiment-id",
            paths.experiment_id,
        ]
        if args.env_file:
            command.extend(["--env-file", str(args.env_file)])
        command.extend([
            "generate-one",
            "--task-id",
            str(task_id),
            "--timeout",
            str(args.timeout),
            "--cost-ceiling",
            str(args.cost_ceiling),
        ])
        if args.resume_provider:
            command.append("--resume-provider")
        attempted += 1
        try:
            process = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[3],
                env=dict(os.environ),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
            if process.returncode:
                blocked = f"Task {task_id} worker failed: {process.stdout[-1000:]}"
        except subprocess.TimeoutExpired:
            CostLedger(paths.ledger, args.cost_ceiling).retain_open_reservations(
                f"Task {task_id} worker was terminated at the {args.timeout}-second outer timeout.",
                stage_prefix="generation",
            )
            _recover_interrupted_task(paths, task, args.cost_ceiling, timed_out=True)
            blocked = f"Task {task_id} exceeded the {args.timeout}-second outer timeout."
        artifact = json.loads(target.read_text("utf-8")) if target.exists() else None
        if artifact:
            completed += int(bool(artifact.get("generation", {}).get("complete")))
            pause = artifact.get("generation", {}).get("finish_reason")
            if pause == "restart":
                CostLedger(paths.ledger, args.cost_ceiling).retain_open_reservations(
                    f"Task {task_id} provider request was interrupted; charge outcome is uncertain.",
                    stage_prefix="generation",
                )
            if pause in {
                "quota",
                "provider_quota",
                "provider_connection",
                "rate_limit",
                "spending_ceiling",
                "benchmark_timeout",
                "restart",
            }:
                blocked = blocked or f"Task {task_id} paused: {pause}."
        if blocked:
            break
    _write_export(paths)
    CostLedger(paths.ledger, args.cost_ceiling).retain_open_reservations(
        "The batch ended with no live worker owning this request; charge outcome is uncertain.",
        stage_prefix="generation",
    )
    return {
        "attempted": attempted,
        "completed": completed,
        "selected": len(tasks),
        "blocked": blocked,
    }


def _recover_interrupted_task(
    paths: ExperimentPaths, task: dict, cost_ceiling: float, *, timed_out: bool
) -> dict | None:
    from paperqa.research.service import ResearchService

    app = ResearchService(paths.app_state)
    try:
        run_id = hashlib.sha256(
            f"{paths.experiment_id}:{_task_id(task)}:attempt-1".encode()
        ).hexdigest()[:12]
        job = app.jobs.get(run_id)
        if not job:
            return None
        if timed_out:
            job.status = "stopped"
            job.pause_reason = "benchmark_timeout"
            job.error = f"Benchmark worker terminated after {RUN_TIMEOUT_SECONDS} seconds."
            app.save_job(job)
            app.finish_turn(job)
        return _save_task_artifact(paths, task, job, timed_out=timed_out)
    finally:
        app.close()


def _judge_configuration() -> tuple[str, str, str]:
    backend = os.getenv("LLM_BACKEND", "openai").casefold()
    if backend == "openrouter":
        return (
            backend,
            os.environ["OPENROUTER_API_KEY"],
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    if backend != "openai":
        raise ValueError("LLM_BACKEND must be 'openai' or 'openrouter'.")
    return (
        backend,
        os.environ["OPENAI_API_KEY"],
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def _evaluation_env(*, backend: str, proxy_url: str, base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env["LLM_BACKEND"] = backend
    if backend == "openrouter":
        env["OPENROUTER_BASE_URL"] = proxy_url
        env["OPENROUTER_API_KEY"] = "metered-locally"
    else:
        env["OPENAI_BASE_URL"] = proxy_url
        env["OPENAI_API_KEY"] = "metered-locally"
    return env


def evaluate(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    load_env_file(args.env_file)
    preflight_stage = "evaluation" if args.stage in {"all", "evaluation"} else args.stage
    okay, preflight_result = preflight(
        argparse.Namespace(
            workspace=paths.workspace,
            experiment_id=paths.experiment_id,
            env_file=args.env_file,
            stage=preflight_stage,
            save=True,
        )
    )
    if not okay:
        return {"status": "blocked", "preflight": preflight_result}
    if not paths.export.exists() or not read_jsonl(paths.export):
        return {"status": "blocked", "detail": "No completed generated reports are available."}
    backend, api_key, upstream = _judge_configuration()
    ledger = CostLedger(paths.ledger, args.cost_ceiling)
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    result = {"status": "complete", "stages": {}, "started_at": utc_now()}
    if args.stage in {"all", "evaluation", "race"}:
        result["stages"]["race"] = _run_race(
            paths, ledger, backend, api_key, upstream, args.resume
        )
    if args.stage in {"all", "evaluation", "fact"}:
        result["stages"]["fact"] = _run_fact(
            paths, ledger, backend, api_key, upstream, args.resume
        )
    result["completed_at"] = utc_now()
    result["ledger"] = ledger.snapshot()
    write_json(paths.evaluation / "summary.json", result)
    return result


def _run_race(paths, ledger, backend, api_key, upstream, resume: bool) -> dict:
    marker = paths.evaluation / "race-complete.json"
    if resume and marker.exists():
        return json.loads(marker.read_text("utf-8"))
    python = evaluator_python(paths)
    output = paths.evaluation / "race"
    output.mkdir(parents=True, exist_ok=True)
    raw_data = output / "raw"
    cleaned_data = output / "cleaned"
    raw_data.mkdir(parents=True, exist_ok=True)
    cleaned_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.export, raw_data / f"{paths.export.stem}.jsonl")
    command = [
        str(python),
        "deepresearch_bench_race.py",
        paths.export.stem,
        "--raw_data_dir",
        str(raw_data),
        "--cleaned_data_dir",
        str(cleaned_data),
        "--query_file",
        str(paths.tasks),
        "--output_dir",
        str(output),
        "--max_workers",
        "1",
        "--only_en",
        "--limit",
        str(len(read_jsonl(paths.export))),
    ]
    started = time.perf_counter()
    with metered_proxy(ledger=ledger, upstream=upstream, api_key=api_key, stage="race") as proxy:
        log = _run(
            command,
            cwd=paths.upstream,
            env=_evaluation_env(backend=backend, proxy_url=proxy, base=dict(os.environ)),
        )
    value = {
        "status": "complete",
        "seconds": round(time.perf_counter() - started, 3),
        "command": command,
        "log_tail": log[-4000:],
    }
    write_json(marker, value)
    return value


def _run_fact(paths, ledger, backend, api_key, upstream, resume: bool) -> dict:
    marker = paths.evaluation / "fact-complete.json"
    if resume and marker.exists():
        return json.loads(marker.read_text("utf-8"))
    python = evaluator_python(paths)
    output = paths.evaluation / "fact"
    output.mkdir(parents=True, exist_ok=True)
    stages = [
        (
            "extract",
            [
                "-m",
                "utils.extract",
                "--raw_data_path",
                str(paths.export),
                "--output_path",
                str(output / "extracted.jsonl"),
                "--query_data_path",
                str(paths.tasks),
                "--n_total_process",
                "1",
            ],
        ),
        (
            "deduplicate",
            [
                "-m",
                "utils.deduplicate",
                "--raw_data_path",
                str(output / "extracted.jsonl"),
                "--output_path",
                str(output / "deduplicated.jsonl"),
                "--query_data_path",
                str(paths.tasks),
                "--n_total_process",
                "1",
            ],
        ),
        (
            "scrape",
            [
                "-m",
                "utils.scrape",
                "--raw_data_path",
                str(output / "deduplicated.jsonl"),
                "--output_path",
                str(output / "scraped.jsonl"),
                "--n_total_process",
                "1",
            ],
        ),
        (
            "validate",
            [
                "-m",
                "utils.validate",
                "--raw_data_path",
                str(output / "scraped.jsonl"),
                "--output_path",
                str(output / "validated.jsonl"),
                "--query_data_path",
                str(paths.tasks),
                "--n_total_process",
                "1",
            ],
        ),
    ]
    logs = {}
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[3]
    windows_launcher = repo / "scripts" / "run_deepresearch_module.py"
    with metered_proxy(ledger=ledger, upstream=upstream, api_key=api_key, stage="fact") as proxy:
        env = _evaluation_env(backend=backend, proxy_url=proxy, base=dict(os.environ))
        for name, arguments in stages:
            stage_marker = output / f".{name}-complete"
            if resume and stage_marker.exists():
                logs[name] = "resumed: already complete"
                continue
            command = (
                [str(python), str(windows_launcher), arguments[1], *arguments[2:]]
                if os.name == "nt" and arguments[:1] == ["-m"]
                else [str(python), *arguments]
            )
            logs[name] = _run(command, cwd=paths.upstream, env=env)[-4000:]
            stage_marker.write_text(utc_now(), "utf-8")
    diagnostics = fact_diagnostics(output / "validated.jsonl")
    write_json(output / "diagnostics.json", diagnostics)
    if diagnostics["scorable_reports"] and diagnostics["evaluated_citations"]:
        arguments = [
            "-m",
            "utils.stat",
            "--input_path",
            str(output / "validated.jsonl"),
            "--output_path",
            str(output / "fact_result.txt"),
        ]
        stage_marker = output / ".stat-complete"
        if not (resume and stage_marker.exists()):
            command = (
                [str(python), str(windows_launcher), arguments[1], *arguments[2:]]
                if os.name == "nt"
                else [str(python), *arguments]
            )
            logs["stat"] = _run(command, cwd=paths.upstream, env=dict(os.environ))[-4000:]
            stage_marker.write_text(utc_now(), "utf-8")
    else:
        (output / "fact_result.txt").write_text(
            "incomplete: no citations were scorable by the official FACT aggregation\n",
            "utf-8",
        )
        logs["stat"] = (
            "not run: official stat divides by zero when no report/citation is scorable"
        )
    value = {
        "status": "complete",
        "seconds": round(time.perf_counter() - started, 3),
        "logs": logs,
        "diagnostics": diagnostics,
    }
    write_json(marker, value)
    return value


def fact_diagnostics(validated_path: Path) -> dict:
    rows = read_jsonl(validated_path) if validated_path.exists() else []
    no_citations = unreadable = validation_errors = unknown = evaluated = supported = 0
    per_task = []
    for row in rows:
        citations = row.get("citations")
        if not citations:
            no_citations += 1
        task_evaluated = task_supported = task_errors = task_unknown = 0
        for citation in (row.get("citations_deduped") or {}).values():
            if citation.get("validate_error") is not None:
                validation_errors += 1
                task_errors += 1
                if not citation.get("url_content"):
                    unreadable += 1
                continue
            for result in citation.get("validate_res") or []:
                if result.get("result") == "unknown":
                    unknown += 1
                    task_unknown += 1
                else:
                    evaluated += 1
                    task_evaluated += 1
                    if result.get("result") == "supported":
                        supported += 1
                        task_supported += 1
        per_task.append({
            "id": row.get("id"),
            "extracted_citations": len(citations) if isinstance(citations, list) else 0,
            "evaluated_citations": task_evaluated,
            "supported_citations": task_supported,
            "unknown_citations": task_unknown,
            "validation_errors": task_errors,
            "support_rate": task_supported / task_evaluated if task_evaluated else None,
        })
    scorable = sum(1 for item in per_task if item["evaluated_citations"])
    return {
        "reports": len(rows),
        "scorable_reports": scorable,
        "reports_without_citations": no_citations,
        "unreadable_sources": unreadable,
        "validation_errors": validation_errors,
        "unknown_citations": unknown,
        "evaluated_citations": evaluated,
        "supported_citations": supported,
        "support_rate": supported / evaluated if evaluated else None,
        "per_task": per_task,
    }


def _markdown_html(value: str) -> str:
    try:
        from markdown_it import MarkdownIt

        return (
            MarkdownIt("commonmark", {"html": False, "linkify": True})
            .enable("table")
            .render(value)
        )
    except ImportError:
        escaped = html.escape(value)
        escaped = re.sub(
            r"\[([^\]]+)\]\((https?://[^)]+)\)",
            r'<a href="\2" target="_blank" rel="noopener">\1</a>',
            escaped,
        )
        return "".join(f"<p>{line}</p>" for line in escaped.split("\n\n"))


def _read_optional(path: Path) -> str | None:
    return path.read_text("utf-8") if path.exists() else None


def report(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    tasks = read_jsonl(paths.tasks) if paths.tasks.exists() else []
    items = []
    attempted = completed = scored = 0
    race_rows_path = paths.evaluation / "race" / "raw_results.jsonl"
    race_by_id = (
        {_task_id(row): row for row in read_jsonl(race_rows_path)}
        if race_rows_path.exists()
        else {}
    )
    diagnostics_path = paths.evaluation / "fact" / "diagnostics.json"
    fact_diagnostic = (
        json.loads(diagnostics_path.read_text("utf-8")) if diagnostics_path.exists() else {}
    )
    fact_by_id = {_task_id(row): row for row in fact_diagnostic.get("per_task", [])}
    ledger_snapshot = CostLedger(paths.ledger, args.cost_ceiling).snapshot()
    for task in tasks:
        artifact_path = _artifact_path(paths, _task_id(task))
        artifact = (
            json.loads(artifact_path.read_text("utf-8")) if artifact_path.exists() else None
        )
        if artifact:
            if "attempts" not in artifact:
                artifact["attempts"] = [
                    {
                        "attempt": 1,
                        "run_id": artifact.get("generation", {}).get("run_id"),
                        "segments": [artifact.get("generation", {})],
                        "selected": True,
                    }
                ]
                write_json(artifact_path, artifact)
            article, mappings, unresolved = resolve_citations(
                str(artifact.get("raw_answer") or artifact.get("article") or ""),
                list(artifact.get("evidence") or []),
            )
            artifact["article"] = article
            artifact["citation_mappings"] = mappings
            artifact["unresolved_citations"] = unresolved
            artifact["flags"] = _saved_artifact_flags(artifact)
            artifact["generation"]["wall_seconds"] = round(
                max(
                    float(artifact["generation"].get("wall_seconds", 0) or 0),
                    float(artifact.get("usage", {}).get("seconds", 0) or 0),
                ),
                2,
            )
            artifact["cost"] = _task_cost(
                ledger_snapshot,
                _task_id(task),
                assign_legacy=_task_id(task) == PILOT_TASK_IDS[0],
            )
            write_json(artifact_path, artifact)
            attempted += 1
            completed += int(bool(artifact.get("generation", {}).get("complete")))
        item = artifact or {
            "id": task.get("id"),
            "prompt": _prompt(task),
            "generation": {"status": "not attempted"},
        }
        item["evaluation"] = {
            "race": race_by_id.get(_task_id(task)),
            "fact": fact_by_id.get(_task_id(task)),
            "judge_explanation": (
                "The pinned official evaluator saves dimension scores but does not expose its criterion analyses."
                if _task_id(task) in race_by_id
                else None
            ),
        }
        items.append(item)
    _write_export(paths)
    race_text = _read_optional(paths.evaluation / "race" / "race_result.txt")
    fact_text = _read_optional(paths.evaluation / "fact" / "fact_result.txt")
    scored = sum(
        1
        for item in items
        if item.get("evaluation", {}).get("race")
        and (item.get("evaluation", {}).get("fact") or {}).get("evaluated_citations", 0)
    )
    evaluation_complete = (
        bool(race_text and fact_text)
        and completed == len(tasks)
        and len(race_by_id) == completed
        and int(fact_diagnostic.get("reports", 0)) == completed
        and not str(fact_text).startswith("incomplete:")
    )
    summary = {
        "experiment_id": paths.experiment_id,
        "generated_at": utc_now(),
        "counts": {
            "selected": len(tasks),
            "attempted": attempted,
            "completed": completed,
            "scored": scored,
        },
        "evaluation_complete": evaluation_complete,
        "race_result": race_text,
        "fact_result": fact_text,
        "fact_diagnostics": fact_diagnostic,
        "cost_ledger": ledger_snapshot,
        "tasks": items,
    }
    write_json(paths.report_json, summary)
    cards = []
    summary_rows = []
    for item in items:
        generation = item.get("generation", {})
        evaluation = item.get("evaluation", {})
        usage = item.get("usage", {})
        cost = item.get("cost", {})
        wall_seconds = generation.get("wall_seconds")
        seconds_text = f"{float(wall_seconds):.2f}" if wall_seconds is not None else "—"
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('id')))}</td>"
            f"<td>{html.escape(str(generation.get('status', 'not attempted')))}</td>"
            f"<td>{html.escape(seconds_text)}</td>"
            f"<td>{html.escape(str(usage.get('model_calls', '—')))}</td>"
            f"<td>{html.escape(str(usage.get('tool_calls', '—')))}</td>"
            f"<td>{html.escape(str(usage.get('tokens', '—')))}</td>"
            f"<td>{len(item.get('citation_mappings', []))}</td>"
            f"<td>${float(cost.get('spent_usd', 0)):.6f}</td>"
            "</tr>"
        )
        links = "".join(
            f'<li><a href="{html.escape(str(source["url"]))}">{html.escape(str(source["label"]))}</a></li>'
            for source in item.get("citation_mappings", [])
        )
        cards.append(f"""
        <section><h2>Task {html.escape(str(item.get("id")))}</h2>
        <p class="status">{html.escape(str(generation.get("status")))} · {html.escape(seconds_text)} s · {html.escape(str(usage.get("model_calls", "—")))} model calls · {html.escape(str(usage.get("tool_calls", "—")))} tool calls · {html.escape(str(usage.get("tokens", "—")))} tokens · {len(item.get("citation_mappings", []))} resolved citations · {len(item.get("unresolved_citations", []))} unresolved · ${float(cost.get("spent_usd", 0)):.6f}</p>
        <details><summary>Original prompt</summary><pre>{html.escape(str(item.get("prompt", "")))}</pre></details>
        <article>{_markdown_html(str(item.get("article", "No completed report.")))}</article>
        <details><summary>Per-task evaluation</summary><pre>{html.escape(json.dumps(evaluation, ensure_ascii=False, indent=2))}</pre></details>
        <details><summary>Resolved sources</summary><ul>{links or "<li>None</li>"}</ul></details>
        <details><summary>Observable agent events</summary><pre>{html.escape(json.dumps(item.get("events", []), ensure_ascii=False, indent=2))}</pre></details>
        </section>""")
    incomplete = (
        ""
        if summary["evaluation_complete"]
        else '<div class="warning">Evaluation is incomplete; no five-task aggregate is reported.</div>'
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>PaperQA DeepResearch Bench pilot</title><style>
    body{{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:auto;padding:32px;color:#18202a;background:#f5f7fa}} section{{background:white;border:1px solid #dce2e9;border-radius:12px;padding:24px;margin:24px 0;box-shadow:0 2px 8px #0000000c}} article{{overflow-wrap:anywhere}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccd4dd;padding:8px;text-align:left}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;padding:12px;border-radius:8px}} .summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .summary div{{background:white;padding:16px;border-radius:10px;border:1px solid #dce2e9}} .warning{{background:#fff3cd;border:1px solid #e8cf78;padding:14px;border-radius:8px}} .status{{color:#586574}} a{{color:#1267b0}}</style></head><body>
    <h1>PaperQA · DeepResearch Bench five-task pilot</h1>{incomplete}
    <div class="summary"><div>Selected<br><strong>{len(tasks)}</strong></div><div>Attempted<br><strong>{attempted}</strong></div><div>Completed<br><strong>{completed}</strong></div><div>Scored<br><strong>{scored}</strong></div></div>
    <section><h2>Generation summary</h2><table><thead><tr><th>Task</th><th>Status</th><th>Seconds</th><th>Model calls</th><th>Tool calls</th><th>Tokens</th><th>Citations</th><th>Reported cost</th></tr></thead><tbody>{"".join(summary_rows)}</tbody></table></section>
    <section><h2>Official evaluation output</h2><p>RACE values are reference-relative scores, not percentages of questions answered correctly. Generation and evaluation latency are recorded separately.</p><h3>RACE</h3><pre>{html.escape(race_text or "Not completed")}</pre><h3>FACT</h3><pre>{html.escape(fact_text or "Not completed")}</pre><h3>FACT failure accounting</h3><pre>{html.escape(json.dumps(fact_diagnostic, ensure_ascii=False, indent=2) if fact_diagnostic else "Not completed")}</pre></section>
    {"".join(cards)}</body></html>"""
    paths.report_html.write_text(document, "utf-8")
    return summary


def _saved_artifact_flags(artifact: dict) -> list[str]:
    flags = list(artifact.get("flags") or [])
    events = artifact.get("events") or []
    titles = [str(event.get("title", "")).casefold() for event in events]
    if any("finalizing with available evidence" in title for title in titles):
        flags.append("forced_finalization")
    if any("repeated web search blocked" in title for title in titles):
        flags.append("repeated_search")
    if artifact.get("generation", {}).get("status") == "unsure":
        flags.extend(["incomplete_answer", "unsupported_or_unverified_claims"])
    if artifact.get("unresolved_citations"):
        flags.extend(["unresolved_citations", "unsupported_or_unverified_claims"])
    if any(_unreadable_source(source) for source in artifact.get("evidence") or []):
        flags.append("inaccessible_sources")
    if any(
        event.get("stage") == "tool_error"
        and any(code in str(event.get("detail", "")) for code in ("403", "404", "unreadable"))
        for event in events
    ):
        flags.append("inaccessible_sources")
    return list(dict.fromkeys(flags))


def _task_cost(ledger: dict, task_id: int, *, assign_legacy: bool = False) -> dict:
    entries = [
        entry
        for entry in ledger.get("entries", [])
        if entry.get("stage") == f"generation:task-{task_id}"
        or (assign_legacy and entry.get("stage") == "generation")
    ]
    return {
        "requests": len(entries),
        "spent_usd": sum(float(entry.get("actual_usd", 0) or 0) for entry in entries),
        "retained_reservations_usd": sum(
            float(entry.get("reserved_usd", 0) or 0)
            for entry in entries
            if entry.get("status") in {"reserved", "uncertain"}
        ),
        "reported": all(entry.get("status") == "reconciled" for entry in entries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--env-file", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Pin data and create evaluator env.")
    prepare_parser.add_argument("--skip-evaluator-install", action="store_true")
    prepare_parser.set_defaults(handler=prepare)

    preflight_parser = subparsers.add_parser("preflight", help="Validate one or all stages.")
    preflight_parser.add_argument(
        "--stage", choices=("all", "generation", "evaluation", "race", "fact"), default="all"
    )
    preflight_parser.set_defaults(handler=lambda value: preflight(value)[1])

    generate_parser = subparsers.add_parser("generate", help="Generate the frozen reports.")
    generate_parser.add_argument("--only", type=int, action="append", choices=PILOT_TASK_IDS)
    generate_parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_SECONDS)
    generate_parser.add_argument("--cost-ceiling", type=float, default=10.0)
    generate_parser.add_argument(
        "--resume-provider",
        action="store_true",
        help="Explicitly resume paused provider work; clarification pauses remain unfinished.",
    )
    generate_parser.set_defaults(handler=generate_batch)

    generate_one_parser = subparsers.add_parser("generate-one", help=argparse.SUPPRESS)
    generate_one_parser.add_argument(
        "--task-id", type=int, required=True, choices=PILOT_TASK_IDS
    )
    generate_one_parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_SECONDS)
    generate_one_parser.add_argument("--cost-ceiling", type=float, default=10.0)
    generate_one_parser.add_argument("--resume-provider", action="store_true")
    generate_one_parser.set_defaults(
        handler=lambda value: generate(
            argparse.Namespace(**{
                **vars(value),
                "only": [value.task_id],
            })
        )
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Run official RACE/FACT stages.")
    evaluate_parser.add_argument(
        "--stage", choices=("all", "evaluation", "race", "fact"), default="all"
    )
    evaluate_parser.add_argument("--cost-ceiling", type=float, default=10.0)
    evaluate_parser.add_argument("--resume", action="store_true")
    evaluate_parser.set_defaults(handler=evaluate)

    report_parser = subparsers.add_parser("report", help="Build JSON and HTML results.")
    report_parser.add_argument("--cost-ceiling", type=float, default=10.0)
    report_parser.set_defaults(handler=report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.handler(args)
        displayed = result
        if args.command == "report":
            paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
            displayed = {
                "counts": result["counts"],
                "evaluation_complete": result["evaluation_complete"],
                "results_json": str(paths.report_json),
                "report_html": str(paths.report_html),
            }
        print(json.dumps(displayed, ensure_ascii=True, indent=2))
        if args.command == "preflight":
            return 0 if result.get("ok") else 2
        return 0
    except KeyboardInterrupt:
        print("Interrupted; saved stage artifacts can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
