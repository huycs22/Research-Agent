"""Reproducible LitQA2 closed-corpus benchmark.

Run with ``python -m paperqa.benchmarks.litqa --help`` or ``pqa-litqa``.
Evaluation labels remain in benchmark artifacts and are never added to an agent
prompt or retrieval index.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import aiohttp
from langchain_core.documents import Document

from paperqa.research.library import (
    BM25Retriever,
    HybridRetriever,
    PaperLoader,
    ZeroSafeBM25Retriever,
    _expand_passages,
    digest,
    embeddings_for,
    split_pages,
)
from paperqa.research.parsing import PARSER_VERSION
from paperqa.research.safety import PublicResolver, public_url

from .cost import OFFICIAL_OPENAI_PRICES, PRICE_SOURCES, CostLedger
from .cost_callback import BenchmarkCostCallback
from .deepresearch import code_fingerprint, load_env_file

DATASET_REPOSITORY = "futurehouse/lab-bench"
DATASET_REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
DATASET_FILE = "LitQA2/train-00000-of-00001.parquet"
DATASET_URL = (
    f"https://huggingface.co/datasets/{DATASET_REPOSITORY}/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILE}"
)
EXPERIMENT_ID = "paperqa-litqa2-closed-v1"
DEFAULT_WORKSPACE = Path(".litqa-bench")
MODEL = "cloudflare/@cf/zai-org/glm-4.7-flash"
MINILM = "st-sentence-transformers/all-MiniLM-L6-v2"
RETRIEVERS = ("bm25", "minilm", "hybrid")
SCOPES = ("paired", "shared")
RETRIEVAL_K = 20
ANSWER_EVIDENCE_K = 6
CHUNK_TOKENS = 254
OVERLAP_TOKENS = 32
RUN_TIMEOUT_SECONDS = 240
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_DATASET_BYTES = 10 * 1024 * 1024
ACQUISITION_TIMEOUT_SECONDS = 60
PILOT_SIZE = 20
PILOT_SEED = "paperqa-litqa2-closed-v1"
PRICE_VERIFIED_AT = "2026-09-26"
ACQUISITION_RESOLVER_VERSION = 2
CHOICE_INSTRUCTION = (
    "Use only the associated papers in the closed library. Explain the evidence with exact "
    "source-ID citations, then end with exactly `FINAL_CHOICE: <label>`. Choose the "
    "Insufficient evidence option when the papers do not support another choice."
)


@dataclass(frozen=True)
class ExperimentPaths:
    workspace: Path
    experiment_id: str = EXPERIMENT_ID

    @property
    def cache(self) -> Path:
        return self.workspace / "pdf-cache"

    @property
    def experiment(self) -> Path:
        return self.workspace / "experiments" / self.experiment_id

    @property
    def dataset(self) -> Path:
        return self.experiment / "litqa2.parquet"

    @property
    def evaluator_rows(self) -> Path:
        return self.experiment / "evaluator-rows.jsonl"

    @property
    def manifest(self) -> Path:
        return self.experiment / "manifest.json"

    @property
    def acquisition(self) -> Path:
        return self.experiment / "acquisition.jsonl"

    @property
    def pages(self) -> Path:
        return self.experiment / "pages.jsonl"

    @property
    def chunks(self) -> Path:
        return self.experiment / "chunks.jsonl"

    @property
    def alignments(self) -> Path:
        return self.experiment / "alignments.jsonl"

    @property
    def alignment_overrides(self) -> Path:
        return self.experiment / "alignment-overrides.json"

    @property
    def corpus(self) -> Path:
        return self.experiment / "corpus.json"

    @property
    def vectors(self) -> Path:
        return self.experiment / "indexes" / "minilm"

    @property
    def retrieval(self) -> Path:
        return self.experiment / "retrieval.jsonl"

    @property
    def pilot(self) -> Path:
        return self.experiment / "pilot.json"

    @property
    def runs(self) -> Path:
        return self.experiment / "runs"

    @property
    def shared_runs(self) -> Path:
        return self.experiment / "runs-shared"

    @property
    def agent_runs(self) -> Path:
        return self.experiment / "agent-runs.jsonl"

    @property
    def shared_agent_runs(self) -> Path:
        return self.experiment / "agent-runs-shared.jsonl"

    @property
    def agent_state(self) -> Path:
        return self.experiment / "agent-state"

    @property
    def shared_agent_state(self) -> Path:
        return self.experiment / "agent-state-shared"

    @property
    def ledger(self) -> Path:
        return self.experiment / "cost-ledger.json"

    @property
    def results_json(self) -> Path:
        return self.experiment / "results.json"

    @property
    def results_csv(self) -> Path:
        return self.experiment / "results.csv"

    @property
    def report_html(self) -> Path:
        return self.experiment / "report.html"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def verify_frozen_inputs(paths: ExperimentPaths) -> dict:
    """Reject accidental evaluation against data other than the frozen corpus."""
    if not paths.corpus.exists():
        raise RuntimeError("Run `pqa-litqa index` before evaluation.")
    corpus = json.loads(paths.corpus.read_text("utf-8"))
    checks = ((paths.dataset, corpus.get("dataset_sha256")), (paths.chunks, corpus.get("chunk_sha256")))
    for path, expected in checks:
        if not path.exists() or not expected or sha256_file(path) != expected:
            raise RuntimeError(
                f"Frozen input changed: {path}. Start a new experiment ID and prepare/index again."
            )
    return corpus


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, values: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), "utf-8"
    )
    temporary.replace(path)


def consolidate_agent_runs(paths: ExperimentPaths, scope: str = "paired") -> list[dict]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown agent scope {scope!r}.")
    runs = paths.runs if scope == "paired" else paths.shared_runs
    output = paths.agent_runs if scope == "paired" else paths.shared_agent_runs
    rows = [json.loads(path.read_text("utf-8")) for path in sorted(runs.glob("*.json"))]
    for row in rows:
        row.setdefault("scope", scope)
    write_jsonl(output, rows)
    return rows


def normalize_doi(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    return value.strip().rstrip("/.").casefold()


def doi_filename(doi: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", normalize_doi(doi)) + ".pdf"


def selected_rows(rows: list[dict], only: list[str] | None) -> list[dict]:
    if not only:
        return rows
    wanted = set(only)
    found = [row for row in rows if row["id"] in wanted]
    missing = wanted - {row["id"] for row in found}
    if missing:
        raise ValueError(f"Unknown LitQA2 question IDs: {sorted(missing)}")
    return found


def _download_dataset(target: Path) -> None:
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "PaperQA-LitQA/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(MAX_DATASET_BYTES + 1)
    if len(data) > MAX_DATASET_BYTES or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError("Pinned LitQA2 download is not a bounded Parquet file.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(target)


def load_dataset(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('Install benchmark dependencies with `pip install -e ".[benchmarks]"`.') from exc
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["sources"] = [str(value) for value in row.get("sources") or []]
        row["distractors"] = [str(value) for value in row.get("distractors") or []]
    return rows


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    if not paths.dataset.exists():
        _download_dataset(paths.dataset)
    rows = load_dataset(paths.dataset)
    if len(rows) != 199:
        raise ValueError(f"Pinned LitQA2 dataset has {len(rows)} rows; expected 199.")
    write_jsonl(paths.evaluator_rows, rows)
    source_urls = sorted({source for row in rows for source in row["sources"]})
    manifest = {
        "version": 1,
        "experiment_id": paths.experiment_id,
        "created_at": utc_now(),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
            "file": DATASET_FILE,
            "sha256": sha256_file(paths.dataset),
            "questions": len(rows),
            "source_urls": len(source_urls),
        },
        "configuration": {
            "model": MODEL,
            "reasoning_effort": "medium",
            "retrievers": list(RETRIEVERS),
            "scopes": list(SCOPES),
            "chunk_tokens": CHUNK_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "retrieval_k": RETRIEVAL_K,
            "answer_evidence_k": ANSWER_EVIDENCE_K,
            "pilot_size": PILOT_SIZE,
            "pilot_seed": PILOT_SEED,
            "web_search": False,
            "cost_ceiling_usd": float(args.cost_ceiling),
            "price_verified_at": PRICE_VERIFIED_AT,
            "answer_instruction": CHOICE_INSTRUCTION,
        },
        "dependencies": {
            name: _package_version(name)
            for name in (
                "paper-qa",
                "pypdf",
                "pyarrow",
                "sentence-transformers",
                "chromadb",
            )
        },
        "pricing": {
            "rates": {
                name: price.__dict__ for name, price in OFFICIAL_OPENAI_PRICES.items()
            },
            "sources": PRICE_SOURCES,
        },
        "code": code_fingerprint(Path.cwd()),
    }
    write_json(paths.manifest, manifest)
    return {"questions": len(rows), "source_urls": len(source_urls), "manifest": str(paths.manifest)}


class DownloadClient:
    """Bounded public downloader with connection-time DNS validation and host pacing."""

    def __init__(self, concurrency: int = 2, host_delay: float = 0.4):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.host_delay = host_delay
        self.host_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_request: dict[str, float] = {}

    async def fetch(
        self,
        url: str,
        limit: int = MAX_PDF_BYTES,
        headers: dict[str, str] | None = None,
        timeout_seconds: int = 45,
    ) -> tuple[bytes, str, str]:
        connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False)
        async with self.semaphore, aiohttp.ClientSession(
            connector=connector,
            trust_env=False,
            cookie_jar=aiohttp.DummyCookieJar(),
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            headers={"User-Agent": "PaperQA-LitQA/1.0"},
        ) as client:
            last_error = None
            for attempt in range(2):
                try:
                    current = url
                    for _ in range(5):
                        public_url(current)
                        host = urlparse(current).hostname or ""
                        async with self.host_locks[host]:
                            delay = self.host_delay - (
                                time.monotonic() - self.last_request.get(host, 0)
                            )
                            if delay > 0:
                                await asyncio.sleep(delay)
                            self.last_request[host] = time.monotonic()
                            async with client.get(
                                current, allow_redirects=False, headers=headers
                            ) as response:
                                if response.status in {301, 302, 303, 307, 308}:
                                    current = urljoin(
                                        current, response.headers.get("Location", "")
                                    )
                                    continue
                                if response.status in {429, 500, 502, 503, 504}:
                                    response.raise_for_status()
                                response.raise_for_status()
                                data = bytearray()
                                async for part in response.content.iter_chunked(65536):
                                    data.extend(part)
                                    if len(data) > limit:
                                        raise ValueError(f"Response exceeds {limit} bytes.")
                                return (
                                    bytes(data),
                                    response.headers.get("Content-Type", ""),
                                    str(response.url),
                                )
                    raise ValueError("Too many redirects.")
                except aiohttp.ClientResponseError as exc:
                    if exc.status not in {429, 500, 502, 503, 504}:
                        raise
                    last_error = exc
                    if attempt < 1:
                        await asyncio.sleep(0.5 * (2**attempt))
                except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError) as exc:
                    last_error = exc
                    if attempt < 1:
                        await asyncio.sleep(0.5 * (2**attempt))
            assert last_error is not None
            raise last_error


def _looks_like_supplement(url: str | None) -> bool:
    return bool(
        url
        and re.search(
            r"(?:moesm|supplement|suppinfo|/esm/|[_-]esm(?:[_.-]|$)|mmc\d|mediaobjects)",
            url,
            re.I,
        )
    )


def _quarantine_cached_supplement(record: dict, cache: Path) -> None:
    cached = Path(record["path"])
    if not cached.is_file() or cached.parent.resolve() != cache.resolve():
        return
    rejected = cached.with_name(
        f"{cached.stem}.rejected-supplement-{record.get('sha256', '')[:8]}.pdf"
    )
    if not rejected.exists():
        cached.replace(rejected)


def _pdf_candidate_links(page: bytes, base: str) -> list[str]:
    text = page.decode("utf-8", errors="replace")
    candidates = []
    metadata_patterns = (
        r'<meta[^>]+(?:name|property)=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']citation_pdf_url["\']',
    )
    for pattern in metadata_patterns:
        candidates.extend(urljoin(base, value.replace("&amp;", "&")) for value in re.findall(pattern, text, re.I))
    hrefs = [
        urljoin(base, value.replace("&amp;", "&"))
        for value in re.findall(r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']', text, re.I)
    ]
    candidates.extend(url for url in hrefs if not _looks_like_supplement(url))
    return [url for url in dict.fromkeys(candidates) if not _looks_like_supplement(url)][:4]


async def _json_candidate(client: DownloadClient, url: str) -> dict:
    data, _, _ = await client.fetch(url, 2_000_000, timeout_seconds=15)
    return json.loads(data)


async def resolve_candidates(client: DownloadClient, doi: str, email: str | None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    # Stable publisher routes are useful for repositories whose landing pages
    # require JavaScript. Every response still passes the same PDF validation.
    prefix, _, suffix = doi.partition("/")
    if prefix == "10.1101":
        candidates.extend([
            (f"https://www.biorxiv.org/content/{doi}v1.full.pdf", "biorxiv-v1"),
            (f"https://www.biorxiv.org/content/{doi}.full.pdf", "biorxiv"),
        ])
    elif prefix == "10.1038":
        candidates.append((f"https://www.nature.com/articles/{suffix}.pdf", "nature"))
    elif prefix == "10.1073":
        candidates.append((f"https://www.pnas.org/doi/pdf/{doi}", "pnas"))
    elif prefix == "10.7554" and suffix.casefold().startswith("elife."):
        candidates.append((f"https://elifesciences.org/articles/{suffix.split('.', 1)[1]}.pdf", "elife"))
    queries = {
        "europe-pmc": (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
            + f"query=DOI:{quote(doi)}&format=json&pageSize=3"
        ),
        "openalex": f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='')}",
        "crossref": f"https://api.crossref.org/works/{quote(doi, safe='')}",
        "semantic-scholar": (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"DOI:{quote(doi, safe='')}?fields=openAccessPdf"
        ),
    }
    if email:
        queries["unpaywall"] = (
            f"https://api.unpaywall.org/v2/{quote(doi, safe='')}?email={quote(email)}"
        )
    responses = await asyncio.gather(
        *[_json_candidate(client, url) for url in queries.values()],
        return_exceptions=True,
    )
    payloads = {
        name: response
        for name, response in zip(queries, responses, strict=True)
        if isinstance(response, dict)
    }

    payload = payloads.get("europe-pmc", {})
    if payload:
        for item in payload.get("resultList", {}).get("result", []):
            if pmcid := item.get("pmcid"):
                # PMC's 2026 Cloud Service exposes each permissible PDF as an
                # anonymous S3 object under a PMCID/version prefix. Most records
                # have one version; trying three bounded candidates avoids a
                # multi-million-row inventory download.
                for version in range(1, 4):
                    article_version = f"{pmcid}.{version}"
                    candidates.append((
                        "https://pmc-oa-opendata.s3.amazonaws.com/"
                        f"{article_version}/{article_version}.pdf",
                        f"pmc-cloud-v{version}",
                    ))
                candidates.append((f"https://europepmc.org/articles/{pmcid}?pdf=render", "europe-pmc"))
    payload = payloads.get("openalex", {})
    if payload:
        locations = [payload.get("best_oa_location"), payload.get("primary_location")]
        locations.extend(payload.get("locations") or [])
        for location in locations:
            if isinstance(location, dict) and location.get("pdf_url"):
                candidates.append((location["pdf_url"], "openalex"))
    payload = payloads.get("crossref", {})
    if payload:
        message = payload.get("message", {})
        for link in message.get("link") or []:
            if isinstance(link, dict) and link.get("URL") and (
                link.get("content-type") == "application/pdf"
                or str(link.get("URL")).casefold().endswith(".pdf")
            ):
                candidates.append((link["URL"], "crossref-text-mining"))
    payload = payloads.get("semantic-scholar", {})
    if payload:
        location = payload.get("openAccessPdf") or {}
        if location.get("url"):
            candidates.append((location["url"], "semantic-scholar"))
    payload = payloads.get("unpaywall", {})
    if payload:
        locations = [payload.get("best_oa_location"), *(payload.get("oa_locations") or [])]
        for location in locations:
            if isinstance(location, dict) and location.get("url_for_pdf"):
                candidates.append((location["url_for_pdf"], "unpaywall"))
    candidates.append((f"https://doi.org/{doi}", "doi"))
    return list(dict.fromkeys(candidates))


def validate_pdf(data: bytes) -> None:
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-65536:]:
        raise ValueError("Response is not a complete PDF.")


async def acquire_one(
    client: DownloadClient,
    doi: str,
    paths: ExperimentPaths,
    local_files: dict[str, str],
    email: str | None,
    source_urls: list[str] | None = None,
) -> dict:
    started = time.perf_counter()
    target = paths.cache / doi_filename(doi)
    local = local_files.get(doi)
    search_roots = [paths.workspace / "sample-001"]
    if local:
        search_roots.insert(0, Path(local))
    reusable = None
    for candidate in search_roots:
        if candidate.is_file():
            reusable = candidate
            break
        possible = candidate / doi_filename(doi)
        if possible.is_file():
            reusable = possible
            break
    if target.exists():
        reusable = target
    if reusable:
        try:
            data = await asyncio.to_thread(reusable.read_bytes)
            validate_pdf(data)
            if reusable != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return {
                "doi": doi,
                "source_urls": source_urls or [],
                "resolver_version": ACQUISITION_RESOLVER_VERSION,
                "status": "downloaded",
                "method": "local-cache",
                "resolved_url": None,
                "path": str(target),
                "sha256": digest(data),
                "bytes": len(data),
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception:
            pass
    errors = []
    for url, method in await resolve_candidates(client, doi, email):
        try:
            data, kind, final_url = await client.fetch(url)
            if not data.startswith(b"%PDF-") and "html" in kind.casefold():
                links = _pdf_candidate_links(data, final_url)
                if not links:
                    raise ValueError("Landing page exposed no PDF link.")
                for link in links:
                    try:
                        pdf, _, pdf_url = await client.fetch(link)
                        validate_pdf(pdf)
                        data, final_url = pdf, pdf_url
                        method += ":landing-page"
                        break
                    except Exception as exc:
                        errors.append(f"{type(exc).__name__}: {str(exc)[:180]}")
                else:
                    raise ValueError("Landing-page PDF links were unreadable.")
            validate_pdf(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".download")
            temporary.write_bytes(data)
            temporary.replace(target)
            return {
                "doi": doi,
                "source_urls": source_urls or [],
                "resolver_version": ACQUISITION_RESOLVER_VERSION,
                "status": "downloaded",
                "method": method,
                "resolved_url": final_url,
                "path": str(target),
                "sha256": digest(data),
                "bytes": len(data),
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {str(exc)[:180]}")
    return {
        "doi": doi,
        "source_urls": source_urls or [],
        "resolver_version": ACQUISITION_RESOLVER_VERSION,
        "status": "missing",
        "method": None,
        "resolved_url": None,
        "path": None,
        "sha256": None,
        "bytes": 0,
        "seconds": round(time.perf_counter() - started, 3),
        "error": "; ".join(errors[-8:])[:2000],
    }


async def acquire_bounded(
    client: DownloadClient,
    doi: str,
    paths: ExperimentPaths,
    local_files: dict[str, str],
    email: str | None,
    source_urls: list[str],
) -> dict:
    try:
        return await asyncio.wait_for(
            acquire_one(client, doi, paths, local_files, email, source_urls),
            timeout=ACQUISITION_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return {
            "doi": doi,
            "source_urls": source_urls,
            "resolver_version": ACQUISITION_RESOLVER_VERSION,
            "status": "missing",
            "method": None,
            "resolved_url": None,
            "path": None,
            "sha256": None,
            "bytes": 0,
            "seconds": ACQUISITION_TIMEOUT_SECONDS,
            "error": f"Acquisition exceeded {ACQUISITION_TIMEOUT_SECONDS} seconds.",
        }


async def download_async(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    load_env_file(args.env_file)
    rows = selected_rows(read_jsonl(paths.evaluator_rows), args.only)
    existing = {item["doi"]: item for item in read_jsonl(paths.acquisition)}
    dois = sorted({normalize_doi(source) for row in rows for source in row["sources"]})
    local_files = {}
    if args.local_map:
        local_files = {
            normalize_doi(key): str(value)
            for key, value in json.loads(args.local_map.read_text("utf-8")).items()
        }
    client = DownloadClient()
    email = args.unpaywall_email or os.getenv("UNPAYWALL_EMAIL")
    source_urls = defaultdict(list)
    for row in rows:
        for source in row["sources"]:
            source_urls[normalize_doi(source)].append(source)
    for doi, urls in source_urls.items():
        if doi in existing:
            existing[doi]["source_urls"] = sorted(set(urls))
    for record in existing.values():
        if record.get("status") != "downloaded" or not _looks_like_supplement(
            record.get("resolved_url")
        ):
            continue
        await asyncio.to_thread(_quarantine_cached_supplement, record, paths.cache)
        record.update({
            "status": "rejected_supplement",
            "path": None,
            "error": "Landing-page candidate was a supplementary file, not the article PDF.",
        })
    pending = [
        doi
        for doi in dois
        if existing.get(doi, {}).get("status") != "downloaded"
        and (
            existing.get(doi, {}).get("status") == "rejected_supplement"
            or args.retry_missing
            or existing.get(doi, {}).get("resolver_version") != ACQUISITION_RESOLVER_VERSION
        )
    ]
    open_prefix_priority = {
        "10.1101": 0,
        "10.7554": 1,
        "10.1073": 2,
        "10.1186": 3,
        "10.3390": 4,
    }
    pending.sort(key=lambda doi: (open_prefix_priority.get(doi.split("/", 1)[0], 10), doi))
    for offset in range(0, len(pending), 2):
        results = await asyncio.gather(*[
            acquire_bounded(
                client,
                doi,
                paths,
                local_files,
                email,
                sorted(set(source_urls[doi])),
            )
            for doi in pending[offset : offset + 2]
        ])
        existing.update({item["doi"]: item for item in results})
        write_jsonl(paths.acquisition, [existing[key] for key in sorted(existing)])
        downloaded = sum(item.get("status") == "downloaded" for item in existing.values())
        print(
            f"acquisition {min(offset + 2, len(pending))}/{len(pending)} pending; "
            f"{downloaded} PDFs available",
            file=sys.stderr,
        )
    write_jsonl(paths.acquisition, [existing[key] for key in sorted(existing)])
    selected = [existing[doi] for doi in dois if doi in existing]
    return {
        "sources": len(dois),
        "downloaded": sum(item["status"] == "downloaded" for item in selected),
        "missing": sum(item["status"] != "downloaded" for item in selected),
        "acquisition": str(paths.acquisition),
    }


def download(args) -> dict:
    return asyncio.run(download_async(args))


def _document_from_dict(item: dict) -> Document:
    return Document(page_content=item["page_content"], metadata=item["metadata"], id=item.get("id"))


async def index_async(args) -> dict:
    index_started = time.perf_counter()
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    rows = selected_rows(read_jsonl(paths.evaluator_rows), args.only)
    required = {normalize_doi(source) for row in rows for source in row["sources"]}
    acquisition = {item["doi"]: item for item in read_jsonl(paths.acquisition)}
    old_pages = read_jsonl(paths.pages)
    pages_by_doi: dict[str, list[dict]] = defaultdict(list)
    for item in old_pages:
        pages_by_doi[item["metadata"]["doi"]].append(item)
    prior_corpus = json.loads(paths.corpus.read_text("utf-8")) if paths.corpus.exists() else {}
    parse_status: dict[str, dict] = dict(prior_corpus.get("parse_status", {}))
    for doi in sorted(required):
        record = acquisition.get(doi)
        if not record or record.get("status") != "downloaded":
            parse_status[doi] = {"status": "missing_pdf", "seconds": 0}
            continue
        if pages_by_doi.get(doi) and all(
            page["metadata"].get("file_sha256") == record["sha256"] for page in pages_by_doi[doi]
        ):
            previous = parse_status.get(doi, {})
            parse_status[doi] = {
                **previous,
                "status": "parsed",
                "seconds": float(previous.get("seconds", 0)),
                "reused": True,
            }
            continue
        started = time.perf_counter()
        try:
            parsed = await PaperLoader(Path(record["path"])).aload()
            if not any(page.page_content.strip() for page in parsed):
                raise ValueError("PDF parser returned no text.")
            pages_by_doi[doi] = []
            for page in parsed:
                page.metadata.update({
                    "doi": doi,
                    "title": doi,
                    "citation": doi,
                    "file_sha256": record["sha256"],
                    "document_id": record["sha256"],
                })
                pages_by_doi[doi].append(page.model_dump())
            parse_status[doi] = {
                "status": "parsed",
                "seconds": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            pages_by_doi.pop(doi, None)
            parse_status[doi] = {
                "status": "parse_failed",
                "seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
    page_rows = [page for doi in sorted(pages_by_doi) for page in pages_by_doi[doi]]
    write_jsonl(paths.pages, page_rows)

    embedding_started = time.perf_counter()
    embedding = await asyncio.to_thread(embeddings_for, MINILM)
    client = embedding._client
    tokenizer = client.tokenizer
    first_module = getattr(client, "_first_module", lambda: None)()
    model_config = getattr(getattr(first_module, "auto_model", None), "config", None)
    model_name_or_path = str(getattr(model_config, "_name_or_path", MINILM))
    revision_match = re.search(r"[\\/]snapshots[\\/]([^\\/]+)", model_name_or_path)
    embedding_revision = (
        getattr(model_config, "_commit_hash", None)
        or (revision_match.group(1) if revision_match else model_name_or_path)
    )
    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))
    chunks: list[Document] = []
    for page_data in page_rows:
        page = _document_from_dict(page_data)
        page_chunks = split_pages([page], count, CHUNK_TOKENS)
        normalized_page = normalize_text(page.page_content)
        for position, chunk in enumerate(page_chunks):
            metadata = chunk.metadata
            raw_start = int(metadata.get("start_index", 0))
            raw_end = raw_start + len(chunk.page_content)
            metadata["position"] = position
            metadata["normalized_start"] = len(normalize_text(page.page_content[:raw_start]))
            metadata["normalized_end"] = min(
                len(normalized_page), len(normalize_text(page.page_content[:raw_end]))
            )
            metadata["source_id"] = "src-" + digest(
                f"{metadata['document_id']}|{metadata['page']}|{raw_start}|{chunk.page_content}"
            )[:24]
            chunk.id = metadata["source_id"]
            chunks.append(chunk)
    chunk_rows = [chunk.model_dump() for chunk in chunks]
    chunk_text = "".join(
        json.dumps(value, ensure_ascii=False) + "\n" for value in chunk_rows
    )
    # Path.write_text uses platform newline translation. Hash the exact bytes
    # that write_jsonl will place on disk so Windows and POSIX freezes agree.
    chunk_bytes = chunk_text.replace("\n", os.linesep).encode("utf-8")
    candidate_chunk_sha256 = digest(chunk_bytes)
    evaluation_started = (
        paths.retrieval.exists()
        or paths.pilot.exists()
        or any(paths.runs.glob("*.json"))
        or any(paths.shared_runs.glob("*.json"))
    )
    if (
        prior_corpus
        and prior_corpus.get("chunk_sha256") != candidate_chunk_sha256
        and evaluation_started
    ):
        raise RuntimeError(
            "Corpus membership changed after evaluation started. Use a new --experiment-id."
        )
    write_jsonl(paths.chunks, chunk_rows)
    paths.vectors.mkdir(parents=True, exist_ok=True)
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma

    vector = Chroma(
        collection_name="litqa",
        embedding_function=embedding,
        persist_directory=str(paths.vectors),
        client_settings=ChromaSettings(anonymized_telemetry=False),
    )
    existing_ids = set(vector.get(include=[]).get("ids", []))
    missing = [chunk for chunk in chunks if chunk.id not in existing_ids]
    if missing:
        for offset in range(0, len(missing), 512):
            batch = missing[offset : offset + 512]
            await vector.aadd_documents(batch, ids=[str(chunk.id) for chunk in batch])
            print(
                f"indexed {min(offset + len(batch), len(missing))}/{len(missing)} chunks",
                file=sys.stderr,
            )
    stale = existing_ids - {str(chunk.id) for chunk in chunks}
    if stale:
        await asyncio.to_thread(vector.delete, list(stale))
    embedding_seconds = time.perf_counter() - embedding_started
    alignments = align_rows(rows, page_rows, chunks)
    override_payload = None
    if args.alignment_overrides:
        override_payload = json.loads(args.alignment_overrides.read_text("utf-8"))
        write_json(paths.alignment_overrides, override_payload)
    elif paths.alignment_overrides.exists():
        override_payload = json.loads(paths.alignment_overrides.read_text("utf-8"))
    alignments = apply_alignment_overrides(alignments, rows, chunks, override_payload)
    prior_alignments = {
        item["question_id"]: item for item in read_jsonl(paths.alignments)
    }
    prior_alignments.update({item["question_id"]: item for item in alignments})
    write_jsonl(paths.alignments, [prior_alignments[key] for key in sorted(prior_alignments)])
    parser_name = f"paperqa.research.parsing.text-only-v{PARSER_VERSION}"
    corpus = {
        "frozen_at": utc_now(),
        "documents": sum(item["status"] == "parsed" for item in parse_status.values()),
        "pages": len(page_rows),
        "chunks": len(chunks),
        "chunk_sha256": candidate_chunk_sha256,
        "dataset_sha256": sha256_file(paths.dataset),
        "acquisition_sha256": sha256_file(paths.acquisition),
        "alignment_override_sha256": sha256_file(paths.alignment_overrides)
        if paths.alignment_overrides.exists()
        else None,
        "parser": parser_name,
        "embedding": MINILM,
        "embedding_revision": embedding_revision,
        "model_max_sequence_length": client.max_seq_length,
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "index_seconds": round(time.perf_counter() - index_started, 3),
        "embedding_seconds": round(embedding_seconds, 3),
        "parse_status": parse_status,
    }
    write_json(paths.corpus, corpus)
    return corpus


def index(args) -> dict:
    return asyncio.run(index_async(args))


_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).translate(_LIGATURES)
    value = re.sub(r"(?<=\w)[\-‐]\s+(?=\w)", "", value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _word_count(value: str) -> int:
    return len(re.findall(r"\w+", value, re.UNICODE))


def align_rows(rows: list[dict], pages: list[dict], chunks: list[Document]) -> list[dict]:
    pages_by_doi: dict[str, list[dict]] = defaultdict(list)
    chunks_by_page: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for page in pages:
        pages_by_doi[page["metadata"]["doi"]].append(page)
    for chunk in chunks:
        chunks_by_page[(chunk.metadata["doi"], str(chunk.metadata["page"]))].append(chunk)
    output = []
    for row in rows:
        gold = normalize_text(row.get("key-passage") or "")
        if not gold:
            output.append({"question_id": row["id"], "status": "missing_annotation", "spans": [], "relevant_ids": []})
            continue
        spans, relevant = [], set()
        span_coverages = []
        approximate = []
        for source in row["sources"]:
            doi = normalize_doi(source)
            page_ranges = []
            combined_parts = []
            separator_ranges = []
            cursor = 0
            for page in pages_by_doi.get(doi, []):
                normalized_page = normalize_text(page["page_content"])
                if combined_parts:
                    combined_parts.append(" ")
                    separator_ranges.append((cursor, cursor + 1))
                    cursor += 1
                page_ranges.append((page, cursor, normalized_page))
                combined_parts.append(normalized_page)
                cursor += len(normalized_page)
            combined = "".join(combined_parts)
            doi_matched = False
            start = combined.find(gold)
            while start >= 0:
                doi_matched = True
                end = start + len(gold)
                match_intervals = [
                    (max(start, left), min(end, right))
                    for left, right in separator_ranges
                    if min(end, right) > max(start, left)
                ]
                for page, page_start, normalized_page in page_ranges:
                    local_start = max(0, start - page_start)
                    local_end = min(len(normalized_page), end - page_start)
                    if local_end <= local_start:
                        continue
                    page_number = str(page["metadata"]["page"])
                    spans.append({
                        "doi": doi,
                        "page": page_number,
                        "start": local_start,
                        "end": local_end,
                        "match_start": start,
                        "match_end": end,
                    })
                    for chunk in chunks_by_page[(doi, page_number)]:
                        chunk_start = int(chunk.metadata["normalized_start"])
                        chunk_end = int(chunk.metadata["normalized_end"])
                        left = max(local_start, chunk_start)
                        right = min(local_end, chunk_end)
                        if right <= left:
                            continue
                        overlap = _word_count(normalized_page[left:right])
                        gold_tokens = _word_count(gold)
                        chunk_tokens = _word_count(normalize_text(chunk.page_content))
                        required = gold_tokens if gold_tokens < 20 else max(
                            20, (min(gold_tokens, chunk_tokens) + 1) // 2
                        )
                        if overlap >= required:
                            relevant.add(chunk.metadata["source_id"])
                        match_intervals.append((page_start + left, page_start + right))
                merged = []
                for left, right in sorted(match_intervals):
                    if merged and left <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                    else:
                        merged.append((left, right))
                span_coverages.append(
                    sum(right - left for left, right in merged) / max(1, end - start)
                )
                start = combined.find(gold, start + 1)
            if not doi_matched:
                for page, _, _ in page_ranges:
                    page_number = str(page["metadata"]["page"])
                    for chunk in chunks_by_page[(doi, page_number)]:
                        score = SequenceMatcher(
                            None,
                            gold[:4000],
                            normalize_text(chunk.page_content)[:4000],
                            autojunk=False,
                        ).ratio()
                        approximate.append({
                            "source_id": chunk.metadata["source_id"],
                            "doi": doi,
                            "page": page_number,
                            "similarity": round(score, 4),
                        })
        output.append({
            "question_id": row["id"],
            "status": "aligned" if spans and relevant else "unaligned",
            "spans": spans,
            "relevant_ids": sorted(relevant),
            "gold_span_coverage": mean(span_coverages) if span_coverages else None,
            "approximate_candidates": sorted(
                approximate, key=lambda item: item["similarity"], reverse=True
            )[:5],
        })
    return output


def apply_alignment_overrides(
    alignments: list[dict],
    rows: list[dict],
    chunks: list[Document],
    payload: dict | None,
) -> list[dict]:
    if not payload:
        return alignments
    if payload.get("version") != 1 or not isinstance(payload.get("overrides"), dict):
        raise ValueError("Alignment overrides must contain version 1 and an overrides object.")
    row_sources = {
        row["id"]: {normalize_doi(source) for source in row["sources"]} for row in rows
    }
    chunk_by_id = {str(chunk.metadata["source_id"]): chunk for chunk in chunks}
    by_question = {item["question_id"]: item for item in alignments}
    for question_id, source_ids in payload["overrides"].items():
        if question_id not in by_question:
            raise ValueError(f"Alignment override references unknown question {question_id!r}.")
        if not isinstance(source_ids, list) or not source_ids:
            raise ValueError(f"Alignment override for {question_id!r} must list passage IDs.")
        unknown = set(source_ids) - set(chunk_by_id)
        if unknown:
            raise ValueError(f"Alignment override contains unknown passage IDs: {sorted(unknown)}")
        outside = [
            source_id
            for source_id in source_ids
            if chunk_by_id[source_id].metadata["doi"] not in row_sources[question_id]
        ]
        if outside:
            raise ValueError(f"Alignment override contains passages outside associated papers: {outside}")
        by_question[question_id]["status"] = "aligned_override"
        by_question[question_id]["relevant_ids"] = sorted(set(source_ids))
        by_question[question_id]["override_version"] = 1
    return [by_question[item["question_id"]] for item in alignments]


class PreparedLibrary:
    """Read-only view over one frozen chunk collection and retriever."""

    def __init__(
        self,
        chunks: list[Document],
        vectors: Path,
        retriever: str,
        allowed_document_ids: set[str] | None = None,
    ):
        if retriever not in RETRIEVERS:
            raise ValueError(f"Unknown retriever {retriever!r}.")
        self.retriever = retriever
        self.allowed_document_ids = allowed_document_ids
        self.documents = [
            doc
            for doc in chunks
            if allowed_document_ids is None or doc.metadata["document_id"] in allowed_document_ids
        ]
        self.vectors = vectors
        self.vector = None
        self.sparse = None

    def list(self) -> list[dict]:
        values = {}
        for doc in self.documents:
            values[doc.metadata["document_id"]] = doc.metadata["title"]
        return [{"id": key, "name": values[key]} for key in sorted(values)]

    async def ensure(self) -> None:
        if self.retriever in {"bm25", "hybrid"} and self.sparse is None and self.documents:
            self.sparse = BM25Retriever.from_documents(
                self.documents,
                k=min(RETRIEVAL_K * 3, len(self.documents)),
            )
            self.sparse = ZeroSafeBM25Retriever(inner=self.sparse)
        if self.retriever in {"minilm", "hybrid"} and self.vector is None:
            from chromadb.config import Settings as ChromaSettings
            from langchain_chroma import Chroma

            embedding = await asyncio.to_thread(embeddings_for, MINILM)
            self.vector = Chroma(
                collection_name="litqa",
                embedding_function=embedding,
                persist_directory=str(self.vectors),
                client_settings=ChromaSettings(anonymized_telemetry=False),
            )

    async def search(self, query: str, document_ids: list[str] | None, k: int) -> list[Document]:
        await self.ensure()
        allowed = set(document_ids or [])
        docs = [
            doc
            for doc in self.documents
            if not allowed
            or doc.metadata["document_id"] in allowed
            or doc.metadata.get("filename") in allowed
            or doc.metadata.get("doi") in allowed
        ]
        if not docs:
            return []
        k = max(1, min(int(k), RETRIEVAL_K))
        sparse = None
        if self.retriever in {"bm25", "hybrid"} and allowed:
            sparse = BM25Retriever.from_documents(
                docs,
                k=min(k * 3, len(docs)),
            )
            sparse = ZeroSafeBM25Retriever(inner=sparse)
        elif self.retriever in {"bm25", "hybrid"}:
            sparse = self.sparse
        dense = None
        if self.retriever in {"minilm", "hybrid"}:
            ids = sorted({doc.metadata["document_id"] for doc in docs})
            dense = self.vector.as_retriever(
                search_kwargs={"k": min(k * 3, len(docs)), "filter": {"document_id": {"$in": ids}}}
            )
        return await HybridRetriever(dense=dense, sparse=sparse, k=k).ainvoke(query)

    async def read(self, ids: list[str], neighbors: int = 1) -> list[Document]:
        return _expand_passages(
            self.documents, ids, neighbors, preserve_pages=True
        )

    async def source_ids(self) -> set[str]:
        return {doc.metadata["source_id"] for doc in self.documents}

    async def corpus_identity(self) -> str:
        payload = json.dumps(
            {
                "chunks": sorted(doc.metadata["source_id"] for doc in self.documents),
                "retriever": self.retriever,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _question_document_ids(row: dict, acquisition: dict[str, dict]) -> set[str]:
    return {
        acquisition[normalize_doi(source)]["sha256"]
        for source in row["sources"]
        if acquisition.get(normalize_doi(source), {}).get("status") == "downloaded"
    }


def retrieval_metrics(ranked: list[str], relevant: set[str]) -> dict:
    if not relevant:
        raise ValueError("Retrieval metrics require at least one relevant passage.")
    output = {}
    for cutoff in (1, 3, 5, 6, 10, 20):
        hits = relevant & set(ranked[:cutoff])
        output[f"recall@{cutoff}"] = len(hits) / len(relevant)
        output[f"hit@{cutoff}"] = bool(hits)
    first = next((rank for rank, source_id in enumerate(ranked[:20], 1) if source_id in relevant), None)
    output["mrr@20"] = 1 / first if first else 0.0
    return output


async def retrieve_async(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    verify_frozen_inputs(paths)
    rows = selected_rows(read_jsonl(paths.evaluator_rows), args.only)
    chunks = [_document_from_dict(item) for item in read_jsonl(paths.chunks)]
    alignments = {item["question_id"]: item for item in read_jsonl(paths.alignments)}
    acquisition = {item["doi"]: item for item in read_jsonl(paths.acquisition)}
    results = {
        (item["question_id"], item["scope"], item["retriever"]): item
        for item in read_jsonl(paths.retrieval)
    }
    libraries: dict[tuple[str, str, tuple[str, ...]], PreparedLibrary] = {}
    for row in rows:
        alignment = alignments.get(row["id"], {})
        if alignment.get("status") not in {"aligned", "aligned_override"}:
            continue
        relevant = set(alignment["relevant_ids"])
        paired_ids = _question_document_ids(row, acquisition)
        for scope in SCOPES:
            allowed = paired_ids if scope == "paired" else None
            for retriever in RETRIEVERS:
                key = row["id"], scope, retriever
                if key in results and not args.force:
                    continue
                library_key = scope, retriever, tuple(sorted(allowed or []))
                library = libraries.get(library_key)
                if library is None:
                    library = PreparedLibrary(chunks, paths.vectors, retriever, allowed)
                    await library.ensure()
                    libraries[library_key] = library
                started = time.perf_counter()
                ranked_docs = await library.search(row["question"], None, RETRIEVAL_K)
                ranked = [doc.metadata["source_id"] for doc in ranked_docs]
                results[key] = {
                    "question_id": row["id"],
                    "scope": scope,
                    "retriever": retriever,
                    "ranked_ids": ranked,
                    "relevant_ids": sorted(relevant),
                    "metrics": retrieval_metrics(ranked, relevant),
                    "seconds": round(time.perf_counter() - started, 4),
                }
                write_jsonl(paths.retrieval, [results[item] for item in sorted(results)])
    return {"evaluations": len(results), "retrieval": str(paths.retrieval)}


def retrieve(args) -> dict:
    return asyncio.run(retrieve_async(args))


def choice_prompt(row: dict) -> tuple[str, str, dict[str, str]]:
    choices = [*row["distractors"], row["ideal"], "Insufficient evidence / abstain"]
    seed = int(hashlib.sha256(f"{PILOT_SEED}:{row['id']}".encode()).hexdigest(), 16)
    random.Random(seed).shuffle(choices)
    mapping = {chr(65 + index): value for index, value in enumerate(choices)}
    gold = next(label for label, value in mapping.items() if value == row["ideal"])
    lines = "\n".join(f"{label}. {value}" for label, value in mapping.items())
    prompt = (
        f"{row['question']}\n\nChoices:\n{lines}\n\n"
        + CHOICE_INSTRUCTION
    )
    return prompt, gold, mapping


def parse_choice(answer: str, labels: set[str]) -> str | None:
    matches = re.findall(r"FINAL_CHOICE:\s*([A-Z])\b", answer, re.I)
    choice = matches[0].upper() if len(matches) == 1 else None
    return choice if choice in labels else None


def citation_metrics(
    answer: str,
    sources: list[dict],
    associated_document_ids: set[str],
    relevant_ids: set[str],
) -> dict:
    cited = set(re.findall(r"src-[a-f0-9]{24}", answer))
    known = {source["id"] for source in sources}
    associated = {
        source["id"]
        for source in sources
        if source.get("document_id") in associated_document_ids
    }
    return {
        "coverage": bool(cited),
        "valid": len(cited & known) / len(cited) if cited else 0.0,
        "associated_source": len(cited & associated) / len(cited) if cited else 0.0,
        "gold_passage_proxy": len(cited & relevant_ids) / len(cited) if cited else 0.0,
    }


def select_pilot(rows: list[dict], acquisition: dict[str, dict], corpus: dict) -> list[str]:
    parsed = {
        doi for doi, status in corpus.get("parse_status", {}).items() if status.get("status") == "parsed"
    }
    eligible = [
        row["id"]
        for row in rows
        if row["sources"] and all(normalize_doi(source) in parsed for source in row["sources"])
        and all(acquisition.get(normalize_doi(source), {}).get("sha256") for source in row["sources"])
    ]
    return sorted(
        eligible, key=lambda question_id: hashlib.sha256(f"{PILOT_SEED}:{question_id}".encode()).hexdigest()
    )[:PILOT_SIZE]


async def run_async(args) -> dict:
    from paperqa.research.service import ResearchService, execute
    from paperqa.research.storage import RunRecord

    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    scope = getattr(args, "scope", "paired")
    if scope not in SCOPES:
        raise ValueError(f"Unknown agent scope {scope!r}.")
    runs_path = paths.runs if scope == "paired" else paths.shared_runs
    state_path = paths.agent_state if scope == "paired" else paths.shared_agent_state
    verify_frozen_inputs(paths)
    load_env_file(args.env_file)
    rows = read_jsonl(paths.evaluator_rows)
    by_id = {row["id"]: row for row in rows}
    acquisition = {item["doi"]: item for item in read_jsonl(paths.acquisition)}
    corpus = json.loads((paths.experiment / "corpus.json").read_text("utf-8"))
    chunks = [_document_from_dict(item) for item in read_jsonl(paths.chunks)]
    alignments = {item["question_id"]: item for item in read_jsonl(paths.alignments)}
    if paths.pilot.exists():
        pilot = json.loads(paths.pilot.read_text("utf-8"))["question_ids"]
    else:
        pilot = select_pilot(rows, acquisition, corpus)
        write_json(paths.pilot, {"seed": PILOT_SEED, "question_ids": pilot})
    if args.only:
        wanted = set(args.only)
        missing = wanted - set(pilot)
        if missing:
            raise ValueError(f"Selected questions are not in the frozen pilot: {sorted(missing)}")
        pilot = [question_id for question_id in pilot if question_id in wanted]
    runs_path.mkdir(parents=True, exist_ok=True)
    app = ResearchService(state_path)
    ledger = CostLedger(paths.ledger, args.cost_ceiling)
    attempted = completed = 0
    try:
        for question_id in pilot:
            row = by_id[question_id]
            associated = _question_document_ids(row, acquisition)
            allowed = associated if scope == "paired" else None
            for retriever in RETRIEVERS:
                target = runs_path / f"{question_id}-{retriever}.json"
                prior_artifact = json.loads(target.read_text("utf-8")) if target.exists() else {}
                if prior_artifact.get("status") in {"complete", "unsure"}:
                    continue
                if (
                    prior_artifact.get("status") in {"failed", "stopped"}
                    and not args.retry_failed
                ):
                    continue
                restart_paused = prior_artifact.get("status") == "paused" and getattr(
                    args, "restart_paused", False
                )
                if restart_paused:
                    archived_attempt = int(prior_artifact.get("attempt", 1))
                    write_json(
                        target.parent
                        / "history"
                        / f"{target.stem}.attempt-{archived_attempt}.paused.json",
                        prior_artifact,
                    )
                resuming_pause = prior_artifact.get("status") == "paused" and not restart_paused
                attempt = int(prior_artifact.get("attempt", 1 if resuming_pause else 0))
                if not resuming_pause:
                    attempt += 1
                prompt, gold, mapping = choice_prompt(row)
                run_id = prior_artifact.get("run_id") if resuming_pause else None
                run_id = run_id or hashlib.sha256(
                    f"{paths.experiment_id}:{scope}:{question_id}:{retriever}:attempt-{attempt}".encode()
                ).hexdigest()[:12]
                job = app.jobs.get(run_id)
                if job is None:
                    conversation_id = app.begin_turn(prompt)
                    job = RunRecord(
                        id=run_id,
                        conversation_id=conversation_id,
                        question=prompt,
                        model=MODEL,
                        embedding=retriever,
                        mode="research",
                        route="research",
                    )
                    app.jobs[job.id] = job
                if job.status == "paused" and not args.resume_provider:
                    continue
                trace: list[dict] = []

                def observe(name, inputs, outputs, trace=trace):
                    trace.append({"tool": name, "inputs": inputs, "outputs": outputs})

                cost_stage = f"litqa:{scope}:{question_id}:{retriever}:attempt-{attempt}"
                callback = BenchmarkCostCallback(
                    ledger,
                    MODEL,
                    maximum_output_tokens=4096,
                    stage=cost_stage,
                )
                library = PreparedLibrary(chunks, paths.vectors, retriever, allowed)
                await library.ensure()
                payload = {
                    "web_search": False,
                    "answer_max_sources": 5,
                    "evidence_k": ANSWER_EVIDENCE_K,
                    "reasoning_effort": "medium",
                    "local_conversation_fast_path": False,
                    "completion_contract": {
                        "marker": "FINAL_CHOICE",
                        "labels": sorted(mapping),
                        "abstention_label": next(
                            label
                            for label, value in mapping.items()
                            if value == "Insufficient evidence / abstain"
                        ),
                    },
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
                            callbacks=[callback],
                            library=library,
                            retrieval_observer=observe,
                        ),
                        timeout=args.timeout,
                    )
                except TimeoutError:
                    timed_out = True
                    ledger.retain_open_reservations(
                        "Benchmark timeout left provider billing uncertain.",
                        stage_prefix=cost_stage,
                    )
                    job.status, job.pause_reason = "stopped", "benchmark_timeout"
                    job.error = f"Benchmark timeout reached after {args.timeout} seconds."
                    app.save_job(job)
                    app.finish_turn(job)
                answer = str((job.result or {}).get("answer", ""))
                chosen = parse_choice(answer, set(mapping))
                answer_status = dict(
                    (job.result or {}).get("agent", {}).get("answer_status", {})
                )
                sources = list((job.result or {}).get("sources", []))
                relevant = set(alignments.get(question_id, {}).get("relevant_ids", []))
                inspected = {
                    output.get("id")
                    for item in trace
                    for output in item.get("outputs", [])
                    if isinstance(output, dict)
                }
                cost_entries = [
                    item
                    for item in ledger.snapshot().get("entries", [])
                    if item.get("stage") == cost_stage
                ]
                artifact = {
                    "question_id": question_id,
                    "retriever": retriever,
                    "scope": scope,
                    "run_id": run_id,
                    "attempt": attempt,
                    "terminal": job.status in {"complete", "unsure", "failed", "stopped"},
                    "status": job.status,
                    "pause_reason": job.pause_reason,
                    "timed_out": timed_out,
                    "answer": answer,
                    "choice": chosen,
                    "gold_choice": gold,
                    "correct": chosen == gold,
                    "abstained": chosen is not None
                    and mapping[chosen] == "Insufficient evidence / abstain",
                    "malformed": chosen is None,
                    "answer_available": bool(answer_status.get("available", answer.strip())),
                    "format_valid": bool(answer_status.get("format_valid", chosen is not None)),
                    "fallback_abstention": bool(
                        answer_status.get("fallback_abstention", False)
                    ),
                    "execution_status": (job.result or {}).get("execution_status"),
                    "choice_mapping": mapping,
                    "sources": sources,
                    "retrieval_trace": trace,
                    "inspected_gold_passage": bool(inspected & relevant),
                    "citation": citation_metrics(answer, sources, associated, relevant),
                    "usage": dict(job.usage),
                    "cost": {
                        "usd": sum(
                            float(item.get("actual_usd", item.get("reserved_usd", 0)))
                            for item in cost_entries
                        ),
                        "calls": len(cost_entries),
                        "uncertain_calls": sum(
                            item.get("status") in {"uncertain", "reserved"}
                            for item in cost_entries
                        ),
                        "entries": cost_entries,
                    },
                    "verification": (job.result or {}).get("agent", {}).get("verification", {}),
                    "seconds": round(time.perf_counter() - started, 3),
                    "error": job.error,
                }
                write_json(target, artifact)
                consolidate_agent_runs(paths, scope)
                if job.status == "complete":
                    completed += 1
                if job.pause_reason in {
                    "provider_quota",
                    "provider_connection",
                    "provider_unavailable",
                    "spending_ceiling",
                }:
                    return {
                        "scope": scope,
                        "attempted": attempted,
                        "completed": completed,
                        "paused": True,
                    }
    finally:
        app.close()
    return {"scope": scope, "attempted": attempted, "completed": completed, "paused": False}


def run(args) -> dict:
    return asyncio.run(run_async(args))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))]


def _average_metrics(items: list[dict]) -> dict:
    if not items:
        return {}
    keys = sorted(items[0]["metrics"])
    return {key: mean(float(item["metrics"][key]) for item in items) for key in keys}


def _summarize_answers(run_rows: list[dict]) -> dict:
    answer_summary = {}
    for retriever in RETRIEVERS:
        selected = [item for item in run_rows if item["retriever"] == retriever]
        answered = [item for item in selected if not item["malformed"] and not item["abstained"]]
        verification_statuses: dict[str, int] = defaultdict(int)
        for item in selected:
            verification_statuses[str(item.get("verification", {}).get("status", "missing"))] += 1
        answer_summary[retriever] = {
            "runs": len(selected),
            "completion_rate": sum(item["status"] == "complete" for item in selected)
            / len(selected)
            if selected
            else 0,
            "accuracy": sum(item["correct"] for item in selected) / len(selected)
            if selected
            else 0,
            "precision": sum(item["correct"] for item in answered) / len(answered)
            if answered
            else 0,
            "answered_accuracy": sum(item["correct"] for item in answered) / len(answered)
            if answered
            else 0,
            "answer_coverage": len(answered) / len(selected) if selected else 0,
            "abstention_rate": sum(item["abstained"] for item in selected) / len(selected)
            if selected
            else 0,
            "citation_coverage": mean(item["citation"]["coverage"] for item in selected)
            if selected
            else 0,
            "citation_validity": mean(item["citation"]["valid"] for item in selected)
            if selected
            else 0,
            "citation_associated_source": mean(
                item["citation"]["associated_source"] for item in selected
            )
            if selected
            else 0,
            "citation_gold_passage_proxy": mean(
                item["citation"]["gold_passage_proxy"] for item in selected
            )
            if selected
            else 0,
            "inspected_gold_passage_rate": mean(
                item["inspected_gold_passage"] for item in selected
            )
            if selected
            else 0,
            "latency_median": median([item["seconds"] for item in selected])
            if selected
            else None,
            "latency_p95": _percentile([item["seconds"] for item in selected], 0.95),
            "verification_statuses": dict(sorted(verification_statuses.items())),
            "cost_usd": sum(float(item.get("cost", {}).get("usd", 0)) for item in selected),
            "model_calls": sum(int(item.get("cost", {}).get("calls", 0)) for item in selected),
        }
    return answer_summary


def report(args) -> dict:
    paths = ExperimentPaths(args.workspace.resolve(), args.experiment_id)
    corpus_manifest = verify_frozen_inputs(paths)
    experiment_manifest = (
        json.loads(paths.manifest.read_text("utf-8")) if paths.manifest.exists() else {}
    )
    acquisition = read_jsonl(paths.acquisition)
    alignments = read_jsonl(paths.alignments)
    retrieval_rows = read_jsonl(paths.retrieval)
    run_rows_by_scope = {
        "paired": consolidate_agent_runs(paths, "paired"),
        "shared": consolidate_agent_runs(paths, "shared"),
    }
    retrieval_summary = {}
    for scope in SCOPES:
        question_sets = [
            {
                item["question_id"]
                for item in retrieval_rows
                if item["scope"] == scope and item["retriever"] == retriever
            }
            for retriever in RETRIEVERS
        ]
        common_questions = set.intersection(*question_sets) if question_sets else set()
        for retriever in RETRIEVERS:
            available = [
                item for item in retrieval_rows if item["scope"] == scope and item["retriever"] == retriever
            ]
            selected = [item for item in available if item["question_id"] in common_questions]
            retrieval_summary[f"{scope}:{retriever}"] = {
                "questions": len(selected),
                "available_questions": len(available),
                "common_question_set": True,
                **_average_metrics(selected),
                "latency_median": median([item["seconds"] for item in selected]) if selected else None,
                "latency_p95": _percentile([item["seconds"] for item in selected], 0.95),
            }
    answer_summaries = {
        scope: _summarize_answers(rows) for scope, rows in run_rows_by_scope.items()
    }
    answer_summary = answer_summaries["paired"]
    ledger = json.loads(paths.ledger.read_text("utf-8")) if paths.ledger.exists() else {}
    acquisition_seconds = [float(item.get("seconds", 0)) for item in acquisition]
    parsing_seconds = [
        float(item.get("seconds", 0))
        for item in corpus_manifest.get("parse_status", {}).values()
    ]
    retrieval_eligible = sum(
        item["status"] in {"aligned", "aligned_override"} for item in alignments
    )
    retrieval_complete = all(
        summary["available_questions"] == retrieval_eligible
        for summary in retrieval_summary.values()
    )
    paired_complete = len(run_rows_by_scope["paired"]) == PILOT_SIZE * len(RETRIEVERS) and all(
        item["status"] in {"complete", "unsure"} for item in run_rows_by_scope["paired"]
    )
    shared_started = bool(run_rows_by_scope["shared"])
    shared_complete = len(run_rows_by_scope["shared"]) == PILOT_SIZE * len(RETRIEVERS) and all(
        item["status"] in {"complete", "unsure"} for item in run_rows_by_scope["shared"]
    )
    results = {
        "experiment_id": paths.experiment_id,
        "generated_at": utc_now(),
        "code": code_fingerprint(Path.cwd()),
        "manifest_code": experiment_manifest.get("code"),
        "complete": retrieval_complete
        and paired_complete
        and (not shared_started or shared_complete),
        "retrieval_complete": retrieval_complete,
        "paired_agent_complete": paired_complete,
        "shared_agent_complete": shared_complete,
        "corpus": {
            "source_urls": len({
                url for item in acquisition for url in item.get("source_urls", [])
            }),
            "normalized_dois": len(acquisition),
            "downloaded": sum(item["status"] == "downloaded" for item in acquisition),
            "missing": sum(item["status"] != "downloaded" for item in acquisition),
            "parsed_documents": corpus_manifest.get("documents", 0),
            "parse_failures": sum(
                item.get("status") == "parse_failed"
                for item in corpus_manifest.get("parse_status", {}).values()
            ),
            "aligned": sum(item["status"] in {"aligned", "aligned_override"} for item in alignments),
            "aligned_by_override": sum(item["status"] == "aligned_override" for item in alignments),
            "unaligned": sum(item["status"] == "unaligned" for item in alignments),
            "missing_annotations": sum(item["status"] == "missing_annotation" for item in alignments),
            "retrieval_eligible_questions": retrieval_eligible,
            "mean_gold_span_coverage": mean([
                float(item["gold_span_coverage"])
                for item in alignments
                if item.get("gold_span_coverage") is not None
            ]) if any(item.get("gold_span_coverage") is not None for item in alignments) else None,
        },
        "retrieval": retrieval_summary,
        "answers": answer_summary,
        "answers_by_scope": answer_summaries,
        "cost": {
            "limit_usd": ledger.get("limit_usd", args.cost_ceiling),
            "spent_usd": ledger.get("spent_usd", 0),
            "reserved_usd": ledger.get("reserved_usd", 0),
        },
        "latency": {
            "acquisition_median": median(acquisition_seconds) if acquisition_seconds else None,
            "acquisition_p95": _percentile(acquisition_seconds, 0.95),
            "parsing_median": median(parsing_seconds) if parsing_seconds else None,
            "parsing_p95": _percentile(parsing_seconds, 0.95),
            "index_total": corpus_manifest.get("index_seconds"),
            "embedding_index": corpus_manifest.get("embedding_seconds"),
        },
    }
    write_json(paths.results_json, results)
    csv_rows = []
    for item in retrieval_rows:
        csv_rows.append({
            "kind": "retrieval",
            "question_id": item["question_id"],
            "scope": item["scope"],
            "retriever": item["retriever"],
            "correct": "",
            "mrr@20": item["metrics"]["mrr@20"],
            "seconds": item["seconds"],
        })
    for scope, run_rows in run_rows_by_scope.items():
        for item in run_rows:
            csv_rows.append({
                "kind": "answer",
                "question_id": item["question_id"],
                "scope": scope,
                "retriever": item["retriever"],
                "correct": item["correct"],
                "mrr@20": "",
                "seconds": item["seconds"],
            })
    paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "question_id",
                "scope",
                "retriever",
                "correct",
                "mrr@20",
                "seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    sections = [
        "<h1>LitQA2 closed-corpus benchmark</h1>",
        f"<p><strong>Status:</strong> {'complete' if results['complete'] else 'incomplete'}</p>",
        "<h2>Corpus</h2><pre>" + html.escape(json.dumps(results["corpus"], indent=2)) + "</pre>",
        "<h2>Retrieval</h2><pre>" + html.escape(json.dumps(retrieval_summary, indent=2)) + "</pre>",
        "<h2>Answers and citations by scope</h2><pre>"
        + html.escape(json.dumps(answer_summaries, indent=2))
        + "</pre>",
        "<h2>Cost</h2><pre>" + html.escape(json.dumps(results["cost"], indent=2)) + "</pre>",
        "<h2>Stage latency</h2><pre>" + html.escape(json.dumps(results["latency"], indent=2)) + "</pre>",
        "<h2>Supporting artifacts</h2><ul>"
        '<li><a href="acquisition.jsonl">Acquisition records</a></li>'
        '<li><a href="alignments.jsonl">Gold-passage alignments</a></li>'
        '<li><a href="retrieval.jsonl">Standalone retrieval results</a></li>'
        '<li><a href="agent-runs.jsonl">Per-question agent results</a></li>'
        '<li><a href="agent-runs-shared.jsonl">Shared-corpus agent results</a></li>'
        '<li><a href="runs/">Individual resumable run artifacts</a></li></ul>',
        "<p>Gold-passage citation matching is an evidence proxy, not a claim-entailment score.</p>",
    ]
    paths.report_html.write_text(
        "<!doctype html><meta charset=utf-8><title>LitQA2 benchmark</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;"
        "padding:0 20px}pre{background:#f5f5f5;padding:16px;overflow:auto}</style>"
        + "".join(sections),
        "utf-8",
    )
    return results


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--only", action="append", help="Restrict work to a LitQA2 question ID.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--cost-ceiling", type=float, default=10.0)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Download and pin LitQA2 metadata.")
    prepare_parser.set_defaults(handler=prepare)
    download_parser = subparsers.add_parser("download", help="Acquire and validate source PDFs.")
    _add_selection(download_parser)
    download_parser.add_argument("--local-map", type=Path, help="JSON mapping DOI to local PDF path.")
    download_parser.add_argument("--unpaywall-email", default=os.getenv("UNPAYWALL_EMAIL"))
    download_parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Retry terminal acquisition failures with the current resolver.",
    )
    download_parser.set_defaults(handler=download)
    index_parser = subparsers.add_parser("index", help="Parse, chunk, align, and index PDFs.")
    _add_selection(index_parser)
    index_parser.add_argument(
        "--alignment-overrides",
        type=Path,
        help="Versioned JSON overrides for reviewed approximate passage matches.",
    )
    index_parser.set_defaults(handler=index)
    retrieve_parser = subparsers.add_parser("retrieve", help="Evaluate paired and shared retrieval.")
    _add_selection(retrieve_parser)
    retrieve_parser.add_argument("--force", action="store_true")
    retrieve_parser.set_defaults(handler=retrieve)
    run_parser = subparsers.add_parser(
        "run", help="Run the frozen 20-question paired or shared-corpus pilot."
    )
    _add_selection(run_parser)
    run_parser.add_argument(
        "--scope",
        choices=SCOPES,
        default="paired",
        help="Expose associated papers only (paired) or the complete frozen corpus (shared).",
    )
    run_parser.add_argument("--timeout", type=int, default=RUN_TIMEOUT_SECONDS)
    run_parser.add_argument("--resume-provider", action="store_true")
    run_parser.add_argument(
        "--restart-paused",
        action="store_true",
        help="Archive an unrecoverable paused checkpoint and start a new isolated attempt.",
    )
    run_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Start a new isolated attempt for failed or timed-out runs.",
    )
    run_parser.set_defaults(handler=run)
    report_parser = subparsers.add_parser("report", help="Create JSON, CSV, and HTML reports.")
    report_parser.set_defaults(handler=report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command != "prepare" and not ExperimentPaths(
            args.workspace.resolve(), args.experiment_id
        ).evaluator_rows.exists():
            raise RuntimeError("Run `pqa-litqa prepare` first.")
        result = args.handler(args)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; saved benchmark artifacts can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
