"""Local LangChain retrieval, independent of PaperQA's LLM/indexing pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from .config import DEFAULT_EMBEDDING
from .parsing import PARSER_VERSION, SUPPORTED_SUFFIXES, parse_document

SUFFIXES = SUPPORTED_SUFFIXES
_RESOURCE_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_RESOURCE_CACHE_LOCK = threading.RLock()
_INDEX_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
INDEX_FORMAT_VERSION = 4
MAX_RETRIEVAL_CANDIDATES = 60


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold()))


def _index_lock(path: Path) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), str(path.resolve()))
    with _RESOURCE_CACHE_LOCK:
        return _INDEX_LOCKS.setdefault(key, asyncio.Lock())


def _expand_passages(
    documents: list[Document],
    source_ids: list[str],
    neighbors: int,
    *,
    preserve_pages: bool,
    limit: int = 20,
) -> list[Document]:
    """Return requested passages first, then nearby chunks in stable distance order."""
    by_id = {doc.metadata["source_id"]: doc for doc in documents}
    requested = list(dict.fromkeys(source_ids))
    missing = [source_id for source_id in requested if source_id not in by_id]
    if missing:
        raise ValueError("Unknown passage IDs: " + ", ".join(missing))
    selected = [by_id[source_id] for source_id in requested]
    effective_limit = max(limit, len(selected))
    seen = set(requested)
    radius = max(0, min(int(neighbors), 2))
    candidates: list[tuple[int, int, int, Document]] = []
    for request_index, item in enumerate(selected):
        position = int(item.metadata["position"])
        for corpus_index, doc in enumerate(documents):
            if doc.metadata["source_id"] in seen:
                continue
            if doc.metadata["document_id"] != item.metadata["document_id"]:
                continue
            if preserve_pages and doc.metadata.get("page") != item.metadata.get("page"):
                continue
            distance = abs(int(doc.metadata["position"]) - position)
            if distance <= radius:
                candidates.append((distance, request_index, corpus_index, doc))
    for _, _, _, doc in sorted(candidates, key=lambda value: value[:3]):
        source_id = doc.metadata["source_id"]
        if source_id in seen:
            continue
        selected.append(doc)
        seen.add(source_id)
        if len(selected) >= effective_limit:
            break
    return selected[:effective_limit]


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@lru_cache(maxsize=3)
def embeddings_for(name: str):
    if name == "sparse":
        return None
    if name.startswith(("st-", "hybrid-st-")):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Local embeddings are unavailable: the PyTorch installation could not load. "
                "Check your trusted Python installation and Windows application-control logs, "
                "or explicitly select BM25 (sparse) or configured OpenAI embeddings."
            ) from exc

        model = name.removeprefix("hybrid-").removeprefix("st-")
        if "/" not in model:
            model = "sentence-transformers/" + model
        common = {
            "model_name": model,
            "encode_kwargs": {"normalize_embeddings": True},
        }
        try:
            # Avoid slow Hub availability checks on every local-server restart.
            # A first-time setup still falls back to the normal download path.
            return HuggingFaceEmbeddings(
                **common,
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": False,
                    "local_files_only": True,
                },
            )
        except OSError:
            return HuggingFaceEmbeddings(
                **common,
                model_kwargs={"device": "cpu", "trust_remote_code": False},
            )
    raise ValueError("Unsupported embedding model.")


class PaperLoader:
    """Load a local document into page-preserving LangChain documents."""

    def __init__(self, path: Path):
        self.path = path

    async def aload(self):
        return [document async for document in self.alazy_load()]

    async def alazy_load(self):
        content_hash = await asyncio.to_thread(lambda: digest(self.path.read_bytes()))
        pages = await asyncio.wait_for(
            asyncio.to_thread(parse_document, self.path, content_hash), timeout=45
        )
        for page in pages:
            yield page


class HybridRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dense: Any = None
    sparse: Any = None
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager):
        groups = [
            retriever.invoke(query)
            for retriever in (self.dense, self.sparse)
            if retriever is not None
        ]
        scores, documents = {}, {}
        for group in groups:
            for rank, doc in enumerate(group, 1):
                key = doc.metadata["source_id"]
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                documents[key] = doc
        return [
            documents[key]
            for key in sorted(scores, key=lambda key: (-scores[key], key))[: self.k]
        ]


class ZeroSafeBM25Retriever(BaseRetriever):
    """Suppress BM25's arbitrary documents when no lexical term overlaps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    inner: Any

    def _get_relevant_documents(self, query: str, *, run_manager):
        terms = _tokens(query)
        if not terms:
            return []
        return [doc for doc in self.inner.invoke(query) if terms & _tokens(doc.page_content)]


class BM25Retriever(BaseRetriever):
    """Small rank-bm25 adapter, avoiding the deprecated community integration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    documents: list[Document]
    engine: Any
    k: int = 6

    @classmethod
    def from_documents(cls, documents: list[Document], *, k: int):
        from rank_bm25 import BM25Okapi

        tokenized = [re.findall(r"\w+", doc.page_content.casefold()) for doc in documents]
        return cls(documents=documents, engine=BM25Okapi(tokenized), k=k)

    def _get_relevant_documents(self, query: str, *, run_manager):
        tokens = re.findall(r"\w+", query.casefold())
        if not tokens:
            return []
        scores = self.engine.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [self.documents[index] for index in ranked[: self.k]]


def split_pages(pages, count, size):
    """Token-bounded chunks with exact character offsets and <=32-token overlap.

    Counting uses the selected embedding tokenizer, not an approximate word size.
    Cutting characters (rather than decoding token slices) preserves Unicode.
    """
    chunks = []
    for page in pages:
        text, start = page.page_content, 0
        while start < len(text):
            low, high = start + 1, min(len(text), start + size * 12)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if count(text[start:middle]) <= size:
                    end, low = middle, middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("Embedding token limit cannot fit one character.")
            if end < len(text):
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if boundary > start + (end - start) // 2:
                    end = boundary + 1
            content = text[start:end]
            while count(content) > size:
                end -= 1
                content = text[start:end]
            if content.strip():
                chunks.append(
                    Document(
                        page_content=content, metadata={**page.metadata, "start_index": start}
                    )
                )
            if end == len(text):
                break
            overlap = end
            # A small bounded scan avoids assuming tokens map one-to-one to chars.
            while overlap > start + 1 and end - overlap < 128:
                if count(text[overlap - 1 : end]) > min(32, size // 5):
                    break
                overlap -= 1
            start = max(start + 1, overlap)
    return chunks


class Library:
    def __init__(
        self,
        papers: Path,
        directory: Path,
        embedding=DEFAULT_EMBEDDING,
        supplied_docs=None,
        meter=None,
    ):
        self.papers, self.directory = papers.resolve(), directory
        self.embedding, self.supplied_docs = embedding, supplied_docs
        self.meter = meter
        self.documents: list[Document] | None = None
        self.vector = None
        self._sparse_cache: dict[tuple[str, ...], ZeroSafeBM25Retriever] = {}
        self._ready_lock = asyncio.Lock()

    def list(self):
        if self.supplied_docs is not None:
            documents = getattr(self.supplied_docs, "documents", self.supplied_docs)
            values = {
                doc.metadata["document_id"]: doc.metadata.get("title", "Untitled")
                for doc in documents
            }
            return [{"id": key, "name": values[key]} for key in sorted(values)]
        return (
            [
                {"id": path.name, "name": path.name}
                for path in sorted(self.papers.iterdir())
                if path.suffix.lower() in SUFFIXES and path.is_file() and not path.is_symlink()
            ]
            if self.papers.exists()
            else []
        )

    async def ensure(self):
        async with self._ready_lock:
            async with _index_lock(self.directory / self.embedding):
                await self._ensure()

    async def _ensure(self):
        if self.documents is not None:
            return
        cache_key = None
        source_signature = None
        if self.supplied_docs is None:
            source_paths = [
                path
                for path in sorted(self.papers.iterdir())
                if path.is_file() and path.suffix.lower() in SUFFIXES and not path.is_symlink()
            ]
            signature_items = []
            for path in source_paths:
                content_hash = await asyncio.to_thread(
                    lambda value=path: digest(value.read_bytes())
                )
                signature_items.append((path.name, content_hash))
            source_signature = tuple(signature_items)
            cache_key = (str(self.papers), str(self.directory.resolve()), self.embedding)
            with _RESOURCE_CACHE_LOCK:
                cached = _RESOURCE_CACHE.get(cache_key)
            if cached and cached["signature"] == source_signature:
                self.documents = cached["documents"]
                self.vector = cached["vector"]
                self._sparse_cache = cached.get("sparse_cache", {})
                return
            # An empty benchmark/library has nothing to embed.  Short-circuit before
            # constructing SentenceTransformer/OpenAI clients so web-only research is
            # independent of local model availability and API credentials.
            if not source_signature:
                self.documents = []
                self.vector = None
                with _RESOURCE_CACHE_LOCK:
                    _RESOURCE_CACHE[cache_key] = {
                        "signature": source_signature,
                        "documents": [],
                        "vector": None,
                        "sparse_cache": {},
                    }
                return
        embedding = await asyncio.to_thread(embeddings_for, self.embedding)
        if self.embedding.startswith(("st-", "hybrid-st-")):
            client = embedding._client
            size = min(384, client.max_seq_length - 2)

            def count(text: str) -> int:
                # The splitter intentionally probes over-limit candidate slices
                # before binary-searching down to ``size``. Suppress the
                # tokenizer's misleading warning for those probes; final chunks
                # are still checked below and never exceed the embedding limit.
                return len(
                    client.tokenizer.encode(
                        text, add_special_tokens=False, verbose=False
                    )
                )
        else:
            import tiktoken

            size = 384
            encoder = tiktoken.get_encoding("cl100k_base")

            def count(text: str) -> int:
                return len(encoder.encode(text, disallowed_special=()))
        fingerprint = digest(
            f"v{INDEX_FORMAT_VERSION + 1}|{self.embedding}|{size}|32|char-offsets|parser-v{PARSER_VERSION}"
        )[:24]
        root = self.directory / "langchain-v2" / fingerprint
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "documents.json"
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
        old_ids = {
            item["metadata"]["source_id"]
            for entry in manifest.values()
            for item in entry["chunks"]
        }
        new_manifest: dict[str, dict] = {}
        changed: list[Document] = []
        if self.supplied_docs is not None:
            # Caller-provided LangChain documents are authoritative.
            pages = list(getattr(self.supplied_docs, "documents", self.supplied_docs))
            supplied_identity = json.dumps(
                [
                    {
                        "text": doc.page_content,
                        "document_id": doc.metadata["document_id"],
                        "title": doc.metadata["title"],
                        "citation": doc.metadata["citation"],
                        "page": doc.metadata["page"],
                    }
                    for doc in pages
                ],
                ensure_ascii=False,
                sort_keys=True,
            )
            entries = [("provided", digest(supplied_identity), pages)]
        else:
            entries = []
            signature_hashes = dict(source_signature or ())
            for item in self.list():
                path = self.papers / item["name"]
                content_hash = signature_hashes.get(path.name) or await asyncio.to_thread(
                    lambda p=path: digest(p.read_bytes())
                )
                old = manifest.get(path.name)
                if old and old["hash"] == content_hash:
                    new_manifest[path.name] = old
                    continue
                entries.append((path.name, content_hash, await PaperLoader(path).aload()))
        for name, content_hash, pages in entries:
            old = manifest.get(name)
            if old and old["hash"] == content_hash:
                new_manifest[name] = old
                continue
            chunks = await asyncio.to_thread(split_pages, pages, count, size)
            for position, chunk in enumerate(chunks):
                meta = chunk.metadata
                meta["position"] = position
                meta["source_id"] = (
                    "src-"
                    + digest(
                        f"{meta['document_id']}|{meta['page']}|{meta.get('start_index', 0)}|{chunk.page_content}"
                    )[:24]
                )
                chunk.id = meta["source_id"]
            new_manifest[name] = {
                "hash": content_hash,
                "chunks": [doc.model_dump() for doc in chunks],
            }
            changed.extend(chunks)
        documents = [
            Document(**chunk) for entry in new_manifest.values() for chunk in entry["chunks"]
        ]
        if embedding is not None:
            from chromadb.config import Settings as ChromaSettings
            from langchain_chroma import Chroma

            self.vector = Chroma(
                collection_name="research",
                embedding_function=embedding,
                persist_directory=str(root / "vectors"),
                client_settings=ChromaSettings(anonymized_telemetry=False),
            )
            new_ids = {doc.metadata["source_id"] for doc in documents}
            removed = old_ids - new_ids
            if removed:
                await asyncio.to_thread(self.vector.delete, list(removed))
            if changed:
                # Upsert is repeat-safe if a process stopped before committing the manifest.
                await self.vector.aadd_documents(changed, ids=[doc.id for doc in changed])
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(new_manifest, ensure_ascii=False), "utf-8")
        temporary.replace(manifest_path)
        self.documents = documents
        if cache_key is not None:
            with _RESOURCE_CACHE_LOCK:
                _RESOURCE_CACHE[cache_key] = {
                    "signature": source_signature,
                    "documents": documents,
                    "vector": self.vector,
                    "sparse_cache": self._sparse_cache,
                }

    def _sparse_retriever(self, docs: list[Document]):
        key = tuple(sorted(doc.metadata["source_id"] for doc in docs))
        cached = self._sparse_cache.get(key)
        if cached is not None:
            return cached
        inner = BM25Retriever.from_documents(
            docs,
            k=min(MAX_RETRIEVAL_CANDIDATES, len(docs)),
        )
        cached = ZeroSafeBM25Retriever(inner=inner)
        self._sparse_cache[key] = cached
        return cached

    async def search(self, query: str, document_ids: list[str] | None, k: int):
        await self.ensure()
        docs = self.documents or []
        if document_ids:
            docs = [
                doc
                for doc in docs
                if doc.metadata["document_id"] in document_ids
                or doc.metadata["filename"] in document_ids
            ]
        if not docs:
            return []
        k = max(1, min(k, 20))
        sparse = self._sparse_retriever(docs)
        dense = None
        if self.vector:
            kwargs: dict = {"k": k * 3}
            if document_ids:
                kwargs["filter"] = {
                    "document_id": {"$in": list({doc.metadata["document_id"] for doc in docs})}
                }
            dense = self.vector.as_retriever(search_kwargs=kwargs)
        retriever = HybridRetriever(
            dense=dense,
            sparse=sparse
            if self.embedding == "sparse" or self.embedding.startswith("hybrid-")
            else None,
            k=k,
        )
        return await retriever.ainvoke(query)

    async def source_ids(self) -> set[str]:
        await self.ensure()
        return {doc.metadata["source_id"] for doc in self.documents or []}

    async def corpus_identity(self) -> str:
        """Stable identity for search caches and persisted retrieval observations."""
        await self.ensure()
        payload = {
            "embedding": self.embedding,
            "source_ids": sorted(
                doc.metadata["source_id"] for doc in self.documents or []
            ),
        }
        return digest(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    async def read(self, ids: list[str], neighbors: int = 1):
        await self.ensure()
        return _expand_passages(
            self.documents or [], ids, neighbors, preserve_pages=True
        )


def source_record(doc: Document, query: str = "") -> dict:
    meta = doc.metadata
    page = meta.get("page", "")
    return {
        "id": meta["source_id"],
        "name": f"{meta['title']}" + (f" page {page}" if page else ""),
        "docname": meta["title"],
        "document_id": meta["document_id"],
        "citation": meta.get("citation", meta["title"]),
        "filename": meta.get("filename") or None,
        "url": meta.get("url"),
        "page": page,
        "offset": meta.get("start_index", 0),
        "summary": doc.page_content,
        "excerpt": doc.page_content,
        "query": query,
        "used": False,
        "score": None,
        "provenance": "read passage",
    }
