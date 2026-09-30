"""Configuration shared by the research service, UI, and benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CLOUDFLARE_MODEL = "cloudflare/@cf/zai-org/glm-4.7-flash"
DEFAULT_EMBEDDING = "hybrid-st-sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_EMBEDDINGS = {
    "sparse",
    "st-sentence-transformers/all-MiniLM-L6-v2",
    "st-multi-qa-MiniLM-L6-cos-v1",
    "st-sentence-transformers/all-mpnet-base-v2",
    DEFAULT_EMBEDDING,
}
REASONING_EFFORTS = {"low", "medium", "high"}


@dataclass(frozen=True)
class EvidencePolicy:
    """One source of truth for retrieval and publication limits."""

    search_results: int = 6
    max_cited_passages: int = 8
    max_source_documents: int = 5

    def __post_init__(self) -> None:
        if not 1 <= self.search_results <= 20:
            raise ValueError("search_results must be between 1 and 20")
        if not 1 <= self.max_cited_passages <= 20:
            raise ValueError("max_cited_passages must be between 1 and 20")
        if not 1 <= self.max_source_documents <= 20:
            raise ValueError("max_source_documents must be between 1 and 20")


@dataclass(frozen=True)
class AgentConfig:
    """Public configuration for one research service or run."""

    home: Path
    embedding: str = DEFAULT_EMBEDDING
    reasoning_effort: str = "medium"
    web_search: bool = True
    evidence: EvidencePolicy = EvidencePolicy()

    def __post_init__(self) -> None:
        if self.embedding not in SUPPORTED_EMBEDDINGS:
            raise ValueError(f"Unsupported embedding: {self.embedding}")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError("reasoning_effort must be low, medium, or high")


def default_data_directory() -> Path:
    configured = os.getenv("PQA_UI_HOME") or os.getenv("PQA_HOME")
    return Path(configured) if configured else Path.home() / ".paperqa" / "ui"

