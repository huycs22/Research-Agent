# PaperQA research agent

A focused local research application that searches a paper library, reads exact
passages, and produces cited answers. It uses LangChain/LangGraph, Cloudflare
Workers AI GLM-4.7-Flash, BM25, local MiniLM embeddings, and Chroma.

## Install and run

```powershell
uv sync --extra local --extra benchmarks --extra dev
uv run pqa-ui --data-dir .test-ui-data
```

Configure the model through environment variables or an ignored `.env` file:

```text
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
REASONING_EFFORT=medium
```

The UI opens at `http://127.0.0.1:8877/`. Its data directory contains `papers/`,
`indexes/`, `conversations/`, `runs-v2/`, and `research-checkpoints.sqlite`.

## Python API

```python
from pathlib import Path
from paperqa import EvidencePolicy, ResearchService

service = ResearchService(Path(".research-data"))
run = await service.run(
    "What does the paper report about retrieval accuracy?",
    embedding="hybrid-st-sentence-transformers/all-MiniLM-L6-v2",
    web_search=False,
    evidence_policy=EvidencePolicy(
        search_results=6,
        max_cited_passages=8,
        max_source_documents=5,
    ),
)
print(run.result["answer"] if run.result else run.error)
service.close()
```

`ResearchService` also provides `inspect`, `stop`, `resume`, conversation listing,
and deletion. Auto, Research, and Chat modes share the same local router. Prepared
benchmark libraries can be passed through `library=` when calling `run`.

Existing data remains in place. Conversation IDs, run IDs, source IDs, and the
`runs-v2`/checkpoint formats are preserved; the new parser/index identity writes
to a new namespace instead of mutating older indexes or saved citations.

## Benchmarks

LitQA closed-corpus commands:

```powershell
uv run pqa-litqa prepare
uv run pqa-litqa download
uv run pqa-litqa index
uv run pqa-litqa retrieve
uv run pqa-litqa run
uv run pqa-litqa report
```

DeepResearch Bench commands:

```powershell
uv run pqa-deepresearch prepare
uv run pqa-deepresearch generate
uv run pqa-deepresearch evaluate
uv run pqa-deepresearch report
```

DeepResearch generation uses Cloudflare. Its pinned external evaluator optionally
uses OpenAI/OpenRouter and Jina credentials; those integrations are isolated from
the agent provider configuration.

## Development

```powershell
uv run pytest
uv run ruff check src tests
```

See [architecture](docs/architecture.md),
[LitQA details](docs/litqa2-closed-benchmark.md), and
[DeepResearch details](docs/deepresearch-benchmark.md).

This project retains the original Apache-2.0 license and upstream attribution
in `LICENSE` and `NOTICE`.
