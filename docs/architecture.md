# Research-agent architecture

`ResearchService` is the public boundary. Python callers, the web UI, and both
benchmarks use it for conversations, runs, stop/resume, and deletion. Durable
conversation JSON, run JSON, and LangGraph SQLite checkpoints remain in the
configured data directory.

The focused package boundaries are:

| Module | Responsibility |
|---|---|
| `research/config.py` | Agent settings and the shared evidence policy |
| `research/service.py` | Public lifecycle API and durable execution |
| `research/runtime.py` | LangGraph phases and permitted tool execution |
| `research/router.py` | Shared local intent routing for UI and API |
| `research/validation.py` | Citation and completion-contract validation |
| `research/parsing.py` | Versioned PDF/TXT/Markdown/HTML extraction |
| `research/library.py` | Chunking, BM25, MiniLM, Chroma, and fusion |
| `research/storage.py` | Backward-readable conversation and run records |
| `research/providers.py`, `safety.py` | Cloudflare calls, retries, timeouts, and accounting |
| `ui/app.py` | Thin HTTP/static-file adapter over the service |
| `benchmarks/` | LitQA and DeepResearch execution, scoring, reporting, and neutral cost utilities |

The runtime has four persisted phases: `research`, `finalization`, `correction`,
and `finished`. Research tools can run only in the research phase. Reaching a
loop threshold changes phase and records its exact reason. Finalization sees a
fresh context containing the current request, remembered constraints, and the
selected citable evidence. Historical tool output and rejected drafts remain in
the checkpoint but are not sent again as answer evidence.

`EvidencePolicy` defines the shared defaults: six search results, eight passage
citations, and five source documents. Retrieval retains all requested passages
and neighbor expansions. Finalization selects within the policy and verification
accepts citations only from that selected set. One final answer and one correction
remain available after research thresholds are reached.

`validation.py` owns deterministic citation-ID parsing, raw tool-markup detection,
and completion-contract scoring. The orchestration layer calls these checks but
does not define benchmark-specific answer labels or gold answers.

`Library` parses PDF, text, Markdown, and HTML into page-preserving LangChain
documents. It creates token-bounded chunks with stable offsets and source IDs.
Sparse retrieval uses `rank-bm25`; dense retrieval uses local sentence-transformer
embeddings in Chroma; hybrid retrieval uses reciprocal-rank fusion.

Cloudflare Workers AI GLM-4.7-Flash is the only agent model. The OpenAI-compatible
LangChain client is an HTTP client for Cloudflare and does not expose OpenAI as an
agent provider. DeepResearch Bench may separately invoke its pinned optional
OpenAI/OpenRouter judges.
