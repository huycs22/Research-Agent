# Research-agent stability verification

## Scope

This verification precedes any new LitQA experiment. It exercises the shared
research service with web search disabled and leaves existing LitQA artifacts
unchanged. The implementation stabilizes requested-passage expansion, cache and
corpus identities, zero-overlap BM25 behavior, numeric excerpts, retrieval
traces, execution phases, completion contracts, provider recovery, and durable
conversation memory.

The opt-in runner is:

```powershell
# Local BM25, MiniLM, and Hybrid retrieval only; no provider calls.
.venv/Scripts/python.exe scripts/verify_agent_stability.py

# Full resumable Cloudflare matrix. A paused provider run is not retried unless
# --resume-provider is supplied explicitly.
.venv/Scripts/python.exe scripts/verify_agent_stability.py --live
```

Artifacts are written under the ignored `.agent-stability/` directory. The
three-retriever offline run is preserved under `.agent-stability/offline/`, so
it cannot overwrite live reports. Each live artifact
records expected and observed behavior, the exact model-visible
evidence context, retrieval/tool traces, latency, usage, verification state, and
request-level cost entries. The runner reserves each physical provider call in
a new $1 ledger and, when present, the existing LitQA $10 cumulative ledger.
Uncertain calls retain their reservations.

## Focused-refactor validation

The clean locked environment passed **50 Python tests** and **6 JavaScript UI
tests**. Ruff also passed across `src`, `scripts`, and `tests`. The Python suite
covers both benchmark pipelines, cost reservations, web parsing, retrieval,
durable service behavior, and the exact guard/citation regression.

In that regression, six search results expand to eleven passages, research hits
its guard, finalization receives at most eight passages from at most five
documents, an invalid nine-citation draft is rejected, and the reserved
correction produces a terminal cited answer. Additional focused checks cover:

- simple conversation with zero provider/tool calls and mixed greeting/research
  routing through the shared router;
- requested-passage expansion, zero-overlap BM25 rejection, and stable evidence
  limits;
- exact completion markers, conflicting markers, abstention, and raw tool-call
  markup detection;
- restart isolation plus deletion of conversation JSON, run records, and SQLite
  checkpoints;
- LitQA alignment/retrieval/report fixtures and DeepResearch cost/proxy/report
  fixtures; and
- Cloudflare-only agent configuration and browser rendering of citations,
  tables, lists, code, and safe links.

## Real local retrieval

The four-document fixture includes numerical evidence, two studies that must not
be conflated, a conflicting replication, and a distractor containing embedded
instructions. All three retrievers returned the numeric evidence:

| Retriever | Cold/check time | Returned chunks | Numeric passage found |
|---|---:|---:|---|
| BM25 | 0.235 s | 3 | yes |
| MiniLM L6 | 11.007 s | 4 | yes |
| Hybrid | 0.323 s | 4 | yes |

MiniLM produced a normalized 384-dimensional vector. The MiniLM check was a real
local model load, and the dense/hybrid checks used the real persisted Chroma
index rather than mocked embeddings.

## Preserved live Cloudflare evidence

The cleanup did not spend money or rerun the paid matrix. The following results
are preserved from the immediately preceding stabilization run and its ledgers;
they are not a post-refactor benchmark result.

Both basic conversational prompts completed locally with zero model calls and
zero tool calls. The account initially configured in `.env` had exhausted its
daily free allocation. An alternate user-supplied credential was injected only
into the live process, was not printed or saved, and successfully completed the
matrix.

The initial full run executed all **15 research cases** (five prompts across
BM25, MiniLM, and Hybrid) plus **eight memory turns and one isolation probe**.
It exposed three problems that terminal-status-only reporting had hidden:

- two absent-evidence answers returned unrelated partial excerpts instead of an
  explicit limitation;
- the follow-up calculation found and calculated the correct value but lost the
  result while falling back after missing citations;
- conflict answers inferred significance, power, or causal explanations beyond
  the fixture text, especially when provider verification timed out.

The runtime now states corpus absence explicitly, preserves the current turn's
bounded calculator result with units and source provenance, runs claim support
verification even after loop finalization, and gives both answer generation and
verification explicit rules against unsupported significance and causality.
The report now separates execution completeness from semantic acceptance and
checks memory content rather than only terminal status.

Clean focused reruns then passed:

| Check | Runs | Result |
|---|---:|---|
| Absent evidence | 3 retrievers | All explicitly reported the missing evidence |
| Memory and isolation | 8 + 1 | All facts, correction, preferences, calculation, restart, and isolation checks passed |
| Conflict synthesis | 3 retrievers | Required facts present; prohibited unsupported interpretations absent |

Several otherwise valid answers remain labeled `unresolved` because Cloudflare
verification calls intermittently timed out or returned HTTP 500. They pass the
deterministic content checks but are not described as model-verified. The final
conflict answers correctly distinguish the documented study differences from a
demonstrated causal explanation, although cautious hypotheses still require
human judgment.

Across the original and focused stability experiments, the dedicated ledgers
record **$0.0250536475 spent** and **$0.0284748610 reserved** for uncertain
provider calls. The shared cumulative $10 ledger records **$0.2771408190 spent**
and **$0.1941794740 reserved**. All values remain below both ceilings.
