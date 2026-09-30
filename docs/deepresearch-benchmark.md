# DeepResearch Bench pilot

PaperQA includes a reproducible, isolated five-task English pilot for the original
[DeepResearch Bench](https://deepresearch-bench.github.io/). It evaluates reports,
not short multiple-choice answers. The pilot is diagnostic and is not a leaderboard
submission or a statistically representative benchmark result.

The frozen task IDs are 51, 59, 67, 68, and 71. Generation uses Cloudflare
GLM-4.7-Flash at medium reasoning in Research mode, with public search enabled and
the Hybrid MiniLM + BM25 configuration selected. Each task receives an empty library,
a fresh conversation, one attempt, and a ten-minute outer timeout. The empty library
is intentional: this pilot measures the agent's public research behavior, not local
embedding quality.

## Commands

Run from the repository root with the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch prepare
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch preflight --stage generation
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch generate
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch preflight --stage evaluation
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch evaluate --resume
.\.venv\Scripts\python.exe -m paperqa.benchmarks.deepresearch report
```

Global options (such as `--workspace`, `--experiment-id`, and `--env-file`) must
precede the subcommand. `prepare --skip-evaluator-install` downloads and pins the
official source and data without creating its separate environment. `generate
--only 51` is useful for an explicitly partial smoke run. Re-running `generate`
does not replace terminal attempts. A provider-paused task is resumed only with
`--resume-provider`; a clarification request stays unfinished because the runner
must not invent a user's answer.

## Credentials and spending

Generation requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Official
RACE/FACT evaluation uses either OpenAI direct (default) or OpenRouter and requires
the corresponding API key. FACT also requires `JINA_API_KEY` for source scraping.
Missing judge credentials do not prevent report generation.

Generation uses a LangChain callback and evaluation uses a localhost-only metering
proxy. Before every GLM or judge call they reserve the request's maximum possible
output at the verified official token price. After a successful response they
reconcile the reservation using provider usage. An uncertain failure keeps its
reservation. The persistent ledger refuses a request that would take combined spend
and reservations above US$10. The ceiling cannot be silently changed for an existing
experiment.

The official pinned evaluator currently uses GPT-5.5 for RACE, GPT-5.6 Luna for
cleaning, and GPT-5.4 Mini for FACT. GLM-4.7-Flash generation uses Cloudflare's
published token-equivalent rates ($0.0605/M input, $0.40/M output); the manifest
records the official pricing URL and verification date.

## Artifacts

Artifacts live under `.deepresearch-bench/experiments/paperqa-drb-en5-v1/`:

- `manifest.json`: upstream commit, dataset/config/code hashes, dirty working-tree
  fingerprint, model settings, limitations, prices, and dates.
- `tasks.jsonl`: only the five exact prompts visible to the generator.
- `tasks/task-*.json`: raw answer, converted report, plan, observable events,
  evidence, source mapping, usage, timing, flags, and finish status.
- `paperqa-glm47flash.jsonl`: official `id`, `prompt`, `article` export.
- `agent-state/`: isolated conversations, checkpoints, runs, evidence, and indexes.
- `evaluation/`: resumable upstream RACE and FACT stage outputs.
- `cost-ledger.json`: persistent request reservations and reported spend.
- `results.json` and `report.html`: machine-readable and human-readable summaries.

Reference reports and criteria stay in the evaluator checkout and are only passed to
the official scoring programs. They are never copied into the agent's state or prompt.
Reports with incomplete evaluation always show attempted, completed, and scored
counts and are labeled incomplete rather than presenting a five-task aggregate.
