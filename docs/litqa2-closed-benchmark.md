# LitQA2 closed-corpus benchmark

This benchmark evaluates the local LangGraph research agent without web tools. It uses the
pinned `futurehouse/lab-bench` LitQA2 Parquet revision
`5c77cec648430f30611808808861eb86f81d5eaa` and compares BM25, normalized
`all-MiniLM-L6-v2`, and equal-weight reciprocal-rank fusion over one canonical chunk set.

Install the benchmark dependencies:

```powershell
pip install -e ".[benchmarks]"
```

Run the stages in order. Every command is resumable, and `--only QUESTION_ID` can be repeated
where the command supports selection.

```powershell
pqa-litqa prepare
pqa-litqa download --local-map local-pdfs.json
pqa-litqa index
pqa-litqa retrieve
pqa-litqa run --scope paired --env-file .env
pqa-litqa run --scope shared --env-file .env
pqa-litqa report
```

Artifacts are written under
`.litqa-bench/experiments/paperqa-litqa2-closed-v1/`. Validated PDFs are shared between
experiments in `.litqa-bench/pdf-cache/`. `download` records every unresolved or blocked DOI;
it never substitutes an abstract. A local replacement map is a JSON object whose keys are DOI
strings and whose values are PDF paths. Unpaywall is queried only when `--unpaywall-email` or
`UNPAYWALL_EMAIL` is configured. Terminal failures are not repeated on an ordinary resume;
use `--retry-missing` after credentials, network access, or resolver availability changes.

`index` freezes PDF-derived pages, 254-token page-bounded chunks with at most 32 tokens of
overlap, hashes, parser information, and the MiniLM model revision. Once retrieval or answer
evaluation starts, changing corpus membership requires a new `--experiment-id`.

Exact normalized passage matches enter the primary metrics automatically. Approximate matches
are review candidates only. To accept reviewed candidates, create a versioned file such as:

```json
{
  "version": 1,
  "overrides": {
    "question-id": ["src-0123456789abcdef01234567"]
  }
}
```

Then run `pqa-litqa index --alignment-overrides reviewed.json`. Overrides may reference only
chunks from the question's associated papers.

The paid pilot is selected once by seeded question-ID hashing from questions whose complete
paired corpus parsed successfully. Paired scope exposes only associated papers; shared scope
exposes every document in the frozen corpus without revealing the gold source mapping. Each
scope has isolated conversations and resumable artifacts, while both use the same cumulative
cost ledger. Runs are sequential and interleave the three retrievers per question. All model
calls reserve budget before submission; the default cumulative ceiling is US$10.
Provider-paused work resumes only with `--resume-provider`, while failed or timed-out work
starts a new isolated attempt only with `--retry-failed`. If a saved provider checkpoint is
repeatedly unrecoverable, `--restart-paused` archives that artifact under the scope's `history/`
directory and starts a new isolated attempt without touching completed answers.

The HTML report distinguishes paired and shared retrieval, reports the common eligible
denominator, acquisition/alignment failures, answer and citation metrics, stage latency, model
calls, and retained uncertain charges. Gold-passage citation matching is an evidence proxy, not
a claim-entailment metric. This is a custom closed-corpus evaluation, not a reproduction of the
published LAB-Bench protocol.
