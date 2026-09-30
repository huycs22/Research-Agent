# Focused-agent cleanup inventory

The pre-cleanup source is recoverable from
`.refactor-snapshots/pre-focused-agent-20260930-102813.zip` (29,448,703 bytes).
The archive contains the source, tests, scripts, documentation, package metadata,
and repository-hosting configuration that existed before the refactor. It excludes `.env`,
papers, conversations, indexes, checkpoints, PDF caches, and experiment data.

## Removed application code

- Original Aviary/LDP agent: `src/paperqa/agents/`, `_ldp_shims.py`
- Original `Docs` pipeline and model abstractions: `docs.py`, `core.py`, `llms.py`,
  `prompts.py`, `settings.py`, and `types.py`
- Metadata/discovery integrations: `clients/`, `sources/`, and `contrib/`
- Old configuration presets: `configs/`
- Old reader/utilities after their required text-only behavior moved into
  `research/parsing.py`: `readers.py`, `utils.py`, and `paths.py`
- Deprecated fixed UI agent and monolithic server: `ui/research_agent.py` and
  `ui/server.py`

## Removed project material

- Upstream package workspace in `packages/`
- Legacy hosted-CI workflow configuration
- Stale upstream-only contributor, mailmap, attributes, and pre-commit files
- Upstream tutorials and test cassettes/fixtures
- Tests for removed CLIs, agents, metadata clients, clinical-trial tools,
  settings, and old UI migrations
- Deprecated migration/fixture scripts

## Preserved

- LangChain/LangGraph research runtime, local retrieval, memory, web tools, UI,
  LitQA, DeepResearch Bench, and focused tests
- Licenses and attribution
- Every runtime data directory, including `.litqa-bench`, `.deepresearch-bench`,
  `.agent-stability`, `.research-verification`, and `.test-ui-*`
- `.git`, `.env`, local papers, indexes, conversations, checkpoints, and ledgers
