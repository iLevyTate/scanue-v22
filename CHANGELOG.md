# Changelog

## 1.3.0 — 2026-07-30

Full audit and repair of the multi-agent pipeline (PR #15). Verified by 217
offline tests, ruff and mypy in CI (Python 3.11–3.13), and a live end-to-end
validation against real Ollama models (all checks passing, including
schema-honored delegation and token capture).

### Routing — the multi-agent system is now actually multi-agent
- DLPFC's delegation decision was destroyed by a response reformatter before
  the router read it; nearly every run silently collapsed to MPFC-only. The
  decision is now requested as a schema-validated object
  (`with_structured_output`) with the text parser as fallback, and every run
  records `delegation_source` in its session log.
- Removed an OFC keyword collision that selected reward processing whenever
  MPFC's own role was described; keyword matching is word-boundary anchored.

### Honest failure reporting
- Runs with failed specialists were logged `completed: true` with no trace of
  the failures, and the failed agents' error strings were fed to MPFC as peer
  "insights". Failed agents are now excluded from synthesis, MPFC is told
  which perspectives are missing, the CLI reports partial results, and the
  session log carries `degraded` and `agent_errors`.

### Resource bounds
- HITL feedback history was unbounded and injected into every prompt of every
  run; ~25 entries filled an 8k context window, then every agent failed at
  once. Now recency-windowed and character-budgeted (configurable via
  `SCANUE_FEEDBACK_*`).
- `num_ctx` / `max_tokens` configurable per model; Ollama's silent prompt
  truncation is now detectable; `logs/` pruned to the most recent 50; state
  paths anchored to the project root (`SCANUE_STATE_DIR` overrides).

### Resilience
- Retries with exponential backoff for all providers (previously OpenAI-only,
  leaving the default all-Ollama config with none).
- Per-model `timeout:` in config is honoured (a hardcoded 30 s previously won).

### Observability
- Per-stage token usage, finish reason, model/provider attribution and prompt
  capture; run-level totals and wall clock; truncated generations flagged.
- `scripts/validate.py`: one-command validation against a real provider.

### Quality
- Substantive prompts for all four specialists; MPFC explicitly synthesizes,
  attributes and reconciles peer analyses; `agents/specialized.py` generated
  from a single table.
- Packaging via `pyproject.toml` (`pip install -e .`, `scanue` console
  script); `requirements.txt` removed; ruff + mypy blocking in CI; accurate
  state typing; EOF/empty-input CLI fixes; JSON log corruption on
  unserializable values fixed; DOI badge corrected to this repository's own
  Zenodo record (10.5281/zenodo.14510406).

## 1.2.0-beta and earlier

See git history and the release notes on GitHub.
