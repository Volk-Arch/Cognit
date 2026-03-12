# TODO

## Done

- [x] Graceful degradation: tree-sitter import via try/except + `_HAS_INDEX` flag
- [x] Module-level CodeIndex cache (`_code_index_cache`) — no rebuild on each route
- [x] `/index` command: project_summary (no args) and search_summary (with query), `--rebuild`
- [x] Replace `_chain_ask` with `_ephemeral_eval_ask` (full eval) in all call-sites
- [x] Remove `_chain_ask` and `_save_current_kv` (dead code)
- [x] Add `_read_pattern_source` — re-reads source files from pattern metadata
- [x] Add analyst stage to pipeline (after navigator, before agents)
- [x] License audit for public release (PolyForm-Noncommercial-1.0.0, .gitignore)
- [x] i18n: cognit_i18n.py — bilingual EN/RU messages, agent roles, HELP
- [x] i18n: all .py files — comments/docstrings to English, prints → msg()
- [x] i18n: lang selection in cognit_setup.py, stored in .echo.json
- [x] i18n: README.md and ARTICLE.md translated to English (RU as .ru.md)
- [x] Remove /no_think (Qwen3-specific), clean up Russian LLM prompts
- [x] Move system prompt to cognit_i18n.py (bilingual)
- [x] Move pipeline stage comments to cognit_i18n.py (STAGE_COMMENTS)
- [x] cognit_setup.py: all prints → msg(), _AGENT_TEMPLATES → cognit_i18n.py
- [x] Lightweight git hook: cognit_hook.py (no model loading, stale detection, BM25 update)
- [x] Stale pattern management: mark_stale/clear_stale/is_stale in cognit_core.py
- [x] LLM reranking: BM25 top-20 → LLM rerank → top-8 (retrieve-then-rerank)
- [x] ARTICLE.md: added "Why This Architecture Looks Like the Brain" section (EN + RU)

## Backlog

- [ ] Try Qwen2.5-Coder-7B-Instruct as replacement model
- [ ] Test git hook + LLM reranking end-to-end
- [ ] Tree-sitter support for other languages (JS/TS, Go, Rust)
- [ ] S3 pattern sync between machines
- [ ] Web interface (read-only pattern dashboard)
- [ ] Tests: unit tests for cognit_index.py, cognit_patch.py
