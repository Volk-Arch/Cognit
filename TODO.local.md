# TODO

## Done

- [x] Graceful degradation: tree-sitter import через try/except + `_HAS_INDEX` флаг
- [x] Кеш CodeIndex на уровне модуля (`_code_index_cache`) — не пересобирается на каждый route
- [x] `/index` команда: project_summary (без аргументов) и search_summary (с запросом), `--rebuild`
- [x] Замена `_chain_ask` на `_ephemeral_eval_ask` (full eval) во всех call-site'ах
- [x] Удаление `_chain_ask` и `_save_current_kv` (мёртвый код)
- [x] Добавлен `_read_pattern_source` — перечитывает source files из метаданных паттерна

## Backlog

- [ ] Neural ranking: Transformer ранжирует BM25-кандидатов (опционально, после `route`)
- [ ] Поддержка других языков в tree-sitter (JS/TS, Go, Rust)
- [ ] S3-синхронизация паттернов между машинами
- [ ] Веб-интерфейс (read-only дашборд паттернов)
- [ ] Тесты: unit-тесты для cognit_index.py, cognit_patch.py
