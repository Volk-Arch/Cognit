# Cognit — контекст для Claude

Система персистентного нейронного контекста для локальных LLM.
Полная документация: `README.md`.

## Файлы проекта

| Файл | Назначение |
|---|---|
| `cognit.py` | Единая точка входа |
| `cognit_transformer.py` | Transformer-бэкенд (Qwen3-8B). KV-cache → `.pkl` |
| `cognit_index.py` | Tree-sitter навигатор: AST-парсинг Python + BM25 |
| `cognit_core.py` | Общие утилиты: git, file hash, паттерны, route |
| `cognit_pipeline.py` | Конфигурация пайплайна агентов (`pipeline.json`) |
| `cognit_patch.py` | Применение unified diff (`/patch`), мульти-файл |
| `cognit_agents.py` | Работа с agents/: чтение текста, список агентов |
| `cognit_setup.py` | Одноразовая настройка: `.echo.json`, git-хук, `agents/` |

## Ключевые API

```python
# cognit_core.py
git_repo_name() → str
git_branch() → str
make_patterns_dir(repo, branch) → str
file_hash(path) → str                    # SHA-256 [:16]
pattern_exists(patterns_dir, name) → bool
check_stale_sources(patterns_dir, name)   # → список изменённых файлов
collect_text_files(path) → list[Path]
save_route(patterns_dir, index, task, files) → Path
load_last_route(patterns_dir) → dict | None

# cognit_index.py
CodeIndex(project_dir, cache_dir)
  .build() → int
  .search(query, top_k) → list[SearchResult]
  .files_for_query(query, top_k) → list[str]
  .project_summary() → str
  .search_summary(query, top_k) → str

# cognit_patch.py
_extract_diff(text) → str | None
_extract_all_diffs(text) → list[str]
_diff_target(diff) → str | None
apply_patch(diff, file_path) → bool
```

## Технические нюансы

- `llm.tokenize(..., special=True)` — обязательно для ChatML (`<|im_start|>`, `<|im_end|>`)
- KV-cache continuation ненадёжна для инжектов 700+ токенов → использовать полный eval (`save_pattern` + `ask_pattern`)
- Агенты в пайплайне работают через полный eval, временные паттерны `_agent_<id>` удаляются после использования
- `/patch` резолвит пути через `client_project` из `.echo.json`
