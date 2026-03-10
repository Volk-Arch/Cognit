# Cognit — контекст проекта для Claude

## Что это

Система персистентного нейронного контекста для локальных LLM.
Модель один раз читает текст → состояние сохраняется в файл → все следующие вопросы идут без повторной передачи контекста.

## Файлы проекта

| Файл | Назначение |
|---|---|
| `cognit.py` | Единая точка входа — запускает Transformer |
| `cognit_transformer.py` | Transformer-бэкенд (Qwen3-8B). KV-cache → `.pkl` |
| `cognit_index.py` | Tree-sitter навигатор: AST-парсинг Python + BM25-поиск по символам |
| `cognit_core.py` | Общие утилиты: git-хелперы, file hash, паттерны, route save/load |
| `cognit_pipeline.py` | Конфигурация пайплайна агентов (`pipeline.json`) |
| `cognit_patch.py` | Применение unified diff к файлам (`/patch`), мульти-файл (`_extract_all_diffs`) |
| `cognit_agents.py` | Работа с agents/: чтение текста (`read_agent_text`), список агентов |
| `cognit_setup.py` | Одноразовая настройка: `.echo.json`, `.gitignore`, git-хук, `agents/` |
| `README.md` | Полная документация: концепция, команды, сценарии, настройка |

## Архитектура

```
echo_patterns/
  <repo>/
    <branch>/
      <name>.pkl           ← KV-cache состояние модели
      <name>.json          ← метаданные (токены, дата, backend, source_files, hashes)
      _code_index.json     ← кеш tree-sitter индекса

agents/            ← знания о проекте (в git клиентского проекта!)
  style/global.md
  arch/overview.md
  context/project.md

models/            ← GGUF-файлы (в .gitignore)
  Qwen3-8B-GGUF/
```

## Два репо: Cognit + клиентский проект

Cognit живёт в своём репо. Клиентский проект — отдельный git.
`cognit_setup.py` спрашивает путь к клиентскому проекту → устанавливает
post-commit хук туда с **абсолютным путём** к `cognit_transformer.py`.

```json
// .echo.json — хранит путь к клиентскому проекту + конфиг модели
{
  "backend": "transformer",
  "transformer": { "model_path": "...", "n_gpu_layers": -1, "n_ctx": 8192, "max_tokens": 512 },
  "client_project": "C:/path/to/demo-project"
}
```

⚠️ При переносе Cognit в другую папку — перезапустить `python cognit_setup.py`
(обновит абсолютный путь в хуке и `.echo.json`).

## Ключевые технические детали

**KV-cache (Transformer):**
- `llm.save_state()` / `llm.load_state()` из llama-cpp-python
- Размер растёт с количеством токенов
- `llm.tokenize(..., special=True)` — обязательно для ChatML (`<|im_start|>`, `<|im_end|>`)
- `load_pattern` проверяет совместимость модели (квантизация) через `meta["model"]`

**Tree-sitter навигатор (cognit_index.py):**
- `CodeIndex(project_dir)` — AST-парсинг всех `.py` файлов
- BM25-поиск по именам символов (функции, классы, импорты) + докстрингам
- Инкрементальный: пропускает файлы с неизменённым хешем
- Кеш на диск: `_code_index.json` в директории паттернов
- Зависимости: `tree-sitter`, `tree-sitter-python`

## CLI-команды

```
use <имя>            — выбрать активный паттерн
<вопрос>             — ask (следует политике: grow/retrain)
? <вопрос>           — временно сменить политику (grow↔retrain)
route <задача>       — найти файлы через tree-sitter индекс → пайплайн
/load <имя> @<путь>  — загрузить файл или папку как паттерн
/load ?<имя> @<путь> — загрузить с принудительным retrain
/patch               — применить все diff-блоки из ответа к файлам
/patch @<файл>       — применить к конкретному файлу (override)
/edit @<файл> <задача> — свежее чтение файла → unified diff
/agents              — список агентов из agents/
/agent <имя> [имя2 ...]  — ambient: каждый вопрос через [паттерн + агенты]
/agent off           — выключить ambient
/review @<файл>      — ревью файла (агент style, эфемерно)
/review <агент> @<файл>  — ревью через конкретный агент
/review              — ревью последнего ответа модели
/style @<файл>       — проверка стиля файла
/list                — список паттернов
/help                — справка
/exit                — выход
```

## Флаги запуска

```bash
python cognit.py                              # единая точка входа
python cognit_transformer.py                  # Transformer напрямую
python cognit_transformer.py --refresh-file <path>  # headless: пересоздать паттерн
python cognit_transformer.py --status         # headless: проверить актуальность
python cognit_setup.py                        # полная настройка
python cognit_setup.py agents                 # только создать agents/
```

## cognit_core.py — ключевые функции

```python
git_repo_name()                          # → имя репо из git
git_branch()                             # → ветка, "/" → "-"
make_patterns_dir(repo, branch) → str    # создаёт и возвращает путь
file_hash(path) → str                    # SHA-256 первых+последних 64KB, [:16]
pattern_exists(patterns_dir, name) → bool
check_stale_sources(patterns_dir, name)  # → список изменённых файлов
collect_text_files(path) → list[Path]    # git ls-files или rglob
save_route(patterns_dir, index, task, files) → Path   # _route_last.json
load_last_route(patterns_dir) → dict | None            # None если >24ч
```

## cognit_index.py — ключевые функции

```python
CodeIndex(project_dir, cache_dir)        # AST-индекс проекта
  .build() → int                         # парсит .py файлы, возвращает кол-во
  .search(query, top_k) → list[SearchResult]  # BM25-поиск по символам
  .files_for_query(query, top_k) → list[str]  # уникальные файлы
  .file_summary(filepath) → str          # краткое описание файла
  .project_summary() → str              # обзор всего проекта
  .search_summary(query, top_k) → str   # текстовый отчёт для промпта
```

## Post-commit хук

Устанавливается в клиентский проект. При `git commit`:
- Определяет изменённые файлы
- Вызывает `python /abs/path/cognit_transformer.py --refresh-file <file>` для каждого
- Паттерны, использующие изменённый файл, пересоздаются автоматически

## Пайплайн агентов — техника

Агенты в `_run_pipeline` работают через **полный eval** (не KV-cache continuation):
1. `_read_agent_text(name)` — сырой текст из `agents/<name>/`
2. `save_pattern("_agent_<id>", agent_text + shared)` — полный eval с нуля
3. `ask_pattern("_agent_<id>", role + "/no_think")` — вопрос к свежему KV-cache
4. Временный паттерн удаляется после использования

Кодер работает аналогично: `save_pattern("_pipeline", shared)` → `ask_pattern`.

KV-cache continuation (`load_state` + `eval(inject)` + `generate`) ненадёжна
для больших инжектов (700+ токенов) — модель сразу генерирует `<|im_end|>`.

## cognit_patch.py — ключевые функции

```python
_extract_diff(text) → str | None         # первый diff-блок из ответа
_extract_all_diffs(text) → list[str]     # все diff-блоки (мульти-файл)
_diff_target(diff) → str | None          # путь из заголовка +++
_is_new_file_diff(diff) → bool           # True если --- /dev/null
apply_patch(diff, file_path) → bool      # применить diff, создать .cognit.bak
```

`/patch` в CLI резолвит относительные пути через `client_project` из `.echo.json`.

## Статус проекта

Proof of concept. Основной функционал работает, покрыт документацией.
Следующие возможные шаги: S3-синхронизация паттернов, веб-интерфейс, тесты.
