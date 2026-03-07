# Cognit — контекст проекта для Claude

## Что это

Система персистентного нейронного контекста для локальных LLM.
Модель один раз читает текст → состояние сохраняется в файл → все следующие вопросы идут без повторной передачи контекста.

## Файлы проекта

| Файл | Назначение |
|---|---|
| `echo_poc.py` | Transformer-бэкенд (Qwen3-8B). KV-cache → `.pkl` |
| `echo_rwkv.py` | RWKV-бэкенд. Рекуррентное состояние, без лимита контекста |
| `echo_core.py` | Общие утилиты: git-хелперы, file hash, паттерны, route/expand save/load |
| `echo_setup.py` | Одноразовая настройка: `.echo.json`, `.gitignore`, git-хук, `agents/` |
| `README.md` | Полная документация: концепция, команды, сценарии, настройка |

## Архитектура

```
echo_patterns/
  <repo>/
    <branch>/
      <name>.pkl   ← состояние модели (KV-cache или RWKV state)
      <name>.json  ← метаданные (токены, дата, backend, source_files, hashes)

agents/            ← знания о проекте (в git!)
  style/global.md
  arch/overview.md
  context/project.md

models/            ← GGUF-файлы (в .gitignore)
  Qwen3-8B-GGUF/
  rwkv/
```

## Два репо: Cognit + клиентский проект

Cognit живёт в своём репо. Клиентский проект — отдельный git.
`echo_setup.py` спрашивает путь к клиентскому проекту → устанавливает
post-commit хук туда с **абсолютным путём** к `echo_poc.py`.

```json
// .echo.json — хранит путь к клиентскому проекту
{
  "backend": "transformer",
  "model_name": "Qwen3-8B-Q4_K_M",
  "client_project": "C:/path/to/demo-project"
}
```

⚠️ При переносе Cognit в другую папку — перезапустить `python echo_setup.py`
(обновит абсолютный путь в хуке и `.echo.json`).

## Ключевые технические детали

**KV-cache (Transformer):**
- `llm.save_state()` / `llm.load_state()` из llama-cpp-python
- Размер растёт с количеством токенов

**RWKV state:**
- Фиксированный размер (~98 KB для 7B) вне зависимости от объёма текста
- Chunked eval: `llm.eval(chunk)` по 512 токенов — нет лимита контекста

**Гибридный рабочий процесс (RWKV → Transformer):**
1. RWKV индексирует всю кодовую базу (`/load repo @src/`)
2. `route <задача>` → RWKV определяет нужные файлы → сохраняет `_route_last.json`
3. Авто-handoff: RWKV выгружается (`del llm` + `gc.collect()`) → запускает `echo_poc.py --auto-route`
4. Transformer подхватывает файлы из `_route_last.json` без вопросов

**Обратный handoff (Transformer → RWKV):**
1. В Transformer: `expand <задача>` → сохраняет `_expand_last.json` → выгружается → запускает `echo_rwkv.py --auto-expand`
2. RWKV стартует, показывает задачу и source-файлы из контекста Transformer
3. Пользователь загружает нужный индекс → задаёт вопросы / делает route

## CLI-команды (одинаковы в обоих бэкендах)

```
use <имя>            — выбрать активный паттерн
<вопрос>             — ask (паттерн растёт, диалог накапливается)
? <вопрос>           — peek (паттерн не меняется, снимок)
/load <имя> @<путь>  — загрузить файл или папку как паттерн
/list                — список паттернов
/exit                — выход

route <задача>       — только echo_rwkv.py: маршрутизация файлов
expand <задача>      — только echo_poc.py: передать задачу в RWKV (handoff)
```

## Флаги запуска

```bash
python echo_poc.py                        # интерактивный режим
python echo_poc.py --auto-route           # авто-загрузка из _route_last.json (после route в RWKV)
python echo_poc.py --refresh-file <path>  # headless: пересоздать паттерн для файла
python echo_poc.py --status               # headless: проверить актуальность паттернов

python echo_rwkv.py                       # интерактивный режим
python echo_rwkv.py --auto-expand         # запуск с отображением задачи от Transformer

python echo_setup.py          # полная настройка
python echo_setup.py agents   # только создать agents/ в клиентском проекте
```

## echo_core.py — ключевые функции

```python
git_repo_name()                          # → имя репо из git
git_branch()                             # → ветка, "/" → "-"
make_patterns_dir(repo, branch) → str    # создаёт и возвращает путь
file_hash(path) → str                    # SHA-256 первых+последних 64KB, [:16]
check_stale_sources(patterns_dir, name)  # → список изменённых файлов
save_route(patterns_dir, index, task, files) → Path         # _route_last.json
load_last_route(patterns_dir) → dict | None                  # None если >24ч
save_expand_request(patterns_dir, task, from_pattern, from_sources) → Path  # _expand_last.json
load_expand_request(patterns_dir) → dict | None              # None если >24ч
```

## Post-commit хук

Устанавливается в клиентский проект. При `git commit`:
- Определяет изменённые файлы
- Вызывает `python /abs/path/echo_poc.py --refresh-file <file>` для каждого
- Паттерны, использующие изменённый файл, пересоздаются автоматически

## Статус проекта

Proof of concept. Всё работает, покрыто документацией (README.md).
Следующие возможные шаги: S3-синхронизация паттернов, веб-интерфейс, тесты.
