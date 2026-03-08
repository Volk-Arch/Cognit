# Cognit — контекст проекта для Claude

## Что это

Система персистентного нейронного контекста для локальных LLM.
Модель один раз читает текст → состояние сохраняется в файл → все следующие вопросы идут без повторной передачи контекста.

## Файлы проекта

| Файл | Назначение |
|---|---|
| `cognit.py` | Оркестратор — единая точка входа, переключает Transformer ↔ RWKV в одном процессе |
| `cognit_transformer.py` | Transformer-бэкенд (Qwen3-8B). KV-cache → `.pkl` |
| `cognit_rwkv.py` | RWKV-бэкенд. Рекуррентное состояние, без лимита контекста |
| `cognit_core.py` | Общие утилиты: git-хелперы, file hash, паттерны, route/expand save/load |
| `cognit_patch.py` | Применение unified diff к файлам (`/patch`) |
| `cognit_agents.py` | Работа с agents/: чтение текста, список агентов |
| `cognit_setup.py` | Одноразовая настройка: `.echo.json`, `.gitignore`, git-хук, `agents/` |
| `README.md` | Полная документация: концепция, команды, сценарии, настройка |

## Архитектура

```
echo_patterns/
  <repo>/
    <branch>/
      <name>.pkl   ← состояние модели (KV-cache или RWKV state)
      <name>.json  ← метаданные (токены, дата, backend, source_files, hashes)

agents/            ← знания о проекте (в git клиентского проекта!)
  style/global.md
  arch/overview.md
  context/project.md

models/            ← GGUF-файлы (в .gitignore)
  Qwen3-8B-GGUF/
  rwkv/
```

## Два репо: Cognit + клиентский проект

Cognit живёт в своём репо. Клиентский проект — отдельный git.
`cognit_setup.py` спрашивает путь к клиентскому проекту → устанавливает
post-commit хук туда с **абсолютным путём** к `cognit_transformer.py`.

```json
// .echo.json — хранит путь к клиентскому проекту
{
  "backend": "transformer",
  "model_name": "Qwen3-8B-Q4_K_M",
  "client_project": "C:/path/to/demo-project"
}
```

⚠️ При переносе Cognit в другую папку — перезапустить `python cognit_setup.py`
(обновит абсолютный путь в хуке и `.echo.json`).

## Ключевые технические детали

**KV-cache (Transformer):**
- `llm.save_state()` / `llm.load_state()` из llama-cpp-python
- Размер растёт с количеством токенов

**RWKV state:**
- Фиксированный размер (~98 KB для 7B) вне зависимости от объёма текста
- Chunked eval: `llm.eval(chunk)` по 512 токенов — нет лимита контекста

**In-process switching (оркестратор):**
- `cli_loop()` возвращает `dict | None` вместо subprocess + sys.exit
- `init_model()` / `unload_model()` в каждом бэкенде
- Handoff RWKV→Transformer: `{"action": "route", "task": ..., "files": [...]}`
- Handoff Transformer→RWKV: `{"action": "expand", "task": ..., "from_pattern": ...}`

## CLI-команды

### Обе модели
```
use <имя>            — выбрать активный паттерн
<вопрос>             — ask (следует политике: grow/retrain)
? <вопрос>           — временно сменить политику (grow↔retrain)
/load <имя> @<путь>  — загрузить файл или папку как паттерн
/load ?<имя> @<путь> — загрузить с принудительным retrain
/list                — список паттернов
/help                — справка
/exit                — выход
```

### Только Transformer (cognit_transformer.py)
```
expand <задача>             — передать задачу в RWKV (handoff)
/patch                      — применить diff из последнего ответа к файлу
/patch @<файл>              — применить к конкретному файлу
/agents                     — список агентов из agents/
/agent <имя> [имя2 ...]     — ambient: каждый вопрос через [паттерн + агенты]
/agent off                  — выключить ambient
/review @<файл>             — ревью файла (агент style, эфемерно)
/review <агент> @<файл>     — ревью через конкретный агент
/review                     — ревью последнего ответа модели
/style @<файл>              — проверка стиля файла
```

### Только RWKV (cognit_rwkv.py)
```
route <задача>       — найти файлы для задачи → handoff на Transformer
```

## Флаги запуска

```bash
python cognit.py                              # оркестратор (из .echo.json)
python cognit.py --rwkv                       # принудительно RWKV
python cognit.py --transformer                # принудительно Transformer
python cognit_transformer.py                  # Transformer напрямую
python cognit_transformer.py --refresh-file <path>  # headless: пересоздать паттерн
python cognit_transformer.py --status         # headless: проверить актуальность
python cognit_rwkv.py                         # RWKV напрямую
python cognit_rwkv.py --auto-expand           # принять pending задачу от Transformer
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
save_route(patterns_dir, index, task, files) → Path         # _route_last.json
load_last_route(patterns_dir) → dict | None                  # None если >24ч
save_expand_request(patterns_dir, task, from_pattern, from_sources) → Path
load_expand_request(patterns_dir) → dict | None              # None если >24ч
```

## Post-commit хук

Устанавливается в клиентский проект. При `git commit`:
- Определяет изменённые файлы
- Вызывает `python /abs/path/cognit_transformer.py --refresh-file <file>` для каждого
- Паттерны, использующие изменённый файл, пересоздаются автоматически

## Статус проекта

Proof of concept. Всё работает, покрыто документацией (README.md).
Следующие возможные шаги: S3-синхронизация паттернов, веб-интерфейс, тесты.
