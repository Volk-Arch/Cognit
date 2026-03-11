# Cognit — Persistent Neural Context

> **TL;DR для тех кто в теме:**
> Сохраняем KV-cache Transformer на диск — это и есть "паттерны".
> Tree-sitter парсит весь проект (AST + BM25) и пишет навигационное мемо.
> Пайплайн агентов (`pipeline.json`) последовательно накапливает контекст: навигатор → советники → кодер.
> Всё in-process, одна модель (Qwen3-8B), без переключений.

## Как это работает

Нейросеть один раз читает файлы и **сохраняет своё внутреннее состояние** на диск. Все последующие вопросы идут к уже «знающей» модели — файлы не передаются повторно.

Это не RAG и не векторный поиск. Модель держит код в памяти так, как держит его человек — целиком, с пониманием связей между модулями.

**Навигация** по кодовой базе — через tree-sitter индекс: AST-парсинг Python-файлов + BM25-поиск по именам символов и докстрингам. Детерминистично, быстро, без GPU.

**Паттерны** привязаны к git-ветке — переключился на другую ветку, Cognit автоматически использует её паттерны. При `git commit` паттерны изменённых файлов обновляются через post-commit хук.

---

## Пайплайн агентов

Когда пользователь описывает задачу, Cognit запускает последовательный пайплайн с накопленным контекстом:

```
Задача
  │
  ├─ [1] tree-sitter Navigator → мемо: «bayes_update() строки 42-67, зависит от validate_input()»
  │
  ├─ [2] context agent     → мемо: «цель проекта — демо байесовских методов»
  │
  ├─ [3] arch agent        → мемо: «не нарушать сигнатуру float→float, зависимости X→Y»
  │
  ├─ [4] style agent       → мемо: «snake_case, type hints обязательны, docstring»
  │
  └─ [5] coder             → unified diff по всему накопленному контексту
              │
              ▼
           /patch → файл обновлён
```

Каждая стадия получает **весь контекст предыдущих** — кодер видит файлы + навигационное мемо + мнения всех советников. Агенты пайплайна работают через полный eval (`save_pattern` → `ask_pattern`) — текст агента + накопленный контекст оцениваются заново на каждой стадии. Временные паттерны `_agent_<id>` удаляются после использования.

Порядок и состав стадий задаются в `pipeline.json` клиентского проекта.

```json
{
  "stages": [
    {"id": "navigator", "type": "navigator", "enabled": true},
    {"id": "context",   "type": "agent", "name": "context", "enabled": true},
    {"id": "arch",      "type": "agent", "name": "arch",    "enabled": true},
    {"id": "style",     "type": "agent", "name": "style",   "enabled": true},
    {"id": "coder",     "type": "coder",                    "enabled": true}
  ]
}
```

Можно менять порядок, отключать стадии (`"enabled": false`), редактировать `role` — инструкцию для каждой стадии.

---

## Сценарии использования

**Поправить функцию** — вводишь задачу → tree-sitter находит файл → советники добавляют контекст → кодер пишет diff → `/patch`.

**Создать новый файл** — навигатор находит похожие файлы как образец → советники задают стиль → кодер создаёт файл через `--- /dev/null` diff → `/patch`.

**Детальный анализ** — `/load auth @src/auth.py` → вопросы напрямую к Transformer, диалог накапливается.

**Ревью по соглашениям** — `/review @file` стекает знание style-агента + файл в один вопрос. Паттерн не меняется — эфемерная сессия.

**База знаний команды** — `agents/` с описанием архитектуры, стиля, контекста хранится в git и загружается при каждой сессии.

---

## Установка и настройка

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
pip install tree-sitter tree-sitter-python
```

Скачать модель (GGUF) и положить в `models/`:
- **Transformer**: `Qwen3-8B-Q4_K_M.gguf` (~4.7 GB VRAM)

Одноразовая настройка:

```bash
python cognit_setup.py
```

Создаёт `.echo.json`, шаблоны `agents/`, `pipeline.json` и устанавливает git-хуки.

### Конфигурация (`.echo.json`)

```json
{
  "backend": "transformer",
  "transformer": {
    "model_path":   "models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
    "n_gpu_layers": -1,
    "n_ctx":        8192,
    "max_tokens":   512
  },
  "client_project": "C:/path/to/your/project"
}
```

---

## Быстрый старт

### Типичная сессия — задача неизвестна (пайплайн)

```
python cognit.py

🧠> поправить функцию bayes_update — сделать более расширенной
   📇 Индекс: 3 файлов, 12 символов
   📍 Найдено 4 символа в 1 файле:
      • main.py  (bayes_update, _read_prob, main)

🚀 Пайплайн (5 стадий)
  ✓ [nav   ] navigator
  ✓ [agent ] context
  ✓ [agent ] arch
  ✓ [agent ] style
  ✓ [coder ] coder
  💡 Diff готов → /patch

🧠 [_pipeline]> /patch
✅ Патч применён → main.py
```

### Типичная сессия — файл известен (ручной режим)

```
python cognit.py

🧠> /load auth @src/auth.py
🧠 [auth]> есть ли баги в JWT-проверке?
🧠 [auth]> /edit @src/auth.py убери хардкод RS256
💡 Применить? → /patch
🧠 [auth]> /patch
✅ Патч применён → src/auth.py
```

---

## Техническая информация

### Поток задачи

```
cognit.py (единая точка входа)
  │
  └── Transformer CLI
        ├── задача без паттерна → tree-sitter навигация → пайплайн
        ├── route <задача> → tree-sitter навигация → пайплайн
        ├── /load @file → паттерн → ручной диалог
        └── /patch, /edit, /review, /agent...
```

### Правка кода (/patch и /edit)

**`/patch`** — извлекает **все** unified diff блоки из последнего ответа и применяет к файлам по очереди. Относительные пути из заголовков `+++` резолвятся через `client_project` из `.echo.json`. Перед записью создаётся `.cognit.bak`. Если diff нет — предыдущий ответ инжектируется как контекст и модель переспрашивается.

**`/edit @file задача`** — читает файл заново (не из KV-cache), передаёт содержимое в модель, просит unified diff. Для точных правок конкретного места.

### Хранение паттернов

```
echo_patterns/
  <repo>/
    <branch>/
      <name>.pkl            ← состояние нейросети (KV-cache)
      <name>.json            ← метаданные: файлы-источники, хэши, дата, модель
      _pipeline.pkl/.json    ← временный паттерн кодера (перезаписывается)
      _pipeline_log_*.md     ← лог пайплайна (задача, мемо агентов, diff)
      _code_index.json       ← кеш tree-sitter индекса
```

Паттерны привязаны к модели — `load_pattern` проверяет совместимость квантизации (`meta["model"]` vs текущая модель).

### Политика обновления (grow_policy)

| Политика | Условие | Поведение |
|---|---|---|
| `retrain` | `main`/`master`, `agents/` | Пересоздаётся при коммите |
| `grow ~` | Feature-ветки | Пропускается хуком, накапливает диалог |

```
🧠 [auth]>   ← retrain  🧠 [auth~]>  ← grow
```

### agents/ — база знаний проекта

Markdown-документы со спецификой проекта, хранятся в git клиентского проекта:

```
agents/
  style/global.md      ← именование, форматирование, запреты
  arch/overview.md     ← модули, зависимости, поток запроса
  context/project.md   ← цель проекта, бизнес-правила
```

**Авто-инициализация** — при запуске Cognit сканирует `agents/` и создаёт паттерны автоматически.

**Ручной ambient-режим** — `/agent style arch` включает агентов для каждого следующего вопроса в текущей сессии. Полезен при ручной работе с `/load`. В пайплайне агенты подключаются автоматически.

---

### Добавить нового агента

**Шаг 1 — Создать файл знаний** в клиентском проекте:

```bash
mkdir -p agents/security
# Пишем что агент должен "знать"
cat > agents/security/global.md << 'EOF'
# Security Agent

## Принципы
- Никогда не логировать токены, пароли, секреты
- Все внешние входные данные валидировать до использования
- SQL: только параметризованные запросы
EOF
```

**Шаг 2 — Добавить в пайплайн** (`pipeline.json` клиентского проекта):

```json
{
  "stages": [
    {"id": "navigator", "type": "navigator", "enabled": true},
    {"id": "context",   "type": "agent", "name": "context",  "enabled": true},
    {"id": "security",  "type": "agent", "name": "security", "enabled": true,
     "role": "Задача: {task}\n\nКакие риски безопасности надо учесть? Напиши кратко."},
    {"id": "coder",     "type": "coder",                     "enabled": true}
  ]
}
```

### Git-интеграция

**post-commit** — при каждом коммите обновляет паттерны изменённых файлов.

```bash
python cognit_transformer.py --refresh-file src/auth.py  # пересоздать паттерн вручную
python cognit_transformer.py --status                    # показать устаревшие паттерны
```

### Структура проектов

```
my-project/              ← клиентский git (твой код)
  agents/                ← база знаний (в git!)
  pipeline.json          ← порядок стадий пайплайна (в git!)

Cognit/                  ← этот репо (рядом)
  cognit.py              ← единая точка входа
  cognit_transformer.py  ← Transformer-бэкенд (KV-cache, Qwen3)
  cognit_index.py        ← tree-sitter навигатор (AST + BM25)
  cognit_pipeline.py     ← пайплайн агентов (конфиг)
  cognit_core.py         ← общие утилиты (git, паттерны, хэши)
  cognit_patch.py        ← применение unified diff (мульти-файл)
  cognit_agents.py       ← работа с agents/ (чтение текста, список)
  cognit_setup.py        ← одноразовая настройка
```

---

## Ограничения

**Контекст Transformer ограничен 8192 токенами** — ~300–400 строк кода. Пайплайн обрезает файлы свыше 8000 символов.

**Качество пайплайна зависит от agents/** — если `style/global.md` содержит только шаблон, style-агент не добавит ценности. Заполни руководства под свой проект.

**Паттерны не переносимы между машинами** — привязаны к версии модели и llama-cpp-python. S3-синхронизация запланирована.

**Токенизация ChatML** — `llm.tokenize()` вызывается с `special=True` для корректной обработки специальных токенов Qwen3 (`<|im_start|>`, `<|im_end|>`).

**Только Python** — tree-sitter навигатор парсит `.py` файлы. Для других языков нужно добавить соответствующие tree-sitter грамматики.

---

## Шпаргалка

**Задача неизвестна — пайплайн запускается сам:**
```
python cognit.py
🧠> поправь функцию bayes_update   ← просто пишешь задачу
   [tree-sitter навигирует + пайплайн агентов]
💡 Diff готов → /patch
🧠 [_pipeline]> /patch              ← применяем
```

**Файл известен — ручной режим:**
```
python cognit.py
🧠> /load auth @src/auth.py
🧠 [auth]> есть ли баги?
🧠 [auth]> /edit @src/auth.py убери дублирование
🧠 [auth]> /patch
```

```bash
# Запуск
python cognit.py               # единая точка входа

# Паттерны
/load auth @src/auth.py        # загрузить файл → паттерн 'auth'
/load repo @src/               # загрузить всю папку
/load ?auth @src/auth.py       # принудительный retrain
use auth                       # переключиться на паттерн
/list                          # список паттернов
/exit                          # выход

# Вопросы
как работает функция login?    # ask — паттерн растёт (если grow)
? как работает функция login?  # peek — инвертирует политику на один вопрос

# Навигация и индекс
route добавить rate limiting   # tree-sitter → найти файлы → пайплайн
/index                         # обзор проекта (все символы)
/index bayes update            # BM25-поиск по символам

# Правка кода
/patch                         # все diff-блоки из ответа → применить по файлам
/patch @src/other.py           # применить к конкретному файлу (override)
/edit @src/auth.py убери дублирование  # свежее чтение файла → diff

# Агенты  [ручной режим]
/agents                        # список агентов
/agent style arch              # ambient: каждый вопрос через [auth + style + arch]
/agent off                     # выключить ambient
/review @src/auth.py           # разовое ревью через style-агент
/review arch @src/auth.py      # ревью через конкретный агент

# Headless
python cognit_transformer.py --status               # устаревшие паттерны
python cognit_transformer.py --refresh-file src/auth.py  # пересоздать паттерн
python cognit_setup.py                              # перенастроить
python cognit_setup.py agents                       # создать agents/
```
