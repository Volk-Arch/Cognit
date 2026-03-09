# Cognit — Persistent Neural Context

> **TL;DR для тех кто в теме:**
> Сохраняем KV-cache Transformer и рекуррентное состояние RWKV на диск — это и есть "паттерны".
> RWKV (~98 KB фиксированного состояния) индексирует весь проект и пишет навигационное мемо.
> Пайплайн агентов (`pipeline.json`) последовательно накапливает контекст: RWKV → советники → кодер.
> Переключение моделей in-process через возврат handoff-dict из `cli_loop()`, без subprocess.

## Как это работает

Нейросеть один раз читает файлы и **сохраняет своё внутреннее состояние** на диск. Все последующие вопросы идут к уже «знающей» модели — файлы не передаются повторно.

Это не RAG и не векторный поиск. Модель держит код в памяти так, как держит его человек — целиком, с пониманием связей между модулями.

Система работает **последовательно на одной видеокарте** — модели не запускаются параллельно, при переключении одна выгружается из VRAM, другая загружается.

В системе **две модели** с разными ролями:

| | RWKV | Transformer |
|---|---|---|
| Роль | Навигатор + Индекс | Советники + Кодер |
| Контекст | **Без ограничений** | 8192 токенов |
| Паттерн | ~98 KB (фиксированный) | Растёт с диалогом |
| Сильная сторона | Весь проект целиком | Точная работа с файлом |

**Пользователь всегда общается с Transformer.** RWKV — внутренний инструмент: запускается автоматически для индексирования и навигации, CLI не показывает. Единая точка входа — `cognit.py`.

**Паттерны** привязаны к git-ветке — переключился на другую ветку, Cognit автоматически использует её паттерны. При `git commit` паттерны изменённых файлов обновляются через post-commit хук.

---

## Пайплайн агентов

Когда пользователь описывает задачу, Cognit запускает последовательный пайплайн с накопленным контекстом:

```
Задача
  │
  ├─ [1] RWKV Navigator    → мемо: «bayes_update() строки 42-67, зависит от validate_input()»
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

Каждая стадия получает **весь контекст предыдущих** — кодер видит файлы + RWKV мемо + мнения всех советников. Порядок и состав стадий задаются в `pipeline.json` клиентского проекта.

```json
{
  "stages": [
    {"id": "navigator", "type": "rwkv",  "enabled": true},
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

**Поправить функцию** — вводишь задачу → RWKV находит файл и пишет где смотреть → советники добавляют контекст → кодер пишет diff → `/patch`.

**Создать новый файл** — RWKV находит похожие файлы как образец → советники задают стиль → кодер создаёт файл через `--- /dev/null` diff → `/patch`.

**Детальный анализ** — `/load auth @src/auth.py` → вопросы напрямую к Transformer, диалог накапливается.

**Ревью по соглашениям** — `/review @file` стекает знание style-агента + файл в один вопрос. Паттерн не меняется — эфемерная сессия.

**База знаний команды** — `agents/` с описанием архитектуры, стиля, контекста хранится в git и загружается при каждой сессии.

---

## Установка и настройка

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Скачать модели (GGUF) и положить в `models/`:
- **Transformer**: `Qwen3-8B-Q4_K_M.gguf` (~4.7 GB VRAM)
- **RWKV**: `RWKV-6-World-7B-Q4_K_M.gguf` (~4.1 GB VRAM) — нужен для пайплайна

> Обе модели не нужны одновременно — RWKV выгружается перед стартом Transformer.

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
  "rwkv": {
    "model_path":   "models/rwkv/RWKV-6-World-7B-Q4_K_M.gguf",
    "n_gpu_layers": -1,
    "n_ctx":        1024,
    "max_tokens":   512,
    "chunk_size":   512
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
   ⚙️  RWKV: навигирую...
   → Файлы: main.py
   → Мемо: bayes_update() строки 42-67, входные параметры float [0,1]

🚀 Пайплайн (5 стадий)
  ✓ [rwkv  ] navigator
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
  ├── Transformer CLI (всегда)
  │     ├── задача без паттерна → handoff to RWKV
  │     ├── /load @file → паттерн → ручной диалог
  │     └── /patch, /edit, /review, /agent...
  │
  └── RWKV headless (автоматически при задаче)
        ├── строит индекс проекта (один раз)
        ├── пишет навигационное мемо (файлы + где смотреть)
        └── возвращает handoff → Transformer запускает пайплайн
```

### Правка кода (/patch и /edit)

**`/patch`** — извлекает unified diff из последнего ответа, показывает и применяет к файлу. Перед записью создаётся `.cognit.bak`. Если diff нет — предыдущий ответ инжектируется как контекст и модель переспрашивается.

**`/edit @file задача`** — читает файл заново (не из KV-cache), передаёт содержимое в модель, просит unified diff. Для точных правок конкретного места.

### Хранение паттернов

```
echo_patterns/
  <repo>/
    <branch>/
      <name>.pkl   ← состояние нейросети (KV-cache или RWKV state)
      <name>.json  ← метаданные: файлы-источники, хэши, дата, диалоги
```

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
- Файловые пути: проверять на path traversal (../)
- Зависимости: проверять CVE перед добавлением

## Типичные уязвимости в этом проекте
- JWT-секрет в .env, не в коде
- CORS разрешён только для перечисленных origins
EOF
```

Содержимое `global.md` — это и есть знание агента. Пиши конкретно: правила, запреты, типичные ошибки, примеры. Чем конкретнее — тем полезнее мемо в пайплайне.

**Шаг 2 — Обучить агента** (создать KV-cache паттерн):

```bash
# Вариант A: перезапустить Cognit — агенты инициализируются автоматически
python cognit.py

# Вариант B: вручную через /load
python cognit.py
🧠> /load security @agents/security/global.md
🧠 [security]> /exit
```

Паттерн сохраняется в `echo_patterns/<repo>/<branch>/security.pkl`.

**Шаг 3 — Добавить в пайплайн** (`pipeline.json` клиентского проекта):

```json
{
  "passes": 1,
  "stages": [
    {"id": "navigator", "type": "rwkv",  "enabled": true},
    {"id": "context",   "type": "agent", "name": "context",  "enabled": true},
    {"id": "arch",      "type": "agent", "name": "arch",     "enabled": true},
    {"id": "security",  "type": "agent", "name": "security", "enabled": true,
     "role": "Задача: {task}\n\nКакие риски безопасности надо учесть? Напиши кратко (2-4 предложения). Не пиши код."},
    {"id": "style",     "type": "agent", "name": "style",    "enabled": true},
    {"id": "coder",     "type": "coder",                     "enabled": true}
  ]
}
```

Поле `role` — инструкция агенту. `{task}` подставляется автоматически. Агент видит всё что написали предыдущие стадии + `role` как вопрос.

**Проверка:**
```
python cognit.py
🧠> /agents
  style    (retrain)  agents/style/global.md
  arch     (retrain)  agents/arch/overview.md
  context  (retrain)  agents/context/project.md
  security (retrain)  agents/security/global.md   ← появился

🧠> поправить авторизацию в src/auth.py
  [RWKV навигирует]
  [agent] context
  [agent] arch
  [agent] security   ← участвует в пайплайне
  [agent] style
  [coder] coder
💡 Diff готов → /patch
```

**Обновить знание агента** — отредактировать `global.md` и переобучить:
```bash
# Изменили agents/security/global.md
python cognit.py
🧠> /load ?security @agents/security/global.md   # ? = принудительный retrain
🧠 [security]> /exit
```

Или просто закоммитить изменение — post-commit хук пересоздаст паттерн автоматически (политика `retrain` для `agents/`).

### Git-интеграция

**post-commit** — при каждом коммите обновляет паттерны изменённых файлов.

**post-checkout** — при смене ветки проверяет наличие RWKV-индекса. Если нет — предупреждает.

```bash
python cognit_transformer.py --refresh-file src/auth.py  # пересоздать паттерн вручную
python cognit_transformer.py --status                    # показать устаревшие паттерны
python cognit_rwkv.py --check-index                      # есть ли RWKV-индекс
python cognit_rwkv.py --build-index src/                 # создать RWKV-индекс headless
```

### Структура проектов

```
my-project/              ← клиентский git (твой код)
  agents/                ← база знаний (в git!)
  pipeline.json          ← порядок стадий пайплайна (в git!)

Cognit/                  ← этот репо (рядом)
  cognit.py              ← единая точка входа
  cognit_transformer.py  ← Transformer-бэкенд (KV-cache, Qwen3)
  cognit_rwkv.py         ← RWKV-бэкенд (навигатор, индекс)
  cognit_pipeline.py     ← пайплайн агентов (конфиг + runner)
  cognit_core.py         ← общие утилиты (git, паттерны, хэши)
  cognit_patch.py        ← применение unified diff
  cognit_agents.py       ← работа с agents/ (чтение, список)
  cognit_setup.py        ← одноразовая настройка
```

---

## Ограничения

**Контекст Transformer ограничен 8192 токенами** — ~300–400 строк кода. Пайплайн обрезает файлы свыше 8000 символов.

**RWKV теряет детали при сжатии** — рекуррентное состояние фиксированного размера (~98 KB). Для большой кодовой базы ранние файлы вытесняются поздними. Для навигации достаточно; для детального анализа — нет.

**Качество пайплайна зависит от agents/** — если `style/global.md` содержит только шаблон, style-агент не добавит ценности. Заполни руководства под свой проект.

**Паттерны не переносимы между машинами** — привязаны к версии модели и llama-cpp-python. S3-синхронизация запланирована.

---

## Шпаргалка

**Задача неизвестна — пайплайн запускается сам:**
```
python cognit.py
🧠> поправь функцию bayes_update   ← просто пишешь задачу
   [RWKV навигирует + пайплайн агентов]
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
python cognit.py --rwkv        # RWKV CLI (ручное построение индекса)

# Паттерны  [Transformer]
/load auth @src/auth.py        # загрузить файл → паттерн 'auth'
/load repo @src/               # загрузить всю папку
/load ?auth @src/auth.py       # принудительный retrain
use auth                       # переключиться на паттерн
/list                          # список паттернов
/exit                          # выход

# Вопросы  [Transformer]
как работает функция login?    # ask — паттерн растёт (если grow)
? как работает функция login?  # peek — инвертирует политику на один вопрос

# Правка кода
/patch                         # diff из последнего ответа → применить
/patch @src/other.py           # применить к конкретному файлу
/edit @src/auth.py убери дублирование  # свежее чтение файла → diff

# Агенты  [ручной режим]
/agents                        # список агентов
/agent style arch              # ambient: каждый вопрос через [auth + style + arch]
/agent off                     # выключить ambient
/review @src/auth.py           # разовое ревью через style-агент
/review arch @src/auth.py      # ревью через конкретный агент

# Pipeline
# Настраивается в pipeline.json клиентского проекта
# Запускается автоматически при вводе задачи без паттерна

# Headless
python cognit_transformer.py --status               # устаревшие паттерны
python cognit_transformer.py --refresh-file src/auth.py  # пересоздать паттерн
python cognit_rwkv.py --check-index                 # есть ли RWKV-индекс
python cognit_rwkv.py --build-index src/            # создать индекс headless
python cognit_setup.py                              # перенастроить
python cognit_setup.py agents                       # создать agents/
```
