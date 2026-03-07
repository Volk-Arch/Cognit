# Cognit — Persistent Neural Context

Локальная LLM с персистентной памятью. Модель один раз читает текст — состояние сохраняется в файл. Все последующие вопросы задаются без повторной передачи контекста.

## Два режима

| | `echo_poc.py` | `echo_rwkv.py` |
|---|---|---|
| Архитектура | Transformer (Qwen3, Phi, ...) | RWKV |
| Хранение памяти | KV-cache | Рекуррентное состояние |
| Лимит контекста | 8192 токенов | **Без ограничений** |
| Точность на деталях | Высокая | Чуть ниже (сжатие) |
| Команда `route` | — | ✅ (маршрутизация файлов) |
| Команда `expand` | ✅ (передать задачу в RWKV) | — |
| Когда использовать | Отдельные файлы, точные вопросы | Большие кодовые базы, глобальный индекс |

**Гибридный рабочий процесс:**
- RWKV → Transformer: `route` определяет нужные файлы → Transformer загружает только их
- Transformer → RWKV: `expand` передаёт задачу в RWKV — за широким контекстом, роутингом или ответами по всей кодовой базе

---

## Сценарии использования

**🗺 Разобраться в чужом проекте**
RWKV читает всю кодовую базу целиком за один проход. Задаёшь вопросы об архитектуре, модулях, потоке данных — без передачи файлов снова.
```
python echo_rwkv.py
🧠> /load repo @src/
🧠 [repo]> как устроена авторизация?
🧠 [repo]> какие файлы отвечают за платежи?
```

**🔍 Детальный анализ / код-ревью файла**
Transformer читает конкретный файл. `?` (peek) — проверяешь без изменения паттерна, обычный вопрос — диалог накапливается.
```
python echo_poc.py
🧠> /load auth @src/auth/middleware.py
🧠 [auth]> ? есть ли уязвимости?          ← паттерн не меняется
🧠 [auth]> как исправить JWT-алгоритм?    ← диалог растёт
```

**✅ Ревью с соглашениями команды**
Загружаешь агент стиля (из `agents/`) и файл — AI знает правила проекта и проверяет код по ним.
```
🧠> /load style @agents/style/
🧠> /load pr    @src/api/routes.py
🧠> use style
🧠 [style]> соответствует ли routes.py нашим соглашениям?
```

**🎯 Найти нужные файлы для задачи**
RWKV определяет, какие файлы затронет задача. Transformer получает только их — точная работа без лишнего контекста.
```
python echo_rwkv.py
🧠 [repo]> route добавить rate limiting
→ авто-handoff: Transformer загружает только 3 нужных файла из 50
```

**📚 База знаний команды**
`agents/` с описанием архитектуры, стиля, контекста — хранится в git, загружается при каждой сессии.
```
🧠> /load arch    @agents/arch/
🧠> /load context @agents/context/
🧠 [arch]> куда добавить новый микросервис?
```

---

## Установка

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

Модели скачать и положить в `models/`:

**Transformer** (`echo_poc.py`):
- `Qwen3-8B-Q4_K_M.gguf` — ~4.7 GB VRAM (рекомендуется)
- `Phi-3.5-mini-instruct-Q4_K_M.gguf` — ~2.5 GB VRAM

**RWKV** (`echo_rwkv.py`), папка `models/rwkv/`:
- `RWKV-6-World-7B-Q4_K_M.gguf` — ~4.1 GB VRAM (рекомендуется)
- `RWKV-6-World-3B-Q4_K_M.gguf` — ~2.0 GB VRAM
- `RWKV-6-World-1B6-Q4_K_M.gguf` — ~1.0 GB VRAM

> Обе модели не нужны одновременно — RWKV выгружается из VRAM перед стартом Transformer.

---

## Настройка (одноразово)

```bash
cd C:/git/Cognit
python echo_setup.py
```

| Вопрос | Ответ |
|---|---|
| Бэкенд | Enter (transformer) |
| Модель | Enter (дефолт) |
| Путь к проекту | `C:/git/DemoAI` |
| Хук в DemoAI | `y` |
| Хук в Cognit | `n` |
| Создать agents/ | `y` |

```
   ✅ client_project сохранён в .echo.json
   ✅ Хук установлен: C:/git/DemoAI/.git/hooks/post-commit
   ✅ Создан: agents/style/global.md
   ✅ Создан: agents/arch/overview.md
   ✅ Создан: agents/context/project.md
✅ Настройка завершена.
```

**Что создалось:**
```
Cognit/.echo.json              ← конфиг (в .gitignore, не в git)
DemoAI/.git/hooks/post-commit  ← авто-обновление паттернов при коммите
DemoAI/agents/                 ← шаблоны базы знаний
```

---

## Быстрый старт: Transformer

```bash
python echo_poc.py
```
```
🧠 Cognit Transformer  |  Загрузка модели...
✅ Готов
📋 Паттернов: 0
🧠>
```

### Загрузить файл и поговорить

```
🧠> /load hello @C:/git/DemoAI/hello.py
   Токенов: 47  →  eval...
✅ hello.pkl  (31 KB)
🧠 [hello]>

🧠 [hello]> что делает этот код?
💬 Загрузка паттерна 'hello'... (файл не передаётся повторно!)
...ответ...
💾 hello.pkl обновлён  (+8 токенов, диалогов: 1)

🧠 [hello]> ? есть ли потенциальные ошибки?
💬 Peek (паттерн не изменится)...
...ответ...
(паттерн не изменился)
```

### Загрузить весь проект

```
🧠> /load project @C:/git/DemoAI/
   Загружаю 5 файлов из C:/git/DemoAI/  (git ls-files)
   (.gitignore уважается автоматически)
✅ project.pkl  (52 KB)
```

### Персистентность — перезапуск

```
🧠 [hello]> /exit
```
```bash
python echo_poc.py
```
```
📋 Паттернов: 3

  • hello    (55 токенов, 31 KB, диалогов: 1)
    Превью:  def hello(): print("Hello, World!")...

  • style    (180 токенов, 30 KB, диалогов: 1)
  • arch     (210 токенов, 35 KB, диалогов: 0)

🧠> use hello
🧠 [hello]> продолжим — что ещё улучшить?
```

Паттерны на месте. Файл не перечитывается — только вопрос.

---

## Быстрый старт: RWKV

```bash
python echo_rwkv.py
```
```
🧠 Cognit RWKV  |  Загрузка модели...  (~4.1 GB VRAM)
✅ Готов
🧠>
```

### Индексация без лимита контекста

```
🧠> /load repo @C:/git/DemoAI/
   # Проект: DemoAI
   Загружаю 5 файлов  (git ls-files)
   Всего токенов: 520
   [██████████████████████████████] 100%  520/520 токенов
✅ repo.pkl  (98 KB)
   Размер фиксирован — не растёт с объёмом текста
```

> **Ключевой момент:** любой объём текста → всегда **98 KB** состояния.
> Для Transformer — каждый токен занимает место в контексте. Для RWKV — нет.

### route → авто-handoff на Transformer

```
🧠 [repo]> route добавить rate limiting для POST /login

──────────────────────────────────────────────────────────
Для этой задачи потребуются:
- src/auth/middleware.py  — rate limit логично поставить до JWT-проверки
- src/api/routes.py       — роут /login определён здесь
- src/db/connection.py    — для Redis-подключения счётчиков
──────────────────────────────────────────────────────────

   Маршрут сохранён → _route_last.json

Передать задачу Transformer? [Y/n]: y
   Выгрузка RWKV из VRAM...  (освобождено ~4.1 GB)
   Запуск echo_poc.py --auto-route...
```

**Transformer запускается автоматически:**
```
🧠 Cognit Transformer  |  Загрузка модели...
✅ Готов

   ⚡ Обнаружен маршрут от RWKV (0 мин. назад):
   Задача: добавить rate limiting для POST /login
   Загружаю автоматически...

✅ middleware.pkl  (47 KB)
✅ routes.pkl      (71 KB)
✅ db_conn.pkl     (31 KB)

🧠 [middleware]>
```

> **RWKV выгружена → Transformer загружена.** Одна видеокарта, два инструмента последовательно.

---

## Команды

Интерфейс одинаков для обоих режимов.

### `use <имя>`

Выбрать паттерн как активный. Все последующие вопросы идут к нему.

### `<вопрос>` — ask

Просто пишешь вопрос — модель отвечает из памяти паттерна. Паттерн **обновляется**, диалог накапливается.

### `? <вопрос>` — peek

Ответ без изменения паттерна. Для разовых проверок к фиксированному снимку.

### `/load <имя> @<путь>`

Загрузить файл или **целую папку** как паттерн. После загрузки паттерн активируется автоматически.

```
/load auth    @src/auth.py          ← файл
/load style   @agents/style/        ← вся папка (.md, .txt, .py, ...)
/load repo    @src/                 ← кодовая база (RWKV, без лимита)
/load project @C:/git/DemoAI/       ← весь проект (git ls-files)
/load note    Важно: не использовать глобальные переменные  ← текст напрямую
```

Директория собирается через `git ls-files` — `.gitignore` уважается автоматически.

### `route <задача>`  *(только echo_rwkv.py)*

Спрашивает RWKV-паттерн, какие файлы нужны для задачи. Предлагает авто-handoff на Transformer.

### `expand <задача>`  *(только echo_poc.py)*

Сохраняет задачу в `_expand_last.json`, выгружает Transformer из VRAM и запускает `echo_rwkv.py --auto-expand`. RWKV стартует и сразу показывает задачу и файлы из контекста Transformer.

### `/list` / `/help` / `/exit`

Список паттернов, справка, выход.

---

## agents/ — база знаний проекта

`agents/` — Markdown-документы, которые «обучают» AI на специфику твоего проекта. Хранятся в git клиентского проекта, загружаются при каждой сессии.

### Инициализация

```bash
python echo_setup.py        # создаёт agents/ вместе с остальной настройкой
python echo_setup.py agents # только agents/, если уже настроено
```

Создаёт шаблоны:

```
agents/
  style/
    global.md        ← именование, форматирование, импорты
    commands.md      ← стиль CLI-команд
  arch/
    overview.md      ← модули, зависимости, поток данных
  context/
    project.md       ← цель проекта, бизнес-правила
```

### Пример заполнения

**`agents/style/global.md`:**
```markdown
# Стиль кода

## Именование
- snake_case для функций и переменных
- Классы: PascalCase
- Константы: UPPER_SNAKE_CASE

## Форматирование
- Отступы: 4 пробела, строки до 100 символов
- f-строки везде, без .format()

## Запрещено
- Глобальные переменные
- print() в продакшне — только logging
- Секреты в коде — только os.getenv()
```

**`agents/arch/overview.md`:**
```markdown
# Архитектура

## Стек
- Python 3.11, FastAPI, PostgreSQL

## Структура
- src/auth/  — JWT-авторизация
- src/api/   — роуты и схемы
- src/db/    — подключение и запросы

## Поток запроса
HTTP → middleware.py → routes.py → queries.py → DB

## Что избегать
- Бизнес-логика в routes.py — только в services/
- Прямые SQL в роутах — только через db/queries.py
```

### Использование

```bash
cd C:/git/DemoAI
git add agents/
git commit -m "Add agent knowledge base"
```
```
🧠> /load style @C:/git/DemoAI/agents/style/
🧠> /load arch  @C:/git/DemoAI/agents/arch/

🧠> use style
🧠 [style]> как именовать новую функцию для обработки токенов?
```

---

## Git-интеграция

### Post-commit хук

После каждого коммита в клиентском проекте хук автоматически обновляет паттерны для изменённых файлов:

```
git commit -m "fix: JWT algorithm check"
   ↓
.git/hooks/post-commit
   ↓
python /abs/path/echo_poc.py --refresh-file src/auth/middleware.py
   ↓
Паттерны, использующие этот файл, пересоздаются
```

```
[main a3f2d1c] fix: JWT algorithm check

[Cognit] Checking patterns for changed files...
```

### Ручные флаги

```bash
python echo_poc.py --refresh-file src/auth.py  # пересоздать паттерн для файла
python echo_poc.py --status                    # проверить актуальность паттернов
```

```
⚠️  Устарел: middleware  (src/auth/middleware.py изменился)
✅ Все паттерны актуальны
```

`--status` в CI:

```yaml
- name: Check Cognit patterns
  run: python echo_poc.py --status || echo "Паттерны устарели"
```

### Автообновление в сессии

Перед каждым вопросом система проверяет — не изменился ли исходник паттерна:

```
⚠️  Файлы изменились с момента создания паттерна 'auth':
    • src/auth.py
   Пересоздать паттерн? [y/N]
```

### Структура путей (git-aware)

```
echo_patterns/
  <repo>/
    <branch>/
      auth.pkl    ← паттерн ветки main
      auth.json
    feature-auth/
      auth.pkl    ← паттерн ветки feature/auth
```

Та же структура — ключ S3 для синхронизации между командой:
```
s3://your-bucket/echo-patterns/<repo>/<branch>/<name>.pkl
```

---

## Структура проектов

**Cognit-репозиторий живёт отдельно от клиентского кода.**

```
my-project/          ← основной гит (твой код)
  src/
  agents/            ← знания о проекте (в git!)
    style/global.md
    arch/overview.md
  .gitignore

Cognit/              ← Cognit-репо (рядом)
  echo_poc.py
  echo_rwkv.py
  echo_setup.py
  echo_core.py
  echo_patterns/     ← нейросетевые состояния (в .gitignore)
  models/            ← GGUF-файлы (в .gitignore)
  .echo.json         ← путь к клиентскому проекту (в .gitignore)
```

`echo_setup.py` спрашивает путь к клиентскому проекту → устанавливает post-commit хук туда с **абсолютным путём** к `echo_poc.py`.

⚠️ При переносе Cognit в другую папку — перезапустить `python echo_setup.py`.

---

## Конфигурация

**echo_poc.py** (Transformer):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MODEL_PATH` | `models/Qwen3-8B-GGUF/...` | Путь к GGUF |
| `N_CTX` | `8192` | Контекстное окно (токены) |
| `N_GPU_LAYERS` | `-1` | Слоёв на GPU (-1 = все) |
| `MAX_TOKENS` | `512` | Максимум токенов в ответе |

**echo_rwkv.py** (RWKV):

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MODEL_PATH` | `models/rwkv/...` | Путь к GGUF |
| `N_CTX` | `1024` | Буфер генерации (не лимит истории) |
| `CHUNK_SIZE` | `512` | Размер чанка при обработке |
| `N_GPU_LAYERS` | `-1` | Слоёв на GPU (-1 = все) |
| `MAX_TOKENS` | `512` | Максимум токенов в ответе |

---

## Шпаргалка

```bash
# Запуск
python echo_poc.py          # Transformer
python echo_rwkv.py         # RWKV (безлимитный контекст)

# Внутри сессии
/load name @path            # загрузить файл или папку
use name                    # переключиться на паттерн
вопрос                      # ask — ответ, паттерн обновляется
? вопрос                    # peek — ответ без изменения паттерна
/list                       # список всех паттернов
/exit                       # выход
route задача                # RWKV: найти файлы → handoff на Transformer
expand задача               # Transformer: передать задачу в RWKV

# Headless
python echo_poc.py --status              # проверить устаревшие паттерны
python echo_poc.py --refresh-file path  # пересоздать паттерн вручную
python echo_setup.py                    # перенастроить (после смены папки)
python echo_setup.py agents             # создать agents/ в клиентском проекте
```
