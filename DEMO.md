# Echo PoC — Полный демо-сценарий

Демонстрирует полный цикл: настройка проекта → базы знаний → индексация RWKV → маршрутизация → авто-смена на Transformer → код-ревью → персистентность.

---

## Предварительные условия

```
models/
  Qwen3-8B-GGUF/
    Qwen3-8B-Q4_K_M.gguf     ← ~4.7 GB
  rwkv/
    RWKV-6-World-7B-Q4_K_M.gguf  ← ~4.1 GB
```

> Обе модели не нужны одновременно — RWKV выгружается перед стартом Transformer.

---

## Шаг 0 — Структура проекта

Демо работает на условном проекте `my-api`:

```
my-api/
  src/
    auth/
      middleware.py    ← JWT-мидлвар
      utils.py         ← хэшер паролей
    api/
      routes.py        ← FastAPI-роуты
      models.py        ← Pydantic-схемы
    db/
      connection.py    ← подключение к БД
      queries.py       ← SQL-запросы
  tests/
    test_auth.py
```

Echo-файлы лежат **рядом** с проектом или в отдельном `my-api-ai/` — не важно для демо.

---

## Шаг 1 — Настройка (одноразово)

```bash
python echo_setup.py
```

Интерактивный вывод:

```
╔══════════════════════════════════════════════╗
║  🧠 Echo PoC — Настройка проекта            ║
╚══════════════════════════════════════════════╝

   Git-репозиторий: my-api  (C:/dev/my-api)

── Конфигурация проекта (.echo.json) ──────────────────
   Бэкенд (transformer / rwkv) [transformer]: transformer
   Имя модели (без пути и расширения) [Qwen3-8B-Q4_K_M]: Qwen3-8B-Q4_K_M
   ✅ Создан .echo.json

── .gitignore ──────────────────────────────────────────
   ✅ Добавлено в .gitignore: echo_patterns/

── post-commit хук ─────────────────────────────────────
Установить post-commit хук? [Y/n]: y
   ✅ Хук установлен: .git/hooks/post-commit

── agents/ — знания о проекте ──────────────────────────
Создать папку agents/ с шаблонами? [Y/n]: y
   ✅ Создан: agents/style/global.md
   ✅ Создан: agents/style/commands.md
   ✅ Создан: agents/arch/overview.md
   ✅ Создан: agents/context/project.md

✅ Настройка завершена.
```

**Что создано:**

```
.echo.json              ← модель зафиксирована (для всей команды)
.gitignore              ← echo_patterns/ исключены из git
.git/hooks/post-commit  ← авто-обновление паттернов при коммите
agents/                 ← шаблоны базы знаний (пустые)
```

---

## Шаг 2 — База знаний агентов

Заполняем `agents/` информацией о проекте. Это делается один раз — потом хранится в git.

### `agents/style/global.md`

```markdown
# Глобальный стиль кода

## Именование
- Переменные и функции: snake_case
- Классы: PascalCase
- Константы: UPPER_SNAKE_CASE

## Форматирование
- Отступы: 4 пробела
- Максимальная длина строки: 100 символов
- f-строки везде где возможно, без .format()

## Паттерны
- Не использовать глобальные переменные
- Зависимости через FastAPI Depends(), не хардкодить
- Все ошибки через HTTPException с понятным detail

## Тесты
- pytest, fixtures в conftest.py
- Мокать внешние зависимости через pytest-mock
```

### `agents/arch/overview.md`

```markdown
# Архитектура my-api

## Стек
- Python 3.11, FastAPI, PostgreSQL, Redis

## Структура
- src/auth/  — JWT-авторизация (middleware + utils)
- src/api/   — роуты и Pydantic-схемы
- src/db/    — подключение и SQL-запросы

## Поток запроса
HTTP → middleware.py (JWT verify) → routes.py → queries.py → DB

## Что избегать
- Бизнес-логику в routes.py — только в services/
- Прямые SQL в роутах — только через db/queries.py
- Хранить секреты в коде — только через os.getenv()
```

Фиксируем в git:

```bash
git add agents/
git commit -m "Add agent knowledge base"
```

---

## Шаг 3 — RWKV: индексация кодовой базы

RWKV читает весь `src/` без ограничений контекста.

```bash
python echo_rwkv.py
```

```
🧠 Echo RWKV  |  Загрузка модели...
   Модель: RWKV-6-World-7B-Q4_K_M.gguf  (4.1 GB VRAM)
   ✅ Готов

📋 Паттернов: 0

🧠> /load repo @src/
```

```
   Загрузка директории: src/
   Найдено файлов: 6 (.py)
   Всего ~3 800 токенов

   Обработка чанками (CHUNK_SIZE=512):
   [██████████████████████████████] 100%  3800/3800 токенов

✅ Паттерн сохранён: echo_patterns/my-api/main/repo.pkl
   Размер: 98 KB  (фиксирован, не растёт с объёмом текста)
   Диалогов: 0

🧠 [repo]> _
```

> **Ключевой момент:** 3 800 токенов уместились в **98 KB** фиксированного состояния.
> Для Transformer это был бы весь контекст; для RWKV — просто обновление скрытого состояния.

---

## Шаг 4 — RWKV: база знаний агентов

Загружаем `agents/` чтобы RWKV знал стиль и архитектуру:

```
🧠 [repo]> /load style @agents/style/
```

```
   Загрузка директории: agents/style/
   Найдено файлов: 2 (.md)
   Всего ~420 токенов

   Обработка...
✅ Паттерн сохранён: echo_patterns/my-api/main/style.pkl

🧠 [style]> use repo
🧠 [repo]> _
```

Переключаемся обратно на `repo` — в нём уже весь код.

---

## Шаг 5 — RWKV: маршрутизация задачи

Задача: добавить rate limiting в API.

```
🧠 [repo]> route добавить rate limiting для POST /login
```

```
   🗺️  Маршрутизация задачи: добавить rate limiting для POST /login
   Спрашиваю RWKV...

──────────────────────────────────────────────────────────
Для добавления rate limiting к POST /login потребуются:

- src/auth/middleware.py   — здесь проходит авторизация, rate limit логично поставить до JWT-проверки
- src/api/routes.py        — роут /login определён здесь, нужно добавить Depends()
- src/db/connection.py     — для Redis-подключения (хранение счётчиков)
──────────────────────────────────────────────────────────

   Команды для echo_poc.py:
   /load middleware @src/auth/middleware.py
   /load routes @src/api/routes.py
   /load db_conn @src/db/connection.py

   Маршрут сохранён → _route_last.json

Передать задачу Transformer? [Y/n]: y
```

```
   Выгрузка RWKV из VRAM...  (освобождено ~4.1 GB)
   Запуск echo_poc.py --auto-route...
```

---

## Шаг 6 — Transformer: авто-загрузка

`echo_poc.py` запускается автоматически:

```
🧠 Echo Transformer  |  Загрузка модели...
   Модель: Qwen3-8B-Q4_K_M.gguf  (4.7 GB VRAM)
   ✅ Готов

📋 Паттернов: 0

   ⚡ Обнаружен маршрут от RWKV (1 мин. назад):
   Задача: добавить rate limiting для POST /login
   Файлы:
     /load middleware @src/auth/middleware.py
     /load routes @src/api/routes.py
     /load db_conn @src/db/connection.py

   Загружаю автоматически...
```

```
   Загрузка 'middleware' из src/auth/middleware.py...
   Токенов: 312  →  eval...
✅ middleware.pkl  (47 KB)

   Загрузка 'routes' из src/api/routes.py...
   Токенов: 528  →  eval...
✅ routes.pkl  (71 KB)

   Загрузка 'db_conn' из src/db/connection.py...
   Токенов: 187  →  eval...
✅ db_conn.pkl  (31 KB)

   Активный паттерн: middleware
🧠 [middleware]> _
```

> **RWKV выгружена → Transformer загружена.** Оба инструмента работали на одной видеокарте последовательно.

---

## Шаг 7 — Transformer: код-ревью

### 7.1 — Изучение middleware

```
🧠 [middleware]> что делает текущий middleware и как он устроен?
```

```
💬 Загрузка паттерна 'middleware'... (не передаётся повторно!)
   Токенов вопроса: 14

──────────────────────────────────────────────────────────
JWTMiddleware перехватывает каждый запрос до роутов.
Извлекает токен из заголовка Authorization: Bearer <token>,
декодирует через python-jose, кладёт user_id в request.state.

Публичные пути (/docs, /health, /login) пропускаются без проверки.
──────────────────────────────────────────────────────────
💾 middleware.pkl обновлён  (+14 токенов, диалогов: 1)
```

### 7.2 — Безопасный peek без изменения паттерна

```
🧠 [middleware]> ? есть ли уязвимости в текущей реализации?
```

```
💬 Peek (паттерн не изменится)...

──────────────────────────────────────────────────────────
Замечено:
1. jwt.decode() не проверяет алгоритм — уязвимость "alg:none"
   Нужно: algorithms=["HS256"] явно
2. Токен не валидируется на отзыв (нет проверки в Redis/БД)
3. Исключения jose.JWTError не логируются — сложно отлаживать
──────────────────────────────────────────────────────────
(паттерн не изменился)
```

> `?` использован для "одноразового" вопроса — паттерн остаётся чистым снимком кода.

### 7.3 — Переключение на роуты

```
🧠 [middleware]> use routes
🧠 [routes]> как реализован роут /login?
```

```
──────────────────────────────────────────────────────────
POST /login принимает LoginRequest (username, password).
Вызывает authenticate_user() из auth.utils, возвращает JWT.
Никакого rate limiting нет — прямой вызов без ограничений.
──────────────────────────────────────────────────────────
```

### 7.4 — Проектирование решения

```
🧠 [routes]> покажи как добавить rate limiting через slowapi без изменения бизнес-логики
```

```
──────────────────────────────────────────────────────────
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, body: LoginRequest):
    # существующая логика без изменений
    ...

Limiter подключается через app.state.limiter = limiter
и обработчик _rate_limit_exceeded_handler.
──────────────────────────────────────────────────────────
💾 routes.pkl обновлён  (диалогов: 2)
```

### 7.5 — Проверка стиля через агент

```
🧠 [routes]> use style
```

Паттерн `style` не создан в Transformer — загружаем:

```
🧠 [style]> /load style @agents/style/
```

```
✅ style.pkl  (28 KB)
🧠 [style]> _
```

```
🧠 [style]> соответствует ли код в routes.py нашим соглашениям?
```

```
──────────────────────────────────────────────────────────
По style/global.md:
✅ snake_case соблюдён
✅ Зависимости через Depends()
⚠️  В одном месте использован .format() вместо f-строки (строка 47)
⚠️  HTTPException в route /login не указывает detail — нарушает соглашение
──────────────────────────────────────────────────────────
```

---

## Шаг 8 — Персистентность: перезапуск

Выходим и перезапускаем:

```
🧠 [style]> /exit
```

```bash
python echo_poc.py
```

```
🧠 Echo Transformer  |  Загрузка модели...
   ✅ Готов

📋 Паттернов: 4

  • db_conn    [TRANSFORMER]  (187 токенов, 31 KB)
    Сохранён: 2024-01-15 14:23  |  Диалогов: 0

  • middleware  [TRANSFORMER]  (326 токенов, 47 KB)
    Сохранён: 2024-01-15 14:24  |  Диалогов: 1
    Превью:   JWTMiddleware перехватывает каждый запрос до роутов...

  • routes      [TRANSFORMER]  (541 токенов, 71 KB)
    Сохранён: 2024-01-15 14:26  |  Диалогов: 2
    Превью:   POST /login принимает LoginRequest (username, password)...

  • style       [TRANSFORMER]  (420 токенов, 28 KB)
    Сохранён: 2024-01-15 14:31  |  Диалогов: 1
    Превью:   Глобальный стиль кода: snake_case, 4 пробела...

🧠> use middleware
🧠 [middleware]> продолжим — что ещё нужно исправить?
```

Диалог продолжается. Контекст не передавался повторно.

---

## Шаг 9 — Git-интеграция: авто-обновление

Вносим изменения в middleware и коммитим:

```bash
# Правим уязвимость с алгоритмом
git add src/auth/middleware.py
git commit -m "fix: JWT algorithm check"
```

```
[main a3f2d1c] fix: JWT algorithm check
 1 file changed, 2 insertions(+), 1 deletion(-)

🧠 Echo: проверяю паттерны для изменённых файлов...
⚠️  Паттерн 'middleware' устарел (src/auth/middleware.py изменился)
   Пересоздаю...
✅ middleware.pkl обновлён  (329 токенов, 48 KB)
```

> Хук запустил `echo_poc.py --refresh-file src/auth/middleware.py` автоматически.
> Следующая сессия уже работает с актуальным кодом.

---

## Итог: что показали

| Этап | Инструмент | Результат |
|------|-----------|-----------|
| Настройка | `echo_setup.py` | `.echo.json`, `agents/`, git-хук |
| База знаний | `agents/*.md` → git | Стиль и архитектура в версии |
| Индексация 3 800 токенов | RWKV | 98 KB фиксированного состояния |
| Маршрутизация задачи | `route` (RWKV) | 3 нужных файла из 6 |
| Смена модели | авто-handoff | RWKV → Transformer без перезапуска вручную |
| Код-ревью (ask) | Transformer | Диалог накапливается в паттерне |
| Безопасный вопрос (peek) | `?` prefix | Паттерн не меняется |
| Проверка стиля | `style` паттерн | Агент знает соглашения проекта |
| Персистентность | перезапуск | 4 паттерна восстановлены, диалоги сохранены |
| Авто-обновление | post-commit хук | Паттерн пересоздан при изменении файла |

---

## Быстрая шпаргалка по командам

```
# Первый раз
python echo_setup.py              ← настройка проекта

# RWKV-сессия (большой контекст)
python echo_rwkv.py
/load repo @src/                  ← вся кодовая база
route добавить авторизацию        ← маршрутизация → авто-смена на Transformer

# Transformer-сессия (точная работа)
python echo_poc.py
/load auth @src/auth.py           ← конкретный файл
use auth                          ← выбрать активный паттерн
что делает login()?               ← ask (паттерн растёт)
? есть ли уязвимости?             ← peek (паттерн не меняется)
use style                         ← переключиться на паттерн стиля
/list                             ← показать все паттерны
```
