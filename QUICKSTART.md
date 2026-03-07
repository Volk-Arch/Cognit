# Echo PoC — Quickstart & Demo

Полный обход функционала: команды + ожидаемый вывод.

---

## Требования

```
models/
  Qwen3-8B-GGUF/
    Qwen3-8B-Q4_K_M.gguf        ← ~4.7 GB VRAM (Transformer)
  rwkv/
    RWKV-6-World-7B-Q4_K_M.gguf ← ~4.1 GB VRAM (RWKV)
```

> Обе модели не нужны одновременно — RWKV выгружается перед стартом Transformer.

```
C:/git/Cognit/   ← Echo (отсюда всё запускаем)
C:/git/DemoAI/   ← Твой проект (туда ставим хук, там agents/)
```

---

## 1. Настройка (одноразово)

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

## 2. Агенты — база знаний проекта

Заполняем один раз, потом хранится в git DemoAI.

**`DemoAI/agents/style/global.md`:**
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

**`DemoAI/agents/arch/overview.md`:**
```markdown
# Архитектура DemoAI

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

Коммитим в DemoAI:
```bash
cd C:/git/DemoAI
git add agents/
git commit -m "Add agent knowledge base"
```

---

## 3. Transformer — загрузить файл и поговорить

```bash
cd C:/git/Cognit
python echo_poc.py
```

```
🧠 Echo Transformer  |  Загрузка модели...
✅ Готов
📋 Паттернов: 0
🧠>
```

### Загрузить файл
```
🧠> /load hello @C:/git/DemoAI/hello.py
```
```
   Токенов: 47  →  eval...
✅ hello.pkl  (31 KB)
🧠 [hello]>
```

### Ask — паттерн растёт
```
🧠 [hello]> что делает этот код?
```
```
💬 Загрузка паттерна 'hello'... (файл не передаётся повторно!)
...ответ...
💾 hello.pkl обновлён  (+8 токенов, диалогов: 1)
```

### Peek — снимок без изменений
```
🧠 [hello]> ? есть ли потенциальные ошибки?
```
```
💬 Peek (паттерн не изменится)...
...ответ...
(паттерн не изменился)
```

### Загрузить агентов
```
🧠> /load style @C:/git/DemoAI/agents/style/
🧠> /load arch  @C:/git/DemoAI/agents/arch/
```

```
🧠> use style
🧠 [style]> как именовать новую функцию для обработки токенов?

🧠 [style]> use arch
🧠 [arch]> куда добавить новый эндпоинт?
```

### Загрузить весь проект
```
🧠> /load project @C:/git/DemoAI/
```
```
   Загружаю 5 файлов из C:/git/DemoAI/  (git ls-files)
   (.gitignore уважается автоматически)
✅ project.pkl  (52 KB)
```

---

## 4. Персистентность — перезапуск

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

## 5. RWKV — индексация без лимита контекста

```bash
python echo_rwkv.py
```
```
🧠 Echo RWKV  |  Загрузка модели...  (~4.1 GB VRAM)
✅ Готов
🧠>
```

```
🧠> /load repo @C:/git/DemoAI/
```
```
   # Проект: DemoAI
   Загружаю 5 файлов  (git ls-files)
   Всего токенов: 520
   [██████████████████████████████] 100%  520/520 токенов
✅ repo.pkl  (98 KB)
   Размер фиксирован — не растёт с объёмом текста
```

> **Ключевой момент:** любой объём текста → всегда **98 KB** состояния.
> Для Transformer — каждый токен занимает место в контексте. Для RWKV — нет.

```
🧠 [repo]> как устроен проект?
🧠 [repo]> ? какие файлы отвечают за авторизацию?
```

---

## 6. RWKV route → авто-смена на Transformer

```
🧠 [repo]> route добавить rate limiting для POST /login
```
```
   🗺️  Маршрутизация: добавить rate limiting для POST /login

──────────────────────────────────────────────────────────
Для этой задачи потребуются:
- src/auth/middleware.py  — rate limit логично поставить до JWT-проверки
- src/api/routes.py       — роут /login определён здесь
- src/db/connection.py    — для Redis-подключения счётчиков
──────────────────────────────────────────────────────────

   Команды для echo_poc.py:
   /load middleware @src/auth/middleware.py
   /load routes     @src/api/routes.py
   /load db_conn    @src/db/connection.py

   Маршрут сохранён → _route_last.json

Передать задачу Transformer? [Y/n]: y
```
```
   Выгрузка RWKV из VRAM...  (освобождено ~4.1 GB)
   Запуск echo_poc.py --auto-route...
```

**Transformer запускается автоматически:**
```
🧠 Echo Transformer  |  Загрузка модели...
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

## 7. Код-ревью в Transformer

### Изучить middleware
```
🧠 [middleware]> что делает текущий middleware и как он устроен?
```
```
💬 Загрузка паттерна 'middleware'... (не передаётся повторно!)

──────────────────────────────────────────────────────────
JWTMiddleware перехватывает каждый запрос до роутов.
Извлекает токен из Authorization: Bearer <token>,
декодирует через python-jose, кладёт user_id в request.state.
Публичные пути (/docs, /health, /login) пропускаются.
──────────────────────────────────────────────────────────
💾 middleware.pkl обновлён  (диалогов: 1)
```

### Peek — проверить уязвимости без сохранения
```
🧠 [middleware]> ? есть ли уязвимости в текущей реализации?
```
```
💬 Peek (паттерн не изменится)...

──────────────────────────────────────────────────────────
1. jwt.decode() не проверяет алгоритм — уязвимость "alg:none"
   Нужно: algorithms=["HS256"] явно
2. Токен не валидируется на отзыв (нет проверки в Redis)
3. JWTError не логируется — сложно отлаживать
──────────────────────────────────────────────────────────
(паттерн не изменился)
```

### Переключиться и спроектировать решение
```
🧠 [middleware]> use routes
🧠 [routes]> покажи как добавить rate limiting через slowapi
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
──────────────────────────────────────────────────────────
💾 routes.pkl обновлён  (диалогов: 2)
```

### Проверить стиль через агент
```
🧠 [routes]> use style
🧠 [style]> соответствует ли код в routes.py нашим соглашениям?
```
```
──────────────────────────────────────────────────────────
✅ snake_case соблюдён
✅ Зависимости через Depends()
⚠️  В одном месте использован .format() вместо f-строки (строка 47)
⚠️  HTTPException без detail — нарушает соглашение
──────────────────────────────────────────────────────────
```

---

## 8. Git-хук — авто-обновление паттернов

```bash
cd C:/git/DemoAI
# отредактируй файл
git add src/auth/middleware.py
git commit -m "fix: JWT algorithm check"
```
```
[main a3f2d1c] fix: JWT algorithm check

[Echo] Checking patterns for changed files...
```

Паттерн `middleware` пересоздаётся автоматически.

**Проверить статус вручную:**
```bash
cd C:/git/Cognit
python echo_poc.py --status
```
```
⚠️  Устарел: middleware  (src/auth/middleware.py изменился)
```
или
```
✅ Все паттерны актуальны
```

---

## Итог: что показали

| Этап | Инструмент | Результат |
|---|---|---|
| Настройка | `echo_setup.py` | `.echo.json`, `agents/`, git-хук |
| База знаний | `agents/*.md` → git DemoAI | Стиль и архитектура в версии |
| Загрузка файла | `/load` + eval | Нейросеть "прочитала" код один раз |
| Ask / Peek | вопрос / `? вопрос` | Диалог накапливается или не меняется |
| Персистентность | перезапуск | Паттерны восстанавливаются с диска |
| Индексация | RWKV | Любой объём → 98 KB фиксированного состояния |
| Маршрутизация | `route` (RWKV) | 3 нужных файла из 6 определено автоматически |
| Смена модели | авто-handoff | RWKV → Transformer на одной видеокарте |
| Код-ревью | Transformer | Диалог накапливается, peek не меняет паттерн |
| Проверка стиля | `style` паттерн | Агент знает соглашения проекта |
| Авто-обновление | post-commit хук | Паттерн пересоздан при изменении файла |

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
route задача                # RWKV: найти файлы → handoff

# Headless
python echo_poc.py --status              # проверить устаревшие паттерны
python echo_poc.py --refresh-file path  # пересоздать паттерн вручную
python echo_setup.py                    # перенастроить (после смены папки)
```
