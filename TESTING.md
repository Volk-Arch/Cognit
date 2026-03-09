# Cognit — Сценарий тестирования

Пошаговый прогон от чистой установки до первых правок кода.

---

## Предварительные требования

- Python 3.11+
- CUDA 12.1+ (для GPU-режима) или CPU (медленно, но работает)
- ~6 GB свободной VRAM для одной модели
- git

---

## Шаг 1 — Клонируем репо

```bash
git clone https://github.com/kriusov/Cognit.git
cd Cognit
```

---

## Шаг 2 — Устанавливаем зависимости

**GPU (CUDA 12.1):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**CPU (без GPU):**
```bash
pip install llama-cpp-python
```

Проверка:
```bash
python -c "from llama_cpp import Llama; print('OK')"
```

---

## Шаг 3 — Скачиваем модели

Создаём папку и скачиваем хотя бы одну модель:

```bash
mkdir -p models/Qwen3-8B-GGUF
mkdir -p models/rwkv
```

**Transformer (Qwen3-8B, ~4.7 GB VRAM):**
Скачать `Qwen3-8B-Q4_K_M.gguf` с HuggingFace:
```
https://huggingface.co/Qwen/Qwen3-8B-GGUF
```
Положить в `models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`

**RWKV (~4.1 GB VRAM, опционально — нужен для route/expand):**
Скачать `RWKV-6-World-7B-Q4_K_M.gguf` с HuggingFace:
```
https://huggingface.co/bartowski/RWKV-x060-World-7B-v2.1-GGUF
```
Положить в `models/rwkv/RWKV-6-World-7B-Q4_K_M.gguf`

> Для теста только Transformer достаточно.

---

## Шаг 4 — Подготавливаем клиентский проект

Нужен любой git-репо с кодом. Можно создать тестовый:

```bash
mkdir -p /path/to/my-project/src
cd /path/to/my-project
git init
echo "def hello():\n    return 1" > src/main.py
git add . && git commit -m "init"
cd -  # вернуться в Cognit/
```

---

## Шаг 5 — Настраиваем Cognit

```bash
python cognit_setup.py
```

Отвечаем на вопросы:
```
Путь к клиентскому проекту: /path/to/my-project
Путь к Transformer модели [models/Qwen3-8B-GGUF/...]: <Enter>
GPU layers [-1]: <Enter>
Путь к RWKV модели [models/rwkv/...]: <Enter>
Установить post-commit хук? [Y/n]: y
```

**Проверка — `.echo.json` создан:**
```bash
cat .echo.json
```
Должно быть:
```json
{
  "backend": "transformer",
  "transformer": { "model_path": "models/...", ... },
  "rwkv": { ... },
  "client_project": "/path/to/my-project"
}
```

**Проверка — шаблоны агентов созданы:**
```bash
ls /path/to/my-project/agents/
# style/  arch/  context/
cat /path/to/my-project/agents/style/global.md
# Должен быть реальный Python style guide, не заглушки
```

---

## Шаг 6 — Smoke: первый запуск

```bash
python cognit.py
```

Ожидаемый вывод:
```
╔══════════════════════════════════════════════╗
║  Cognit · Transformer (Qwen3-8B-Q4_K_M)     ║
╚══════════════════════════════════════════════╝

Модель загружена | GPU | слои: -1 | ctx: 8192

🔄 Авто-инициализация агентов (3 из 3)...
   Загружаю агента: style  (...)
   🃏 Генерирую card.md для style... ✓
   Загружаю агента: arch  (...)
   🃏 Генерирую card.md для arch... ✓
   Загружаю агента: context  (...)
   🃏 Генерирую card.md для context... ✓

   🎯 Создаю оркестратор из 3 card.md...

📋 Паттернов: 4  (style, arch, context, orchestrator)

🧠>
```

**Что проверяем:**
- Модель загрузилась без ошибок
- Агенты инициализировались автоматически (паттернов не было — создались)
- card.md сгенерированы для каждого агента
- Оркестратор создан из card.md

---

## Шаг 7 — Загружаем файл и задаём вопрос

```
🧠> /load main @src/main.py
🧠 [main]> что делает эта функция?
```

**Что проверяем:**
- Паттерн создался: появилось `echo_patterns/<repo>/main/main.pkl`
- Модель ответила на вопрос о `hello()`
- Промпт переключился на `🧠 [main]>`

---

## Шаг 8 — grow_policy

```
🧠 [main]> есть ли способ сделать это лучше?
```

Должно появиться `[main~]` — паттерн растёт:

Проверяем в `/list`:
```
🧠 [main~]> /list
  main~   (grow)  tokens: ...
  style   (retrain)
  arch    (retrain)
  context (retrain)
```

---

## Шаг 9 — Авто-выбор агентов

Спрашиваем что-то про стиль:
```
🧠 [main~]> как правильно назвать эту функцию по конвенциям стиля?
```

Ожидаем:
```
   🎯 Оркестратор... style
🤖 Авто-агенты: style
[модель отвечает с учётом style-агента]
```

Если оркестратора нет — должно быть keyword-fallback:
```
🤖 Авто-агенты: style
```

---

## Шаг 10 — /patch: правим код

Просим модель исправить что-то:
```
🧠 [main~]> переименуй функцию hello в greet и добавь параметр name: str
```

Если в ответе есть diff-блок — сразу пробуем:
```
🧠 [main~]> /patch
```

Ожидаем:
```
Применить патч к src/main.py? [y/N]: y
✅ Патч применён → src/main.py
   Бэкап: src/main.py.cognit.bak
```

Если diff не было в ответе:
```
⚠️  Diff не найден в ответе. Запрашиваю у модели...
[модель генерирует diff]
Применить патч к src/main.py? [y/N]: y
```

**Проверка:**
```bash
cat /path/to/my-project/src/main.py
# Должна быть функция greet(name: str)
cat /path/to/my-project/src/main.py.cognit.bak
# Исходный файл
```

---

## Шаг 11 — /agent ambient

```
🧠 [main~]> /agent style arch
🔗 Ambient агенты: style + arch
   Каждый вопрос будет проходить через [main + style + arch]

🧠 [main~ + style + arch]> добавь валидацию входного параметра
```

Промпт изменился, агенты учитываются при каждом ответе.

Выключить:
```
🧠 [main~ + style + arch]> /agent off
🔕 Ambient агенты отключены
🧠 [main~]>
```

---

## Шаг 12 — /review эфемерно

```
🧠 [main~]> /review @src/main.py
```

Одноразовая проверка файла через style-агент. Паттерн `main~` не должен измениться после этого.

---

## Шаг 13 — git-хук (если установлен)

```bash
cd /path/to/my-project
echo "# comment" >> src/main.py
git add src/main.py
git commit -m "add comment"
```

Должен запуститься post-commit хук:
```
[Cognit] Обновляю паттерны...
   ~ main: grow-паттерн, пропускаем (накапливает диалог)
[Cognit] Готово.
```

`main` имеет политику `grow` (feature-ветка) — пропускается. Если создать `retrain`-паттерн, он пересоздастся.

---

## Шаг 14 — повторный запуск

```bash
# Ctrl+C или /exit
python cognit.py
```

Паттерны уже есть — загрузка быстрая:
```
📋 Паттернов: 4  (style, arch, context, orchestrator)
🧠> use main
🧠 [main~]> продолжим — что ещё исправить?
```

Файл не перечитывается. Контекст персистентный.

---

## Шаг 15 — route + expand (если есть RWKV)

Создаём RWKV-индекс:
```bash
python cognit.py --rwkv
🧠> /load repo @src/
```

Затем в Transformer:
```bash
python cognit.py
🧠> route добавить логирование в main.py
```

Ожидаем автоматическое переключение: Transformer → RWKV (находит файлы) → Transformer (загружает их).

---

## Типичные сценарии (краткая справка)

### Сценарий A: знаешь файл → сразу Transformer

```
python cognit.py

🧠> /load auth @src/auth/middleware.py
🧠 [auth]> есть ли уязвимости в JWT-проверке?
🧠 [auth]> исправь алгоритм
🧠 [auth]> /patch
```

`/patch` — если в ответе нет diff-блока, автоматически запрашивает его у модели, показывает изменения и применяет с созданием `.cognit.bak` бэкапа.

### Сценарий B: не знаешь файл → route прямо из Transformer

Если RWKV-индекс уже создан (`/load repo @src/` в RWKV-режиме):

```
python cognit.py

🧠> route добавить rate limiting для POST /login

   Найден RWKV-индекс: repo
⚙️  Переключение: Transformer → RWKV  (авто-маршрут: добавить rate limiting...)
⚡ Авто-маршрут: «добавить rate limiting для POST /login»
   Индекс: repo

→ Файлы: middleware.py, routes.py, db/connection.py

⚙️  Переключение: RWKV → Transformer  (файлов: 3)

🧠 [middleware]> как сейчас реализовано ограничение запросов?
```

Создать индекс (один раз): `python cognit.py --rwkv` → `/load repo @src/`

### Сценарий C: начал в Transformer, понял что мало контекста → expand

```
python cognit.py

🧠> /load auth @src/auth/middleware.py
🧠 [auth]> как это соотносится с остальной авторизацией?
🧠 [auth]> expand нужен контекст по всей авторизации

⚙️  Переключение: Transformer → RWKV
🧠> use repo
🧠 [repo]> расскажи об архитектуре авторизации в проекте
```

### Сценарий D: персистентность между сессиями

```bash
# Завершили сессию, перезапустили
python cognit.py
📋 Паттернов: 3
🧠> use auth
🧠 [auth]> продолжим — что ещё проверить?
```

Файл не перечитывается — только вопрос.

---

## Что считать успехом

| Проверка | Критерий |
|---|---|
| Smoke | Модель загрузилась, паттерны авто-созданы |
| card.md | Файлы `agents/*/card.md` созданы после первого запуска |
| Оркестратор | `echo_patterns/.../orchestrator.pkl` существует |
| grow ~ | `/list` показывает `~` для feature-паттернов |
| /patch | `main.py` изменён, `.cognit.bak` создан |
| /agent | Промпт меняется, `/agent off` возвращает обычный |
| Повторный запуск | Вопрос без перезагрузки файла работает |
| post-commit хук | Grow-паттерн пропускается, retrain — пересоздаётся |
