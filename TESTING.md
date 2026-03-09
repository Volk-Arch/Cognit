# Cognit — Сценарий тестирования

Пошаговый прогон от чистой установки до первых правок кода.

---

## Предварительные требования

- Python 3.11+
- CUDA 12.1+ (для GPU-режима) или CPU (медленно, но работает)
- ~6 GB свободной VRAM для одной модели (модели работают последовательно, не параллельно)
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

**RWKV (~4.1 GB VRAM, опционально — нужен для пайплайна):**
Скачать `RWKV-6-World-7B-Q4_K_M.gguf` с HuggingFace:
```
https://huggingface.co/bartowski/RWKV-x060-World-7B-v2.1-GGUF
```
Положить в `models/rwkv/RWKV-6-World-7B-Q4_K_M.gguf`

> Для базового теста только Transformer достаточно.

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

**Проверка — pipeline.json создан:**
```bash
cat /path/to/my-project/pipeline.json
```
Должно быть 5 стадий: navigator, context, arch, style, coder.

---

## Шаг 6 — Smoke: первый запуск

```bash
python cognit.py
```

Ожидаемый вывод:
```
Cognit · Transformer (Qwen3-8B-Q4_K_M) | GPU | слои: -1 | ctx: 8192

🔄 Авто-инициализация агентов (3 из 3)...
   Загружаю агента: style  (...)
   🃏 Генерирую card.md для style... ✓
   Загружаю агента: arch  (...)
   🃏 Генерирую card.md для arch... ✓
   Загружаю агента: context  (...)
   🃏 Генерирую card.md для context... ✓

📋 Паттернов: 3
  • style  [T]  2026-03-09
  • arch   [T]  2026-03-09
  • context [T] 2026-03-09

🧠>
```

**Что проверяем:**
- Модель загрузилась без ошибок (однострочный баннер без рамки)
- Агенты инициализировались автоматически (паттернов не было — создались)
- `card.md` сгенерированы для каждого агента в `agents/*/card.md`
- `/list` — компактный формат, одна строка на паттерн

---

## Шаг 7 — Загружаем файл и задаём вопрос (ручной режим)

```
🧠> /load main @src/main.py
🧠 [main]> что делает эта функция?
```

**Что проверяем:**
- Паттерн создался: `echo_patterns/<repo>/main/main.pkl`
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

## Шаг 9 — Авто-выбор агентов (keyword)

Спрашиваем что-то про стиль:
```
🧠 [main~]> как правильно назвать эту функцию по конвенциям стиля?
```

Ожидаем:
```
🤖 Авто-агенты: style
[модель отвечает с учётом style-агента]
```

Keyword-matching по словам: `стиль`, `style`, `именован`, `назван`, `конвенц` — подключает style-агента.

---

## Шаг 10 — /patch: правим код

Просим модель исправить что-то:
```
🧠 [main~]> переименуй функцию hello в greet и добавь параметр name: str
```

Если в ответе есть diff-блок — появится подсказка:
```
💡 Есть код в ответе → /patch
```

Пробуем:
```
🧠 [main~]> /patch
```

Ожидаем:
```
Применить патч к src/main.py? [y/N]: y
✅ Патч применён → src/main.py
   Бэкап: src/main.py.cognit.bak
```

Если diff не было в ответе — модель переспрашивается автоматически:
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

## Шаг 10б — /patch для нового файла

Просим создать новый файл:
```
🧠 [main~]> создай src/utils.py с функцией format_name(name: str) -> str
```

Ожидаем diff с `--- /dev/null`:
```diff
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,5 @@
+def format_name(name: str) -> str:
+    ...
```

```
🧠 [main~]> /patch
НОВЫЙ ФАЙЛ: src/utils.py
Создать? [y/N]: y
✅ Файл создан → src/utils.py
```

---

## Шаг 10в — /edit: точная правка файла

`/edit` читает файл заново (не из KV-cache паттерна) и сразу просит diff:

```
🧠 [main~]> /edit @src/main.py добавь docstring к функции greet
```

Ожидаем:
```
📄 /edit src/main.py  →  добавь docstring к функции greet
[модель читает файл и выдаёт unified diff]
💡 Применить? → /patch
```

Затем:
```
🧠 [main~]> /patch
✅ Патч применён → src/main.py
```

**Когда /edit лучше /patch:**
- Нужна точная правка конкретной строки, а не «по памяти» из паттерна
- Паттерн давно создан и файл мог измениться
- Хочешь гарантированно получить diff, а не объяснение

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
📋 Паттернов: 3  (style, arch, context)
🧠> use main
🧠 [main~]> продолжим — что ещё исправить?
```

Файл не перечитывается. Контекст персистентный.

---

## Шаг 15 — Пайплайн (если есть RWKV)

Тест автоматического пайплайна. RWKV запускается headless — пользователь его не видит.

### Сценарий A: задача описана текстом — пайплайн запускается сам

```bash
python cognit.py
🧠> поправить функцию hello — сделать более расширенной
```

Ожидаем:
```
⚙️  RWKV: навигирую...
   → Файлы: src/main.py
   → Мемо: hello() в main.py строки 1-3, нет зависимостей

🚀 Пайплайн (5 стадий)
  ✓ [rwkv  ] navigator
  ✓ [agent ] context
  ✓ [agent ] arch
  ✓ [agent ] style
  ✓ [coder ] coder
  💡 Diff готов → /patch

🧠 [_pipeline]>
```

```
🧠 [_pipeline]> /patch
✅ Патч применён → src/main.py
```

**Что проверяем:**
- RWKV запустился headless и завершился до отображения пайплайна
- Каждая стадия отмечена `✓`
- `_pipeline` паттерн — временный (retrain, будет сброшен при следующем пайплайне)

### Сценарий B: создать новый файл через пайплайн

```bash
python cognit.py
🧠> создать новый файл src/validators.py с функцией validate_email
```

Ожидаем пайплайн, в конце diff с `--- /dev/null`, затем:
```
🧠 [_pipeline]> /patch
НОВЫЙ ФАЙЛ: src/validators.py
Создать? [y/N]: y
✅ Файл создан → src/validators.py
```

### Сценарий C: pipeline.json — отключить стадию

Редактируем `/path/to/my-project/pipeline.json`:
```json
{
  "stages": [
    {"id": "navigator", "type": "rwkv",  "enabled": true},
    {"id": "context",   "type": "agent", "name": "context", "enabled": false},
    {"id": "arch",      "type": "agent", "name": "arch",    "enabled": true},
    {"id": "style",     "type": "agent", "name": "style",   "enabled": true},
    {"id": "coder",     "type": "coder",                    "enabled": true}
  ]
}
```

Запускаем задачу:
```
🧠> поправить что-нибудь
```

В выводе пайплайна: `context` помечен `✗`, остальные `✓`.

---

## Что считать успехом

| Проверка | Критерий |
|---|---|
| Smoke | Однострочный баннер, модель загрузилась без ошибок |
| card.md | Файлы `agents/*/card.md` созданы после первого запуска |
| pipeline.json | Файл создан в клиентском проекте при `cognit_setup.py` |
| grow ~ | `/list` — компактный формат, `~` для feature-паттернов |
| /patch (правка) | `main.py` изменён, `.cognit.bak` создан |
| /patch (новый файл) | `utils.py` создан, `.bak` не создаётся |
| /edit | Файл читается свежим, diff генерируется без обращения к KV-cache |
| /agent | Промпт меняется, `/agent off` возвращает обычный |
| Keyword agents | Вопрос про стиль → `🤖 Авто-агенты: style` без оркестратора |
| Пайплайн headless | RWKV запускается/завершается без CLI → Transformer запускает стадии |
| pipeline.json disable | Отключённая стадия → `✗` в выводе, пропускается |
| Повторный запуск | Вопрос без перезагрузки файла работает |
| post-commit хук | Grow-паттерн пропускается, retrain — пересоздаётся |
| Сброс | Удалить `echo_patterns/` → первый пайплайн пересоздаёт всё |
