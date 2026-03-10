# Cognit — Сценарий тестирования

Пошаговый прогон от чистой установки до первых правок кода.

---

## Предварительные требования

- Python 3.11+
- CUDA 12.1+ (для GPU-режима) или CPU (медленно, но работает)
- ~5 GB свободной VRAM для модели
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
pip install tree-sitter tree-sitter-python
```

**CPU (без GPU):**
```bash
pip install llama-cpp-python
pip install tree-sitter tree-sitter-python
```

Проверка:
```bash
python -c "from llama_cpp import Llama; print('OK')"
python -c "from cognit_index import CodeIndex; print('OK')"
```

---

## Шаг 3 — Скачиваем модель

Создаём папку и скачиваем модель:

```bash
mkdir -p models/Qwen3-8B-GGUF
```

**Transformer (Qwen3-8B, ~4.7 GB VRAM):**
Скачать `Qwen3-8B-Q4_K_M.gguf` с HuggingFace:
```
https://huggingface.co/Qwen/Qwen3-8B-GGUF
```
Положить в `models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`

---

## Шаг 4 — Подготавливаем клиентский проект

Нужен любой git-репо с Python-кодом. Можно создать тестовый:

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
  "client_project": "/path/to/my-project"
}
```

**Проверка — шаблоны агентов созданы:**
```bash
ls /path/to/my-project/agents/
# style/  arch/  context/
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
🧠 Transformer · Qwen3-8B-Q4_K_M · my-project/main

🔄 Авто-инициализация агентов (3 из 3)...
   Загружаю агента: style  (...)
   Загружаю агента: arch  (...)
   Загружаю агента: context  (...)

📋 Паттернов: 3
  • style  [T]  2026-03-10
  • arch   [T]  2026-03-10
  • context [T] 2026-03-10

🧠>
```

---

## Шаг 7 — Загружаем файл и задаём вопрос (ручной режим)

```
🧠> /load main @src/main.py
🧠 [main]> что делает эта функция?
```

**Что проверяем:**
- Паттерн создался
- Модель ответила на вопрос о `hello()`
- Промпт переключился на `🧠 [main]>`

---

## Шаг 8 — grow_policy

```
🧠 [main]> есть ли способ сделать это лучше?
```

Должно появиться `[main~]` — паттерн растёт.

---

## Шаг 9 — Авто-выбор агентов (keyword)

```
🧠 [main~]> как правильно назвать эту функцию по конвенциям стиля?
```

Ожидаем:
```
🤖 Авто-агенты: style
```

---

## Шаг 10 — /patch: правим код

```
🧠 [main~]> переименуй функцию hello в greet и добавь параметр name: str
🧠 [main~]> /patch
```

**Путь резолвится через `client_project`** — относительный `main.py` из diff-заголовка автоматически превращается в абсолютный путь клиентского проекта.

---

## Шаг 10б — /patch для нового файла

```
🧠 [main~]> создай src/utils.py с функцией format_name
🧠 [main~]> /patch
```

Ожидаем `НОВЫЙ ФАЙЛ: src/utils.py`.

---

## Шаг 10в — /patch: несколько файлов

Если модель сгенерировала несколько diff-блоков:

```
🧠 [main~]> добавь валидацию в main.py и создай src/validators.py
🧠 [main~]> /patch
```

Ожидаем: `📋 Найдено 2 diff-блоков`, отдельный запрос подтверждения для каждого.

---

## Шаг 11 — Пайплайн (tree-sitter навигация)

Тест автоматического пайплайна. Tree-sitter находит файлы, пайплайн запускается.

### Сценарий A: задача описана текстом

```bash
python cognit.py
🧠> поправить функцию hello — сделать более расширенной
```

Ожидаем:
```
📇 Индекс: N файлов, M символов
📍 Найдено X символов в Y файлах:
   • main.py  (hello)

   Запустить пайплайн? [Y/n]: y

🚀 Пайплайн (5 стадий)
  ✓ [nav   ] navigator

  [agent] context
  ...
  [agent] arch
  ...
  [agent] style
  ...
  [coder] coder
  💡 Diff готов → /patch
```

**Что проверяем:**
- Tree-sitter нашёл файл с функцией
- Каждый агент создаёт временный паттерн `_agent_<id>` (полный eval)
- Агенты дают осмысленные мемо (не `<|im_end|>`)
- Временные паттерны удалены после использования

### Сценарий B: команда `route`

```
🧠> route добавить валидацию email
```

Ожидаем: tree-sitter поиск → файлы → предложение запустить пайплайн.

### Сценарий C: pipeline.json — отключить стадию

Редактируем `pipeline.json`:
```json
{
  "stages": [
    {"id": "navigator", "type": "navigator", "enabled": true},
    {"id": "context",   "type": "agent", "name": "context", "enabled": false},
    ...
  ]
}
```

В выводе пайплайна: `context` помечен `✗`, пропускается.

---

## Что считать успехом

| Проверка | Критерий |
|---|---|
| Smoke | Модель загрузилась без ошибок |
| card.md | Файлы `agents/*/card.md` созданы после первого запуска |
| pipeline.json | Файл создан в клиентском проекте при `cognit_setup.py` |
| grow ~ | `/list` — компактный формат, `~` для feature-паттернов |
| /patch (правка) | `main.py` изменён, `.cognit.bak` создан |
| /patch (новый файл) | `utils.py` создан, `.bak` не создаётся |
| /edit | Файл читается свежим, diff генерируется |
| /agent | Промпт меняется, `/agent off` возвращает обычный |
| Keyword agents | Вопрос про стиль → `🤖 Авто-агенты: style` |
| Tree-sitter навигация | Задача без паттерна → индекс находит файлы → пайплайн |
| pipeline.json disable | Отключённая стадия → `✗` в выводе, пропускается |
| /patch мульти-файл | Два diff-блока → два запроса подтверждения |
| /patch путь | Относительный путь из diff → резолвится в `client_project` |
| Совместимость модели | Паттерн от Q6_K + модель Q4_K_M → предупреждение |
| Повторный запуск | Вопрос без перезагрузки файла работает |
| post-commit хук | Grow-паттерн пропускается, retrain — пересоздаётся |
| Пайплайн агенты | Каждый агент: save_pattern → ask_pattern → ответ (не `<\|im_end\|>`) |
| Сброс | Удалить `echo_patterns/` → первый пайплайн пересоздаёт всё |
