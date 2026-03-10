#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_setup.py — одноразовая настройка Cognit для проекта
========================================================
Запуск:
    python cognit_setup.py

Что делает:
    1. Создаёт .echo.json  — фиксирует модель и бэкенд для команды
    2. Добавляет echo_patterns/ в .gitignore
    3. Устанавливает post-commit хук (опционально)
    4. Показывает S3-команды для синхронизации паттернов
"""

import os
import sys
import json
import subprocess
from pathlib import Path

ECHO_CONFIG   = ".echo.json"
GITIGNORE     = ".gitignore"
PATTERNS_BASE = "echo_patterns"


# =============================================================================
# ХЕЛПЕРЫ
# =============================================================================
def _git_root() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True
        )
        return Path(out.stdout.strip())
    except Exception:
        return None


def _git_repo_name(root: Path) -> str:
    return root.name


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{prompt}{suffix}: ").strip().lower()
    if not val:
        return default
    return val == "y"


def _ask_client_project() -> Path | None:
    """Спрашивает путь к основному (клиентскому) проекту."""
    print("\n── Основной проект (для git-хука) ──────────────────────")
    print("   Cognit живёт в своём репо. Укажи путь к проекту,")
    print("   который хочешь анализировать — хук поставится туда.")

    # Подставляем сохранённый путь как дефолт
    existing_client = ""
    if Path(ECHO_CONFIG).exists():
        try:
            with open(ECHO_CONFIG, encoding="utf-8") as f:
                existing_client = json.load(f).get("client_project", "")
        except Exception:
            pass

    raw = _ask("   Путь к проекту (Enter — пропустить)", existing_client)
    if not raw:
        return None

    p = Path(raw).expanduser().resolve()
    if not p.exists():
        print(f"   ⚠️  Путь не найден: {p}")
        return None
    if not (p / ".git").exists():
        print(f"   ⚠️  {p} — не git-репозиторий (нет папки .git)")
        return None

    print(f"   Проект: {p.name}  ({p})")
    return p


# =============================================================================
# ШАГ 1: .echo.json
# =============================================================================
def setup_config():
    print("\n── Конфигурация проекта (.echo.json) ──────────────────")

    existing = {}
    if Path(ECHO_CONFIG).exists():
        with open(ECHO_CONFIG, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"   Найден существующий {ECHO_CONFIG}")

    ex_t = existing.get("transformer", {})

    t_model_path = _ask(
        "   Путь к Transformer-модели (GGUF)",
        ex_t.get("model_path", "models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    )

    config = {
        "backend":      "transformer",
        "patterns_dir": PATTERNS_BASE,
        "transformer": {
            "model_path":   t_model_path,
            "n_gpu_layers": ex_t.get("n_gpu_layers", -1),
            "n_ctx":        ex_t.get("n_ctx", 8192),
            "max_tokens":   ex_t.get("max_tokens", 512),
        },
        "_note": (
            "Все участники команды должны использовать одну модель. "
            "Паттерны несовместимы между разными моделями."
        ),
    }

    with open(ECHO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Создан {ECHO_CONFIG}")
    return config


# =============================================================================
# ШАГ 2: .gitignore
# =============================================================================
def setup_gitignore():
    print("\n── .gitignore ──────────────────────────────────────────")

    lines_to_add = [
        "# Cognit — паттерны не версионируются, хранятся локально или в S3",
        f"{PATTERNS_BASE}/",
    ]

    existing_lines = []
    if Path(GITIGNORE).exists():
        existing_lines = Path(GITIGNORE).read_text(encoding="utf-8").splitlines()

    to_add = [l for l in lines_to_add if l not in existing_lines]
    if not to_add:
        print(f"   ✅ {GITIGNORE} уже содержит нужные записи")
        return

    with open(GITIGNORE, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(to_add) + "\n")

    for line in to_add:
        if not line.startswith("#"):
            print(f"   ✅ Добавлено в {GITIGNORE}: {line}")


# =============================================================================
# ШАГ 3: post-commit хук
# =============================================================================
def _make_hook_script(echo_script_path: str) -> str:
    """Генерирует скрипт хука с абсолютным путём к echo-скрипту."""
    # Используем forward slashes — Git sh на Windows их понимает
    posix_path = Path(echo_script_path).as_posix()
    return f"""\
#!/bin/sh
# Cognit — post-commit hook
# echo-скрипт: {posix_path}

if git rev-parse HEAD~1 >/dev/null 2>&1; then
    CHANGED=$(git diff HEAD~1 HEAD --name-only)
else
    CHANGED=$(git show --name-only --pretty="" HEAD)
fi

if [ -z "$CHANGED" ]; then
    exit 0
fi

ECHO_SCRIPT="{posix_path}"
if [ ! -f "$ECHO_SCRIPT" ]; then
    exit 0
fi

echo "[Cognit] Checking patterns for changed files..."
echo "$CHANGED" | while read -r file; do
    python "$ECHO_SCRIPT" --refresh-file "$file" 2>/dev/null
done
"""


def setup_hook(target_git_root: Path, echo_script_path: Path):
    """Устанавливает post-commit хук в указанный git-репозиторий."""
    hook_path = target_git_root / ".git" / "hooks" / "post-commit"

    if hook_path.exists():
        overwrite = _ask_yn(
            f"   Хук уже существует в {target_git_root.name}. Перезаписать?",
            default=False,
        )
        if not overwrite:
            print("   Пропускаю.")
            return

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(_make_hook_script(str(echo_script_path)), encoding="utf-8")

    try:
        import stat
        st = os.stat(hook_path)
        os.chmod(hook_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    print(f"   ✅ Хук установлен: {hook_path}")


# =============================================================================
# ШАГ 4: S3-инструкции
# =============================================================================
def show_s3_instructions(config: dict, repo_name: str):
    print("\n── Синхронизация паттернов через S3 ───────────────────")
    bucket = "your-bucket"
    prefix = "echo-patterns"

    print(f"""
   Структура пути паттернов:
     Локально: {PATTERNS_BASE}/<repo>/<branch>/<name>.pkl
     В S3:     s3://{bucket}/{prefix}/<repo>/<branch>/<name>.pkl

   Загрузить паттерны текущей ветки в S3:
     aws s3 sync {PATTERNS_BASE}/{repo_name}/ \\
         s3://{bucket}/{prefix}/{repo_name}/

   Скачать паттерны с S3:
     aws s3 sync s3://{bucket}/{prefix}/{repo_name}/ \\
         {PATTERNS_BASE}/{repo_name}/

   Скачать конкретную ветку:
     aws s3 sync s3://{bucket}/{prefix}/{repo_name}/main/ \\
         {PATTERNS_BASE}/{repo_name}/main/

   Метаданные каждого паттерна (.json) содержат:
     backend, model, repo, branch, source_files
   → Можно выбрать нужный файл по этим полям без загрузки .pkl
""")


# =============================================================================
# ШАГ 5: agents/ — знания о проекте
# =============================================================================

# Шаблоны для каждого типа агента
_AGENT_TEMPLATES = {
    "style/global.md": """\
# Стиль кода — DemoAI

## Именование
- Все функции и переменные: `snake_case`
- Приватные хелперы (не часть публичного API): `_underscore_prefix` — например `_read_prob`
- Параметры вероятностей: полные имена, не сокращать:
  - `prior_h` (не `p`, не `prior`)
  - `p_e_given_h` (не `peh`, не `likelihood`)
  - `p_e_given_not_h` (не `pnh`)
  - `posterior` (не `post`, не `result`)
- Математическое отрицание в именах: суффикс `_not_h` (не `_neg`, не `_complement`)
- Счётчики шагов: `step` (не `i`, не `n`, не `iteration`)
- Константы: `UPPER_SNAKE_CASE` если добавляются на уровне модуля

## Форматирование
- Отступы: 4 пробела (без табов)
- Максимальная длина строки: 88 символов
- Пустые строки: 2 между топ-уровневыми функциями
- Кавычки: двойные для строк пользователю, одинарные допустимы внутри f-строк
- Float-вывод: всегда `.6f` для вероятностей — `f"{value:.6f}"`

## Импорты
- Первая строка файла: `from __future__ import annotations`
- Порядок: stdlib → (сторонние, если появятся) → локальные
- `import *` запрещён

## Типизация
- Тип-хинты обязательны для всех функций (параметры + возвращаемое значение)
- `float` для вероятностей, `str` для промптов, `None` для процедур
- Python 3.10+ синтаксис: `X | None` вместо `Optional[X]`

## Сообщения пользователю
- Язык: русский (проект русскоязычный)
- Стиль: вежливый императив — «Введите...», «Попробуйте...», «Вероятность должна быть...»
- Формат ошибки валидации: одно предложение, конкретное исправление
- Формат результата: два числа в столбик с выравниванием:
  ```
    P(H) было:     0.300000
    P(H|E) стало:  0.462963
  ```
- Заголовок шага: `f"\\nШаг {step}:"` — с пустой строкой перед

## Обработка ошибок
- `ZeroDivisionError`: поднимать из `bayes_update` с понятным русским сообщением
- Ловить явно: `except ZeroDivisionError as e:` — не голый `except:`
- `KeyboardInterrupt`: ловить в `main`, печатать `"\\nВыход."` и завершать
- `ValueError`: ловить в `_read_prob` при `float()`, продолжать цикл
- Не выводить traceback пользователю

## Антипаттерны (запрещено)
- Сокращать имена параметров вероятностей (`peh`, `pnh`, `ph`)
- Форматировать float без указания точности (только `.6f` или явная другая)
- Добавлять глобальное состояние (все данные через параметры функций)
- Разрывать цепочку `prior → bayes_update → posterior → новый prior` скрытыми присваиваниями
- Смешивать вычисление и вывод: `bayes_update` не должна печатать ничего
""",

    "style/commands.md": """\
# Стиль интерактивного CLI — DemoAI

## Промпты ввода
- Формат: `"Введите {описание}: "` — двоеточие и пробел перед курсором
- Шаговые промпты: с отступом `"  Введите P(E|H): "` (2 пробела)
- Поддержка запятой как десятичного разделителя (`.replace(",", ".")`)
- Команда выхода: `q`, `quit`, `exit` — регистронезависимо, в любой точке ввода

## Вывод результатов
- Два числа в столбик, выравнивание по `:`:
  ```
    P(H) было:     0.300000
    P(H|E) стало:  0.462963
  ```
- Всегда `.6f` для вероятностей — 6 знаков после запятой
- Пустая строка перед каждым шагом: `print(f"\\nШаг {step}:")`

## Сообщения об ошибках
- Одна строка, без `❌` — проект учебный, строгий тон не нужен
- Примеры: `"Введите число (например 0.2) или q для выхода."`
- `"Вероятность должна быть в диапазоне [0, 1]."`
- `f"Ошибка: {e}"` — для ZeroDivisionError

## Завершение
- `"\\nВыход."` — по `q` или Ctrl+C, с пустой строкой перед
- Нет `sys.exit()` с ненулевым кодом для нормального выхода пользователя
""",

    "arch/overview.md": """\
# Архитектура DemoAI

## Структура проекта
```
main.py   — весь код, три функции
README.md — описание
```
Нет пакетов, нет подпапок, нет зависимостей кроме stdlib (`sys`).

## Три функции и их роли

### `_read_prob(prompt: str) -> float`
- Единственная точка ввода от пользователя
- Бесконечный цикл до получения корректного float в [0, 1]
- Обрабатывает: нечисловой ввод (ValueError), диапазон, команду выхода (KeyboardInterrupt)
- Не знает о байесовской логике — только валидация числа-вероятности
- Если нужно добавить новый тип ввода — создавать аналогичный хелпер `_read_*`

### `bayes_update(prior_h, p_e_given_h, p_e_given_not_h) -> float`
- Чистая функция: только вычисление, никакого I/O
- Реализует формулу: `P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|¬H)*(1-P(H))]`
- Единственный способ упасть: `ZeroDivisionError` когда знаменатель == 0.0
- Всегда вызывается с тремя float, возвращает один float
- Если добавляется новая байесовская формула — рядом с этой функцией, тем же стилем

### `main() -> None`
- Точка входа и весь UI-цикл
- Ловит исключения от `bayes_update` и `KeyboardInterrupt`
- Управляет состоянием: `prior` (float) и `step` (int)
- Итеративный паттерн: `prior = posterior` в конце каждого шага

## Поток данных
```
_read_prob("P(H)")  →  prior
  ↓
loop:
  _read_prob("P(E|H)")    →  p_e_h
  _read_prob("P(E|¬H)")   →  p_e_nh
  bayes_update(prior, p_e_h, p_e_nh)  →  posterior
  prior = posterior
  step += 1
```

## Правила расширения

**Куда добавлять новые вычисления:** рядом с `bayes_update`, как отдельная чистая функция.
Пример: `log_odds_update(prior_h, likelihood_ratio) -> float`.

**Куда добавлять новые типы ввода:** новый хелпер `_read_*`, вызывать из `main`.
Пример: `_read_label(prompt: str) -> str` для именованных гипотез.

**Куда добавлять вывод:** только в `main`. `bayes_update` не должна ничего печатать.

**Куда НЕ добавлять глобальное состояние:** всё передаётся через параметры.

## Критические инварианты
- `bayes_update` остаётся чистой функцией — без print, без input, без side effects
- Знаменатель в `bayes_update` проверяется перед делением — `ZeroDivisionError` если 0.0
- `prior` после каждого шага = предыдущий `posterior` — цепочка не должна прерываться
- Все вероятности в [0, 1] — `_read_prob` гарантирует это на входе
""",

    "context/project.md": """\
# Контекст проекта — DemoAI

## Что это
Учебный интерактивный калькулятор байесовского обновления вероятностей.
Пользователь вводит априорную вероятность P(H) и на каждом шаге — новые свидетельства
P(E|H) и P(E|¬H). Программа показывает как вера обновляется итерационно по формуле Байеса.

## Цель
Демонстрация байесовского мышления: как накопление свидетельств меняет вероятность гипотезы.
Не production-инструмент — образовательное демо для студентов теории вероятностей.

## Аудитория
Студенты и начинающие разработчики изучающие вероятностный вывод.
Русскоязычный интерфейс — все сообщения на русском.

## Технологии
- Python 3.10+ (используется синтаксис `X | None`, `from __future__ import annotations`)
- Только стандартная библиотека (`sys`) — никаких внешних зависимостей
- Никакой БД, файловой системы, HTTP — полностью stateless

## Принципы
- **Простота прежде всего** — одна задача, один файл, три функции
- **Никаких зависимостей** — если нужна библиотека, скорее всего это перебор
- **Чистые функции** — вычисление отделено от ввода/вывода
- **Понятные ошибки** — пользователь всегда знает что ввести чтобы продолжить

## Что уместно добавлять
- Именованные гипотезы: `"Введите название гипотезы: "` → `"H: дождь идёт"`
- История шагов: список `(prior, p_e_h, p_e_nh, posterior)` с выводом в конце
- Альтернативные формы байесовского обновления (log-odds, likelihood ratio)
- CSV-экспорт истории шагов через `csv.writer`
- Режим пакетной обработки через аргументы командной строки (`argparse`)

## Что НЕ уместно добавлять
- Веб-интерфейс, HTTP-сервер, REST API
- База данных, сессии, пользователи
- Внешние библиотеки (numpy, scipy) — для учебного демо избыточно
- Многопоточность / асинхронность
- GUI (tkinter, PyQt) — проект CLI-ориентированный

## Байесовская формула (для справки)
```
P(H|E) = P(E|H) × P(H)
         ─────────────────────────────────────
         P(E|H) × P(H) + P(E|¬H) × (1 − P(H))
```
Где:
- P(H) — априорная вероятность гипотезы (до наблюдения свидетельства)
- P(E|H) — вероятность свидетельства если гипотеза верна (likelihood)
- P(E|¬H) — вероятность свидетельства если гипотеза неверна
- P(H|E) — апостериорная вероятность (после наблюдения, становится новым prior)
""",
}


def setup_agents(target_dir: Path | None = None):
    """Создаёт папку agents/ в клиентском проекте (или CWD если не указан)."""
    print("\n── agents/ — знания о проекте ──────────────────────────")

    base = target_dir if target_dir else Path(".")
    agents_dir = base / "agents"

    if target_dir:
        print(f"   Создаю в: {target_dir}")

    created = []

    for rel_path, content in _AGENT_TEMPLATES.items():
        target = agents_dir / rel_path
        if target.exists():
            print(f"   • {rel_path}  (уже существует, пропускаю)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created.append(rel_path)
        print(f"   ✅ Создан: {target}")

    if not created:
        print("   Все файлы уже существуют.")
        return

    agents_path = agents_dir.resolve()
    print(f"""
   Структура agents/:
     {agents_path}/
       style/global.md    — стиль кода (дефолт: Python best practices)
       style/commands.md  — стиль CLI-вывода (дефолт: заполнен)
       arch/overview.md   — архитектура (замени на свою структуру)
       context/project.md — цели, решения, ограничения (замени)

   Cognit при следующем запуске автоматически:
     1. Загрузит агентов из этих папок и создаст KV-cache паттерны
     2. Подключит агентов в пайплайн и ambient-режим

   Можно сразу запустить — дефолтные файлы уже рабочие.
   Лучший результат — после заполнения arch/ и context/ под свой проект.
""")

    print("   Совет: agents/ нужно добавить в git клиентского проекта.")
    print(f"   cd {base.resolve()} && git add agents/ && git commit -m 'Add agent knowledge base'")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import sys

    # Субкоманды: python cognit_setup.py agents
    if len(sys.argv) > 1:
        subcmd = sys.argv[1].lower()
        if subcmd == "agents":
            print("""
╔══════════════════════════════════════════════╗
║  🧠 Cognit — Инициализация agents/          ║
╚══════════════════════════════════════════════╝""")
            # Читаем client_project из .echo.json если есть
            client = None
            if Path(ECHO_CONFIG).exists():
                try:
                    cp = json.loads(Path(ECHO_CONFIG).read_text(encoding="utf-8")).get("client_project", "")
                    if cp and Path(cp).exists():
                        client = Path(cp)
                except Exception:
                    pass
            setup_agents(client)
            return
        else:
            print(f"Неизвестная команда: {subcmd}")
            print("Использование: python cognit_setup.py [agents]")
            sys.exit(1)

    print("""
╔══════════════════════════════════════════════╗
║  🧠 Cognit — Настройка проекта             ║
╚══════════════════════════════════════════════╝""")

    git_root = _git_root()
    if git_root:
        repo_name = _git_repo_name(git_root)
        print(f"\n   Git-репозиторий: {repo_name}  ({git_root})")
    else:
        repo_name = "local"
        print("\n   ⚠️  Git-репозиторий не обнаружен. Паттерны будут в echo_patterns/local/")

    config = setup_config()
    setup_gitignore()

    # Абсолютный путь к скрипту (для прописывания в хук)
    echo_dir = Path(__file__).parent.resolve()
    transformer_script = echo_dir / "cognit_transformer.py"

    # Хуки в основном (клиентском) проекте
    client_root = _ask_client_project()
    if client_root:
        if _ask_yn(f"\n   Установить post-commit хук в {client_root.name}?", default=True):
            setup_hook(client_root, transformer_script)
        # Сохраняем путь к клиентскому проекту в .echo.json
        config["client_project"] = str(client_root)
        with open(ECHO_CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"   ✅ client_project сохранён в {ECHO_CONFIG}")

    # Хук в самом Cognit-репозитории (опционально)
    if git_root and _ask_yn("\nУстановить post-commit хук и в Cognit-репозитории?", default=False):
        setup_hook(git_root, transformer_script)

    if _ask_yn("\nСоздать папку agents/ с шаблонами?", default=True):
        setup_agents(client_root)  # None если клиент не задан → создаст в CWD

    # Создаём pipeline.json в клиентском проекте если его нет
    if client_root:
        try:
            from cognit_pipeline import save_default_pipeline
            save_default_pipeline(str(client_root))
        except ImportError:
            pass

    show_s3_instructions(config, repo_name)

    print("✅ Настройка завершена.\n")


if __name__ == "__main__":
    main()
