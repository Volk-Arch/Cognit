#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
echo_setup.py — одноразовая настройка Echo для проекта
========================================================
Запуск:
    python echo_setup.py

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
    print("   Echo живёт в своём репо. Укажи путь к проекту,")
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

    backend = _ask(
        "   Бэкенд (transformer / rwkv)",
        existing.get("backend", "transformer")
    )

    default_model = {
        "transformer": "Qwen3-8B-Q4_K_M",
        "rwkv": "RWKV-6-World-7B-Q4_K_M",
    }.get(backend, "Qwen3-8B-Q4_K_M")

    model_name = _ask(
        "   Имя модели (без пути и расширения)",
        existing.get("model_name", default_model)
    )

    config = {
        "backend":    backend,
        "model_name": model_name,
        "patterns_dir": PATTERNS_BASE,
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
        "# Echo PoC — паттерны не версионируются, хранятся локально или в S3",
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
# Echo PoC — post-commit hook
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

echo "[Echo] Checking patterns for changed files..."
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
# Глобальный стиль кода

Опиши здесь общие соглашения проекта. Примеры:

## Именование
- Переменные и функции: snake_case
- Классы: PascalCase
- Константы: UPPER_SNAKE_CASE
- Приватные поля: _prefix

## Форматирование
- Отступы: 4 пробела
- Максимальная длина строки: 100 символов
- Кавычки: двойные для строк

## Импорты
- Стандартная библиотека → сторонние → локальные
- Не использовать `import *`

## Документация
- Докстринги в формате: ...
""",

    "style/commands.md": """\
# Стиль команд и CLI

Опиши соглашения для команд, если проект предоставляет CLI.

## Именование команд
- ...

## Флаги
- ...

## Примеры вывода
- ...
""",

    "arch/overview.md": """\
# Архитектура проекта

## Структура папок
```
src/
  ...
```

## Основные модули
- `module_a` — отвечает за...
- `module_b` — отвечает за...

## Поток данных
1. ...
2. ...

## Ключевые зависимости
- ...

## Что избегать
- ...
""",

    "context/project.md": """\
# Контекст проекта

## Цель проекта
...

## Целевая аудитория
...

## Текущий статус
...

## Важные решения и их причины
- Решение X принято потому что...

## Что не делать
- ...
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
       style/global.md    — глобальный стиль кода
       style/commands.md  — стиль команд/CLI
       arch/overview.md   — архитектура проекта
       context/project.md — контекст и бизнес-правила

   Заполни шаблоны → загрузи в echo как паттерны:
     /load style @{agents_path}/style/
     /load arch  @{agents_path}/arch/
""")

    print("   Совет: agents/ нужно добавить в git клиентского проекта.")
    print(f"   cd {base.resolve()} && git add agents/ && git commit -m 'Add agent knowledge base'")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import sys

    # Субкоманды: python echo_setup.py agents
    if len(sys.argv) > 1:
        subcmd = sys.argv[1].lower()
        if subcmd == "agents":
            print("""
╔══════════════════════════════════════════════╗
║  🧠 Echo — Инициализация agents/             ║
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
            print("Использование: python echo_setup.py [agents]")
            sys.exit(1)

    print("""
╔══════════════════════════════════════════════╗
║  🧠 Echo PoC — Настройка проекта            ║
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

    # Абсолютный путь к echo-скрипту (для прописывания в хук)
    echo_dir = Path(__file__).parent.resolve()
    backend = config.get("backend", "transformer")
    script_name = "echo_rwkv.py" if backend == "rwkv" else "echo_poc.py"
    echo_script = echo_dir / script_name

    # Хук в основном (клиентском) проекте
    client_root = _ask_client_project()
    if client_root:
        if _ask_yn(f"\n   Установить хук в {client_root.name}?", default=True):
            setup_hook(client_root, echo_script)
        # Сохраняем путь к клиентскому проекту в .echo.json
        config["client_project"] = str(client_root)
        with open(ECHO_CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"   ✅ client_project сохранён в {ECHO_CONFIG}")

    # Хук в самом Echo-репозитории (опционально)
    if git_root and _ask_yn("\nУстановить хук и в Echo-репозитории?", default=False):
        setup_hook(git_root, echo_script)

    if _ask_yn("\nСоздать папку agents/ с шаблонами?", default=True):
        setup_agents(client_root)  # None если клиент не задан → создаст в CWD

    show_s3_instructions(config, repo_name)

    print("✅ Настройка завершена.\n")


if __name__ == "__main__":
    main()
