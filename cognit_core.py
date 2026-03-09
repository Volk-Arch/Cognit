#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_core.py — Общие утилиты для cognit_transformer.py и cognit_rwkv.py
=============================================================
Не загружает никаких моделей. Чистые функции для работы с:
  - git-контекстом (repo, branch, путь к паттернам)
  - хешированием файлов
  - чтением метаданных паттернов
  - CLI-хелперами (список, подсказки)
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

PATTERNS_BASE = "echo_patterns"


# =============================================================================
# GIT-КОНТЕКСТ
# =============================================================================
def git_repo_name() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True
        )
        return Path(out.stdout.strip()).name
    except Exception:
        return "local"


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return out.stdout.strip().replace("/", "-")
    except Exception:
        return "main"


def make_patterns_dir(repo: str, branch: str) -> str:
    """Возвращает путь к директории паттернов и создаёт её."""
    d = f"{PATTERNS_BASE}/{repo}/{branch}"
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# ХЕШИРОВАНИЕ ФАЙЛОВ
# =============================================================================
def file_hash(path: str) -> str:
    """SHA-256 начала + конца файла. Быстро, достаточно для детекции изменений."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if size > 65536:
            f.seek(max(0, size - 65536))
            h.update(f.read(65536))
    return h.hexdigest()[:16]


# =============================================================================
# МЕТАДАННЫЕ ПАТТЕРНОВ
# =============================================================================
def read_meta(patterns_dir: str, name: str) -> dict | None:
    """Читает .json метаданные паттерна. Возвращает None если нет файла."""
    p = Path(patterns_dir) / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_meta(patterns_dir: str, name: str, meta: dict):
    p = Path(patterns_dir) / f"{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def pattern_exists(patterns_dir: str, name: str) -> bool:
    return (Path(patterns_dir) / f"{name}.pkl").exists()


def list_pattern_names(patterns_dir: str) -> list[str]:
    """Имена всех паттернов (без _служебных)."""
    return [
        p.stem for p in sorted(Path(patterns_dir).glob("*.pkl"))
        if not p.stem.startswith("_")
    ]


def default_grow_policy(branch: str, source_paths: list[str]) -> str:
    """
    Определяет политику обновления паттерна по умолчанию.

    retrain — пересоздаётся при изменении файлов (диалог сбрасывается):
      - ветки main / master
      - паттерны из agents/

    grow — накапливает диалог, не пересоздаётся автоматически:
      - feature-ветки (всё остальное)
    """
    if branch in ("main", "master"):
        return "retrain"
    for p in source_paths:
        if "agents/" in p.replace("\\", "/"):
            return "retrain"
    return "grow"


# =============================================================================
# CLI-ХЕЛПЕРЫ
# =============================================================================
def print_patterns_list(patterns_dir: str):
    """Показывает все паттерны — одна строка на паттерн."""
    metas = [
        p for p in sorted(Path(patterns_dir).glob("*.json"))
        if not p.stem.startswith("_")
    ]
    if not metas:
        print("   Паттернов нет — загрузи: /load <имя> @<файл>")
        return

    print(f"📋 Паттернов: {len(metas)}")
    for p in metas:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        backend = m.get("backend", "?")[0].upper()   # T / R
        policy  = "~" if m.get("grow_policy") == "grow" else ""
        saved   = m.get("saved_at", "")[:10]          # 2026-03-09
        asks    = m.get("n_asks", 0)
        asks_s  = f"  {asks}д" if asks else ""
        print(f"  • {m['name']}{policy}  [{backend}]  {saved}{asks_s}")


def hint_patterns(patterns_dir: str):
    """Одна строка с именами паттернов — подсказка при отсутствии активного."""
    names = list_pattern_names(patterns_dir)
    if names:
        print(f"   Паттерны: {', '.join(names)}")
    else:
        print("   Паттернов нет — загрузи: /load <имя> @<файл>")


# =============================================================================
# ПРОВЕРКА АКТУАЛЬНОСТИ
# =============================================================================
def check_stale_sources(patterns_dir: str, name: str) -> list[str]:
    """
    Возвращает список изменившихся/удалённых source_files для паттерна.
    Пустой список = всё актуально.
    """
    meta = read_meta(patterns_dir, name)
    if not meta:
        return []

    changed = []
    for src in meta.get("source_files", []):
        path = src["path"]
        if not os.path.exists(path):
            changed.append(f"{path}  (не найден)")
        elif file_hash(path) != src["hash"]:
            changed.append(path)
    return changed


# =============================================================================
# ROUTE: ПОСЛЕДНИЙ МАРШРУТ
# =============================================================================
def save_route(patterns_dir: str, index: str, task: str, files: list[str]):
    """Сохраняет последний результат route в _route_last.json."""
    route_path = Path(patterns_dir) / "_route_last.json"
    with open(route_path, "w", encoding="utf-8") as f:
        json.dump({
            "index":     index,
            "task":      task,
            "files":     files,
            "routed_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    return route_path


def load_last_route(patterns_dir: str) -> dict | None:
    """Читает _route_last.json. Возвращает None если нет или старше 24 часов."""
    route_path = Path(patterns_dir) / "_route_last.json"
    if not route_path.exists():
        return None
    with open(route_path, encoding="utf-8") as f:
        data = json.load(f)

    # Игнорируем маршруты старше 24 часов
    try:
        routed_at = datetime.fromisoformat(data["routed_at"])
        age_hours = (datetime.now() - routed_at).total_seconds() / 3600
        if age_hours > 24:
            return None
    except Exception:
        return None

    return data


# =============================================================================
# EXPAND: ЗАПРОС TRANSFORMER → RWKV
# =============================================================================
def save_expand_request(patterns_dir: str, task: str, from_pattern: str, from_sources: list[str]):
    """Сохраняет запрос Transformer к RWKV в _expand_last.json."""
    expand_path = Path(patterns_dir) / "_expand_last.json"
    with open(expand_path, "w", encoding="utf-8") as f:
        json.dump({
            "task":         task,
            "from_pattern": from_pattern,
            "from_sources": from_sources,
            "requested_at": datetime.now().isoformat(),
        }, f, ensure_ascii=False, indent=2)
    return expand_path


def load_expand_request(patterns_dir: str) -> dict | None:
    """Читает _expand_last.json. Возвращает None если нет или старше 24 часов."""
    expand_path = Path(patterns_dir) / "_expand_last.json"
    if not expand_path.exists():
        return None
    with open(expand_path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        age_hours = (datetime.now() - datetime.fromisoformat(data["requested_at"])).total_seconds() / 3600
        if age_hours > 24:
            return None
    except Exception:
        return None
    return data


# =============================================================================
# СБОР ФАЙЛОВ ДЛЯ /load (уважает .gitignore)
# =============================================================================

# Расширения текстовых файлов которые имеет смысл читать модели
TEXT_EXTENSIONS = {
    # Код
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".vue", ".svelte", ".sh", ".bash", ".ps1",
    # Данные / конфиг
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    # Документация
    ".md", ".txt", ".rst",
}


def collect_text_files(dir_path: str | Path) -> list[Path]:
    """
    Возвращает текстовые файлы из директории.

    Если это git-репо — использует `git ls-files` (автоматически
    уважает .gitignore и возвращает только отслеживаемые файлы).
    Иначе — rglob с фильтром по TEXT_EXTENSIONS.
    """
    p = Path(dir_path).resolve()

    # Пробуем git ls-files
    try:
        result = subprocess.run(
            ["git", "-C", str(p), "ls-files", "--cached", "."],
            capture_output=True, text=True, check=True
        )
        files = []
        for line in result.stdout.splitlines():
            f = (p / line).resolve()
            if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS:
                files.append(f)
        if files:
            return sorted(files)
    except Exception:
        pass

    # Fallback: rglob без git
    return sorted(
        f for f in p.rglob("*")
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS
    )
