#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_pipeline.py — конфигурируемый пайплайн агентов.

Пайплайн описывается в pipeline.json в корне клиентского проекта.
Каждая стадия получает накопленный контекст предыдущих и добавляет своё мемо.
Финальная стадия типа "coder" пишет unified diff.

Типы стадий:
  navigator — навигация через tree-sitter индекс (мемо вставляется автоматически)
  agent     — KV-cache агент пишет своё мемо
  coder     — Transformer строит context-паттерн и генерирует diff
  reviewer  — проверяет diff кодера по tree-sitter структуре, убирает дубли

Параметры верхнего уровня:
  passes — сколько раз прогнать агентов (default 1).
           passes=2: агенты видят мемо друг друга во втором прогоне → уточнение.
           Navigator, coder и reviewer всегда запускаются один раз независимо от passes.

Пример pipeline.json:
  {
    "passes": 2,
    "stages": [
      {"id": "navigator", "type": "navigator", "enabled": true},
      {"id": "context",   "type": "agent", "name": "context", "enabled": true},
      {"id": "arch",      "type": "agent", "name": "arch",    "enabled": true},
      {"id": "style",     "type": "agent", "name": "style",   "enabled": true},
      {"id": "coder",     "type": "coder",                    "enabled": true}
    ]
  }
"""

import json
from pathlib import Path

PIPELINE_FILENAME = "pipeline.json"

# Порядок и состав стадий по умолчанию
DEFAULT_PIPELINE: dict = {
    "passes": 1,    # 1 = один прогон; 2 = двойной прогон агентов
    "stages": [
        {
            "id":      "navigator",
            "type":    "navigator",
            "enabled": True,
            "comment": "Мемо от tree-sitter навигатора: какие файлы, где смотреть"
        },
        {
            "id":      "context",
            "type":    "agent",
            "name":    "context",
            "enabled": True,
            "role":    "Задача: {task}\n\nЧто важно учесть с точки зрения целей и требований проекта? "
                       "Напиши кратко (2-4 предложения). Не пиши код.",
            "comment": "Агент: контекст проекта, бизнес-требования"
        },
        {
            "id":      "arch",
            "type":    "agent",
            "name":    "arch",
            "enabled": True,
            "role":    "Задача: {task}\n\nКакие архитектурные ограничения важны? "
                       "Что нельзя сломать? Напиши кратко (2-4 предложения). Не пиши код.",
            "comment": "Агент: архитектура, зависимости, паттерны"
        },
        {
            "id":      "style",
            "type":    "agent",
            "name":    "style",
            "enabled": True,
            "role":    "Задача: {task}\n\nКакие соглашения по стилю кода нужно соблюсти? "
                       "Напиши кратко (2-4 предложения). Не пиши код.",
            "comment": "Агент: стиль, форматирование, именование"
        },
        {
            "id":      "coder",
            "type":    "coder",
            "enabled": True,
            "role":    "Измени существующие файлы для реализации задачи. "
                       "Напиши unified diff. Только ```diff блок, без объяснений.",
            "comment": "Финальная стадия: кодер пишет diff по всему контексту"
        },
        {
            "id":      "reviewer",
            "type":    "reviewer",
            "enabled": True,
            "role":    "Ты ревьюер. Проверь diff:\n"
                       "1. Правильные ли файлы и функции изменены?\n"
                       "2. Нет ли дублирования кода?\n"
                       "3. Совпадают ли номера строк @@ с реальными?\n"
                       "Если есть ошибки — напиши исправленный ```diff блок.\n"
                       "Если всё верно — выведи тот же diff без изменений.",
            "comment": "Ревью: проверяет diff по tree-sitter структуре, убирает дубли"
        }
    ]
}


def load_pipeline(client_project: str) -> dict:
    """
    Загружает pipeline.json из клиентского проекта.
    Если файла нет — возвращает DEFAULT_PIPELINE.
    Добавляет недостающие role из дефолта; сохраняет passes из файла.
    """
    if not client_project:
        return DEFAULT_PIPELINE
    p = Path(client_project) / PIPELINE_FILENAME
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            # Заполняем недостающие поля из дефолта
            defaults_by_id = {s["id"]: s for s in DEFAULT_PIPELINE["stages"]}
            for stage in cfg.get("stages", []):
                if "role" not in stage and stage.get("id") in defaults_by_id:
                    stage["role"] = defaults_by_id[stage["id"]].get("role", "")
                # Миграция: rwkv → navigator
                if stage.get("type") == "rwkv":
                    stage["type"] = "navigator"
            # passes по умолчанию 1 если не задан
            cfg.setdefault("passes", 1)
            return cfg
        except Exception as e:
            print(f"⚠️  Ошибка чтения pipeline.json: {e} — использую дефолт")
    return DEFAULT_PIPELINE


def save_default_pipeline(client_project: str) -> Path | None:
    """Записывает pipeline.json в клиентский проект если его нет."""
    if not client_project:
        return None
    p = Path(client_project) / PIPELINE_FILENAME
    if p.exists():
        return p  # не перезаписываем существующий
    p.write_text(json.dumps(DEFAULT_PIPELINE, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(f"✅ pipeline.json создан: {p}")
    return p


def describe_pipeline(pipeline: dict) -> str:
    """Возвращает читаемое описание стадий (и числа прогонов если > 1)."""
    passes = pipeline.get("passes", 1)
    lines  = []
    for s in pipeline.get("stages", []):
        enabled = "✓" if s.get("enabled", True) else "✗"
        stype   = s.get("type", "?")
        name    = s.get("name", s.get("id", ""))
        # agent-стадии показывают x passes если > 1
        repeat  = f" ×{passes}" if passes > 1 and stype == "agent" else ""
        lines.append(f"  {enabled} [{stype:6}] {name}{repeat}")
    return "\n".join(lines)
