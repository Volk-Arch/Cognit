#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_agents.py — утилиты для работы с агентами (agents/).
Нет зависимостей на LLM.
"""

import json
import cognit_core as core
from pathlib import Path


def echo_config() -> dict:
    """Читает .echo.json. Возвращает {} если файл не найден."""
    p = Path(".echo.json")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_agents(patterns_dir: str) -> list[tuple[str, bool]]:
    """
    Возвращает список агентов из {client_project}/agents/.
    Каждый элемент: (имя, загружен_ли_паттерн).
    """
    agents_root = Path(echo_config().get("client_project", "")) / "agents"
    if not agents_root.exists():
        return []
    return [
        (d.name, core.pattern_exists(patterns_dir, d.name))
        for d in sorted(agents_root.iterdir()) if d.is_dir()
    ]


def read_agent_text(agent_name: str) -> str:
    """Читает сырой текст агента из {client_project}/agents/<name>/."""
    cfg = echo_config()
    agent_dir = Path(cfg.get("client_project", "")) / "agents" / agent_name
    if not agent_dir.exists():
        return ""
    files = core.collect_text_files(agent_dir)
    if not files:
        return ""
    return "\n\n---\n\n".join(
        f.read_text(encoding="utf-8", errors="ignore") for f in files
    )
