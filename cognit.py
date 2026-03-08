#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit.py — единая точка входа Cognit.

Переключает Transformer ↔ RWKV в одном процессе:
  - expand (Transformer → RWKV): нужен обзор всей кодовой базы
  - route <задача> (Transformer → RWKV → Transformer): найти файлы через RWKV-индекс
  - route <задача> (RWKV → Transformer): найти файлы и отдать Transformer

Запуск:
    python cognit.py              # бэкенд из .echo.json (по умолчанию transformer)
    python cognit.py --rwkv       # принудительно RWKV
    python cognit.py --transformer  # принудительно Transformer

Запуск конкретного бэкенда напрямую:
    python cognit_transformer.py
    python cognit_rwkv.py
"""

import sys
import json
from pathlib import Path


def _default_backend() -> str:
    p = Path(".echo.json")
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if cfg.get("backend") == "rwkv":
                return "rwkv"
        except Exception:
            pass
    return "transformer"


def main():
    if "--rwkv" in sys.argv:
        backend = "rwkv"
    elif "--transformer" in sys.argv:
        backend = "transformer"
    else:
        backend = _default_backend()

    pending = None

    while True:
        if backend == "transformer":
            import cognit_transformer as t
            if t.llm is None:
                t.init_model()
            result = t.cli_loop(pending=pending)
            t.unload_model()
        else:
            import cognit_rwkv as r
            if r.llm is None:
                r.init_model()
            result = r.cli_loop(pending=pending)
            r.unload_model()

        if result is None:
            break  # /exit — нормальный выход

        # Переключаем бэкенд по action из handoff
        action = result.get("action", "")
        if action == "expand":
            print(f"\n⚙️  Переключение: Transformer → RWKV  (задача: {result.get('task', '')[:60]})")
            backend = "rwkv"
        elif action == "expand_route":
            # route из Transformer: переключаемся в RWKV только для авто-маршрута
            print(f"\n⚙️  Переключение: Transformer → RWKV  (авто-маршрут: {result.get('task', '')[:60]})")
            backend = "rwkv"
        elif action == "route":
            print(f"\n⚙️  Переключение: RWKV → Transformer  (файлов: {len(result.get('files', []))})")
            backend = "transformer"
        else:
            break

        pending = result


if __name__ == "__main__":
    main()
