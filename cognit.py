#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit.py — единая точка входа Cognit.

Пользователь ВСЕГДА говорит с Transformer.
RWKV — внутренний инструмент для индексирования и маршрутизации, не показывает CLI.

Запуск:
    python cognit.py              # Transformer (по умолчанию)
    python cognit.py --rwkv       # RWKV в интерактивном режиме (для индексации вручную)
    python cognit.py --transformer  # явно Transformer
"""

import sys


def _default_backend() -> str:
    return "transformer"


def _rwkv_headless(action: str, pending: dict) -> dict | None:
    """
    Запускает RWKV headless (без интерактивного CLI):
      - expand       → строит индекс если нет, маршрутизирует задачу
      - expand_route → маршрутизирует через существующий индекс
    Возвращает route dict или None.
    """
    import cognit_rwkv as r
    if r.llm is None:
        r.init_model()

    try:
        if action == "expand":
            result = r.headless_route(
                pending.get("task", ""),
                pending.get("original_question", ""),
            )
        else:  # expand_route
            result = r.headless_route_with_index(
                pending.get("index", ""),
                pending.get("task", ""),
                pending.get("original_question", ""),
            )
    finally:
        r.unload_model()

    return result


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
            auto_route = pending is not None and pending.get("auto", False)
            result = t.cli_loop(pending=pending, auto_route=auto_route)
            t.unload_model()

            if result is None:
                break  # /exit

            action = result.get("action", "")

            if action in ("expand", "expand_route"):
                # RWKV работает headless — пользователь его не видит
                print(f"\n⚙️  RWKV: {'индексирую и ' if action == 'expand' else ''}маршрутизирую...")
                route = _rwkv_headless(action, result)
                if route:
                    pending = route
                    # Остаёмся на Transformer — он подхватит route в следующей итерации
                else:
                    print("⚠️  Не удалось определить файлы. Продолжаю без маршрутизации.")
                    pending = None
                # backend не меняем — всегда Transformer

            elif action == "route":
                # Прямой route от интерактивного RWKV (--rwkv режим)
                pending = result
                backend = "transformer"

            else:
                break

        else:
            # Явный интерактивный RWKV (python cognit.py --rwkv)
            import cognit_rwkv as r
            if r.llm is None:
                r.init_model()
            result = r.cli_loop(pending=pending)
            r.unload_model()

            if result is None:
                break  # /exit

            if result.get("action") == "route":
                backend = "transformer"
                pending = result
            else:
                break


if __name__ == "__main__":
    main()
