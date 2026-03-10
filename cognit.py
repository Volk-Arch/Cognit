#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit.py — единая точка входа Cognit.

Запускает Transformer-бэкенд с интерактивным CLI.
Маршрутизация файлов — через tree-sitter индекс (cognit_index.py).

Запуск:
    python cognit.py
"""


def main():
    import cognit_transformer as t
    t.init_model()
    try:
        t.cli_loop()
    finally:
        t.unload_model()


if __name__ == "__main__":
    main()
