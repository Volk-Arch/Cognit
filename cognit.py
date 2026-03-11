#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit.py — single entry point for Cognit.

Starts the Transformer backend with interactive CLI.
File routing — via tree-sitter index (cognit_index.py).

Run:
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
