#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_patch.py — применение unified diff к файлам.
Нет зависимостей на LLM.
"""

import re
from pathlib import Path


def _extract_diff(text: str) -> str | None:
    """Извлекает unified diff из ответа модели (```diff блок или сырой diff)."""
    m = re.search(r'```diff\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(
        r'(---[ \t]+\S[^\n]*\n\+\+\+[ \t]+\S[^\n]*\n(?:@@[^\n]*\n(?:[+\- \\][^\n]*\n?)*)+)',
        text, re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def _extract_all_diffs(text: str) -> list[str]:
    """Извлекает все unified diff блоки из ответа модели."""
    diffs = re.findall(r'```diff\s*\n(.*?)```', text, re.DOTALL)
    if diffs:
        return [d.strip() for d in diffs if d.strip()]
    # Fallback: один diff в сыром формате
    single = _extract_diff(text)
    return [single] if single else []


def _diff_target(diff: str) -> str | None:
    """Извлекает путь целевого файла из заголовка +++ diff."""
    m = re.search(r'^\+\+\+[ \t]+(?:b/)?(.+?)(?:[ \t]+\d{4}|$)', diff, re.MULTILINE)
    if m:
        path = m.group(1).strip()
        return None if path == '/dev/null' else path
    return None


def _is_new_file_diff(diff: str) -> bool:
    """Возвращает True если diff создаёт новый файл (--- /dev/null)."""
    return bool(re.search(r'^---[ \t]+/dev/null', diff, re.MULTILINE))


def _apply_unified_diff(diff_text: str, orig_lines: list[str]) -> list[str]:
    """Чистая Python-реализация применения unified diff."""
    result = []
    orig_pos = 0
    hunk_re = re.compile(r'^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@')
    diff_lines = diff_text.splitlines()
    i = 0

    while i < len(diff_lines) and not diff_lines[i].startswith('@@'):
        i += 1

    while i < len(diff_lines):
        m = hunk_re.match(diff_lines[i])
        if not m:
            i += 1
            continue
        old_start = int(m.group(1)) - 1  # → 0-indexed
        i += 1

        while orig_pos < old_start and orig_pos < len(orig_lines):
            result.append(orig_lines[orig_pos])
            orig_pos += 1

        while i < len(diff_lines) and not hunk_re.match(diff_lines[i]):
            line = diff_lines[i]
            i += 1
            if not line:
                continue
            prefix, content = line[0], line[1:]
            if prefix == ' ':
                if orig_pos < len(orig_lines):
                    result.append(orig_lines[orig_pos])
                    orig_pos += 1
            elif prefix == '-':
                orig_pos += 1
            elif prefix == '+':
                result.append(content + '\n')

    while orig_pos < len(orig_lines):
        result.append(orig_lines[orig_pos])
        orig_pos += 1

    return result


def apply_patch(diff: str, file_path: str) -> bool:
    """Применяет unified diff к файлу. Создаёт .cognit.bak перед изменением.
    Поддерживает создание новых файлов (--- /dev/null)."""
    p = Path(file_path)
    is_new = _is_new_file_diff(diff) or not p.exists()

    if is_new:
        orig = ""
    else:
        try:
            orig = p.read_text(encoding='utf-8')
        except Exception as e:
            print(f"❌ Не могу прочитать файл: {e}")
            return False

    try:
        new_lines = _apply_unified_diff(diff, orig.splitlines(keepends=True))
    except Exception as e:
        print(f"❌ Ошибка применения патча: {e}")
        return False

    # Создаём папки если нужно
    p.parent.mkdir(parents=True, exist_ok=True)

    bak = file_path + ".cognit.bak"
    if not is_new:
        Path(bak).write_text(orig, encoding='utf-8')
    try:
        p.write_text(''.join(new_lines), encoding='utf-8')
        if is_new:
            print(f"✅ Файл создан → {file_path}")
        else:
            print(f"✅ Патч применён → {file_path}")
            print(f"   Бэкап: {bak}")
        return True
    except Exception as e:
        print(f"❌ Не могу записать файл: {e}")
        return False
