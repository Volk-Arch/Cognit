#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_patch.py — applying unified diff to files.
No LLM dependencies.
"""

import re
from pathlib import Path

from cognit_i18n import msg


def _extract_diff(text: str) -> str | None:
    """Extracts unified diff from model response (```diff block or raw diff)."""
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
    """Extracts all unified diff blocks from model response."""
    diffs = re.findall(r'```diff\s*\n(.*?)```', text, re.DOTALL)
    if diffs:
        return [d.strip() for d in diffs if d.strip()]
    # Fallback: single diff in raw format
    single = _extract_diff(text)
    return [single] if single else []


def _diff_target(diff: str) -> str | None:
    """Extracts the target file path from the +++ diff header."""
    m = re.search(r'^\+\+\+[ \t]+(?:b/)?(.+?)(?:[ \t]+\d{4}|$)', diff, re.MULTILINE)
    if m:
        path = m.group(1).strip()
        return None if path == '/dev/null' else path
    return None


def _is_new_file_diff(diff: str) -> bool:
    """Returns True if the diff creates a new file (--- /dev/null)."""
    return bool(re.search(r'^---[ \t]+/dev/null', diff, re.MULTILINE))


def _apply_unified_diff(diff_text: str, orig_lines: list[str]) -> list[str]:
    """Pure Python implementation of unified diff application."""
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
    """Applies unified diff to a file. Creates .cognit.bak before modification.
    Supports creating new files (--- /dev/null)."""
    p = Path(file_path)
    is_new = _is_new_file_diff(diff) or not p.exists()

    if is_new:
        orig = ""
    else:
        try:
            orig = p.read_text(encoding='utf-8')
        except Exception as e:
            print(msg("err_cannot_read_file", error=e))
            return False

    try:
        new_lines = _apply_unified_diff(diff, orig.splitlines(keepends=True))
    except Exception as e:
        print(msg("err_patch_failed", path=file_path, error=e))
        return False

    # Create directories if needed
    p.parent.mkdir(parents=True, exist_ok=True)

    bak = file_path + ".cognit.bak"
    if not is_new:
        Path(bak).write_text(orig, encoding='utf-8')
    try:
        p.write_text(''.join(new_lines), encoding='utf-8')
        if is_new:
            print(msg("ok_file_created", path=file_path))
        else:
            print(msg("ok_patch_applied", path=file_path))
            print(msg("info_backup", path=bak))
        return True
    except Exception as e:
        print(msg("err_patch_failed", path=file_path, error=e))
        return False
