#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_hook.py — lightweight post-commit hook handler
======================================================
Called by the git post-commit hook. Does NOT load any LLM models.

What it does:
  1. Detects changed files from the last commit
  2. Marks affected KV-cache patterns as stale (metadata only)
  3. Incrementally updates the BM25/tree-sitter index
  4. Patterns are lazily retrained on next interactive /ask
"""

import os
import json
import subprocess
from pathlib import Path

import cognit_core as core
from cognit_i18n import msg, set_lang


ECHO_CONFIG = ".echo.json"


def _read_config() -> dict:
    """Read .echo.json from the Cognit directory."""
    p = Path(ECHO_CONFIG)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _changed_files(client_project: str) -> list[str]:
    """Get list of files changed in the last commit (absolute paths)."""
    client = Path(client_project).resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(client), "diff", "HEAD~1", "HEAD", "--name-only"],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError:
        # First commit — no HEAD~1
        try:
            result = subprocess.run(
                ["git", "-C", str(client), "show", "--name-only", "--pretty=", "HEAD"],
                capture_output=True, text=True, check=True
            )
        except Exception:
            return []

    files = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line:
            files.append(str((client / line).resolve()))
    return files


def _mark_stale_patterns(patterns_dir: str, changed_files: list[str]) -> int:
    """Scan patterns and mark those affected by changed files as stale."""
    pdir = Path(patterns_dir)
    if not pdir.exists():
        return 0

    marked = 0
    for meta_path in sorted(pdir.glob("*.json")):
        if meta_path.stem.startswith("_"):
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        sources = meta.get("source_files", [])
        if not sources:
            continue

        # Find which source files have changed
        stale_paths = []
        for src in sources:
            stored_path = str(Path(src["path"]).resolve())
            if stored_path not in changed_files:
                continue
            # Verify hash actually differs
            if os.path.exists(src["path"]) and core.file_hash(src["path"]) != src["hash"]:
                stale_paths.append(src["path"])

        if stale_paths and core.mark_stale(patterns_dir, meta["name"], stale_paths):
            marked += 1

    return marked


def _update_index(client_project: str, cache_dir: str):
    """Incrementally rebuild the BM25/tree-sitter index."""
    try:
        from cognit_index import CodeIndex
    except ImportError:
        return  # tree-sitter not available
    idx = CodeIndex(client_project, cache_dir=cache_dir)
    idx.build()


def post_commit_hook():
    """Main entry point for the post-commit hook."""
    config = _read_config()
    if not config:
        return

    set_lang(config.get("lang", "en"))

    client_project = config.get("client_project", "")
    if not client_project or not Path(client_project).exists():
        return

    patterns_base = config.get("patterns_dir", core.PATTERNS_BASE)

    # Determine repo/branch for patterns path
    orig_dir = os.getcwd()
    try:
        os.chdir(client_project)
        repo = core.git_repo_name()
        branch = core.git_branch()
    finally:
        os.chdir(orig_dir)

    patterns_dir = f"{patterns_base}/{repo}/{branch}"
    if not Path(patterns_dir).exists():
        print(msg("hook_no_patterns_dir"))
        return

    # 1. Get changed files
    changed = _changed_files(client_project)
    if not changed:
        return

    print(msg("hook_checking", count=len(changed)))

    # 2. Mark affected patterns as stale
    marked = _mark_stale_patterns(patterns_dir, changed)
    if marked:
        print(msg("hook_marked_stale", count=marked))
    else:
        print(msg("hook_all_current"))

    # 3. Incrementally update BM25 index
    _update_index(client_project, patterns_dir)
    print(msg("hook_index_updated"))


if __name__ == "__main__":
    post_commit_hook()
