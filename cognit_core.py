#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_core.py — Common utilities for cognit_transformer.py
=============================================================
Does not load any models. Pure functions for working with:
  - git context (repo, branch, patterns path)
  - file hashing
  - reading pattern metadata
  - CLI helpers (listing, hints)
"""

import os
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

from cognit_i18n import msg

PATTERNS_BASE = "echo_patterns"


# =============================================================================
# GIT CONTEXT
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
    """Returns the path to the patterns directory and creates it."""
    d = f"{PATTERNS_BASE}/{repo}/{branch}"
    os.makedirs(d, exist_ok=True)
    return d


# =============================================================================
# FILE HASHING
# =============================================================================
def file_hash(path: str) -> str:
    """SHA-256 of file head + tail. Fast, sufficient for change detection."""
    h = hashlib.sha256()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h.update(f.read(65536))
        if size > 65536:
            f.seek(max(0, size - 65536))
            h.update(f.read(65536))
    return h.hexdigest()[:16]


# =============================================================================
# PATTERN METADATA
# =============================================================================
def read_meta(patterns_dir: str, name: str) -> dict | None:
    """Reads .json pattern metadata. Returns None if file is missing."""
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
    """Names of all patterns (excluding _internal ones)."""
    return [
        p.stem for p in sorted(Path(patterns_dir).glob("*.pkl"))
        if not p.stem.startswith("_")
    ]


def default_grow_policy(branch: str, source_paths: list[str]) -> str:
    """
    Determines the default pattern update policy.

    retrain — recreated when files change (dialogue is reset):
      - main / master branches
      - patterns from agents/

    grow — accumulates dialogue, not recreated automatically:
      - feature branches (everything else)
    """
    if branch in ("main", "master"):
        return "retrain"
    for p in source_paths:
        if "agents/" in p.replace("\\", "/"):
            return "retrain"
    return "grow"


# =============================================================================
# CLI HELPERS
# =============================================================================
def print_patterns_list(patterns_dir: str):
    """Shows all patterns — one line per pattern."""
    metas = [
        p for p in sorted(Path(patterns_dir).glob("*.json"))
        if not p.stem.startswith("_")
    ]
    if not metas:
        print(msg("info_patterns_list_empty"))
        return

    print(msg("info_patterns_list_count", count=len(metas)))
    for p in metas:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        backend = m.get("backend", "?")[0].upper()   # T / R
        policy  = "~" if m.get("grow_policy") == "grow" else ""
        stale   = "!" if m.get("stale") else ""
        saved   = m.get("saved_at", "")[:10]          # 2026-03-09
        asks    = m.get("n_asks", 0)
        asks_s  = f"  {asks}q" if asks else ""
        print(f"  • {m['name']}{stale}{policy}  [{backend}]  {saved}{asks_s}")


def hint_patterns(patterns_dir: str):
    """One-line pattern names hint when no active pattern is selected."""
    names = list_pattern_names(patterns_dir)
    if names:
        print(msg("info_hint_patterns") + f"  {', '.join(names)}")
    else:
        print(msg("info_patterns_list_empty"))


# =============================================================================
# STALENESS CHECK
# =============================================================================
def check_stale_sources(patterns_dir: str, name: str) -> list[str]:
    """
    Returns a list of changed/deleted source_files for a pattern.
    Empty list = everything is up to date.
    """
    meta = read_meta(patterns_dir, name)
    if not meta:
        return []

    changed = []
    for src in meta.get("source_files", []):
        path = src["path"]
        if not os.path.exists(path):
            changed.append(f"{path}  (not found)")
        elif file_hash(path) != src["hash"]:
            changed.append(path)
    return changed


def mark_stale(patterns_dir: str, name: str, changed_paths: list[str]) -> bool:
    """Mark pattern as stale. Merges changed_paths into stale_files. Returns True if state changed."""
    meta = read_meta(patterns_dir, name)
    if not meta:
        return False
    existing = set(meta.get("stale_files", []))
    merged = existing | set(changed_paths)
    if meta.get("stale") and merged == existing:
        return False
    meta["stale"] = True
    meta["stale_files"] = sorted(merged)
    write_meta(patterns_dir, name, meta)
    return True


def clear_stale(patterns_dir: str, name: str):
    """Remove stale flag from pattern metadata (after retrain)."""
    meta = read_meta(patterns_dir, name)
    if not meta:
        return
    changed = False
    if "stale" in meta:
        del meta["stale"]
        changed = True
    if "stale_files" in meta:
        del meta["stale_files"]
        changed = True
    if changed:
        write_meta(patterns_dir, name, meta)


def is_stale(patterns_dir: str, name: str) -> bool:
    """Quick check for stale flag."""
    meta = read_meta(patterns_dir, name)
    return bool(meta and meta.get("stale"))


# =============================================================================
# ROUTE: LAST ROUTE
# =============================================================================
def save_route(patterns_dir: str, index: str, task: str, files: list[str]):
    """Saves the last route result to _route_last.json."""
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
    """Reads _route_last.json. Returns None if missing or older than 24 hours."""
    route_path = Path(patterns_dir) / "_route_last.json"
    if not route_path.exists():
        return None
    with open(route_path, encoding="utf-8") as f:
        data = json.load(f)

    # Ignore routes older than 24 hours
    try:
        routed_at = datetime.fromisoformat(data["routed_at"])
        age_hours = (datetime.now() - routed_at).total_seconds() / 3600
        if age_hours > 24:
            return None
    except Exception:
        return None

    return data


# =============================================================================
# FILE COLLECTION FOR /load (respects .gitignore)
# =============================================================================

# Text file extensions that are meaningful for the model to read
TEXT_EXTENSIONS = {
    # Code
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".vue", ".svelte", ".sh", ".bash", ".ps1",
    # Data / config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
    # Documentation
    ".md", ".txt", ".rst",
}


def collect_text_files(dir_path: str | Path,
                       exclude_dirs: set[str] | None = None) -> list[Path]:
    """
    Returns text files from a directory.

    If inside a git repo — uses `git ls-files` (automatically
    respects .gitignore and returns only tracked files).
    Otherwise — rglob filtered by TEXT_EXTENSIONS.

    exclude_dirs: directory names to exclude (e.g. {"agents"}).
    """
    p = Path(dir_path).resolve()
    excl = exclude_dirs or set()

    def _keep(f: Path) -> bool:
        if f.suffix.lower() not in TEXT_EXTENSIONS:
            return False
        # Exclude files inside forbidden directories
        return not any(part in excl for part in f.relative_to(p).parts[:-1])

    # Try git ls-files
    try:
        result = subprocess.run(
            ["git", "-C", str(p), "ls-files", "--cached", "."],
            capture_output=True, text=True, check=True
        )
        files = []
        for line in result.stdout.splitlines():
            f = (p / line).resolve()
            if f.is_file() and _keep(f):
                files.append(f)
        if files:
            return sorted(files)
    except Exception:
        pass

    # Fallback: rglob without git
    return sorted(f for f in p.rglob("*") if f.is_file() and _keep(f))
