#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_pipeline.py — configurable agent pipeline.

The pipeline is described in pipeline.json in the client project root.
Each stage receives the accumulated context from previous stages and adds its own memo.
The final stage of type "coder" writes unified diff.

Stage types:
  navigator — navigation via tree-sitter index (memo is inserted automatically)
  agent     — KV-cache agent writes its own memo
  coder     — Transformer builds context pattern and generates diff
  reviewer  — checks coder's diff against tree-sitter structure, removes duplicates

Top-level parameters:
  passes — how many times to run agents (default 1).
           passes=2: agents see each other's memos in the second run -> refinement.
           Navigator, coder and reviewer always run once regardless of passes.

Example pipeline.json:
  {
    "passes": 2,
    "stages": [
      {"id": "navigator", "type": "navigator", "enabled": true},
      {"id": "analyst",   "type": "agent", "name": "analyst", "enabled": true},
      {"id": "context",   "type": "agent", "name": "context", "enabled": true},
      {"id": "arch",      "type": "agent", "name": "arch",    "enabled": true},
      {"id": "style",     "type": "agent", "name": "style",   "enabled": true},
      {"id": "coder",     "type": "coder",                    "enabled": true}
    ]
  }
"""

import json
from pathlib import Path

from cognit_i18n import msg, agent_role

PIPELINE_FILENAME = "pipeline.json"

# Default stage order and composition
DEFAULT_PIPELINE: dict = {
    "passes": 1,    # 1 = single run; 2 = double run of agents
    "stages": [
        {
            "id":      "navigator",
            "type":    "navigator",
            "enabled": True,
            "comment": "Memo from tree-sitter navigator: which files, where to look"
        },
        {
            "id":      "analyst",
            "type":    "agent",
            "name":    "analyst",
            "enabled": True,
            "comment": "Analyst: reviews code + task -> concrete change plan"
        },
        {
            "id":      "context",
            "type":    "agent",
            "name":    "context",
            "enabled": True,
            "comment": "Agent: project context, business requirements"
        },
        {
            "id":      "arch",
            "type":    "agent",
            "name":    "arch",
            "enabled": True,
            "comment": "Agent: architecture, dependencies, patterns"
        },
        {
            "id":      "style",
            "type":    "agent",
            "name":    "style",
            "enabled": True,
            "comment": "Agent: style, formatting, naming"
        },
        {
            "id":      "coder",
            "type":    "coder",
            "enabled": True,
            "comment": "Final stage: coder writes diff using full context"
        },
        {
            "id":      "reviewer",
            "type":    "reviewer",
            "enabled": True,
            "comment": "Review: checks diff against tree-sitter structure, removes duplicates"
        }
    ]
}


def load_pipeline(client_project: str) -> dict:
    """
    Loads pipeline.json from the client project.
    If file is missing — returns DEFAULT_PIPELINE.
    Adds missing roles from default; preserves passes from file.
    """
    if not client_project:
        return _with_roles(DEFAULT_PIPELINE)
    p = Path(client_project) / PIPELINE_FILENAME
    if p.exists():
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
            # Fill missing roles from i18n (language-aware)
            for stage in cfg.get("stages", []):
                if "role" not in stage:
                    sid = stage.get("name", stage.get("id", ""))
                    role = agent_role(sid)
                    if role:
                        stage["role"] = role
                # Migration: rwkv -> navigator
                if stage.get("type") == "rwkv":
                    stage["type"] = "navigator"
            # passes defaults to 1 if not specified
            cfg.setdefault("passes", 1)
            return cfg
        except Exception as e:
            print(msg("warn_pipeline_error", error=e))
    return _with_roles(DEFAULT_PIPELINE)


def _with_roles(pipeline: dict) -> dict:
    """Inject language-aware roles from cognit_i18n into stages missing a role."""
    import copy
    result = copy.deepcopy(pipeline)
    for stage in result.get("stages", []):
        if "role" not in stage:
            sid = stage.get("name", stage.get("id", ""))
            role = agent_role(sid)
            if role:
                stage["role"] = role
    return result


def save_default_pipeline(client_project: str) -> Path | None:
    """Writes pipeline.json to the client project if it doesn't exist."""
    if not client_project:
        return None
    p = Path(client_project) / PIPELINE_FILENAME
    if p.exists():
        return p  # do not overwrite existing
    p.write_text(json.dumps(DEFAULT_PIPELINE, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print(msg("ok_pipeline_created", path=p))
    return p


def describe_pipeline(pipeline: dict) -> str:
    """Returns a human-readable description of stages (and pass count if > 1)."""
    passes = pipeline.get("passes", 1)
    lines  = []
    for s in pipeline.get("stages", []):
        enabled = "✓" if s.get("enabled", True) else "✗"
        stype   = s.get("type", "?")
        name    = s.get("name", s.get("id", ""))
        # agent stages show x passes if > 1
        repeat  = f" ×{passes}" if passes > 1 and stype == "agent" else ""
        lines.append(f"  {enabled} [{stype:6}] {name}{repeat}")
    return "\n".join(lines)
