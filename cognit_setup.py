#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_setup.py — one-time Cognit setup for a project
======================================================
Usage:
    python cognit_setup.py

What it does:
    1. Creates .echo.json  — locks the model and backend for the team
    2. Adds echo_patterns/ to .gitignore
    3. Installs post-commit hook (optional)
    4. Shows S3 commands for pattern sync
"""

import os
import sys
import json
import subprocess
from pathlib import Path

from cognit_i18n import msg, set_lang, agent_templates

ECHO_CONFIG   = ".echo.json"
GITIGNORE     = ".gitignore"
PATTERNS_BASE = "echo_patterns"


# =============================================================================
# HELPERS
# =============================================================================
def _git_root() -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True
        )
        return Path(out.stdout.strip())
    except Exception:
        return None


def _git_repo_name(root: Path) -> str:
    return root.name


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{prompt}{suffix}: ").strip().lower()
    if not val:
        return default
    return val == "y"


def _ask_client_project() -> Path | None:
    """Ask for the path to the main (client) project."""
    print(msg("setup_header"))
    print(msg("setup_project_intro1"))
    print(msg("setup_project_intro2"))

    # Use saved path as default
    existing_client = ""
    if Path(ECHO_CONFIG).exists():
        try:
            with open(ECHO_CONFIG, encoding="utf-8") as f:
                existing_client = json.load(f).get("client_project", "")
        except Exception:
            pass

    raw = _ask(msg("setup_ask_project_path"), existing_client)
    if not raw:
        return None

    p = Path(raw).expanduser().resolve()
    if not p.exists():
        print(msg("setup_path_not_found", path=p))
        return None
    if not (p / ".git").exists():
        print(msg("setup_not_git_repo", path=p))
        return None

    print(msg("setup_project_name", name=p.name, path=p))
    return p


# =============================================================================
# STEP 1: .echo.json
# =============================================================================
def setup_config():
    print(msg("setup_config_header"))

    existing = {}
    if Path(ECHO_CONFIG).exists():
        with open(ECHO_CONFIG, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(msg("setup_found_existing", file=ECHO_CONFIG))

    ex_t = existing.get("transformer", {})

    t_model_path = _ask(
        msg("setup_ask_model_path"),
        ex_t.get("model_path", "models/Qwen/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    )

    lang = _ask("   Language / Язык (en/ru)", existing.get("lang", "en"))
    if lang not in ("en", "ru"):
        lang = "en"
    set_lang(lang)

    config = {
        "backend":      "transformer",
        "patterns_dir": PATTERNS_BASE,
        "lang":         lang,
        "transformer": {
            "model_path":   t_model_path,
            "n_gpu_layers": ex_t.get("n_gpu_layers", -1),
            "n_ctx":        ex_t.get("n_ctx", 8192),
            "max_tokens":   ex_t.get("max_tokens", 512),
        },
        "_note": (
            "All team members must use the same model. "
            "Patterns are incompatible between different models."
        ),
    }

    with open(ECHO_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(msg("setup_config_created", file=ECHO_CONFIG))
    return config


# =============================================================================
# STEP 2: .gitignore
# =============================================================================
def setup_gitignore():
    print(msg("setup_gitignore_header"))

    lines_to_add = [
        "# Cognit — patterns are not versioned, stored locally or in S3",
        f"{PATTERNS_BASE}/",
    ]

    existing_lines = []
    if Path(GITIGNORE).exists():
        existing_lines = Path(GITIGNORE).read_text(encoding="utf-8").splitlines()

    to_add = [l for l in lines_to_add if l not in existing_lines]
    if not to_add:
        print(msg("setup_gitignore_ok", file=GITIGNORE))
        return

    with open(GITIGNORE, "a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(to_add) + "\n")

    for line in to_add:
        if not line.startswith("#"):
            print(msg("setup_gitignore_added", file=GITIGNORE, line=line))


# =============================================================================
# STEP 3: post-commit hook
# =============================================================================
def _make_hook_script(cognit_dir: str) -> str:
    """Generate lightweight hook script. No model loading — just marks stale + updates index."""
    # Use forward slashes — Git sh on Windows understands them
    posix_dir = Path(cognit_dir).as_posix()
    return f"""\
#!/bin/sh
# Cognit — post-commit hook (lightweight, no model loading)
# cognit dir: {posix_dir}

cd "{posix_dir}" && python cognit_hook.py 2>/dev/null
"""


def setup_hook(target_git_root: Path, cognit_dir: Path):
    """Install post-commit hook into the specified git repository."""
    hook_path = target_git_root / ".git" / "hooks" / "post-commit"

    if hook_path.exists():
        overwrite = _ask_yn(
            msg("setup_hook_exists", name=target_git_root.name),
            default=False,
        )
        if not overwrite:
            print(msg("setup_skip"))
            return

    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(_make_hook_script(str(cognit_dir)), encoding="utf-8")

    try:
        import stat
        st = os.stat(hook_path)
        os.chmod(hook_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    print(msg("setup_hook_installed", path=hook_path))


# =============================================================================
# STEP 4: S3 instructions
# =============================================================================
def show_s3_instructions(config: dict, repo_name: str):
    print(msg("setup_s3_header"))
    bucket = "your-bucket"
    prefix = "echo-patterns"

    print(f"""
   Pattern path structure:
     Local:  {PATTERNS_BASE}/<repo>/<branch>/<name>.pkl
     S3:     s3://{bucket}/{prefix}/<repo>/<branch>/<name>.pkl

   Upload current branch patterns to S3:
     aws s3 sync {PATTERNS_BASE}/{repo_name}/ \\
         s3://{bucket}/{prefix}/{repo_name}/

   Download patterns from S3:
     aws s3 sync s3://{bucket}/{prefix}/{repo_name}/ \\
         {PATTERNS_BASE}/{repo_name}/

   Download specific branch:
     aws s3 sync s3://{bucket}/{prefix}/{repo_name}/main/ \\
         {PATTERNS_BASE}/{repo_name}/main/

   Each pattern metadata (.json) contains:
     backend, model, repo, branch, source_files
   → You can pick the right file by these fields without downloading .pkl
""")


# =============================================================================
# STEP 5: agents/ — project knowledge
# =============================================================================

def setup_agents(target_dir: Path | None = None):
    """Create agents/ folder in the client project (or CWD if not specified)."""
    print(msg("setup_agents_header"))

    base = target_dir if target_dir else Path(".")
    agents_dir = base / "agents"

    if target_dir:
        print(msg("setup_creating_in", dir=target_dir))

    created = []

    for rel_path, content in agent_templates().items():
        target = agents_dir / rel_path
        if target.exists():
            print(msg("setup_file_exists", path=rel_path))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        created.append(rel_path)
        print(msg("setup_file_created", path=target))

    if not created:
        print(msg("setup_all_exist"))
        return

    agents_path = agents_dir.resolve()
    print(f"""
   agents/ structure:
     {agents_path}/
       style/global.md    — code style (default: Python best practices)
       style/commands.md  — CLI output style (default: filled in)
       arch/overview.md   — architecture (replace with your structure)
       context/project.md — goals, decisions, constraints (replace)

   Cognit on next run will automatically:
     1. Load agents from these folders and create KV-cache patterns
     2. Connect agents to pipeline and ambient mode

   You can run immediately — default files are already functional.
   Best results — after filling arch/ and context/ for your project.
""")

    print(msg("setup_agents_tip"))
    print(f"   cd {base.resolve()} && git add agents/ && git commit -m 'Add agent knowledge base'")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import sys

    # Load language from .echo.json if available
    if Path(ECHO_CONFIG).exists():
        try:
            with open(ECHO_CONFIG, encoding="utf-8") as f:
                _cfg = json.load(f)
            set_lang(_cfg.get("lang", "en"))
        except Exception:
            pass

    # Subcommands: python cognit_setup.py agents
    if len(sys.argv) > 1:
        subcmd = sys.argv[1].lower()
        if subcmd == "agents":
            print(msg("setup_banner_agents"))
            # Read client_project from .echo.json if available
            client = None
            if Path(ECHO_CONFIG).exists():
                try:
                    cp = json.loads(Path(ECHO_CONFIG).read_text(encoding="utf-8")).get("client_project", "")
                    if cp and Path(cp).exists():
                        client = Path(cp)
                except Exception:
                    pass
            setup_agents(client)
            return
        else:
            print(msg("setup_unknown_cmd", cmd=subcmd))
            print(msg("setup_usage"))
            sys.exit(1)

    print(msg("setup_banner_main"))

    git_root = _git_root()
    if git_root:
        repo_name = _git_repo_name(git_root)
        print(msg("setup_git_repo", name=repo_name, path=git_root))
    else:
        repo_name = "local"
        print(msg("setup_no_git"))

    config = setup_config()
    setup_gitignore()

    # Absolute path to Cognit directory (for the hook)
    cognit_dir = Path(__file__).parent.resolve()

    # Hook in the main (client) project
    client_root = _ask_client_project()
    if client_root:
        if _ask_yn(msg("setup_ask_hook_client", name=client_root.name), default=True):
            setup_hook(client_root, cognit_dir)
        # Save client project path to .echo.json
        config["client_project"] = str(client_root)
        with open(ECHO_CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(msg("setup_client_saved", file=ECHO_CONFIG))

    # Hook in Cognit repo itself (optional)
    if git_root and _ask_yn(msg("setup_ask_hook_cognit"), default=False):
        setup_hook(git_root, cognit_dir)

    if _ask_yn(msg("setup_ask_agents"), default=True):
        setup_agents(client_root)  # None if no client → creates in CWD

    # Create pipeline.json in client project if it doesn't exist
    if client_root:
        try:
            from cognit_pipeline import save_default_pipeline
            save_default_pipeline(str(client_root))
        except ImportError:
            pass

    show_s3_instructions(config, repo_name)

    print(msg("setup_done"))


if __name__ == "__main__":
    main()
