#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""cognit_transformer.py — Transformer backend (Qwen2.5-Coder). KV-cache → .pkl."""

import os
import re
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import cognit_core as core
from cognit_patch import (_extract_diff, _extract_all_diffs, _diff_target,
                          _is_new_file_diff, apply_patch as _apply_patch)
from cognit_agents import echo_config as _echo_config, list_agents as _list_agents, read_agent_text as _read_agent_text
from cognit_i18n import msg, set_lang, HELP as HELP_TEXTS
try:
    from cognit_index import CodeIndex, SearchResult as _SearchResult
    _HAS_INDEX = True
except ImportError:
    CodeIndex = None
    _SearchResult = None
    _HAS_INDEX = False

# =============================================================================
# CONFIGURATION — read from .echo.json (key "transformer"), defaults below
# =============================================================================
_c           = _echo_config().get("transformer", {})
MODEL_PATH   = _c.get("model_path",   "models/Qwen/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
MODEL_NAME   = Path(MODEL_PATH).stem
N_CTX        = _c.get("n_ctx",        8192)
N_GPU_LAYERS = _c.get("n_gpu_layers", -1)
MAX_TOKENS   = _c.get("max_tokens",   512)

set_lang(_echo_config().get("lang", "en"))

REPO_NAME    = core.git_repo_name()
BRANCH_NAME  = core.git_branch()
PATTERNS_DIR = core.make_patterns_dir(REPO_NAME, BRANCH_NAME)

# ChatML stop tokens
# </think> is NOT included: some models output <think></think> before the answer
STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]

# Stop patterns for the answer: model starts playing other roles → stop
# Checked ONLY after answer_started
# </think> is NOT included: it appears in recent_text right after answer_started → false stop
ANSWER_STOP_PATTERNS = ["Human:", "User:", "\nHuman:", "\nUser:"]

# =============================================================================
# INITIALIZATION
# =============================================================================
try:
    from llama_cpp import Llama
except ImportError:
    print(msg("err_llama_not_installed"))
    print("   pip install llama-cpp-python --extra-index-url "
          "https://abetlen.github.io/llama-cpp-python/whl/cu121")
    sys.exit(1)

llm = None  # initialized via init_model()
os.makedirs(PATTERNS_DIR, exist_ok=True)


def init_model():
    """Load Transformer model into VRAM."""
    global llm
    if not os.path.exists(MODEL_PATH):
        print(msg("err_model_not_found", path=MODEL_PATH))
        print(msg("err_set_model_path"))
        sys.exit(1)
    print(msg("status_loading_model", path=MODEL_PATH))
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    print(msg("ok_transformer_loaded", device='GPU' if N_GPU_LAYERS != 0 else 'CPU'))
    print(msg("info_patterns_dir", dir=PATTERNS_DIR, repo=REPO_NAME, branch=BRANCH_NAME))


def unload_model():
    """Unload model from VRAM."""
    global llm
    if llm is not None:
        del llm
        llm = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass

# =============================================================================
# PATTERN OPERATIONS
# =============================================================================
def _check_and_refresh(name: str):
    """Check pattern freshness. If files changed — offer to recreate."""
    meta = core.read_meta(PATTERNS_DIR, name)
    if not meta:
        return

    # Combine pre-computed stale_files (from hook) with live check
    stale_files = set(meta.get("stale_files", []))
    live_changed = core.check_stale_sources(PATTERNS_DIR, name)
    all_changed = sorted(stale_files | set(live_changed))

    if not all_changed:
        return

    print(msg("warn_files_changed", name=name))
    for c in all_changed:
        print(f"    • {c}")

    try:
        ans = input(msg("prompt_recreate")).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if ans != "y":
        print(msg("info_continue_old"))
        return

    texts, paths = [], []
    for src in meta.get("source_files", []):
        p = src["path"]
        if os.path.exists(p):
            texts.append(Path(p).read_text(encoding="utf-8"))
            paths.append(p)

    if texts:
        save_pattern(name, "\n\n".join(texts), source_files=paths)
        core.clear_stale(PATTERNS_DIR, name)


def _context_prompt(text: str) -> str:
    """Build a prompt with context in ChatML format."""
    return (
        "<|im_start|>system\n"
        + msg("system_prompt_context") + "\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        + msg("user_load_context") + f"\n\n{text.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        + msg("assistant_context_loaded") + "\n"
        "<|im_end|>\n"
    )


def _question_prompt(question: str) -> str:
    """Append a question to the already loaded KV-cache."""
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def save_pattern(name: str, text: str, source_files: list[str] = None, grow_policy: str = None):
    """
    Run text through LLM, save KV-cache.
    After this the model 'knows' the text without needing to read it again.

    grow_policy: 'grow' | 'retrain' | None (auto by branch/path)
    """
    if grow_policy is None:
        grow_policy = core.default_grow_policy(BRANCH_NAME, source_files or [])
    print(msg("status_creating_pattern", name=name, policy=grow_policy))

    prompt = _context_prompt(text)
    tokens = llm.tokenize(prompt.encode("utf-8"), special=True)

    if len(tokens) > N_CTX - 64:
        print(msg("warn_text_too_long", n_tokens=len(tokens), limit=N_CTX - 64))
        tokens = tokens[:N_CTX - 64]

    print(msg("info_token_count", count=len(tokens)))
    print(msg("status_processing"))

    llm.reset()
    llm.eval(tokens)
    state = llm.save_state()

    # Save state
    pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
    meta_path    = Path(PATTERNS_DIR) / f"{name}.json"

    with open(pattern_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = pattern_path.stat().st_size / 1024
    meta = {
        "name":         name,
        "backend":      "transformer",
        "model":        MODEL_NAME,
        "repo":         REPO_NAME,
        "branch":       BRANCH_NAME,
        "grow_policy":  grow_policy,
        "n_tokens":     len(tokens),
        "size_kb":      round(size_kb, 1),
        "preview":      text[:300],
        "saved_at":     datetime.now().isoformat(),
        "n_asks":       0,
        "source_files": [{"path": p, "hash": core.file_hash(p)} for p in (source_files or [])],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(msg("ok_pattern_saved", name=name, size_kb=size_kb, n_tokens=llm.n_tokens))


def load_pattern(name: str) -> bool:
    """Restore KV-cache from file."""
    pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
    if not pattern_path.exists():
        print(msg("err_pattern_not_found", name=name))
        return False

    # Model compatibility check: KV-cache from a different quantization = garbage
    meta = core.read_meta(PATTERNS_DIR, name)
    if meta and meta.get("model") and meta["model"] != MODEL_NAME:
        print(msg("warn_model_mismatch", name=name, model=meta['model'])
              + f"current model {MODEL_NAME}")
        print(msg("info_recreate_pattern", name=name))
        return False

    with open(pattern_path, "rb") as f:
        state = pickle.load(f)

    llm.load_state(state)

    # load_state() restores KV-cache, but n_tokens (write position)
    # may not be restored — depends on llama-cpp-python version.
    if meta and "n_tokens" in meta:
        llm.n_tokens = meta["n_tokens"]

    return True


def ask_pattern(name: str, question: str, grow: bool = True) -> str:
    """
    Load KV-cache and answer a question.
    Context text is NOT sent again — only the question.

    grow=True (default): after answering, KV-cache is updated —
    pattern accumulates the entire dialogue (growing session).
    """
    print(f"💬 [{name}]", end="  ", flush=True)
    _check_and_refresh(name)
    if not load_pattern(name):
        return

    # Append only the question — context is already in KV-cache
    question_prompt = _question_prompt(question)
    q_tokens = llm.tokenize(question_prompt.encode("utf-8"), add_bos=False, special=True)
    print(f"(KV: {llm.n_tokens})", end="  ", flush=True)

    print(f"+{len(q_tokens)} tok")
    print(f"\n❓ {question}")
    print("─" * 50)

    # reset=False — do not reset loaded KV-cache!
    # Some models output <think>...</think> before the answer — hide this block
    collected = []
    answer_started = False
    think_dots = 0  # dot indicator counter during think phase
    junk_streak = 0  # consecutive "garbage" tokens

    for token_id in llm.generate(q_tokens, reset=False, temp=0.3, top_p=0.9, repeat_penalty=1.1):
        piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        collected.append(piece)
        full = "".join(collected)

        # Strip stop tokens from the piece before output
        clean_piece = piece
        for stop in STOP_TOKENS:
            if stop in clean_piece:
                clean_piece = clean_piece[:clean_piece.index(stop)]

        if not answer_started:
            if "</think>" in full:
                # Erase dot indicator
                if think_dots:
                    print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)
                after = full.split("</think>", 1)[1]
                for stop in STOP_TOKENS:
                    if stop in after:
                        after = after[:after.index(stop)]
                if after.strip():
                    print(after, end="", flush=True)
                answer_started = True
            elif not any(("<|im" in full, "<think>" in full)):
                # Model answers without <think> — output immediately
                print(clean_piece, end="", flush=True)
                answer_started = True
            elif "<think>" in full and len(collected) % 12 == 0:
                # Progress indicator during think phase
                think_dots += 1
                print(".", end="", flush=True)
        else:
            print(clean_piece, end="", flush=True)

        tail = full[-60:]
        if any(stop in tail for stop in STOP_TOKENS):
            break
        if len(collected) >= MAX_TOKENS:
            break
        # Fake turn detector: model started playing as Human/User
        if answer_started:
            recent_text = "".join(collected[-10:])
            if any(p in recent_text for p in ANSWER_STOP_PATTERNS):
                print(msg("warn_fake_turn"))
                break
            # Duplicate code block detector: second ```diff/```python → stop
            fence_count = full.count("```")
            if fence_count >= 4:  # 2 blocks = 4 fence markers
                print(msg("warn_dup_code_block"))
                break
        # Infinite loop detector: multiple window sizes
        _loop_found = False
        for win in (30, 60, 90):
            if len(collected) >= win * 2:
                recent = "".join(collected[-win:])
                before = "".join(collected[-win * 2:-win])
                if recent == before:
                    _loop_found = True
                    break
        if _loop_found:
            if not answer_started and think_dots:
                print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)
            print(msg("warn_infinite_loop"))
            break
        # Think phase limit: >400 tokens without </think> → stop
        if not answer_started and len(collected) > 400:
            if think_dots:
                print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)
            print(msg("warn_think_too_long"))
            break
        # Garbage token detector: `, spaces, \n in a row → stop
        if piece.strip("`\n\r \t") == "":
            junk_streak += 1
            if junk_streak >= 6:
                print(msg("warn_garbage"))
                break
        else:
            junk_streak = 0

    # Erase dots if think did not finish
    if think_dots and not answer_started:
        print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)

    # If model answered without think block — output everything
    if not answer_started:
        print("".join(collected), end="", flush=True)

    print("\n" + "─" * 50)

    # Growing session: save updated KV-cache back to pattern
    if grow:
        state = llm.save_state()
        pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
        meta_path    = Path(PATTERNS_DIR) / f"{name}.json"

        with open(pattern_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {"name": name, "backend": "transformer", "preview": question[:300]}

        meta["n_tokens"]   = llm.n_tokens
        meta["size_kb"]    = round(pattern_path.stat().st_size / 1024, 1)
        meta["updated_at"] = datetime.now().isoformat()
        meta.setdefault("n_asks", 0)
        meta["n_asks"] += 1

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"💾 {name}  ({meta['size_kb']} KB, {meta['n_asks']}q)")

    # Clean up response: remove think block and stop tokens
    response = "".join(collected)
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    for stop in STOP_TOKENS + ANSWER_STOP_PATTERNS:
        if stop in response:
            response = response[:response.index(stop)]
    response = response.strip()

    return response


def _extract_line_range(memo: str, filename: str) -> tuple[int, int] | None:
    """
    Parse line range from navigation memo for a specific file.
    Looks for patterns: 'строки 42-67', 'lines 42-67', ':42-67'.
    First searches near the filename, then across the entire memo.
    """
    patterns = [
        r'стр(?:оки?)?\s+(\d+)\s*[-–—]\s*(\d+)',   # строки 42-67, стр. 42-67
        r'lines?\s+(\d+)\s*[-–—]\s*(\d+)',          # lines 42-67, line 42-67
        r':(\d+)\s*[-–—]\s*(\d+)',                  # :42-67
    ]
    stem = Path(filename).stem.lower()
    fname_pos = memo.lower().find(stem)
    zones = []
    if fname_pos >= 0:
        zones.append(memo[max(0, fname_pos - 10): fname_pos + 200])
    zones.append(memo)

    for zone in zones:
        for pat in patterns:
            m = re.search(pat, zone, re.IGNORECASE)
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


def _focused_file_content(filepath: str, memo: str, context_lines: int = 15) -> str:
    """
    Return a focused file fragment if navigation memo contains line numbers.
    Fragment = specified lines +/- context_lines, with line numbers.
    If no line numbers — full file up to FILE_LIMIT characters.
    """
    FILE_LIMIT = 8000
    try:
        raw = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

    lines = raw.splitlines()
    rng   = _extract_line_range(memo, Path(filepath).name)

    if rng:
        start_line, end_line = rng
        start_idx = max(0, start_line - 1 - context_lines)   # 0-indexed
        end_idx   = min(len(lines), end_line + context_lines)

        parts = []
        if start_idx > 0:
            parts.append(f"... (lines 1–{start_idx} skipped)")
        for i in range(start_idx, end_idx):
            parts.append(f"{i + 1}: {lines[i]}")
        if end_idx < len(lines):
            parts.append(f"... (lines {end_idx + 1}–{len(lines)} skipped)")

        excerpt = "\n".join(parts)
        print(msg("info_code_fragment", start=start_line, end=end_line)
              + f"(±{context_lines}) of {len(lines)}")
        return excerpt[:FILE_LIMIT]

    # No range — full file
    if len(raw) > FILE_LIMIT:
        return raw[:FILE_LIMIT] + "\n... (truncated)"
    return raw


def _run_pipeline(task: str, files: list[str], nav_memo: str = "") -> str:
    """
    Sequential agent pipeline with accumulated context.

    Stage order is taken from pipeline.json of the client project.
    Each stage receives all accumulated context and adds its own memo.
    Supports passes > 1: agents run N times, each subsequent pass
    sees memos from all previous ones → refinement.
    Reasoning log is written to _pipeline_log.md in the client project.
    Final stage 'coder' generates unified diff.

    nav_memo: context note from tree-sitter navigator (search_summary).
    Returns coder response (diff).
    """
    from cognit_pipeline import load_pipeline, describe_pipeline

    cfg        = _echo_config() or {}
    client_dir = cfg.get("client_project", "")
    pipeline   = load_pipeline(client_dir)
    stages     = [s for s in pipeline.get("stages", []) if s.get("enabled", True)]
    passes     = max(1, int(pipeline.get("passes", 1)))

    # ── Reasoning log ────────────────────────────────────────────────────────
    log_path: Path | None = Path(client_dir) / "_pipeline_log.md" if client_dir else None
    if log_path:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        log_path.write_text(
            f"# Pipeline: {task[:80]}\n_{ts}_\n\n",
            encoding="utf-8",
        )

    def _log(section: str, content: str) -> None:
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"## {section}\n{content.strip()}\n\n")

    # ── Pipeline description ─────────────────────────────────────────────────
    suffix = f", {passes} passes" if passes > 1 else ""
    print(msg("info_pipeline_stages", n=len(stages), suffix=suffix))
    print(describe_pipeline(pipeline))
    print("─" * 50)

    # Remove previous temp pattern if any
    for ext in (".pkl", ".json"):
        p = Path(PATTERNS_DIR) / f"_pipeline{ext}"
        if p.exists():
            p.unlink()

    # ── Shared context: task + files (focused by memo) ───────────────────────
    shared = f"## Task\n{task}\n\n"
    _log("Task", task)

    for fpath in files:
        try:
            content = _focused_file_content(fpath, nav_memo)
            fname   = Path(fpath).name
            shared += f"## File: {fname}\n```\n{content}\n```\n\n"
            _log(f"File: {fname}", f"```\n{content}\n```")
        except Exception as e:
            print(f"   ⚠️  {fpath}: {e}")

    # ── Navigation (tree-sitter) — once at the beginning ─────────────────────
    nav_stage = next((s for s in stages if s.get("type") in ("navigator", "rwkv")), None)
    if nav_stage and nav_memo:
        shared += f"## Navigation\n{nav_memo}\n\n"
        _log("Navigation", nav_memo)
        print(f"\n[nav] navigator")
        print(msg("info_nav_memo", memo=nav_memo[:120]))
    elif nav_stage:
        print(msg("info_nav_no_memo"))

    # Only agent stages; coder and reviewer — separate at the end
    agent_stages   = [s for s in stages if s.get("type") == "agent"]
    coder_stage    = next((s for s in stages if s.get("type") == "coder"), None)
    reviewer_stage = next((s for s in stages if s.get("type") == "reviewer"), None)

    # ── Agent passes (passes times) ──────────────────────────────────────────
    coder_response = ""

    for pass_num in range(1, passes + 1):
        if passes > 1:
            print(f"\n{'═' * 50}")
            print(msg("info_pipeline_run", num=pass_num, total=passes))
            print(f"{'═' * 50}")
            _log(f"─── Run {pass_num} / {passes}", "")

        for stage in agent_stages:
            sid        = stage.get("id", "agent")
            agent_name = stage.get("name", sid)
            role       = stage.get("role", "").replace("{task}", task)
            label      = f"{sid}" + (f"  (pass {pass_num})" if passes > 1 else "")

            print(f"\n[agent] {label}")

            # Read agent text from agents/<name>/
            agent_text = _read_agent_text(agent_name)

            # Full eval: agent knowledge + shared pipeline context
            # Agent without text (e.g. analyst) works only with shared context + role
            # (KV-cache continuation breaks on large injections — save_pattern is reliable)
            tmp_name = f"_agent_{sid}"
            if agent_text:
                combined = f"## Agent knowledge: {agent_name}\n{agent_text}\n\n---\n\n{shared}"
            else:
                combined = shared
            save_pattern(tmp_name, combined, grow_policy="retrain")
            memo_result = ask_pattern(tmp_name, role, grow=False)
            memo = memo_result or ""

            # Remove temporary pattern
            for ext in (".pkl", ".json"):
                p = Path(PATTERNS_DIR) / f"{tmp_name}{ext}"
                if p.exists():
                    p.unlink()

            if memo and memo.strip():
                title = f"{sid}" + (f" · pass {pass_num}" if passes > 1 else "")
                shared += f"## {title}\n{memo.strip()}\n\n"
                _log(title, memo.strip())
            else:
                print(msg("warn_agent_no_memo", name=agent_name))

    # ── Coder — final stage ──────────────────────────────────────────────────
    if coder_stage:
        role       = coder_stage.get("role", "").replace("{task}", task)
        file_names = ", ".join(Path(f).name for f in files)

        print(f"\n[coder] coder")
        print(msg("info_context_stats", words=len(shared.split()), files=len(files)))

        save_pattern("_pipeline", shared, grow_policy="retrain")
        coder_q = (
            f"Task: {task}\n\n"
            f"Files: {file_names}\n\n"
            f"{role}"
        )
        coder_response = ask_pattern("_pipeline", coder_q, grow=False)

        if coder_response:
            _log("Coder (diff)", coder_response)

    # ── Reviewer — check diff via tree-sitter ────────────────────────────────
    if reviewer_stage and coder_response and "```" in coder_response:
        from cognit_patch import _extract_all_diffs, _diff_target

        role = reviewer_stage.get("role", "").replace("{task}", task)
        print(f"\n[reviewer] reviewer")

        # Tree-sitter structure of affected files
        tree_info = ""
        if _HAS_INDEX:
            idx = _get_code_index()
            if idx:
                diffs = _extract_all_diffs(coder_response)
                for d in diffs:
                    target = _diff_target(d)
                    if target:
                        abs_target = target
                        if not os.path.isabs(target) and client_dir:
                            abs_target = os.path.join(client_dir, target)
                        tree_info += idx.file_summary(abs_target) + "\n\n"

        # Reviewer context: task + source files + diff + tree
        review_ctx = f"## Task\n{task}\n\n"
        for fpath in files:
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")[:6000]
                review_ctx += f"## File: {Path(fpath).name}\n```\n{content}\n```\n\n"
            except Exception:
                pass
        review_ctx += f"## Coder diff\n```diff\n{coder_response}\n```\n\n"
        if tree_info:
            review_ctx += f"## File structure (tree-sitter)\n{tree_info}\n"

        save_pattern("_pipeline", review_ctx, grow_policy="retrain")
        reviewed = ask_pattern("_pipeline", role, grow=False)

        if reviewed and "```" in reviewed:
            coder_response = reviewed
            _log("Reviewer (corrected)", reviewed)
            print(msg("ok_diff_corrected"))
        else:
            _log("Reviewer (unchanged)", reviewed or "(no response)")
            print(msg("info_diff_unchanged"))

    if coder_response and "```" in coder_response:
        print(msg("info_diff_ready"))

    if log_path and log_path.exists():
        print(msg("info_log_path", path=log_path))

    return coder_response or ""


def list_patterns():
    core.print_patterns_list(PATTERNS_DIR)


def _find_or_load_agent(agent_name: str) -> str | None:
    """
    Find or auto-load an agent pattern by name.
    Looks for pattern agent_name; if not found — reads .echo.json → client_project/agents/<name>/.
    """
    if core.pattern_exists(PATTERNS_DIR, agent_name):
        return agent_name

    cfg = _echo_config()
    if not cfg:
        print(msg("err_agent_echo_missing", name=agent_name))
        print(msg("info_load_manually", name=agent_name))
        return None

    agent_dir = Path(cfg.get("client_project", "")) / "agents" / agent_name
    if not agent_dir.exists():
        print(msg("err_agent_not_found", name=agent_name, dir=agent_dir))
        available = _list_agents(PATTERNS_DIR)
        if available:
            names = ", ".join(n for n, _ in available)
            print(msg("info_available_agents", names=names))
        else:
            print(msg("info_create_agents"))
        return None

    print(msg("status_loading_agent", name=agent_name, dir=agent_dir))
    _load_path(agent_name, str(agent_dir))
    return agent_name if core.pattern_exists(PATTERNS_DIR, agent_name) else None


def _read_pattern_source(name: str) -> str:
    """Re-read pattern source files from metadata."""
    meta = core.read_meta(PATTERNS_DIR, name)
    if not meta:
        return ""
    parts = []
    for src in meta.get("source_files", []):
        p = Path(src["path"])
        if p.exists():
            try:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    if not parts:
        agent_text = _read_agent_text(name)
        if agent_text:
            return agent_text
    return "\n\n---\n\n".join(parts)


def _ephemeral_eval_ask(texts: list[str], question: str,
                        tmp_name: str = "_ephemeral_tmp") -> str:
    """
    Full-eval ask: combines texts → save_pattern → ask_pattern → cleanup.

    Reliable alternative to _chain_ask for large injections (700+ tokens).
    Each text is combined via separator and run through full eval.
    Temporary pattern is deleted after use.
    """
    combined = "\n\n---\n\n".join(t for t in texts if t and t.strip())
    if not combined.strip():
        return ""
    print(msg("info_full_eval", n=len(combined)))
    save_pattern(tmp_name, combined)
    result = ask_pattern(tmp_name, question, grow=False)
    for ext in (".pkl", ".json"):
        p = Path(PATTERNS_DIR) / f"{tmp_name}{ext}"
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    return result or ""


def _llm_expand_query(task: str) -> list[str]:
    """
    Ask the LLM to suggest likely function/class names for an abstract task.
    Returns extra search terms to combine with the original query.
    """
    prompt = (
        f"Task: {task}\n\n"
        "What Python function or class names might be related to this task?\n"
        "List 5-10 likely names (snake_case for functions, CamelCase for classes).\n"
        "Return ONLY the names, one per line. No explanations."
    )
    print(msg("info_expanding_query"))
    response = _ephemeral_eval_ask([prompt], "List likely symbol names:",
                                   tmp_name="_expand_tmp")
    if not response:
        return []
    terms = []
    for line in response.strip().splitlines():
        name = line.strip().lstrip("•-0123456789.) ").strip()
        # Keep only valid identifiers
        if name and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            terms.append(name)
    return terms[:10]


def _llm_rerank(task: str, candidates_text: str, top_k: int = 8) -> list[int]:
    """
    Ask the LLM to rerank search candidates by relevance to the task.
    Returns list of candidate numbers (1-indexed) in order of relevance.
    Falls back to empty list on failure (caller keeps BM25 order).
    """
    prompt = (
        f"Task: {task}\n\n"
        "Rank these code symbols by relevance to the task (most relevant first).\n"
        "Return ONLY the numbers, separated by commas. No explanations.\n\n"
        f"{candidates_text}"
    )
    print(msg("info_reranking", count=candidates_text.count("\n") + 1))
    response = _ephemeral_eval_ask([prompt], "Which numbers are most relevant? Return comma-separated list.",
                                   tmp_name="_rerank_tmp")
    if not response:
        return []
    # Parse comma-separated integers
    try:
        nums = []
        for token in response.replace("\n", ",").split(","):
            token = token.strip().rstrip(".")
            if token.isdigit():
                nums.append(int(token))
        # Deduplicate while preserving order
        seen: set[int] = set()
        result: list[int] = []
        for n in nums:
            if n not in seen:
                seen.add(n)
                result.append(n)
            if len(result) >= top_k:
                break
        return result
    except Exception:
        return []


def _do_edit(file_path: str, task: str, base_pattern: str | None = None) -> str:
    """
    Read file FRESH and ask the model to produce a unified diff.

    Difference from /load + ask:
      - File is read right now (not from KV-cache) → exact lines, exact numbers
      - base_pattern is loaded as background (project context, agent)
      - If base_pattern is not set — temporary pattern from the file itself

    Returns model response (expected unified diff) or "" on error.
    """
    p = Path(file_path)
    if not p.exists():
        print(msg("err_file_not_found", path=file_path))
        return ""

    content = p.read_text(encoding="utf-8", errors="ignore")
    n_lines = len(content.splitlines())
    print(msg("status_reading_file", path=p.resolve(), lines=n_lines, chars=len(content)))

    file_block = (
        f"# File to edit: {p.resolve()}\n\n"
        f"```\n{content}\n```"
    )
    edit_question = (
        f"{task}\n\n"
        "Output a unified diff for this change. Use exact line numbers from the file.\n"
        "Format:\n"
        "--- a/filename\n"
        "+++ b/filename\n"
        "@@ -N,M +N,M @@\n"
        " context line\n"
        "-removed line\n"
        "+added line\n"
        "Only diff, no explanations."
    )

    if base_pattern and core.pattern_exists(PATTERNS_DIR, base_pattern):
        # Full eval: source + file → reliable for large files
        base_source = _read_pattern_source(base_pattern)
        return _ephemeral_eval_ask([base_source, file_block], edit_question, "_edit_tmp")
    else:
        # No active pattern — temporary from the file itself
        return _ephemeral_eval_ask([file_block], edit_question, "_edit_tmp")


# =============================================================================
# CLI
# =============================================================================


def _load_path(name: str, raw_path: str, force_policy: str = None):
    """Load a file or directory as a pattern."""
    p = Path(raw_path)
    if p.is_dir():
        files = core.collect_text_files(p)
        if not files:
            print(msg("err_empty_folder", path=raw_path))
            return
        texts, paths = [], []
        header = f"# Project: {p.name}\nPath: {p.resolve()}\nFiles: {len(files)}\n"
        texts.append(header)
        for f in files:
            try:
                texts.append(f"# {f.resolve()}\n\n{f.read_text(encoding='utf-8', errors='ignore')}")
                paths.append(str(f.resolve()))
            except Exception:
                pass
        print(msg("status_loading_files", count=len(files), path=raw_path))
        save_pattern(name, "\n\n---\n\n".join(texts), source_files=paths, grow_policy=force_policy)
    elif p.is_file():
        save_pattern(name, p.read_text(encoding="utf-8", errors="ignore"),
                     source_files=[str(p.resolve())], grow_policy=force_policy)
    else:
        print(msg("err_path_not_found", path=raw_path))


def _load_mix(name: str, raw_paths: list[str], force_policy: str = None):
    """Load multiple files/folders as a single composite pattern."""
    all_texts: list[str] = []
    all_sources: list[str] = []
    for raw_path in raw_paths:
        p = Path(raw_path)
        if p.is_dir():
            files = core.collect_text_files(p)
            if not files:
                print(msg("warn_empty_folder", path=raw_path))
                continue
            header = f"# {p.name}  ({p.resolve()})\n"
            block_texts = [header]
            for f in files:
                try:
                    block_texts.append(
                        f"# {f.resolve()}\n\n{f.read_text(encoding='utf-8', errors='ignore')}"
                    )
                    all_sources.append(str(f.resolve()))
                except Exception:
                    pass
            all_texts.append("\n\n---\n\n".join(block_texts))
            print(msg("info_folder_loaded", name=p.name, count=len(files)))
        elif p.is_file():
            all_texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            all_sources.append(str(p))
            print(f"   + {p.name}")
        else:
            print(msg("warn_path_not_found", path=raw_path))
    if not all_texts:
        print(msg("err_no_files"))
        return
    combined = "\n\n===\n\n".join(all_texts)
    print(msg("info_composite", name=name, sources=len(raw_paths), files=len(all_sources)))
    save_pattern(name, combined, source_files=all_sources, grow_policy=force_policy)


def _hint_patterns():
    core.hint_patterns(PATTERNS_DIR)


def _auto_init_agents():
    """Auto-load missing agents from client project's agents/ directory."""
    cfg = _echo_config()
    client = cfg.get("client_project", "")
    agents = _list_agents(PATTERNS_DIR)
    missing = [(name, loaded) for name, loaded in agents if not loaded]

    if missing and client:
        print(msg("status_auto_init", missing=len(missing), total=len(agents)))
        for name, _ in missing:
            agent_dir = Path(client) / "agents" / name
            if agent_dir.exists():
                print(msg("status_loading_agent", name=name, dir=agent_dir))
                _load_path(name, str(agent_dir))
            else:
                print(msg("warn_agent_folder_missing", dir=agent_dir))
        print()


_code_index_cache = None
_code_index_project = ""


def _get_code_index(rebuild: bool = False):
    """Return CodeIndex for the client project (cached in module)."""
    global _code_index_cache, _code_index_project
    if not _HAS_INDEX:
        print(msg("warn_tree_sitter_missing"))
        return None
    cfg = _echo_config()
    client_dir = cfg.get("client_project", "")
    if not client_dir or not Path(client_dir).exists():
        return None
    if _code_index_cache is not None and _code_index_project == client_dir and not rebuild:
        return _code_index_cache
    idx = CodeIndex(client_dir, cache_dir=PATTERNS_DIR)
    idx.build()
    _code_index_cache = idx
    _code_index_project = client_dir
    return idx


_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "about",
    "it", "its", "this", "that", "and", "or", "but", "not", "no",
    "fix", "add", "make", "get", "set", "use", "all", "how", "what",
    "исправь", "добавь", "сделай", "как", "что", "все", "это", "для",
    "нужно", "надо", "можно",
}


def _extract_grep_terms(task: str) -> list[str]:
    """Extract meaningful terms from task for grep search."""
    tokens = re.split(r'[^a-zA-Z0-9а-яА-ЯёЁ_]+', task.lower())
    return [t for t in tokens if len(t) >= 4 and t not in _STOP_WORDS][:5]


def _merge_unique(target: list, source: list) -> int:
    """Append items from source to target if (filepath, symbol.name) not already present."""
    seen = {(r.filepath, r.symbol.name) for r in target}
    added = 0
    for r in source:
        key = (r.filepath, r.symbol.name)
        if key not in seen:
            seen.add(key)
            target.append(r)
            added += 1
    return added


def _llm_chunked_scan(task: str, idx) -> list:
    """Last-resort: scan all project files via LLM to find relevant code."""
    print(msg("info_full_scan"))
    all_files = sorted(idx.symbols.keys())
    results = []
    for filepath in all_files:
        try:
            content = Path(filepath).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(content) > 6000:
            content = content[:6000]
        fname = Path(filepath).name
        print(msg("info_scan_checking", name=fname))
        prompt = f"Task: {task}\n\nFile: {fname}\n```\n{content}\n```"
        response = _ephemeral_eval_ask(
            [prompt],
            "Does this file contain code relevant to the task? "
            "If yes, list the function/class names. If no, say NO.",
            tmp_name="_scan_tmp"
        )
        if not response or "NO" in response.strip().upper()[:10]:
            continue
        # Match response to indexed symbols
        resp_lower = response.lower()
        matched = False
        for sym in idx.symbols.get(filepath, []):
            if sym.kind in ("function", "class") and sym.name.lower() in resp_lower:
                results.append(_SearchResult(filepath, sym, 0.5, f"scan:{sym.name}"))
                matched = True
        if not matched:
            # File is relevant but no specific symbol matched — add first function/class
            for sym in idx.symbols.get(filepath, []):
                if sym.kind in ("function", "class"):
                    results.append(_SearchResult(filepath, sym, 0.3, "scan:file"))
                    break
        if len(results) >= 10:
            break
    if results:
        print(msg("info_scan_found", count=len(results)))
    else:
        print(msg("info_scan_nothing"))
    return results


def _route_via_index(task: str) -> tuple[list[str], str]:
    """
    Cascading search: find relevant files for a task.

    Escalation from cheap to expensive:
    [1] BM25 over symbols        — free, <100ms
    [2] LLM query expansion      — ~2-3s, if [1] < 5
    [3] grep over codebase       — free, <500ms
    [4] find_dependencies (AST)  — free, <200ms
    [5] LLM rerank               — ~2-3s
    [6] chunked full scan        — ~30-120s, last resort
    """
    idx = _get_code_index()
    if idx is None:
        print(msg("warn_client_project"))
        return [], ""

    cfg = _echo_config()
    rerank_enabled = cfg.get("transformer", {}).get("rerank", True)
    llm_ok = llm is not None and rerank_enabled

    # ── [1] BM25 ────────────────────────────────────────────────────
    results = idx.search(task, top_k=20)

    # ── [2] LLM expansion (if < 5) ─────────────────────────────────
    if llm_ok and len(results) < 5:
        extra_terms = _llm_expand_query(task)
        if extra_terms:
            expanded = idx.search(task + " " + " ".join(extra_terms), top_k=20)
            added = _merge_unique(results, expanded)
            print(msg("info_expand_done", count=len(extra_terms), total=len(results)))

    # ── [3] grep (if still < 5) ────────────────────────────────────
    if len(results) < 5:
        grep_terms = _extract_grep_terms(task)
        if grep_terms:
            print(msg("info_grep_searching"))
            all_grep: list = []
            for term in grep_terms:
                all_grep.extend(idx.grep_files(term, max_results=30))
            if all_grep:
                grep_as_sr = idx.grep_to_search_results(all_grep)
                added = _merge_unique(results, grep_as_sr)
                unique_files = len({g.filepath for g in all_grep})
                print(msg("info_grep_found", count=len(all_grep), files=unique_files))

    # ── [4] find dependencies (enrich found results) ───────────────
    if results:
        print(msg("info_deps_searching"))
        found_names = [r.symbol.name for r in results[:8]]
        found_files = list({r.filepath for r in results[:8]})
        deps = idx.find_dependencies(found_names, found_files)
        added = _merge_unique(results, deps)
        if added:
            print(msg("info_deps_found", count=added))

    # ── [5] LLM rerank (if > 3 candidates) ────────────────────────
    if llm_ok and len(results) > 3:
        candidates_text = idx.format_candidates(results)
        reranked = _llm_rerank(task, candidates_text, top_k=8)
        if reranked:
            reordered = [results[i - 1] for i in reranked if 1 <= i <= len(results)]
            selected = set(reranked)
            for i, r in enumerate(results, 1):
                if i not in selected:
                    reordered.append(r)
            results = reordered
            print(msg("info_rerank_done", count=len(reranked)))
        else:
            print(msg("info_rerank_fallback"))

    # ── [6] chunked full scan (last resort) ────────────────────────
    if not results and llm_ok:
        results = _llm_chunked_scan(task, idx)

    if not results:
        print(msg("warn_no_symbols"))
        return [], ""

    # ── Output ─────────────────────────────────────────────────────
    seen: set[str] = set()
    files: list[str] = []
    for r in results:
        if r.filepath not in seen:
            seen.add(r.filepath)
            files.append(r.filepath)

    context_note = idx.search_summary(task, top_k=10)

    print(msg("info_symbols_found", n_symbols=len(results), n_files=len(files)))
    for f in files:
        rel = f.replace(idx.project_dir, "").lstrip("/\\")
        file_results = [r for r in results if r.filepath == f]
        symbols_str = ", ".join(r.symbol.name for r in file_results[:3])
        print(f"   • {rel}  ({symbols_str})")

    return files, context_note


def cli_loop() -> None:
    """
    Main interactive loop.
    Returns None on /exit.
    """
    print(msg("info_header", model=MODEL_NAME[:28], repo=REPO_NAME, branch=BRANCH_NAME))

    list_patterns()
    _auto_init_agents()

    active = None
    active_policy = None
    last_response = ""
    ambient_agents = []  # list of agents for ambient mode ([] = disabled)

    if not active:
        print(msg("info_intro"))
        print(msg("info_intro_load"))

    while True:
        try:
            if active:
                if ambient_agents:
                    agents_str = " + ".join(ambient_agents)
                    prompt_str = f"🧠 [{active} + {agents_str}]> "
                else:
                    marker = "~" if active_policy == "grow" else ""
                    prompt_str = f"🧠 [{active}{marker}]> "
            else:
                prompt_str = "🧠> "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue

            # ── Slash commands ───────────────────────────────────────────────
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=2)
                cmd = parts[0].lower() if parts else ""

                if cmd in ("exit", "quit"):
                    print(msg("ok_exit"))
                    return None

                elif cmd == "help":
                    import cognit_i18n
                    print(cognit_i18n.HELP[cognit_i18n.LANG])

                elif cmd == "list":
                    list_patterns()

                elif cmd == "index":
                    query = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                    idx = _get_code_index(rebuild=("--rebuild" in query))
                    if idx is None:
                        print(msg("err_index_unavailable"))
                        continue
                    query = query.replace("--rebuild", "").strip()
                    if query:
                        print(idx.search_summary(query))
                    else:
                        print(idx.project_summary())

                elif cmd == "load":
                    if len(parts) < 3:
                        print(msg("err_load_usage"))
                        print(msg("info_load_composite"))
                        print(msg("info_load_retrain"))
                        continue
                    name = parts[1]
                    force_policy = None
                    if name.startswith("?"):
                        name = name[1:]
                        force_policy = "retrain"
                    at_paths = [p[1:] for p in parts[2:] if p.startswith("@")]
                    if len(at_paths) > 1:
                        _load_mix(name, at_paths, force_policy=force_policy)
                    elif len(at_paths) == 1:
                        _load_path(name, at_paths[0], force_policy=force_policy)
                    else:
                        save_pattern(name, " ".join(parts[2:]), grow_policy=force_policy)
                    active = name  # automatically switch to new pattern
                    active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                    print(msg("info_active_pattern", name=active))

                elif cmd == "edit":
                    # /edit @src/config.py change 0-1 to percentages
                    if len(parts) < 3 or not parts[1].startswith("@"):
                        print(msg("err_edit_usage"))
                        print(msg("info_edit_example"))
                        continue
                    edit_path = parts[1][1:]
                    edit_task = parts[2]
                    result = _do_edit(edit_path, edit_task, active)
                    if result:
                        last_response = result
                        print(msg("info_apply_patch"))

                elif cmd == "patch":
                    if not active:
                        print(msg("err_select_pattern"))
                        continue
                    diffs = _extract_all_diffs(last_response) if last_response else []
                    if not diffs:
                        if not last_response:
                            print(msg("err_no_previous"))
                            continue
                        print(msg("info_reformatting"))
                        base_source = _read_pattern_source(active)
                        last_response = _ephemeral_eval_ask(
                            [base_source, f"Previous response:\n\n{last_response}"],
                            "Show the changes from this response as a unified diff (```diff block).",
                            "_patch_tmp",
                        )
                        diffs = _extract_all_diffs(last_response)
                    if not diffs:
                        print(msg("err_no_diff"))
                        continue

                    # Resolve paths relative to client_project
                    cfg_patch = _echo_config()
                    client_dir_patch = cfg_patch.get("client_project", "")
                    # Explicit @file override (only for single diff)
                    explicit_target = parts[1].lstrip("@") if len(parts) > 1 else None

                    if len(diffs) > 1:
                        print(msg("info_diffs_found", count=len(diffs)))

                    applied_any = False
                    for diff in diffs:
                        target = explicit_target or _diff_target(diff)
                        if not target:
                            print(msg("err_file_not_specified"))
                            continue
                        # Resolve relative paths against client_project
                        if not os.path.isabs(target) and not os.path.exists(target) and client_dir_patch:
                            target = os.path.join(client_dir_patch, target)
                        is_new = _is_new_file_diff(diff) or not os.path.exists(target)
                        preview = diff.split('\n')
                        label = "NEW FILE" if is_new else target
                        print(f"\n📋 Diff ({len(preview)} lines)  →  {label}")
                        print("─" * 60)
                        for ln in preview[:50]:
                            print(ln)
                        if len(preview) > 50:
                            print(msg("info_more_lines", count=len(preview) - 50))
                        print("─" * 60)
                        try:
                            action = "Create" if is_new else "Apply"
                            ans = input(f"   {action} '{target}'? [y/N] ").strip().lower()
                        except (KeyboardInterrupt, EOFError):
                            continue
                        if ans == "y":
                            _apply_patch(diff, target)
                            applied_any = True

                    # After /patch in pipeline — reset coder context
                    if applied_any and active == "_pipeline":
                        active = None
                        active_policy = None
                        print(msg("ok_pipeline_done"))

                elif cmd == "agents":
                    agents = _list_agents(PATTERNS_DIR)
                    if not agents:
                        print(msg("info_no_agents"))
                    else:
                        print(msg("info_agent_count", count=len(agents)))
                        for name, loaded in agents:
                            tag = " [loaded]" if loaded else ""
                            print(f"   • {name}{tag}")
                        print()

                elif cmd == "agent":
                    # Parse all agent names: /agent arch style context
                    agent_names = user_input[1:].split()[1:]
                    if not agent_names or agent_names[0].lower() == "off":
                        ambient_agents = []
                        print(msg("info_agents_disabled"))
                    else:
                        loaded_names = []
                        for agent_name in agent_names:
                            loaded = _find_or_load_agent(agent_name)
                            if loaded:
                                loaded_names.append(agent_name)
                        if loaded_names:
                            ambient_agents = loaded_names
                            agents_str = " + ".join(loaded_names)
                            print(msg("ok_agents_enabled", names=agents_str))
                            print(msg("info_question_flow", pattern=active or '?', agents=agents_str))
                            print(msg("info_ambient_ephemeral"))
                            print(msg("info_disable_agents"))

                elif cmd in ("review", "style"):
                    default_q = (
                        "Do a code review considering project rules. What needs to be fixed?"
                        if cmd == "review" else
                        "Check compliance with our style standards. List violations."
                    )
                    # Syntax: /review [agent] @<file> [question]
                    # parts[1] — agent or @file; if not @, it's the agent name
                    rest = parts[1:]
                    agent_name = "style"  # default
                    if rest and not rest[0].startswith("@"):
                        agent_name = rest[0]
                        rest = rest[1:]

                    # Content source: @file/folder or last_response
                    if rest and rest[0].startswith("@"):
                        content_path = rest[0][1:]
                        question = " ".join(rest[1:]) if len(rest) > 1 else default_q
                        p = Path(content_path)
                        if p.is_file():
                            content = p.read_text(encoding="utf-8", errors="ignore")
                            label = content_path
                        elif p.is_dir():
                            files = core.collect_text_files(p)
                            content = "\n\n---\n\n".join(
                                f.read_text(encoding="utf-8", errors="ignore") for f in files
                            )
                            label = f"{content_path}/ ({len(files)} files)"
                        else:
                            print(msg("err_content_not_found", path=content_path))
                            continue
                    elif last_response:
                        content = last_response
                        question = " ".join(rest) if rest else default_q
                        label = "last model response"
                    else:
                        print(msg("err_specify_file", cmd=cmd))
                        agents = _list_agents(PATTERNS_DIR)
                        if agents:
                            print(msg("info_available_agents", names=', '.join(n for n, _ in agents)))
                        continue
                    print(msg("info_agent_analyzing", name=agent_name, label=label))
                    agent_source = _read_agent_text(agent_name)
                    if agent_source:
                        last_response = _ephemeral_eval_ask(
                            [agent_source, content], question, "_review_tmp")
                    else:
                        print(msg("warn_agent_not_found_rev", name=agent_name))

                else:
                    print(msg("err_unknown_cmd", cmd=cmd))

            # ── use <name> ──────────────────────────────────────────────────
            elif user_input.lower().startswith("use ") or user_input.lower() == "use":
                name = user_input[4:].strip()
                if not name:
                    print(msg("err_use_usage"))
                    _hint_patterns()
                    continue
                if not (Path(PATTERNS_DIR) / f"{name}.pkl").exists():
                    print(msg("err_pattern_not_found_use", name=name))
                    _hint_patterns()
                    continue
                active = name
                active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                policy_label = " [~grow]" if active_policy == "grow" else " [retrain]"
                print(msg("ok_pattern_selected", name=active, policy=policy_label))

            # ── ? question → temporarily switch policy ──────────────────────
            elif user_input.startswith("?"):
                question = user_input[1:].strip()
                if not question:
                    print(msg("err_question_after_qmark"))
                    continue
                if not active:
                    print(msg("err_select_pattern"))
                    _hint_patterns()
                    continue
                if ambient_agents:
                    agent_text = "\n\n---\n\n".join(
                        t for name in ambient_agents if (t := _read_agent_text(name))
                    )
                    if agent_text:
                        base_source = _read_pattern_source(active)
                        last_response = _ephemeral_eval_ask(
                            [base_source, agent_text], question, "_ambient_tmp")
                    else:
                        last_response = ask_pattern(active, question, grow=False)
                else:
                    # ? flips policy: grow→don't save, retrain→save
                    flipped_grow = (active_policy == "retrain")
                    last_response = ask_pattern(active, question, grow=flipped_grow)

            # ── route <task> → find files via tree-sitter index ─────────────
            elif user_input.lower().startswith("route ") or user_input.lower() == "route":
                task = user_input[6:].strip()
                if not task:
                    print(msg("err_route_usage"))
                    print(msg("info_route_example"))
                    continue
                files, nav_memo = _route_via_index(task)
                if files:
                    try:
                        ans = input(msg("prompt_run_pipeline")).strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        ans = "n"
                    if ans != "n":
                        last_response = _run_pipeline(task, files, nav_memo)
                        active = "_pipeline"
                        active_policy = "retrain"

            # ── plain text → ask ─────────────────────────────────────────────
            else:
                if not active:
                    # No pattern — auto-route via tree-sitter index
                    files, nav_memo = _route_via_index(user_input)
                    if files:
                        try:
                            ans = input(msg("prompt_run_pipeline")).strip().lower()
                        except (KeyboardInterrupt, EOFError):
                            ans = "n"
                        if ans != "n":
                            last_response = _run_pipeline(user_input, files, nav_memo)
                            active = "_pipeline"
                            active_policy = "retrain"
                    else:
                        print(msg("err_files_not_found"))
                    continue

                if ambient_agents:
                    agent_text = "\n\n---\n\n".join(
                        t for name in ambient_agents if (t := _read_agent_text(name))
                    )
                    if agent_text:
                        base_source = _read_pattern_source(active)
                        last_response = _ephemeral_eval_ask(
                            [base_source, agent_text], user_input, "_ambient_tmp")
                    else:
                        last_response = ask_pattern(active, user_input, grow=(active_policy == "grow"))
                else:
                    last_response = ask_pattern(active, user_input, grow=(active_policy == "grow"))

                # After any ask: if response contains code — hint /patch
                if last_response and "```" in last_response:
                    print(msg("info_code_found"))

        except KeyboardInterrupt:
            print(msg("ok_exit"))
            return None
        except Exception as e:
            print(msg("err_general", error=e))
            import traceback
            traceback.print_exc()


# =============================================================================
# HEADLESS MODE (for post-commit hook and scripts)
# =============================================================================
def headless_refresh_file(file_path: str):
    """
    Recreate all patterns that reference file_path.
    Called by post-commit hook: python cognit_transformer.py --refresh-file src/auth.py
    Does not ask interactive questions — everything is automatic.
    """
    if not Path(PATTERNS_DIR).exists():
        return

    if not os.path.exists(file_path):
        print(msg("warn_file_not_found_ref", path=file_path))
        return

    new_hash = core.file_hash(file_path)
    refreshed = 0

    for meta_path in sorted(Path(PATTERNS_DIR).glob("*.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sources = meta.get("source_files", [])
        affected = [s for s in sources if s["path"] == file_path]
        if not affected:
            continue

        name = meta["name"]
        policy = meta.get("grow_policy", "retrain")

        # Check if file changed since pattern creation
        if all(s["hash"] == new_hash for s in affected):
            print(msg("ok_pattern_current", name=name))
            continue

        if policy == "grow":
            print(msg("info_grow_skip", name=name))
            continue

        print(msg("status_recreating", name=name, path=file_path))
        texts, paths = [], []
        for src in sources:
            p = src["path"]
            if os.path.exists(p):
                texts.append(Path(p).read_text(encoding="utf-8"))
                paths.append(p)

        if texts:
            save_pattern(name, "\n\n".join(texts), source_files=paths)
            refreshed += 1

    if refreshed:
        print(msg("ok_patterns_updated", count=refreshed))
    elif refreshed == 0:
        print(msg("info_no_patterns_using", path=file_path))


def headless_status():
    """
    Check freshness of all patterns.
    Exit code: 0 = all current, 1 = there are stale ones.
    """
    if not Path(PATTERNS_DIR).exists():
        print(msg("info_no_patterns"))
        sys.exit(0)

    stale = []
    for meta_path in sorted(Path(PATTERNS_DIR).glob("*.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        sources = meta.get("source_files", [])
        if not sources:
            continue

        for src in sources:
            path = src["path"]
            if not os.path.exists(path):
                stale.append((meta["name"], path, "not found"))
            elif core.file_hash(path) != src["hash"]:
                stale.append((meta["name"], path, "modified"))

    if not stale:
        print(msg("ok_all_current"))
        sys.exit(0)

    print(msg("warn_stale_patterns", count=len(stale)))
    for name, path, reason in stale:
        print(f"   • {name}: {path}  ({reason})")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--refresh-file" and len(sys.argv) > 2:
            print(msg("warn_refresh_deprecated"))
            init_model()
            headless_refresh_file(sys.argv[2])
        elif sys.argv[1] == "--status":
            headless_status()  # model not needed
        else:
            print(msg("err_unknown_flag", flag=sys.argv[1]))
            print(msg("info_available_flags"))
            sys.exit(1)
    else:
        init_model()
        cli_loop()
