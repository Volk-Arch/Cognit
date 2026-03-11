#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""cognit_transformer.py — Transformer backend (Qwen3). KV-cache → .pkl."""

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
try:
    from cognit_index import CodeIndex
    _HAS_INDEX = True
except ImportError:
    CodeIndex = None
    _HAS_INDEX = False

# =============================================================================
# КОНФИГУРАЦИЯ — читается из .echo.json (ключ "transformer"), defaults ниже
# =============================================================================
_c           = _echo_config().get("transformer", {})
MODEL_PATH   = _c.get("model_path",   "models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
MODEL_NAME   = Path(MODEL_PATH).stem
N_CTX        = _c.get("n_ctx",        8192)
N_GPU_LAYERS = _c.get("n_gpu_layers", -1)
MAX_TOKENS   = _c.get("max_tokens",   512)

REPO_NAME    = core.git_repo_name()
BRANCH_NAME  = core.git_branch()
PATTERNS_DIR = core.make_patterns_dir(REPO_NAME, BRANCH_NAME)

# Стоп-токены Qwen3 (ChatML)
# </think> НЕ включаем в основной список: Qwen3 выдаёт <think></think> перед ответом
STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]

# Стоп-паттерны для ответа: модель начинает играть за других → стоп
# Проверяются ТОЛЬКО после answer_started
# </think> НЕ включаем: попадает в recent_text сразу после answer_started → ложный стоп
ANSWER_STOP_PATTERNS = ["Human:", "User:", "\nHuman:", "\nUser:"]

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ
# =============================================================================
try:
    from llama_cpp import Llama
except ImportError:
    print("❌ llama-cpp-python не установлен.")
    print("   pip install llama-cpp-python --extra-index-url "
          "https://abetlen.github.io/llama-cpp-python/whl/cu121")
    sys.exit(1)

llm = None  # инициализируется через init_model()
os.makedirs(PATTERNS_DIR, exist_ok=True)


def init_model():
    """Загружает Transformer-модель в VRAM."""
    global llm
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        print("   Укажите правильный путь в MODEL_PATH")
        sys.exit(1)
    print(f"Загрузка модели: {MODEL_PATH} ...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    print(f"✅ Transformer загружен  |  {'GPU' if N_GPU_LAYERS != 0 else 'CPU'}")
    print(f"   Паттерны: {PATTERNS_DIR}  [{REPO_NAME} / {BRANCH_NAME}]")


def unload_model():
    """Выгружает модель из VRAM."""
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
# РАБОТА С ПАТТЕРНАМИ
# =============================================================================
def _check_and_refresh(name: str):
    """Проверяет актуальность паттерна. Если файлы изменились — предлагает пересоздать."""
    changed = core.check_stale_sources(PATTERNS_DIR, name)
    if not changed:
        return

    print(f"\n⚠️  Файлы изменились с момента создания паттерна '{name}':")
    for c in changed:
        print(f"    • {c}")

    try:
        ans = input("   Пересоздать паттерн? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if ans != "y":
        print("   Продолжаем со старым паттерном.")
        return

    meta = core.read_meta(PATTERNS_DIR, name)
    texts, paths = [], []
    for src in meta.get("source_files", []):
        p = src["path"]
        if os.path.exists(p):
            texts.append(Path(p).read_text(encoding="utf-8"))
            paths.append(p)

    if texts:
        save_pattern(name, "\n\n".join(texts), source_files=paths)


def _context_prompt(text: str) -> str:
    """Формирует промпт с контекстом в формате ChatML (Qwen3)."""
    return (
        "<|im_start|>system\n"
        "Ты — ассистент с предзагруженным контекстом. "
        "Отвечай на вопросы, опираясь на загруженный контекст. "
        "Если информации в контексте недостаточно, скажи об этом. "
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Загрузи контекст:\n\n{text.strip()}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Контекст загружен. Готов отвечать на вопросы.\n"
        "<|im_end|>\n"
    )


def _question_prompt(question: str) -> str:
    """Добавляет вопрос к уже загруженному KV-cache."""
    return (
        "<|im_start|>user\n"
        f"{question}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def save_pattern(name: str, text: str, source_files: list[str] = None, grow_policy: str = None):
    """
    Прогоняет текст через LLM, сохраняет KV-cache.
    После этого модель 'знает' текст без необходимости читать его снова.

    grow_policy: 'grow' | 'retrain' | None (авто по ветке/пути)
    """
    if grow_policy is None:
        grow_policy = core.default_grow_policy(BRANCH_NAME, source_files or [])
    print(f"\n📝 Формирование паттерна '{name}'  [{grow_policy}]...")

    prompt = _context_prompt(text)
    tokens = llm.tokenize(prompt.encode("utf-8"), special=True)

    if len(tokens) > N_CTX - 64:
        print(f"⚠️  Текст слишком длинный ({len(tokens)} токенов), обрезаем до {N_CTX - 64}")
        tokens = tokens[:N_CTX - 64]

    print(f"   Токенов в контексте: {len(tokens)}")
    print("   Обработка (eval)...")

    llm.reset()
    llm.eval(tokens)
    state = llm.save_state()

    # Сохраняем состояние
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

    print(f"✅ {name}  ({size_kb:.0f} KB, {llm.n_tokens} tok)")


def load_pattern(name: str) -> bool:
    """Восстанавливает KV-cache из файла."""
    pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
    if not pattern_path.exists():
        print(f"❌ Паттерн '{name}' не найден. Используйте 'list' для просмотра.")
        return False

    # Проверка совместимости модели: KV-cache от другой квантизации = мусор
    meta = core.read_meta(PATTERNS_DIR, name)
    if meta and meta.get("model") and meta["model"] != MODEL_NAME:
        print(f"⚠️  Паттерн '{name}' создан для {meta['model']}, "
              f"текущая модель {MODEL_NAME}")
        print(f"   Пересоздай: /load ?{name} @<путь>")
        return False

    with open(pattern_path, "rb") as f:
        state = pickle.load(f)

    llm.load_state(state)

    # load_state() восстанавливает KV-cache, но n_tokens (позиция записи)
    # может не восстановиться — зависит от версии llama-cpp-python.
    if meta and "n_tokens" in meta:
        llm.n_tokens = meta["n_tokens"]

    return True


def ask_pattern(name: str, question: str, grow: bool = True) -> str:
    """
    Загружает KV-cache и отвечает на вопрос.
    Текст контекста НЕ передаётся повторно — только вопрос.

    grow=True (по умолчанию): после ответа KV-cache обновляется —
    паттерн накапливает весь диалог (растущая сессия).
    """
    print(f"💬 [{name}]", end="  ", flush=True)
    _check_and_refresh(name)
    if not load_pattern(name):
        return

    # Добавляем только вопрос — контекст уже в KV-cache
    question_prompt = _question_prompt(question)
    q_tokens = llm.tokenize(question_prompt.encode("utf-8"), add_bos=False, special=True)
    print(f"(KV: {llm.n_tokens})", end="  ", flush=True)

    print(f"+{len(q_tokens)} tok")
    print(f"\n❓ {question}")
    print("─" * 50)

    # reset=False — не сбрасывать загруженный KV-cache!
    # Qwen3 выдаёт <think>...</think> перед ответом — скрываем этот блок
    collected = []
    answer_started = False
    think_dots = 0  # счётчик точек-индикатора во время think-фазы
    junk_streak = 0  # подряд идущие «мусорные» токены

    for token_id in llm.generate(q_tokens, reset=False, temp=0.3, top_p=0.9, repeat_penalty=1.1):
        piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        collected.append(piece)
        full = "".join(collected)

        # Обрезаем стоп-токены из куска перед выводом
        clean_piece = piece
        for stop in STOP_TOKENS:
            if stop in clean_piece:
                clean_piece = clean_piece[:clean_piece.index(stop)]

        if not answer_started:
            if "</think>" in full:
                # Стираем точки-индикатор
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
                # Модель отвечает без <think> — выводим сразу
                print(clean_piece, end="", flush=True)
                answer_started = True
            elif "<think>" in full and len(collected) % 12 == 0:
                # Индикатор прогресса во время think-фазы
                think_dots += 1
                print(".", end="", flush=True)
        else:
            print(clean_piece, end="", flush=True)

        tail = full[-60:]
        if any(stop in tail for stop in STOP_TOKENS):
            break
        if len(collected) >= MAX_TOKENS:
            break
        # Детектор фейковых turn'ов: модель начала играть за Human/User
        if answer_started:
            recent_text = "".join(collected[-10:])
            if any(p in recent_text for p in ANSWER_STOP_PATTERNS):
                print("\n⚠️  Фейковый turn — генерация остановлена")
                break
            # Детектор повторных code-блоков: второй ```diff/```python → стоп
            fence_count = full.count("```")
            if fence_count >= 4:  # 2 блока = 4 fence-маркера
                print("\n⚠️  Повторный code-блок — генерация остановлена")
                break
        # Детектор зацикливания: несколько размеров окна
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
            print("\n⚠️  Зацикливание — генерация остановлена")
            break
        # Лимит think-фазы: >400 токенов без </think> → стоп
        if not answer_started and len(collected) > 400:
            if think_dots:
                print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)
            print("⚠️  Think-фаза слишком длинная — генерация остановлена")
            break
        # Детектор мусорных токенов: `, пробелы, \n подряд → стоп
        if piece.strip("`\n\r \t") == "":
            junk_streak += 1
            if junk_streak >= 6:
                print("\n⚠️  Мусорный вывод — генерация остановлена")
                break
        else:
            junk_streak = 0

    # Стираем точки если think не завершился
    if think_dots and not answer_started:
        print("\r" + " " * (think_dots + 10) + "\r", end="", flush=True)

    # Если модель ответила без think-блока — вывести всё
    if not answer_started:
        print("".join(collected), end="", flush=True)

    print("\n" + "─" * 50)

    # Растущая сессия: сохраняем обновлённый KV-cache обратно в паттерн
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

        print(f"💾 {name}  ({meta['size_kb']} KB, {meta['n_asks']}д)")

    # Очистка ответа: убираем think-блок и стоп-токены
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
    Парсит диапазон строк из навигационного мемо для конкретного файла.
    Ищет паттерны: 'строки 42-67', 'lines 42-67', ':42-67'.
    Сначала ищет вблизи имени файла, потом по всему мемо.
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
    Возвращает сфокусированный фрагмент файла если в навигационном мемо есть номера строк.
    Фрагмент = указанные строки ± context_lines, с номерами строк.
    Если номеров нет — полный файл до FILE_LIMIT символов.
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
            parts.append(f"... (строки 1–{start_idx} пропущены)")
        for i in range(start_idx, end_idx):
            parts.append(f"{i + 1}: {lines[i]}")
        if end_idx < len(lines):
            parts.append(f"... (строки {end_idx + 1}–{len(lines)} пропущены)")

        excerpt = "\n".join(parts)
        print(f"   📍 Фрагмент: строки {start_line}–{end_line} "
              f"(±{context_lines}) из {len(lines)}")
        return excerpt[:FILE_LIMIT]

    # Нет диапазона — полный файл
    if len(raw) > FILE_LIMIT:
        return raw[:FILE_LIMIT] + "\n... (обрезано)"
    return raw


def _run_pipeline(task: str, files: list[str], nav_memo: str = "") -> str:
    """
    Последовательный пайплайн агентов с накопленным контекстом.

    Порядок стадий берётся из pipeline.json клиентского проекта.
    Каждая стадия получает весь накопленный контекст и добавляет своё мемо.
    Поддерживает passes > 1: агенты прогоняются N раз, каждый следующий
    прогон видит мемо всех предыдущих → уточнение.
    Лог рассуждений пишется в _pipeline_log.md клиентского проекта.
    Финальная стадия 'coder' генерирует unified diff.

    nav_memo: контекстная заметка от tree-sitter навигатора (search_summary).
    Возвращает ответ кодера (diff).
    """
    from cognit_pipeline import load_pipeline, describe_pipeline

    cfg        = _echo_config() or {}
    client_dir = cfg.get("client_project", "")
    pipeline   = load_pipeline(client_dir)
    stages     = [s for s in pipeline.get("stages", []) if s.get("enabled", True)]
    passes     = max(1, int(pipeline.get("passes", 1)))

    # ── Лог рассуждений ──────────────────────────────────────────────────────
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

    # ── Описание пайплайна ───────────────────────────────────────────────────
    suffix = f", {passes} прогона" if passes > 1 else ""
    print(f"\n🚀 Пайплайн  ({len(stages)} стадий{suffix})")
    print(describe_pipeline(pipeline))
    print("─" * 50)

    # Удаляем предыдущий temp-паттерн если был
    for ext in (".pkl", ".json"):
        p = Path(PATTERNS_DIR) / f"_pipeline{ext}"
        if p.exists():
            p.unlink()

    # ── Shared context: задача + файлы (сфокусированные по мемо) ────────────
    shared = f"## Задача\n{task}\n\n"
    _log("Задача", task)

    for fpath in files:
        try:
            content = _focused_file_content(fpath, nav_memo)
            fname   = Path(fpath).name
            shared += f"## Файл: {fname}\n```\n{content}\n```\n\n"
            _log(f"Файл: {fname}", f"```\n{content}\n```")
        except Exception as e:
            print(f"   ⚠️  {fpath}: {e}")

    # ── Навигация (tree-sitter) — однократно в начале ────────────────────────
    nav_stage = next((s for s in stages if s.get("type") in ("navigator", "rwkv")), None)
    if nav_stage and nav_memo:
        shared += f"## Навигация\n{nav_memo}\n\n"
        _log("Навигация", nav_memo)
        print(f"\n[nav] navigator")
        print(f"   → {nav_memo[:120]}")
    elif nav_stage:
        print(f"\n[nav] navigator  — нет мемо (пропуск)")

    # Только agent-стадии; coder и reviewer — отдельно в конце
    agent_stages   = [s for s in stages if s.get("type") == "agent"]
    coder_stage    = next((s for s in stages if s.get("type") == "coder"), None)
    reviewer_stage = next((s for s in stages if s.get("type") == "reviewer"), None)

    # ── Прогоны агентов (passes раз) ─────────────────────────────────────────
    coder_response = ""

    for pass_num in range(1, passes + 1):
        if passes > 1:
            print(f"\n{'═' * 50}")
            print(f"  Прогон {pass_num} / {passes}")
            print(f"{'═' * 50}")
            _log(f"─── Прогон {pass_num} / {passes}", "")

        for stage in agent_stages:
            sid        = stage.get("id", "agent")
            agent_name = stage.get("name", sid)
            role       = stage.get("role", "").replace("{task}", task)
            label      = f"{sid}" + (f"  (прогон {pass_num})" if passes > 1 else "")

            print(f"\n[agent] {label}")

            # Читаем текст агента из agents/<name>/
            agent_text = _read_agent_text(agent_name)
            if not agent_text:
                print(f"   ⚠️  Агент '{agent_name}' — текст не найден — пропуск")
                continue

            # Full eval: знания агента + shared контекст пайплайна
            # (KV-cache continuation ломается на больших инжектах — save_pattern надёжен)
            tmp_name = f"_agent_{sid}"
            combined = f"## Знания агента: {agent_name}\n{agent_text}\n\n---\n\n{shared}"
            save_pattern(tmp_name, combined, grow_policy="retrain")
            memo_result = ask_pattern(tmp_name, role + "\n/no_think", grow=False)
            memo = memo_result or ""

            # Удаляем временный паттерн
            for ext in (".pkl", ".json"):
                p = Path(PATTERNS_DIR) / f"{tmp_name}{ext}"
                if p.exists():
                    p.unlink()

            if memo and memo.strip():
                title = f"{sid}" + (f" · прогон {pass_num}" if passes > 1 else "")
                shared += f"## {title}\n{memo.strip()}\n\n"
                _log(title, memo.strip())
            else:
                print(f"   ⚠️  Агент '{agent_name}' не дал мемо — пропуск")

    # ── Кодер — финальная стадия ─────────────────────────────────────────────
    if coder_stage:
        role       = coder_stage.get("role", "").replace("{task}", task)
        file_names = ", ".join(Path(f).name for f in files)

        print(f"\n[coder] coder")
        print(f"   Контекст: {len(shared.split())} слов  |  {len(files)} файл(ов)")

        save_pattern("_pipeline", shared, grow_policy="retrain")
        coder_q = (
            f"Задача: {task}\n\n"
            f"Файлы: {file_names}\n\n"
            f"{role}"
        )
        coder_response = ask_pattern("_pipeline", coder_q, grow=False)

        if coder_response:
            _log("Coder (diff)", coder_response)

    # ── Ревьюер — проверка diff по tree-sitter ────────────────────────────────
    if reviewer_stage and coder_response and "```" in coder_response:
        from cognit_patch import _extract_all_diffs, _diff_target

        role = reviewer_stage.get("role", "").replace("{task}", task)
        print(f"\n[reviewer] reviewer")

        # Tree-sitter структура затронутых файлов
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

        # Контекст ревьюера: задача + исходные файлы + diff + tree
        review_ctx = f"## Задача\n{task}\n\n"
        for fpath in files:
            try:
                content = Path(fpath).read_text(encoding="utf-8", errors="ignore")[:6000]
                review_ctx += f"## Файл: {Path(fpath).name}\n```\n{content}\n```\n\n"
            except Exception:
                pass
        review_ctx += f"## Diff от кодера\n```diff\n{coder_response}\n```\n\n"
        if tree_info:
            review_ctx += f"## Структура файлов (tree-sitter)\n{tree_info}\n"

        save_pattern("_pipeline", review_ctx, grow_policy="retrain")
        reviewed = ask_pattern("_pipeline", role + "\n/no_think", grow=False)

        if reviewed and "```" in reviewed:
            coder_response = reviewed
            _log("Reviewer (исправлен)", reviewed)
            print("   ✓ Diff скорректирован")
        else:
            _log("Reviewer (без изменений)", reviewed or "(нет ответа)")
            print("   ✓ Diff без изменений")

    if coder_response and "```" in coder_response:
        print("\n💡 Diff готов → /patch")

    if log_path and log_path.exists():
        print(f"\n📄 Лог: {log_path}")

    return coder_response or ""


def list_patterns():
    core.print_patterns_list(PATTERNS_DIR)


def _find_or_load_agent(agent_name: str) -> str | None:
    """
    Находит или авто-загружает паттерн агента по имени.
    Ищет паттерн agent_name; если нет — читает .echo.json → client_project/agents/<name>/.
    """
    if core.pattern_exists(PATTERNS_DIR, agent_name):
        return agent_name

    cfg = _echo_config()
    if not cfg:
        print(f"❌ Паттерн '{agent_name}' не найден и .echo.json отсутствует.")
        print(f"   Загрузи вручную: /load {agent_name} @agents/{agent_name}/")
        return None

    agent_dir = Path(cfg.get("client_project", "")) / "agents" / agent_name
    if not agent_dir.exists():
        print(f"❌ Не найден агент '{agent_name}': {agent_dir}")
        available = _list_agents(PATTERNS_DIR)
        if available:
            names = ", ".join(n for n, _ in available)
            print(f"   Доступные агенты: {names}")
        else:
            print("   Создай agents/: python cognit_setup.py agents")
        return None

    print(f"   Авто-загрузка агента '{agent_name}': {agent_dir} ...")
    _load_path(agent_name, str(agent_dir))
    return agent_name if core.pattern_exists(PATTERNS_DIR, agent_name) else None


def _read_pattern_source(name: str) -> str:
    """Перечитывает исходные файлы паттерна из метаданных."""
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

    Надёжная альтернатива _chain_ask для больших инжектов (700+ токенов).
    Каждый текст объединяется через separator и прогоняется через полный eval.
    Временный паттерн удаляется после использования.
    """
    combined = "\n\n---\n\n".join(t for t in texts if t and t.strip())
    if not combined.strip():
        return ""
    print(f"\n🔄 Full-eval: {len(combined)} символов")
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


def _do_edit(file_path: str, task: str, base_pattern: str | None = None) -> str:
    """
    Читает файл СВЕЖО и просит модель выдать unified diff.

    Разница с /load + ask:
      - Файл читается прямо сейчас (не из KV-cache) → точные строки, точные номера
      - base_pattern загружается как фон (проектный контекст, агент)
      - Если base_pattern не задан — временный паттерн из самого файла

    Возвращает ответ модели (ожидается unified diff) или "" при ошибке.
    """
    p = Path(file_path)
    if not p.exists():
        print(f"❌ Файл не найден: {file_path}")
        return ""

    content = p.read_text(encoding="utf-8", errors="ignore")
    n_lines = len(content.splitlines())
    print(f"\n📄 Читаю файл: {p.resolve()}  ({n_lines} строк, {len(content)} символов)")

    file_block = (
        f"# Файл для редактирования: {p.resolve()}\n\n"
        f"```\n{content}\n```"
    )
    edit_question = (
        f"{task}\n\n"
        "Выдай unified diff для этого изменения. Используй точные номера строк из файла.\n"
        "Формат:\n"
        "--- a/имя_файла\n"
        "+++ b/имя_файла\n"
        "@@ -N,M +N,M @@\n"
        " контекстная строка\n"
        "-удалённая строка\n"
        "+добавленная строка\n"
        "Только diff, без объяснений."
    )

    if base_pattern and core.pattern_exists(PATTERNS_DIR, base_pattern):
        # Full eval: source + file → reliable for large files
        base_source = _read_pattern_source(base_pattern)
        return _ephemeral_eval_ask([base_source, file_block], edit_question, "_edit_tmp")
    else:
        # Нет активного паттерна — временный из самого файла
        return _ephemeral_eval_ask([file_block], edit_question, "_edit_tmp")


# =============================================================================
# CLI
# =============================================================================
HELP = """
📖 КОМАНДЫ:
  use <имя>            — выбрать активный паттерн (память модели)
  <вопрос>             — спросить  (следует политике паттерна: grow сохраняет, retrain — нет)
  ? <вопрос>           — спросить с временной сменой политики (grow↔retrain)
  route <задача>       — найти файлы через tree-sitter индекс → запустить пайплайн
  /load <имя> @<файл>        — загрузить файл как паттерн
  /load <имя> @<д1> @<д2>   — composite: несколько папок в одном eval-проходе
  /load <имя> <текст>        — загрузить текст как паттерн
  /edit @<файл> <задача> — читает файл свежо → точный unified diff → /patch
  /patch               — применить unified diff из последнего ответа к файлу
  /patch @<файл>       — применить к конкретному файлу (override)
  /agents              — список доступных агентов
  /agent <имя> [имя2 ...] — включить ambient агенты (все вопросы через [паттерн + агенты])
  /agent off           — выключить ambient агенты
  /review @<файл>      — код-ревью (агент style + файл, паттерн не меняется)
  /review arch @<файл> — ревью через любой агент из agents/
  /review              — ревью последнего ответа модели (diff/код)
  /style @<файл>       — проверка стиля файла
  /index               — обзор проекта (tree-sitter символы)
  /index <запрос>      — поиск символов по запросу (BM25)
  /index --rebuild     — пересобрать индекс
  /list                — показать все паттерны
  /help                — эта справка
  /exit                — выход

ПРИМЕР:
  /load auth @src/auth.py             ← загрузить файл
  use auth                            ← переключиться на него
  что делает функция login?           ← просто пишешь вопрос
  исправь JWT                         ← просишь исправление
  /agent style arch                   ← включить ambient: стиль + архитектура одновременно
  /review @src/auth.py                ← ревью файла (стиль + код)
  /edit @src/auth.py исправь JWT      ← читает файл свежо → точный diff
  /patch                              ← применяем diff к auth.py
  route добавить rate limiting        ← найти нужные файлы → пайплайн
  /index bayes update                 ← найти символы по запросу
"""


def _load_path(name: str, raw_path: str, force_policy: str = None):
    """Загружает файл или директорию как паттерн."""
    p = Path(raw_path)
    if p.is_dir():
        files = core.collect_text_files(p)
        if not files:
            print(f"❌ Папка пуста или нет подходящих файлов: {raw_path}")
            return
        texts, paths = [], []
        header = f"# Проект: {p.name}\nПуть: {p.resolve()}\nФайлов: {len(files)}\n"
        texts.append(header)
        for f in files:
            try:
                texts.append(f"# {f.resolve()}\n\n{f.read_text(encoding='utf-8', errors='ignore')}")
                paths.append(str(f.resolve()))
            except Exception:
                pass
        print(f"   Загружаю {len(files)} файлов из {raw_path}/  (git ls-files / rglob)")
        save_pattern(name, "\n\n---\n\n".join(texts), source_files=paths, grow_policy=force_policy)
    elif p.is_file():
        save_pattern(name, p.read_text(encoding="utf-8", errors="ignore"),
                     source_files=[str(p.resolve())], grow_policy=force_policy)
    else:
        print(f"❌ Не найден файл или папка: {raw_path}")


def _load_mix(name: str, raw_paths: list[str], force_policy: str = None):
    """Загружает несколько файлов/папок как единый composite паттерн."""
    all_texts: list[str] = []
    all_sources: list[str] = []
    for raw_path in raw_paths:
        p = Path(raw_path)
        if p.is_dir():
            files = core.collect_text_files(p)
            if not files:
                print(f"   ⚠️  Папка пуста или нет подходящих файлов: {raw_path}")
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
            print(f"   + {p.name}/  ({len(files)} файлов)")
        elif p.is_file():
            all_texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            all_sources.append(str(p))
            print(f"   + {p.name}")
        else:
            print(f"   ⚠️  Не найден: {raw_path}")
    if not all_texts:
        print("❌ Ни один путь не содержит файлов.")
        return
    combined = "\n\n===\n\n".join(all_texts)
    print(f"   Composite: {name}  (источников: {len(raw_paths)}, файлов: {len(all_sources)})")
    save_pattern(name, combined, source_files=all_sources, grow_policy=force_policy)


def _hint_patterns():
    core.hint_patterns(PATTERNS_DIR)


def _auto_init_agents():
    """Авто-загрузка ненайденных агентов из agents/ клиентского проекта."""
    cfg = _echo_config()
    client = cfg.get("client_project", "")
    agents = _list_agents(PATTERNS_DIR)
    missing = [(name, loaded) for name, loaded in agents if not loaded]

    if missing and client:
        print(f"\n🔄 Авто-инициализация агентов ({len(missing)} из {len(agents)})...")
        for name, _ in missing:
            agent_dir = Path(client) / "agents" / name
            if agent_dir.exists():
                print(f"   Загружаю агента: {name}  ({agent_dir})")
                _load_path(name, str(agent_dir))
            else:
                print(f"   ⚠️  Папка агента не найдена: {agent_dir}")
        print()


_code_index_cache = None
_code_index_project = ""


def _get_code_index(rebuild: bool = False):
    """Возвращает CodeIndex для клиентского проекта (кеширует в модуле)."""
    global _code_index_cache, _code_index_project
    if not _HAS_INDEX:
        print("⚠️  tree-sitter не установлен. pip install tree-sitter tree-sitter-python")
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


def _route_via_index(task: str) -> tuple[list[str], str]:
    """
    Находит релевантные файлы для задачи через tree-sitter индекс.

    1. CodeIndex.search(task) → кандидаты (файлы + символы)
    2. Формирует контекстную заметку (какие символы нашлись)
    3. Возвращает (files, context_note)

    Без переключения моделей — всё in-process.
    """
    idx = _get_code_index()
    if idx is None:
        print("⚠️  client_project не задан в .echo.json — маршрутизация невозможна")
        return [], ""

    # Поиск по индексу
    results = idx.search(task, top_k=15)
    if not results:
        print("⚠️  Индекс не нашёл релевантных символов")
        return [], ""

    # Уникальные файлы (по порядку score)
    seen: set[str] = set()
    files: list[str] = []
    for r in results:
        if r.filepath not in seen:
            seen.add(r.filepath)
            files.append(r.filepath)

    # Контекстная заметка для пайплайна
    context_note = idx.search_summary(task, top_k=10)

    # Показываем что нашли
    print(f"\n📍 Найдено {len(results)} символов в {len(files)} файлах:")
    for f in files:
        rel = f.replace(idx.project_dir, "").lstrip("/\\")
        file_results = [r for r in results if r.filepath == f]
        symbols_str = ", ".join(r.symbol.name for r in file_results[:3])
        print(f"   • {rel}  ({symbols_str})")

    return files, context_note


def cli_loop() -> None:
    """
    Главный интерактивный цикл.
    Возвращает None при /exit.
    """
    print(f"\n🧠 Transformer · {MODEL_NAME[:28]} · {REPO_NAME}/{BRANCH_NAME}")

    list_patterns()
    _auto_init_agents()

    active = None
    active_policy = None
    last_response = ""
    ambient_agents = []  # список агентов для ambient-режима ([] = выключен)

    if not active:
        print("\n💡 Введи вопрос или задачу — Cognit найдёт нужные файлы автоматически.")
        print("   Или /load <имя> @<путь> для работы с конкретным файлом.")

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

            # ── Slash-команды ────────────────────────────────────────────────
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=2)
                cmd = parts[0].lower() if parts else ""

                if cmd in ("exit", "quit"):
                    print("👋 Выход")
                    return None

                elif cmd == "help":
                    print(HELP)

                elif cmd == "list":
                    list_patterns()

                elif cmd == "index":
                    query = " ".join(parts[1:]).strip() if len(parts) > 1 else ""
                    idx = _get_code_index(rebuild=("--rebuild" in query))
                    if idx is None:
                        print("❌ Индекс недоступен (проверьте client_project и tree-sitter)")
                        continue
                    query = query.replace("--rebuild", "").strip()
                    if query:
                        print(idx.search_summary(query))
                    else:
                        print(idx.project_summary())

                elif cmd == "load":
                    if len(parts) < 3:
                        print("❌ Использование: /load <имя> @<файл/папка>  или  /load <имя> <текст>")
                        print("   Несколько путей: /load <имя> @dir1/ @dir2/  (composite паттерн)")
                        print("   Префикс ? форсирует retrain: /load ?name @path")
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
                    active = name  # автоматически переключаемся на новый паттерн
                    active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                    print(f"   Активный паттерн: {active}")

                elif cmd == "edit":
                    # /edit @src/config.py поменяй 0-1 на проценты
                    if len(parts) < 3 or not parts[1].startswith("@"):
                        print("❌ Использование: /edit @<файл> <задача>")
                        print("   Пример: /edit @src/config.py поменяй 0-1 на проценты")
                        continue
                    edit_path = parts[1][1:]
                    edit_task = parts[2]
                    result = _do_edit(edit_path, edit_task, active)
                    if result:
                        last_response = result
                        print("\n💡 Применить? → /patch")

                elif cmd == "patch":
                    if not active:
                        print("❌ Сначала выбери паттерн: use <имя>")
                        continue
                    diffs = _extract_all_diffs(last_response) if last_response else []
                    if not diffs:
                        if not last_response:
                            print("❌ Нет предыдущего ответа. Задай вопрос об изменениях, затем /patch.")
                            continue
                        print("   Diff не найден — переформатирую предыдущий ответ...")
                        base_source = _read_pattern_source(active)
                        last_response = _ephemeral_eval_ask(
                            [base_source, f"Предыдущий ответ:\n\n{last_response}"],
                            "Покажи изменения из этого ответа в виде unified diff (```diff блок).",
                            "_patch_tmp",
                        )
                        diffs = _extract_all_diffs(last_response)
                    if not diffs:
                        print("❌ Модель не вернула unified diff. Попроси явно.")
                        continue

                    # Resolve paths relative to client_project
                    cfg_patch = _echo_config()
                    client_dir_patch = cfg_patch.get("client_project", "")
                    # Explicit @file override (only for single diff)
                    explicit_target = parts[1].lstrip("@") if len(parts) > 1 else None

                    if len(diffs) > 1:
                        print(f"\n📋 Найдено {len(diffs)} diff-блоков")

                    applied_any = False
                    for diff in diffs:
                        target = explicit_target or _diff_target(diff)
                        if not target:
                            print("❌ Файл не определён. Используй: /patch @<файл>")
                            continue
                        # Resolve relative paths against client_project
                        if not os.path.isabs(target) and not os.path.exists(target) and client_dir_patch:
                            target = os.path.join(client_dir_patch, target)
                        is_new = _is_new_file_diff(diff) or not os.path.exists(target)
                        preview = diff.split('\n')
                        label = "НОВЫЙ ФАЙЛ" if is_new else target
                        print(f"\n📋 Diff ({len(preview)} строк)  →  {label}")
                        print("─" * 60)
                        for ln in preview[:50]:
                            print(ln)
                        if len(preview) > 50:
                            print(f"   ... (ещё {len(preview) - 50} строк)")
                        print("─" * 60)
                        try:
                            action = "Создать" if is_new else "Применить"
                            ans = input(f"   {action} '{target}'? [y/N] ").strip().lower()
                        except (KeyboardInterrupt, EOFError):
                            continue
                        if ans == "y":
                            _apply_patch(diff, target)
                            applied_any = True

                    # После /patch в пайплайне — сбросить контекст кодера
                    if applied_any and active == "_pipeline":
                        active = None
                        active_policy = None
                        print("🔄 Пайплайн завершён, контекст сброшен.")

                elif cmd == "agents":
                    agents = _list_agents(PATTERNS_DIR)
                    if not agents:
                        print("📂 Нет агентов. Создай: python cognit_setup.py agents")
                    else:
                        print(f"\n📋 Агентов: {len(agents)}")
                        for name, loaded in agents:
                            tag = " [загружен]" if loaded else ""
                            print(f"   • {name}{tag}")
                        print()

                elif cmd == "agent":
                    # Парсим все имена агентов: /agent arch style context
                    agent_names = user_input[1:].split()[1:]
                    if not agent_names or agent_names[0].lower() == "off":
                        ambient_agents = []
                        print("🔕 Ambient агенты отключены")
                    else:
                        loaded_names = []
                        for agent_name in agent_names:
                            loaded = _find_or_load_agent(agent_name)
                            if loaded:
                                loaded_names.append(agent_name)
                        if loaded_names:
                            ambient_agents = loaded_names
                            agents_str = " + ".join(loaded_names)
                            print(f"🔗 Ambient агенты: {agents_str}")
                            print(f"   Каждый вопрос будет проходить через [{active or '?'} + {agents_str}]")
                            print(f"   Паттерн не сохраняется в ambient режиме (эфемерно).")
                            print(f"   Отключить: /agent off")

                elif cmd in ("review", "style"):
                    default_q = (
                        "Сделай код-ревью с учётом правил проекта. Что нужно исправить?"
                        if cmd == "review" else
                        "Проверь соответствие нашим стандартам стиля. Перечисли нарушения."
                    )
                    # Синтаксис: /review [агент] @<файл> [вопрос]
                    # parts[1] — агент или @file; если не @, это имя агента
                    rest = parts[1:]
                    agent_name = "style"  # по умолчанию
                    if rest and not rest[0].startswith("@"):
                        agent_name = rest[0]
                        rest = rest[1:]

                    # Источник контента: @file/папка или last_response
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
                            label = f"{content_path}/ ({len(files)} файлов)"
                        else:
                            print(f"❌ Не найден: {content_path}")
                            continue
                    elif last_response:
                        content = last_response
                        question = " ".join(rest) if rest else default_q
                        label = "последний ответ модели"
                    else:
                        print(f"❌ Укажи файл: /{cmd} [агент] @<файл>  или сначала задай вопрос")
                        agents = _list_agents(PATTERNS_DIR)
                        if agents:
                            print(f"   Агенты: {', '.join(n for n, _ in agents)}")
                        continue
                    print(f"   Агент: {agent_name}  |  Анализирую: {label}")
                    agent_source = _read_agent_text(agent_name)
                    if agent_source:
                        last_response = _ephemeral_eval_ask(
                            [agent_source, content], question, "_review_tmp")
                    else:
                        print(f"⚠️  Агент {agent_name} не найден")

                else:
                    print(f"❌ Неизвестная команда '/{cmd}'. Введите '/help'.")

            # ── use <имя> ────────────────────────────────────────────────────
            elif user_input.lower().startswith("use ") or user_input.lower() == "use":
                name = user_input[4:].strip()
                if not name:
                    print("❌ Использование: use <имя>")
                    _hint_patterns()
                    continue
                if not (Path(PATTERNS_DIR) / f"{name}.pkl").exists():
                    print(f"❌ Паттерн '{name}' не найден.")
                    _hint_patterns()
                    continue
                active = name
                active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                policy_label = " [~grow]" if active_policy == "grow" else " [retrain]"
                print(f"✅ Активный паттерн: {active}{policy_label}")

            # ── ? вопрос → временно сменить политику ────────────────────────
            elif user_input.startswith("?"):
                question = user_input[1:].strip()
                if not question:
                    print("❌ Введите вопрос после ?")
                    continue
                if not active:
                    print("❌ Сначала выбери паттерн: use <имя>")
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
                    # ? флипает политику: grow→не сохранять, retrain→сохранять
                    flipped_grow = (active_policy == "retrain")
                    last_response = ask_pattern(active, question, grow=flipped_grow)

            # ── route <задача> → найти файлы через tree-sitter индекс ───────
            elif user_input.lower().startswith("route ") or user_input.lower() == "route":
                task = user_input[6:].strip()
                if not task:
                    print("❌ Использование: route <задача>")
                    print("   Пример: route добавить rate limiting для POST /login")
                    continue
                files, nav_memo = _route_via_index(task)
                if files:
                    try:
                        ans = input("\n   Запустить пайплайн? [Y/n] ").strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        ans = "n"
                    if ans != "n":
                        last_response = _run_pipeline(task, files, nav_memo)
                        active = "_pipeline"
                        active_policy = "retrain"

            # ── просто текст → ask ───────────────────────────────────────────
            else:
                if not active:
                    # Нет паттерна — авто-маршрутизируем через tree-sitter индекс
                    files, nav_memo = _route_via_index(user_input)
                    if files:
                        try:
                            ans = input("\n   Запустить пайплайн? [Y/n] ").strip().lower()
                        except (KeyboardInterrupt, EOFError):
                            ans = "n"
                        if ans != "n":
                            last_response = _run_pipeline(user_input, files, nav_memo)
                            active = "_pipeline"
                            active_policy = "retrain"
                    else:
                        print("❌ Не удалось найти файлы. Загрузи вручную: /load <имя> @<путь>")
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

                # После любого ask: если в ответе есть код — подсказываем /patch
                if last_response and "```" in last_response:
                    print("\n💡 Есть код в ответе → /patch")

        except KeyboardInterrupt:
            print("\n👋 Выход")
            return None
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


# =============================================================================
# HEADLESS-РЕЖИМ (для post-commit хука и скриптов)
# =============================================================================
def headless_refresh_file(file_path: str):
    """
    Пересоздаёт все паттерны, которые ссылаются на file_path.
    Вызывается post-commit хуком: python cognit_transformer.py --refresh-file src/auth.py
    Не задаёт интерактивных вопросов — всё автоматически.
    """
    if not Path(PATTERNS_DIR).exists():
        return

    if not os.path.exists(file_path):
        print(f"   ⚠️  Файл не найден: {file_path}")
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

        # Проверяем, изменился ли файл с момента создания паттерна
        if all(s["hash"] == new_hash for s in affected):
            print(f"   ✓ {name}: актуален")
            continue

        if policy == "grow":
            print(f"   ~ {name}: grow-паттерн, пропускаем (накапливает диалог)")
            continue

        print(f"   ♻  {name}: пересоздаю (изменён {file_path})...")
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
        print(f"   ✅ Обновлено паттернов: {refreshed}")
    elif refreshed == 0:
        print(f"   ℹ  Нет паттернов, использующих {file_path}")


def headless_status():
    """
    Проверяет актуальность всех паттернов.
    Выходной код: 0 = всё актуально, 1 = есть устаревшие.
    """
    if not Path(PATTERNS_DIR).exists():
        print("📂 Паттернов нет.")
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
                stale.append((meta["name"], path, "не найден"))
            elif core.file_hash(path) != src["hash"]:
                stale.append((meta["name"], path, "изменён"))

    if not stale:
        print("✅ Все паттерны актуальны.")
        sys.exit(0)

    print(f"⚠️  Устаревших паттернов: {len(stale)}")
    for name, path, reason in stale:
        print(f"   • {name}: {path}  ({reason})")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--refresh-file" and len(sys.argv) > 2:
            init_model()
            headless_refresh_file(sys.argv[2])
        elif sys.argv[1] == "--status":
            headless_status()  # не нужна модель
        else:
            print(f"Неизвестный флаг: {sys.argv[1]}")
            print("Флаги: --refresh-file <path>, --status")
            sys.exit(1)
    else:
        init_model()
        cli_loop()
