#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Cognit — Persistent Neural Context (Transformer)
====================================================
Концепция: KV-cache локальной LLM = "паттерн активности" = персистентная память.

Установка:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

Модель (скачать GGUF, например с HuggingFace):
    Qwen3-8B-Q4_K_M.gguf  (~4.7 GB VRAM)  ← рекомендую

Запуск:
    python cognit.py                   # через оркестратор (рекомендую)
    python cognit_transformer.py       # напрямую
    python cognit_transformer.py --status              # проверить устаревшие паттерны
    python cognit_transformer.py --refresh-file <path> # пересоздать паттерн вручную

Команды:
    use <имя>                   — выбрать активный паттерн
    <вопрос>                    — задать вопрос (следует политике: grow/retrain)
    ? <вопрос>                  — временно сменить политику (grow↔retrain)
    expand <задача>             — передать задачу в RWKV (handoff)
    route <задача>              — найти файлы через RWKV-индекс → вернуться с файлами
    /load <имя> @<путь>          — загрузить файл/папку как паттерн
    /load <имя> @<д1> @<д2>     — composite: несколько источников, один eval-проход
    /load ?<имя> @<путь>         — загрузить с принудительным retrain
    /load <имя> <текст>          — загрузить текст напрямую
    /list                       — список паттернов
    /patch                      — применить diff из последнего ответа к файлу
    /patch @<файл>              — применить к конкретному файлу
    /agents                     — список агентов из agents/ (авто-создаются при старте)
    /agent <имя> [имя2 ...]     — включить ambient (все вопросы через [паттерн + агенты])
    /agent off                  — выключить ambient
    /review @<файл>             — ревью файла (агент style, эфемерно)
    /review <агент> @<файл>     — ревью через конкретный агент
    /review                     — ревью последнего ответа модели
    /style @<файл>              — проверка стиля файла
    /help                       — справка
    /exit                       — выход
"""

import os
import re
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import cognit_core as core
from cognit_patch import _extract_diff, _diff_target, apply_patch as _apply_patch
from cognit_agents import echo_config as _echo_config, list_agents as _list_agents, read_agent_text as _read_agent_text

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
# </think> НЕ включаем: Qwen3 всегда выдаёт <think></think> перед ответом,
# останавливаемся только когда ответ закончен
STOP_TOKENS = ["<|im_end|>", "<|im_start|>"]

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
        "Если информации в контексте недостаточно, предложи пользователю команду: "
        "expand <краткое описание задачи> — это переключит на RWKV с доступом ко всей кодовой базе. "
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
    tokens = llm.tokenize(prompt.encode("utf-8"))

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

    with open(pattern_path, "rb") as f:
        state = pickle.load(f)

    llm.load_state(state)
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
    q_tokens = llm.tokenize(question_prompt.encode("utf-8"), add_bos=False)

    print(f"+{len(q_tokens)} tok")
    print(f"\n❓ {question}")
    print("─" * 50)

    # reset=False — не сбрасывать загруженный KV-cache!
    # Qwen3 выдаёт <think>...</think> перед ответом — скрываем этот блок
    collected = []
    answer_started = False

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
                after = full.split("</think>", 1)[1]
                for stop in STOP_TOKENS:
                    if stop in after:
                        after = after[:after.index(stop)]
                if after.strip():
                    print(after, end="", flush=True)
                answer_started = True
        else:
            print(clean_piece, end="", flush=True)

        tail = full[-60:]
        if any(stop in tail for stop in STOP_TOKENS):
            break
        if len(collected) >= MAX_TOKENS:
            break
        # Детектор зацикливания: последние 40 токенов == предыдущие 40
        if len(collected) >= 80:
            recent = "".join(collected[-40:])
            before = "".join(collected[-80:-40])
            if recent == before:
                print("\n⚠️  Зацикливание — генерация остановлена")
                break

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

    # Авто-детекция предложения expand от модели
    response = "".join(collected)
    handoff = _maybe_expand(name, question, response)
    return response, handoff  # tuple[str, dict | None]


def _peek_pattern(name: str, prompt: str) -> str | None:
    """
    Тихий инференс на паттерне без вывода и без сохранения состояния.
    Используется для роутинговых запросов к оркестратору.
    """
    pkl = Path(PATTERNS_DIR) / f"{name}.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            state = pickle.load(f)
        llm.load_state(state)
        # /no_think подавляет расширенный reasoning Qwen3 (экономит токены)
        q_tokens = llm.tokenize(_question_prompt(prompt + "\n/no_think").encode("utf-8"), add_bos=False)
        collected = []
        for token_id in llm.generate(q_tokens, reset=False, temp=0.1, top_p=0.9):
            piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
            collected.append(piece)
            full = "".join(collected)
            if any(stop in full[-60:] for stop in STOP_TOKENS):
                break
            if len(collected) >= 400:  # 120 → 400: think-блок один съедает ~200 токенов
                break
        response = "".join(collected)
        # Qwen3 оборачивает reasoning в <think>...</think> — берём только ответ
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()
        for stop in STOP_TOKENS:
            if stop in response:
                response = response[:response.index(stop)]
        return response.strip()
    except Exception:
        return None


def _generate_agent_card(name: str, agent_dir: Path):
    """
    Генерирует card.md для агента через его собственный KV-cache.
    Модель описывает себя: область знаний и когда её надо активировать.
    Вызывается после создания паттерна если card.md не существует.
    """
    card_path = agent_dir / "card.md"
    if card_path.exists():
        return
    if not core.pattern_exists(PATTERNS_DIR, name):
        return

    prompt = (
        f"Ты агент '{name}'. Опиши себя для системы автоматического роутинга.\n"
        "Ответь ровно двумя строками:\n"
        "Область: [что ты знаешь — одна строка]\n"
        "Применяй когда: [в каких ситуациях активировать — одна строка]\n"
        "Только эти две строки. Без комментариев."
    )
    print(f"   🃏 Генерирую card.md для {name}...", end=" ", flush=True)
    response = _peek_pattern(name, prompt)
    if not response:
        print("(нет ответа)")
        return

    lines = [l.strip() for l in response.splitlines() if l.strip() and not l.startswith("<")]
    card_lines = [l for l in lines if l.startswith("Область:") or l.startswith("Применяй когда:")]
    if not card_lines:
        card_lines = lines[:3]  # fallback: первые строки

    card_path.write_text(f"# {name}\n\n" + "\n".join(card_lines) + "\n", encoding="utf-8")
    print("✓")


def _maybe_expand(pattern_name: str, question: str, response: str) -> dict | None:
    """
    Проверяет, предложила ли модель expand. Если да — предлагает запустить его.
    Возвращает handoff dict или None.
    """
    lower = response.lower()
    if "expand" not in lower:
        return None

    # Пробуем вытащить задачу из фразы вида "expand <задача>"
    m = re.search(r'expand\s+([^\n`<]{5,150})', response, re.IGNORECASE)
    if m:
        suggested_task = m.group(1).strip().rstrip('.,!?`').strip()
    else:
        suggested_task = question

    print(f"\n🔄 Expand → RWKV: «{suggested_task[:70]}»")
    return _handoff_to_rwkv(pattern_name, suggested_task, original_question=question)


def _maybe_route_suggestion(question: str, response: str) -> dict | None:
    """
    Проверяет, предложила ли модель route (нужны конкретные файлы).
    Работает только если есть RWKV-индекс.
    Возвращает expand_route handoff или None.
    """
    lower = response.lower()
    if "route" not in lower:
        return None

    index = _find_rwkv_index()
    if not index:
        return None  # нет индекса — нечего предлагать

    m = re.search(r'route\s+([^\n`<]{5,150})', response, re.IGNORECASE)
    suggested_task = m.group(1).strip().rstrip('.,!?`').strip() if m else question

    print(f"\n🗺  Route → RWKV: «{suggested_task[:70]}»")
    return {"action": "expand_route", "task": suggested_task, "index": index,
            "original_question": question}


def _maybe_chain_handoff(active: str, question: str, response: str) -> dict | None:
    """
    После ответа агента проверяет: предложил ли он expand или route.
    Возвращает первый найденный handoff или None.
    Вызывается после _chain_ask чтобы цепочки агентов могли запускать переключение.
    """
    handoff = _maybe_expand(active, question, response)
    if handoff:
        return handoff
    return _maybe_route_suggestion(question, response)


def list_patterns():
    core.print_patterns_list(PATTERNS_DIR)


# =============================================================================
# CHAIN — агент-цепочки (/review, /style)
# =============================================================================

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


def _chain_ask(base_pattern: str, extra_text: str, question: str) -> str:
    """
    Стекает extra_text поверх KV-cache base_pattern и задаёт вопрос.
    Состояние НЕ сохраняется — эфемерная сессия (паттерн не меняется).
    """
    print(f"\n🔗 Цепочка: [{base_pattern}] + контент")
    if not load_pattern(base_pattern):
        return ""

    # Eval дополнительного контента поверх загруженного состояния
    injection = f"\n\n# Контент для анализа:\n\n{extra_text.strip()}\n"
    inj_tokens = llm.tokenize(injection.encode("utf-8"), add_bos=False)
    if inj_tokens:
        if len(inj_tokens) > N_CTX // 2:
            print(f"⚠️  Контент большой ({len(inj_tokens)} токенов), обрезаем")
            inj_tokens = inj_tokens[:N_CTX // 2]
        print(f"   Контент: {len(inj_tokens)} токенов  (eval, не сохраняется)")
        llm.eval(inj_tokens)

    q_prompt = _question_prompt(question)
    q_tokens = llm.tokenize(q_prompt.encode("utf-8"), add_bos=False)

    print(f"\n❓ {question}")
    print("─" * 50)

    collected = []
    answer_started = False

    for token_id in llm.generate(q_tokens, reset=False, temp=0.3, top_p=0.9, repeat_penalty=1.1):
        piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        collected.append(piece)
        full = "".join(collected)

        clean_piece = piece
        for stop in STOP_TOKENS:
            if stop in clean_piece:
                clean_piece = clean_piece[:clean_piece.index(stop)]

        if not answer_started:
            if "</think>" in full:
                after = full.split("</think>", 1)[1]
                for stop in STOP_TOKENS:
                    if stop in after:
                        after = after[:after.index(stop)]
                if after.strip():
                    print(after, end="", flush=True)
                answer_started = True
        else:
            print(clean_piece, end="", flush=True)

        tail = full[-60:]
        if any(stop in tail for stop in STOP_TOKENS):
            break
        if len(collected) >= MAX_TOKENS:
            break
        if len(collected) >= 80:
            recent = "".join(collected[-40:])
            before = "".join(collected[-80:-40])
            if recent == before:
                print("\n⚠️  Зацикливание — генерация остановлена")
                break

    if not answer_started:
        print("".join(collected), end="", flush=True)

    print("\n" + "─" * 50)
    print("   🔗 Цепочка завершена — паттерн не изменён")

    return "".join(collected)


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
        # Инжектируем содержимое файла поверх KV-cache паттерна
        return _chain_ask(base_pattern, file_block, edit_question)
    else:
        # Нет активного паттерна — временный из самого файла (не сохраняется)
        _EDIT_TMP = "_edit_tmp"
        _load_path(_EDIT_TMP, file_path)
        resp, _ = ask_pattern(_EDIT_TMP, edit_question, grow=False)
        return resp or ""


# =============================================================================
# CLI
# =============================================================================
HELP = """
📖 КОМАНДЫ:
  use <имя>            — выбрать активный паттерн (память модели)
  <вопрос>             — спросить  (следует политике паттерна: grow сохраняет, retrain — нет)
  ? <вопрос>           — спросить с временной сменой политики (grow↔retrain)
  expand <задача>      — передать задачу в RWKV и выгрузиться
  route <задача>       — найти файлы через RWKV-индекс → вернуться с файлами
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
  expand нужен контекст по auth       ← передать задачу в RWKV (обзор)
  route добавить rate limiting        ← найти нужные файлы через RWKV-индекс
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
    """
    При старте проверяет агентов из agents/ клиентского проекта.
    Если паттерн агента не создан — создаёт автоматически.
    Также создаёт оркестратор-паттерн из card.md файлов агентов.
    Вызывается один раз в начале cli_loop().
    """
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
                # Генерируем card.md из KV-cache если его ещё нет
                _generate_agent_card(name, agent_dir)
            else:
                print(f"   ⚠️  Папка агента не найдена: {agent_dir}")
        print()

    # Генерируем card.md для уже загруженных агентов у которых карточки нет
    if client:
        agents_root = Path(client) / "agents"
        for name, loaded in agents:
            if not loaded:
                continue
            agent_dir = agents_root / name
            if agent_dir.exists() and not (agent_dir / "card.md").exists():
                _generate_agent_card(name, agent_dir)

    # Оркестратор: создаём из card.md если паттерна ещё нет
    if client and not core.pattern_exists(PATTERNS_DIR, "orchestrator"):
        agents_root = Path(client) / "agents"
        cards = sorted(agents_root.glob("*/card.md")) if agents_root.exists() else []
        if cards:
            combined = "\n\n---\n\n".join(
                c.read_text(encoding="utf-8", errors="ignore") for c in cards
            )
            print(f"   🎯 Создаю оркестратор из {len(cards)} card.md...")
            save_pattern("orchestrator", combined, grow_policy="retrain")
            print()


# Ключевые слова для авто-подбора агентов.
# Агент активируется если вопрос содержит хотя бы одно слово из списка.
# Пополняется при появлении новых агентов — или через keywords.txt в папке агента.
_AGENT_KEYWORDS: dict[str, list[str]] = {
    "style":   ["стиль", "style", "форматирование", "naming", "конвенция",
                "оформление", "lint", "pep8", "отступ", "импорт"],
    "arch":    ["архитектур", "arch", "структур", "design", "паттерн", "pattern",
                "зависимост", "модул", "слой", "layer", "интерфейс"],
    "context": ["контекст", "context", "проект", "цель", "требовани", "бизнес",
                "зачем", "почему", "задача"],
}


def _load_agent_keywords(agent_name: str) -> list[str]:
    """Читает keywords.txt из папки агента, если есть. Дополняет _AGENT_KEYWORDS."""
    cfg = _echo_config()
    kw_path = Path(cfg.get("client_project", "")) / "agents" / agent_name / "keywords.txt"
    if not kw_path.exists():
        return []
    return [line.strip() for line in kw_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def _suggest_agents(question: str) -> list[str]:
    """
    Подбирает агентов по ключевым словам в вопросе.
    Только среди агентов у которых уже есть загруженный паттерн.
    """
    q = question.lower()
    suggested = []
    for agent_name, keywords in _AGENT_KEYWORDS.items():
        if not core.pattern_exists(PATTERNS_DIR, agent_name):
            continue
        extra = _load_agent_keywords(agent_name)
        if any(kw in q for kw in keywords + extra):
            suggested.append(agent_name)
    return suggested


def _orchestrator_agents(question: str, available: list[str]) -> list[str]:
    """
    Запрашивает оркестратор-паттерн для выбора агентов под конкретный вопрос.
    Возвращает список агентов или [] если оркестратора нет либо он не нужен.
    """
    if not core.pattern_exists(PATTERNS_DIR, "orchestrator") or not available:
        return []
    routing_prompt = (
        f"Вопрос пользователя: \"{question[:200]}\"\n"
        f"Доступные агенты: {', '.join(available)}\n"
        "Какие агенты нужны? Перечисли только имена через запятую. "
        "Если ни один не нужен — ответь: none"
    )
    print("   🎯 Оркестратор...", end=" ", flush=True)
    response = _peek_pattern("orchestrator", routing_prompt)
    if not response:
        print()
        return []
    result = [n for n in available if n.lower() in response.lower()]
    print(", ".join(result) if result else "none")
    return result


def _find_rwkv_index() -> str | None:
    """
    Ищет любой RWKV-паттерн в PATTERNS_DIR.
    Возвращает имя первого найденного или None.
    """
    for meta_path in sorted(Path(PATTERNS_DIR).glob("*.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("backend") == "rwkv":
                return meta["name"]
        except Exception:
            pass
    return None


def _handoff_to_rwkv(active: str, task: str, original_question: str = "") -> dict:
    """Передаёт задачу в RWKV. Возвращает dict для in-process switch через оркестратор."""
    meta = core.read_meta(PATTERNS_DIR, active) if active else {}
    sources = [s["path"] for s in meta.get("source_files", [])]
    core.save_expand_request(PATTERNS_DIR, task, active or "", sources)
    return {
        "action":            "expand",
        "task":              task,
        "original_question": original_question,
        "from_pattern":      active or "",
        "from_sources":      sources,
        "requested_at":      datetime.now().isoformat(),
    }


def cli_loop(auto_route: bool = False, pending: dict = None) -> dict | None:
    """
    Главный интерактивный цикл.
    Возвращает dict handoff (action=expand) или None при /exit.
    При запуске через когнит.py — используется pending для авто-маршрута.
    """
    print(f"\n🧠 Transformer · {MODEL_NAME[:28]} · {REPO_NAME}/{BRANCH_NAME}")

    list_patterns()
    _auto_init_agents()

    # Если RWKV оставил маршрут — загружаем файлы (авто или с вопросом)
    active = None
    active_policy = None
    last_response = ""
    ambient_agents = []  # список агентов для ambient-режима ([] = выключен)
    route = pending if (pending and pending.get("action") == "route") else core.load_last_route(PATTERNS_DIR)
    if route:
        age_min = int((datetime.now() - datetime.fromisoformat(route["routed_at"])).total_seconds() / 60)
        print(f"\n💡 Маршрут от RWKV ({age_min} мин назад): «{route['task']}»")
        print(f"   Файлов: {len(route['files'])}")
        for f in route["files"]:
            print(f"   • {f}")

        if auto_route:
            ans = "y"
            print("   Загружаю автоматически...")
        else:
            try:
                ans = input("   Загрузить все? [y/N] ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                ans = "n"

        if ans == "y":
            last_loaded = None
            for fpath in route["files"]:
                if os.path.exists(fpath):
                    stem = Path(fpath).stem.replace("-", "_").replace(".", "_")
                    _load_path(stem, fpath)
                    last_loaded = stem
                else:
                    print(f"   ⚠️  Не найден: {fpath}")
            if last_loaded:
                active = last_loaded  # сразу активируем последний загруженный
                active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                print(f"\n✅ Готов. Активный паттерн: {active}")

                # Zero-touch: авто-задаём исходный вопрос пользователя
                orig_q = route.get("original_question", "")
                if orig_q and route.get("auto"):
                    print(f"\n🎯 Авто-вопрос: «{orig_q}»")
                    valid_files = [f for f in route.get("files", []) if os.path.exists(f)]
                    if len(valid_files) == 1:
                        # Один файл → читаем свежо, точный diff
                        last_response = _do_edit(valid_files[0], orig_q, active)
                        if last_response:
                            print("\n💡 Применить? → /patch")
                    else:
                        # Несколько файлов → ask через KV-cache
                        last_response, handoff = ask_pattern(
                            active, orig_q, grow=(active_policy == "grow")
                        )
                        if handoff:
                            return handoff

    if not active:
        print("\nВыбери паттерн командой 'use <имя>' или '/help' для справки.")

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
                    diff = _extract_diff(last_response) if last_response else None
                    if not diff:
                        if not last_response:
                            print("❌ Нет предыдущего ответа. Задай вопрос об изменениях, затем /patch.")
                            continue
                        # Diff не найден — просим модель переформатировать
                        print("   Diff не найден в последнем ответе — запрашиваю у модели...")
                        last_response, _ = ask_pattern(
                            active,
                            "Покажи изменения из предыдущего ответа в виде unified diff (```diff блок).",
                            grow=False,
                        )
                        diff = _extract_diff(last_response)
                    if not diff:
                        print("❌ Модель не вернула unified diff. Попроси явно.")
                        continue
                    # Файл: из аргумента /patch @file, или из заголовка +++ diff
                    target = parts[1].lstrip("@") if len(parts) > 1 else _diff_target(diff)
                    if not target:
                        print("❌ Файл не определён. Используй: /patch @<файл>")
                        continue
                    if not os.path.exists(target):
                        print(f"❌ Файл не найден: {target}")
                        continue
                    preview = diff.split('\n')
                    print(f"\n📋 Diff ({len(preview)} строк)  →  {target}")
                    print("─" * 60)
                    for ln in preview[:50]:
                        print(ln)
                    if len(preview) > 50:
                        print(f"   ... (ещё {len(preview) - 50} строк)")
                    print("─" * 60)
                    try:
                        ans = input(f"   Применить к '{target}'? [y/N] ").strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        continue
                    if ans == "y":
                        _apply_patch(diff, target)

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
                    agent = _find_or_load_agent(agent_name)
                    if agent:
                        last_response = _chain_ask(agent, content, question)
                        handoff = _maybe_chain_handoff(active or agent, question, last_response)
                        if handoff:
                            return handoff

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
                _loaded = [n for n, ok in _list_agents(PATTERNS_DIR) if ok and n != "orchestrator"]
                effective_agents = ambient_agents or _orchestrator_agents(question, _loaded) or _suggest_agents(question)
                if effective_agents:
                    if not ambient_agents:
                        print(f"🤖 Авто-агенты: {', '.join(effective_agents)}")
                    agent_text = "\n\n---\n\n".join(
                        t for name in effective_agents if (t := _read_agent_text(name))
                    )
                    if agent_text:
                        last_response = _chain_ask(active, agent_text, question)
                        handoff = _maybe_chain_handoff(active, question, last_response)
                        if handoff:
                            return handoff
                    else:
                        last_response, handoff = ask_pattern(active, question, grow=False)
                        if handoff:
                            return handoff
                else:
                    # ? флипает политику: grow→не сохранять, retrain→сохранять
                    flipped_grow = (active_policy == "retrain")
                    last_response, handoff = ask_pattern(active, question, grow=flipped_grow)
                    if handoff:
                        return handoff

            # ── expand <задача> → handoff to RWKV ───────────────────────────
            elif user_input.lower().startswith("expand ") or user_input.lower() == "expand":
                task = user_input[7:].strip()
                if not task:
                    print("❌ Использование: expand <задача>")
                    print("   Пример: expand нужен контекст по модулю авторизации")
                    continue
                return _handoff_to_rwkv(active, task)

            # ── route <задача> → авто-маршрут через RWKV-индекс ─────────────
            elif user_input.lower().startswith("route ") or user_input.lower() == "route":
                task = user_input[6:].strip()
                if not task:
                    print("❌ Использование: route <задача>")
                    print("   Пример: route добавить rate limiting для POST /login")
                    continue
                index = _find_rwkv_index()
                if not index:
                    print("❌ Нет RWKV-индекса. Создай его сначала:")
                    print("   python cognit.py --rwkv")
                    print("   /load repo @src/")
                    print("   Затем работай как обычно — Transformer подхватит файлы.")
                    continue
                print(f"   Найден RWKV-индекс: {index}")
                return {
                    "action": "expand_route",
                    "task":   task,
                    "index":  index,
                }

            # ── просто текст → ask ───────────────────────────────────────────
            else:
                if not active:
                    print("❌ Сначала выбери паттерн: use <имя>")
                    _hint_patterns()
                    continue
                _loaded = [n for n, ok in _list_agents(PATTERNS_DIR) if ok and n != "orchestrator"]
                effective_agents = ambient_agents or _orchestrator_agents(user_input, _loaded) or _suggest_agents(user_input)
                if effective_agents:
                    if not ambient_agents:
                        print(f"🤖 Авто-агенты: {', '.join(effective_agents)}")
                    agent_text = "\n\n---\n\n".join(
                        t for name in effective_agents if (t := _read_agent_text(name))
                    )
                    if agent_text:
                        last_response = _chain_ask(active, agent_text, user_input)
                        handoff = _maybe_chain_handoff(active, user_input, last_response)
                        if handoff:
                            return handoff
                    else:
                        last_response, handoff = ask_pattern(active, user_input, grow=(active_policy == "grow"))
                        if handoff:
                            return handoff
                else:
                    # Следуем политике паттерна: grow сохраняет, retrain — нет
                    last_response, handoff = ask_pattern(active, user_input, grow=(active_policy == "grow"))
                    if handoff:
                        return handoff

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
        if sys.argv[1] == "--auto-route":
            init_model()
            cli_loop(auto_route=True)
        elif sys.argv[1] == "--refresh-file" and len(sys.argv) > 2:
            init_model()
            headless_refresh_file(sys.argv[2])
        elif sys.argv[1] == "--status":
            headless_status()  # не нужна модель
        else:
            print(f"Неизвестный флаг: {sys.argv[1]}")
            print("Флаги: --auto-route, --refresh-file <path>, --status")
            sys.exit(1)
    else:
        init_model()
        cli_loop()
