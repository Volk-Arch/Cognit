#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Cognit — Persistent Neural Context (RWKV)
============================================
RWKV — рекуррентная архитектура без ограничения на размер контекста.
Текст любой длины прогоняется чанками; состояние накапливается итеративно.
Сохранённый паттерн — фиксированного размера, независимо от объёма текста.

Установка:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

Модель (скачать GGUF с HuggingFace):
    RWKV-6-World-7B-Q4_K_M.gguf   (~4.1 GB VRAM)  ← рекомендую
    RWKV-6-World-3B-Q4_K_M.gguf   (~2.0 GB VRAM)  ← если нужно меньше
    RWKV-6-World-1B6-Q4_K_M.gguf  (~1.0 GB VRAM)  ← минимальный вариант

    Источник: https://huggingface.co/BlinkDL/rwkv-6-world

Запуск:
    python cognit.py --rwkv            # через оркестратор (рекомендую)
    python cognit_rwkv.py              # напрямую
    python cognit_rwkv.py --auto-expand  # принять pending задачу от Transformer

Команды:
    use <имя>                   — выбрать активный паттерн
    <вопрос>                    — задать вопрос (следует политике: grow/retrain)
    ? <вопрос>                  — временно сменить политику (grow↔retrain)
    route <задача>              — найти файлы для задачи → handoff на Transformer
    /load <имя> @<файл/папка>   — загрузить файл/папку (без лимита по размеру!)
    /load ?<имя> @<путь>        — загрузить с принудительным retrain
    /load <имя> <текст>         — загрузить текст напрямую
    /list                       — список паттернов
    /help                       — справка
    /exit                       — выход
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import cognit_core as core

# =============================================================================
# КОНФИГУРАЦИЯ — поменяй MODEL_PATH на свой путь к GGUF
# =============================================================================
MODEL_PATH    = "models/rwkv/RWKV-6-World-7B-Q4_K_M.gguf"
MODEL_NAME    = Path(MODEL_PATH).stem
N_GPU_LAYERS  = -1
MAX_TOKENS    = 512

REPO_NAME    = core.git_repo_name()
BRANCH_NAME  = core.git_branch()
PATTERNS_DIR = core.make_patterns_dir(REPO_NAME, BRANCH_NAME)

# RWKV не имеет KV-cache в традиционном смысле.
# N_CTX здесь — только буфер для генерации ответа, не для истории.
# Историю хранит рекуррентное состояние (фиксированного размера).
N_CTX = 1024

# Размер чанка при обработке длинного текста.
# Меньше = больше прогресс-баров, медленнее. 512 — хороший баланс.
CHUNK_SIZE = 512

# Стоп-последовательности RWKV-World
STOP_SEQS = ["\nUser:", "\n\nUser:", "\nSystem:"]

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
    """Загружает RWKV-модель в VRAM."""
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
    print(f"✅ RWKV загружен  |  {'GPU' if N_GPU_LAYERS != 0 else 'CPU'}")
    print(f"   Контекст: ∞ (рекуррентное состояние) | Чанк: {CHUNK_SIZE} токенов")
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
    """Промпт для загрузки контекста (формат RWKV-World)."""
    return (
        f"User: Прочитай и запомни этот контекст:\n\n{text.strip()}\n\n"
        "Assistant: Контекст загружен. Готов отвечать на вопросы. "
        "Если для задачи нужна точная работа с конкретными файлами — "
        "предложи команду route <задача>, чтобы передать управление Transformer.\n\n"
    )


def _question_prompt(question: str) -> str:
    """Промпт вопроса, добавляемый к уже загруженному состоянию."""
    return f"User: {question}\n\nAssistant:"


def save_pattern(name: str, text: str, source_files: list[str] = None, grow_policy: str = None):
    """
    Прогоняет текст через RWKV чанками, сохраняет рекуррентное состояние.

    Принципиальное отличие от Transformer:
    - Нет ограничения по длине — текст любого размера
    - Состояние после обработки фиксированного размера (не растёт с текстом)
    - Каждый чанк продолжает с того места, где остановился предыдущий

    grow_policy: 'grow' | 'retrain' | None (авто по ветке/пути)
    """
    if grow_policy is None:
        grow_policy = core.default_grow_policy(BRANCH_NAME, source_files or [])
    print(f"\n📝 Формирование паттерна '{name}'  [RWKV · {grow_policy}]...")

    prompt = _context_prompt(text)
    tokens = llm.tokenize(prompt.encode("utf-8"))
    total  = len(tokens)

    print(f"   Всего токенов: {total}  (без ограничений по размеру!)")
    print(f"   Обработка чанками по {CHUNK_SIZE} токенов...")

    llm.reset()
    for start in range(0, total, CHUNK_SIZE):
        chunk = tokens[start : start + CHUNK_SIZE]
        llm.n_tokens = 0  # сбрасываем позицию буфера, RWKV-состояние сохраняется
        llm.eval(chunk)
        done = min(start + CHUNK_SIZE, total)
        pct  = done * 100 // total
        bar  = "█" * (done * 30 // total) + "░" * (30 - done * 30 // total)
        print(f"\r   [{bar}] {pct:3d}%  {done}/{total} токенов", end="", flush=True)
    print()

    state = llm.save_state()

    pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
    meta_path    = Path(PATTERNS_DIR) / f"{name}.json"

    with open(pattern_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = pattern_path.stat().st_size / 1024
    meta = {
        "name":         name,
        "backend":      "rwkv",
        "model":        MODEL_NAME,
        "repo":         REPO_NAME,
        "branch":       BRANCH_NAME,
        "grow_policy":  grow_policy,
        "n_tokens":     total,
        "size_kb":      round(size_kb, 1),
        "preview":      text[:300],
        "saved_at":     datetime.now().isoformat(),
        "n_asks":       0,
        "source_files": [{"path": p, "hash": core.file_hash(p)} for p in (source_files or [])],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Паттерн сохранён: {pattern_path}  ({size_kb:.0f} KB)")
    print(f"   Обработано токенов: {total}  |  Размер паттерна фиксирован (не зависит от объёма текста)")
    print(f"   Вопросы: ask {name} <вопрос>")


def load_pattern(name: str) -> bool:
    """Восстанавливает рекуррентное состояние из файла."""
    pattern_path = Path(PATTERNS_DIR) / f"{name}.pkl"
    if not pattern_path.exists():
        print(f"❌ Паттерн '{name}' не найден. Используйте 'list' для просмотра.")
        return False

    meta_path = Path(PATTERNS_DIR) / f"{name}.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("backend") == "transformer":
            print(f"⚠️  Паттерн '{name}' создан Transformer-бэкендом.")
            print("   Загрузите его через cognit_transformer.py или пересоздайте здесь.")
            return False

    with open(pattern_path, "rb") as f:
        state = pickle.load(f)
    llm.load_state(state)
    return True


def ask_pattern(name: str, question: str, grow: bool = True):
    """
    Загружает рекуррентное состояние и отвечает на вопрос.
    Текст контекста НЕ передаётся повторно.

    grow=True:  после ответа состояние обновляется (растущая сессия).
    grow=False: состояние не меняется (peek).
    """
    print(f"\n💬 Загрузка паттерна '{name}'...")
    _check_and_refresh(name)
    if not load_pattern(name):
        return

    q_prompt = _question_prompt(question)
    q_tokens = llm.tokenize(q_prompt.encode("utf-8"), add_bos=False)

    print(f"   Токенов вопроса: {len(q_tokens)}  (контекст не передаётся повторно!)")
    print(f"\n❓ {question}")
    print("─" * 50)

    collected = []

    # RWKV склонен к зацикливанию — нужны более высокий repeat_penalty
    # и явный детектор петель
    for token_id in llm.generate(q_tokens, reset=False, temp=0.5, top_p=0.9, repeat_penalty=1.3):
        piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        collected.append(piece)
        full = "".join(collected)

        # Обрезаем стоп-последовательности перед выводом
        clean = piece
        for stop in STOP_SEQS:
            if stop in clean:
                clean = clean[:clean.index(stop)]

        print(clean, end="", flush=True)

        # Стоп по явным токенам
        tail = full[-60:]
        if any(stop in tail for stop in STOP_SEQS):
            break

        # Детектор зацикливания: последние 40 токенов == предыдущие 40
        if len(collected) >= 80:
            recent = "".join(collected[-40:])
            before = "".join(collected[-80:-40])
            if recent == before:
                break
        if len(collected) >= MAX_TOKENS:
            break

    print("\n" + "─" * 50)

    # Растущая сессия: сохраняем обновлённое состояние
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
            meta = {"name": name, "backend": "rwkv", "preview": question[:300]}

        meta["size_kb"]    = round(pattern_path.stat().st_size / 1024, 1)
        meta["updated_at"] = datetime.now().isoformat()
        meta.setdefault("n_asks", 0)
        meta["n_asks"] += 1

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"💾 Паттерн обновлён  ({meta['size_kb']} KB, диалогов: {meta['n_asks']})")

    # Авто-детекция предложения route от модели
    return _maybe_route(name, question, "".join(collected))


def _maybe_route(pattern_name: str, question: str, response: str) -> dict | None:
    """
    Проверяет, предложила ли модель route. Если да — предлагает запустить его прямо сейчас.
    Возвращает handoff dict или None.
    """
    import re
    lower = response.lower()
    if "route" not in lower:
        return None

    m = re.search(r'route\s+([^\n`<]{5,150})', response, re.IGNORECASE)
    if m:
        suggested_task = m.group(1).strip().rstrip('.,!?`').strip()
    else:
        suggested_task = question

    try:
        ans = input(f"\n🗺  Запустить route? [{suggested_task[:60]}] [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return None

    if ans != "n":
        return route_pattern(pattern_name, suggested_task)
    return None


def route_pattern(index_name: str, task: str) -> dict | None:
    """
    Использует RWKV-индекс для определения файлов, релевантных задаче.
    Возвращает handoff dict или None если файлы не определены.
    """
    print(f"\n🗺  Маршрутизация задачи через индекс '{index_name}'...")
    _check_and_refresh(index_name)
    if not load_pattern(index_name):
        return

    route_prompt = (
        f"User: Задача: {task}\n"
        "Перечисли ТОЛЬКО пути к файлам, которые нужно изучить для этой задачи. "
        "Один путь на строку. Никакого другого текста.\n\n"
        "Assistant:"
    )
    r_tokens = llm.tokenize(route_prompt.encode("utf-8"), add_bos=False)

    print(f"   Токенов запроса: {len(r_tokens)}")
    print("─" * 50)

    collected = []
    # Температура 0.2 — детерминированный вывод для списка файлов
    for token_id in llm.generate(r_tokens, reset=False, temp=0.2, top_p=0.9, repeat_penalty=1.3):
        piece = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        collected.append(piece)
        full = "".join(collected)

        clean = piece
        for stop in STOP_SEQS:
            if stop in clean:
                clean = clean[:clean.index(stop)]
        print(clean, end="", flush=True)

        tail = full[-60:]
        if any(stop in tail for stop in STOP_SEQS):
            break
        if len(collected) >= 200:
            break
        if len(collected) >= 80:
            recent = "".join(collected[-40:])
            before = "".join(collected[-80:-40])
            if recent == before:
                break

    print("\n" + "─" * 50)

    # Парсим пути из ответа
    raw = "".join(collected)
    for stop in STOP_SEQS:
        if stop in raw:
            raw = raw[:raw.index(stop)]

    files = []
    for line in raw.strip().splitlines():
        line = line.strip().lstrip("•-*123456789. ")
        # Принимаем строки, похожие на пути к файлам
        if line and len(line) > 2 and not line.startswith("#"):
            if "/" in line or "\\" in line or (
                "." in line and " " not in line and not line.startswith(".")
            ):
                files.append(line)

    if not files:
        print("⚠️  Не удалось распознать пути в ответе.")
        print("   Попробуйте переформулировать задачу или уточнить индекс.")
        return None

    print(f"\n📂 Релевантных файлов: {len(files)}")
    print("\n   Команды для Transformer:")
    for fpath in files:
        stem = Path(fpath).stem.replace("-", "_").replace(".", "_")
        print(f"   /load {stem} @{fpath}")

    route_path = core.save_route(PATTERNS_DIR, index_name, task, files)
    print(f"\n💾 Маршрут сохранён: {route_path}")
    return _handoff_to_transformer(index_name, task, files)


def list_patterns():
    core.print_patterns_list(PATTERNS_DIR)


# =============================================================================
# CLI
# =============================================================================
HELP = """
📖 КОМАНДЫ:
  use <имя>            — выбрать активный паттерн (память модели)
  <вопрос>             — спросить  (следует политике паттерна: grow сохраняет, retrain — нет)
  ? <вопрос>           — спросить с временной сменой политики (grow↔retrain)
  route <задача>       — какие файлы нужны? (использует активный паттерн как индекс)
  /load <имя> @<файл>  — загрузить файл как паттерн (без лимита по размеру!)
  /load <имя> <текст>  — загрузить текст как паттерн
  /list                — показать все паттерны
  /help                — эта справка
  /exit                — выход

ПРИМЕР (гибридный режим):
  /load repo @src/        ← проиндексировать всю кодовую базу
  use repo
  route добавить логин    ← какие файлы нужны? → передаётся Transformer автоматически
  что такое WorldModel?   ← обычный вопрос по индексу

ПРИМЕР (получение задачи от Transformer):
  Transformer пишет: expand нужен контекст по auth
  RWKV запускается автоматически, показывает задачу
  use repo → задай вопрос по задаче → route если нужны конкретные файлы
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
                texts.append(f"# {f.relative_to(p)}\n\n{f.read_text(encoding='utf-8', errors='ignore')}")
                paths.append(str(f))
            except Exception:
                pass
        print(f"   Загружаю {len(files)} файлов из {raw_path}/  (git ls-files / rglob)")
        save_pattern(name, "\n\n---\n\n".join(texts), source_files=paths, grow_policy=force_policy)
    elif p.is_file():
        save_pattern(name, p.read_text(encoding="utf-8", errors="ignore"),
                     source_files=[str(p)], grow_policy=force_policy)
    else:
        print(f"❌ Не найден файл или папка: {raw_path}")


def _hint_patterns():
    core.hint_patterns(PATTERNS_DIR)


def _handoff_to_transformer(index_name: str, task: str, files: list[str]) -> dict:
    """Передаёт задачу Transformer. Возвращает dict для in-process switch."""
    return {
        "action":    "route",
        "task":      task,
        "index":     index_name,
        "files":     files,
        "routed_at": datetime.now().isoformat(),
    }


def cli_loop(auto_expand: bool = False, pending: dict = None) -> dict | None:
    """
    Главный интерактивный цикл RWKV.
    Возвращает dict handoff (action=route) или None при /exit.
    """
    print(f"""
╔══════════════════════════════════════════════╗
║  🧠 Cognit · RWKV ({MODEL_NAME[:20]})
║  Рекуррентное состояние · ∞ контекст
╚══════════════════════════════════════════════╝""")

    list_patterns()

    # Авто-маршрут: Transformer вызвал route <задача> → RWKV сразу выполняет и возвращает файлы
    if pending and pending.get("action") == "expand_route":
        index = pending.get("index", "")
        task  = pending.get("task", "")
        print(f"\n⚡ Авто-маршрут: «{task}»")
        print(f"   Индекс: {index}")
        if core.pattern_exists(PATTERNS_DIR, index):
            result = route_pattern(index, task)
            return result  # None если файлы не разобраны, dict route если успешно
        else:
            print(f"❌ RWKV-паттерн '{index}' не найден в {PATTERNS_DIR}")
            print(f"   Индекс мог быть удалён или находится на другой ветке.")
            print(f"   Создай заново: python cognit.py --rwkv → /load repo @src/")
            return None

    # Если Transformer оставил запрос — показываем его
    expand = pending if (pending and pending.get("action") == "expand") else core.load_expand_request(PATTERNS_DIR)
    if expand:
        age_min = int((datetime.now() - datetime.fromisoformat(expand["requested_at"])).total_seconds() / 60)
        print(f"\n💡 Запрос от Transformer ({age_min} мин назад):")
        print(f"   Задача: «{expand['task']}»")
        if expand.get("from_pattern"):
            print(f"   Паттерн: {expand['from_pattern']}")
        if expand.get("from_sources"):
            print("   Файлы из контекста:")
            for src in expand["from_sources"]:
                print(f"   • {src}")
    else:
        print("\nВыбери паттерн командой 'use <имя>' или '/help' для справки.")

    active = None  # текущий активный паттерн
    active_policy = None

    while True:
        try:
            if active:
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
                        print("   Префикс ? форсирует retrain: /load ?name @path")
                        continue
                    name, raw = parts[1], parts[2]
                    force_policy = None
                    if name.startswith("?"):
                        name = name[1:]
                        force_policy = "retrain"
                    if raw.startswith("@"):
                        _load_path(name, raw[1:], force_policy=force_policy)
                    else:
                        save_pattern(name, raw, grow_policy=force_policy)
                    active = name  # автоматически переключаемся на новый паттерн
                    active_policy = (core.read_meta(PATTERNS_DIR, active) or {}).get("grow_policy", "retrain")
                    print(f"   Активный паттерн: {active}")

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

            # ── route <задача> — использует активный паттерн как индекс ─────
            elif user_input.lower().startswith("route ") or user_input.lower() == "route":
                task = user_input[6:].strip()
                if not task:
                    print("❌ Введите задачу: route <задача>")
                    continue
                if not active:
                    print("❌ Сначала выбери паттерн-индекс: use <имя>")
                    _hint_patterns()
                    continue
                result = route_pattern(active, task)
                if result:
                    return result

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
                # ? флипает политику: grow→не сохранять, retrain→сохранять
                flipped_grow = (active_policy == "retrain")
                result = ask_pattern(active, question, grow=flipped_grow)
                if result:
                    return result

            # ── просто текст → ask ───────────────────────────────────────────
            else:
                if not active:
                    print("❌ Сначала выбери паттерн: use <имя>")
                    _hint_patterns()
                    continue
                result = ask_pattern(active, user_input, grow=(active_policy == "grow"))
                if result:
                    return result

        except KeyboardInterrupt:
            print("\n👋 Выход")
            return None
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto-expand":
            init_model()
            cli_loop(auto_expand=True)
        else:
            print(f"Неизвестный флаг: {sys.argv[1]}")
            print("Флаги: --auto-expand")
            sys.exit(1)
    else:
        init_model()
        cli_loop()
