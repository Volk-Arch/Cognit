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

Команды:
    context <имя> <текст>    — обработать текст → сохранить паттерн (без лимита!)
    context <имя> @<файл>    — то же, но текст из файла
    ask  <имя> <вопрос>      — ответить + накопить диалог в паттерн
    peek <имя> <вопрос>      — ответить без изменения паттерна
    list                     — показать все паттерны
    help / exit
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime

import echo_core as core

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
print(f"✅ Модель загружена  |  Устройство: {'GPU' if N_GPU_LAYERS != 0 else 'CPU'}")
print(f"   Контекст: ∞ (рекуррентное состояние) | Чанк: {CHUNK_SIZE} токенов")
print(f"   Паттерны: {PATTERNS_DIR}  [{REPO_NAME} / {BRANCH_NAME}]")
os.makedirs(PATTERNS_DIR, exist_ok=True)

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
        "Assistant: Контекст загружен. Готов отвечать на вопросы.\n\n"
    )


def _question_prompt(question: str) -> str:
    """Промпт вопроса, добавляемый к уже загруженному состоянию."""
    return f"User: {question}\n\nAssistant:"


def save_pattern(name: str, text: str, source_files: list[str] = None):
    """
    Прогоняет текст через RWKV чанками, сохраняет рекуррентное состояние.

    Принципиальное отличие от Transformer:
    - Нет ограничения по длине — текст любого размера
    - Состояние после обработки фиксированного размера (не растёт с текстом)
    - Каждый чанк продолжает с того места, где остановился предыдущий
    """
    print(f"\n📝 Формирование паттерна '{name}'  [RWKV]...")

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
            print("   Загрузите его через echo_poc.py или пересоздайте здесь.")
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


def route_pattern(index_name: str, task: str):
    """
    Использует RWKV-индекс для определения файлов, релевантных задаче.
    Задаёт структурированный вопрос, парсит пути из ответа.
    Выводит готовые команды 'context' для echo_poc.py (Transformer).
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
        return

    print(f"\n📂 Релевантных файлов: {len(files)}")
    print("\n   Команды для echo_poc.py (Transformer):")
    for fpath in files:
        # Имя паттерна — stem без спецсимволов
        stem = Path(fpath).stem.replace("-", "_").replace(".", "_")
        print(f"   /load {stem} @{fpath}")

    route_path = core.save_route(PATTERNS_DIR, index_name, task, files)
    print(f"\n💾 Маршрут сохранён: {route_path}")
    print("   Запусти echo_poc.py — он предложит загрузить эти файлы автоматически.")


def list_patterns():
    core.print_patterns_list(PATTERNS_DIR)


# =============================================================================
# CLI
# =============================================================================
HELP = """
📖 КОМАНДЫ:
  use <имя>            — выбрать активный паттерн (память модели)
  <вопрос>             — спросить  (паттерн растёт)
  ? <вопрос>           — спросить без изменения паттерна
  route <задача>       — какие файлы нужны? (использует активный паттерн как индекс)
  /load <имя> @<файл>  — загрузить файл как паттерн (без лимита по размеру!)
  /load <имя> <текст>  — загрузить текст как паттерн
  /list                — показать все паттерны
  /help                — эта справка
  /exit                — выход

ПРИМЕР (гибридный режим):
  /load repo @src/        ← проиндексировать всю кодовую базу
  use repo
  route добавить логин    ← какие файлы нужны? (вывод → скопируй в echo_poc.py)
  что такое WorldModel?   ← обычный вопрос по индексу
"""


def _load_path(name: str, raw_path: str):
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
        save_pattern(name, "\n\n---\n\n".join(texts), source_files=paths)
    elif p.is_file():
        save_pattern(name, p.read_text(encoding="utf-8", errors="ignore"), source_files=[str(p)])
    else:
        print(f"❌ Не найден файл или папка: {raw_path}")


def _hint_patterns():
    core.hint_patterns(PATTERNS_DIR)


def _handoff_to_transformer():
    """
    После route предлагает передать управление Transformer-модели.
    Выгружает RWKV из VRAM, запускает echo_poc.py --auto-route в том же терминале.
    """
    print("\n🔄 Передать задачу Transformer (echo_poc.py)? [Y/n] ", end="", flush=True)
    try:
        ans = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        return

    if ans == "n":
        return

    print("   Выгружаю RWKV из VRAM...")
    global llm
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass  # llama-cpp-python без torch тоже освободит память

    print("   Запускаю Transformer... 🤝\n")
    import subprocess
    # Запускаем echo_poc.py в том же терминале; rwkv-процесс завершается
    subprocess.Popen([sys.executable, "echo_poc.py", "--auto-route"])
    sys.exit(0)


def cli_loop():
    print("""
╔══════════════════════════════════════════════╗
║  🧠 Cognit — Persistent Neural Context      ║
║  RWKV · Рекуррентное состояние · ∞ контекст  ║
╚══════════════════════════════════════════════╝""")

    list_patterns()
    print("\nВыбери паттерн командой 'use <имя>' или '/help' для справки.")

    active = None  # текущий активный паттерн

    while True:
        try:
            prompt_str = f"🧠 [{active}]> " if active else "🧠> "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue

            # ── Slash-команды ────────────────────────────────────────────────
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=2)
                cmd = parts[0].lower() if parts else ""

                if cmd in ("exit", "quit"):
                    print("👋 Выход")
                    break

                elif cmd == "help":
                    print(HELP)

                elif cmd == "list":
                    list_patterns()

                elif cmd == "load":
                    if len(parts) < 3:
                        print("❌ Использование: /load <имя> @<файл/папка>  или  /load <имя> <текст>")
                        continue
                    name, raw = parts[1], parts[2]
                    if raw.startswith("@"):
                        _load_path(name, raw[1:])
                    else:
                        save_pattern(name, raw)
                    active = name  # автоматически переключаемся на новый паттерн
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
                print(f"✅ Активный паттерн: {active}")

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
                route_pattern(active, task)
                _handoff_to_transformer()  # предложить передать управление

            # ── ? вопрос → peek ──────────────────────────────────────────────
            elif user_input.startswith("?"):
                question = user_input[1:].strip()
                if not question:
                    print("❌ Введите вопрос после ?")
                    continue
                if not active:
                    print("❌ Сначала выбери паттерн: use <имя>")
                    _hint_patterns()
                    continue
                ask_pattern(active, question, grow=False)

            # ── просто текст → ask ───────────────────────────────────────────
            else:
                if not active:
                    print("❌ Сначала выбери паттерн: use <имя>")
                    _hint_patterns()
                    continue
                ask_pattern(active, user_input, grow=True)

        except KeyboardInterrupt:
            print("\n👋 Выход")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    cli_loop()
