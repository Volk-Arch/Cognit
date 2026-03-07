#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Echo PoC — Persistent Neural Context (Transformer)
====================================================
Концепция: KV-cache локальной LLM = "паттерн активности" = персистентная память.

Установка:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

Модель (скачать GGUF, например с HuggingFace):
    Qwen3-8B-Q4_K_M.gguf  (~4.7 GB VRAM)  ← рекомендую

Команды:
    context <имя> <текст>    — обработать текст → сохранить KV-cache
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
MODEL_PATH   = "models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
MODEL_NAME   = Path(MODEL_PATH).stem
N_CTX        = 8192
N_GPU_LAYERS = -1
MAX_TOKENS   = 512

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
    """Формирует промпт с контекстом в формате ChatML (Qwen3)."""
    return (
        "<|im_start|>system\n"
        "Ты — ассистент с предзагруженным контекстом. "
        "Отвечай на вопросы, опираясь только на загруженный контекст. "
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


def save_pattern(name: str, text: str, source_files: list[str] = None):
    """
    Прогоняет текст через LLM, сохраняет KV-cache.
    После этого модель 'знает' текст без необходимости читать его снова.
    """
    print(f"\n📝 Формирование паттерна '{name}'...")

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
        "n_tokens":     len(tokens),
        "size_kb":      round(size_kb, 1),
        "preview":      text[:300],
        "saved_at":     datetime.now().isoformat(),
        "n_asks":       0,
        "source_files": [{"path": p, "hash": core.file_hash(p)} for p in (source_files or [])],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Паттерн сохранён: {pattern_path}  ({size_kb:.0f} KB)")
    print(f"   Вопросы: ask {name} <вопрос>")


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


def ask_pattern(name: str, question: str, grow: bool = True):
    """
    Загружает KV-cache и отвечает на вопрос.
    Текст контекста НЕ передаётся повторно — только вопрос.

    grow=True (по умолчанию): после ответа KV-cache обновляется —
    паттерн накапливает весь диалог (растущая сессия).
    """
    print(f"\n💬 Загрузка паттерна '{name}'...")
    _check_and_refresh(name)
    if not load_pattern(name):
        return

    # Добавляем только вопрос — контекст уже в KV-cache
    question_prompt = _question_prompt(question)
    q_tokens = llm.tokenize(question_prompt.encode("utf-8"), add_bos=False)

    print(f"   Токенов вопроса: {len(q_tokens)}  (контекст не передаётся повторно!)")
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

        print(f"💾 Паттерн обновлён: +{len(q_tokens)} токенов  ({meta['size_kb']} KB, диалогов: {meta['n_asks']})")


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
  /load <имя> @<файл>  — загрузить файл как паттерн
  /load <имя> <текст>  — загрузить текст как паттерн
  /list                — показать все паттерны
  /help                — эта справка
  /exit                — выход

ПРИМЕР:
  /load auth @src/auth.py        ← загрузить файл
  use auth                       ← переключиться на него
  что делает функция login?      ← просто пишешь вопрос
  ? есть ли SQL-инъекции?        ← peek: не меняет паттерн
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


def cli_loop(auto_route: bool = False):
    print("""
╔══════════════════════════════════════════════╗
║  🧠 ECHO — Persistent Neural Context         ║
║  Transformer · KV-cache                      ║
╚══════════════════════════════════════════════╝""")

    list_patterns()

    # Если RWKV оставил маршрут — загружаем файлы (авто или с вопросом)
    active = None
    route = core.load_last_route(PATTERNS_DIR)
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
                print(f"\n✅ Готов. Активный паттерн: {active}")

    if not active:
        print("\nВыбери паттерн командой 'use <имя>' или '/help' для справки.")

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


# =============================================================================
# HEADLESS-РЕЖИМ (для post-commit хука и скриптов)
# =============================================================================
def headless_refresh_file(file_path: str):
    """
    Пересоздаёт все паттерны, которые ссылаются на file_path.
    Вызывается post-commit хуком: python echo_poc.py --refresh-file src/auth.py
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
        # Проверяем, изменился ли файл с момента создания паттерна
        if all(s["hash"] == new_hash for s in affected):
            print(f"   ✓ {name}: актуален")
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
            # Запущен из echo_rwkv.py после route — загружаем файлы без вопросов
            cli_loop(auto_route=True)
        elif sys.argv[1] == "--refresh-file" and len(sys.argv) > 2:
            headless_refresh_file(sys.argv[2])
        elif sys.argv[1] == "--status":
            headless_status()
        else:
            print(f"Неизвестный флаг: {sys.argv[1]}")
            print("Флаги: --auto-route, --refresh-file <path>, --status")
            sys.exit(1)
    else:
        cli_loop()
