#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""cognit_i18n.py — bilingual messages (EN/RU) for Cognit CLI."""

LANG = "en"  # overridden by .echo.json at startup


def set_lang(lang: str):
    global LANG
    LANG = lang if lang in ("en", "ru") else "en"


def msg(key: str, **kwargs) -> str:
    """Return localized message. Falls back to EN, then to raw key."""
    entry = _MESSAGES.get(key)
    if entry is None:
        return key
    text = entry.get(LANG, entry.get("en", key))
    return text.format(**kwargs) if kwargs else text


# ---------------------------------------------------------------------------
# Agent roles (LLM prompts) — used by cognit_pipeline.py
# ---------------------------------------------------------------------------
AGENT_ROLES = {
    "analyst": {
        "en": "Task: {task}\n\n"
              "Analyze the code and describe a concrete plan of changes: "
              "which functions/lines need to change, what exactly to do, why. "
              "Write briefly (2-4 sentences). Do not write code.",
        "ru": "Задача: {task}\n\n"
              "Проанализируй код и опиши конкретный план изменений: "
              "какие функции/строки нужно изменить, что именно сделать, почему. "
              "Напиши кратко (2-4 предложения). Не пиши код.",
    },
    "context": {
        "en": "Task: {task}\n\n"
              "What is important to consider from the project goals and requirements perspective? "
              "Write briefly (2-4 sentences). Do not write code.",
        "ru": "Задача: {task}\n\n"
              "Что важно учесть с точки зрения целей и требований проекта? "
              "Напиши кратко (2-4 предложения). Не пиши код.",
    },
    "arch": {
        "en": "Task: {task}\n\n"
              "What architectural constraints are important? "
              "What must not be broken? Write briefly (2-4 sentences). Do not write code.",
        "ru": "Задача: {task}\n\n"
              "Какие архитектурные ограничения важны? "
              "Что нельзя сломать? Напиши кратко (2-4 предложения). Не пиши код.",
    },
    "style": {
        "en": "Task: {task}\n\n"
              "What code style conventions should be followed? "
              "Write briefly (2-4 sentences). Do not write code.",
        "ru": "Задача: {task}\n\n"
              "Какие соглашения по стилю кода нужно соблюсти? "
              "Напиши кратко (2-4 предложения). Не пиши код.",
    },
    "coder": {
        "en": "Modify existing files to implement the task. "
              "Write unified diff. Only ```diff block, no explanations.",
        "ru": "Измени существующие файлы для реализации задачи. "
              "Напиши unified diff. Только ```diff блок, без объяснений.",
    },
    "reviewer": {
        "en": "You are a reviewer. Check the diff:\n"
              "1. Are the correct files and functions changed?\n"
              "2. Is there code duplication?\n"
              "3. Do the line numbers @@ match the real ones?\n"
              "If there are errors — write a corrected ```diff block.\n"
              "If everything is correct — output the same diff unchanged.",
        "ru": "Ты ревьюер. Проверь diff:\n"
              "1. Правильные ли файлы и функции изменены?\n"
              "2. Нет ли дублирования кода?\n"
              "3. Совпадают ли номера строк @@ с реальными?\n"
              "Если есть ошибки — напиши исправленный ```diff блок.\n"
              "Если всё верно — выведи тот же diff без изменений.",
    },
}


def agent_role(name: str) -> str:
    """Return agent role prompt in current language."""
    entry = AGENT_ROLES.get(name)
    if entry is None:
        return ""
    return entry.get(LANG, entry.get("en", ""))


# ---------------------------------------------------------------------------
# Pipeline stage comments (human-readable, written to pipeline.json)
# ---------------------------------------------------------------------------
STAGE_COMMENTS = {
    "navigator": {
        "en": "Memo from tree-sitter navigator: which files, where to look",
        "ru": "Памятка от tree-sitter навигатора: какие файлы, где искать",
    },
    "analyst": {
        "en": "Analyst: reviews code + task -> concrete change plan",
        "ru": "Аналитик: смотрит код + задачу → конкретный план изменений",
    },
    "context": {
        "en": "Agent: project context, business requirements",
        "ru": "Агент: контекст проекта, бизнес-требования",
    },
    "arch": {
        "en": "Agent: architecture, dependencies, patterns",
        "ru": "Агент: архитектура, зависимости, паттерны",
    },
    "style": {
        "en": "Agent: style, formatting, naming",
        "ru": "Агент: стиль, форматирование, нейминг",
    },
    "coder": {
        "en": "Final stage: coder writes diff using full context",
        "ru": "Финальный этап: кодер пишет diff используя весь контекст",
    },
    "reviewer": {
        "en": "Review: checks diff against tree-sitter structure, removes duplicates",
        "ru": "Ревью: проверяет diff по tree-sitter структуре, убирает дубли",
    },
}


def stage_comment(stage_id: str) -> str:
    """Return localized pipeline stage comment."""
    entry = STAGE_COMMENTS.get(stage_id)
    if entry is None:
        return ""
    return entry.get(LANG, entry.get("en", ""))


# ---------------------------------------------------------------------------
# HELP text
# ---------------------------------------------------------------------------
HELP = {
    "en": """
📖 COMMANDS:
  use <name>              — select active pattern (model memory)
  <question>              — ask  (follows pattern policy: grow saves, retrain — no)
  ? <question>            — ask with temporary policy switch (grow↔retrain)
  route <task>            — find files via tree-sitter index → run pipeline
  /load <name> @<file>    — load file as pattern
  /load <name> @<d1> @<d2> — composite: multiple folders in single eval pass
  /load <name> <text>     — load text as pattern
  /edit @<file> <task>    — reads file fresh → precise unified diff → /patch
  /patch                  — apply unified diff from last response to file
  /patch @<file>          — apply to specific file (override)
  /agents                 — list available agents
  /agent <n> [n2 ...]     — enable ambient agents (all questions through [pattern + agents])
  /agent off              — disable ambient agents
  /review @<file>         — code review (style agent + file, pattern unchanged)
  /review arch @<file>    — review through any agent from agents/
  /review                 — review last model response (diff/code)
  /style @<file>          — check file style
  /index                  — project overview (tree-sitter symbols)
  /index <query>          — search symbols by query (BM25)
  /index --rebuild        — rebuild index
  /list                   — show all patterns
  /help                   — this help
  /exit                   — exit

EXAMPLE:
  /load auth @src/auth.py             ← load file
  use auth                            ← switch to it
  what does the login function do?    ← just write a question
  fix JWT                             ← request a fix
  /agent style arch                   ← enable ambient: style + architecture
  /review @src/auth.py                ← review file (style + code)
  /edit @src/auth.py fix JWT          ← reads file fresh → precise diff
  /patch                              ← apply diff to auth.py
  route add rate limiting             ← find files → pipeline
  /index bayes update                 ← search symbols by query
""",
    "ru": """
📖 КОМАНДЫ:
  use <имя>               — выбрать активный паттерн (память модели)
  <вопрос>                — спросить  (следует политике паттерна: grow сохраняет, retrain — нет)
  ? <вопрос>              — спросить с временной сменой политики (grow↔retrain)
  route <задача>          — найти файлы через tree-sitter индекс → запустить пайплайн
  /load <имя> @<файл>     — загрузить файл как паттерн
  /load <имя> @<д1> @<д2> — composite: несколько папок в одном eval-проходе
  /load <имя> <текст>     — загрузить текст как паттерн
  /edit @<файл> <задача>  — читает файл свежо → точный unified diff → /patch
  /patch                  — применить unified diff из последнего ответа к файлу
  /patch @<файл>          — применить к конкретному файлу (override)
  /agents                 — список доступных агентов
  /agent <имя> [имя2 ...] — включить ambient агенты (все вопросы через [паттерн + агенты])
  /agent off              — выключить ambient агенты
  /review @<файл>         — код-ревью (агент style + файл, паттерн не меняется)
  /review arch @<файл>    — ревью через любой агент из agents/
  /review                 — ревью последнего ответа модели (diff/код)
  /style @<файл>          — проверка стиля файла
  /index                  — обзор проекта (tree-sitter символы)
  /index <запрос>         — поиск символов по запросу (BM25)
  /index --rebuild        — пересобрать индекс
  /list                   — показать все паттерны
  /help                   — эта справка
  /exit                   — выход

ПРИМЕР:
  /load auth @src/auth.py             ← загрузить файл
  use auth                            ← переключиться на него
  что делает функция login?           ← просто пишешь вопрос
  исправь JWT                         ← просишь исправление
  /agent style arch                   ← включить ambient: стиль + архитектура
  /review @src/auth.py                ← ревью файла (стиль + код)
  /edit @src/auth.py исправь JWT      ← читает файл свежо → точный diff
  /patch                              ← применяем diff к auth.py
  route добавить rate limiting        ← найти нужные файлы → пайплайн
  /index bayes update                 ← найти символы по запросу
""",
}


# ---------------------------------------------------------------------------
# All UI messages — {key: {en: "...", ru: "..."}}
# ---------------------------------------------------------------------------
_MESSAGES = {
    # ── System prompt (ChatML context) ──────────────────────────────────────
    "system_prompt_context":    {"en": "You are an assistant with preloaded context. "
                                       "Answer questions based on the loaded context. "
                                       "If there is not enough information in the context, say so.",
                                 "ru": "Ты — ассистент с предзагруженным контекстом. "
                                       "Отвечай на вопросы, опираясь на загруженный контекст. "
                                       "Если информации в контексте недостаточно, скажи об этом."},
    "user_load_context":        {"en": "Load context:",
                                 "ru": "Загрузи контекст:"},
    "assistant_context_loaded": {"en": "Context loaded. Ready to answer questions.",
                                 "ru": "Контекст загружен. Готов отвечать на вопросы."},

    # ── Init & model ─────────────────────────────────────────────────────────
    "err_llama_not_installed":   {"en": "❌ llama-cpp-python is not installed.",
                                  "ru": "❌ llama-cpp-python не установлен."},
    "err_model_not_found":      {"en": "❌ Model not found: {path}",
                                  "ru": "❌ Модель не найдена: {path}"},
    "err_set_model_path":       {"en": "   Specify the correct path in MODEL_PATH",
                                  "ru": "   Укажите правильный путь в MODEL_PATH"},
    "status_loading_model":     {"en": "Loading model: {path} ...",
                                  "ru": "Загрузка модели: {path} ..."},
    "ok_transformer_loaded":    {"en": "✅ Transformer loaded  |  {device}",
                                  "ru": "✅ Transformer загружен  |  {device}"},
    "info_patterns_dir":        {"en": "   Patterns: {dir}  [{repo} / {branch}]",
                                  "ru": "   Паттерны: {dir}  [{repo} / {branch}]"},

    # ── Stale check ──────────────────────────────────────────────────────────
    "warn_files_changed":       {"en": "\n⚠️  Files changed since pattern '{name}' was created:",
                                  "ru": "\n⚠️  Файлы изменились с момента создания паттерна '{name}':"},
    "info_continue_old":        {"en": "   Continuing with the old pattern.",
                                  "ru": "   Продолжаем со старым паттерном."},
    "prompt_recreate":          {"en": "   Recreate pattern? [y/N] ",
                                  "ru": "   Пересоздать паттерн? [y/N] "},

    # ── Pattern operations ───────────────────────────────────────────────────
    "status_creating_pattern":  {"en": "\n📝 Creating pattern '{name}'  [{policy}]...",
                                  "ru": "\n📝 Формирование паттерна '{name}'  [{policy}]..."},
    "warn_text_too_long":       {"en": "⚠️  Text too long ({n_tokens} tokens), truncating to {limit}",
                                  "ru": "⚠️  Текст слишком длинный ({n_tokens} токенов), обрезаем до {limit}"},
    "info_token_count":         {"en": "   Tokens in context: {count}",
                                  "ru": "   Токенов в контексте: {count}"},
    "status_processing":        {"en": "   Processing (eval)...",
                                  "ru": "   Обработка (eval)..."},
    "ok_pattern_saved":         {"en": "✅ {name}  ({size_kb:.0f} KB, {n_tokens} tok)",
                                  "ru": "✅ {name}  ({size_kb:.0f} KB, {n_tokens} tok)"},
    "err_pattern_not_found":    {"en": "❌ Pattern '{name}' not found. Use 'list' to view.",
                                  "ru": "❌ Паттерн '{name}' не найден. Используйте 'list' для просмотра."},
    "warn_model_mismatch":      {"en": "⚠️  Pattern '{name}' was created for {model}, ",
                                  "ru": "⚠️  Паттерн '{name}' создан для {model}, "},
    "info_recreate_pattern":    {"en": "   Recreate: /load ?{name} @<path>",
                                  "ru": "   Пересоздай: /load ?{name} @<путь>"},

    # ── Ask / generation ─────────────────────────────────────────────────────
    "warn_fake_turn":           {"en": "\n⚠️  Fake turn — generation stopped",
                                  "ru": "\n⚠️  Фейковый turn — генерация остановлена"},
    "warn_dup_code_block":      {"en": "\n⚠️  Duplicate code block — generation stopped",
                                  "ru": "\n⚠️  Повторный code-блок — генерация остановлена"},
    "warn_infinite_loop":       {"en": "\n⚠️  Infinite loop — generation stopped",
                                  "ru": "\n⚠️  Зацикливание — генерация остановлена"},
    "warn_think_too_long":      {"en": "⚠️  Think phase too long — generation stopped",
                                  "ru": "⚠️  Think-фаза слишком длинная — генерация остановлена"},
    "warn_garbage":             {"en": "\n⚠️  Garbage output — generation stopped",
                                  "ru": "\n⚠️  Мусорный вывод — генерация остановлена"},
    "warn_script_drift":        {"en": "\n⚠️  Script drift detected — generation stopped",
                                  "ru": "\n⚠️  Обнаружена смена языка/скрипта — генерация остановлена"},

    # ── Pipeline ─────────────────────────────────────────────────────────────
    "info_pipeline_stages":     {"en": "\n🚀 Pipeline  ({n} stages{suffix})",
                                  "ru": "\n🚀 Пайплайн  ({n} стадий{suffix})"},
    "info_pipeline_run":        {"en": "  Run {num} / {total}",
                                  "ru": "  Прогон {num} / {total}"},
    "info_nav_memo":            {"en": "   → {memo}",
                                  "ru": "   → {memo}"},
    "info_nav_no_memo":         {"en": "\n[nav] navigator  — no memo (skipped)",
                                  "ru": "\n[nav] navigator  — нет мемо (пропуск)"},
    "warn_agent_no_memo":       {"en": "   ⚠️  Agent '{name}' did not provide memo — skipped",
                                  "ru": "   ⚠️  Агент '{name}' не дал мемо — пропуск"},
    "info_context_stats":       {"en": "   Context: {words} words  |  {files} file(s)",
                                  "ru": "   Контекст: {words} слов  |  {files} файл(ов)"},
    "ok_diff_corrected":        {"en": "   ✓ Diff corrected",
                                  "ru": "   ✓ Diff скорректирован"},
    "info_diff_unchanged":      {"en": "   ✓ Diff unchanged",
                                  "ru": "   ✓ Diff без изменений"},
    "info_diff_ready":          {"en": "\n💡 Diff ready → /patch",
                                  "ru": "\n💡 Diff готов → /patch"},
    "info_log_path":            {"en": "\n📄 Log: {path}",
                                  "ru": "\n📄 Лог: {path}"},
    "ok_pipeline_done":         {"en": "🔄 Pipeline completed, context reset.",
                                  "ru": "🔄 Пайплайн завершён, контекст сброшен."},
    "info_code_fragment":       {"en": "   📍 Fragment: lines {start}–{end} ",
                                  "ru": "   📍 Фрагмент: строки {start}–{end} "},

    # ── Agent operations ─────────────────────────────────────────────────────
    "err_agent_echo_missing":   {"en": "❌ Pattern '{name}' not found and .echo.json is missing.",
                                  "ru": "❌ Паттерн '{name}' не найден и .echo.json отсутствует."},
    "info_load_manually":       {"en": "   Load manually: /load {name} @agents/{name}/",
                                  "ru": "   Загрузи вручную: /load {name} @agents/{name}/"},
    "err_agent_not_found":      {"en": "❌ Agent '{name}' not found: {dir}",
                                  "ru": "❌ Не найден агент '{name}': {dir}"},
    "info_available_agents":    {"en": "   Available agents: {names}",
                                  "ru": "   Доступные агенты: {names}"},
    "info_create_agents":       {"en": "   Create agents/: python cognit_setup.py agents",
                                  "ru": "   Создай agents/: python cognit_setup.py agents"},
    "status_loading_agent":     {"en": "   Auto-loading agent '{name}': {dir} ...",
                                  "ru": "   Авто-загрузка агента '{name}': {dir} ..."},
    "status_auto_init":         {"en": "\n🔄 Auto-initializing agents ({missing} of {total})...",
                                  "ru": "\n🔄 Авто-инициализация агентов ({missing} из {total})..."},
    "warn_agent_folder_missing":{"en": "   ⚠️  Agent folder not found: {dir}",
                                  "ru": "   ⚠️  Папка агента не найдена: {dir}"},
    "info_no_agents":           {"en": "📂 No agents. Create: python cognit_setup.py agents",
                                  "ru": "📂 Нет агентов. Создай: python cognit_setup.py agents"},
    "info_agent_count":         {"en": "\n📋 Agents: {count}",
                                  "ru": "\n📋 Агентов: {count}"},
    "info_agents_disabled":     {"en": "🔕 Ambient agents disabled",
                                  "ru": "🔕 Ambient агенты отключены"},
    "ok_agents_enabled":        {"en": "🔗 Ambient agents: {names}",
                                  "ru": "🔗 Ambient агенты: {names}"},
    "info_question_flow":       {"en": "   Each question will pass through [{pattern} + {agents}]",
                                  "ru": "   Каждый вопрос будет проходить через [{pattern} + {agents}]"},
    "info_ambient_ephemeral":   {"en": "   Pattern is not saved in ambient mode (ephemeral).",
                                  "ru": "   Паттерн не сохраняется в ambient режиме (эфемерно)."},
    "info_disable_agents":      {"en": "   Disable: /agent off",
                                  "ru": "   Отключить: /agent off"},
    "warn_agent_not_found_rev": {"en": "⚠️  Agent {name} not found",
                                  "ru": "⚠️  Агент {name} не найден"},
    "info_agent_analyzing":     {"en": "   Agent: {name}  |  Analyzing: {label}",
                                  "ru": "   Агент: {name}  |  Анализирую: {label}"},

    # ── Full eval ────────────────────────────────────────────────────────────
    "info_full_eval":           {"en": "\n🔄 Full-eval: {n} characters",
                                  "ru": "\n🔄 Full-eval: {n} символов"},

    # ── File operations ──────────────────────────────────────────────────────
    "err_file_not_found":       {"en": "❌ File not found: {path}",
                                  "ru": "❌ Файл не найден: {path}"},
    "status_reading_file":      {"en": "\n📄 Reading file: {path}  ({lines} lines, {chars} characters)",
                                  "ru": "\n📄 Читаю файл: {path}  ({lines} строк, {chars} символов)"},
    "err_empty_folder":         {"en": "❌ Folder is empty or no suitable files: {path}",
                                  "ru": "❌ Папка пуста или нет подходящих файлов: {path}"},
    "status_loading_files":     {"en": "   Loading {count} files from {path}/  (git ls-files / rglob)",
                                  "ru": "   Загружаю {count} файлов из {path}/  (git ls-files / rglob)"},
    "err_path_not_found":       {"en": "❌ File or folder not found: {path}",
                                  "ru": "❌ Не найден файл или папка: {path}"},
    "warn_empty_folder":        {"en": "   ⚠️  Folder is empty or no suitable files: {path}",
                                  "ru": "   ⚠️  Папка пуста или нет подходящих файлов: {path}"},
    "info_folder_loaded":       {"en": "   + {name}/  ({count} files)",
                                  "ru": "   + {name}/  ({count} файлов)"},
    "warn_path_not_found":      {"en": "   ⚠️  Not found: {path}",
                                  "ru": "   ⚠️  Не найден: {path}"},
    "err_no_files":             {"en": "❌ No paths contain files.",
                                  "ru": "❌ Ни один путь не содержит файлов."},
    "info_composite":           {"en": "   Composite: {name}  (sources: {sources}, files: {files})",
                                  "ru": "   Composite: {name}  (источников: {sources}, файлов: {files})"},

    # ── Index / navigation ───────────────────────────────────────────────────
    "warn_tree_sitter_missing": {"en": "⚠️  tree-sitter not installed. pip install tree-sitter tree-sitter-python",
                                  "ru": "⚠️  tree-sitter не установлен. pip install tree-sitter tree-sitter-python"},
    "warn_client_project":      {"en": "⚠️  client_project not set in .echo.json — routing impossible",
                                  "ru": "⚠️  client_project не задан в .echo.json — маршрутизация невозможна"},
    "warn_no_symbols":          {"en": "⚠️  Index found no relevant symbols",
                                  "ru": "⚠️  Индекс не нашёл релевантных символов"},
    "info_symbols_found":       {"en": "\n📍 Found {n_symbols} symbols in {n_files} files:",
                                  "ru": "\n📍 Найдено {n_symbols} символов в {n_files} файлах:"},
    "err_index_unavailable":    {"en": "❌ Index unavailable (check client_project and tree-sitter)",
                                  "ru": "❌ Индекс недоступен (проверьте client_project и tree-sitter)"},

    # ── CLI header & intro ───────────────────────────────────────────────────
    "info_header":              {"en": "\n🧠 Transformer · {model} · {repo}/{branch}",
                                  "ru": "\n🧠 Transformer · {model} · {repo}/{branch}"},
    "info_intro":               {"en": "\n💡 Enter a question or task — Cognit will find files automatically.",
                                  "ru": "\n💡 Введи вопрос или задачу — Cognit найдёт нужные файлы автоматически."},
    "info_intro_load":          {"en": "   Or /load <name> @<path> to work with a specific file.",
                                  "ru": "   Или /load <имя> @<путь> для работы с конкретным файлом."},
    "ok_exit":                  {"en": "👋 Goodbye",
                                  "ru": "👋 Выход"},

    # ── CLI commands ─────────────────────────────────────────────────────────
    "err_load_usage":           {"en": "❌ Usage: /load <name> @<file/folder>  or  /load <name> <text>",
                                  "ru": "❌ Использование: /load <имя> @<файл/папка>  или  /load <имя> <текст>"},
    "info_load_composite":      {"en": "   Multiple paths: /load <name> @dir1/ @dir2/  (composite pattern)",
                                  "ru": "   Несколько путей: /load <имя> @dir1/ @dir2/  (composite паттерн)"},
    "info_load_retrain":        {"en": "   Prefix ? forces retrain: /load ?name @path",
                                  "ru": "   Префикс ? форсирует retrain: /load ?name @path"},
    "info_active_pattern":      {"en": "   Active pattern: {name}",
                                  "ru": "   Активный паттерн: {name}"},
    "err_edit_usage":           {"en": "❌ Usage: /edit @<file> <task>",
                                  "ru": "❌ Использование: /edit @<файл> <задача>"},
    "info_edit_example":        {"en": "   Example: /edit @src/config.py change 0-1 to percentages",
                                  "ru": "   Пример: /edit @src/config.py поменяй 0-1 на проценты"},
    "info_apply_patch":         {"en": "\n💡 Apply? → /patch",
                                  "ru": "\n💡 Применить? → /patch"},
    "err_select_pattern":       {"en": "❌ First select a pattern: use <name>",
                                  "ru": "❌ Сначала выбери паттерн: use <имя>"},
    "err_no_previous":          {"en": "❌ No previous response. Ask a question about changes, then /patch.",
                                  "ru": "❌ Нет предыдущего ответа. Задай вопрос об изменениях, затем /patch."},
    "info_reformatting":        {"en": "   Diff not found — reformatting previous response...",
                                  "ru": "   Diff не найден — переформатирую предыдущий ответ..."},
    "err_no_diff":              {"en": "❌ Model did not return unified diff. Ask explicitly.",
                                  "ru": "❌ Модель не вернула unified diff. Попроси явно."},
    "info_diffs_found":         {"en": "\n📋 Found {count} diff blocks",
                                  "ru": "\n📋 Найдено {count} diff-блоков"},
    "err_file_not_specified":   {"en": "❌ File not specified. Use: /patch @<file>",
                                  "ru": "❌ Файл не определён. Используй: /patch @<файл>"},
    "info_more_lines":          {"en": "   ... ({count} more lines)",
                                  "ru": "   ... (ещё {count} строк)"},
    "info_code_found":          {"en": "\n💡 Code found in response → /patch",
                                  "ru": "\n💡 Есть код в ответе → /patch"},
    "err_unknown_cmd":          {"en": "❌ Unknown command '/{cmd}'. Enter '/help'.",
                                  "ru": "❌ Неизвестная команда '/{cmd}'. Введите '/help'."},
    "err_use_usage":            {"en": "❌ Usage: use <name>",
                                  "ru": "❌ Использование: use <имя>"},
    "err_pattern_not_found_use":{"en": "❌ Pattern '{name}' not found.",
                                  "ru": "❌ Паттерн '{name}' не найден."},
    "ok_pattern_selected":      {"en": "✅ Active pattern: {name}{policy}",
                                  "ru": "✅ Активный паттерн: {name}{policy}"},
    "err_question_after_qmark": {"en": "❌ Enter a question after ?",
                                  "ru": "❌ Введите вопрос после ?"},
    "err_route_usage":          {"en": "❌ Usage: route <task>",
                                  "ru": "❌ Использование: route <задача>"},
    "info_route_example":       {"en": "   Example: route add rate limiting for POST /login",
                                  "ru": "   Пример: route добавить rate limiting для POST /login"},
    "prompt_run_pipeline":      {"en": "\n   Run pipeline? [Y/n] ",
                                  "ru": "\n   Запустить пайплайн? [Y/n] "},
    "err_files_not_found":      {"en": "❌ Could not find files. Load manually: /load <name> @<path>",
                                  "ru": "❌ Не удалось найти файлы. Загрузи вручную: /load <имя> @<путь>"},
    "err_general":              {"en": "❌ Error: {error}",
                                  "ru": "❌ Ошибка: {error}"},

    "err_content_not_found":    {"en": "❌ Not found: {path}",
                                  "ru": "❌ Не найден: {path}"},
    "err_specify_file":         {"en": "❌ Specify file: /{cmd} [agent] @<file>  or ask a question first",
                                  "ru": "❌ Укажи файл: /{cmd} [агент] @<файл>  или сначала задай вопрос"},

    # ── Headless ─────────────────────────────────────────────────────────────
    "warn_file_not_found_ref":  {"en": "   ⚠️  File not found: {path}",
                                  "ru": "   ⚠️  Файл не найден: {path}"},
    "ok_pattern_current":       {"en": "   ✓ {name}: current",
                                  "ru": "   ✓ {name}: актуален"},
    "info_grow_skip":           {"en": "   ~ {name}: grow-pattern, skipped (accumulates dialogue)",
                                  "ru": "   ~ {name}: grow-паттерн, пропускаем (накапливает диалог)"},
    "status_recreating":        {"en": "   ♻  {name}: recreating (modified {path})...",
                                  "ru": "   ♻  {name}: пересоздаю (изменён {path})..."},
    "ok_patterns_updated":      {"en": "   ✅ Updated patterns: {count}",
                                  "ru": "   ✅ Обновлено паттернов: {count}"},
    "info_no_patterns_using":   {"en": "   ℹ  No patterns using {path}",
                                  "ru": "   ℹ  Нет паттернов, использующих {path}"},
    "info_no_patterns":         {"en": "📂 No patterns.",
                                  "ru": "📂 Паттернов нет."},
    "ok_all_current":           {"en": "✅ All patterns are current.",
                                  "ru": "✅ Все паттерны актуальны."},
    "warn_stale_patterns":      {"en": "⚠️  Stale patterns: {count}",
                                  "ru": "⚠️  Устаревших паттернов: {count}"},
    "err_unknown_flag":         {"en": "Unknown flag: {flag}",
                                  "ru": "Неизвестный флаг: {flag}"},
    "info_available_flags":     {"en": "Flags: --refresh-file <path>, --status",
                                  "ru": "Флаги: --refresh-file <path>, --status"},
    "warn_refresh_deprecated":  {"en": "⚠️  --refresh-file is deprecated. Run 'python cognit_setup.py' to update your hook.",
                                  "ru": "⚠️  --refresh-file устарел. Запустите 'python cognit_setup.py' для обновления хука."},

    # ── cognit_hook.py ────────────────────────────────────────────────────────
    "hook_checking":            {"en": "[Cognit] Checking {count} changed file(s)...",
                                  "ru": "[Cognit] Проверяю {count} изменённых файл(ов)..."},
    "hook_marked_stale":        {"en": "[Cognit] Marked {count} pattern(s) as stale.",
                                  "ru": "[Cognit] Отмечено устаревших паттернов: {count}."},
    "hook_index_updated":       {"en": "[Cognit] Index updated.",
                                  "ru": "[Cognit] Индекс обновлён."},
    "hook_all_current":         {"en": "[Cognit] All patterns current.",
                                  "ru": "[Cognit] Все паттерны актуальны."},
    "hook_no_patterns_dir":     {"en": "[Cognit] No patterns directory, skipping.",
                                  "ru": "[Cognit] Каталог паттернов не найден, пропускаю."},

    # ── LLM reranking ────────────────────────────────────────────────────────
    "info_expanding_query":     {"en": "   🔍 LLM query expansion (BM25 found < 5)...",
                                  "ru": "   🔍 LLM расширение запроса (BM25 нашёл < 5)..."},
    "info_expand_done":         {"en": "   ✅ Expanded: +{count} terms → {total} candidates",
                                  "ru": "   ✅ Расширено: +{count} терминов → {total} кандидатов"},
    "info_reranking":           {"en": "   🔄 LLM reranking {count} candidates...",
                                  "ru": "   🔄 LLM ранжирование {count} кандидатов..."},
    "info_rerank_done":         {"en": "   ✅ Reranked → top {count} files",
                                  "ru": "   ✅ Переранжировано → топ {count} файлов"},
    "info_rerank_fallback":     {"en": "   ⚠️  Rerank failed, using BM25 order",
                                  "ru": "   ⚠️  Ранжирование не удалось, используется BM25"},
    "info_grep_searching":      {"en": "   🔎 Grep search in codebase...",
                                  "ru": "   🔎 Grep поиск по кодовой базе..."},
    "info_grep_found":          {"en": "   ✅ Grep: {count} matches in {files} files",
                                  "ru": "   ✅ Grep: {count} совпадений в {files} файлах"},
    "info_deps_searching":      {"en": "   🔗 Finding dependencies...",
                                  "ru": "   🔗 Поиск зависимостей..."},
    "info_deps_found":          {"en": "   ✅ Dependencies: +{count} related symbols",
                                  "ru": "   ✅ Зависимости: +{count} связанных символов"},
    "info_full_scan":           {"en": "   🔬 Full scan (last resort)...",
                                  "ru": "   🔬 Полный скан (крайний случай)..."},
    "info_scan_checking":       {"en": "   📄 Scanning {name}...",
                                  "ru": "   📄 Сканирование {name}..."},
    "info_scan_found":          {"en": "   ✅ Scan found {count} relevant files",
                                  "ru": "   ✅ Скан нашёл {count} релевантных файлов"},
    "info_scan_nothing":        {"en": "   ⚠️  Full scan found nothing",
                                  "ru": "   ⚠️  Полный скан ничего не нашёл"},

    # ── cognit_core.py ───────────────────────────────────────────────────────
    "info_patterns_list_empty": {"en": "   No patterns — load: /load <name> @<file>",
                                  "ru": "   Паттернов нет — загрузи: /load <имя> @<файл>"},
    "info_patterns_list_count": {"en": "📋 Patterns: {count}",
                                  "ru": "📋 Паттернов: {count}"},
    "info_hint_patterns":       {"en": "\n💡 Hint: /load <name> @<file>  to create a pattern",
                                  "ru": "\n💡 Подсказка: /load <имя> @<файл>  чтобы создать паттерн"},

    # ── cognit_patch.py ──────────────────────────────────────────────────────
    "err_cannot_read_file":     {"en": "❌ Cannot read file: {error}",
                                  "ru": "❌ Не могу прочитать файл: {error}"},
    "ok_file_created":          {"en": "✅ File created → {path}",
                                  "ru": "✅ Файл создан → {path}"},
    "ok_patch_applied":         {"en": "✅ Patch applied → {path}",
                                  "ru": "✅ Патч применён → {path}"},
    "info_backup":              {"en": "   Backup: {path}",
                                  "ru": "   Бэкап: {path}"},
    "err_patch_failed":         {"en": "❌ Patch failed for {path}: {error}",
                                  "ru": "❌ Ошибка применения патча к {path}: {error}"},
    "prompt_apply_patch":       {"en": "   Apply '{path}'? [y/N] ",
                                  "ru": "   Применить '{path}'? [y/N] "},

    # ── cognit_pipeline.py ───────────────────────────────────────────────────
    "warn_pipeline_error":      {"en": "⚠️  Error reading pipeline.json: {error} — using default",
                                  "ru": "⚠️  Ошибка чтения pipeline.json: {error} — использую дефолт"},
    "ok_pipeline_created":      {"en": "✅ pipeline.json created: {path}",
                                  "ru": "✅ pipeline.json создан: {path}"},

    # ── cognit_index.py ──────────────────────────────────────────────────────
    "info_index_stats":         {"en": "📇 Index: {files} files, {symbols} symbols",
                                  "ru": "📇 Индекс: {files} файлов, {symbols} символов"},

    # ── cognit_setup.py ───────────────────────────────────────────────────
    "setup_header":             {"en": "\n── Main project (for git hook) ──────────────────────",
                                  "ru": "\n── Основной проект (для git-хука) ──────────────────────"},
    "setup_project_intro1":     {"en": "   Cognit lives in its own repo. Specify the path to the project",
                                  "ru": "   Cognit живёт в своём репо. Укажи путь к проекту,"},
    "setup_project_intro2":     {"en": "   you want to analyze — the hook will be installed there.",
                                  "ru": "   который хочешь анализировать — хук поставится туда."},
    "setup_ask_project_path":   {"en": "   Path to project (Enter — skip)",
                                  "ru": "   Путь к проекту (Enter — пропустить)"},
    "setup_path_not_found":     {"en": "   ⚠️  Path not found: {path}",
                                  "ru": "   ⚠️  Путь не найден: {path}"},
    "setup_not_git_repo":       {"en": "   ⚠️  {path} — not a git repository (no .git folder)",
                                  "ru": "   ⚠️  {path} — не git-репозиторий (нет папки .git)"},
    "setup_project_name":       {"en": "   Project: {name}  ({path})",
                                  "ru": "   Проект: {name}  ({path})"},
    "setup_config_header":      {"en": "\n── Project configuration (.echo.json) ──────────────────",
                                  "ru": "\n── Конфигурация проекта (.echo.json) ──────────────────"},
    "setup_found_existing":     {"en": "   Found existing {file}",
                                  "ru": "   Найден существующий {file}"},
    "setup_ask_model_path":     {"en": "   Path to Transformer model (GGUF)",
                                  "ru": "   Путь к Transformer-модели (GGUF)"},
    "setup_config_created":     {"en": "   ✅ Created {file}",
                                  "ru": "   ✅ Создан {file}"},
    "setup_gitignore_header":   {"en": "\n── .gitignore ──────────────────────────────────────────",
                                  "ru": "\n── .gitignore ──────────────────────────────────────────"},
    "setup_gitignore_ok":       {"en": "   ✅ {file} already contains required entries",
                                  "ru": "   ✅ {file} уже содержит нужные записи"},
    "setup_gitignore_added":    {"en": "   ✅ Added to {file}: {line}",
                                  "ru": "   ✅ Добавлено в {file}: {line}"},
    "setup_hook_exists":        {"en": "   Hook already exists in {name}. Overwrite?",
                                  "ru": "   Хук уже существует в {name}. Перезаписать?"},
    "setup_skip":               {"en": "   Skipping.",
                                  "ru": "   Пропускаю."},
    "setup_hook_installed":     {"en": "   ✅ Hook installed: {path}",
                                  "ru": "   ✅ Хук установлен: {path}"},
    "setup_s3_header":          {"en": "\n── Pattern sync via S3 ───────────────────",
                                  "ru": "\n── Синхронизация паттернов через S3 ───────────────────"},
    "setup_agents_header":      {"en": "\n── agents/ — project knowledge ──────────────────────────",
                                  "ru": "\n── agents/ — знания о проекте ──────────────────────────"},
    "setup_creating_in":        {"en": "   Creating in: {dir}",
                                  "ru": "   Создаю в: {dir}"},
    "setup_file_exists":        {"en": "   • {path}  (already exists, skipping)",
                                  "ru": "   • {path}  (уже существует, пропускаю)"},
    "setup_file_created":       {"en": "   ✅ Created: {path}",
                                  "ru": "   ✅ Создан: {path}"},
    "setup_all_exist":          {"en": "   All files already exist.",
                                  "ru": "   Все файлы уже существуют."},
    "setup_agents_tip":         {"en": "   Tip: agents/ should be added to git of the client project.",
                                  "ru": "   Совет: agents/ нужно добавить в git клиентского проекта."},
    "setup_unknown_cmd":        {"en": "Unknown command: {cmd}",
                                  "ru": "Неизвестная команда: {cmd}"},
    "setup_usage":              {"en": "Usage: python cognit_setup.py [agents]",
                                  "ru": "Использование: python cognit_setup.py [agents]"},
    "setup_git_repo":           {"en": "\n   Git repository: {name}  ({path})",
                                  "ru": "\n   Git-репозиторий: {name}  ({path})"},
    "setup_no_git":             {"en": "\n   ⚠️  Git repository not found. Patterns will be in echo_patterns/local/",
                                  "ru": "\n   ⚠️  Git-репозиторий не обнаружен. Паттерны будут в echo_patterns/local/"},
    "setup_client_saved":       {"en": "   ✅ client_project saved to {file}",
                                  "ru": "   ✅ client_project сохранён в {file}"},
    "setup_ask_hook_client":    {"en": "\n   Install post-commit hook in {name}?",
                                  "ru": "\n   Установить post-commit хук в {name}?"},
    "setup_ask_hook_cognit":    {"en": "\nInstall post-commit hook in Cognit repo too?",
                                  "ru": "\nУстановить post-commit хук и в Cognit-репозитории?"},
    "setup_ask_agents":         {"en": "\nCreate agents/ folder with templates?",
                                  "ru": "\nСоздать папку agents/ с шаблонами?"},
    "setup_done":               {"en": "✅ Setup complete.\n",
                                  "ru": "✅ Настройка завершена.\n"},
    "setup_banner_agents":      {"en": "\n╔══════════════════════════════════════════════╗\n║  🧠 Cognit — Initialize agents/              ║\n╚══════════════════════════════════════════════╝",
                                  "ru": "\n╔══════════════════════════════════════════════╗\n║  🧠 Cognit — Инициализация agents/          ║\n╚══════════════════════════════════════════════╝"},
    "setup_banner_main":        {"en": "\n╔══════════════════════════════════════════════╗\n║  🧠 Cognit — Project Setup                   ║\n╚══════════════════════════════════════════════╝",
                                  "ru": "\n╔══════════════════════════════════════════════╗\n║  🧠 Cognit — Настройка проекта             ║\n╚══════════════════════════════════════════════╝"},
}


# ---------------------------------------------------------------------------
# Agent templates — example knowledge base for setup
# ---------------------------------------------------------------------------
_AGENT_TEMPLATES_EN = {
    "style/global.md": """\
# Code Style — DemoAI

## Naming
- All functions and variables: `snake_case`
- Private helpers (not part of public API): `_underscore_prefix` — e.g. `_read_prob`
- Probability parameters: full names, no abbreviations:
  - `prior_h` (not `p`, not `prior`)
  - `p_e_given_h` (not `peh`, not `likelihood`)
  - `p_e_given_not_h` (not `pnh`)
  - `posterior` (not `post`, not `result`)
- Mathematical negation in names: `_not_h` suffix (not `_neg`, not `_complement`)
- Step counters: `step` (not `i`, not `n`, not `iteration`)
- Constants: `UPPER_SNAKE_CASE` if added at module level

## Formatting
- Indentation: 4 spaces (no tabs)
- Max line length: 88 characters
- Blank lines: 2 between top-level functions
- Quotes: double for user-facing strings, single allowed inside f-strings
- Float output: always `.6f` for probabilities — `f"{value:.6f}"`

## Imports
- First line: `from __future__ import annotations`
- Order: stdlib -> (third-party if any) -> local
- `import *` is forbidden

## Type Hints
- Type hints required for all functions (parameters + return value)
- `float` for probabilities, `str` for prompts, `None` for procedures
- Python 3.10+ syntax: `X | None` instead of `Optional[X]`

## User Messages
- Style: polite imperative — "Enter...", "Try...", "Probability must be..."
- Validation error format: one sentence, specific correction
- Result format: two numbers in a column, aligned by `:`:
  ```
    P(H) was:      0.300000
    P(H|E) now:    0.462963
  ```
- Step header: `f"\\nStep {step}:"` — with blank line before

## Error Handling
- `ZeroDivisionError`: raise from `bayes_update` with a clear message
- Catch explicitly: `except ZeroDivisionError as e:` — no bare `except:`
- `KeyboardInterrupt`: catch in `main`, print `"\\nExit."` and terminate
- `ValueError`: catch in `_read_prob` on `float()`, continue loop
- Never show traceback to user

## Anti-patterns (forbidden)
- Abbreviating probability parameter names (`peh`, `pnh`, `ph`)
- Formatting float without precision (only `.6f` or explicit other)
- Adding global state (all data via function parameters)
- Breaking the `prior -> bayes_update -> posterior -> new prior` chain with hidden assignments
- Mixing computation and output: `bayes_update` must not print anything
""",

    "style/commands.md": """\
# Interactive CLI Style — DemoAI

## Input Prompts
- Format: `"Enter {description}: "` — colon and space before cursor
- Step prompts: indented `"  Enter P(E|H): "` (2 spaces)
- Support comma as decimal separator (`.replace(",", ".")`)
- Exit command: `q`, `quit`, `exit` — case-insensitive, at any input point

## Output
- Two numbers in a column, aligned by `:`:
  ```
    P(H) was:      0.300000
    P(H|E) now:    0.462963
  ```
- Always `.6f` for probabilities — 6 decimal places
- Blank line before each step: `print(f"\\nStep {step}:")`

## Error Messages
- One line, no emoji — educational project, no harsh tone
- Examples: `"Enter a number (e.g. 0.2) or q to quit."`
- `"Probability must be in range [0, 1]."`
- `f"Error: {e}"` — for ZeroDivisionError

## Exit
- `"\\nExit."` — on `q` or Ctrl+C, with blank line before
- No `sys.exit()` with non-zero code for normal user exit
""",

    "arch/overview.md": """\
# Architecture — DemoAI

## Project Structure
```
main.py   — all code, three functions
README.md — description
```
No packages, no subdirectories, no dependencies beyond stdlib (`sys`).

## Three Functions and Their Roles

### `_read_prob(prompt: str) -> float`
- The only user input point
- Infinite loop until a valid float in [0, 1] is received
- Handles: non-numeric input (ValueError), range, exit command (KeyboardInterrupt)
- Knows nothing about Bayesian logic — only probability number validation
- To add a new input type — create a similar `_read_*` helper

### `bayes_update(prior_h, p_e_given_h, p_e_given_not_h) -> float`
- Pure function: computation only, no I/O
- Implements: `P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|~H)*(1-P(H))]`
- Only failure mode: `ZeroDivisionError` when denominator == 0.0
- Always called with three floats, returns one float
- New Bayesian formula — add next to this function, same style

### `main() -> None`
- Entry point and entire UI loop
- Catches exceptions from `bayes_update` and `KeyboardInterrupt`
- Manages state: `prior` (float) and `step` (int)
- Iterative pattern: `prior = posterior` at end of each step

## Data Flow
```
_read_prob("P(H)")  ->  prior
  |
loop:
  _read_prob("P(E|H)")    ->  p_e_h
  _read_prob("P(E|~H)")   ->  p_e_nh
  bayes_update(prior, p_e_h, p_e_nh)  ->  posterior
  prior = posterior
  step += 1
```

## Extension Rules

**Where to add new computations:** next to `bayes_update`, as a separate pure function.
Example: `log_odds_update(prior_h, likelihood_ratio) -> float`.

**Where to add new input types:** new `_read_*` helper, call from `main`.
Example: `_read_label(prompt: str) -> str` for named hypotheses.

**Where to add output:** only in `main`. `bayes_update` must not print anything.

**Where NOT to add global state:** everything passed via parameters.

## Critical Invariants
- `bayes_update` remains a pure function — no print, no input, no side effects
- Denominator in `bayes_update` is checked before division — `ZeroDivisionError` if 0.0
- `prior` after each step = previous `posterior` — the chain must not break
- All probabilities in [0, 1] — `_read_prob` guarantees this at input
""",

    "context/project.md": """\
# Project Context — DemoAI

## What It Is
An educational interactive Bayesian probability update calculator.
The user enters a prior probability P(H) and at each step — new evidence
P(E|H) and P(E|~H). The program shows how belief updates iteratively via Bayes' theorem.

## Goal
Demonstrate Bayesian thinking: how accumulating evidence changes hypothesis probability.
Not a production tool — an educational demo for probability theory students.

## Audience
Students and beginner developers studying probabilistic inference.

## Technologies
- Python 3.10+ (`X | None` syntax, `from __future__ import annotations`)
- Standard library only (`sys`) — no external dependencies
- No DB, filesystem, HTTP — fully stateless

## Principles
- **Simplicity first** — one task, one file, three functions
- **No dependencies** — if you need a library, it's probably overkill
- **Pure functions** — computation separated from I/O
- **Clear errors** — user always knows what to enter to continue

## What Is Appropriate to Add
- Named hypotheses: `"Enter hypothesis name: "` -> `"H: it's raining"`
- Step history: list of `(prior, p_e_h, p_e_nh, posterior)` with summary at the end
- Alternative Bayesian update forms (log-odds, likelihood ratio)
- CSV export of step history via `csv.writer`
- Batch processing mode via command-line arguments (`argparse`)

## What Is NOT Appropriate to Add
- Web interface, HTTP server, REST API
- Database, sessions, users
- External libraries (numpy, scipy) — overkill for educational demo
- Multithreading / async
- GUI (tkinter, PyQt) — project is CLI-oriented

## Bayes' Formula (reference)
```
P(H|E) = P(E|H) * P(H)
         ─────────────────────────────────────
         P(E|H) * P(H) + P(E|~H) * (1 - P(H))
```
Where:
- P(H) — prior probability of hypothesis (before observing evidence)
- P(E|H) — probability of evidence given hypothesis is true (likelihood)
- P(E|~H) — probability of evidence given hypothesis is false
- P(H|E) — posterior probability (after observation, becomes new prior)
""",
}

_AGENT_TEMPLATES_RU = {
    "style/global.md": """\
# Стиль кода — DemoAI

## Именование
- Все функции и переменные: `snake_case`
- Приватные хелперы (не часть публичного API): `_underscore_prefix` — например `_read_prob`
- Параметры вероятностей: полные имена, не сокращать:
  - `prior_h` (не `p`, не `prior`)
  - `p_e_given_h` (не `peh`, не `likelihood`)
  - `p_e_given_not_h` (не `pnh`)
  - `posterior` (не `post`, не `result`)
- Математическое отрицание в именах: суффикс `_not_h` (не `_neg`, не `_complement`)
- Счётчики шагов: `step` (не `i`, не `n`, не `iteration`)
- Константы: `UPPER_SNAKE_CASE` если добавляются на уровне модуля

## Форматирование
- Отступы: 4 пробела (без табов)
- Максимальная длина строки: 88 символов
- Пустые строки: 2 между топ-уровневыми функциями
- Кавычки: двойные для строк пользователю, одинарные допустимы внутри f-строк
- Float-вывод: всегда `.6f` для вероятностей — `f"{value:.6f}"`

## Импорты
- Первая строка файла: `from __future__ import annotations`
- Порядок: stdlib → (сторонние, если появятся) → локальные
- `import *` запрещён

## Типизация
- Тип-хинты обязательны для всех функций (параметры + возвращаемое значение)
- `float` для вероятностей, `str` для промптов, `None` для процедур
- Python 3.10+ синтаксис: `X | None` вместо `Optional[X]`

## Сообщения пользователю
- Язык: русский (проект русскоязычный)
- Стиль: вежливый императив — «Введите...», «Попробуйте...», «Вероятность должна быть...»
- Формат ошибки валидации: одно предложение, конкретное исправление
- Формат результата: два числа в столбик с выравниванием:
  ```
    P(H) было:     0.300000
    P(H|E) стало:  0.462963
  ```
- Заголовок шага: `f"\\nШаг {step}:"` — с пустой строкой перед

## Обработка ошибок
- `ZeroDivisionError`: поднимать из `bayes_update` с понятным русским сообщением
- Ловить явно: `except ZeroDivisionError as e:` — не голый `except:`
- `KeyboardInterrupt`: ловить в `main`, печатать `"\\nВыход."` и завершать
- `ValueError`: ловить в `_read_prob` при `float()`, продолжать цикл
- Не выводить traceback пользователю

## Антипаттерны (запрещено)
- Сокращать имена параметров вероятностей (`peh`, `pnh`, `ph`)
- Форматировать float без указания точности (только `.6f` или явная другая)
- Добавлять глобальное состояние (все данные через параметры функций)
- Разрывать цепочку `prior → bayes_update → posterior → новый prior` скрытыми присваиваниями
- Смешивать вычисление и вывод: `bayes_update` не должна печатать ничего
""",

    "style/commands.md": """\
# Стиль интерактивного CLI — DemoAI

## Промпты ввода
- Формат: `"Введите {описание}: "` — двоеточие и пробел перед курсором
- Шаговые промпты: с отступом `"  Введите P(E|H): "` (2 пробела)
- Поддержка запятой как десятичного разделителя (`.replace(",", ".")`)
- Команда выхода: `q`, `quit`, `exit` — регистронезависимо, в любой точке ввода

## Вывод результатов
- Два числа в столбик, выравнивание по `:`:
  ```
    P(H) было:     0.300000
    P(H|E) стало:  0.462963
  ```
- Всегда `.6f` для вероятностей — 6 знаков после запятой
- Пустая строка перед каждым шагом: `print(f"\\nШаг {step}:")`

## Сообщения об ошибках
- Одна строка, без `❌` — проект учебный, строгий тон не нужен
- Примеры: `"Введите число (например 0.2) или q для выхода."`
- `"Вероятность должна быть в диапазоне [0, 1]."`
- `f"Ошибка: {e}"` — для ZeroDivisionError

## Завершение
- `"\\nВыход."` — по `q` или Ctrl+C, с пустой строкой перед
- Нет `sys.exit()` с ненулевым кодом для нормального выхода пользователя
""",

    "arch/overview.md": """\
# Архитектура DemoAI

## Структура проекта
```
main.py   — весь код, три функции
README.md — описание
```
Нет пакетов, нет подпапок, нет зависимостей кроме stdlib (`sys`).

## Три функции и их роли

### `_read_prob(prompt: str) -> float`
- Единственная точка ввода от пользователя
- Бесконечный цикл до получения корректного float в [0, 1]
- Обрабатывает: нечисловой ввод (ValueError), диапазон, команду выхода (KeyboardInterrupt)
- Не знает о байесовской логике — только валидация числа-вероятности
- Если нужно добавить новый тип ввода — создавать аналогичный хелпер `_read_*`

### `bayes_update(prior_h, p_e_given_h, p_e_given_not_h) -> float`
- Чистая функция: только вычисление, никакого I/O
- Реализует формулу: `P(H|E) = P(E|H)*P(H) / [P(E|H)*P(H) + P(E|¬H)*(1-P(H))]`
- Единственный способ упасть: `ZeroDivisionError` когда знаменатель == 0.0
- Всегда вызывается с тремя float, возвращает один float
- Если добавляется новая байесовская формула — рядом с этой функцией, тем же стилем

### `main() -> None`
- Точка входа и весь UI-цикл
- Ловит исключения от `bayes_update` и `KeyboardInterrupt`
- Управляет состоянием: `prior` (float) и `step` (int)
- Итеративный паттерн: `prior = posterior` в конце каждого шага

## Поток данных
```
_read_prob("P(H)")  →  prior
  ↓
loop:
  _read_prob("P(E|H)")    →  p_e_h
  _read_prob("P(E|¬H)")   →  p_e_nh
  bayes_update(prior, p_e_h, p_e_nh)  →  posterior
  prior = posterior
  step += 1
```

## Правила расширения

**Куда добавлять новые вычисления:** рядом с `bayes_update`, как отдельная чистая функция.
Пример: `log_odds_update(prior_h, likelihood_ratio) -> float`.

**Куда добавлять новые типы ввода:** новый хелпер `_read_*`, вызывать из `main`.
Пример: `_read_label(prompt: str) -> str` для именованных гипотез.

**Куда добавлять вывод:** только в `main`. `bayes_update` не должна ничего печатать.

**Куда НЕ добавлять глобальное состояние:** всё передаётся через параметры.

## Критические инварианты
- `bayes_update` остаётся чистой функцией — без print, без input, без side effects
- Знаменатель в `bayes_update` проверяется перед делением — `ZeroDivisionError` если 0.0
- `prior` после каждого шага = предыдущий `posterior` — цепочка не должна прерываться
- Все вероятности в [0, 1] — `_read_prob` гарантирует это на входе
""",

    "context/project.md": """\
# Контекст проекта — DemoAI

## Что это
Учебный интерактивный калькулятор байесовского обновления вероятностей.
Пользователь вводит априорную вероятность P(H) и на каждом шаге — новые свидетельства
P(E|H) и P(E|¬H). Программа показывает как вера обновляется итерационно по формуле Байеса.

## Цель
Демонстрация байесовского мышления: как накопление свидетельств меняет вероятность гипотезы.
Не production-инструмент — образовательное демо для студентов теории вероятностей.

## Аудитория
Студенты и начинающие разработчики изучающие вероятностный вывод.
Русскоязычный интерфейс — все сообщения на русском.

## Технологии
- Python 3.10+ (используется синтаксис `X | None`, `from __future__ import annotations`)
- Только стандартная библиотека (`sys`) — никаких внешних зависимостей
- Никакой БД, файловой системы, HTTP — полностью stateless

## Принципы
- **Простота прежде всего** — одна задача, один файл, три функции
- **Никаких зависимостей** — если нужна библиотека, скорее всего это перебор
- **Чистые функции** — вычисление отделено от ввода/вывода
- **Понятные ошибки** — пользователь всегда знает что ввести чтобы продолжить

## Что уместно добавлять
- Именованные гипотезы: `"Введите название гипотезы: "` → `"H: дождь идёт"`
- История шагов: список `(prior, p_e_h, p_e_nh, posterior)` с выводом в конце
- Альтернативные формы байесовского обновления (log-odds, likelihood ratio)
- CSV-экспорт истории шагов через `csv.writer`
- Режим пакетной обработки через аргументы командной строки (`argparse`)

## Что НЕ уместно добавлять
- Веб-интерфейс, HTTP-сервер, REST API
- База данных, сессии, пользователи
- Внешние библиотеки (numpy, scipy) — для учебного демо избыточно
- Многопоточность / асинхронность
- GUI (tkinter, PyQt) — проект CLI-ориентированный

## Байесовская формула (для справки)
```
P(H|E) = P(E|H) × P(H)
         ─────────────────────────────────────
         P(E|H) × P(H) + P(E|¬H) × (1 − P(H))
```
Где:
- P(H) — априорная вероятность гипотезы (до наблюдения свидетельства)
- P(E|H) — вероятность свидетельства если гипотеза верна (likelihood)
- P(E|¬H) — вероятность свидетельства если гипотеза неверна
- P(H|E) — апостериорная вероятность (после наблюдения, становится новым prior)
""",
}


def agent_templates() -> dict:
    """Return agent templates for the current language."""
    return _AGENT_TEMPLATES_RU if LANG == "ru" else _AGENT_TEMPLATES_EN
