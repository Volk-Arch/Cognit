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
