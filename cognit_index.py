#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Igor Kriusov <kriusovia@gmail.com>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
cognit_index.py — AST-based code index via tree-sitter.

Зависимости:
    pip install tree-sitter tree-sitter-python

Без зависимостей на LLM. Используется для навигации по кодовой базе:
  - tree-sitter парсит Python-файлы → символы (функции, классы, импорты)
  - BM25-style поиск по именам + докстрингам
  - Кеш индекса на диск для быстрого перезапуска

Использование:
    from cognit_index import CodeIndex
    idx = CodeIndex("/path/to/project")
    idx.build()
    results = idx.search("bayes update", top_k=5)
"""

import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cognit_core as core


# =============================================================================
# МОДЕЛИ ДАННЫХ
# =============================================================================
@dataclass
class Symbol:
    """Один символ (функция, класс, импорт, переменная) из AST."""
    name: str
    kind: str              # "function", "class", "import", "assignment"
    filepath: str
    line_start: int        # 1-indexed
    line_end: int          # 1-indexed
    signature: str = ""    # def foo(a, b) -> int   или   class Foo(Base)
    docstring: str = ""
    methods: list[str] = field(default_factory=list)  # только для классов


@dataclass
class SearchResult:
    """Результат поиска."""
    filepath: str
    symbol: Symbol
    score: float
    context: str = ""      # краткое описание для вывода


# =============================================================================
# AST-ПАРСЕР (tree-sitter)
# =============================================================================
def _init_parser():
    """Инициализирует tree-sitter parser для Python."""
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        raise ImportError(
            "tree-sitter не установлен. Установи:\n"
            "  pip install tree-sitter tree-sitter-python"
        )
    lang = Language(tspython.language())
    parser = Parser(lang)
    return parser


def _node_text(node) -> str:
    """Извлекает текст из узла AST."""
    return node.text.decode("utf-8") if node and node.text else ""


def _extract_docstring(body_node) -> str:
    """Извлекает докстринг из первого expression_statement в теле."""
    if body_node is None:
        return ""
    for child in body_node.named_children:
        if child.type == "expression_statement":
            expr = child.named_children[0] if child.named_children else None
            if expr and expr.type == "string":
                raw = _node_text(expr)
                # Убираем тройные кавычки
                for q in ('"""', "'''"):
                    if raw.startswith(q) and raw.endswith(q):
                        return raw[3:-3].strip()
                return raw.strip("\"'").strip()
        break  # докстринг — только первый statement
    return ""


def _extract_parameters(node) -> str:
    """Извлекает список параметров из function_definition."""
    params = node.child_by_field_name("parameters")
    return _node_text(params) if params else "()"


def _extract_return_type(node) -> str:
    """Извлекает возвращаемый тип из function_definition."""
    ret = node.child_by_field_name("return_type")
    if ret:
        return " -> " + _node_text(ret)
    return ""


def _parse_file(parser, filepath: str) -> list[Symbol]:
    """Парсит один Python-файл через tree-sitter → список символов."""
    try:
        source = Path(filepath).read_bytes()
    except Exception:
        return []

    tree = parser.parse(source)
    root = tree.root_node
    symbols: list[Symbol] = []

    for node in root.named_children:
        line_start = node.start_point[0] + 1  # 0-indexed → 1-indexed
        line_end = node.end_point[0] + 1

        # ── Функция ──────────────────────────────────────────────────
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else "?"
            params = _extract_parameters(node)
            ret = _extract_return_type(node)
            body = node.child_by_field_name("body")
            doc = _extract_docstring(body)
            symbols.append(Symbol(
                name=name, kind="function", filepath=filepath,
                line_start=line_start, line_end=line_end,
                signature=f"def {name}{params}{ret}",
                docstring=doc,
            ))

        # ── Класс ────────────────────────────────────────────────────
        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node) if name_node else "?"
            superclasses = node.child_by_field_name("superclasses")
            bases = f"({_node_text(superclasses)})" if superclasses else ""
            body = node.child_by_field_name("body")
            doc = _extract_docstring(body)

            methods = []
            if body:
                for child in body.named_children:
                    if child.type == "function_definition":
                        m_name = child.child_by_field_name("name")
                        if m_name:
                            methods.append(_node_text(m_name))

            symbols.append(Symbol(
                name=name, kind="class", filepath=filepath,
                line_start=line_start, line_end=line_end,
                signature=f"class {name}{bases}",
                docstring=doc, methods=methods,
            ))

        # ── Импорт ───────────────────────────────────────────────────
        elif node.type == "import_statement":
            text = _node_text(node)
            symbols.append(Symbol(
                name=text, kind="import", filepath=filepath,
                line_start=line_start, line_end=line_end,
            ))

        elif node.type == "import_from_statement":
            text = _node_text(node)
            symbols.append(Symbol(
                name=text, kind="import", filepath=filepath,
                line_start=line_start, line_end=line_end,
            ))

        # ── Присваивание верхнего уровня ─────────────────────────────
        elif node.type == "expression_statement":
            expr = node.named_children[0] if node.named_children else None
            if expr and expr.type == "assignment":
                left = expr.child_by_field_name("left")
                if left:
                    name = _node_text(left)
                    symbols.append(Symbol(
                        name=name, kind="assignment", filepath=filepath,
                        line_start=line_start, line_end=line_end,
                    ))

        # ── Decorated functions/classes ───────────────────────────────
        elif node.type == "decorated_definition":
            inner = None
            for child in node.named_children:
                if child.type in ("function_definition", "class_definition"):
                    inner = child
                    break
            if inner and inner.type == "function_definition":
                name_node = inner.child_by_field_name("name")
                name = _node_text(name_node) if name_node else "?"
                params = _extract_parameters(inner)
                ret = _extract_return_type(inner)
                body = inner.child_by_field_name("body")
                doc = _extract_docstring(body)
                symbols.append(Symbol(
                    name=name, kind="function", filepath=filepath,
                    line_start=line_start, line_end=line_end,
                    signature=f"def {name}{params}{ret}",
                    docstring=doc,
                ))
            elif inner and inner.type == "class_definition":
                name_node = inner.child_by_field_name("name")
                name = _node_text(name_node) if name_node else "?"
                superclasses = inner.child_by_field_name("superclasses")
                bases = f"({_node_text(superclasses)})" if superclasses else ""
                body = inner.child_by_field_name("body")
                doc = _extract_docstring(body)
                methods = []
                if body:
                    for child in body.named_children:
                        if child.type == "function_definition":
                            m_name = child.child_by_field_name("name")
                            if m_name:
                                methods.append(_node_text(m_name))
                symbols.append(Symbol(
                    name=name, kind="class", filepath=filepath,
                    line_start=line_start, line_end=line_end,
                    signature=f"class {name}{bases}",
                    docstring=doc, methods=methods,
                ))

    return symbols


# =============================================================================
# BM25-ПОИСК
# =============================================================================
def _tokenize_query(text: str) -> list[str]:
    """Разбивает текст на токены для поиска (lowercase, split by non-alphanum)."""
    return [t for t in re.split(r'[^a-zA-Z0-9а-яА-ЯёЁ_]+', text.lower()) if len(t) >= 2]


def _symbol_text(sym: Symbol) -> str:
    """Собирает весь searchable текст символа."""
    parts = [sym.name]
    if sym.docstring:
        parts.append(sym.docstring)
    if sym.signature:
        parts.append(sym.signature)
    if sym.methods:
        parts.extend(sym.methods)
    return " ".join(parts).lower()


def _bm25_score(query_tokens: list[str], doc_text: str,
                avg_dl: float, n_docs: int, df: dict[str, int],
                k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 score для одного документа."""
    doc_tokens = _tokenize_query(doc_text)
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0

    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        f = tf[qt]
        idf = math.log((n_docs - df.get(qt, 0) + 0.5) / (df.get(qt, 0) + 0.5) + 1.0)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf * numerator / denominator

    return score


# =============================================================================
# CodeIndex — ОСНОВНОЙ КЛАСС
# =============================================================================
class CodeIndex:
    """
    AST-based code index для Python-проекта.

    Использование:
        idx = CodeIndex("/path/to/project")
        idx.build()
        results = idx.search("bayes update", top_k=5)
        print(idx.project_summary())
    """

    def __init__(self, project_dir: str, cache_dir: str = ""):
        """
        Args:
            project_dir: путь к корню проекта (с .py файлами)
            cache_dir:   путь для кеша индекса (default: echo_patterns/<repo>/<branch>/)
        """
        self.project_dir = str(Path(project_dir).resolve())
        self.cache_dir = cache_dir
        self.symbols: dict[str, list[Symbol]] = {}   # filepath → [Symbol]
        self.file_hashes: dict[str, str] = {}         # filepath → hash
        self._parser = None

    def build(self) -> int:
        """
        Парсит все .py файлы в project_dir через tree-sitter.
        Возвращает количество обработанных файлов.
        Инкрементальный: пропускает файлы, чей хеш не изменился.
        """
        self._parser = _init_parser()

        # Загружаем кеш если есть
        cached = self._load_cache()
        cached_hashes = cached.get("file_hashes", {}) if cached else {}
        cached_symbols = {}
        if cached:
            for filepath, sym_dicts in cached.get("symbols", {}).items():
                cached_symbols[filepath] = [Symbol(**sd) for sd in sym_dicts]

        # Собираем .py файлы
        py_files = [
            f for f in core.collect_text_files(self.project_dir)
            if f.suffix == ".py"
        ]

        parsed = 0
        for fpath in py_files:
            fstr = str(fpath)
            try:
                h = core.file_hash(fstr)
            except Exception:
                continue

            self.file_hashes[fstr] = h

            # Инкрементальный: пропускаем если хеш совпадает
            if fstr in cached_hashes and cached_hashes[fstr] == h and fstr in cached_symbols:
                self.symbols[fstr] = cached_symbols[fstr]
                continue

            syms = _parse_file(self._parser, fstr)
            self.symbols[fstr] = syms
            parsed += 1

        # Удаляем файлы которых больше нет
        existing = {str(f) for f in py_files}
        for old in list(self.symbols.keys()):
            if old not in existing:
                del self.symbols[old]
                self.file_hashes.pop(old, None)

        # Сохраняем кеш
        self._save_cache()

        total_symbols = sum(len(s) for s in self.symbols.values())
        print(f"📇 Индекс: {len(self.symbols)} файлов, {total_symbols} символов"
              f" ({parsed} обновлено)")
        return len(self.symbols)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        BM25-поиск по символам индекса.
        Возвращает top_k результатов, отсортированных по score.
        """
        query_tokens = _tokenize_query(query)
        if not query_tokens:
            return []

        # Строим все "документы" (один документ = один символ)
        all_docs: list[tuple[Symbol, str]] = []
        for filepath, syms in self.symbols.items():
            for sym in syms:
                doc_text = _symbol_text(sym)
                all_docs.append((sym, doc_text))

        if not all_docs:
            return []

        # Document frequency
        n_docs = len(all_docs)
        df: dict[str, int] = {}
        all_texts_tokenized = []
        for _, doc_text in all_docs:
            tokens = set(_tokenize_query(doc_text))
            all_texts_tokenized.append(tokens)
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        avg_dl = sum(len(_tokenize_query(dt)) for _, dt in all_docs) / max(n_docs, 1)

        # Score
        results: list[SearchResult] = []
        for (sym, doc_text) in all_docs:
            score = _bm25_score(query_tokens, doc_text, avg_dl, n_docs, df)
            if score > 0:
                context = sym.signature or sym.name
                if sym.docstring:
                    doc_short = sym.docstring[:80]
                    context += f"  # {doc_short}"
                results.append(SearchResult(
                    filepath=sym.filepath, symbol=sym,
                    score=score, context=context,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def files_for_query(self, query: str, top_k: int = 5) -> list[str]:
        """
        Возвращает уникальные файлы, релевантные запросу.
        Удобный хелпер для маршрутизации.
        """
        results = self.search(query, top_k=top_k * 3)
        seen: set[str] = set()
        files: list[str] = []
        for r in results:
            if r.filepath not in seen:
                seen.add(r.filepath)
                files.append(r.filepath)
                if len(files) >= top_k:
                    break
        return files

    def file_summary(self, filepath: str) -> str:
        """Краткое описание файла: классы, функции, импорты."""
        syms = self.symbols.get(filepath, [])
        if not syms:
            return f"{Path(filepath).name}: (пустой или не проиндексирован)"

        lines = [f"### {Path(filepath).name}"]
        for sym in syms:
            if sym.kind == "class":
                methods_str = ", ".join(sym.methods[:5])
                if len(sym.methods) > 5:
                    methods_str += f" (+{len(sym.methods) - 5})"
                lines.append(f"  class {sym.name}  [{sym.line_start}-{sym.line_end}]"
                             + (f"  методы: {methods_str}" if methods_str else ""))
            elif sym.kind == "function":
                doc = f"  # {sym.docstring[:60]}" if sym.docstring else ""
                lines.append(f"  {sym.signature}  [{sym.line_start}-{sym.line_end}]{doc}")
            elif sym.kind == "import":
                lines.append(f"  {sym.name}")
        return "\n".join(lines)

    def project_summary(self) -> str:
        """Обзор всего проекта: файлы с ключевыми символами."""
        lines = [f"# Проект: {Path(self.project_dir).name}",
                 f"Файлов: {len(self.symbols)}\n"]
        for filepath in sorted(self.symbols.keys()):
            lines.append(self.file_summary(filepath))
            lines.append("")
        return "\n".join(lines)

    def search_summary(self, query: str, top_k: int = 10) -> str:
        """
        Текстовый отчёт по поиску — для вставки в промпт Transformer.
        Группирует результаты по файлам.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return f"По запросу «{query}» ничего не найдено."

        # Группируем по файлу
        by_file: dict[str, list[SearchResult]] = {}
        for r in results:
            by_file.setdefault(r.filepath, []).append(r)

        lines = [f"Найдено {len(results)} символов в {len(by_file)} файлах:\n"]
        for filepath, file_results in by_file.items():
            rel = _relative_path(filepath, self.project_dir)
            lines.append(f"### {rel}")
            for r in file_results:
                sym = r.symbol
                score_str = f"[{r.score:.1f}]"
                if sym.kind == "function":
                    lines.append(f"  {score_str} {sym.signature}  (строки {sym.line_start}-{sym.line_end})")
                elif sym.kind == "class":
                    lines.append(f"  {score_str} {sym.signature}  (строки {sym.line_start}-{sym.line_end})")
                else:
                    lines.append(f"  {score_str} {sym.name}")
                if sym.docstring:
                    lines.append(f"         {sym.docstring[:100]}")
            lines.append("")
        return "\n".join(lines)

    # ─── Кеш ──────────────────────────────────────────────────────────
    def _cache_path(self) -> Path | None:
        if self.cache_dir:
            return Path(self.cache_dir) / "_code_index.json"
        return None

    def _save_cache(self) -> None:
        p = self._cache_path()
        if not p:
            return
        try:
            data = {
                "project_dir": self.project_dir,
                "file_hashes": self.file_hashes,
                "symbols": {
                    fp: [asdict(s) for s in syms]
                    for fp, syms in self.symbols.items()
                },
            }
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить кеш индекса: {e}")

    def _load_cache(self) -> dict | None:
        p = self._cache_path()
        if not p or not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("project_dir") != self.project_dir:
                return None  # другой проект
            return data
        except Exception:
            return None


# =============================================================================
# УТИЛИТЫ
# =============================================================================
def _relative_path(filepath: str, base: str) -> str:
    """Возвращает относительный путь если возможно."""
    try:
        return str(Path(filepath).relative_to(base))
    except ValueError:
        return filepath
