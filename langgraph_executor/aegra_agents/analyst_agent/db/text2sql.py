"""Безопасное исполнение SQL, написанного моделью (text2sql).

Агент пишет обычные SELECT'ы по документированной схеме, поэтому защита стоит
на стороне базы, а не промпта:

* ``PRAGMA query_only`` + авторизатор sqlite3 — разрешены только чтение и
  вызовы функций; запись, DDL, ATTACH и PRAGMA отклоняются на этапе подготовки
  запроса, то есть до выполнения;
* один стейтмент за вызов (``sqlite3`` и сам откажется выполнять несколько,
  но явная проверка даёт понятную ошибку);
* обработчик прогресса рвёт слишком долгий запрос;
* капы на строки, колонки и длину ячейки — контекст модели ограничен.

Текст ошибки самокорректирующий: модель видит, какие вью и колонки существуют,
и переписывает запрос без помощи человека.
"""
from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# Разрешаем только чтение и функции; всё остальное — deny на этапе prepare.
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)

_COMMENT_LINE_RE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z_0-9]*)", re.IGNORECASE)
_NO_COLUMN_RE = re.compile(r"no such column:\s*(\S+)", re.IGNORECASE)
_NO_TABLE_RE = re.compile(r"no such table:\s*(\S+)", re.IGNORECASE)
_NO_FUNCTION_RE = re.compile(r"no such function:\s*(\S+)", re.IGNORECASE)

_HINT_TAIL = (
    "Исправь запрос и вызови инструмент ещё раз: один SELECT, "
    "до {max_rows} строк и {max_cols} колонок."
)


@dataclass
class QueryResult:
    """Результат запроса: либо строки, либо человекочитаемая ошибка."""

    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    elapsed_ms: float = 0.0
    error: str | None = None
    note: str | None = None


class SafeQueryRunner:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        max_rows: int = 200,
        max_cols: int = 20,
        max_cell: int = 120,
        timeout_s: float = 3.0,
    ) -> None:
        self.conn = conn
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.max_cell = max_cell
        self.timeout_s = timeout_s
        self._installed = False

    # --- установка/снятие защиты -----------------------------------------

    def _authorize(self, action: int, *_args: Any) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def install(self) -> None:
        self.conn.execute("PRAGMA query_only = 1")
        self.conn.set_authorizer(self._authorize)
        self._installed = True

    def uninstall(self) -> None:
        self.conn.set_authorizer(None)
        self.conn.execute("PRAGMA query_only = 0")
        self._installed = False

    @contextmanager
    def writable(self) -> Iterator[None]:
        """Временно снимает защиту — только для наших собственных записей
        (например, отметки отклонения моделью), не для SQL от модели."""
        was_installed = self._installed
        if was_installed:
            self.uninstall()
        try:
            yield
        finally:
            if was_installed:
                self.install()

    # --- справка по схеме -------------------------------------------------

    def _views(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def _columns_of(self, name: str) -> list[str]:
        try:
            with self.writable():
                rows = self.conn.execute(f"PRAGMA table_info({name})").fetchall()
        except sqlite3.Error:
            return []
        return [r[1] for r in rows]

    def _referenced(self, sql: str) -> list[str]:
        known = set(self._views())
        out: list[str] = []
        for name in _TABLE_REF_RE.findall(sql):
            if name in known and name not in out:
                out.append(name)
        return out

    # --- сообщения об ошибках --------------------------------------------

    def _tail(self) -> str:
        return _HINT_TAIL.format(max_rows=self.max_rows, max_cols=self.max_cols)

    def _explain(self, exc: Exception, sql: str) -> str:
        text = str(exc)
        lines = [f"ОШИБКА SQL: {text}"]
        if "not authorized" in text.lower():
            lines.append("Разрешён только SELECT: запись, DDL, ATTACH и PRAGMA запрещены.")
        elif "interrupted" in text.lower():
            lines.append(
                "Запрос слишком тяжёлый или долгий: добавь фильтр по date/person_key "
                "либо LIMIT."
            )
        elif m := _NO_COLUMN_RE.search(text):
            column = m.group(1).split(".")[-1]
            lines.append(f"Колонки {column} нет.")
            for view in self._referenced(sql) or ["v_fact"]:
                cols = self._columns_of(view)
                if cols:
                    lines.append(f"Колонки {view}: {', '.join(cols)}.")
        elif _NO_TABLE_RE.search(text):
            lines.append(f"Доступные представления: {', '.join(self._views())}.")
        elif m := _NO_FUNCTION_RE.search(text):
            lines.append(
                "Из функций доступны только стандартные SQLite плюс ru_lower(x) "
                "и pct(a, b)."
            )
        lines.append(self._tail())
        return "\n".join(lines)

    # --- исполнение -------------------------------------------------------

    def _precheck(self, sql: str) -> str | None:
        stripped = _COMMENT_BLOCK_RE.sub(" ", _COMMENT_LINE_RE.sub(" ", sql)).strip()
        stripped = stripped.rstrip(";").strip()
        if not stripped:
            return f"ОШИБКА SQL: пустой запрос.\n{self._tail()}"
        head = stripped.lstrip("( ").split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            return (
                f"ОШИБКА SQL: запрос должен начинаться с SELECT или WITH, а начинается "
                f"с {head}.\n{self._tail()}"
            )
        if not sqlite3.complete_statement(stripped + ";"):
            return f"ОШИБКА SQL: незавершённый запрос.\n{self._tail()}"
        # Несколько стейтменов: после отделения первого остаётся хвост.
        for i in range(len(stripped)):
            if stripped[i] == ";" and sqlite3.complete_statement(stripped[: i + 1]):
                if stripped[i + 1 :].strip():
                    return (
                        "ОШИБКА SQL: за один вызов выполняется один запрос.\n"
                        + self._tail()
                    )
        return None

    def run(self, sql: str) -> QueryResult:
        if not self._installed:
            self.install()
        problem = self._precheck(sql or "")
        if problem:
            return QueryResult(error=problem)

        deadline = time.monotonic() + self.timeout_s
        self.conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
        started = time.monotonic()
        try:
            cur = self.conn.execute(sql)
            columns = [d[0] for d in (cur.description or [])]
            if len(columns) > self.max_cols:
                return QueryResult(
                    error=(
                        f"ОШИБКА SQL: в выборке {len(columns)} колонок, максимум "
                        f"{self.max_cols}. Перечисли нужные колонки явно.\n{self._tail()}"
                    )
                )
            fetched = cur.fetchmany(self.max_rows + 1)
        except sqlite3.Error as exc:
            return QueryResult(error=self._explain(exc, sql))
        finally:
            self.conn.set_progress_handler(None, 0)

        truncated = len(fetched) > self.max_rows
        rows = [self._clip_row(r) for r in fetched[: self.max_rows]]
        return QueryResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            note=(
                f"показаны первые {self.max_rows} строк — это НЕ полное число случаев"
                if truncated
                else None
            ),
        )

    def _clip_row(self, row: Any) -> tuple:
        out = []
        for value in tuple(row):
            if isinstance(value, str) and len(value) > self.max_cell:
                value = value[: self.max_cell] + "…"
            out.append(value)
        return tuple(out)


__all__ = ["QueryResult", "SafeQueryRunner"]
