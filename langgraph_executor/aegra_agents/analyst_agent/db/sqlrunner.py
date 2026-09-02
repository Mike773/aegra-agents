"""Параметризованные SQL-шаблоны: загрузка, исполнение, рендер.

Один и тот же ``.sql``-файл используют инструменты агента, обогащение промпта и
отладочный CLI (``scripts/sql_debug.py``) — поэтому запрос на проме
воспроизводится дословно, без «а что же там выполнилось на самом деле».

Шапка файла разбирается машинно::

    -- name: tool_metric_history
    -- summary: Ряд значений показателя по периодам
    -- params: person_key:str, metric:str, element:str? = NULL, limit:int = 60
    -- returns: date, fact, plan, plan_status
    -- render: table

``?`` в типе — необязательный параметр; после ``=`` идёт значение по умолчанию.
Параметры биндятся как ``:name`` — конкатенации строк в SQL нет нигде.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .text2sql import QueryResult

_SQL_DIR = Path(__file__).with_name("sql")
_HEADER_RE = re.compile(r"^--\s*(name|summary|params|returns|render)\s*:\s*(.*)$", re.I)
_PARAM_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z_0-9]*)\s*:\s*(?P<type>str|int|float|bool)"
    r"(?P<optional>\?)?\s*(?:=\s*(?P<default>.+))?$"
)

_MAX_ROWS_DEFAULT = 200
_RENDER_MAX_ROWS = 60
_RENDER_MAX_CELL = 80


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: Literal["str", "int", "float", "bool"]
    optional: bool
    default: Any = None


@dataclass(frozen=True)
class SqlTemplate:
    name: str
    summary: str
    params: dict[str, ParamSpec]
    returns: list[str]
    render: str
    body: str


def _parse_default(raw: str | None, type_: str) -> Any:
    if raw is None:
        return None
    text = raw.strip()
    if text.upper() in ("NULL", "NONE", ""):
        return None
    if type_ == "int":
        return int(text)
    if type_ == "float":
        return float(text)
    if type_ == "bool":
        return text.strip().casefold() in ("1", "true", "да", "yes")
    return text.strip("'\"")


def _parse_params(raw: str) -> dict[str, ParamSpec]:
    params: dict[str, ParamSpec] = {}
    depth = 0
    chunk = ""
    # Разбиваем по запятым верхнего уровня: значение по умолчанию может
    # содержать запятую внутри скобок.
    for ch in raw:
        if ch == "," and depth == 0:
            if chunk.strip():
                spec = _parse_param(chunk)
                params[spec.name] = spec
            chunk = ""
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        chunk += ch
    if chunk.strip():
        spec = _parse_param(chunk)
        params[spec.name] = spec
    return params


def _parse_param(chunk: str) -> ParamSpec:
    m = _PARAM_RE.match(chunk.strip())
    if not m:
        raise ValueError(f"не разобран параметр шаблона: {chunk!r}")
    type_ = m.group("type")
    has_default = m.group("default") is not None
    return ParamSpec(
        name=m.group("name"),
        type=type_,  # type: ignore[arg-type]
        optional=bool(m.group("optional")) or has_default,
        default=_parse_default(m.group("default"), type_),
    )


def _parse_template(name: str, text: str) -> SqlTemplate:
    header: dict[str, str] = {}
    body_lines: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m and not body_lines:
            header[m.group(1).lower()] = m.group(2).strip()
            continue
        body_lines.append(line)
    return SqlTemplate(
        name=header.get("name", name),
        summary=header.get("summary", ""),
        params=_parse_params(header.get("params", "")),
        returns=[c.strip() for c in header.get("returns", "").split(",") if c.strip()],
        render=header.get("render", "table"),
        body="\n".join(body_lines).strip(),
    )


@lru_cache(maxsize=None)
def load_sql(name: str) -> SqlTemplate:
    """Читает и разбирает шаблон по имени файла (без расширения)."""
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", name):
        raise ValueError(f"недопустимое имя шаблона: {name!r}")
    path = _SQL_DIR / f"{name}.sql"
    if not path.is_file():
        raise FileNotFoundError(f"нет шаблона {name}.sql")
    return _parse_template(name, path.read_text(encoding="utf-8"))


def list_templates(prefix: str | None = None) -> list[SqlTemplate]:
    """Все шаблоны каталога (кроме схемы), опционально с фильтром по префиксу."""
    out: list[SqlTemplate] = []
    for path in sorted(_SQL_DIR.glob("*.sql")):
        if path.stem == "schema":
            continue
        if prefix and not path.stem.startswith(prefix):
            continue
        out.append(load_sql(path.stem))
    return out


def _coerce(value: Any, spec: ParamSpec) -> Any:
    if value is None:
        return None
    if spec.type == "int":
        return int(value)
    if spec.type == "float":
        return float(value)
    if spec.type == "bool":
        return 1 if value else 0
    text = str(value).strip()
    return text or None


def run_template(
    conn: sqlite3.Connection,
    name: str,
    *,
    max_rows: int = _MAX_ROWS_DEFAULT,
    **params: Any,
) -> QueryResult:
    """Исполняет шаблон с именованными параметрами."""
    tpl = load_sql(name)
    unknown = set(params) - set(tpl.params)
    if unknown:
        raise ValueError(
            f"{name}: неизвестные параметры: {', '.join(sorted(unknown))}; "
            f"допустимы: {', '.join(tpl.params)}"
        )
    bound: dict[str, Any] = {}
    for pname, spec in tpl.params.items():
        if pname in params and params[pname] is not None:
            bound[pname] = _coerce(params[pname], spec)
        elif spec.optional:
            bound[pname] = spec.default
        else:
            raise ValueError(f"{name}: не задан обязательный параметр {pname}")
    try:
        cur = conn.execute(tpl.body, bound)
        columns = [d[0] for d in (cur.description or [])]
        fetched = cur.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        return QueryResult(error=f"ОШИБКА SQL в шаблоне {name}: {exc}")
    truncated = len(fetched) > max_rows
    return QueryResult(
        columns=columns,
        rows=[tuple(r) for r in fetched[:max_rows]],
        truncated=truncated,
        note=(
            f"показаны первые {max_rows} строк — это НЕ полное число случаев"
            if truncated
            else None
        ),
    )


# --- рендер ---------------------------------------------------------------

def _fmt(value: Any, round_to: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        text = f"{round(value, round_to):g}"
        return text
    return str(value)


def _clip(text: str, max_cell: int) -> str:
    return text if len(text) <= max_cell else text[: max_cell - 1] + "…"


def render_markdown(
    res: QueryResult,
    *,
    max_rows: int = _RENDER_MAX_ROWS,
    drop_empty_cols: bool = True,
    round_to: int = 2,
    max_cell: int = _RENDER_MAX_CELL,
) -> str:
    """Результат → компактная Markdown-таблица; пустые колонки отбрасываются."""
    if res.error:
        return res.error
    if not res.rows:
        return "Данных не нашлось."
    rows = res.rows[:max_rows]
    cells = [[_clip(_fmt(v, round_to), max_cell) for v in row] for row in rows]
    keep = list(range(len(res.columns)))
    if drop_empty_cols:
        keep = [i for i in keep if any(row[i] for row in cells)]
    if not keep:
        return "Данных не нашлось."
    header = "| " + " | ".join(res.columns[i] for i in keep) + " |"
    sep = "| " + " | ".join("---" for _ in keep) + " |"
    body = ["| " + " | ".join(row[i] for i in keep) + " |" for row in cells]
    out = [header, sep, *body]
    if res.truncated or len(res.rows) > max_rows:
        out.append(
            res.note or f"показаны первые {len(rows)} строк — это НЕ полное число случаев"
        )
    return "\n".join(out)


def render_kv(res: QueryResult, *, round_to: int = 2) -> str:
    """Одна строка результата → «колонка: значение» по строке на пару."""
    if res.error:
        return res.error
    if not res.rows:
        return ""
    row = res.rows[0]
    return "\n".join(
        f"{col}: {_fmt(value, round_to)}"
        for col, value in zip(res.columns, row)
        if value is not None and str(value) != ""
    )


def render_lines(res: QueryResult, template: str, *, round_to: int = 2) -> list[str]:
    """Строки результата → тексты по шаблону вида '{metric}: {fact}'."""
    if res.error or not res.rows:
        return []
    out: list[str] = []
    for row in res.rows:
        values = {col: _fmt(v, round_to) for col, v in zip(res.columns, row)}
        out.append(template.format(**values))
    return out


def render_csv(res: QueryResult) -> str:
    if res.error:
        return res.error
    lines = [",".join(res.columns)]
    for row in res.rows:
        lines.append(",".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


__all__ = [
    "ParamSpec",
    "QueryResult",
    "SqlTemplate",
    "list_templates",
    "load_sql",
    "render_csv",
    "render_kv",
    "render_lines",
    "render_markdown",
    "run_template",
]
