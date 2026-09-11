"""Рейтинг по звёздам: место сотрудника среди коллег уровня за квартал.

Уровни называются так, как они подписаны в агрегатах («ГОСБ», «ТБ»); коды и
служебные имена полей в выдачу не попадают. Инструмент привязывается только
когда рейтинг пришёл (а он приходит только вместе со звёздами).
"""
from __future__ import annotations

import re
from typing import Any

from ..agent.guards import blank_to_none
from ..db.sqlrunner import run_template

_QUARTER_RE = re.compile(r"(?<!\d)([1-4])(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def parse_quarter(text: Any) -> tuple[int | None, int | None]:
    """«3 квартал 2026», «2026 Q3», «Q3», «3» → (год, квартал); пусто → (None, None)."""
    text = blank_to_none(text)
    if not isinstance(text, str) or not text.strip():
        return None, None
    year_m = _YEAR_RE.search(text)
    year = int(year_m.group(1)) if year_m else None
    rest = _YEAR_RE.sub(" ", text)
    quarter_m = _QUARTER_RE.search(rest)
    quarter = int(quarter_m.group(1)) if quarter_m else None
    return year, quarter


def _period(year: Any, quarter: Any) -> str:
    return f"{quarter} квартал {year}"


def rating_levels(db: Any) -> list[tuple[str, str]]:
    """Уровни рейтинга (код, название) от узкой группы к широкой — как в базе."""
    rows = db.conn.execute(
        "SELECT level, MIN(level_name) AS level_name, MIN(level_order) AS level_order "
        "FROM rating GROUP BY level ORDER BY level_order, level"
    ).fetchall()
    return [(str(r["level"]), str(r["level_name"])) for r in rows]


def levels_text(db: Any) -> str:
    """«ГОСБ (GOSB), ТБ (TB), Сбер (SBER)»; если названия нет — только код."""
    parts = []
    for code, name in rating_levels(db):
        parts.append(code if name.casefold() == code.casefold() else f"{name} ({code})")
    return ", ".join(parts)


def resolve_level(db: Any, text: Any) -> str | None:
    """Название или код уровня → код из базы: точное совпадение, затем
    подстрока в обе стороны (модель пишет «в ГОСБ», «по территории»)."""
    text = blank_to_none(text)
    if not isinstance(text, str) or not text.strip():
        return None
    norm = " ".join(text.split()).casefold()
    levels = rating_levels(db)
    for code, name in levels:
        if norm in (code.casefold(), name.casefold()):
            return code
    hits = [
        code for code, name in levels
        if any(k and (k in norm or norm in k) for k in (code.casefold(), name.casefold()))
    ]
    return hits[0] if len(hits) == 1 else None


def star_rating_text(
    ctx: Any,
    *,
    person: str | None = None,
    quarter: str | None = None,
    level: str | None = None,
) -> str:
    ctx.used_data_tools = True
    db = ctx.db
    person_key = ctx.person_key
    person = blank_to_none(person)
    level = blank_to_none(level)
    if person:
        resolved = db.resolve_person(person)
        if resolved is None:
            # Модель кладёт название уровня («ГОСБ») в person — это уровень, не человек.
            as_level = resolve_level(db, person)
            if as_level is None or level:
                return (
                    f"Сотрудник «{person}» не найден в данных. Если это уровень "
                    f"рейтинга, передай его аргументом level; уровни: {levels_text(db)}."
                )
            level = as_level
        else:
            person_key = resolved
    level_code: str | None = None
    if level:
        level_code = resolve_level(db, level)
        if level_code is None:
            return f"Уровня «{level}» в рейтинге нет. Уровни: {levels_text(db)}."
    year, q = parse_quarter(quarter)
    if quarter and year is None and q is None:
        return (
            f"Не разобрал квартал «{quarter}». Назови его как «3 квартал 2026» "
            "или оставь пустым — покажу все."
        )

    res = run_template(
        db.conn, "enrich_rating", person_key=person_key, year=year, quarter=q,
        level=level_code,
    )
    rows = [dict(zip(res.columns, row)) for row in res.rows]
    fio_row = ctx.db.conn.execute(
        "SELECT fio FROM person WHERE person_key = ?", (person_key,)
    ).fetchone()
    who = fio_row["fio"] if fio_row and fio_row["fio"] else person_key
    if not rows:
        when = f" за {_period(year, q)}" if q and year else ""
        return f"Рейтинга по звёздам у {who}{when} нет."

    by_period: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_period.setdefault((r["year"], r["quarter"]), []).append(r)
    lines = [f"Рейтинг по звёздам: {who} (уровни от узкой группы к широкой)"]
    for (yr, qt), items in by_period.items():
        bits = []
        for r in items:
            size = f" из {r['staff']}" if r["staff"] else ""
            bits.append(f"{r['level_name']} — {r['place']} место{size}")
        lines.append(f"- {_period(yr, qt)}: " + "; ".join(bits) + ".")
    return "\n".join(lines)


__all__ = ["levels_text", "parse_quarter", "rating_levels", "resolve_level", "star_rating_text"]
