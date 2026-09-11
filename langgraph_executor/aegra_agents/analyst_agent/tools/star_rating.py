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


def star_rating_text(
    ctx: Any, *, person: str | None = None, quarter: str | None = None
) -> str:
    ctx.used_data_tools = True
    person_key = ctx.person_key
    person = blank_to_none(person)
    if person:
        resolved = ctx.db.resolve_person(person)
        if resolved is None:
            return f"Сотрудник «{person}» не найден в данных."
        person_key = resolved
    year, q = parse_quarter(quarter)
    if quarter and year is None and q is None:
        return (
            f"Не разобрал квартал «{quarter}». Назови его как «3 квартал 2026» "
            "или оставь пустым — покажу все."
        )

    res = run_template(
        ctx.db.conn, "enrich_rating", person_key=person_key, year=year, quarter=q,
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


__all__ = ["parse_quarter", "star_rating_text"]
