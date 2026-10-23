"""Карта отклонений сотрудника — с неё агент начинает разбор показателей.

Считает её SQL-шаблон ``db/sql/deviation_map.sql``: тот же запрос гоняется на
дампе через ``scripts/sql_debug.py``. Инструмент отдаёт модели только колонку
``problems`` — связный текст по показателю (план, динамика, группы с местом,
вывод), остальные колонки шаблона нужны для отладки.
"""
from __future__ import annotations

from typing import Any

from ..agent.guards import blank_to_none
from ..db.sqlrunner import run_template

DEFAULT_DEPTH = 2
MAX_DEPTH = 9
# Полные тексты — пока укладываются сюда; выдача инструмента режется на
# DEFAULT_OUTPUT_CAP (20 000), а хвост с остальными именами должен влезть.
_TEXT_BUDGET = 15_000
# На каждый из двух списков имён (не поместились / в норме).
_NAMES_BUDGET = 2_000


def _names_line(prefix: str, names: list[str]) -> str:
    """«prefix: A, B, C.»; не влезло в _NAMES_BUDGET — «… и ещё N.»"""
    shown: list[str] = []
    used = len(prefix)
    for name in names:
        if shown and used + len(name) + 2 > _NAMES_BUDGET:
            break
        shown.append(name)
        used += len(name) + 2
    tail = f" и ещё {len(names) - len(shown)}" if len(shown) < len(names) else ""
    return f"{prefix}: " + ", ".join(shown) + tail + "."


def deviation_map_text(ctx: Any, *, person: str | None = None, depth: Any = None) -> str:
    ctx.used_data_tools = True
    db = ctx.db
    person_key = ctx.person_key
    person = blank_to_none(person)
    if person:
        resolved = db.resolve_person(person)
        if resolved is None:
            return f"Сотрудник «{person}» не найден в данных."
        person_key = resolved
    try:
        depth = int(blank_to_none(depth) or DEFAULT_DEPTH)
    except (TypeError, ValueError):
        depth = DEFAULT_DEPTH
    depth = min(max(depth, 1), MAX_DEPTH)

    res = run_template(db.conn, "deviation_map", person_key=person_key, max_depth=depth,
                       max_rows=1000)
    if res.error:
        return res.error
    rows = [dict(zip(res.columns, row)) for row in res.rows]
    fio = db.conn.execute(
        "SELECT fio FROM person WHERE person_key = ?", (person_key,)
    ).fetchone()
    who = fio["fio"] if fio and fio["fio"] else person_key
    if not rows:
        return (
            f"Карта отклонений {who} (до {depth} уровня) пуста: нет показателей с "
            "планом либо все звёзды получены."
        )

    lines = [
        f"Карта отклонений: {who}, показатели до {depth} уровня. Сначала "
        "отклонения от плана, затем слабые сигналы, сильные результаты и норма."
    ]
    used = len(lines[0])
    rest: list[str] = []
    norm: list[str] = []
    for r in rows:
        if r["signal"] == "норма":
            norm.append(r["metric"])
            continue
        text = f"- {r['problems']}"
        # Первый (самый важный) показатель — всегда целиком.
        if len(lines) > 1 and (rest or used + len(text) > _TEXT_BUDGET):
            rest.append(f"{r['metric']} ({r['signal']})")
            continue
        lines.append(text)
        used += len(text)
    if rest:
        lines.append(_names_line("Не поместились, подробности — через metric_card", rest))
    if norm:
        lines.append(_names_line("В норме", norm))
    return "\n".join(lines)


__all__ = ["DEFAULT_DEPTH", "deviation_map_text"]
