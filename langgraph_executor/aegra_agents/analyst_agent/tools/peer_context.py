"""Сравнение с коллегами по группам от узкой к широкой.

Служебные имена полей и коды уровней в выдачу не попадают: группа называется
так, как она подписана в данных («по офису»), а числа — словами.
"""
from __future__ import annotations

from typing import Any

from ..agent.guards import blank_to_none
from ..db.sqlrunner import run_template


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, digits):g}"
    return str(value)


def peer_context_text(
    ctx: Any, *, metric: str | None, person: str | None = None
) -> str:
    ctx.used_data_tools = True
    # Модель присылает «не задано» и пустой строкой, и пустым объектом.
    metric = blank_to_none(metric)
    if not isinstance(metric, str) or not metric.strip():
        return "Не указано название показателя. Назови показатель и повтори вызов."
    ref = ctx.db.resolve_metric(metric)
    if ref is None:
        return f"Показатель «{metric}» не найден в данных."
    person_key = ctx.person_key
    person = blank_to_none(person)
    if person:
        resolved = ctx.db.resolve_person(person)
        if resolved:
            person_key = resolved

    res = run_template(ctx.db.conn, "enrich_peer", person_key=person_key, row_limit=60)
    rows = [
        dict(zip(res.columns, row))
        for row in res.rows
        if dict(zip(res.columns, row))["metric"] == ref.name
    ]
    if not rows:
        return (
            f"Данных по группам сравнения для «{ref.name}» не пришло — "
            "сравнить с коллегами нечем."
        )

    own = ctx.db.conn.execute(
        "SELECT fact, date, plan_status, rel_status, peer_status FROM v_fact_latest "
        "WHERE person_key = ? AND metric = ? AND element IS NULL",
        (person_key, ref.name),
    ).fetchone()

    lines = [f"«{ref.name}» на фоне коллег (группы от узкой к широкой)"]
    if own is not None and own["fact"] is not None:
        tail = f", {own['rel_status'].replace('_', ' ')}" if own["rel_status"] else ""
        lines.append(f"- Свой результат на {own['date']}: {_num(own['fact'])}{tail}.")
    for r in rows:
        bits = [f"в среднем {_num(r['mean_fact'])}"]
        if r["median"] is not None:
            bits.append(f"медиана {_num(r['median'])}")
        if r["top20_mean_fact"] is not None:
            bits.append(f"у самых сильных {_num(r['top20_mean_fact'])}")
        if r["hit_rate"] is not None:
            bits.append(f"план выполняет {_num(r['hit_rate'], 0)} % коллег")
        if r["rank_raw"]:
            bits.append(f"место сотрудника {r['rank_raw']}")
        elif r["percentile"] is not None:
            bits.append(f"сотрудник выше {_num(r['percentile'], 0)} % коллег")
        size = f" ({r['total_objects']} человек)" if r["total_objects"] else ""
        lines.append(f"- {r['level_name']}{size}: " + ", ".join(bits) + ".")
    return "\n".join(lines)


__all__ = ["peer_context_text"]
