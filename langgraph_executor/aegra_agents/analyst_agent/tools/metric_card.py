"""Карточка показателя — один вызов вместо десятка узких выборок.

Собирает всё, что обычно нужно про показатель: справку, ряд по периодам с
вердиктами, разрезы, состав с весами влияния, справку из базы знаний. Каждый
вызов инструмента стоит отдельного обмена с моделью, поэтому лучше один
широкий, чем пять узких.
"""
from __future__ import annotations

from typing import Any

from ..db.sqlrunner import QueryResult, run_template

_MAX_CANDIDATES = 8


def _rows(res: QueryResult) -> list[dict[str, Any]]:
    return [dict(zip(res.columns, row)) for row in res.rows]


def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{round(value, digits):g}"
    return str(value)


def _verdict(value: Any) -> str:
    return "" if not value else str(value).replace("_", " ")


def _catalog_names(ctx: Any) -> list[str]:
    return [
        r["name"]
        for r in ctx.db.conn.execute(
            "SELECT name FROM metric ORDER BY depth, name LIMIT ?", (_MAX_CANDIDATES,)
        )
    ]


def _header(ctx: Any, name: str) -> list[str]:
    row = ctx.db.conn.execute(
        "SELECT name, description, unit, direction, kind, depth, path, parent_name, "
        "is_star, is_star_metric, star_of, kn_summary, kn_aliases, kn_cumulative, "
        "kn_plan_semantics FROM v_metric WHERE name = ?",
        (name,),
    ).fetchone()
    if row is None:
        return [name]
    lines = [f"ПОКАЗАТЕЛЬ: {row['name']}"]
    if row["description"]:
        lines.append(f"Описание: {row['description']}")
    facts = []
    if row["unit"]:
        facts.append(f"единицы — {row['unit']}")
    if row["direction"]:
        facts.append(
            "больше — лучше" if row["direction"] != "обратная" else "меньше — лучше"
        )
    if row["kind"] and row["kind"] != "уровень":
        facts.append(f"вид — {row['kind']}")
    if row["path"]:
        facts.append(f"место в дереве — {row['path']}")
    if row["is_star"]:
        facts.append("это звезда: числа нет, только получена или нет")
    if row["star_of"]:
        facts.append(f"влияет на звезду «{row['star_of']}»")
    if facts:
        lines.append("Свойства: " + "; ".join(facts) + ".")
    if row["kn_summary"]:
        knowledge = [row["kn_summary"]]
        if row["kn_aliases"]:
            knowledge.append(f"обозначается также: {row['kn_aliases']}")
        if row["kn_cumulative"]:
            knowledge.append("показатель накопительный")
        if row["kn_plan_semantics"]:
            knowledge.append(f"про план: {row['kn_plan_semantics']}")
        lines.append("Справка из базы знаний: " + "; ".join(knowledge))
    return lines


def _periods(ctx: Any, person: str, name: str, date: str | None) -> list[str]:
    res = run_template(
        ctx.db.conn, "tool_metric_card", person_key=person, metric=name, row_limit=40
    )
    rows = _rows(res)
    if date:
        rows = [r for r in rows if r["date"] == date]
    if not rows:
        return ["Значений по периодам нет."]
    lines = ["ЗНАЧЕНИЯ ПО ПЕРИОДАМ (последнее сверху; у показателей даты могут различаться)"]
    for r in rows:
        if r["star_received"] is not None:
            lines.append(
                f"- {r['date']}: звезда " + ("получена" if r["star_received"] else "НЕ получена")
            )
            continue
        bits = [f"факт {_num(r['fact'])}"]
        if r["plan"] is not None:
            bits.append(f"план {_num(r['plan'])}")
        if r["ex"] is not None:
            bits.append(f"выполнение плана {_num(r['ex'], 1)} %")
        if r["rr"] is not None:
            bits.append(f"run rate {_num(r['rr'])}")
        if r["plan_status"]:
            piece = _verdict(r["plan_status"])
            if r["plan_dev_pct"] is not None:
                piece += f" на {abs(r['plan_dev_pct']):.1f} %"
            bits.append(piece)
        if r["pop_status"]:
            piece = _verdict(r["pop_status"])
            if r["pop_change_pct"] is not None:
                piece += f" ({r['pop_change_pct']:+.1f} % к прошлому периоду)"
            bits.append(piece)
        if r["trend_status"]:
            bits.append(f"тренд — {_verdict(r['trend_status'])}")
        if r["rel_status"]:
            bits.append(_verdict(r["rel_status"]))
        if r["is_anomaly"]:
            bits.append("аномалия на фоне коллег")
        lines.append(f"- {r['date']}: " + ", ".join(bits))
    return lines


def _elements(ctx: Any, person: str, name: str) -> list[str]:
    rows = _rows(
        run_template(
            ctx.db.conn, "tool_metric_elements", person_key=person, metric=name
        )
    )
    if not rows:
        return []
    lines = ["РАЗРЕЗЫ (последнее значение каждого)"]
    for r in rows:
        bits = [f"факт {_num(r['fact'])}"]
        if r["plan"] is not None:
            bits.append(f"план {_num(r['plan'])}")
        if r["plan_status"]:
            piece = _verdict(r["plan_status"])
            if r["plan_dev_pct"] is not None:
                piece += f" на {abs(r['plan_dev_pct']):.1f} %"
            bits.append(piece)
        if r["pop_status"]:
            bits.append(_verdict(r["pop_status"]))
        lines.append(f"- {r['element']} ({r['date']}): " + ", ".join(bits))
    return lines


def _children(ctx: Any, person: str, name: str, depth: int) -> list[str]:
    rows = _rows(
        run_template(
            ctx.db.conn, "tool_metric_children",
            person_key=person, metric=name, max_depth=max(1, min(depth, 4)),
        )
    )
    if not rows:
        return []
    lines = ["СОСТАВ (влияющие показатели; у них своя дата последнего значения)"]
    for r in rows:
        indent = "  " * (int(r["child_depth"] or 1) - 1)
        bits = []
        # Доля среди сиблингов и исходный вес влияния — разные числа: у
        # единственного ребёнка доля 100 %, а вес из данных может быть иным.
        if r["share_pct"] is not None:
            bits.append(f"вес среди влияющих {_num(r['share_pct'], 0)} %")
        if r["influent_percent"] is not None and (
            r["share_pct"] is None or round(r["influent_percent"]) != round(r["share_pct"])
        ):
            bits.append(f"влияние по данным {_num(r['influent_percent'], 0)} %")
        if r["fact"] is not None:
            piece = f"факт {_num(r['fact'])}"
            if r["plan"] is not None:
                piece += f" при плане {_num(r['plan'])}"
            bits.append(piece)
        if r["plan_status"]:
            piece = _verdict(r["plan_status"])
            if r["plan_dev_pct"] is not None:
                piece += f" на {abs(r['plan_dev_pct']):.1f} %"
            bits.append(piece)
        if r["pop_status"]:
            bits.append(_verdict(r["pop_status"]))
        if r["date"]:
            bits.append(f"на {r['date']}")
        lines.append(f"{indent}- {r['child']}: " + (", ".join(bits) or "данных нет"))
    return lines


def _peers(ctx: Any, person: str, name: str) -> list[str]:
    rows = _rows(
        run_template(ctx.db.conn, "enrich_peer", person_key=person, row_limit=40)
    )
    rows = [r for r in rows if r["metric"] == name]
    if not rows:
        return []
    lines = ["НА ФОНЕ КОЛЛЕГ"]
    for r in rows:
        bits = [f"среднее {_num(r['mean_fact'])}"]
        if r["top20_mean_fact"] is not None:
            bits.append(f"у сильнейших {_num(r['top20_mean_fact'])}")
        if r["hit_rate"] is not None:
            bits.append(f"план выполняет {_num(r['hit_rate'], 0)} % коллег")
        if r["rank_raw"]:
            bits.append(f"место {r['rank_raw']}")
        lines.append(f"- {r['level_name']}: " + ", ".join(bits))
    return lines


def metric_card_text(
    ctx: Any,
    *,
    metric: str,
    person: str | None = None,
    date: str | None = None,
    depth: int = 2,
) -> str:
    """Карточка показателя одним текстом."""
    ctx.used_data_tools = True
    ref = ctx.db.resolve_metric(metric)
    if ref is None:
        names = ", ".join(_catalog_names(ctx))
        return (
            f"Показатель «{metric}» не найден в данных. "
            f"Есть, например: {names}. Уточни название."
        )
    person_key = ctx.person_key
    if person:
        resolved = ctx.db.resolve_person(person)
        if resolved:
            person_key = resolved

    blocks = [
        "\n".join(_header(ctx, ref.name)),
        "\n".join(_periods(ctx, person_key, ref.name, date)),
    ]
    for part in (
        _elements(ctx, person_key, ref.name),
        _children(ctx, person_key, ref.name, depth),
        _peers(ctx, person_key, ref.name),
    ):
        if part:
            blocks.append("\n".join(part))
    return "\n\n".join(blocks)


__all__ = ["metric_card_text"]
