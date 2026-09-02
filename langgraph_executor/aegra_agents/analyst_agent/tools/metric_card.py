"""Карточка показателя — один вызов вместо десятка узких выборок.

Собирает всё, что обычно нужно про показатель: справку, ряд по периодам с
вердиктами, разрезы, состав с весами влияния, справку из базы знаний. Каждый
вызов инструмента стоит отдельного обмена с моделью, поэтому лучше один
широкий, чем пять узких.
"""
from __future__ import annotations

from typing import Any

from ..agent.guards import blank_to_none
from ..db.sqlrunner import QueryResult, run_template

_MAX_CANDIDATES = 8
# Разрезов у показателя бывают сотни: в карточку идёт только край выборки.
_WORST_ELEMENTS = 8
_BEST_ELEMENTS = 2


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


def _person_info(ctx: Any, person: str, name: str) -> dict[str, Any]:
    """Как показатель выглядит у ЭТОГО человека: есть ли свой итог и сколько
    разрезов. Считается на человека: у руководителя итог может быть, а у
    сотрудника тот же показатель представлен только разрезами."""
    row = ctx.db.conn.execute(
        "SELECT has_aggregate, n_elements FROM v_metric_person "
        "WHERE person_key = ? AND metric = ?",
        (person, name),
    ).fetchone()
    if row is None:
        return {"has_aggregate": 1, "n_elements": 0}
    return {"has_aggregate": row["has_aggregate"], "n_elements": row["n_elements"]}


def _header(ctx: Any, name: str, info: dict[str, Any] | None = None) -> list[str]:
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
    info = info or {}
    n_elements = info.get("n_elements") or 0
    if n_elements and not info.get("has_aggregate"):
        facts.append(
            f"собственного итога нет — показатель есть только разрезами ({n_elements}); "
            "складывать разрезы в итог нельзя"
        )
    elif n_elements:
        facts.append(f"разрезов {n_elements}")
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


def _periods(
    ctx: Any, person: str, name: str, date: str | None, info: dict[str, Any] | None = None
) -> list[str]:
    res = run_template(
        ctx.db.conn, "tool_metric_card", person_key=person, metric=name, row_limit=40
    )
    rows = _rows(res)
    if date:
        rows = [r for r in rows if r["date"] == date]
    if not rows:
        info = info or {}
        if info.get("n_elements") and not info.get("has_aggregate"):
            # Не «данных нет», а «нет собственного итога»: ниже идут разрезы с
            # данными, и без этой оговорки карточка противоречила бы сама себе.
            return [
                "СВОЕГО ИТОГА У ПОКАЗАТЕЛЯ НЕТ: он существует только разрезами "
                f"({info['n_elements']}). Ниже — разрезы; складывать их в итог нельзя."
            ]
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


def _element_line(r: dict[str, Any]) -> str:
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
    return f"- {r['element']} ({r['date']}): " + ", ".join(bits)


def _elements(ctx: Any, person: str, name: str, total: int) -> list[str]:
    """Край выборки разрезов: худшие и пара лучших.

    Перечислять сотни разрезов бессмысленно — модель получает верхушку и
    подсказку, как достать остальное запросом.
    """
    worst = _rows(run_template(
        ctx.db.conn, "tool_metric_elements", person_key=person, metric=name,
        worst_first=True, row_limit=_WORST_ELEMENTS,
    ))
    if not worst:
        return []
    shown = {r["element"] for r in worst}
    best = [
        r for r in _rows(run_template(
            ctx.db.conn, "tool_metric_elements", person_key=person, metric=name,
            worst_first=False, row_limit=_BEST_ELEMENTS + len(shown),
        ))
        if r["element"] not in shown
    ][:_BEST_ELEMENTS]

    lines = ["РАЗРЕЗЫ (последнее значение каждого; худшие сверху)"]
    lines.extend(_element_line(r) for r in worst)
    if best:
        lines.append("Лучшие разрезы:")
        lines.extend(_element_line(r) for r in best)
    total = total or len(worst)
    if total > len(worst) + len(best):
        lines.append(
            f"Показаны {len(worst) + len(best)} из {total} разрезов. Остальные — "
            "запросом с фильтром по имени (LIKE) и LIMIT либо "
            "metric_card с аргументом element."
        )
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


def _single_element(ctx: Any, person: str, name: str, element: str) -> str:
    """Карточка одного разреза: его ряд по периодам."""
    known = [
        r["element"]
        for r in ctx.db.conn.execute(
            "SELECT DISTINCT element FROM v_fact WHERE person_key = ? AND metric = ? "
            "AND element IS NOT NULL ORDER BY element LIMIT ?",
            (person, name, _MAX_CANDIDATES),
        )
    ]
    match = ctx.db.resolve_element(person, name, element)
    if match is None:
        return (
            f"Разрез «{element}» у показателя «{name}» не найден. "
            f"Есть, например: {', '.join(known)}. "
            "Полный список — запросом к v_fact_latest с фильтром по имени."
        )
    rows = _rows(run_template(
        ctx.db.conn, "tool_metric_history", person_key=person, metric=name,
        element=match, row_limit=40,
    ))
    lines = [f"ПОКАЗАТЕЛЬ «{name}», РАЗРЕЗ «{match}» — значения по периодам"]
    if not rows:
        lines.append("Значений нет.")
        return "\n".join(lines)
    for r in rows:
        bits = [f"факт {_num(r['fact'])}"]
        if r["plan"] is not None:
            bits.append(f"план {_num(r['plan'])}")
        if r["plan_status"]:
            piece = _verdict(r["plan_status"])
            bits.append(piece)
        if r["pop_status"]:
            piece = _verdict(r["pop_status"])
            if r["pop_change_pct"] is not None:
                piece += f" ({r['pop_change_pct']:+.1f} % к прошлому периоду)"
            bits.append(piece)
        lines.append(f"- {r['date']}: " + ", ".join(bits))
    return "\n".join(lines)


def metric_card_text(
    ctx: Any,
    *,
    metric: str,
    person: str | None = None,
    date: str | None = None,
    depth: int = 2,
    element: str | None = None,
) -> str:
    """Карточка показателя одним текстом."""
    ctx.used_data_tools = True
    # Модель присылает «не задано» и пустой строкой, и пустым объектом.
    metric = blank_to_none(metric)
    names = ", ".join(_catalog_names(ctx))
    if not isinstance(metric, str) or not metric.strip():
        return (
            "Не указано название показателя. "
            f"Есть, например: {names}. Назови показатель и повтори вызов."
        )
    ref = ctx.db.resolve_metric(metric)
    if ref is None:
        return (
            f"Показатель «{metric}» не найден в данных. "
            f"Есть, например: {names}. Уточни название."
        )
    person_key = ctx.person_key
    person = blank_to_none(person)
    if person and isinstance(person, str):
        resolved = ctx.db.resolve_person(person)
        if resolved:
            person_key = resolved

    element = blank_to_none(element)
    if isinstance(element, str) and element.strip():
        return _single_element(ctx, person_key, ref.name, element.strip())

    info = _person_info(ctx, person_key, ref.name)
    blocks = [
        "\n".join(_header(ctx, ref.name, info)),
        "\n".join(_periods(ctx, person_key, ref.name, date, info)),
    ]
    for part in (
        _elements(ctx, person_key, ref.name, info.get("n_elements") or 0),
        _children(ctx, person_key, ref.name, depth),
        _peers(ctx, person_key, ref.name),
    ):
        if part:
            blocks.append("\n".join(part))
    return "\n\n".join(blocks)


__all__ = ["metric_card_text"]
