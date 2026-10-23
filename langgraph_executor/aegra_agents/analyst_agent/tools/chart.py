"""Инструмент ``build_chart`` — график под ответом в формате plotly.

Два режима. Основной — ``metric``: модель называет показатель, а ряд за
последние ``periods`` периодов (факт, план, медиана ближайшей группы) код берёт
из базы шаблоном ``tool_chart_series``. Модель чисел не переписывает: на
большом объёме GigaChat их путает, а по одной последней точке строил «линию»
из одной даты. Второй — ``traces``: график, который описал руководитель, модель
задаёт типизированным подмножеством plotly (тип, заголовки, серии x/y); линия
по одной точке не принимается.

Код чистит описание и собирает стандартную фигуру
``{"data": [...], "layout": {...}}``, которую узел агента переносит в
``additional_kwargs`` итога (``orchestrator_chart``, обёртка —
``contract.messages.chart_payload``). Библиотека plotly не
нужна: фигура — обычный JSON. Привязывается только при ``interactive_chart=true``.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..agent.guards import blank_to_none
from ..agent.runctx import RunContext
from ..db.sqlrunner import run_template
from ..prompts.chart import (
    CHART_TOOL_DESCRIPTION,
    DEFAULT_PERIODS,
    MAX_PERIODS,
    MAX_POINTS,
    MAX_TRACES,
)

TOOL_NAME = "build_chart"

ChartType = Literal["line", "bar", "scatter"]

# Тип графика модели → тип и режим серии plotly.
_PLOTLY_TRACE: dict[str, dict[str, str]] = {
    "line": {"type": "scatter", "mode": "lines+markers"},
    "bar": {"type": "bar"},
    "scatter": {"type": "scatter", "mode": "markers"},
}

_EMPTY_HINT = (
    "График не принят: передай metric — показатель, который ты подсветил в "
    "ответе (ряд по периодам код возьмёт из данных), либо traces — объекты с "
    "полями name, x и y одинаковой длины, где y — числа из данных. Исправь "
    "аргументы и вызови ещё раз либо переходи к итоговому ответу без графика."
)

_SINGLE_POINT_HINT = (
    "График не принят: у линии одна точка — это не динамика. Передай metric — "
    "показатель, который ты подсветил в ответе, и код построит его ряд за "
    "последние периоды из данных."
)


class Trace(BaseModel):
    """Одна серия: подписи точек и значения той же длины."""

    name: str = Field(..., description="Подпись серии в легенде")
    x: list[str] = Field(..., description="Подписи точек: даты, имена, разрезы")
    y: list[float | None] = Field(..., description="Значения той же длины; null — пропуск")
    type: ChartType | None = Field(
        None, description="Тип серии, если отличается от chart_type"
    )


class ChartArgs(BaseModel):
    chart_type: ChartType = Field(..., description="line | bar | scatter")
    title: str = Field(..., description="Заголовок графика")
    metric: str | None = Field(
        None,
        description=(
            "Показатель, который ты подсветил в ответе: ряд по периодам код возьмёт "
            "из данных, traces тогда не нужны"
        ),
    )
    periods: int | None = Field(
        DEFAULT_PERIODS,
        description=f"Сколько последних периодов показать для metric (до {MAX_PERIODS})",
    )
    element: str | None = Field(None, description="Разрез показателя metric, если фокус на нём")
    person: str | None = Field(
        None, description="Чей ряд для metric; по умолчанию — тот, кого разбираем"
    )
    x_title: str | None = Field(None, description="Подпись оси x")
    y_title: str | None = Field(None, description="Подпись оси y")
    traces: list[Trace] | None = Field(
        None,
        description=(
            f"Серии графика, до {MAX_TRACES} штук — только для графика, который "
            "описал руководитель; без metric"
        ),
    )


def _number(value: Any) -> float | None:
    """Число из значения модели; всё, что числом не читается, — пропуск."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().replace(" ", "").replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _axis_title(value: Any) -> dict[str, Any] | None:
    text = _text(value)
    return {"title": {"text": text}} if text else None


def _clean_trace(raw: Any, default_type: str, index: int) -> dict[str, Any] | None:
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return None
    xs = list(raw.get("x") or [])
    ys = list(raw.get("y") or [])
    if not xs or len(xs) != len(ys):
        return None
    xs = [_text(x) for x in xs[:MAX_POINTS]]
    ys = [_number(y) for y in ys[:MAX_POINTS]]
    if all(y is None for y in ys):
        return None
    kind = raw.get("type") if raw.get("type") in _PLOTLY_TRACE else default_type
    return {
        **_PLOTLY_TRACE[kind],
        "name": _text(raw.get("name")) or f"Серия {index}",
        "x": xs,
        "y": ys,
    }


def build_figure(raw: Any) -> dict[str, Any] | None:
    """Аргументы модели → фигура plotly; None, если не осталось ни одной серии."""
    if isinstance(raw, BaseModel):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return None
    chart_type = raw.get("chart_type") if raw.get("chart_type") in _PLOTLY_TRACE else "line"
    data: list[dict[str, Any]] = []
    for item in raw.get("traces") or []:
        trace = _clean_trace(item, chart_type, len(data) + 1)
        if trace is not None:
            data.append(trace)
        if len(data) >= MAX_TRACES:
            break
    if not data:
        return None
    layout: dict[str, Any] = {}
    title = _text(raw.get("title"))
    if title:
        layout["title"] = {"text": title}
    for key, axis in (("x_title", "xaxis"), ("y_title", "yaxis")):
        axis_title = _axis_title(raw.get(key))
        if axis_title:
            layout[axis] = axis_title
    return {"data": data, "layout": layout}


def _periods(value: Any) -> int:
    try:
        n = int(blank_to_none(value) or DEFAULT_PERIODS)
    except (TypeError, ValueError):
        n = DEFAULT_PERIODS
    return min(max(n, 2), MAX_PERIODS)


def metric_traces(
    ctx: RunContext,
    metric: Any,
    *,
    periods: Any = None,
    element: Any = None,
    person: Any = None,
) -> tuple[list[dict[str, Any]], str]:
    """Ряд показателя из базы → серии графика и подпись ряда для ответа модели.

    Пустой список серий — график строить не из чего; вторая часть — подсказка.
    """
    ref = ctx.db.resolve_metric(metric)
    if ref is None:
        return [], f"График не принят: показатель «{metric}» не найден в данных."
    person_key = ctx.person_key
    person = blank_to_none(person)
    if person:
        resolved = ctx.db.resolve_person(person)
        if resolved is None:
            return [], f"График не принят: сотрудник «{person}» не найден в данных."
        person_key = resolved
    # У звезды чисел нет: строить нечего, нужен её влияющий показатель.
    star = ctx.db.conn.execute(
        "SELECT m.metric_id FROM metric m WHERE m.name = ? AND m.is_star = 1", (ref.name,)
    ).fetchone()
    if star is not None:
        kids = [
            r["name"] for r in ctx.db.conn.execute(
                "SELECT DISTINCT cm.name FROM metric_edge e "
                "JOIN person p ON p.person_id = e.person_id "
                "JOIN metric cm ON cm.metric_id = e.child_id AND cm.is_star_metric = 1 "
                "WHERE p.person_key = ? AND e.parent_id = ? ORDER BY cm.name",
                (person_key, star["metric_id"]),
            )
        ]
        hint = f": {', '.join(kids)}" if kids else ""
        return [], (
            f"График не принят: «{ref.name}» — звезда, у неё нет чисел. Передай в "
            f"metric её влияющий показатель из твоего ответа{hint}."
        )
    element = blank_to_none(element)
    res = run_template(
        ctx.db.conn, "tool_chart_series", person_key=person_key, metric=ref.name,
        element=element, periods=_periods(periods),
    )
    rows = [dict(zip(res.columns, row)) for row in res.rows] if not res.error else []
    if not rows or all(r["fact"] is None for r in rows):
        where = f" по разрезу «{element}»" if element else ""
        return [], (
            f"График не принят: у «{ref.name}»{where} нет ряда значений. Если у "
            "показателя только разрезы, передай element."
        )
    dates = [r["date"] for r in rows]
    traces = [{"name": "Факт", "x": dates, "y": [r["fact"] for r in rows]}]
    if any(r["plan"] not in (None, 0) for r in rows):
        traces.append({"name": "План", "x": dates, "y": [r["plan"] for r in rows]})
    group = rows[-1]["group_name"]
    if any(r["median"] is not None for r in rows):
        label = f"Медиана коллег ({group})" if group else "Медиана коллег"
        traces.append({"name": label, "x": dates, "y": [r["median"] for r in rows]})
    series = ", ".join(t["name"].lower() for t in traces)
    span = dates[0] if len(dates) == 1 else f"{dates[0]} — {dates[-1]}"
    return traces, f"«{ref.name}», периодов {len(dates)} ({span}): {series}"


def open_star_metrics(db: Any, person_key: str) -> list[str]:
    """Влияющие показатели звёзд, не полученных в последнем периоде."""
    return [
        r["name"] for r in db.conn.execute(
            "SELECT DISTINCT cm.name FROM fact s "
            "JOIN person p ON p.person_id = s.person_id "
            "JOIN metric sm ON sm.metric_id = s.metric_id AND sm.is_star = 1 "
            "JOIN metric_edge e ON e.person_id = s.person_id AND e.parent_id = s.metric_id "
            "JOIN metric cm ON cm.metric_id = e.child_id AND cm.is_star_metric = 1 "
            "WHERE p.person_key = ? AND s.is_last_of_series = 1 AND s.star_received = 0 "
            "ORDER BY cm.name",
            (person_key,),
        )
    ]


def make_build_chart(ctx: RunContext) -> StructuredTool:
    def build_chart(
        chart_type: str,
        title: str,
        metric: str | None = None,
        periods: int | None = DEFAULT_PERIODS,
        element: str | None = None,
        person: str | None = None,
        x_title: str | None = None,
        y_title: str | None = None,
        traces: list[Any] | None = None,
    ) -> str:
        what = ""
        if blank_to_none(metric):
            traces, what = metric_traces(
                ctx, metric, periods=periods, element=element, person=person
            )
            if not traces:
                return what
        figure = build_figure({
            "chart_type": chart_type, "title": title or blank_to_none(metric),
            "traces": traces, "x_title": x_title, "y_title": y_title,
        })
        if figure is None:
            return _EMPTY_HINT
        # Линия по одной дате — не динамика: так модель «строила» график по
        # последнему значению, которое видела в ответах инструментов.
        if not what and all(
            t.get("mode") == "lines+markers" and len(t["x"]) < 2 for t in figure["data"]
        ):
            return _SINGLE_POINT_HINT
        # Повторный вызов перезаписывает фигуру: у ответа один график.
        ctx.chart = figure
        head = f"График принят: {what}." if what else (
            f"График принят: серий {len(figure['data'])}."
        )
        return f"{head} Теперь напиши итоговый ответ, не пересказывая график текстом."

    # При неполученных звёздах фокус ответа — их показатели (см. deviation_map),
    # а модель строила график ещё до ответа и брала верхний показатель.
    description = CHART_TOOL_DESCRIPTION
    stars = open_star_metrics(ctx.db, ctx.person_key)
    if stars:
        description += (
            " У сотрудника есть неполученные звёзды: по умолчанию metric — один из "
            f"их показателей ({', '.join(stars)}), тот, что ты назвал главным фокусом."
        )

    return StructuredTool.from_function(
        func=build_chart,
        args_schema=ChartArgs,
        name=TOOL_NAME,
        description=description,
    )


__all__ = ["TOOL_NAME", "build_figure", "make_build_chart", "metric_traces", "open_star_metrics"]
