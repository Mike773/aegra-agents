"""Инструмент ``build_chart`` — график под ответом в формате plotly.

Модель описывает график типизированным подмножеством plotly (тип, заголовки,
серии x/y), код чистит его и собирает стандартную фигуру
``{"data": [...], "layout": {...}}``, которую узел агента переносит в
``additional_kwargs`` итога (``orchestrator_chart``). Библиотека plotly не
нужна: фигура — обычный JSON. Привязывается только при ``interactive_chart=true``.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..agent.runctx import RunContext
from ..prompts.chart import CHART_TOOL_DESCRIPTION, MAX_POINTS, MAX_TRACES

TOOL_NAME = "build_chart"

ChartType = Literal["line", "bar", "scatter"]

# Тип графика модели → тип и режим серии plotly.
_PLOTLY_TRACE: dict[str, dict[str, str]] = {
    "line": {"type": "scatter", "mode": "lines+markers"},
    "bar": {"type": "bar"},
    "scatter": {"type": "scatter", "mode": "markers"},
}

_EMPTY_HINT = (
    "График не принят: нужен хотя бы один объект в traces с полями name, x и y "
    "одинаковой длины, где y — числа из данных. Исправь аргументы и вызови ещё "
    "раз либо переходи к итоговому ответу без графика."
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
    x_title: str | None = Field(None, description="Подпись оси x")
    y_title: str | None = Field(None, description="Подпись оси y")
    traces: list[Trace] = Field(..., description=f"Серии графика, до {MAX_TRACES} штук")


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


def make_build_chart(ctx: RunContext) -> StructuredTool:
    def build_chart(
        chart_type: str,
        title: str,
        traces: list[Any],
        x_title: str | None = None,
        y_title: str | None = None,
    ) -> str:
        figure = build_figure({
            "chart_type": chart_type, "title": title, "traces": traces,
            "x_title": x_title, "y_title": y_title,
        })
        if figure is None:
            return _EMPTY_HINT
        # Повторный вызов перезаписывает фигуру: у ответа один график.
        ctx.chart = figure
        return (
            f"График принят: серий {len(figure['data'])}. Теперь напиши итоговый "
            "ответ, не пересказывая график текстом."
        )

    return StructuredTool.from_function(
        func=build_chart,
        args_schema=ChartArgs,
        name=TOOL_NAME,
        description=CHART_TOOL_DESCRIPTION,
    )


__all__ = ["TOOL_NAME", "build_figure", "make_build_chart"]
