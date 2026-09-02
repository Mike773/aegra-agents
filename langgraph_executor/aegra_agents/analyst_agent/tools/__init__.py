"""Инструменты агента: мало, но широкие.

Каждый вызов инструмента — отдельный обмен с моделью, поэтому один вызов должен
закрывать вопрос целиком. Отсюда `metric_card` (всё об одном показателе сразу)
и свободный `query_sql` вместо десятка узких выборок.

Выдача всегда человекочитаемая: она идёт прямо в контекст модели.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from ..agent.runctx import RunContext
from .metric_card import metric_card_text
from .peer_context import peer_context_text
from .query_sql import make_query_sql
from .search_wiki import make_search_wiki


def build_tools(ctx: RunContext, *, extra: list[Any] | None = None) -> list[Any]:
    """Набор инструментов под флаги запуска."""
    tools: list[Any] = []

    if ctx.text2sql_enabled:
        tools.append(make_query_sql(ctx))

    tools.append(
        StructuredTool.from_function(
            func=lambda metric, person=None, date=None, depth=2: metric_card_text(
                ctx, metric=metric, person=person, date=date, depth=depth
            ),
            name="metric_card",
            description=(
                "Всё об одном показателе за один вызов: описание и единицы, значения "
                "по периодам с планом и вердиктами, разрезы, состав (влияющие "
                "показатели с весами), место среди коллег. Аргументы: metric — "
                "название показателя; person — чей результат (по умолчанию тот, кого "
                "разбираем); date — конкретная дата, если нужна одна; depth — сколько "
                "уровней состава показать."
            ),
        )
    )

    if ctx.use_peer_aggregates:
        tools.append(
            StructuredTool.from_function(
                func=lambda metric, person=None: peer_context_text(
                    ctx, metric=metric, person=person
                ),
                name="peer_context",
                description=(
                    "Сравнение показателя с коллегами по группам от узкой к широкой: "
                    "среднее и медиана группы, уровень сильнейших, доля выполняющих "
                    "план, место сотрудника. Аргументы: metric, person."
                ),
            )
        )

    if ctx.easyrag_enabled and ctx.easyrag_graph is not None:
        tools.append(make_search_wiki(ctx))

    tools.append(
        StructuredTool.from_function(
            func=lambda scope="all", metric=None: ctx.ledger.render(
                scope=scope or "all", metric=metric
            ),
            name="list_deviations",
            description=(
                "Карта отклонений сотрудника целиком, с числами и приоритетом. "
                "scope: all | plan | dynamics | peers | stars | achievements | model "
                "| task:<id>; metric — фильтр по показателю."
            ),
        )
    )

    tools.append(
        StructuredTool.from_function(
            func=lambda metric, kind, text, element=None: ctx.ledger.note(
                metric=metric, kind=kind, text=text, element=element
            ),
            name="note_deviation",
            description=(
                "Зафиксировать свою находку, которой нет в карте отклонений: она "
                "переживёт следующие вопросы руководителя. Аргументы: metric — "
                "показатель; kind — короткий вид находки (например hypothesis); "
                "text — суть находки; element — разрез, если находка про него."
            ),
        )
    )

    tools.extend(extra or [])
    return tools


__all__ = ["build_tools"]
