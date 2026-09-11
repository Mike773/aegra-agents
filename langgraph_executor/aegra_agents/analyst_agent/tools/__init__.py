"""Инструменты агента: мало, но широкие.

Каждый вызов инструмента — отдельный обмен с моделью, поэтому один вызов должен
закрывать вопрос целиком. Отсюда `metric_card` (всё об одном показателе сразу)
и свободный `query_sql` вместо десятка узких выборок.

Выдача всегда человекочитаемая: она идёт прямо в контекст модели.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from ..agent.guards import clean_args
from ..agent.runctx import RunContext
from .metric_card import metric_card_text
from .peer_context import peer_context_text
from .query_sql import make_query_sql
from .search_wiki import make_search_wiki
from .star_rating import star_rating_text


class _ToolArgs(BaseModel):
    """База схем аргументов: типы объявлены явно.

    Без явной схемы `langchain_gigachat` отдаёт модели `type: object` на каждый
    параметр, и GigaChat-3-Ultra послушно шлёт `{}` вместо строки — инструмент
    получает пустое название показателя (замечено на живом прогоне). Пустые
    значения любого вида до проверки типов превращаются в «не задано», чтобы
    инструмент ответил подсказкой, а не ошибкой валидации.
    """

    @model_validator(mode="before")
    @classmethod
    def _blanks_to_none(cls, data: Any) -> Any:
        return clean_args(data) if isinstance(data, dict) else data


class MetricCardArgs(_ToolArgs):
    metric: str | None = Field(None, description="Название показателя")
    person: str | None = Field(
        None, description="Чей результат; по умолчанию — тот, кого разбираем"
    )
    date: str | None = Field(None, description="Конкретная дата, если нужна одна")
    depth: int | None = Field(2, description="Сколько уровней состава показать")
    element: str | None = Field(
        None, description="Название разреза, если нужен ряд по одному разрезу"
    )


class PeerContextArgs(_ToolArgs):
    metric: str | None = Field(None, description="Название показателя")
    person: str | None = Field(None, description="Чей результат")


class StarRatingArgs(_ToolArgs):
    person: str | None = Field(None, description="Чей рейтинг; по умолчанию — тот, кого разбираем")
    quarter: str | None = Field(
        None, description="Квартал, если нужен один, например «3 квартал 2026»"
    )


class ListDeviationsArgs(_ToolArgs):
    scope: str | None = Field(
        "all",
        description="all | plan | dynamics | peers | stars | achievements | model | task:<id>",
    )
    metric: str | None = Field(None, description="Фильтр по показателю")


class NoteDeviationArgs(_ToolArgs):
    metric: str = Field(..., description="Показатель")
    kind: str = Field(..., description="Короткий вид находки, например hypothesis")
    text: str = Field(..., description="Суть находки")
    element: str | None = Field(None, description="Разрез, если находка про него")


def build_tools(ctx: RunContext, *, extra: list[Any] | None = None) -> list[Any]:
    """Набор инструментов под флаги запуска."""
    tools: list[Any] = []

    if ctx.text2sql_enabled:
        tools.append(make_query_sql(ctx))

    def metric_card(
        metric: str | None = None,
        person: str | None = None,
        date: str | None = None,
        depth: int | None = 2,
        element: str | None = None,
    ) -> str:
        return metric_card_text(
            ctx, metric=metric, person=person, date=date,
            depth=depth if depth is not None else 2, element=element,
        )

    tools.append(
        StructuredTool.from_function(
            func=metric_card,
            args_schema=MetricCardArgs,
            name="metric_card",
            description=(
                "Всё об одном показателе за один вызов: описание и единицы, значения "
                "по периодам с планом и вердиктами, худшие разрезы, состав (влияющие "
                "показатели с весами), место среди коллег. Это первый инструмент, "
                "когда нужно разобраться в показателе. Разрезов у показателя бывают "
                "сотни — карточка сама показывает худшие разрезы, а ряд по одному "
                "конкретному разрезу даёт аргумент element. Аргументы: metric — "
                "название показателя; person — чей результат (по умолчанию тот, кого "
                "разбираем); date — конкретная дата, если нужна одна; depth — сколько "
                "уровней состава показать; element — название разреза."
            ),
        )
    )

    if ctx.use_peer_aggregates:
        def peer_context(metric: str | None = None, person: str | None = None) -> str:
            return peer_context_text(ctx, metric=metric, person=person)

        tools.append(
            StructuredTool.from_function(
                func=peer_context,
                args_schema=PeerContextArgs,
                name="peer_context",
                description=(
                    "Сравнение показателя с коллегами по группам от узкой к широкой: "
                    "среднее и медиана группы, уровень сильнейших, доля выполняющих "
                    "план, место сотрудника. Помогает отделить системное отклонение "
                    "от индивидуального. Аргументы: metric, person."
                ),
            )
        )

    # Рейтинг по звёздам приходит только вместе со звёздами — без него
    # инструмента нет, и набор инструментов для прежних датасетов не меняется.
    if ctx.db.has_ratings:
        def star_rating(person: str | None = None, quarter: str | None = None) -> str:
            return star_rating_text(ctx, person=person, quarter=quarter)

        tools.append(
            StructuredTool.from_function(
                func=star_rating,
                args_schema=StarRatingArgs,
                name="star_rating",
                description=(
                    "Рейтинг по звёздам: какое место сотрудник занял среди коллег на "
                    "каждом уровне (от узкой группы к широкой) за квартал и сколько "
                    "человек в рейтинге уровня. Отвечает на «какое место по звёздам», "
                    "«как выглядит на фоне банка/территории». Аргументы: person — "
                    "чей рейтинг; quarter — квартал, если нужен один."
                ),
            )
        )

    if ctx.easyrag_enabled and ctx.easyrag_graph is not None:
        tools.append(make_search_wiki(ctx))

    def list_deviations(scope: str | None = "all", metric: str | None = None) -> str:
        return ctx.ledger.render(scope=scope or "all", metric=metric)

    tools.append(
        StructuredTool.from_function(
            func=list_deviations,
            args_schema=ListDeviationsArgs,
            name="list_deviations",
            description=(
                "Карта отклонений сотрудника целиком: уже посчитана по данным и "
                "отранжирована по важности, в системном промпте — только её верхушка. "
                "scope: all | plan | dynamics | peers | stars | achievements | model "
                "| task:<id>; metric — фильтр по показателю."
            ),
        )
    )

    def note_deviation(
        metric: str, kind: str, text: str, element: str | None = None
    ) -> str:
        return ctx.ledger.note(metric=metric, kind=kind, text=text, element=element)

    tools.append(
        StructuredTool.from_function(
            func=note_deviation,
            args_schema=NoteDeviationArgs,
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
