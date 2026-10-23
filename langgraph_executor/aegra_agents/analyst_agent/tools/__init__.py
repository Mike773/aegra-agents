"""Инструменты агента: мало, но широкие.

Каждый вызов инструмента — отдельный обмен с моделью, поэтому один вызов должен
закрывать вопрос целиком. Отсюда `deviation_map` (все показатели сотрудника
против плана, своей динамики и групп сразу), `metric_card` (всё об одном
показателе) и свободный `query_sql` вместо десятка узких выборок.

Выдача всегда человекочитаемая: она идёт прямо в контекст модели.
"""
from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from ..agent.guards import clean_args
from ..agent.runctx import RunContext
from .chart import make_build_chart
from .deviation_map import DEFAULT_DEPTH, deviation_map_text
from .metric_card import metric_card_text
from .peer_context import peer_context_text
from .query_sql import make_query_sql
from .search_wiki import make_search_wiki
from .star_rating import levels_text, star_rating_text
from .suggestions import make_suggest_followups


class _ToolArgs(BaseModel):
    """База схем аргументов: типы объявлены явно.

    Без явной схемы `langchain_gigachat` отдаёт модели `type: object` на каждый
    параметр, и GigaChat-3-Ultra послушно шлёт `{}` вместо строки — инструмент
    получает пустое название показателя (замечено на живом прогоне). Пустые
    значения любого вида до проверки типов превращаются в «не задано», чтобы
    инструмент ответил подсказкой, а не ошибкой валидации.

    `model_required` — поля, обязательные для модели, но не для Python. Они
    объявлены `str` без default: только так они попадают в `required` схемы
    функции (генератор схем `langchain_gigachat` снимает обязательность с
    любого nullable-поля). Без `required` модель читает поле необязательным и
    зовёт инструмент без него — metric_card без показателя, замечено на живом
    прогоне. Пропуск всё равно даёт подсказку, а не ошибку валидации:
    валидатор подставляет пустую строку, которую инструмент читает как
    «не задано».
    """

    model_required: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def _blanks_to_none(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = clean_args(data)
        for name in cls.model_required:
            if data.get(name) is None:
                data[name] = ""
        return data


class DeviationMapArgs(_ToolArgs):
    person: str | None = Field(
        None, description="Чью карту (ФИО или табельный); по умолчанию — тот, кого разбираем"
    )
    depth: int | None = Field(
        DEFAULT_DEPTH, description="До какого уровня дерева показателей брать (1 — верхний)"
    )


class MetricCardArgs(_ToolArgs):
    model_required = ("metric",)

    metric: str = Field(..., description="Название показателя")
    person: str | None = Field(
        None, description="Чей результат; по умолчанию — тот, кого разбираем"
    )
    date: str | None = Field(None, description="Конкретная дата, если нужна одна")
    depth: int | None = Field(2, description="Сколько уровней состава показать")
    element: str | None = Field(
        None, description="Название разреза, если нужен ряд по одному разрезу"
    )


class PeerContextArgs(_ToolArgs):
    model_required = ("metric",)

    metric: str = Field(..., description="Название показателя")
    person: str | None = Field(None, description="Чей результат")


class StarRatingArgs(_ToolArgs):
    person: str | None = Field(
        None, description="Чей рейтинг (ФИО или табельный); по умолчанию — тот, кого разбираем"
    )
    level: str | None = Field(
        None,
        description=(
            "Уровень рейтинга (ГОСБ, ТБ, Сбер и т.п. — название или код из описания "
            "инструмента), если нужен один; по умолчанию — все уровни"
        ),
    )
    quarter: str | None = Field(
        None, description="Квартал, если нужен один, например «3 квартал 2026»"
    )


def build_tools(ctx: RunContext, *, extra: list[Any] | None = None) -> list[Any]:
    """Набор инструментов под флаги запуска."""
    tools: list[Any] = []

    if ctx.text2sql_enabled:
        tools.append(make_query_sql(ctx))

    def deviation_map(person: str | None = None, depth: int | None = DEFAULT_DEPTH) -> str:
        return deviation_map_text(ctx, person=person, depth=depth)

    tools.append(
        StructuredTool.from_function(
            func=deviation_map,
            args_schema=DeviationMapArgs,
            name="deviation_map",
            description=(
                "Карта отклонений сотрудника: с неё начинается разбор показателей. По "
                "каждому показателю с планом — выполнение плана (по run rate, иначе по "
                "ex, иначе факт к плану), динамика к прошлому периоду, положение во "
                "всех группах сравнения с местом в рейтинге, медианой и долей коллег, "
                "выполняющих план, и вывод: общая это проблема группы или личная, "
                "мягкий ли план, в числе ли лучших. Показатели упорядочены: сначала "
                "отклонения от плана, затем слабые сигналы (план выполнен, но далеко "
                "позади коллег), сильные результаты, норма. Если у сотрудника есть "
                "звёзды, в карте только показатели неполученных звёзд. Аргументы: "
                "person — чью карту (по умолчанию тот, кого разбираем); depth — до "
                "какого уровня дерева показателей брать (1 — только верхний, по "
                f"умолчанию {DEFAULT_DEPTH})."
            ),
        )
    )

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
                "когда нужно разобраться в одном показателе. Разрезов у показателя бывают "
                "сотни — карточка сама показывает худшие разрезы, а ряд по одному "
                "конкретному разрезу даёт аргумент element. Аргументы: metric — "
                "название показателя, обязательно; person — чей результат (по умолчанию тот, кого "
                "разбираем); date — конкретная дата, если нужна одна; depth — сколько "
                "уровней состава показать; element — название разреза. Без metric "
                "карточки нет: сводка показателей верхнего уровня — в системном промпте."
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
        def star_rating(
            person: str | None = None,
            level: str | None = None,
            quarter: str | None = None,
        ) -> str:
            return star_rating_text(ctx, person=person, level=level, quarter=quarter)

        # Уровни — из базы этого запуска: модель должна знать, что «ГОСБ» — это
        # уровень рейтинга (аргумент level), а не имя сотрудника или показателя.
        tools.append(
            StructuredTool.from_function(
                func=star_rating,
                args_schema=StarRatingArgs,
                name="star_rating",
                description=(
                    "Рейтинг по звёздам: какое место сотрудник занял среди коллег на "
                    "каждом уровне (от узкой группы к широкой) за квартал и сколько "
                    "человек в рейтинге уровня. Отвечает на «какое место по звёздам», "
                    "«какой рейтинг в ГОСБ / по банку». Уровни рейтинга в этих данных "
                    f"(от узкой группы к широкой): {levels_text(ctx.db)}. Аргументы: "
                    "person — чей рейтинг (ФИО или табельный); level — уровень из "
                    "списка выше, если нужен один; quarter — квартал, если нужен один."
                ),
            )
        )

    if ctx.easyrag_enabled and ctx.easyrag_graph is not None:
        tools.append(make_search_wiki(ctx))

    # Интерактивные подсказки и график — только по входным параметрам: без них набор
    # инструментов и промпт прежние.
    if ctx.interactive_suggestions:
        tools.append(make_suggest_followups(ctx))
    if ctx.interactive_chart:
        tools.append(make_build_chart(ctx))

    tools.extend(extra or [])
    return tools


__all__ = ["build_tools"]
