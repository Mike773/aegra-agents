"""Состояние графа: публичные каналы и внутренние.

Публичная схема повторяет OrchestratorOutput из v4 поле в поле — прод-клиент
уже завязан на эти имена. Добавлены только каналы составного разбора (задачи и
их результаты). Внутренние каналы (сырой датасет, готовые блоки промпта, карта
отклонений) чекпойнтятся, но наружу не отдаются: они большие и клиенту не нужны.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

Intent = Literal["analytics", "wiki", "save_insight", "chat", "done"]


class TraceStep(TypedDict, total=False):
    """Один шаг рассуждения для сквозного лога (формат v4)."""

    stage: str
    kind: str
    summary: str
    detail: dict


class Task(TypedDict, total=False):
    """Задача составного разбора (dashboard_mode)."""

    id: str
    title: str
    question: str
    metric_names: list[str]
    target_value: float | None
    target_kind: str | None
    depends_on: list[str]


class TaskResult(TypedDict, total=False):
    id: str
    title: str
    question: str
    answer_md: str
    error: str | None


class AnalystOutput(TypedDict, total=False):
    """Публичная (возвращаемая) схема графа."""

    messages: Annotated[list[AnyMessage], add_messages]

    boss_tabnum: str
    employee_tabnum: str
    position: str | None
    direction_key: str | None

    # Первое сообщение диалога: роль, методология, желаемый формат, иногда —
    # список задач. Держится весь диалог и подмешивается в системный промпт.
    briefing: str | None

    # Сквозной лог шагов текущего хода. Строго per-turn: первый узел хода
    # перезаписывает список, downstream дополняет явной конкатенацией.
    reasoning_trace: list[TraceStep]

    metrics_error: str | None
    aggregates_error: str | None
    loaded: bool

    # Внешняя привязка инсайтов; нет обоих — в сервис ничего не пишем.
    source_type: str | None
    source_id: str | None

    intent: Intent | None

    # Что за ход обратились в базу знаний (для описи источников в ответе).
    easyrag_query: str | None
    easyrag_snippets: list[dict]
    easyrag_error: str | None
    easyrag_stub_pages: list[dict]

    # Опорный разбор первого хода: ставится один раз и не перезаписывается.
    metrics_summary: str | None
    # Последний узкий вопрос и ответ по нему.
    analytics_question: str | None
    analytics_answer: str | None
    analytics_error: str | None

    memory_context: str | None
    memory_error: str | None
    memory_saved: bool | None
    memory_save_error: str | None
    memory_loaded: bool

    # Составной разбор: задачи, указатель текущей и накопленные результаты.
    dashboard_mode: bool
    tasks: list[Task]
    task_idx: int
    task_results: list[TaskResult]


class AnalystState(AnalystOutput, total=False):
    """Полный стейт: публичные каналы + внутренние (не отдаются наружу)."""

    # Сырой датасет и предагрегаты: живут между ходами, база пересобирается из них.
    metrics: Any
    aggregates: Any

    turn_kind: str            # initial | followup | dashboard
    turn_no: int

    # Готовые блоки системного промпта, посчитанные один раз на первом ходе.
    schema_doc: str
    enrichment_block: str
    catalog_block: str

    # Карта отклонений на весь диалог (пересобирается и дополняется каждый ход).
    deviations: list[dict]
    insight_written: bool


__all__ = ["AnalystOutput", "AnalystState", "Intent", "Task", "TaskResult", "TraceStep"]
