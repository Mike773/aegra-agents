"""Состояние графа gap_resolver.

Финальный отчёт уходит и текстом (``report``), и ``AIMessage`` (``messages``) —
чтобы граф работал и как batch-джоба (вызов с одним ``direction_key``), и в
чат-контексте aegra. ``direction_key`` читается из ``config.configurable``,
затем из state (паттерн как в kb_chat/doc_manager).
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GapResolverState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Вход.
    direction_key: str
    top_k: int

    # Нерешённые gap'ы направления, сгруппированные по нормализованному вопросу.
    # Элемент: {"id": str, "query": str, "ids": list[str]} (ids — все дубли вопроса).
    gaps: list[dict[str, Any]]

    # Результат разбора.
    resolved: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]
    created_stub_pages: list[str]
    errors: list[dict[str, Any]]
    report: str
