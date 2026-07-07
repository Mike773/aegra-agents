"""Состояние графа external_doc_loader.

Финальный отчёт уходит и текстом (``report``), и ``AIMessage`` (``messages``) —
чтобы граф работал и как batch-джоба, и в чат-контексте aegra (паттерн
gap_resolver).
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ExternalDocLoaderState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # Документы, полученные из внешнего источника.
    # Элемент: {"direction_id": str, "text": str}.
    documents: list[dict[str, Any]]

    # Итог по каждому документу: {"direction_key", "status", ...}, где status —
    # created | duplicate | similar_skipped | replaced.
    results: list[dict[str, Any]]

    errors: list[dict[str, Any]]
    report: str
